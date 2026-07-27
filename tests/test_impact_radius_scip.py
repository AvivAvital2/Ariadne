"""Guardrail for fix #1 — file-based impact_radius on the SCIP graph.

``library.impact_radius(file)`` counts dependent files from the SCIP call graph by
**call site** (``edge.file``) — reliable even when the caller is a SCIP ``local``
symbol — instead of the (empty) legacy import graph. Direct is exact; transitive is a
conservative file-level lower bound; ``scip_indexed`` flags un-indexed sources.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import config
from ariadne_mcp.server_knowledge import ariadne_impact_radius
from ariadne_mcp.service import AriadneService
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
)
from library import Library


@pytest.fixture(autouse=True)
def restore_config():
    saved = config._global_config
    yield
    config._global_config = saved

# ---------------------------------------------------------------------------
# library.impact_radius(file) on the SCIP graph
# ---------------------------------------------------------------------------


def _pysym(pkg: str, descriptor: str, name: str) -> str:
    return f'scip-python python {pkg} 1 {descriptor}#{name}().'


def _doc_multi(rel: str, defs: list[tuple[str, str, tuple[str, ...]]]) -> _ScipDoc:
    """A file with one or more definitions; each may reference other symbols
    (refs land on lines inside that definition so the edge is owned by it).
    """
    occ: list = []
    syms: list = []
    line = 0
    for sym, display, refs in defs:
        occ.append(_ScipOccurrence(symbol=sym, range=(line, 0, line + 9, 0), is_definition=True))
        syms.append(_ScipSymbol(symbol=sym, kind='Method', display_name=display))
        for j, r in enumerate(refs):
            occ.append(_ScipOccurrence(symbol=r, range=(line + 2 + j, 4, line + 2 + j, 8),
                                       is_definition=False))
        line += 12
    return _ScipDoc(relative_path=rel, occurrences=tuple(occ), symbols=tuple(syms))


def _impact_db(tmp_path: Path) -> tuple[Path, Path]:
    """src1 defines Target.run() in mod/target.py. It is called by caller_a.py and
    caller_b.py (src1), x.py (src2, cross-source), a same-file Helper (must be
    excluded as self), and a SCIP `local 0` symbol (must be excluded as noise).
    caller_aa.py calls A → a transitive (2-hop) dependent. src3 is configured but
    has no SCIP data.
    """
    src1, src2, src3 = (tmp_path / n for n in ('src1', 'src2', 'src3'))
    for d in (src1, src2, src3):
        d.mkdir()

    target_sym = _pysym('src1', 'mod/target/Target', 'run')
    helper_sym = _pysym('src1', 'mod/target/Helper', 'aux')
    a_sym = _pysym('src1', 'mod/caller_a/A', 'a')
    b_sym = _pysym('src1', 'mod/caller_b/B', 'b')
    aa_sym = _pysym('src1', 'mod/caller_aa/AA', 'aa')
    x_sym = _pysym('src2', 'x/X', 'x')

    idx1 = ScipIndex(documents=(
        _doc_multi('mod/target.py', [(target_sym, 'run', ()), (helper_sym, 'aux', (target_sym,))]),
        _doc_multi('mod/caller_a.py', [(a_sym, 'a', (target_sym,))]),
        _doc_multi('mod/caller_b.py', [(b_sym, 'b', (target_sym,))]),
        _doc_multi('mod/caller_aa.py', [(aa_sym, 'aa', (a_sym,))]),
        _doc_multi('mod/caller_local.py', [('local 0', 'loc', (target_sym,))]),
    ))
    idx2 = ScipIndex(documents=(_doc_multi('x.py', [(x_sym, 'x', (target_sym,))]),))

    g = CrossSourceGraph()
    g.add_source('src1', index=idx1, language='python')
    g.add_source('src2', index=idx2, language='python')
    g.materialize()

    db = tmp_path / 'impact.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        g.save_to(conn)
    lib.close()

    cfg = tmp_path / 'ariadne.yaml'
    cfg.write_text(
        'sources:\n'
        f'  src1:\n    path: {src1}\n'
        f'  src2:\n    path: {src2}\n'
        f'  src3:\n    path: {src3}\n'
        'default_source: src1\n',
    )
    return db, cfg


class TestFileBasedImpact:
    """impact_radius(file) counts dependent files from the SCIP call graph by call
    site (edge.file) — cross-source, including local-scope call sites, excluding the
    target file itself — and flags un-indexed sources instead of a misleading 0.
    """

    def test_dependents_via_scip_with_breakdown(
        self, tmp_path: Path,
    ) -> None:
        db, cfg = _impact_db(tmp_path)
        config._global_config = config.Config(config_path=cfg)
        root1 = config._global_config.get_all_source_paths()['src1']
        lib = Library(db)
        try:
            r = lib.impact_radius(str(root1 / 'mod/target.py'), depth=2)
        finally:
            lib.close()

        assert r['scip_indexed'] is True
        # caller_a + caller_b (src1) + x (src2) + caller_local (src1) — the local-scope
        # call site now counts (edge.file is reliable); the same-file Helper is self.
        assert r['direct_dependents'] == 4
        assert r['transitive_dependents'] == 1          # caller_aa → A → T
        assert r['total_affected_files'] == 5
        assert r['dependents_by_source'] == {'src1': 4, 'src2': 1}

    def test_unindexed_source_is_flagged_not_a_leaf(
        self, tmp_path: Path,
    ) -> None:
        db, cfg = _impact_db(tmp_path)
        config._global_config = config.Config(config_path=cfg)
        root3 = config._global_config.get_all_source_paths()['src3']
        lib = Library(db)
        try:
            r = lib.impact_radius(str(root3 / 'whatever.py'))
        finally:
            lib.close()

        assert r['scip_indexed'] is False               # NOT a silent 0/leaf
        assert r['direct_dependents'] == 0

    def test_file_outside_known_sources(
        self, tmp_path: Path,
    ) -> None:
        """A path under no configured source resolves to no source → flagged, not a leaf."""
        db, cfg = _impact_db(tmp_path)
        config._global_config = config.Config(config_path=cfg)
        lib = Library(db)
        try:
            r = lib.impact_radius('/nowhere/unknown.py')
        finally:
            lib.close()

        assert r['scip_indexed'] is False
        assert r['direct_dependents'] == 0


# ---------------------------------------------------------------------------
# Slice ③ — MCP tool surfaces scip_indexed honesty + per-source breakdown
# ---------------------------------------------------------------------------


def _wire_service(result: dict) -> None:
    """Point the singleton AriadneService at a stub library returning ``result``."""
    svc = AriadneService()
    svc._library = SimpleNamespace(impact_radius=lambda *_a, **_kw: result)
    AriadneService._instance = svc


class TestToolSurface:
    """Slice ③ — ariadne_impact_radius reflects scip_indexed + dependents_by_source."""

    async def test_not_indexed_source_flagged_in_output(self) -> None:
        _wire_service({
            'file': 'f.py', 'scip_indexed': False, 'direct_dependents': 0,
            'transitive_dependents': 0, 'total_affected_files': 0,
            'dependents_by_source': {}, 'affected_docs': 2, 'affected_tests': 1,
            'radius_score': 1, 'top_dependents': [],
        })
        out = (await ariadne_impact_radius('f.py')).output
        assert 'not SCIP-indexed' in out          # honest signal, not a silent 0
        assert 'Affected docs: 2' in out

    async def test_indexed_output_shows_per_source_breakdown(self) -> None:
        _wire_service({
            'file': 'f.py', 'scip_indexed': True, 'direct_dependents': 3,
            'transitive_dependents': 1, 'total_affected_files': 4,
            'dependents_by_source': {'src1': 3, 'src2': 1}, 'affected_docs': 2,
            'affected_tests': 1, 'radius_score': 8, 'top_dependents': ['a.py', 'b.py'],
        })
        out = (await ariadne_impact_radius('f.py')).output
        assert 'Direct dependents: 3' in out
        assert 'src1 (3)' in out and 'src2 (1)' in out

    async def test_indexed_leaf_omits_breakdown(self) -> None:
        """Indexed file with no dependents: real 0 (not the not-indexed message),
        and no empty per-source line.
        """
        _wire_service({
            'file': 'f.py', 'scip_indexed': True, 'direct_dependents': 0,
            'transitive_dependents': 0, 'total_affected_files': 0,
            'dependents_by_source': {}, 'affected_docs': 0, 'affected_tests': 0,
            'radius_score': 0, 'top_dependents': [],
        })
        out = (await ariadne_impact_radius('f.py')).output
        assert 'Direct dependents: 0' in out
        assert 'not SCIP-indexed' not in out
        assert 'Dependents by source' not in out
