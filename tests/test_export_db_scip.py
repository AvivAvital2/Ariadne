"""Batch 3: SCIP reachability + cross-instance merge safety for the DB slice.

Two contracts, both deliberate *deviations* from scoped-full:

- **Reachability** — a slice must carry the cross-source symbols the source
  calls into (and the edges to them), so ``callers``/``callees``/``impact_radius``
  resolve. Scoped-full keeps only both-endpoints-in-source edges (isolation);
  a shareable bundle wants the opposite (completeness).
- **Merge safety** — ``local N`` symbol ids are renumbered per ``scip merge``,
  so they are NOT stable across instances. Export namespaces them by owning
  source so merging a slice into a populated DB can't overwrite an unrelated
  local symbol on the ``canonical_id`` primary key.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import numpy as np
import pytest

import cli.core as core
from docgen.scip_cross_source import CrossSourceGraph
from export_db import export_source_db, import_source_db
from library import Library

EMBED_DIM = 16

_SYM_COLS = ('canonical_id', 'source_name', 'language', 'file', 'line_start',
             'line_end', 'kind', 'display_name', 'qualified_name', 'parent_qualified_name')


def _vec(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _sym(cid, src, name, parent=None):
    return (cid, src, 'python', f'{src}/x.py', 1, 2, 'Function', name, f'{src}.{name}', parent)


@pytest.fixture(autouse=True)
def _no_token_count(monkeypatch):
    monkeypatch.setattr('library.core.CoreMixin._count_tokens', lambda self, content: None)


class _FakeCfg:
    default_source = None
    db_path = 'ariadne.db'


@pytest.fixture
def cfg(monkeypatch):
    monkeypatch.setattr(core, 'get_config', lambda: _FakeCfg())


@pytest.fixture
def scip_db(tmp_path):
    path = tmp_path / 'full.db'
    lib = Library(path)
    lib.add_document(content_type='explanation', title='SDK', content='body',
                     source_files=['sdk/x.py'], embedding=_vec(1), source_name='sdk')
    lib.close()

    conn = sqlite3.connect(path)
    conn.executemany(
        f"INSERT INTO scip_symbols ({','.join(_SYM_COLS)}) VALUES ({','.join('?' * 10)})",
        [
            _sym('sdk.func', 'sdk', 'func'),
            _sym('local 5', 'sdk', 'x', parent='sdk.func'),       # a local in sdk
            _sym('ariadne.helper', 'ariadne', 'helper'),          # cross-source callee
            _sym('noise.thing', 'noise', 'thing'),                # unrelated
        ],
    )
    conn.executemany(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, edge_type, file, line, confidence) '
        'VALUES (?,?,?,?,?,?)',
        [
            ('sdk.func', 'ariadne.helper', 'call', 'sdk/x.py', 3, 'exact'),   # cross-source
            ('sdk.func', 'local 5', 'call', 'sdk/x.py', 4, 'exact'),          # local callee
            ('noise.thing', 'ariadne.helper', 'call', 'noise/x.py', 1, 'exact'),  # unrelated
        ],
    )
    conn.commit()
    conn.close()
    return {'path': str(path)}


def test_scip_reachability_pulls_cross_source_callees(scip_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(scip_db['path'], 'sdk', out, include_scip=True)

    conn = sqlite3.connect(out)
    try:
        syms = {r[0] for r in conn.execute('SELECT canonical_id FROM scip_symbols')}
        edges = conn.execute('SELECT caller_canonical_id, callee_canonical_id FROM scip_edges').fetchall()
    finally:
        conn.close()

    assert 'sdk.func' in syms
    assert 'ariadne.helper' in syms, 'cross-source callee symbol not pulled into the slice'
    assert ('sdk.func', 'ariadne.helper') in edges, 'cross-source edge missing'
    assert 'noise.thing' not in syms, 'unreachable symbol leaked into the slice'

    g = CrossSourceGraph()
    with sqlite3.connect(out) as gc:
        g.load_from(gc)
    callee_sources = {e.callee.source_name for e in g.callees_of('sdk.func')}
    assert 'ariadne' in callee_sources, 'cross-source callee not queryable in the slice'


def test_local_symbols_are_namespaced_on_export(scip_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(scip_db['path'], 'sdk', out, include_scip=True)
    conn = sqlite3.connect(out)
    try:
        syms = {r[0] for r in conn.execute('SELECT canonical_id FROM scip_symbols')}
        edges = conn.execute('SELECT callee_canonical_id FROM scip_edges').fetchall()
    finally:
        conn.close()
    assert 'sdk::local 5' in syms, 'local symbol not namespaced'
    assert 'local 5' not in syms, 'raw local id must not survive export'
    assert ('sdk::local 5',) in edges, 'edge endpoint not rewritten to match'


def test_merge_does_not_corrupt_an_unrelated_local_symbol(scip_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(scip_db['path'], 'sdk', out, include_scip=True)

    target = tmp_path / 'target.db'
    Library(target).close()
    tconn = sqlite3.connect(target)
    tconn.execute(
        f"INSERT INTO scip_symbols ({','.join(_SYM_COLS)}) VALUES ({','.join('?' * 10)})",
        ('local 5', 'other', 'python', 'other/x.py', 1, 1, 'Function', 'TARGET_LOCAL', 'other.x', None),
    )
    tconn.commit()
    tconn.close()

    import_source_db(target, out)

    tconn = sqlite3.connect(target)
    try:
        row = tconn.execute(
            "SELECT display_name, source_name FROM scip_symbols WHERE canonical_id='local 5'"
        ).fetchone()
    finally:
        tconn.close()
    assert row == ('TARGET_LOCAL', 'other'), 'merge corrupted an unrelated local symbol'


def test_cli_with_scip_flag_includes_scip(scip_db, tmp_path, cfg):
    out = tmp_path / 'slice.db'
    rc = core.cmd_export_db(argparse.Namespace(
        db=Path(scip_db['path']), source='sdk', out=str(out),
        no_embeddings=False, with_scip=True,
    ))
    assert rc == 0
    conn = sqlite3.connect(out)
    try:
        n = conn.execute('SELECT COUNT(*) FROM scip_symbols').fetchone()[0]
    finally:
        conn.close()
    assert n > 0, '--with-scip did not carry SCIP rows'


def test_parser_accepts_with_scip_flag():
    from cli.main import create_parser
    args = create_parser().parse_args(['export-db', '--source', 's', '--out', 'o.db', '--with-scip'])
    assert args.with_scip is True
