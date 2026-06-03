"""Tests for SCIP cross-source enrichment in ``catalog_enrich`` (Phase 2 Change 2).

The catalog generator's prompt currently sees only what's inside the file
(decorators, args, imports, in-file structure). For the LLM to describe
how a Python module fits into the wider codebase, ``enrich_file`` must
attach a ``ScipFileMetadata`` carrying cross-file callers/callees
resolved against the materialized SCIP graph.

These tests pin the contract of that wiring without requiring a real
``.scip`` file or DB — a ``CrossSourceGraph`` is constructed in-memory
with a hand-rolled symbol/edge pair, then passed to ``enrich_file``.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent

from docgen.catalog_enrich import (
    ScipCallee,
    ScipCaller,
    ScipFileMetadata,
    enrich_file,
)
from docgen.scip_cross_source import (
    CrossSourceEdge,
    CrossSourceGraph,
    CrossSourceSymbol,
)


def _write(p: Path, src: str) -> None:
    p.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


def _make_symbol(
    *,
    qualified_name: str,
    source_name: str,
    file: str,
    canonical_id: str | None = None,
    display_name: str | None = None,
    line: int = 1,
) -> CrossSourceSymbol:
    return CrossSourceSymbol(
        canonical_id=canonical_id or f'scip-python python {source_name} 0.1 {qualified_name}.',
        source_name=source_name,
        language='python',
        file=file,
        line_start=line,
        line_end=line,
        kind='Method',
        display_name=display_name or qualified_name.rsplit('.', 1)[-1],
        qualified_name=qualified_name,
        parent_qualified_name=None,
    )


def test_no_graph_means_scip_field_is_none(tmp_path: Path) -> None:
    """Backwards compatible: callers that don't pass ``cross_source_graph``
    still get a working bundle. ``scip`` defaults to None — the prompt
    template should treat None as "no SCIP data" rather than an error.
    """
    f = tmp_path / 'm.py'
    _write(f, '"""m."""\ndef foo() -> int:\n    return 1\n')

    bundle = enrich_file(f, source_root=tmp_path)

    assert bundle is not None
    assert bundle.scip is None


def test_graph_provided_populates_cross_source_callers(tmp_path: Path) -> None:
    """When the graph contains a cross-FILE edge into this file's symbol,
    the bundle's ``scip.callers`` includes a ``ScipCaller`` naming the
    remote source and qualified name. Bites the stub that returns an
    empty ``ScipFileMetadata``.
    """
    f = tmp_path / 'service.py'
    _write(f, '"""service."""\ndef validate_token() -> bool:\n    return True\n')

    local = _make_symbol(
        qualified_name='service.validate_token',
        source_name='ariadne',
        file='service.py',
        line=2,
    )
    remote = _make_symbol(
        qualified_name='auth_service.login',
        source_name='pyproject',
        file='auth_service.py',
        line=12,
    )
    edge = CrossSourceEdge(
        caller=remote,
        callee=local,
        edge_type='call',
        file='auth_service.py',
        line=12,
    )

    graph = CrossSourceGraph()
    graph._symbols = {
        local.canonical_id: local,
        remote.canonical_id: remote,
    }
    graph._edges = [edge]
    graph._known_source_names = {'ariadne', 'pyproject'}

    bundle = enrich_file(f, source_root=tmp_path, cross_source_graph=graph)

    assert bundle is not None
    assert bundle.scip is not None
    assert any(
        c.local_qualified_name == 'service.validate_token'
        and c.remote_source_name == 'pyproject'
        and c.remote_qualified_name == 'auth_service.login'
        for c in bundle.scip.callers
    ), (
        'expected at least one ScipCaller with remote_source_name=pyproject, '
        f'got {bundle.scip.callers!r}'
    )


def test_graph_provided_populates_cross_source_callees(tmp_path: Path) -> None:
    """Inverse of the callers test. Symbol in this file calls a symbol
    elsewhere → ``scip.callees`` carries a ``ScipCallee`` naming the
    target.
    """
    f = tmp_path / 'orchestrator.py'
    _write(f, '"""orchestrator."""\ndef run() -> None:\n    pass\n')

    local = _make_symbol(
        qualified_name='orchestrator.run',
        source_name='ariadne',
        file='orchestrator.py',
        line=2,
    )
    remote = _make_symbol(
        qualified_name='shared_lib.helper',
        source_name='shared',
        file='shared_lib.py',
        line=5,
    )
    edge = CrossSourceEdge(
        caller=local,
        callee=remote,
        edge_type='call',
        file='orchestrator.py',
        line=3,
    )

    graph = CrossSourceGraph()
    graph._symbols = {
        local.canonical_id: local,
        remote.canonical_id: remote,
    }
    graph._edges = [edge]
    graph._known_source_names = {'ariadne', 'shared'}

    bundle = enrich_file(f, source_root=tmp_path, cross_source_graph=graph)

    assert bundle is not None
    assert bundle.scip is not None
    assert any(
        c.local_qualified_name == 'orchestrator.run'
        and c.remote_source_name == 'shared'
        and c.remote_qualified_name == 'shared_lib.helper'
        for c in bundle.scip.callees
    ), (
        'expected at least one ScipCallee with remote_source_name=shared, '
        f'got {bundle.scip.callees!r}'
    )


async def test_orchestrator_passes_graph_to_enrich_file(tmp_path: Path) -> None:
    """The orchestrator's ``_catalog_generate`` path must thread the
    graph it loaded in ``__aenter__`` through to ``enrich_file``.
    Otherwise the bundle's ``scip`` field stays None and the prompt
    template never sees cross-source data — even though the graph
    sits right there in memory.
    """
    from unittest.mock import AsyncMock, patch

    from docgen import catalog_enrich
    from docgen.generator import DocGenerator
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
    from docgen.scip_cross_source import CrossSourceGraph

    f = tmp_path / 'm.py'
    _write(f, '"""m."""\ndef foo(): pass\n')

    config = OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'test.db',
        staleness_db_path=tmp_path / 'stale.db',
        dry_run=True,
        catalog_only_generator=True,
    )

    captured: dict = {}
    real_enrich = catalog_enrich.enrich_file

    def spy(*args, **kwargs):
        captured['cross_source_graph'] = kwargs.get('cross_source_graph')
        return real_enrich(*args, **kwargs)

    with patch.object(catalog_enrich, 'enrich_file', spy), patch.object(
        DocGenerator, 'generate_from_elements', new_callable=AsyncMock,
        return_value=[],
    ):
        async with DocGenOrchestrator(config) as orch:
            await orch._process_file(f)

    assert isinstance(captured.get('cross_source_graph'), CrossSourceGraph), (
        'orchestrator did not pass a CrossSourceGraph to enrich_file; '
        f'got {type(captured.get("cross_source_graph"))}'
    )


def test_architecture_prompt_renders_cross_source_callers(tmp_path: Path) -> None:
    """End-to-end contract: when ``bundle.scip.callers`` is non-empty,
    the architecture prompt's ``{dependents}`` slot is rendered with the
    SCIP-derived caller list instead of the legacy placeholder.

    Without this, the SCIP enrichment work is invisible — the LLM still
    writes "Dependents: (Analysis not performed)" even though the graph
    knew otherwise.
    """
    from docgen.catalog_enrich import (
        EnrichedElementInfo,
        EnrichedFileBundle,
    )
    from docgen.catalog_extractor import ElementInfo
    from docgen.generator import DocGenerator, GeneratorConfig
    from docgen.prompts import get_template

    fake_element = ElementInfo(
        language='python',
        subtype='function',
        file=str(tmp_path / 'service.py'),
        qualified_name='service.validate_token',
        signature='def validate_token() -> bool:',
        line_start=1,
        line_end=3,
        col_start=0,
        col_end=10,
    )
    bundle = EnrichedFileBundle(
        path=tmp_path / 'service.py',
        language='python',
        module_name='service',
        elements=(EnrichedElementInfo(element=fake_element),),
        scip=ScipFileMetadata(
            callers=(
                ScipCaller(
                    local_qualified_name='service.validate_token',
                    remote_qualified_name='auth_service.login',
                    remote_source_name='pyproject',
                    remote_file='auth_service.py',
                    remote_line=12,
                ),
            ),
        ),
    )

    gen = DocGenerator(
        config=GeneratorConfig(
            api_key='test-not-used',
            doc_types=('architecture',),
        ),
    )
    template = get_template('architecture')
    prompt = gen._format_prompt_from_bundle(template, bundle, source_code='# stub')

    assert 'auth_service.login' in prompt, (
        'expected the cross-source caller to appear in the architecture '
        f'prompt; got:\n{prompt}'
    )
    assert '(in `pyproject`,' in prompt
    assert '(Analysis not performed)' not in prompt


def test_within_file_edges_are_excluded(tmp_path: Path) -> None:
    """Same-file references add nothing the in-file element list doesn't
    already describe. The contract excludes them so the prompt's
    Cross-Source section doesn't repeat what the elements already show.
    """
    f = tmp_path / 'mod.py'
    _write(f, '"""mod."""\ndef a(): pass\n\ndef b(): a()\n')

    sym_a = _make_symbol(
        qualified_name='mod.a',
        source_name='ariadne',
        file='mod.py',
        line=2,
    )
    sym_b = _make_symbol(
        qualified_name='mod.b',
        source_name='ariadne',
        file='mod.py',
        line=4,
    )
    edge = CrossSourceEdge(
        caller=sym_b,
        callee=sym_a,
        edge_type='call',
        file='mod.py',
        line=4,
    )

    graph = CrossSourceGraph()
    graph._symbols = {
        sym_a.canonical_id: sym_a,
        sym_b.canonical_id: sym_b,
    }
    graph._edges = [edge]
    graph._known_source_names = {'ariadne'}

    bundle = enrich_file(f, source_root=tmp_path, cross_source_graph=graph)

    assert bundle is not None
    assert bundle.scip is not None
    # No callers/callees because the only edge is same-file.
    assert bundle.scip.callers == ()
    assert bundle.scip.callees == ()
