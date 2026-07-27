"""Functional tests that drive Ariadne's MCP tools over a REAL client↔server
session (in-memory transport — no socket, no fakes).

Unlike the contract tests that call the tool functions directly, these connect
an actual ``ClientSession`` to the actual FastMCP server and invoke tools by
name over the JSON-RPC protocol — hard evidence the onboarding tool chain and
the ariadne_onboard progress stream work end-to-end over MCP. The no-LLM admin
tools (source_add / discover / estimate / list_sources) run for real against a
synthetic project; the paid onboard pipeline is stubbed so the tool + its
progress notifications can be exercised without an LLM.
"""
from __future__ import annotations

import pytest

from mcp.shared.memory import create_connected_server_and_client_session

from ariadne_mcp.service import AriadneService


@pytest.fixture(autouse=True)
def _isolated_config(monkeypatch, tmp_path):
    cfg_path = tmp_path / 'ariadne.yaml'
    cfg_path.write_text('sources: {}\n')
    monkeypatch.setenv('ARIADNE_CONFIG', str(cfg_path))
    monkeypatch.chdir(tmp_path)
    import config as config_module
    monkeypatch.setattr(config_module, '_global_config', None, raising=False)
    from ariadne_mcp.service import AriadneService
    monkeypatch.setattr(AriadneService, '_instance', None, raising=False)
    return cfg_path


def _synthetic_source(root):
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg' / 'a.py').write_text('def a():\n    return 1\n')
    (root / 'pkg' / 'b.py').write_text('def b():\n    return 2\n')


async def test_onboarding_tool_chain_over_real_mcp_session(tmp_path):
    import ariadne_mcp.server as mcp_server

    src = tmp_path / 'proj'
    _synthetic_source(src)

    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        await session.initialize()

        # source_add — real config write
        r = await session.call_tool('ariadne_source_add', {'name': 'proj', 'path': str(src)})
        assert not r.isError, r.content
        d = r.structuredContent
        assert d['source'] == 'proj' and d['created'] is True

        # discover — real filesystem walk + manifest write
        r = await session.call_tool('ariadne_discover', {'source': 'proj'})
        assert not r.isError, r.content
        langs = {lc['language'] for lc in r.structuredContent['languages']}
        assert 'python' in langs
        assert r.structuredContent['manifest_written'] is True

        # estimate — real (no-LLM) cost model over the walked files
        r = await session.call_tool(
            'ariadne_estimate', {'source': 'proj', 'model': 'claude-opus-4-8'})
        assert not r.isError, r.content
        assert r.structuredContent['file_count'] == 2
        assert r.structuredContent['total_cost_usd'] > 0

        # list_sources — the source we added is present
        r = await session.call_tool('ariadne_list_sources', {})
        assert not r.isError, r.content
        assert 'proj' in {s['name'] for s in r.structuredContent['sources']}


async def test_onboard_tool_over_session_streams_progress(tmp_path, monkeypatch):
    import ariadne_mcp.server as mcp_server
    from cli.onboard_pipeline import OnboardResult

    src = tmp_path / 'proj'
    (src / 'pkg').mkdir(parents=True)
    (src / 'pkg' / 'a.py').write_text('def a():\n    return 1\n')

    # Stub the paid pipeline (no LLM): fire one progress event, return counts.
    # Signature mirrors run_onboard_pipeline (incl. include_free_phases —
    # the tool sets it; test_mcp_onboard.py's fake carries it the same way).
    async def fake_pipeline(source, model, doc_types, *, mode='live',
                            concurrency=None, progress=None, db_path=None,
                            verbose=False, include_free_phases=False):
        if progress is not None:
            await progress('Generating documentation', 2, 3)
        return OnboardResult(docs_written=5, themes_found=2, themes_ok=True)

    monkeypatch.setattr('cli.onboard_pipeline.run_onboard_pipeline', fake_pipeline)

    seen: list = []

    async def on_progress(progress, total=None, message=None):
        seen.append(message)

    async with create_connected_server_and_client_session(mcp_server.mcp) as session:
        await session.initialize()
        await session.call_tool('ariadne_source_add', {'name': 'proj', 'path': str(src)})

        # Populate the catalog with file_index docs — the REAL "files indexed"
        # signal. Deliberately 2, distinct from the 1-file filesystem walk, so
        # the assertion proves files_indexed reflects what was actually
        # cataloged (the in-memory server shares this process's service
        # singleton, so the docs are visible to the onboard tool).
        _lib = AriadneService.get().library
        for i in range(2):
            _lib.add_document(
                content_type='catalog',
                title=f'file_index:proj:mod{i}.py',
                content=f'Catalog index for mod{i}.py -- 1 elements.',
                source_files=[f'proj/mod{i}.py'],
                metadata={'kind': 'file_index', 'source_name': 'proj',
                          'language': 'python'},
                source_name='proj',
            )

        r = await session.call_tool(
            'ariadne_onboard', {'source': 'proj'}, progress_callback=on_progress)
        assert not r.isError, r.content
        d = r.structuredContent
        assert d['docs_written'] == 5          # from the (stubbed) pipeline
        assert d['themes_found'] == 2
        # Real cataloged-file count (2 file_index docs), NOT the 1-file walk.
        assert d['files_indexed'] == 2
        assert 'coverage_percent' in d

    # the tool's ctx.report_progress reached the client over the MCP protocol
    assert any('Generating' in (m or '') for m in seen), \
        f'expected a progress notification over MCP; got {seen}'
