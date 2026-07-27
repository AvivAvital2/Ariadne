"""Contract test for the ``ariadne_onboard`` MCP tool — the structured
"Generate" step (Step 5) of the onboarding wizard.

The tool runs the PAID phases via ``run_onboard_pipeline`` and reports the
"ready" stats. Here the pipeline is faked (no LLM, no real generation), but
coverage is computed against a real synthetic source, so the response reconciles
pipeline output (docs/themes) with filesystem-derived coverage (files/percent).

A single evolving test, one demand at a time:

  D1 — onboarding a source returns an OnboardResponse whose pipeline stats
       (docs_written, themes_found, themes_ok) come from the pipeline, whose
       files_indexed is the REAL count of catalogued files (file_index docs),
       NOT a filesystem walk, and whose coverage_percent is computed from the
       real source.
  D2 — the tool's progress callback bridges each phase to ctx.report_progress.
  D3 — batch=True maps to mode='batch' reaching the pipeline.
  D4 — explicit doc_types are passed to the pipeline (as a tuple) and echoed.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from ariadne_mcp.server_admin import ariadne_onboard, ariadne_source_add
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


class _FakeCtx:
    """Records report_progress calls (the SSE-bound progress bridge)."""

    def __init__(self) -> None:
        self.progress: list[tuple[float, float | None, str | None]] = []

    async def report_progress(self, progress, total=None, message=None):
        self.progress.append((progress, total, message))

    async def info(self, *a, **k):
        pass


async def test_onboard_tool_evolves_through_contract(monkeypatch, tmp_path):
    from cli.onboard_pipeline import OnboardResult

    src = tmp_path / 'proj'
    (src / 'pkg').mkdir(parents=True)
    (src / 'pkg' / 'a.py').write_text('def a():\n    return 1\n')
    (src / 'pkg' / 'b.py').write_text('def b():\n    return 2\n')
    await ariadne_source_add('proj', path=str(src))

    # Fake the paid pipeline (no LLM): record its inputs, fire progress once,
    # return fixed counts. Patched where the service imports it from.
    calls: list[dict] = []

    async def fake_pipeline(source, model, doc_types, *, mode='live',
                            concurrency=None, progress=None, db_path=None,
                            verbose=False, include_free_phases=False):
        calls.append({'source': source, 'model': model, 'doc_types': doc_types,
                      'mode': mode, 'concurrency': concurrency,
                      'include_free_phases': include_free_phases})
        if progress is not None:
            await progress('Generating documentation', 2, 3)
        return OnboardResult(docs_written=5, themes_found=2, themes_ok=True)

    monkeypatch.setattr(
        'cli.onboard_pipeline.run_onboard_pipeline', fake_pipeline)

    # Populate the catalog with file_index docs — the REAL "files indexed"
    # signal (catalog-sync writes one per catalogued source file). Deliberately
    # 3, distinct from the 2-file filesystem walk, so the assertion proves
    # files_indexed reflects what was actually cataloged, not a directory walk.
    # Synthetic source_files that don't match the real tree, so coverage stays 0%.
    _lib = AriadneService.get().library
    for i in range(3):
        _lib.add_document(
            content_type='catalog',
            title=f'file_index:proj:mod{i}.py',
            content=f'Catalog index for mod{i}.py -- 1 elements.',
            source_files=[f'proj/mod{i}.py'],
            metadata={'kind': 'file_index', 'source_name': 'proj',
                      'language': 'python'},
            source_name='proj',
        )

    ctx = _FakeCtx()

    # ---- D1: response reconciles pipeline stats + real coverage ------
    resp = await ariadne_onboard('proj', model='claude-opus-4-8', ctx=ctx)
    assert resp.source == 'proj'
    assert resp.model == 'claude-opus-4-8'
    assert resp.mode == 'live'
    assert resp.docs_written == 5          # from the pipeline
    assert resp.themes_found == 2
    assert resp.themes_ok is True
    # files_indexed is the REAL count of catalogued files (3 file_index docs),
    # NOT the 2-file filesystem walk that coverage() performs.
    assert resp.files_indexed == 3
    # coverage computed on the real synthetic source: 2 python files, none
    # documented in a fresh library.
    assert resp.coverage_percent == 0.0
    assert 'explanation' in resp.doc_types  # default set echoed
    # The tool runs a COMPLETE onboard: it must request the free phases
    # (discover → index → catalog-sync) the wizard otherwise never runs, so a
    # SCIP-routed source gets a built index and a populated catalog.
    assert calls[-1]['include_free_phases'] is True

    # ---- D2: progress bridged to ctx.report_progress ----------------
    assert (2, 3, 'Generating documentation') in ctx.progress

    # ---- D3: batch=True → mode='batch' reaches the pipeline ---------
    resp_batch = await ariadne_onboard('proj', batch=True, ctx=ctx)
    assert resp_batch.mode == 'batch'
    assert calls[-1]['mode'] == 'batch'

    # ---- D4: explicit doc_types passed through (tuple) + echoed -----
    resp_types = await ariadne_onboard(
        'proj', doc_types=['explanation', 'qa'], ctx=ctx)
    assert resp_types.doc_types == ['explanation', 'qa']
    assert calls[-1]['doc_types'] == ('explanation', 'qa')
