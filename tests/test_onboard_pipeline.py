"""Contract test for ``run_onboard_pipeline`` — the headless paid-phase core
behind ``ariadne onboard`` and the ``ariadne_onboard`` MCP tool.

A single evolving test, one demand at a time:

  D1 — the three paid phases run in order: catalog-describe → generate → themes.
  D2 — each phase receives the right arguments: the resolved model, the batch
       choice (mode='batch' → describe batch=True + generate batch_mode='always';
       'live' → False/'never'), the concurrency override, and the doc-type CSV.
  D3 — a progress callback fires once per phase with a rising (current, total),
       total == the number of phases, each carrying the phase label.
  D4 — the returned OnboardResult reports the Library's document + theme counts.
  D5 — a themes failure is NON-fatal (pipeline completes, themes_ok False, docs
       still counted, default concurrency applied); a generate (fatal) failure
       raises OnboardError naming the phase, and does NOT run the themes phase.

No LLM calls — the phase commands are faked to record their Namespace and
return a configurable rc; the Library is a synthetic double.
"""
from __future__ import annotations

import pytest

from cli.onboard_pipeline import OnboardError, run_onboard_pipeline


class _FakeLibrary:
    """Records nothing; just answers the two stat reads the pipeline makes."""

    def __init__(self, docs: int, themes: list[str]) -> None:
        self._docs = docs
        self._themes = themes

    def count_documents(self, content_type=None) -> int:
        return self._docs

    def list_themes(self, *, coherent_only: bool = True) -> list[str]:
        return list(self._themes)


@pytest.fixture
def _fakes(monkeypatch):
    """Fake the three phase commands (record their Namespace, return a
    configurable rc) and the Library the pipeline reads stats from."""
    calls: list[tuple[str, object]] = []
    rcs = {'catalog_describe': 0, 'generate': 0, 'themes_build': 0}

    async def _fake_describe(args):
        calls.append(('catalog_describe', args))
        return rcs['catalog_describe']

    async def _fake_generate(args):
        calls.append(('generate', args))
        return rcs['generate']

    async def _fake_themes(args):
        calls.append(('themes_build', args))
        return rcs['themes_build']

    lib = _FakeLibrary(docs=7, themes=['t1', 't2', 't3'])

    monkeypatch.setattr('cli.onboard_pipeline.cmd_catalog_describe', _fake_describe)
    monkeypatch.setattr('cli.onboard_pipeline.cmd_generate', _fake_generate)
    monkeypatch.setattr('cli.onboard_pipeline.cmd_themes_build', _fake_themes)
    monkeypatch.setattr('cli.onboard_pipeline.get_library', lambda db_path=None: lib)
    return calls, rcs, lib


async def test_run_onboard_pipeline_evolves_through_contract(_fakes):
    calls, rcs, lib = _fakes

    progress_events: list[tuple[str, int, int]] = []

    async def _progress(label, current, total):
        progress_events.append((label, current, total))

    # ---- D1 + D2 + D3 + D4: happy path (batch) -----------------------
    result = await run_onboard_pipeline(
        'proj', 'claude-opus-4-8', ('explanation', 'qa'),
        mode='batch', concurrency=6, progress=_progress,
    )

    # D1: phases ran in the canonical order.
    assert [name for name, _ in calls] == [
        'catalog_describe', 'generate', 'themes_build']

    # D2: each phase got the right arguments.
    describe_args = vars(calls[0][1])
    generate_args = vars(calls[1][1])
    themes_args = vars(calls[2][1])
    assert describe_args['model'] == 'claude-opus-4-8'
    assert describe_args['batch'] is True
    assert describe_args['concurrency'] == 6
    assert generate_args['batch_mode'] == 'always'
    assert generate_args['types'] == 'explanation,qa'
    assert generate_args['concurrency'] == 6
    assert themes_args['themes_action'] == 'build'
    assert themes_args['model'] == 'claude-opus-4-8'
    # the themes phase must batch too when the user chose batch — otherwise it
    # silently runs live at full price despite the batch choice.
    assert themes_args.get('batch') is True

    # D3: progress fired once per phase, rising, labelled, total == 3.
    assert [e[0] for e in progress_events] == [
        'Describing catalog elements',
        'Generating documentation',
        'Building themes',
    ]
    assert [(c, t) for _, c, t in progress_events] == [(1, 3), (2, 3), (3, 3)]

    # D4: stats read from the Library.
    assert result.docs_written == 7
    assert result.themes_found == 3
    assert result.themes_ok is True

    # ---- D5: themes failure is non-fatal, live-mode arg mapping ------
    calls.clear()
    progress_events.clear()
    rcs['themes_build'] = 3
    result2 = await run_onboard_pipeline('proj', 'm', ('explanation',), mode='live')
    assert [name for name, _ in calls] == [
        'catalog_describe', 'generate', 'themes_build']
    assert result2.themes_ok is False
    assert result2.docs_written == 7  # a themes failure never discards the docs
    # live mode + default concurrency (describe=4, generate=3)
    assert vars(calls[0][1])['batch'] is False
    assert vars(calls[0][1])['concurrency'] == 4
    assert vars(calls[1][1])['batch_mode'] == 'never'
    assert vars(calls[1][1])['concurrency'] == 3
    assert vars(calls[2][1]).get('batch') is False  # themes live when mode='live'

    # ---- D5b: a fatal (generate) failure raises; themes NOT reached --
    calls.clear()
    rcs['themes_build'] = 0
    rcs['generate'] = 2
    with pytest.raises(OnboardError) as exc:
        await run_onboard_pipeline('proj', 'm', ('explanation',), mode='live')
    # the error names the failing phase and its rc, and carries the rc so
    # the CLI can propagate it
    assert 'Generating documentation' in str(exc.value)
    assert 'rc=2' in str(exc.value)
    assert exc.value.rc == 2
    assert [name for name, _ in calls] == ['catalog_describe', 'generate']
