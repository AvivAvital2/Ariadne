"""Theme summarization runs inside generate's ``_post_process`` (themes are on
by default), so a batch run must hand ``refresh_themes`` a batch strategy there
— otherwise ``--batch`` still summarizes themes live at full price.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import docgen.themes
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _cfg(tmp_path: Path, *, batch_mode: str) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=tmp_path / 'src',
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='anthropic',
        model='claude-3-5-sonnet',
        doc_types=('explanation',),
        validate=False,
        dry_run=False,          # _post_process returns early on dry_run
        batch_mode=batch_mode,
        auto_batch_threshold=200,
        inject_crossrefs=False,  # skip the crossref stage in _post_process
        themes_enabled=True,
    )


def test_themes_batch_strategy_gates_on_batch_mode(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(DocGenOrchestrator, '_batch_strategy', lambda self: 'SENTINEL')
    assert DocGenOrchestrator(_cfg(tmp_path, batch_mode='always'))._themes_batch_strategy() == 'SENTINEL'
    assert DocGenOrchestrator(_cfg(tmp_path, batch_mode='batch'))._themes_batch_strategy() == 'SENTINEL'
    assert DocGenOrchestrator(_cfg(tmp_path, batch_mode='never'))._themes_batch_strategy() is None
    assert DocGenOrchestrator(_cfg(tmp_path, batch_mode='auto'))._themes_batch_strategy() is None


def _capture_refresh(captured: dict):
    async def _fake(library, writer, *, enabled=True, cluster_kwargs=None,
                    summarize_kwargs=None, batch_strategy=None):
        captured['batch_strategy'] = batch_strategy
        return {}
    return _fake


@pytest.mark.asyncio
async def test_post_process_batches_themes_when_batch_mode(tmp_path, monkeypatch) -> None:
    (tmp_path / 'src').mkdir()
    captured: dict = {}
    monkeypatch.setattr(docgen.themes, 'refresh_themes', _capture_refresh(captured))
    monkeypatch.setattr(DocGenOrchestrator, '_batch_strategy', lambda self: 'SENTINEL')

    async with DocGenOrchestrator(_cfg(tmp_path, batch_mode='always')) as orch:
        await orch._post_process([], crossref_progress=None)

    assert captured['batch_strategy'] == 'SENTINEL'   # themes went to the batch path


@pytest.mark.asyncio
async def test_post_process_themes_live_when_not_batch(tmp_path, monkeypatch) -> None:
    (tmp_path / 'src').mkdir()
    captured: dict = {}
    monkeypatch.setattr(docgen.themes, 'refresh_themes', _capture_refresh(captured))
    monkeypatch.setattr(DocGenOrchestrator, '_batch_strategy', lambda self: 'SENTINEL')

    async with DocGenOrchestrator(_cfg(tmp_path, batch_mode='never')) as orch:
        await orch._post_process([], crossref_progress=None)

    assert captured['batch_strategy'] is None          # live per-theme
