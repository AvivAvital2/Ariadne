"""Theme summarization must use the RUN's model, not the global default.

A spool build passes ``--model <spools_model>`` (e.g. claude-sonnet-5) to
generate, but post-processing's theme summarization dropped that on the floor
and fell back to the global model (opus). Regression: the model threads through
``_post_process`` → ``refresh_themes`` → ``generate_themes`` → ``summarize_theme``.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

import docgen.themes
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


@pytest.mark.asyncio
async def test_post_process_themes_use_the_run_model(tmp_path):
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.py').write_text('"""a."""\n', encoding='utf-8')
    config = OrchestratorConfig(
        source_path=src,
        db_path=tmp_path / 'a.db',
        staleness_db_path=tmp_path / 's.db',
        api_key='test-not-used',
        provider='anthropic',
        model='claude-sonnet-5',      # the run's model (as a spool build sets)
        doc_types=('explanation',),
        validate=False,
        dry_run=False,                # dry_run short-circuits _post_process
        themes_enabled=True,
        inject_crossrefs=False,
    )
    spy = AsyncMock(return_value={'path': 'themes', 'summarized': 0})
    async with DocGenOrchestrator(config) as orch:
        with patch.object(docgen.themes, 'refresh_themes', spy):
            await orch._post_process([])

    assert spy.await_count == 1
    assert spy.await_args.kwargs['summarize_kwargs']['model'] == 'claude-sonnet-5'


@pytest.mark.asyncio
async def test_themes_build_phase_uses_the_run_model(tmp_path):
    """The standalone `themes build` phase (run by the spool onboard) must also
    honor the run's model, not fall back to the global default."""
    import argparse

    import cli.themes_cmd as themes_cmd

    spy = AsyncMock(return_value={
        'summarized': 0, 'incoherent': 0, 'failed': 0, 'total_dirty': 0})
    args = argparse.Namespace(
        db=str(tmp_path / 't.db'), source='x', themes_action='build',
        model='claude-sonnet-5', concurrency=3, batch=False, quiet=True,
    )
    with patch.object(docgen.themes, 'refresh_themes', spy):
        await themes_cmd.cmd_themes_build(args)

    assert spy.await_count == 1
    assert spy.await_args.kwargs['summarize_kwargs']['model'] == 'claude-sonnet-5'
