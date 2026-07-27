"""Batched validation-retry (the 'right' fix for #45 Decision 1a).

Batch mode used to drop validation failures with zero retries (to protect the
50% discount). Instead, we now re-submit just the failing prompts as fresh
batches — up to ``max_batch_validation_retries`` rounds — recovering most via
sampling variance while staying in batch. The summary bills the retry rounds
(estimated). Retry batches are dispatched WITHOUT the resume machinery, so a
crash can't leave a subset batch masquerading as the main one.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

import docgen.orchestrator
from docgen.llm.anthropic import BatchStatus, BatchSubmission
from docgen.llm.anthropic_batch import AnthropicBatchStrategy
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

_VALID_DOC = (
    '# Title\n\n'
    'A documentation paragraph long enough to satisfy the validator, with '
    'multiple sentences and no unclosed code blocks.\n\n'
    '## Section\n\n'
    'More documentation content here, again with enough length and structure '
    'to pass the content checks comfortably.\n'
)
_INVALID_DOC = 'too short'  # fails the validator's minimum-length check


def _config(tmp_path: Path, *, max_retries: int = 3) -> OrchestratorConfig:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'a.py').write_text(
        dedent('"""a."""\ndef foo():\n    pass\n'), encoding='utf-8')
    return OrchestratorConfig(
        source_path=src,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='anthropic',
        model='claude-sonnet-5',
        doc_types=('explanation',),
        validate=True,
        dry_run=True,
        batch_mode='always',
        auto_batch_threshold=1,
        inject_crossrefs=False,
        themes_enabled=False,
        max_batch_validation_retries=max_retries,
    )


def _patches(fetch_mock):
    """submit/poll are constant; fetch is the caller-supplied stateful mock."""
    return (
        patch.object(AnthropicBatchStrategy, 'submit_batch',
                     new=AsyncMock(return_value=BatchSubmission(batch_id='b'))),
        patch.object(AnthropicBatchStrategy, 'poll_batch',
                     new=AsyncMock(return_value=BatchStatus(
                         batch_id='b', processing_status='ended',
                         processing=0, succeeded=1, errored=0))),
        patch.object(AnthropicBatchStrategy, 'fetch_batch_results',
                     new=fetch_mock),
    )


async def _run(config, fetch_side_effect):
    fetch_mock = AsyncMock(side_effect=fetch_side_effect)
    submit, poll, fetch = _patches(fetch_mock)
    async with DocGenOrchestrator(config) as orch:
        with submit, poll, fetch:
            result = await orch.run()
    return result, fetch_mock


@pytest.mark.asyncio
async def test_retry_recovers_failed_doc_and_bills(tmp_path, monkeypatch):
    monkeypatch.setattr(docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True)
    # round 0 → invalid; retry round 1 → valid (recovered).
    result, fetch = await _run(_config(tmp_path), [{'0': _INVALID_DOC}, {'0': _VALID_DOC}])
    assert result.validation_initial_failures == 1
    assert result.validation_retry_attempts == 1
    assert result.validation_recovered == 1
    assert result.validation_retry_cost_usd > 0.0
    assert fetch.call_count == 2  # initial + one retry round


@pytest.mark.asyncio
async def test_retry_exhausts_after_max_rounds_and_bills_each(tmp_path, monkeypatch):
    monkeypatch.setattr(docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True)
    # always invalid: round 0 + 3 retries, never recovers.
    result, fetch = await _run(_config(tmp_path, max_retries=3), [{'0': _INVALID_DOC}] * 4)
    assert result.validation_initial_failures == 1
    assert result.validation_retry_attempts == 3      # capped at max
    assert result.validation_recovered == 0
    assert result.validation_retry_cost_usd > 0.0
    assert fetch.call_count == 4  # initial + 3 retries


@pytest.mark.asyncio
async def test_zero_max_disables_retry_but_still_reports_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True)
    result, fetch = await _run(_config(tmp_path, max_retries=0), [{'0': _INVALID_DOC}])
    assert result.validation_initial_failures == 1
    assert result.validation_retry_attempts == 0
    assert result.validation_recovered == 0
    assert result.validation_retry_cost_usd == 0.0
    assert fetch.call_count == 1  # no retry round
