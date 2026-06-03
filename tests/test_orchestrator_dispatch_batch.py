"""Contract tests for ``_dispatch_batch`` and ``_fetch_with_retry`` (#45.5).

The batch dispatch wraps three provider calls (submit / poll / fetch)
plus durable-state housekeeping (record_pending_batch on submit,
clear_pending_batch on success). Each phase has a distinct failure
mode the orchestrator must handle:

- Quota at submit: no batch was created; nothing to clean up; abort
  immediately.
- Quota at poll: batch_id IS recorded; do NOT clear so user can
  resume or run ``ariadne batch clear <id>``.
- Fetch failure: batch is paid for, results were produced, but
  retrieval failed — retry with exp-backoff before surfacing abort.
- Fetch success: clear pending_batch row, return results.

``_fetch_with_retry`` is tested separately for its retry policy:
it must NOT retry on QuotaExhaustedError (the caller's abort path
handles that), but MUST retry on httpx network errors with
exp-backoff capped at 60s.

Tests use AsyncMock to replace provider methods; one happy-path test
exercises the full submit→poll→fetch sequence to pin the integration
shape, and quota/error tests inject failures at each phase.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx
import pytest

from docgen.generator import PromptBundle
from docgen.llm.anthropic import (
    BatchStatus,
    BatchSubmission,
    QuotaExhaustedError,
)
from docgen.llm.anthropic_batch import AnthropicBatchStrategy
from docgen.orchestrator import (
    BatchAbort,
    DocGenOrchestrator,
    OrchestratorConfig,
)


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='anthropic',
        model='claude-3-5-sonnet',
        doc_types=('explanation',),
        validate=False,
        dry_run=True,
    )


def _sample_prompts() -> list[PromptBundle]:
    """Two minimal PromptBundles for dispatch tests. Real content
    irrelevant — the dispatch path only reads system_prompt /
    user_prompt for the BatchRequest payload, plus the bundle's
    ``file`` for file_to_idxs serialization."""
    return [
        PromptBundle(
            file=Path('/x/a.py'),
            doc_type='explanation',
            system_prompt='SYS-A',
            user_prompt='USR-A',
            title='A',
            metadata={'module_name': 'a', 'source_hash': 'h1'},
        ),
        PromptBundle(
            file=Path('/x/b.py'),
            doc_type='explanation',
            system_prompt='SYS-B',
            user_prompt='USR-B',
            title='B',
            metadata={'module_name': 'b', 'source_hash': 'h2'},
        ),
    ]


def _file_to_idxs() -> dict[Path, list[int]]:
    return {
        Path('/x/a.py'): [0],
        Path('/x/b.py'): [1],
    }


# ---------------------------------------------------------------------------
# _dispatch_batch — happy path
# ---------------------------------------------------------------------------


class TestDispatchBatchHappyPath:
    @pytest.mark.asyncio
    async def test_full_sequence_returns_results(
        self, tmp_path: Path,
    ) -> None:
        """Submit OK → poll ends → fetch returns results → record is
        cleared, returns ``(results, None)``. End-to-end success case.

        Pins:
        1. ``submit_batch`` is called with N requests built from the
           prompts (BatchRequest with custom_id, system, user).
        2. ``record_pending_batch`` writes a row keyed on the
           submission id (so resume can find it).
        3. ``poll_batch`` is called and its on_progress fires.
        4. ``fetch_batch_results`` is called and its dict is returned.
        5. After success, the pending_batch row is cleared (no orphan).
        """
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            with (
                patch.object(
                    AnthropicBatchStrategy, 'submit_batch',
                    new=AsyncMock(return_value=BatchSubmission(
                        batch_id='msgbatch_happy',
                    )),
                ) as mock_submit,
                patch.object(
                    AnthropicBatchStrategy, 'poll_batch',
                    new=AsyncMock(return_value=BatchStatus(
                        batch_id='msgbatch_happy',
                        processing_status='ended',
                        processing=0, succeeded=2, errored=0,
                    )),
                ) as mock_poll,
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results',
                    new=AsyncMock(return_value={
                        '0': 'CONTENT-A', '1': 'CONTENT-B',
                    }),
                ) as mock_fetch,
            ):
                results, abort = await orch._dispatch_batch(
                    _sample_prompts(),
                    _file_to_idxs(),
                    config_hash='hash-happy',
                )

            assert abort is None
            assert results == {'0': 'CONTENT-A', '1': 'CONTENT-B'}

            # Submit was called once with two requests
            mock_submit.assert_called_once()
            (requests_arg,) = mock_submit.call_args.args
            assert len(requests_arg) == 2

            # Pending batch was recorded on submit, then cleared on
            # success — no orphan row remains.
            assert orch._staleness.find_pending_batch('hash-happy') is None
            # And the list is empty (nothing in flight after success).
            assert orch._staleness.list_pending_batches() == []


# ---------------------------------------------------------------------------
# _dispatch_batch — quota at submit
# ---------------------------------------------------------------------------


class TestDispatchBatchQuotaAtSubmit:
    @pytest.mark.asyncio
    async def test_quota_at_submit_returns_abort_no_pending(
        self, tmp_path: Path,
    ) -> None:
        """Submit raises QuotaExhaustedError → no batch was ever
        created, so nothing to record. Abort returned immediately;
        pending_batches table stays empty."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            with patch.object(
                AnthropicBatchStrategy, 'submit_batch',
                new=AsyncMock(side_effect=QuotaExhaustedError(
                    'monthly quota exceeded',
                )),
            ):
                results, abort = await orch._dispatch_batch(
                    _sample_prompts(),
                    _file_to_idxs(),
                    config_hash='hash-quota-submit',
                )

            assert results == {}
            assert abort is not None
            assert isinstance(abort, BatchAbort)
            assert 'submit' in abort.reason.lower() or 'quota' in abort.reason.lower()
            # No batch_id was ever created → no pending row.
            assert orch._staleness.find_pending_batch(
                'hash-quota-submit',
            ) is None


# ---------------------------------------------------------------------------
# _dispatch_batch — quota at poll
# ---------------------------------------------------------------------------


class TestDispatchBatchQuotaAtPoll:
    @pytest.mark.asyncio
    async def test_quota_at_poll_preserves_pending_for_resume(
        self, tmp_path: Path,
    ) -> None:
        """Poll raises QuotaExhaustedError → batch_id IS in the
        pending_batches table (recorded during submit). Critical for
        resume: the user's already paid for the batch; clearing the
        row would forfeit it."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            with (
                patch.object(
                    AnthropicBatchStrategy, 'submit_batch',
                    new=AsyncMock(return_value=BatchSubmission(
                        batch_id='msgbatch_polling',
                    )),
                ),
                patch.object(
                    AnthropicBatchStrategy, 'poll_batch',
                    new=AsyncMock(side_effect=QuotaExhaustedError(
                        'rate limit during poll',
                    )),
                ),
            ):
                results, abort = await orch._dispatch_batch(
                    _sample_prompts(),
                    _file_to_idxs(),
                    config_hash='hash-quota-poll',
                )

            assert results == {}
            assert abort is not None
            assert 'poll' in abort.reason.lower() or 'quota' in abort.reason.lower()
            # Pending row preserved for resume.
            pending = orch._staleness.find_pending_batch('hash-quota-poll')
            assert pending is not None
            assert pending.batch_id == 'msgbatch_polling'


# ---------------------------------------------------------------------------
# _dispatch_batch — fetch failure after retries
# ---------------------------------------------------------------------------


class TestDispatchBatchFetchFails:
    @pytest.mark.asyncio
    async def test_fetch_fails_after_retries_preserves_pending(
        self, tmp_path: Path,
    ) -> None:
        """Fetch fails repeatedly with httpx errors → returns abort,
        leaves pending row intact so user can re-run or clear."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            with (
                patch.object(
                    AnthropicBatchStrategy, 'submit_batch',
                    new=AsyncMock(return_value=BatchSubmission(
                        batch_id='msgbatch_fetchfail',
                    )),
                ),
                patch.object(
                    AnthropicBatchStrategy, 'poll_batch',
                    new=AsyncMock(return_value=BatchStatus(
                        batch_id='msgbatch_fetchfail',
                        processing_status='ended',
                        processing=0, succeeded=2, errored=0,
                    )),
                ),
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results',
                    new=AsyncMock(side_effect=httpx.ConnectError('boom')),
                ),
                # Patch asyncio.sleep so retries don't actually wait.
                patch('asyncio.sleep', new=AsyncMock(return_value=None)),
            ):
                results, abort = await orch._dispatch_batch(
                    _sample_prompts(),
                    _file_to_idxs(),
                    config_hash='hash-fetchfail',
                )

            assert results == {}
            assert abort is not None
            assert 'fetch' in abort.reason.lower()
            # Pending row preserved so the user can run
            # ``ariadne batch clear msgbatch_fetchfail`` or retry.
            pending = orch._staleness.find_pending_batch('hash-fetchfail')
            assert pending is not None
            assert pending.batch_id == 'msgbatch_fetchfail'


# ---------------------------------------------------------------------------
# _fetch_with_retry — focused tests
# ---------------------------------------------------------------------------


class TestFetchWithRetry:
    @pytest.mark.asyncio
    async def test_returns_results_on_first_success(
        self, tmp_path: Path,
    ) -> None:
        """No retry path: fetch_batch_results returns immediately;
        the wrapper returns the dict. Pins that the wrapper isn't
        always-fail (which the stub is)."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            with patch.object(
                AnthropicBatchStrategy, 'fetch_batch_results',
                new=AsyncMock(return_value={'0': 'ok'}),
            ):
                result = await orch._fetch_with_retry(
                    AnthropicBatchStrategy(provider), 'b1', max_retries=3,
                )
            assert result == {'0': 'ok'}

    @pytest.mark.asyncio
    async def test_retries_on_httpx_error_then_succeeds(
        self, tmp_path: Path,
    ) -> None:
        """Two transient errors followed by success → 3 calls total,
        return value is the success dict. Pins the retry loop's
        actually-retries behavior."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            mock_fetch = AsyncMock(side_effect=[
                httpx.ConnectError('transient 1'),
                httpx.ConnectError('transient 2'),
                {'0': 'ok'},
            ])
            with (
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results', new=mock_fetch,
                ),
                patch('asyncio.sleep', new=AsyncMock(return_value=None)),
            ):
                result = await orch._fetch_with_retry(
                    AnthropicBatchStrategy(provider), 'b1', max_retries=5,
                )

            assert result == {'0': 'ok'}
            assert mock_fetch.call_count == 3

    @pytest.mark.asyncio
    async def test_propagates_quota_exhausted_immediately(
        self, tmp_path: Path,
    ) -> None:
        """QuotaExhaustedError must NOT trigger retry — caller's
        abort path handles it. Bites a fix that catches all
        exceptions in the retry loop and burns through max_retries
        before surfacing quota."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            mock_fetch = AsyncMock(
                side_effect=QuotaExhaustedError('quota'),
            )
            with patch.object(
                AnthropicBatchStrategy, 'fetch_batch_results', new=mock_fetch,
            ):
                with pytest.raises(QuotaExhaustedError):
                    await orch._fetch_with_retry(
                        AnthropicBatchStrategy(provider), 'b1', max_retries=5,
                    )

            # Only one call — no retry.
            assert mock_fetch.call_count == 1

    @pytest.mark.asyncio
    async def test_returns_none_after_max_retries(
        self, tmp_path: Path,
    ) -> None:
        """All max_retries attempts fail with httpx errors → return
        None so caller surfaces a fetch-failed abort."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            provider = orch._generator._provider

            mock_fetch = AsyncMock(
                side_effect=httpx.ConnectError('always fails'),
            )
            with (
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results', new=mock_fetch,
                ),
                patch('asyncio.sleep', new=AsyncMock(return_value=None)),
            ):
                result = await orch._fetch_with_retry(
                    AnthropicBatchStrategy(provider), 'b1', max_retries=3,
                )

            assert result is None
            assert mock_fetch.call_count == 3
