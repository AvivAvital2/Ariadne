"""Contract tests for the batch fork in ``run()`` + ``_run_batch``
+ first-run prompt + resume (#45.8).

This is the integration commit that ties #45.4 (collect),
#45.5 (dispatch), #45.6 (assemble), and #45.7 (confirm_callback)
together via a fork in ``DocGenOrchestrator.run()``. The fork
fires when:
1. ``BATCH_DISPATCH_IMPLEMENTED`` is True (gated until #45.9).
2. ``resolve_batch_decision`` says batch (provider, batch_mode,
   threshold).

The fork:
- Computes ``config_hash`` from the run config.
- Calls ``find_pending_batch(config_hash)``. If a match exists,
  routes to resume (fetch the in-flight batch instead of submitting
  a new one).
- Otherwise calls ``confirm_callback`` if wired. Declined → returns
  aborted PipelineResult, no submit.
- Otherwise calls ``_run_batch`` which orchestrates collect →
  dispatch → assemble → post-process.

Tests pin:
- Fork routes to ``_run_batch`` when batch resolves + flag on.
- Fork stays on streaming path when flag off (gate downgrades).
- End-to-end happy path: collect+dispatch+assemble landed.
- ``confirm_callback`` fires before submit; declined → aborted with
  no submit.
- Resume picks up matching config_hash; ignores non-matching.

All tests monkey-patch ``BATCH_DISPATCH_IMPLEMENTED = True`` since
the production default stays False until #45.9.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

import docgen.orchestrator
from docgen.generator import DocGenerator, PromptBundle
from docgen.llm.anthropic import (
    BatchStatus,
    BatchSubmission,
)
from docgen.llm.anthropic_batch import AnthropicBatchStrategy
from docgen.orchestrator import (
    DocGenOrchestrator,
    OrchestratorConfig,
    PipelineResult,
)


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


_VALID_DOC = (
    '# Title\n\n'
    'A documentation paragraph long enough to satisfy the validator. '
    'Multiple sentences. No unclosed code blocks.\n\n'
    '## Section\n\n'
    'More documentation content here, again with enough length and '
    'structure to pass content checks.\n'
)


def _make_source_tree(tmp_path: Path) -> Path:
    src = tmp_path / 'src'
    src.mkdir()
    _write(src / 'a.py', '"""a."""\ndef foo(): pass\n')
    _write(src / 'b.py', '"""b."""\ndef bar(): pass\n')
    return src


def _make_config(
    tmp_path: Path,
    source: Path,
    *,
    provider: str = 'anthropic',
    batch_mode: str = 'always',
    auto_threshold: int = 200,
    dry_run: bool = True,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=source,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider=provider,
        model='claude-3-5-sonnet' if provider == 'anthropic' else 'gpt-5.2',
        doc_types=('explanation',),
        validate=False,
        dry_run=dry_run,
        batch_mode=batch_mode,
        auto_batch_threshold=auto_threshold,
        inject_crossrefs=False,
        themes_enabled=False,
    )


# ---------------------------------------------------------------------------
# Fork in run()
# ---------------------------------------------------------------------------


class TestRunBatchFork:
    @pytest.mark.asyncio
    async def test_batch_resolves_calls_run_batch(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Anthropic + batch_mode=always + flag ON → ``_run_batch``
        called instead of streaming. Pins the fork's positive
        branch."""
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source)

        fake_result = PipelineResult(
            files_processed=2, files_skipped=0,
            docs_created=2, docs_failed=0,
        )

        with patch.object(
            DocGenOrchestrator, '_run_batch',
            new=AsyncMock(return_value=fake_result),
        ) as mock_run_batch:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        mock_run_batch.assert_called_once()
        # The fork returns whatever _run_batch returned.
        assert result is fake_result

    @pytest.mark.asyncio
    async def test_flag_off_stays_on_streaming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Anthropic + batch_mode=always but flag OFF (default) →
        gate downgrades batch=True to sync; ``_run_batch`` NOT
        called. Pins that the gate is honored at the fork."""
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', False,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source)

        with (
            patch.object(
                DocGenOrchestrator, '_run_batch',
                new=AsyncMock(),
            ) as mock_run_batch,
            patch.object(
                DocGenerator, '_call_llm',
                new=AsyncMock(return_value=_VALID_DOC),
            ),
        ):
            async with DocGenOrchestrator(config) as orch:
                await orch.run()

        mock_run_batch.assert_not_called()

    @pytest.mark.asyncio
    async def test_openai_provider_uses_batch_fork(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """OpenAI + batch_mode=always + flag ON → resolver returns batch
        (OpenAI's Batch API is now wired via OpenAIBatchStrategy), so
        ``_run_batch`` IS taken. Pins provider eligibility at the fork:
        both wired providers batch; only a backend-less provider stays sync."""
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source, provider='openai')

        with (
            patch.object(
                DocGenOrchestrator, '_run_batch',
                new=AsyncMock(),
            ) as mock_run_batch,
            patch.object(
                DocGenerator, '_call_llm',
                new=AsyncMock(return_value=_VALID_DOC),
            ),
        ):
            async with DocGenOrchestrator(config) as orch:
                await orch.run()

        mock_run_batch.assert_called_once()


# ---------------------------------------------------------------------------
# _run_batch end-to-end through run()
# ---------------------------------------------------------------------------


class TestRunBatchEndToEnd:
    @pytest.mark.asyncio
    async def test_full_batch_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Full batch run: 2 files → submit returns batch_id → poll
        ends → fetch returns 2 results → assemble → returns
        PipelineResult with files_processed=2.

        Pins the integration of #45.4-7: collect, dispatch, assemble
        all run in sequence via _run_batch."""
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source)

        progress_msgs: list[str] = []
        async with DocGenOrchestrator(config) as orch:
            orch.progress_callback = (
                lambda msg, cur, tot: progress_msgs.append(msg)
            )
            provider = orch._generator._provider
            with (
                patch.object(
                    AnthropicBatchStrategy, 'submit_batch',
                    new=AsyncMock(return_value=BatchSubmission(
                        batch_id='msgbatch_e2e',
                    )),
                ) as mock_submit,
                patch.object(
                    AnthropicBatchStrategy, 'poll_batch',
                    new=AsyncMock(return_value=BatchStatus(
                        batch_id='msgbatch_e2e',
                        processing_status='ended',
                        processing=0, succeeded=2, errored=0,
                    )),
                ),
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results',
                    new=AsyncMock(return_value={
                        '0': _VALID_DOC, '1': _VALID_DOC,
                    }),
                ),
            ):
                result = await orch.run()

        # Submit was actually called — without the fork, run() would
        # have gone through streaming and never touched submit_batch.
        assert mock_submit.call_count == 1
        assert isinstance(result, PipelineResult)
        assert result.aborted is False
        # The store/embed phase must emit its own progress — otherwise the
        # bar looks frozen on "downloading results" for the whole (slow)
        # store. Pin that a per-file storing message is emitted.
        assert any('Storing' in m for m in progress_msgs), progress_msgs


# ---------------------------------------------------------------------------
# First-run prompt
# ---------------------------------------------------------------------------


class TestRunBatchFirstRunPrompt:
    @pytest.mark.asyncio
    async def test_callback_fires_before_submit_when_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """confirm_callback is called once before submit_batch.
        Accepted → submit proceeds normally."""
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source)

        captured_msgs: list[str] = []

        async def confirm(msg: str) -> bool:
            captured_msgs.append(msg)
            return True

        async with DocGenOrchestrator(config) as orch:
            orch.confirm_callback = confirm
            provider = orch._generator._provider

            with (
                patch.object(
                    AnthropicBatchStrategy, 'submit_batch',
                    new=AsyncMock(return_value=BatchSubmission(
                        batch_id='msgbatch_prompted',
                    )),
                ) as mock_submit,
                patch.object(
                    AnthropicBatchStrategy, 'poll_batch',
                    new=AsyncMock(return_value=BatchStatus(
                        batch_id='msgbatch_prompted',
                        processing_status='ended',
                        processing=0, succeeded=2, errored=0,
                    )),
                ),
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results',
                    new=AsyncMock(return_value={
                        '0': _VALID_DOC, '1': _VALID_DOC,
                    }),
                ),
            ):
                await orch.run()

            # Callback fired exactly once.
            assert len(captured_msgs) == 1
            # And submit happened after acceptance.
            mock_submit.assert_called_once()

    @pytest.mark.asyncio
    async def test_declined_aborts_with_no_submit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """confirm_callback returns False → run aborts cleanly,
        submit_batch is NOT called. ``unprocessed_files`` lists all
        files queued so the user can resume after acceptance."""
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source)

        async def deny(msg: str) -> bool:
            return False

        async with DocGenOrchestrator(config) as orch:
            orch.confirm_callback = deny
            provider = orch._generator._provider

            with patch.object(
                AnthropicBatchStrategy, 'submit_batch',
                new=AsyncMock(),
            ) as mock_submit:
                result = await orch.run()

            mock_submit.assert_not_called()
            assert result.aborted is True
            assert 'declin' in result.abort_reason.lower()
            # All queued files end up unprocessed since none were
            # submitted.
            assert len(result.unprocessed_files) == 2


# ---------------------------------------------------------------------------
# Resume from pending batch
# ---------------------------------------------------------------------------


class TestRunBatchResume:
    @pytest.mark.asyncio
    async def test_pending_with_matching_config_skips_submit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A pending batch with matching config_hash → fetch only.
        ``submit_batch`` and ``poll_batch`` NOT called.

        Pins the user-visible value of #45.3's pending_batches
        table: a crash mid-poll doesn't forfeit a paid-for batch."""
        import json

        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        source = _make_source_tree(tmp_path)
        config = _make_config(tmp_path, source)

        async with DocGenOrchestrator(config) as orch:
            # Pre-record a pending batch matching this run's config.
            config_hash = config.config_hash()
            prompts_payload = [
                {
                    'file': str(source / 'a.py'),
                    'doc_type': 'explanation',
                    'system_prompt': 'SYS',
                    'user_prompt': 'USR',
                    'title': 'A',
                    'metadata': {'module_name': 'a', 'source_hash': 'h'},
                },
                {
                    'file': str(source / 'b.py'),
                    'doc_type': 'explanation',
                    'system_prompt': 'SYS',
                    'user_prompt': 'USR',
                    'title': 'B',
                    'metadata': {'module_name': 'b', 'source_hash': 'h'},
                },
            ]
            file_to_idxs_payload = {
                str(source / 'a.py'): [0],
                str(source / 'b.py'): [1],
            }
            orch._staleness.record_pending_batch(
                batch_id='msgbatch_resume',
                prompts_json=json.dumps(prompts_payload),
                file_to_idxs_json=json.dumps(file_to_idxs_payload),
                config_hash=config_hash,
            )

            provider = orch._generator._provider
            with (
                patch.object(
                    AnthropicBatchStrategy, 'submit_batch',
                    new=AsyncMock(),
                ) as mock_submit,
                patch.object(
                    AnthropicBatchStrategy, 'poll_batch',
                    new=AsyncMock(),
                ) as mock_poll,
                patch.object(
                    AnthropicBatchStrategy, 'fetch_batch_results',
                    new=AsyncMock(return_value={
                        '0': _VALID_DOC, '1': _VALID_DOC,
                    }),
                ) as mock_fetch,
            ):
                result = await orch.run()

            mock_submit.assert_not_called()
            mock_poll.assert_not_called()
            mock_fetch.assert_called_once_with('msgbatch_resume')
            # Successful resume clears the pending row.
            assert orch._staleness.find_pending_batch(config_hash) is None
            assert result.aborted is False
            # Resume must still surface cache_stats — it lives on the provider
            # (the strategy borrows + records into it), not on the strategy.
            # Regression guard for the provider→strategy rename.
            assert result.cache_stats is not None
