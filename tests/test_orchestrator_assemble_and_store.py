"""Contract tests for ``_assemble_and_store`` (#45.6).

After the batch fetches results, this method:
1. Maps results back to per-file ``GenerationResult`` via the
   ``file_to_idxs`` map.
2. Wraps each successful response into a ``GeneratedDoc`` via
   ``DocGenerator.assemble_doc``.
3. Validates each doc — failures count as ``docs_failed`` with NO
   retry (per Decision 1a / ``BATCH_VALIDATION_RETRY = False``).
4. Stores docs in the library unless ``dry_run``.
5. Records staleness for stored doc_ids.

Tests follow the existing streaming-path convention from
``tests/test_orchestrator_run_pipeline.py``: use ``dry_run=False`` +
patched ``_store_document`` (returning a fake doc with an ``id``)
when a test needs ``docs_generated > 0``, since the streaming
contract is ``docs_generated = len(doc_ids)``. Use ``dry_run=True``
only when verifying the dry-run skip behavior itself.

Tests pin:
- Happy path: all results valid → docs_generated counts match.
- Partial: some custom_ids missing/None → those count as failed.
- Validation no-retry: invalid content → docs_failed += 1, no LLM
  call (paired with passing-validation test).
- Pre-gen failed files: marked as failures with no doc_ids.
- Dry-run: no library writes, doc_ids stays empty (matches
  streaming).
- BATCH_VALIDATION_RETRY constant pin.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.generator import PromptBundle
from docgen.orchestrator import (
    BATCH_VALIDATION_RETRY,
    DocGenOrchestrator,
    OrchestratorConfig,
)
from docgen.validator import ContentValidator, ValidationResult


_VALID_CONTENT = dedent('''\
    # Sample Doc

    This is a documentation paragraph that satisfies the validator's
    minimum length and structure requirements. It has multiple
    sentences. It does not contain any unclosed code fences or other
    structural issues.

    ## Details

    Here are some details about the topic. The validator's main
    concerns are unclosed code blocks, mismatched headings, and
    very short content — none of which apply here.
    ''')


class _FakeDoc:
    """Stand-in for a real Document — only ``id`` matters here."""

    def __init__(self, id_: str) -> None:
        self.id = id_


def _make_config(
    tmp_path: Path, *, dry_run: bool = True, validate: bool = False,
) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='openai',
        model='gpt-5.2',
        doc_types=('explanation',),
        validate=validate,
        dry_run=dry_run,
    )


def _prompts_for_files(*paths: Path) -> tuple[
    list[PromptBundle], dict[Path, list[int]],
]:
    """Build a flat prompts list (one per file) + file_to_idxs map."""
    prompts: list[PromptBundle] = []
    file_to_idxs: dict[Path, list[int]] = {}
    for i, p in enumerate(paths):
        prompts.append(PromptBundle(
            file=p,
            doc_type='explanation',
            system_prompt=f'SYS-{i}',
            user_prompt=f'USR-{i}',
            title=f'Title {i}',
            metadata={'module_name': f'm{i}', 'source_hash': f'h{i}'},
        ))
        file_to_idxs[p] = [i]
    return prompts, file_to_idxs


# ---------------------------------------------------------------------------
# BATCH_VALIDATION_RETRY constant
# ---------------------------------------------------------------------------


class TestBatchValidationRetryConstant:
    def test_constant_is_false_by_default(self) -> None:
        """Pin the contract: batch dispatch does NOT retry validation
        failures. Bites a future change that flips this to True
        without updating the rationale comment + docstring (and
        without addressing the cost regression that mode change
        introduces)."""
        assert BATCH_VALIDATION_RETRY is False


# ---------------------------------------------------------------------------
# Happy path — dry_run=False + patched _store_document
# ---------------------------------------------------------------------------


class TestAssembleAndStoreHappyPath:
    @pytest.mark.asyncio
    async def test_full_results_produce_expected_per_file_counts(
        self, tmp_path: Path,
    ) -> None:
        """All prompts have content → each file gets docs_generated=1
        (one prompt per file), docs_failed=0. ``dry_run=False`` +
        patched ``_store_document`` so doc_ids accumulate (matches
        streaming's ``docs_generated = len(doc_ids)`` contract)."""
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        # Files must exist on disk — record_documentation_async hashes
        # the file content for its source_record.
        a.write_text('x = 1\n', encoding='utf-8')
        b.write_text('y = 2\n', encoding='utf-8')
        prompts, file_to_idxs = _prompts_for_files(a, b)
        results_by_cid = {'0': _VALID_CONTENT, '1': _VALID_CONTENT}

        config = _make_config(tmp_path, dry_run=False)
        async with DocGenOrchestrator(config) as orch:
            with patch.object(
                DocGenOrchestrator, '_store_document',
                new=AsyncMock(side_effect=[
                    _FakeDoc('id-a'), _FakeDoc('id-b'),
                ]),
            ):
                results, _ = await orch._assemble_and_store(
                    prompts, file_to_idxs, results_by_cid,
                    files_to_process=[a, b],
                    pre_gen_failed=[],
                )

        by_path = {r.source_path: r for r in results}
        assert by_path[a].docs_generated == 1
        assert by_path[a].docs_failed == 0
        assert by_path[b].docs_generated == 1
        assert by_path[b].docs_failed == 0


# ---------------------------------------------------------------------------
# Partial results — missing custom_ids
# ---------------------------------------------------------------------------


class TestAssembleAndStorePartialResults:
    @pytest.mark.asyncio
    async def test_missing_or_none_results_count_as_failed(
        self, tmp_path: Path,
    ) -> None:
        """When fetch returns None for a custom_id, that prompt
        counts as ``docs_failed`` but others still process. Pins
        that one bad result doesn't forfeit the whole file/run."""
        a = tmp_path / 'a.py'
        b = tmp_path / 'b.py'
        a.write_text('x = 1\n', encoding='utf-8')
        b.write_text('y = 2\n', encoding='utf-8')
        prompts, file_to_idxs = _prompts_for_files(a, b)
        # Index 0 succeeded; index 1 failed (None).
        results_by_cid = {'0': _VALID_CONTENT, '1': None}

        config = _make_config(tmp_path, dry_run=False)
        async with DocGenOrchestrator(config) as orch:
            with patch.object(
                DocGenOrchestrator, '_store_document',
                new=AsyncMock(side_effect=[_FakeDoc('id-a')]),
            ):
                results, _ = await orch._assemble_and_store(
                    prompts, file_to_idxs, results_by_cid,
                    files_to_process=[a, b],
                    pre_gen_failed=[],
                )

        by_path = {r.source_path: r for r in results}
        assert by_path[a].docs_generated == 1
        assert by_path[a].docs_failed == 0
        assert by_path[b].docs_generated == 0
        assert by_path[b].docs_failed == 1


# ---------------------------------------------------------------------------
# Validation: no retry on failure (Decision 1a)
# ---------------------------------------------------------------------------


class TestAssembleAndStoreValidationNoRetry:
    @pytest.mark.asyncio
    async def test_validation_pass_records_doc(
        self, tmp_path: Path,
    ) -> None:
        """Paired with the failure test below — passing validation
        produces a docs_generated record. Without this paired test,
        a stub returning empty results would pass the failure test
        and we'd never catch a regression where ALL docs are
        wrongly-rejected."""
        a = tmp_path / 'a.py'
        a.write_text('x = 1\n', encoding='utf-8')
        prompts, file_to_idxs = _prompts_for_files(a)
        results_by_cid = {'0': _VALID_CONTENT}

        config = _make_config(tmp_path, dry_run=False, validate=True)
        async with DocGenOrchestrator(config) as orch:
            # Force PASS — assertion stays on the no-retry contract,
            # not the validator's content rules.
            with (
                patch.object(
                    ContentValidator, 'validate',
                    return_value=ValidationResult(
                        is_valid=True, issues=(),
                    ),
                ),
                patch.object(
                    DocGenOrchestrator, '_store_document',
                    new=AsyncMock(side_effect=[_FakeDoc('id-a')]),
                ),
                patch(
                    'docgen.generator.DocGenerator._call_llm',
                    new=AsyncMock(),
                ) as mock_llm,
            ):
                results, _ = await orch._assemble_and_store(
                    prompts, file_to_idxs, results_by_cid,
                    files_to_process=[a],
                    pre_gen_failed=[],
                )
                # Zero LLM calls during assembly — that's the batch
                # dispatch's main saving.
                assert mock_llm.call_count == 0

        by_path = {r.source_path: r for r in results}
        assert by_path[a].docs_generated == 1
        assert by_path[a].docs_failed == 0

    @pytest.mark.asyncio
    async def test_validation_failure_no_retry_no_llm_call(
        self, tmp_path: Path,
    ) -> None:
        """Validation FAIL → docs_failed += 1 with NO retry call to
        the LLM. Pins Decision 1a / BATCH_VALIDATION_RETRY = False.
        Bites a fix that wires the streaming retry into batch (which
        would silently halve the batch discount the user opted into)."""
        a = tmp_path / 'a.py'
        prompts, file_to_idxs = _prompts_for_files(a)
        results_by_cid = {'0': _VALID_CONTENT}

        config = _make_config(tmp_path, dry_run=False, validate=True)
        async with DocGenOrchestrator(config) as orch:
            with (
                patch.object(
                    ContentValidator, 'validate',
                    return_value=ValidationResult(
                        is_valid=False, issues=(),
                    ),
                ),
                patch.object(
                    DocGenOrchestrator, '_store_document',
                    new=AsyncMock(),
                ) as mock_store,
                patch(
                    'docgen.generator.DocGenerator._call_llm',
                    new=AsyncMock(),
                ) as mock_llm,
            ):
                results, _ = await orch._assemble_and_store(
                    prompts, file_to_idxs, results_by_cid,
                    files_to_process=[a],
                    pre_gen_failed=[],
                )
                # NO retry → no LLM call.
                assert mock_llm.call_count == 0
                # Failed validation → not stored.
                assert mock_store.call_count == 0

        by_path = {r.source_path: r for r in results}
        assert by_path[a].docs_generated == 0
        assert by_path[a].docs_failed == 1


# ---------------------------------------------------------------------------
# Pre-gen failed files
# ---------------------------------------------------------------------------


class TestAssembleAndStorePreGenFailed:
    @pytest.mark.asyncio
    async def test_pre_gen_failed_files_have_failed_count(
        self, tmp_path: Path,
    ) -> None:
        """Files in ``pre_gen_failed`` (e.g. SyntaxError during
        collect) get a GenerationResult with docs_failed = number of
        doc_types. They're not in ``file_to_idxs`` so there are no
        prompts to assemble."""
        good = tmp_path / 'good.py'
        bad = tmp_path / 'bad.py'
        good.write_text('x = 1\n', encoding='utf-8')
        # bad doesn't need to exist — it's in pre_gen_failed so
        # _assemble_and_store skips it without touching the file.
        prompts, file_to_idxs = _prompts_for_files(good)
        results_by_cid = {'0': _VALID_CONTENT}

        config = _make_config(tmp_path, dry_run=False)
        async with DocGenOrchestrator(config) as orch:
            with patch.object(
                DocGenOrchestrator, '_store_document',
                new=AsyncMock(side_effect=[_FakeDoc('id-good')]),
            ):
                results, _ = await orch._assemble_and_store(
                    prompts, file_to_idxs, results_by_cid,
                    files_to_process=[good, bad],
                    pre_gen_failed=[bad],
                )

        by_path = {r.source_path: r for r in results}
        assert by_path[bad].docs_generated == 0
        # One doc_type in config → one failure to count.
        assert by_path[bad].docs_failed == 1
        assert by_path[bad].doc_ids == ()
        # Good file processed normally.
        assert by_path[good].docs_generated == 1


# ---------------------------------------------------------------------------
# Dry run — skip library writes
# ---------------------------------------------------------------------------


class TestAssembleAndStoreDryRun:
    @pytest.mark.asyncio
    async def test_dry_run_skips_storage(
        self, tmp_path: Path,
    ) -> None:
        """``dry_run=True`` → ``_store_document`` is NOT called.
        Matches streaming's behavior: in dry-run, ``doc_ids`` stays
        empty so ``docs_generated = len(doc_ids) = 0``. Pins that
        the dry-run guard exists."""
        a = tmp_path / 'a.py'
        prompts, file_to_idxs = _prompts_for_files(a)
        results_by_cid = {'0': _VALID_CONTENT}

        config = _make_config(tmp_path, dry_run=True)
        async with DocGenOrchestrator(config) as orch:
            with patch.object(
                DocGenOrchestrator, '_store_document',
                new=AsyncMock(),
            ) as mock_store:
                results, _ = await orch._assemble_and_store(
                    prompts, file_to_idxs, results_by_cid,
                    files_to_process=[a],
                    pre_gen_failed=[],
                )
                # Storage NOT called under dry_run.
                assert mock_store.call_count == 0

        by_path = {r.source_path: r for r in results}
        # Matches streaming: docs_generated = len(doc_ids) = 0 in
        # dry_run.
        assert by_path[a].docs_generated == 0
        # And not counted as failed either — the assemble succeeded,
        # we just deliberately skipped storage.
        assert by_path[a].docs_failed == 0
