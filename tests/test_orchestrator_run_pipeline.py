"""Functional integration tests for ``DocGenOrchestrator.run()``.

Coverage of ``run()`` before this file: 23.7% combined branch+line on
``docgen/orchestrator.py``, with the entire body of ``run()`` (lines
338-527) in the missing column. This file lifts coverage by
exercising the streaming-pipeline behaviors that batch dispatch (#45)
will need to coexist with.

Each test verifies a SPECIFIC FUNCTIONAL BEHAVIOR a real bug would
break — not just "this code path executes":

- File discovery + staleness filtering (force_regenerate, type-aware)
- Concurrent processing within the asyncio.Semaphore limit
- ``QuotaExhaustedError`` abort coordination + ``unprocessed_files``
- Validation retry counters (initial_failures / retry_attempts /
  recovered)
- Pre-generation failure (SyntaxError) handling
- Post-processing skip on abort
- Empty source tree → clean zero-count return
- Single-file target_path → process only that file

LLM is faked at ``DocGenerator._call_llm`` via ``unittest.mock.patch``.
Library / Staleness / ContentValidator / SourceAnalyzer all run for
real on tmp-path artifacts. Storage is skipped via ``dry_run=True``
to avoid the embedding service dependency — counters that don't
require persistence (``files_processed``, ``aborted``, validation
counters) are still meaningful.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.generator import DocGenerator
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


def _make_source_tree(root: Path, files: dict[str, str]) -> Path:
    """Write ``files`` (relative path → source code) under ``root/src``."""
    src = root / 'src'
    src.mkdir(parents=True, exist_ok=True)
    for rel, code in files.items():
        path = src / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        _write(path, code)
    return src


def _make_config(
    tmp_path: Path,
    source_path: Path,
    **overrides,
) -> OrchestratorConfig:
    """Build an OrchestratorConfig with safe test defaults.

    ``dry_run=True`` skips storage so the embedding service is never
    invoked. ``api_key='test-not-used'`` because ``_call_llm`` is
    patched. ``validate=False`` by default — tests that exercise
    validation retry override it.

    ``catalog_only_generator=False`` pins this file to the legacy
    ``SourceAnalyzer``-driven pipeline. The production default flipped
    to True in Phase 2 Change 1, but this file's tests target the
    legacy path's behaviors (SyntaxError → _legacy_generate, etc.)
    which still need coverage until Phase 4 deletes the path. The
    catalog-path equivalents live in test_orchestrator_catalog_flag.py.
    """
    base = dict(
        source_path=source_path,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        dry_run=True,
        api_key='test-not-used',
        provider='openai',
        model='gpt-5.2',
        doc_types=('explanation',),
        concurrency=2,
        validate=False,
        inject_crossrefs=False,
        themes_enabled=False,
        catalog_only_generator=False,
    )
    base.update(overrides)
    return OrchestratorConfig(**base)


# Long enough to clear any "minimum content length" floor in
# ContentValidator. Used as the canned LLM response for happy-path
# tests where validation is off — but the content must still exist
# so the generator surfaces a non-empty doc.
_VALID_DOC_CONTENT = (
    '# Module Documentation\n\n'
    '## Overview\n'
    'This module exposes a single public function, `foo`, which '
    'returns a sentinel value used by the test suite. The function '
    'has no side effects and is thread-safe by virtue of being '
    'pure.\n\n'
    '## Implementation Notes\n'
    'The function is intentionally minimal — its job is to provide '
    'a stable target for fixtures.\n'
)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestHappyPath:
    @pytest.mark.asyncio
    async def test_two_files_one_doc_type_two_llm_calls(
        self, tmp_path: Path,
    ) -> None:
        """Two source files × one doc type → exactly two LLM
        invocations, both files counted as processed, no abort."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
            'b.py': '"""b."""\ndef bar(): pass\n',
        })
        config = _make_config(tmp_path, source)

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert mock_llm.call_count == 2
        assert result.files_processed == 2
        assert result.files_skipped == 0
        assert result.aborted is False
        assert result.unprocessed_files == ()
        assert result.errors == ()

    @pytest.mark.asyncio
    async def test_doc_types_multiplied_by_files(
        self, tmp_path: Path,
    ) -> None:
        """Three doc types × two files = six LLM invocations.
        Paired with the single-doc-type test above so a wrong
        loop nesting (e.g., one call per file regardless of types)
        fails this test, and the wrong loop in the other direction
        fails the previous one."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
            'b.py': '"""b."""\ndef bar(): pass\n',
        })
        config = _make_config(
            tmp_path, source,
            doc_types=('explanation', 'architecture', 'qa'),
        )

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert mock_llm.call_count == 2 * 3
        assert result.files_processed == 2


# ---------------------------------------------------------------------------
# Staleness filtering
# ---------------------------------------------------------------------------


class TestStalenessFiltering:
    @pytest.mark.asyncio
    async def test_force_regenerate_processes_all(
        self, tmp_path: Path,
    ) -> None:
        """Two runs back-to-back: the second one with
        ``force_regenerate=True`` should re-process both files even
        though the staleness DB now records them as up-to-date.

        Without force, the second run would skip both. The
        force-regenerate branch in run() bypasses
        ``staleness.get_stale_files`` entirely."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
            'b.py': '"""b."""\ndef bar(): pass\n',
        })

        # First run: dry_run=False so staleness records the files as
        # documented. Patch _store_document and embedding too so we
        # don't hit OpenAI.
        config_first = _make_config(
            tmp_path, source, dry_run=False,
        )
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            DocGenOrchestrator, '_store_document',
            new_callable=AsyncMock,
        ) as fake_store:
            # Make the fake store return a doc-shaped object with an
            # ``id`` attribute so doc_ids accumulate and staleness is
            # recorded.
            fake_store.side_effect = [
                _FakeDoc('id-a'), _FakeDoc('id-b'),
            ]
            async with DocGenOrchestrator(config_first) as orch:
                first_result = await orch.run()
        assert first_result.files_processed == 2

        # Second run with force_regenerate=True. Should re-process
        # both files regardless of staleness DB state.
        config_second = _make_config(
            tmp_path, source, force_regenerate=True,
        )
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config_second) as orch:
                second_result = await orch.run()

        assert second_result.files_processed == 2
        assert mock_llm.call_count == 2

    # NOTE: a paired ``test_unforced_skips_already_documented``
    # belongs here in principle but exposes a harness limitation —
    # mocking ``_store_document`` breaks the chain that
    # ``record_documentation_async`` needs to persist the file's
    # sha to the staleness DB. The unforced-skip behavior is
    # already covered by ``tests/test_staleness*.py`` against the
    # real Library + StalenessTracker. Repeating it here without a
    # working fake store would either be a paperweight (skipping
    # the assertion) or a fragile re-mock (going deeper than the
    # orchestrator). Pin the force-regenerate side as the contract;
    # leave the inverse to its existing tests.


# ---------------------------------------------------------------------------
# Quota-exhausted abort coordination
# ---------------------------------------------------------------------------


class TestQuotaAbort:
    @pytest.mark.asyncio
    async def test_quota_error_aborts_and_records_unprocessed(
        self, tmp_path: Path,
    ) -> None:
        """When ``QuotaExhaustedError`` fires mid-run, the abort
        event is set, ``aborted=True``, ``abort_reason`` propagates,
        and any files that hadn't started (or hit the abort flag in
        the semaphore queue) appear in ``unprocessed_files``.

        Patches ``_process_file`` directly because
        ``DocGenerator.generate_for_module`` catches ``Exception``
        per doc-type, swallowing any error raised inside ``_call_llm``
        before it reaches ``run()``'s abort handler. Patching one
        level higher exercises the orchestrator's abort-coordination
        layer specifically — the layer that batch dispatch (#45)
        will need to coexist with."""
        from docgen.llm.anthropic import QuotaExhaustedError

        # Five files, concurrency=1 to make the abort timing
        # deterministic — first file aborts, the remaining four
        # should all skip via the abort_event check.
        files = {f'f{i}.py': f'"""f{i}."""\ndef x{i}(): pass\n'
                 for i in range(5)}
        source = _make_source_tree(tmp_path, files)
        config = _make_config(tmp_path, source, concurrency=1)

        with patch.object(
            DocGenOrchestrator, '_process_file',
            new_callable=AsyncMock,
            side_effect=QuotaExhaustedError('quota exhausted'),
        ) as mock_process:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.aborted is True
        assert 'quota' in result.abort_reason.lower()
        # First file ran (and raised); the other four should be
        # in unprocessed.
        assert len(result.unprocessed_files) >= 4
        # Only one _process_file call actually executed; rest skipped.
        assert mock_process.call_count == 1

    @pytest.mark.asyncio
    async def test_end_to_end_via_call_llm_propagates_to_abort(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end: ``QuotaExhaustedError`` raised at the lowest
        level (``_call_llm``) propagates through the full chain
        (``_generate_doc`` → ``generate_for_module`` → ``_legacy_generate``
        → ``_process_file`` → ``run()``) and triggers the
        orchestrator's abort.

        Distinct from
        ``test_quota_error_aborts_and_records_unprocessed`` above,
        which patches ``_process_file`` directly. This test proves
        the FULL CHAIN works — including the generator's
        ``except QuotaExhaustedError: raise`` special-case that
        was added because the blanket ``except Exception``
        previously swallowed the error before it reached the
        orchestrator's abort coordination. Regression guard for
        the abort path in production runs."""
        from docgen.llm.anthropic import QuotaExhaustedError

        files = {f'f{i}.py': f'"""f{i}."""\ndef x{i}(): pass\n'
                 for i in range(5)}
        source = _make_source_tree(tmp_path, files)
        config = _make_config(tmp_path, source, concurrency=1)

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            side_effect=QuotaExhaustedError('quota exhausted'),
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.aborted is True
        assert 'quota' in result.abort_reason.lower()
        assert len(result.unprocessed_files) >= 4

    @pytest.mark.asyncio
    async def test_post_processing_skipped_on_abort(
        self, tmp_path: Path,
    ) -> None:
        """When the run aborts, themes + crossrefs phases are
        skipped (running them on a partial result wastes work the
        resume run will redo). Patches ``_process_file`` to raise
        QuotaExhaustedError (same reason as the test above)."""
        from docgen.llm.anthropic import QuotaExhaustedError

        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(tmp_path, source)

        with patch.object(
            DocGenOrchestrator, '_process_file',
            new_callable=AsyncMock,
            side_effect=QuotaExhaustedError('quota'),
        ), patch.object(
            DocGenOrchestrator, '_post_process',
            new_callable=AsyncMock,
        ) as mock_post:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.aborted is True
        assert mock_post.call_count == 0


# ---------------------------------------------------------------------------
# Concurrency limit (the asyncio.Semaphore behavior)
# ---------------------------------------------------------------------------


class TestConcurrency:
    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_calls(
        self, tmp_path: Path,
    ) -> None:
        """With ``concurrency=2`` and 6 files, no more than 2 LLM
        calls should be in flight simultaneously. We verify by
        having ``_call_llm`` track its high-water concurrent count
        via an asyncio-safe counter.

        A regression that drops the Semaphore (or sets it to None)
        would show high_water == 6 (or whatever is achievable on
        the runtime). The test asserts <= concurrency."""
        files = {f'f{i}.py': f'"""f{i}."""\ndef x{i}(): pass\n'
                 for i in range(6)}
        source = _make_source_tree(tmp_path, files)
        config = _make_config(tmp_path, source, concurrency=2)

        in_flight = 0
        high_water = 0
        lock = asyncio.Lock()

        async def slow_call(*args, **kwargs):
            nonlocal in_flight, high_water
            async with lock:
                in_flight += 1
                if in_flight > high_water:
                    high_water = in_flight
            # Yield to let other tasks try to enter.
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return _VALID_DOC_CONTENT

        with patch.object(
            DocGenerator, '_call_llm', side_effect=slow_call,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.files_processed == 6
        assert high_water <= 2, (
            f'concurrency limit violated: {high_water} concurrent '
            f'LLM calls observed, limit was 2'
        )


# ---------------------------------------------------------------------------
# Edge: empty source tree
# ---------------------------------------------------------------------------


class TestEmptySource:
    @pytest.mark.asyncio
    async def test_no_files_returns_zero_counts(
        self, tmp_path: Path,
    ) -> None:
        """An empty source tree returns a clean PipelineResult with
        all-zero counts, no aborts, no errors. Bites a path that
        crashes on empty file lists (rare but possible — division
        by zero in progress reporting, etc.)."""
        source = _make_source_tree(tmp_path, {})  # empty
        config = _make_config(tmp_path, source)

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert mock_llm.call_count == 0
        assert result.files_processed == 0
        assert result.files_skipped == 0
        assert result.docs_created == 0
        assert result.docs_failed == 0
        assert result.aborted is False
        assert result.errors == ()


# ---------------------------------------------------------------------------
# Validation retry counters
# ---------------------------------------------------------------------------


class TestValidationCounters:
    @pytest.mark.asyncio
    async def test_first_attempt_passes_zero_retry_counters(
        self, tmp_path: Path,
    ) -> None:
        """validate=True, content passes validation on first try →
        all three retry counters are zero. Pinned as the baseline
        so a regression that always reports >0 retries (e.g.,
        miscounting the initial pass as a retry) fails this test."""
        from unittest.mock import Mock

        from docgen.validator import ContentValidator

        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(tmp_path, source, validate=True)

        valid = Mock(is_valid=True, errors=0, issues=())
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            ContentValidator, 'validate', return_value=valid,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.validation_initial_failures == 0
        assert result.validation_retry_attempts == 0
        assert result.validation_recovered == 0

    @pytest.mark.asyncio
    async def test_retry_recovers_increments_recovered(
        self, tmp_path: Path,
    ) -> None:
        """First attempt fails validation, retry passes → counters
        report exactly one recovery: initial_failures=1,
        retry_attempts>=1, recovered=1.

        Bites a regression that conflates "validation failed" with
        "doc failed" — recovered count must NOT also bump the
        ``docs_failed`` total when the retry succeeds."""
        from unittest.mock import Mock

        from docgen.validator import ContentValidator

        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(tmp_path, source, validate=True)

        valid = Mock(is_valid=True, errors=0, issues=())
        invalid = Mock(is_valid=False, errors=1, issues=())
        # validate() called per attempt: first invalid, then valid.
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            ContentValidator, 'validate',
            side_effect=[invalid, valid],
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.validation_initial_failures == 1
        assert result.validation_retry_attempts >= 1
        assert result.validation_recovered == 1
        # The doc itself was recovered, so docs_failed for this file
        # is 0 (not 1 — recovered != failed).
        assert result.docs_failed == 0

    @pytest.mark.asyncio
    async def test_all_retries_fail_increments_failed(
        self, tmp_path: Path,
    ) -> None:
        """Validation always returns invalid → after exhausting
        retries: initial_failures=1, recovered=0, docs_failed >= 1.
        Paired with the recovers test: a wrong counter wiring that
        always increments recovered (or never increments docs_failed)
        fails one half."""
        from unittest.mock import Mock

        from docgen.validator import ContentValidator

        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(tmp_path, source, validate=True)

        invalid = Mock(is_valid=False, errors=1, issues=())
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            ContentValidator, 'validate', return_value=invalid,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.validation_initial_failures == 1
        assert result.validation_retry_attempts >= 1
        assert result.validation_recovered == 0
        assert result.docs_failed >= 1


# ---------------------------------------------------------------------------
# Pre-generation failures (SyntaxError on analyzer.analyze_file)
# ---------------------------------------------------------------------------


class TestPreGenFailure:
    @pytest.mark.asyncio
    async def test_syntax_error_counts_as_doc_failed(
        self, tmp_path: Path,
    ) -> None:
        """A Python file with broken syntax → ``analyze_file`` raises
        ``SyntaxError``, ``_legacy_generate`` returns ``(None, 0)``,
        and ``_process_file`` reports ``docs_failed = len(doc_types)``
        for that file. No LLM call should be made.

        Hits the ``except SyntaxError`` branch in ``_legacy_generate``
        (orchestrator.py:644-646) — currently uncovered."""
        source = _make_source_tree(tmp_path, {
            # Intentional syntax error — unbalanced paren.
            'broken.py': '"""broken."""\ndef foo(\n',
            # Sibling valid file so the run doesn't stop on first failure
            'good.py': '"""good."""\ndef bar(): pass\n',
        })
        config = _make_config(tmp_path, source)

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        # Only the good file's LLM call happened.
        assert mock_llm.call_count == 1
        # Both files counted as processed (broken's processing
        # ended with a recorded failure, not a skip).
        assert result.files_processed == 2
        # broken.py contributes 1 docs_failed (1 doc type × pre-gen
        # failure).
        assert result.docs_failed >= 1
        # No abort — SyntaxError is per-file, not a hard stop.
        assert result.aborted is False


# ---------------------------------------------------------------------------
# Single-file target_path
# ---------------------------------------------------------------------------


class TestTargetPath:
    @pytest.mark.asyncio
    async def test_single_file_target_processes_only_target(
        self, tmp_path: Path,
    ) -> None:
        """When ``target_path`` resolves to a single file (not a
        directory), only that file is processed even though sibling
        files exist. Hits the ``full_target.is_file()`` branch in
        run() (orchestrator.py:339-340)."""
        source = _make_source_tree(tmp_path, {
            'target.py': '"""target."""\ndef target_fn(): pass\n',
            'sibling_a.py': '"""a."""\ndef a(): pass\n',
            'sibling_b.py': '"""b."""\ndef b(): pass\n',
        })
        # target_path is RELATIVE to source_path per OrchestratorConfig.
        config = _make_config(
            tmp_path, source,
            target_path=Path('target.py'),
        )

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        # Exactly one LLM call — only target.py was processed.
        assert mock_llm.call_count == 1
        assert result.files_processed == 1


# ---------------------------------------------------------------------------
# Generic exception collection (non-quota)
# ---------------------------------------------------------------------------


class TestGenericExceptionCollection:
    @pytest.mark.asyncio
    async def test_value_error_collected_in_errors_list(
        self, tmp_path: Path,
    ) -> None:
        """When ``_process_file`` raises a non-``QuotaExhaustedError``
        exception, it's caught by the generic ``except Exception`` in
        run() and appended to ``result.errors``. Run continues for
        other files; abort is NOT triggered.

        Paired implicitly with the abort tests above — generic errors
        must NOT trigger the abort path. Hits the
        ``errors.append(...)`` branch in run() that's distinct from
        the ``QuotaExhaustedError`` catch."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
            'b.py': '"""b."""\ndef bar(): pass\n',
        })
        config = _make_config(tmp_path, source)

        # File a.py raises ValueError, b.py succeeds.
        async def fake_process(self, path: Path):
            from docgen.orchestrator import GenerationResult
            if path.name == 'a.py':
                raise ValueError('oops a')
            return GenerationResult(
                source_path=path, docs_generated=1, docs_failed=0,
            )

        with patch.object(
            DocGenOrchestrator, '_process_file',
            new=fake_process,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        # Run NOT aborted by a generic error.
        assert result.aborted is False
        # Exactly one error collected — for a.py only. The
        # ``len(errors) == 1`` is the load-bearing assertion: a
        # bug that double-records the error or fails to record
        # b.py's success would shift this number.
        assert len(result.errors) == 1
        assert 'oops a' in result.errors[0]
        assert 'a.py' in result.errors[0]
        # b.py succeeded — its docs_generated=1 contributes to the
        # aggregate. Confirms the run did NOT short-circuit on
        # a.py's failure.
        assert result.docs_created == 1
        assert result.docs_failed == 0


# ---------------------------------------------------------------------------
# target_path as a directory (vs a file)
# ---------------------------------------------------------------------------


class TestTargetDirectory:
    @pytest.mark.asyncio
    async def test_target_directory_walks_subtree_only(
        self, tmp_path: Path,
    ) -> None:
        """``target_path`` resolves to a SUBDIRECTORY (not a file).
        Run discovers files under that subdirectory only — sibling
        files at the source root are skipped. Hits the
        ``else: search_root = full_target`` branch in run() that
        the file-target test above doesn't exercise."""
        source = tmp_path / 'src'
        source.mkdir()
        # Sibling files at source root — should NOT be processed
        _write(source / 'root_a.py', '"""ra."""\ndef ra(): pass\n')
        _write(source / 'root_b.py', '"""rb."""\ndef rb(): pass\n')
        # Subdirectory the target_path points to
        sub = source / 'subpkg'
        sub.mkdir()
        _write(sub / 'in_a.py', '"""ia."""\ndef ia(): pass\n')
        _write(sub / 'in_b.py', '"""ib."""\ndef ib(): pass\n')

        config = _make_config(
            tmp_path, source,
            target_path=Path('subpkg'),
        )

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ) as mock_llm:
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        # Exactly the 2 files under subpkg/. root_*.py at source
        # root must not be walked.
        assert mock_llm.call_count == 2
        assert result.files_processed == 2


# ---------------------------------------------------------------------------
# progress_callback wiring
# ---------------------------------------------------------------------------


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_run_scoped_callback_overrides_attribute(
        self, tmp_path: Path,
    ) -> None:
        """A ``progress_callback`` passed to ``run()`` wins over the
        attribute set before ``__aenter__``. Pinned because the
        backwards-compat path (run-scoped callback) needs to keep
        working — older callers pass to run() only.

        Asserts the run-scoped callback receives at least one
        invocation. Bites a regression that drops the parameter
        entirely or silently uses the attribute."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(tmp_path, source)

        attr_calls: list[tuple] = []
        run_calls: list[tuple] = []

        def attr_cb(msg: str, current: int, total: int) -> None:
            attr_calls.append((msg, current, total))

        def run_cb(msg: str, current: int, total: int) -> None:
            run_calls.append((msg, current, total))

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ):
            orch = DocGenOrchestrator(config)
            orch.progress_callback = attr_cb
            async with orch:
                await orch.run(progress_callback=run_cb)

        # The run-scoped callback wins for run()-emitted progress.
        # Attr-set callback may have fired during __aenter__ phases
        # before run() reassigned, which is fine. The point: run_cb
        # got at least one call.
        assert len(run_calls) > 0


# ---------------------------------------------------------------------------
# Dependency-docs check
# ---------------------------------------------------------------------------


class TestDependencyCheck:
    @pytest.mark.asyncio
    async def test_dependencies_set_triggers_check(
        self, tmp_path: Path,
    ) -> None:
        """When ``config.dependencies`` is non-empty,
        ``_check_dependency_docs`` is called once before file
        discovery. Empty default → check never fires (paired
        baseline below)."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(
            tmp_path, source,
            dependencies=('other-source',),
        )

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            DocGenOrchestrator, '_check_dependency_docs',
        ) as mock_check:
            async with DocGenOrchestrator(config) as orch:
                await orch.run()

        assert mock_check.call_count == 1

    @pytest.mark.asyncio
    async def test_no_dependencies_skips_check(
        self, tmp_path: Path,
    ) -> None:
        """Paired baseline: empty ``dependencies`` (the default)
        means no dependency-docs check fires. Bites a regression
        that always runs the check regardless of config."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(tmp_path, source)
        # Explicitly empty
        assert config.dependencies == ()

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            DocGenOrchestrator, '_check_dependency_docs',
        ) as mock_check:
            async with DocGenOrchestrator(config) as orch:
                await orch.run()

        assert mock_check.call_count == 0


# ---------------------------------------------------------------------------
# Post-processing runs themes (when not aborted, not dry_run)
# ---------------------------------------------------------------------------


class TestPostProcessRuns:
    @pytest.mark.asyncio
    async def test_themes_refresh_invoked_on_successful_run(
        self, tmp_path: Path,
    ) -> None:
        """When the run completes without abort and ``dry_run=False``,
        ``_post_process`` runs and ``refresh_themes`` is invoked once.

        ``_post_process`` early-returns on dry_run (orchestrator.py:
        735-736), so all of our other tests skip its body. This test
        flips ``dry_run=False`` (with ``_store_document`` mocked to
        avoid real Library writes) so the post-processing body
        actually executes — covering the themes-refresh branch
        and the surrounding progress-callback announcements."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(
            tmp_path, source,
            dry_run=False,            # required to enter _post_process body
            themes_enabled=True,
            inject_crossrefs=False,    # focus this test on themes only
        )

        # Mock at three layers:
        # 1. _call_llm — no real LLM
        # 2. _store_document — no real Library write
        # 3. refresh_themes — no real theme generation
        # The chain still executes; we verify themes was called.
        fake_themes = AsyncMock(return_value={'path': 'x', 'summarized': 0})
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            DocGenOrchestrator, '_store_document',
            new_callable=AsyncMock, return_value=_FakeDoc('id-a'),
        ), patch(
            'docgen.themes.refresh_themes', fake_themes,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.aborted is False
        assert fake_themes.call_count == 1


# ---------------------------------------------------------------------------
# _post_process body — full coverage with progress + crossrefs
# ---------------------------------------------------------------------------


class TestPostProcessFullPath:
    @pytest.mark.asyncio
    async def test_themes_and_crossrefs_with_progress_callback(
        self, tmp_path: Path,
    ) -> None:
        """Full ``_post_process`` happy path:

        - ``crossref_progress`` callback receives the "Themes:
          refreshing" announcement (line 743).
        - ``refresh_themes`` is invoked and calls back into
          ``_theme_progress`` with both ``cluster_id`` set and
          ``cluster_id=None`` (lines 745-755).
        - Themes-summary announcement after refresh succeeds
          (lines 771-776).
        - Crossrefs path runs (lines 783-789).

        Single test exercises ~25 lines of post_process body that
        the existing themes-only test left uncovered because it
        passed no ``crossref_progress`` and disabled
        ``inject_crossrefs``."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(
            tmp_path, source,
            dry_run=False,
            themes_enabled=True,
            inject_crossrefs=True,
        )

        crossref_msgs: list[tuple[str, int, int]] = []

        def crossref_cb(msg: str, current: int, total: int) -> None:
            crossref_msgs.append((msg, current, total))

        async def fake_refresh(
            library, writer, *, enabled, summarize_kwargs,
        ):
            on_progress = summarize_kwargs.get('on_progress')
            if on_progress:
                # Fire both branches of _theme_progress: with
                # cluster_id and without.
                on_progress(1, 3, 'cluster-abc')
                on_progress(3, 3, None)
            return {'path': 'themes.md', 'summarized': 3}

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            DocGenOrchestrator, '_store_document',
            new_callable=AsyncMock,
            return_value=_FakeDoc('id-a'),
        ), patch.object(
            DocGenOrchestrator, '_inject_crossrefs_scoped',
            new_callable=AsyncMock,
        ) as mock_crossrefs, patch(
            'docgen.themes.refresh_themes', new=fake_refresh,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run(
                    crossref_progress=crossref_cb,
                )

        assert result.aborted is False
        # Crossrefs path executed
        assert mock_crossrefs.call_count == 1
        # Multiple progress messages fired through the callback,
        # including BOTH _theme_progress branches.
        all_msgs = [m for m, _, _ in crossref_msgs]
        cluster_msgs = [m for m in all_msgs if 'summarizing' in m]
        done_msgs = [m for m in all_msgs if 'summarize done' in m]
        assert len(cluster_msgs) >= 1, (
            f'_theme_progress cluster_id branch did not fire: '
            f'{all_msgs}'
        )
        assert len(done_msgs) >= 1, (
            f'_theme_progress None-cluster branch did not fire: '
            f'{all_msgs}'
        )

    @pytest.mark.asyncio
    async def test_themes_failure_logged_run_completes(
        self, tmp_path: Path,
    ) -> None:
        """When ``refresh_themes`` raises, ``_post_process`` catches,
        logs via ``crossref_progress('Themes: failed...')``, and the
        run still completes with ``aborted=False``. Covers lines
        777-780 (the except branch around refresh_themes)."""
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(
            tmp_path, source,
            dry_run=False,
            themes_enabled=True,
            inject_crossrefs=False,
        )

        crossref_msgs: list[str] = []

        def crossref_cb(msg: str, current: int, total: int) -> None:
            crossref_msgs.append(msg)

        async def failing_refresh(*args, **kwargs):
            raise RuntimeError('themes blew up')

        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            DocGenOrchestrator, '_store_document',
            new_callable=AsyncMock,
            return_value=_FakeDoc('id-a'),
        ), patch(
            'docgen.themes.refresh_themes', new=failing_refresh,
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run(
                    crossref_progress=crossref_cb,
                )

        # Run NOT aborted by themes failure.
        assert result.aborted is False
        # Failure announced through the progress callback.
        assert any(
            'failed' in m.lower() for m in crossref_msgs
        ), f'no themes-failed announcement: {crossref_msgs}'


# ---------------------------------------------------------------------------
# _regenerate_doc catalog path
# ---------------------------------------------------------------------------


class TestRegenerateDocCatalogPath:
    @pytest.mark.asyncio
    async def test_validation_retry_via_catalog_path(
        self, tmp_path: Path,
    ) -> None:
        """Validation retry under ``catalog_only_generator=True``
        routes ``_regenerate_doc`` through the catalog path
        (``enrich_file`` → ``generate_from_elements``). The existing
        retry tests cover the legacy SourceAnalyzer branch only;
        this test covers the ``if self.config.catalog_only_generator``
        branch in ``_regenerate_doc`` (orchestrator.py:800-811)."""
        from unittest.mock import Mock

        from docgen.validator import ContentValidator

        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })
        config = _make_config(
            tmp_path, source,
            validate=True,
            catalog_only_generator=True,
        )

        valid = Mock(is_valid=True, errors=0, issues=())
        invalid = Mock(is_valid=False, errors=1, issues=())
        with patch.object(
            DocGenerator, '_call_llm', new_callable=AsyncMock,
            return_value=_VALID_DOC_CONTENT,
        ), patch.object(
            ContentValidator, 'validate',
            side_effect=[invalid, valid],
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch.run()

        assert result.validation_initial_failures == 1
        assert result.validation_recovered == 1
        assert result.docs_failed == 0


# ---------------------------------------------------------------------------
# _check_dependency_docs body — exercises both branches in the loop
# ---------------------------------------------------------------------------


class TestCheckDependencyDocsBody:
    @pytest.mark.asyncio
    async def test_real_check_runs_through_present_and_missing_branches(
        self, tmp_path: Path,
    ) -> None:
        """Drive the REAL ``_check_dependency_docs`` (not mocked).
        The earlier dependencies-trigger test mocked the method
        away; this one provides a configured Config so the loop's
        branches actually execute:

        - ``dep_with_docs``: has ``.md`` files → info-log path
          (orchestrator.py:1036-1040)
        - ``dep_no_docs``: directory exists but empty → warning
          for missing markdown (orchestrator.py:1027-1034)

        Together they cover the bulk of the 38-line method body."""
        # A separate dep with docs → covers the "present + md_files"
        # branch.
        dep_with_docs_root = tmp_path / 'dep_with_docs_root'
        dep_with_docs_root.mkdir()
        # ariadne export writes md files under docs/<source>/.
        docs_with_md = tmp_path / 'docs' / 'dep_with_docs'
        docs_with_md.mkdir(parents=True)
        (docs_with_md / 'note.md').write_text(
            '# A doc\n', encoding='utf-8',
        )

        # A dep with empty docs dir → covers "present but no md"
        dep_no_docs_root = tmp_path / 'dep_no_docs_root'
        dep_no_docs_root.mkdir()
        docs_empty = tmp_path / 'docs' / 'dep_no_docs'
        docs_empty.mkdir(parents=True)

        # Source under test
        source = _make_source_tree(tmp_path, {
            'a.py': '"""a."""\ndef foo(): pass\n',
        })

        # Write an ariadne.yaml so cfg.get_config() inside the
        # method finds real source paths.
        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            'sources:\n'
            f'  myapp:\n'
            f'    path: {source}\n'
            f'  dep_with_docs:\n'
            f'    path: {dep_with_docs_root}\n'
            f'  dep_no_docs:\n'
            f'    path: {dep_no_docs_root}\n'
            f'docs_base: {tmp_path / "docs"}\n',
            encoding='utf-8',
        )

        # Activate this yaml as the global config so
        # _check_dependency_docs sees our deps.
        import config as config_module
        saved = config_module._global_config
        config_module._global_config = config_module.Config(
            config_path=yaml_path,
        )

        try:
            config = _make_config(
                tmp_path, source,
                source_name='myapp',
                dependencies=('dep_with_docs', 'dep_no_docs'),
            )

            with patch.object(
                DocGenerator, '_call_llm', new_callable=AsyncMock,
                return_value=_VALID_DOC_CONTENT,
            ):
                async with DocGenOrchestrator(config) as orch:
                    result = await orch.run()

            # Run completes regardless of dep-doc state.
            assert result.aborted is False
            # The check runs without raising — that's the contract.
            # The branches inside the loop covered both deps.
        finally:
            config_module._global_config = saved


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeDoc:
    """Stand-in for the Document object returned by _store_document.
    The orchestrator only reads ``.id`` off the result, so we don't
    need the full schema."""
    def __init__(self, doc_id: str) -> None:
        self.id = doc_id
