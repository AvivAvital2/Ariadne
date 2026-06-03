"""Tests for in-loop validation retry in the orchestrator.

A validation failure today silently drops the doc — money is sunk on
the LLM call but no row is created. Often the failure is a
non-deterministic LLM artifact (e.g. it emitted "Description" instead
of "Overview"); a re-roll at the same temperature would have passed.

This module verifies:
1. On validation failure, the orchestrator re-generates the doc up to
   MAX_VALIDATION_RETRIES times before giving up.
2. Stats are tracked: initial failures, retries attempted, recoveries.
3. Stats roll up into PipelineResult so the CLI can summarize.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Inner retry helper — small, focused, easy to test in isolation.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_validate_with_retry_passes_immediately_when_valid():
    """No retries when the first validation passes."""
    from docgen.generator import GeneratedDoc
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
    from docgen.validator import ContentValidator, ValidationResult

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    orch = DocGenOrchestrator(config=cfg)
    orch._validator = MagicMock(spec=ContentValidator)
    orch._validator.validate.return_value = ValidationResult(
        is_valid=True, issues=()
    )

    gd = GeneratedDoc(
        title='t', content='c', doc_type='explanation',
        source_files=(), metadata={},
    )

    regen_calls = 0
    async def regen():
        nonlocal regen_calls
        regen_calls += 1
        return gd

    final, result, retries = await orch._validate_with_retry(gd, regen)

    assert final is gd
    assert result.is_valid
    assert retries == 0
    assert regen_calls == 0, 'should not regenerate when first attempt is valid'


@pytest.mark.asyncio
async def test_validate_with_retry_succeeds_after_one_retry():
    """Initial fail, retry produces valid doc, returns it with retries=1."""
    from docgen.generator import GeneratedDoc
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
    from docgen.validator import ValidationIssue, ValidationResult

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    orch = DocGenOrchestrator(config=cfg)

    bad = ValidationResult(
        is_valid=False,
        issues=(ValidationIssue(
            level='error', code='missing_section',
            message="missing 'Overview' section",
        ),),
    )
    good = ValidationResult(is_valid=True, issues=())

    orch._validator = MagicMock()
    orch._validator.validate.side_effect = [bad, good]

    initial = GeneratedDoc(
        title='t', content='bad', doc_type='explanation',
        source_files=(), metadata={},
    )
    fixed = GeneratedDoc(
        title='t', content='good', doc_type='explanation',
        source_files=(), metadata={},
    )

    async def regen():
        return fixed

    final, result, retries = await orch._validate_with_retry(initial, regen)

    assert final is fixed, 'should return the regenerated doc, not the original'
    assert result.is_valid
    assert retries == 1


@pytest.mark.asyncio
async def test_validate_with_retry_gives_up_after_max_retries():
    """If retries also fail, returns (None, last_result, retries_used)."""
    from docgen.generator import GeneratedDoc
    from docgen.orchestrator import (
        MAX_VALIDATION_RETRIES,
        DocGenOrchestrator,
        OrchestratorConfig,
    )
    from docgen.validator import ValidationIssue, ValidationResult

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    orch = DocGenOrchestrator(config=cfg)

    bad = ValidationResult(
        is_valid=False,
        issues=(ValidationIssue(
            level='error', code='missing_section', message='bad',
        ),),
    )

    orch._validator = MagicMock()
    orch._validator.validate.return_value = bad

    gd = GeneratedDoc(
        title='t', content='never valid', doc_type='explanation',
        source_files=(), metadata={},
    )

    regen_calls = 0
    async def regen():
        nonlocal regen_calls
        regen_calls += 1
        return gd

    final, result, retries = await orch._validate_with_retry(gd, regen)

    assert final is None, 'should give up and return None'
    assert not result.is_valid
    assert retries == MAX_VALIDATION_RETRIES
    assert regen_calls == MAX_VALIDATION_RETRIES


@pytest.mark.asyncio
async def test_validate_with_retry_stops_when_regen_returns_none():
    """If regenerator can't produce a doc (LLM failure), stop early."""
    from docgen.generator import GeneratedDoc
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
    from docgen.validator import ValidationIssue, ValidationResult

    cfg = OrchestratorConfig(
        source_path=Path('/tmp'),
        db_path=Path('/tmp/x.db'),
        staleness_db_path=Path('/tmp/s.db'),
    )
    orch = DocGenOrchestrator(config=cfg)

    bad = ValidationResult(
        is_valid=False,
        issues=(ValidationIssue(
            level='error', code='bad', message='bad',
        ),),
    )
    orch._validator = MagicMock()
    orch._validator.validate.return_value = bad

    gd = GeneratedDoc(
        title='t', content='bad', doc_type='explanation',
        source_files=(), metadata={},
    )

    regen_calls = 0
    async def regen():
        nonlocal regen_calls
        regen_calls += 1

    final, result, retries = await orch._validate_with_retry(gd, regen)

    assert final is None
    assert retries == 1
    assert regen_calls == 1, 'should stop after first None'


# ---------------------------------------------------------------------------
# PipelineResult must surface validation stats so the CLI can summarize.
# ---------------------------------------------------------------------------


def test_pipeline_result_has_validation_stats_fields():
    """PipelineResult must expose validation_initial_failures,
    validation_retry_attempts, validation_recovered.
    """
    from docgen.orchestrator import PipelineResult

    r = PipelineResult(
        files_processed=10, files_skipped=0,
        docs_created=20, docs_failed=2,
        validation_initial_failures=5,
        validation_retry_attempts=8,
        validation_recovered=3,
    )
    assert r.validation_initial_failures == 5
    assert r.validation_retry_attempts == 8
    assert r.validation_recovered == 3


def test_generation_result_has_validation_stats_fields():
    """Per-file GenerationResult also tracks the stats so they aggregate."""
    from docgen.orchestrator import GenerationResult

    r = GenerationResult(
        source_path=Path('/tmp/x'),
        docs_generated=2, docs_failed=0,
        validation_initial_failures=1,
        validation_retry_attempts=2,
        validation_recovered=1,
    )
    assert r.validation_initial_failures == 1
    assert r.validation_retry_attempts == 2
    assert r.validation_recovered == 1
