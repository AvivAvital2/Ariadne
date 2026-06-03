"""Contract for QuotaExhaustedError propagation through DocGenerator.

The orchestrator's abort-coordination machinery (run():427-441) fires
only when QuotaExhaustedError propagates out of _process_file. But
both generator paths have a blanket ``except Exception`` that
swallows it at the doc-type level — meaning task #51's "graceful
abort on quota error" never actually fired in production runs.
Surfaced 2026-05-09 by orchestrator integration-test audit.

These tests pin the contract that QuotaExhaustedError is special-
cased: it MUST re-raise out of the generator, while other
exceptions (ValueError, network glitches, etc.) stay swallowed so
a single bad doc-type doesn't take down the whole file's generation.

Each test is paired (quota propagates / generic still swallowed) so
a too-broad fix that re-raises everything fails the swallow half,
and a no-op stub that swallows everything fails the propagation
half.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.generator import DocGenerator
from docgen.llm.anthropic import QuotaExhaustedError
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _write_module(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


@pytest.fixture
def python_file(tmp_path: Path) -> Path:
    """A small parseable .py file the analyzer can consume."""
    p = tmp_path / 'm.py'
    _write_module(p, '''"""m."""
def foo():
    return 1
''')
    return p


def _make_config(tmp_path: Path) -> OrchestratorConfig:
    return OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'ariadne.db',
        staleness_db_path=tmp_path / 'staleness.db',
        api_key='test-not-used',
        provider='openai',
        model='gpt-5.2',
        doc_types=('explanation',),
        validate=False,
        dry_run=True,
    )


# ---------------------------------------------------------------------------
# Legacy path: generate_for_module
# ---------------------------------------------------------------------------


class TestGenerateForModulePropagatesQuotaError:
    @pytest.mark.asyncio
    async def test_quota_exhausted_propagates(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """When ``_call_llm`` raises ``QuotaExhaustedError``,
        ``generate_for_module`` must re-raise (not log + continue).
        Without this, the orchestrator's abort coordination at
        run():427 never fires — production users hit quota mid-run
        and burn compute on doomed retries instead of gracefully
        aborting."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            with patch.object(
                DocGenerator, '_call_llm', new_callable=AsyncMock,
                side_effect=QuotaExhaustedError('quota exhausted'),
            ):
                with pytest.raises(QuotaExhaustedError):
                    await gen.generate_for_module(
                        metadata, ('explanation',),
                    )

    @pytest.mark.asyncio
    async def test_generic_exception_still_swallowed(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Paired baseline. A generic exception (``ValueError``) must
        STILL be swallowed and logged — the per-doc-type fallback is
        deliberate: one bad doc type shouldn't take down the whole
        file's generation. Bites a too-broad fix that re-raises every
        exception."""
        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None
            metadata = orch._analyzer.analyze_file(python_file)

            with patch.object(
                DocGenerator, '_call_llm', new_callable=AsyncMock,
                side_effect=ValueError('something else'),
            ):
                # Should NOT raise — generic errors are caught and
                # logged; the function returns whatever docs succeeded
                # (here: none, since the only requested type failed).
                result = await gen.generate_for_module(
                    metadata, ('explanation',),
                )
                assert result == []


# ---------------------------------------------------------------------------
# Catalog path: generate_from_elements
# ---------------------------------------------------------------------------


class TestGenerateFromElementsPropagatesQuotaError:
    @pytest.mark.asyncio
    async def test_quota_exhausted_propagates(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Same contract for the catalog-driven path. Both paths
        must propagate ``QuotaExhaustedError``; the orchestrator's
        abort coordination doesn't care which generator path
        produced the error."""
        from docgen.catalog_enrich import enrich_file

        bundle = enrich_file(
            python_file, source_root=python_file.parent,
        )
        assert bundle is not None  # sanity

        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None

            with patch.object(
                DocGenerator, '_call_llm', new_callable=AsyncMock,
                side_effect=QuotaExhaustedError('quota exhausted'),
            ):
                with pytest.raises(QuotaExhaustedError):
                    await gen.generate_from_elements(
                        bundle, doc_types=('explanation',),
                    )

    @pytest.mark.asyncio
    async def test_generic_exception_still_swallowed(
        self, python_file: Path, tmp_path: Path,
    ) -> None:
        """Paired baseline for the catalog path."""
        from docgen.catalog_enrich import enrich_file

        bundle = enrich_file(
            python_file, source_root=python_file.parent,
        )
        assert bundle is not None

        config = _make_config(tmp_path)
        async with DocGenOrchestrator(config) as orch:
            gen = orch._generator
            assert gen is not None

            with patch.object(
                DocGenerator, '_call_llm', new_callable=AsyncMock,
                side_effect=ValueError('something else'),
            ):
                result = await gen.generate_from_elements(
                    bundle, doc_types=('explanation',),
                )
                assert result == []
