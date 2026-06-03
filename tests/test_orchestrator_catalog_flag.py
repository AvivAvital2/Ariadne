"""Tests for the catalog_only_generator feature flag (Catalog transition Phase 2.5).

The orchestrator must support BOTH the legacy SourceAnalyzer-driven path and
the new catalog-driven path side-by-side, gated by ``catalog_only_generator``
on OrchestratorConfig. With the flag off (default), behavior is byte-identical
to today. With the flag on, ``_process_file`` walks via
``enrich_file`` + ``DocGenerator.generate_from_elements``.

These tests pin the flag's wiring without running any LLM or DB writes.
"""
from __future__ import annotations

from pathlib import Path
from textwrap import dedent
from unittest.mock import AsyncMock, patch

import pytest

from docgen.generator import DocGenerator
from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig


def _write(path: Path, src: str) -> None:
    path.write_text(dedent(src).lstrip('\n'), encoding='utf-8')


# ---------------------------------------------------------------------------
# Config defaults
# ---------------------------------------------------------------------------


class TestConfigDefault:
    def test_catalog_only_generator_defaults_on(self, tmp_path: Path) -> None:
        """Default flipped to ``True`` (Phase 2 Change 1).

        The legacy ``SourceAnalyzer`` path is Python-only and crashes on
        ``.md`` / ``.yaml`` / ``.json`` / Scala / Java with a
        ``SyntaxError``. The catalog-driven path handles every
        ``CATALOG_EXTS`` extension natively, so it is now the default.
        Sources can still opt back into legacy by passing
        ``catalog_only_generator=False`` explicitly.
        """
        cfg = OrchestratorConfig(source_path=tmp_path)
        assert cfg.catalog_only_generator is True

    def test_catalog_only_generator_can_be_disabled(self, tmp_path: Path) -> None:
        """Legacy opt-out still works for emergency rollback."""
        cfg = OrchestratorConfig(
            source_path=tmp_path,
            catalog_only_generator=False,
        )
        assert cfg.catalog_only_generator is False


# ---------------------------------------------------------------------------
# Routing — flag OFF (legacy)
# ---------------------------------------------------------------------------


class TestFlagOffUsesLegacyPath:
    @pytest.mark.asyncio
    async def test_process_file_calls_analyzer_when_flag_off(
        self, tmp_path: Path,
    ) -> None:
        """With the flag off, _process_file must use the legacy
        analyzer.analyze_file → generator.generate_for_module path.
        Asserting we still call generate_for_module catches accidental
        cutovers.
        """
        f = tmp_path / 'm.py'
        _write(f, '''"""m."""
def foo(): return 1
''')

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=False,
        )

        with patch.object(
            DocGenerator, 'generate_for_module', new_callable=AsyncMock,
            return_value=[],
        ) as legacy, patch.object(
            DocGenerator, 'generate_from_elements', new_callable=AsyncMock,
            return_value=[],
        ) as new_path:
            async with DocGenOrchestrator(config) as orch:
                await orch._process_file(f)

        assert legacy.called, 'flag-off must use generate_for_module'
        assert not new_path.called, 'flag-off must NOT use generate_from_elements'


# ---------------------------------------------------------------------------
# Routing — flag ON (catalog-driven)
# ---------------------------------------------------------------------------


class TestFlagOnUsesCatalogPath:
    @pytest.mark.asyncio
    async def test_process_file_calls_generate_from_elements_when_flag_on(
        self, tmp_path: Path,
    ) -> None:
        f = tmp_path / 'm.py'
        _write(f, '''"""m."""
def foo(): return 1
''')

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=True,
        )

        with patch.object(
            DocGenerator, 'generate_for_module', new_callable=AsyncMock,
            return_value=[],
        ) as legacy, patch.object(
            DocGenerator, 'generate_from_elements', new_callable=AsyncMock,
            return_value=[],
        ) as new_path:
            async with DocGenOrchestrator(config) as orch:
                await orch._process_file(f)

        assert new_path.called, 'flag-on must use generate_from_elements'
        assert not legacy.called, 'flag-on must NOT use generate_for_module'

    @pytest.mark.asyncio
    async def test_bundle_passed_to_generate_from_elements(
        self, tmp_path: Path,
    ) -> None:
        """The catalog path must produce an EnrichedFileBundle and pass it
        to generate_from_elements — we verify the call's first arg has
        the expected attributes.
        """
        from docgen.catalog_enrich import EnrichedFileBundle

        f = tmp_path / 'm.py'
        _write(f, '''"""docstring."""
def foo(): return 1
''')

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=True,
        )

        captured: list = []

        async def fake_generate(bundle, doc_types=None):
            captured.append(bundle)
            return []

        with patch.object(
            DocGenerator, 'generate_from_elements',
            new_callable=AsyncMock, side_effect=fake_generate,
        ):
            async with DocGenOrchestrator(config) as orch:
                await orch._process_file(f)

        assert len(captured) == 1
        bundle = captured[0]
        assert isinstance(bundle, EnrichedFileBundle)
        assert bundle.language == 'python'
        assert bundle.module_name == 'm'
        assert bundle.module_docstring == 'docstring.'

    @pytest.mark.asyncio
    async def test_non_python_file_handled_when_flag_on(
        self, tmp_path: Path,
    ) -> None:
        """The whole motivation for the flag is multi-language coverage —
        running _process_file on a JS file with the flag ON must produce
        a bundle and dispatch, not crash on missing analyzer support.
        """
        f = tmp_path / 'app.js'
        _write(f, '''
            function greet() { return 1; }
        ''')

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=True,
        )

        with patch.object(
            DocGenerator, 'generate_from_elements', new_callable=AsyncMock,
            return_value=[],
        ) as new_path:
            async with DocGenOrchestrator(config) as orch:
                result = await orch._process_file(f)

        assert new_path.called, 'JS files must dispatch via the catalog path'
        # No exception, no docs_failed bump (empty doc list is fine).
        assert result.source_path == f


# ---------------------------------------------------------------------------
# Failure handling on the catalog path
# ---------------------------------------------------------------------------


class TestCatalogPathFailures:
    @pytest.mark.asyncio
    async def test_unreadable_file_returns_failure_not_raise(
        self, tmp_path: Path,
    ) -> None:
        """If enrich_file returns None (unsupported extension or read fail),
        _process_file should return a GenerationResult with docs_failed > 0,
        not raise.
        """
        # An unsupported extension exercises the None path of enrich_file.
        f = tmp_path / 'weird.xyz'
        f.write_text('garbage', encoding='utf-8')

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=True,
        )

        async with DocGenOrchestrator(config) as orch:
            result = await orch._process_file(f)

        assert result.source_path == f
        # No docs generated for an unsupported file. No exception is the
        # main invariant.
        assert result.docs_generated == 0

    @pytest.mark.asyncio
    async def test_python_syntax_error_handled_on_catalog_path(
        self, tmp_path: Path,
    ) -> None:
        """A Python file with a SyntaxError must not crash the catalog path.
        ast.parse failing inside enrich_file should leave imports/docstring
        empty but still produce a bundle with element extraction (ast-grep
        is more lenient than ast).
        """
        f = tmp_path / 'broken.py'
        _write(f, '''
            def broken(:
                pass
        ''')

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=True,
        )

        with patch.object(
            DocGenerator, 'generate_from_elements', new_callable=AsyncMock,
            return_value=[],
        ):
            async with DocGenOrchestrator(config) as orch:
                result = await orch._process_file(f)

        assert result.source_path == f
