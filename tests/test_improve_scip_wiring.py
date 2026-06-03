"""Tests for ``ariadne improve`` reaching Scala/Java via SCIP (#7).

Two structural fixes in cmd_improve that unlock Scala/Java:
- It calls ``find_python_files`` (Python-only) instead of
  ``find_catalog_files`` — Scala/Java files don't get discovered.
- The OrchestratorConfig it builds doesn't attach ``source_config``,
  so even if a Scala file made it through, the SCIP path can't engage.
- And without ``catalog_only_generator=True``, the legacy SourceAnalyzer
  runs on Scala source and explodes with SyntaxError.

Tests must FAIL until cmd_improve is updated.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class TestImproveDiscoversScalaFiles:
    @pytest.mark.asyncio
    async def test_scala_files_in_source_are_candidates_for_improve(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """``improve`` must use find_catalog_files (multi-language) so
        Scala/Java files surface as 'undocumented' candidates.
        """
        import argparse

        from cli.generation import cmd_improve

        source_dir = tmp_path / 'scalaproject'
        source_dir.mkdir()
        (source_dir / 'X.scala').write_text('class X\n', encoding='utf-8')

        captured = {}

        # Spy on the file-discovery function used by cmd_improve.
        # If cmd_improve still uses find_python_files, this monkeypatch
        # of find_catalog_files won't fire — captured stays empty.
        # Spy mirrors the production signature explicitly so a future
        # signature drift fails loudly instead of being papered over.
        from docgen import staleness

        original = staleness.find_catalog_files

        def spy(path, *, exclude_patterns=None, exclude_dir_names=None):
            captured['called'] = True
            captured['path'] = path
            captured['exclude_dir_names'] = exclude_dir_names
            # Forward only what was supplied — None means "let the real
            # function use its own default", so we don't paper over its
            # contract with our own placeholder defaults.
            fwd: dict = {}
            if exclude_patterns is not None:
                fwd['exclude_patterns'] = exclude_patterns
            if exclude_dir_names is not None:
                fwd['exclude_dir_names'] = exclude_dir_names
            return original(path, **fwd)

        monkeypatch.setattr('docgen.staleness.find_catalog_files', spy)
        monkeypatch.setattr('cli.generation.find_catalog_files', spy, raising=False)

        from config import Config
        cfg = Config()
        cfg._config['sources'] = {
            'scalaproject': {
                'path': str(source_dir),
                'index_kinds': {'scala': 'scip'},
                'scip': {'artifact_path': str(tmp_path / 'idx.scip')},
            }
        }
        cfg._config['default_source'] = 'scalaproject'
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)
        monkeypatch.setattr(
            'cli.generation.get_library', lambda *_a, **_k: MagicMock(
                get_gap_report=lambda days: {'total_misses': 0, 'top_gaps': []},
                list_documents=list,
                usage_by_document=lambda days, limit: [],
                find_low_value_documents=lambda min_serves, days: [],
                find_oversized_documents=lambda max_chars: [],
                get_graph_stats=lambda: {'total_edges': 0},
            ),
        )

        args = argparse.Namespace(
            source='scalaproject', days=30, max_files=10,
            dry_run=True, db=None,
        )
        await cmd_improve(args)

        assert captured.get('called'), (
            "cmd_improve still uses find_python_files; Scala/Java files "
            "won't be discovered as undocumented candidates"
        )
        # cmd_improve must resolve per-source exclusions through Config
        # and forward them to discovery. Without this, exclude_dirs /
        # exempt_dirs in ariadne.yaml would silently not apply to
        # `ariadne improve`, producing surprise candidates.
        expected = cfg.resolve_excluded_dirs('scalaproject')
        assert captured.get('exclude_dir_names') == expected, (
            f"cmd_improve passed exclude_dir_names="
            f"{captured.get('exclude_dir_names')!r}, expected "
            f"cfg.resolve_excluded_dirs('scalaproject')={expected!r}"
        )


class TestImproveAttachesSourceConfig:
    @pytest.mark.asyncio
    async def test_improve_passes_scip_source_config_when_declared(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When the source declares SCIP, cmd_improve must build an
        OrchestratorConfig with source_config set AND
        catalog_only_generator=True (otherwise the legacy SourceAnalyzer
        path runs and SyntaxErrors on Scala).
        """
        import argparse

        from cli.generation import cmd_improve

        source_dir = tmp_path / 'scalaproject'
        source_dir.mkdir()
        (source_dir / 'X.scala').write_text('class X\n', encoding='utf-8')

        captured_config = {}

        class _FakeOrchestrator:
            def __init__(self, config) -> None:
                captured_config['config'] = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_) -> None:
                pass

            async def _process_file(self, path):
                from docgen.orchestrator import GenerationResult
                return GenerationResult(
                    source_path=path, docs_generated=0, docs_failed=0,
                )

        monkeypatch.setattr(
            'docgen.orchestrator.DocGenOrchestrator', _FakeOrchestrator,
        )

        from config import Config
        cfg = Config()
        cfg._config['sources'] = {
            'scalaproject': {
                'path': str(source_dir),
                'index_kinds': {'scala': 'scip'},
                'scip': {'artifact_path': str(tmp_path / 'idx.scip')},
            }
        }
        cfg._config['default_source'] = 'scalaproject'
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)
        monkeypatch.setattr(
            'cli.generation.get_library', lambda *_a, **_k: MagicMock(
                get_gap_report=lambda days: {'total_misses': 0, 'top_gaps': []},
                list_documents=list,
                usage_by_document=lambda days, limit: [],
                find_low_value_documents=lambda min_serves, days: [],
                find_oversized_documents=lambda max_chars: [],
                get_graph_stats=lambda: {'total_edges': 0},
            ),
        )

        args = argparse.Namespace(
            source='scalaproject', days=30, max_files=10,
            dry_run=False, db=None,
        )
        await cmd_improve(args)

        oc = captured_config.get('config')
        assert oc is not None, 'DocGenOrchestrator was not constructed'
        assert oc.source_config is not None, (
            'improve did not attach source_config — Scala/Java will go '
            'through the wrong extraction path'
        )
        assert oc.catalog_only_generator is True, (
            "improve must enable catalog_only_generator for SCIP sources "
            "so the legacy SourceAnalyzer doesn't SyntaxError on Scala"
        )

