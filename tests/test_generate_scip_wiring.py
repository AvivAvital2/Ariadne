"""Tests for ``generate`` reaching the SCIP path for Scala/Java.

Pins the wiring chain that lets ``ariadne generate --source scalaproject``
produce LLM commentary on Scala/Java elements via SCIP-extracted
structural data (SCIP follow-up #1, #2, #3):

  cmd_generate
   └→ OrchestratorConfig(source_config=...)
       └→ DocGenOrchestrator._catalog_generate(...)
           └→ enrich_file(path, source_root, source_config=...)
               └→ extract_elements(..., source_config=source_config)

Tests must FAIL until each link in the chain is in place.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from docgen.scip_config import SourceScipConfig

# ---------------------------------------------------------------------------
# Link 1: enrich_file accepts source_config and forwards it
# ---------------------------------------------------------------------------


class TestEnrichFileForwardsSourceConfig:
    def test_source_config_kwarg_accepted(self, tmp_path: Path) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'm.py'
        f.write_text('def foo(): pass\n', encoding='utf-8')

        cfg = SourceScipConfig(
            repo='r', artifact_path=tmp_path / 'idx.scip',
        )
        # Just calling with source_config must not raise.
        bundle = enrich_file(f, source_root=tmp_path, source_config=cfg)
        assert bundle is not None

    def test_source_config_forwarded_to_extract_elements(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen.catalog_enrich import enrich_file

        f = tmp_path / 'X.scala'
        f.write_text('class X\n', encoding='utf-8')

        captured = {}

        def spy(path, source_root, *, source_config=None):
            captured['source_config'] = source_config
            return []

        monkeypatch.setattr(
            'docgen.catalog_enrich.extract_elements', spy,
        )

        cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'idx.scip',
            index_kinds={'scala': 'scip'},
        )
        enrich_file(f, source_root=tmp_path, source_config=cfg)

        assert captured['source_config'] is cfg


# ---------------------------------------------------------------------------
# Link 2: OrchestratorConfig.source_config + _catalog_generate uses it
# ---------------------------------------------------------------------------


class TestOrchestratorConfigSourceConfig:
    def test_default_is_none(self, tmp_path: Path) -> None:
        from docgen.orchestrator import OrchestratorConfig

        cfg = OrchestratorConfig(source_path=tmp_path)
        assert cfg.source_config is None

    def test_can_carry_source_config(self, tmp_path: Path) -> None:
        from docgen.orchestrator import OrchestratorConfig

        scip_cfg = SourceScipConfig(
            repo='scalaproject', artifact_path=tmp_path / 'idx.scip',
        )
        cfg = OrchestratorConfig(source_path=tmp_path, source_config=scip_cfg)
        assert cfg.source_config is scip_cfg


class TestCatalogGenerateForwardsSourceConfig:
    @pytest.mark.asyncio
    async def test_catalog_generate_passes_source_config_to_enrich_file(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When ``catalog_only_generator=True`` and the orchestrator has a
        SCIP source_config, _catalog_generate must call enrich_file with
        that source_config — otherwise SCIP-backed extraction never runs
        through the generate pipeline.
        """
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        f = tmp_path / 'X.scala'
        f.write_text('class X\n', encoding='utf-8')

        captured = {}

        def spy_enrich(path, *, source_root, source_config=None, **kwargs):
            captured['source_config'] = source_config

        monkeypatch.setattr(
            'docgen.catalog_enrich.enrich_file', spy_enrich,
        )

        scip_cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'idx.scip',
            index_kinds={'scala': 'scip'},
        )
        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            dry_run=True,
            catalog_only_generator=True,
            source_config=scip_cfg,
        )

        async with DocGenOrchestrator(config) as orch:
            await orch._process_file(f)

        assert captured['source_config'] is scip_cfg


# ---------------------------------------------------------------------------
# Link 3: cmd_generate resolves and passes source_config
# ---------------------------------------------------------------------------


class TestCmdGenerateResolvesSourceConfig:
    @pytest.mark.asyncio
    async def test_cmd_generate_attaches_source_config_when_configured(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The CLI must read the source's SCIP block from ``ariadne.yaml``
        and attach it to OrchestratorConfig — otherwise even with
        --catalog-only-generator the orchestrator can't reach SCIP.
        """
        import argparse

        from cli.generation import cmd_generate

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

            async def run(self, progress_callback=None, **_kw):
                from docgen.orchestrator import PipelineResult
                return PipelineResult(
                    files_processed=0, files_skipped=0,
                    docs_created=0, docs_failed=0,
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
        monkeypatch.setattr('cli.generate.get_config', lambda: cfg)

        # dry_run=False: the test asserts on OrchestratorConfig, which is
        # constructed downstream of the dry-run short-circuit. The
        # FakeOrchestrator above absorbs the call.
        args = argparse.Namespace(
            source='scalaproject',
            model=None,
            provider=None,
            api_key=None,
            types='explanation',
            concurrency=1,
            force=False,
            dry_run=False,
            verbose=False,
            path=None,
            db=None,
            no_crossrefs=False,
        )
        await cmd_generate(args)

        oc = captured_config.get('config')
        assert oc is not None, 'DocGenOrchestrator was never instantiated'
        assert oc.source_config is not None, (
            'cmd_generate failed to attach SCIP source_config — '
            'generate cannot reach the SCIP path'
        )
        assert oc.source_config.repo == 'scalaproject'

    @pytest.mark.asyncio
    async def test_cmd_generate_source_config_none_when_not_configured(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """For sources without a ``scip:`` block, source_config remains None
        (the legacy / Python-only behavior). No regression.
        """
        import argparse

        from cli.generation import cmd_generate

        source_dir = tmp_path / 'pythonproject'
        source_dir.mkdir()
        (source_dir / 'm.py').write_text('def foo(): pass\n', encoding='utf-8')

        captured_config = {}

        class _FakeOrchestrator:
            def __init__(self, config) -> None:
                captured_config['config'] = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_) -> None:
                pass

            async def run(self, progress_callback=None, **_kw):
                from docgen.orchestrator import PipelineResult
                return PipelineResult(
                    files_processed=0, files_skipped=0,
                    docs_created=0, docs_failed=0,
                )

        monkeypatch.setattr(
            'docgen.orchestrator.DocGenOrchestrator', _FakeOrchestrator,
        )

        from config import Config
        cfg = Config()
        cfg._config['sources'] = {'pythonproject': {'path': str(source_dir)}}
        cfg._config['default_source'] = 'pythonproject'
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)
        monkeypatch.setattr('cli.generate.get_config', lambda: cfg)

        args = argparse.Namespace(
            source='pythonproject',
            model=None,
            provider=None,
            api_key=None,
            types='explanation',
            concurrency=1,
            force=False,
            dry_run=False,
            verbose=False,
            path=None,
            db=None,
            no_crossrefs=False,
        )
        await cmd_generate(args)

        oc = captured_config.get('config')
        assert oc is not None
        assert oc.source_config is None
