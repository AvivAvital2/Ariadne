"""End-to-end SCIP wiring across every command path.

Pins the contracts that make Scala/Java sources actually work through
``generate``, ``sync``, ``merge``, and ``check`` — not just
``catalog-sync``.

Two structural changes:
- Orchestrator file discovery (run + check_staleness) uses
  ``find_catalog_files`` when ``catalog_only_generator=True``.
- Each CLI command that builds an ``OrchestratorConfig`` resolves
  ``source_config`` from ``ariadne.yaml`` AND auto-enables
  ``catalog_only_generator=True`` when a SCIP block is present.

Tests must FAIL until each change lands.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# orchestrator.run() — discovery follows catalog_only_generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_uses_find_catalog_files_when_flag_on(
    tmp_path: Path, monkeypatch,
) -> None:
    """The gap to fix: flag=True must dispatch to find_catalog_files
    (and NOT to find_python_files, since the latter would silently
    drop Scala/Java/JSON/MD files).
    """
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    calls = {'find_catalog_files': False, 'find_python_files': False}

    def spy_catalog(path, **_kw):
        calls['find_catalog_files'] = True
        return []

    def spy_python(path, **_kw):
        calls['find_python_files'] = True
        return []

    monkeypatch.setattr(
        'docgen.orchestrator.find_catalog_files', spy_catalog, raising=False,
    )
    monkeypatch.setattr(
        'docgen.orchestrator.find_python_files', spy_python,
    )

    config = OrchestratorConfig(
        source_path=tmp_path,
        db_path=tmp_path / 'test.db',
        staleness_db_path=tmp_path / 'stale.db',
        dry_run=True,
        catalog_only_generator=True,
    )
    async with DocGenOrchestrator(config) as orch:
        await orch.run()

    assert calls['find_catalog_files'], (
        "flag=True must dispatch to find_catalog_files; it didn't"
    )
    assert not calls['find_python_files'], (
        'flag=True must NOT call find_python_files; it did'
    )


class TestOrchestratorCheckStalenessDiscovery:
    @pytest.mark.asyncio
    async def test_check_staleness_uses_find_catalog_files_when_flag_on(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        captured = {}

        def spy_catalog(path):
            captured['catalog_called'] = True
            return []

        monkeypatch.setattr(
            'docgen.orchestrator.find_catalog_files', spy_catalog,
            raising=False,
        )

        config = OrchestratorConfig(
            source_path=tmp_path,
            db_path=tmp_path / 'test.db',
            staleness_db_path=tmp_path / 'stale.db',
            catalog_only_generator=True,
        )
        async with DocGenOrchestrator(config) as orch:
            await orch.check_staleness()

        assert captured.get('catalog_called')


# ---------------------------------------------------------------------------
# cmd_generate auto-enables catalog_only_generator when SCIP declared
# ---------------------------------------------------------------------------


def _scip_yaml_config(tmp_path: Path) -> "Config":
    from config import Config
    src = tmp_path / 'scalaproject'
    src.mkdir()
    cfg = Config()
    cfg._config['sources'] = {
        'scalaproject': {
            'path': str(src),
            'index_kinds': {'scala': 'scip'},
            'scip': {'artifact_path': str(tmp_path / 'idx.scip')},
        }
    }
    cfg._config['default_source'] = 'scalaproject'
    return cfg


class TestCmdCheckAutoEnablesFlag:
    @pytest.mark.asyncio
    async def test_check_attaches_source_config_and_enables_flag(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """cmd_check builds an OrchestratorConfig that runs
        check_staleness. For SCIP sources it must enable
        catalog_only_generator so check reports Scala/Java files via
        find_catalog_files, not 0 via the legacy find_python_files.
        """
        import argparse

        from cli.maintenance import cmd_check

        captured_config = {}

        class _FakeOrchestrator:
            def __init__(self, config) -> None:
                captured_config['config'] = config

            async def __aenter__(self):
                return self

            async def __aexit__(self, *_) -> None:
                pass

            async def check_staleness(self):
                return {
                    'total_files': 0, 'stale_files': 0,
                    'undocumented_files': 0, 'up_to_date': 0,
                    'stale_paths': [], 'undocumented_paths': [],
                }

        monkeypatch.setattr(
            'docgen.orchestrator.DocGenOrchestrator', _FakeOrchestrator,
        )

        cfg = _scip_yaml_config(tmp_path)
        monkeypatch.setattr('cli.maintenance.get_config', lambda: cfg)

        args = argparse.Namespace(
            source='scalaproject', verbose=False, db=None,
        )
        await cmd_check(args)

        oc = captured_config.get('config')
        assert oc is not None
        assert oc.source_config is not None, (
            'cmd_check did not attach source_config — check_staleness '
            'for Scala sources will fall back to find_python_files'
        )
        assert oc.catalog_only_generator is True


class TestCmdGenerateAutoEnablesFlag:
    @pytest.mark.asyncio
    async def test_scip_source_auto_enables_catalog_only_generator(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import argparse

        from cli.generation import cmd_generate

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

        cfg = _scip_yaml_config(tmp_path)
        # cmd_generate moved to cli_generate.py; patch BOTH locations so
        # cli_generation re-export and direct cli_generate users pick up
        # the test config.
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)
        monkeypatch.setattr('cli.generate.get_config', lambda: cfg)

        # ``dry_run=False`` so the test goes through OrchestratorConfig
        # construction (the asserted behavior); the FakeOrchestrator above
        # absorbs the call without doing real work. The earlier dry_run=True
        # path now short-circuits to cost-estimate before the orchestrator is
        # built — which would defeat what this test is checking.
        args = argparse.Namespace(
            source='scalaproject', model=None, provider=None, api_key=None,
            types='explanation',
            concurrency=1, force=False, dry_run=False, verbose=False,
            path=None, db=None, no_crossrefs=False,
        )
        await cmd_generate(args)

        oc = captured_config.get('config')
        assert oc is not None
        assert oc.catalog_only_generator is True, (
            'cmd_generate must auto-enable catalog_only_generator when a '
            'source has SCIP configured — otherwise the legacy '
            'SourceAnalyzer SyntaxErrors on Scala/Java'
        )


# ---------------------------------------------------------------------------
# cmd_sync auto-enables catalog_only_generator (both code paths)
# ---------------------------------------------------------------------------


class TestCmdSyncAutoEnablesFlag:
    @pytest.mark.asyncio
    async def test_standard_sync_attaches_scip_config_and_enables_flag(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """cmd_sync (standard branch, dry_run=False) must construct an
        OrchestratorConfig with source_config set AND
        catalog_only_generator=True for SCIP-declared sources. Tooth:
        unconditional `assert oc is not None` — passes ONLY if the
        orchestrator was actually built.
        """
        import argparse

        from cli.sync import cmd_sync

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

            @property
            def config(self):
                return _StubConfig()

        class _StubConfig:
            concurrency = 1

        monkeypatch.setattr(
            'docgen.orchestrator.DocGenOrchestrator', _FakeOrchestrator,
        )

        cfg = _scip_yaml_config(tmp_path)
        monkeypatch.setattr('cli.sync.get_config', lambda: cfg)

        # Stub git interactions so we drive the regen path with a known set.
        import subprocess
        def fake_run(args, **kw):
            class R: returncode = 0; stdout = ''; stderr = ''
            r = R()
            if args[:2] == ['git', 'rev-parse']:
                r.stdout = 'abc123\n'
            elif args[:2] == ['git', 'diff']:
                r.stdout = 'X.scala\n'
            return r
        monkeypatch.setattr(subprocess, 'run', fake_run)

        from unittest.mock import MagicMock
        lib = MagicMock()
        lib.get_sync_state.return_value = ('def456', '2026-04-28T10:00:00')
        lib.find_documents_by_source_files.return_value = []
        lib.deprecate_stale_gotchas.return_value = 0
        monkeypatch.setattr('cli.sync.get_library', lambda *_a, **_k: lib)

        monkeypatch.setattr(
            'git_ops.get_current_branch', lambda p: 'main', raising=False,
        )

        args = argparse.Namespace(
            source='scalaproject', status=False, force=False, dry_run=False,
            skip_generate=False, no_export=True, vs_main=False, branch=False,
            concurrency=1,
            db=None,
        )
        await cmd_sync(args)

        oc = captured_config.get('config')
        assert oc is not None, (
            'cmd_sync did not construct an OrchestratorConfig — sync '
            'for Scala sources cannot reach the SCIP path'
        )
        assert oc.source_config is not None, (
            "OrchestratorConfig built without source_config; SCIP path "
            "won't engage"
        )
        assert oc.catalog_only_generator is True, (
            'OrchestratorConfig built with legacy flag; SyntaxError on '
            'Scala source'
        )


# ---------------------------------------------------------------------------
# Non-SCIP sources still get catalog_only_generator=True (Phase 2 Change 1)
# ---------------------------------------------------------------------------


def _non_scip_yaml_config(tmp_path: Path) -> "Config":
    """Source with no ``index_kinds`` / ``scip`` block — pure ast-grep path."""
    from config import Config
    src = tmp_path / 'plainpy'
    src.mkdir()
    cfg = Config()
    cfg._config['sources'] = {
        'plainpy': {
            'path': str(src),
        }
    }
    cfg._config['default_source'] = 'plainpy'
    return cfg


class TestNonScipSourcesAlsoGetCatalogPath:
    """Phase 2 Change 1: catalog_only_generator must default ON regardless
    of whether a source has SCIP configured. The legacy SourceAnalyzer
    path crashes on .md/.yaml/.json files (which the catalog walker now
    surfaces by default), so non-SCIP sources need the catalog generator
    too — not just SCIP-declared ones.
    """

    @pytest.mark.asyncio
    async def test_cmd_sync_non_scip_source_uses_catalog_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import argparse

        from cli.sync import cmd_sync

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

            @property
            def config(self):
                class _C: concurrency = 1
                return _C()

        monkeypatch.setattr(
            'docgen.orchestrator.DocGenOrchestrator', _FakeOrchestrator,
        )

        cfg = _non_scip_yaml_config(tmp_path)
        monkeypatch.setattr('cli.sync.get_config', lambda: cfg)

        import subprocess
        def fake_run(args, **kw):
            class R: returncode = 0; stdout = ''; stderr = ''
            r = R()
            if args[:2] == ['git', 'rev-parse']:
                r.stdout = 'abc123\n'
            elif args[:2] == ['git', 'diff']:
                # mix of .py and .md so the catalog filter has work to do
                r.stdout = 'main.py\nREADME.md\n'
            return r
        monkeypatch.setattr(subprocess, 'run', fake_run)

        lib = MagicMock()
        lib.get_sync_state.return_value = ('def456', '2026-04-28T10:00:00')
        lib.find_documents_by_source_files.return_value = []
        lib.deprecate_stale_gotchas.return_value = 0
        monkeypatch.setattr('cli.sync.get_library', lambda *_a, **_k: lib)

        monkeypatch.setattr(
            'git_ops.get_current_branch', lambda p: 'main', raising=False,
        )

        args = argparse.Namespace(
            source='plainpy', status=False, force=False, dry_run=False,
            skip_generate=False, no_export=True, vs_main=False, branch=False,
            concurrency=1,
            db=None,
        )
        await cmd_sync(args)

        oc = captured_config.get('config')
        assert oc is not None
        assert oc.catalog_only_generator is True, (
            "cmd_sync built OrchestratorConfig with catalog_only_generator=False "
            "for a non-SCIP source — that routes .md/.yaml/.json regen through "
            "SourceAnalyzer's ast.parse, which SyntaxErrors on non-Python files."
        )

    def test_cmd_generate_non_scip_source_uses_catalog_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """``cli_generate.cmd_generate`` must NOT gate
        ``catalog_only_generator`` on SCIP YAML.
        """
        import argparse
        import asyncio

        from cli.generate import cmd_generate

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

        cfg = _non_scip_yaml_config(tmp_path)
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)
        monkeypatch.setattr('cli.generate.get_config', lambda: cfg)

        args = argparse.Namespace(
            source='plainpy', model=None, provider=None, api_key=None,
            types='explanation',
            concurrency=1, force=False, dry_run=False, verbose=False,
            path=None, db=None, no_crossrefs=False,
        )
        asyncio.run(cmd_generate(args))

        oc = captured_config.get('config')
        assert oc is not None
        assert oc.catalog_only_generator is True, (
            'cmd_generate built OrchestratorConfig with catalog_only_generator=False '
            'for a non-SCIP source — multi-language regen needs the catalog path '
            'unconditionally.'
        )


# ---------------------------------------------------------------------------
# execute_merge auto-enables catalog_only_generator
# ---------------------------------------------------------------------------


class TestExecuteMergeAutoEnablesFlag:
    @pytest.mark.asyncio
    async def test_merge_attaches_scip_config_when_source_declares(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import merge as merge_mod

        source_dir = tmp_path / 'scalaproject'
        source_dir.mkdir()
        (source_dir / 'X.scala').write_text('class X\n')

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

            @property
            def config(self):
                return MagicMock(concurrency=1)

        monkeypatch.setattr(
            'docgen.orchestrator.DocGenOrchestrator', _FakeOrchestrator,
        )

        # Mock get_consumed_docs to return one consumed doc with a Scala source
        from schema import Document

        fake_doc = MagicMock(spec=Document)
        fake_doc.source_files = ['X.scala']
        fake_doc.id = 'd1'
        fake_doc.metadata = {}

        def fake_get_consumed(library, source_path, main_branch):
            return [fake_doc], ['feature/x']

        monkeypatch.setattr(
            'docgen.merge.get_consumed_docs', fake_get_consumed,
            raising=False,
        )

        # Stub subprocess for set_sync_state path
        import subprocess
        def fake_run(args, **kw):
            class R: returncode = 0; stdout = 'abc123\n'
            return R()
        monkeypatch.setattr(subprocess, 'run', fake_run)

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

        library = MagicMock()
        library.set_sync_state = MagicMock()

        await merge_mod.execute_merge(
            library=library, cfg=cfg,
            source_name='scalaproject',
            source_path=source_dir,
            db_path=None,
            since=None,
            skip_generate=False,
            no_export=True,
            delete_consumed=False,
        )

        oc = captured_config.get('config')
        assert oc is not None, (
            'execute_merge did not construct an OrchestratorConfig — '
            'is it filtering Scala files out before reaching the orchestrator?'
        )
        assert oc.source_config is not None
        assert oc.catalog_only_generator is True
