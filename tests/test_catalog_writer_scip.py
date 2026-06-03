"""Tests for catalog_writer + cmd_catalog_sync SCIP integration (Phase C).

Pins:
- CATALOG_EXTS includes .scala/.sbt/.java so iter_catalog_files picks them up.
- SyncSummary carries an optional ``scip_error`` field.
- ``_element_metadata`` includes the ``documentation`` dict when present.
- ``sync_file_catalog`` traps ``ScipError`` and returns a structured
  failure summary instead of letting the exception unwind.
- ``cmd_catalog_sync`` exits 2 when any summary carries a SCIP error.
- ``--allow-degraded`` is wired on the argparse subparser.

Tests must FAIL until the implementation lands.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'scalaproject' source."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'scalaproject')


@pytest.fixture(autouse=True)
def _mock_embedding(monkeypatch):
    """Stub the embedding service so tests don't need OPENAI_API_KEY.

    Previously the test_source_config_reaches_extract_elements test
    incidentally avoided hitting the embedding path because the
    MagicMock library returned a truthy MagicMock from
    ``library.get_document`` (treated as 'doc exists'). After the
    Phase 3 migration, ``scoped.get_document`` correctly returns None
    for a MagicMock (it can't pass the closure source_name check),
    which routes the test through the real ``writer.add_document``
    path — and the writer needs to embed.
    """
    import numpy as np

    async def fake_embed(self, text):
        return np.zeros(3072, dtype=np.float32)

    async def fake_embed_batch(self, texts):
        return [np.zeros(3072, dtype=np.float32) for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr(
        'embedding.EmbeddingService.embed_batch', fake_embed_batch,
    )
    monkeypatch.setattr(
        'embedding.EmbeddingService._get_client', fake_get_client,
    )
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)

# ---------------------------------------------------------------------------
# CATALOG_EXTS
# ---------------------------------------------------------------------------


class TestCatalogExtsExtended:
    def test_scala_extensions_included(self) -> None:
        from docgen.catalog_writer import CATALOG_EXTS

        assert '.scala' in CATALOG_EXTS
        assert '.sbt' in CATALOG_EXTS

    def test_java_extension_included(self) -> None:
        from docgen.catalog_writer import CATALOG_EXTS

        assert '.java' in CATALOG_EXTS


# ---------------------------------------------------------------------------
# SyncSummary.scip_error
# ---------------------------------------------------------------------------


class TestSyncSummaryScipError:
    def test_default_scip_error_is_none(self) -> None:
        from docgen.catalog_writer import SyncSummary

        s = SyncSummary(file='x')
        assert s.scip_error is None

    def test_can_carry_scip_error(self) -> None:
        from docgen.catalog_writer import SyncSummary
        from docgen.scip_config import ScipUnavailableError

        err = ScipUnavailableError(repo='r', reason='index_missing')
        s = SyncSummary(file='x', skipped=True, scip_error=err)
        assert s.scip_error is err
        assert s.skipped is True


# ---------------------------------------------------------------------------
# _element_metadata — documentation round-trip
# ---------------------------------------------------------------------------


class TestElementMetadataDocumentation:
    def test_documentation_omitted_when_none(self) -> None:
        from docgen.catalog_extractor import ElementInfo
        from docgen.catalog_writer import _element_metadata

        el = ElementInfo(
            language='python', subtype='function', file='m.py',
            qualified_name='m.foo', signature='def foo()',
            line_start=1, line_end=1, col_start=0, col_end=10,
            documentation=None,
        )
        meta = _element_metadata(el, 'src')
        assert 'documentation' not in meta

    def test_documentation_included_when_present(self) -> None:
        """A SCIP-extracted element with structured docs must propagate
        ``documentation`` into the catalog metadata so the LLM has it.
        """
        from docgen.catalog_extractor import ElementInfo
        from docgen.catalog_writer import _element_metadata

        doc = {
            'summary': 'Computes.',
            'params': {'x': 'the input'},
            'returns': 'the output',
        }
        el = ElementInfo(
            language='scala', subtype='scala_def', file='X.scala',
            qualified_name='com.example.X.compute',
            signature='def compute',
            line_start=1, line_end=5, col_start=0, col_end=20,
            documentation=doc,
        )
        meta = _element_metadata(el, 'scalaproject')
        assert 'documentation' in meta
        assert meta['documentation'] == doc


# ---------------------------------------------------------------------------
# sync_file_catalog: ScipError → structured summary
# ---------------------------------------------------------------------------


class TestSyncFileCatalogScipError:
    @pytest.mark.asyncio
    async def test_scip_error_returns_skipped_summary(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """When extract_elements raises a ScipError, sync_file_catalog
        must trap it and return a SyncSummary with skipped=True and the
        error attached — NOT propagate the exception (which would crash
        the whole batch sync) AND not silently fall back.
        """
        from docgen import catalog_writer
        from docgen.scip_config import ScipUnavailableError

        f = tmp_path / 'X.scala'
        f.write_text('class X\n', encoding='utf-8')

        def boom(*args, **kwargs):
            raise ScipUnavailableError(
                repo='scalaproject', reason='index_missing',
            )

        monkeypatch.setattr(
            'docgen.catalog_writer.extract_elements', boom,
        )

        # Library / writer mocks — sync_file_catalog reads them but we
        # short-circuit before any DB writes happen.
        from unittest.mock import MagicMock

        library = MagicMock()
        library.get_document.return_value = None
        writer = MagicMock()
        writer.add_document = AsyncMock()

        summary = await catalog_writer.sync_file_catalog(
            library, writer, 'scalaproject', tmp_path, f,
        )
        assert summary.skipped is True
        assert isinstance(summary.scip_error, ScipUnavailableError)
        # Critical: no DB writes happen on the failure path.
        writer.add_document.assert_not_called()


# ---------------------------------------------------------------------------
# CLI: --allow-degraded flag
# ---------------------------------------------------------------------------


class TestCliAllowDegradedFlag:
    def test_argparse_recognizes_allow_degraded(self) -> None:
        """The flag exists on the catalog-sync subparser."""
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['catalog-sync', '--allow-degraded'])
        assert args.command == 'catalog-sync'
        assert args.allow_degraded is True

    def test_default_allow_degraded_false(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['catalog-sync'])
        assert getattr(args, 'allow_degraded', None) is False


# ---------------------------------------------------------------------------
# Wiring: source_config flows from cmd_catalog_sync → sync_source_catalog
# → sync_file_catalog → extract_elements
# ---------------------------------------------------------------------------


class TestSourceConfigWiring:
    """Without this end-to-end wiring the CLI cannot reach the SCIP path
    even when ``ariadne.yaml`` has it declared. This catches the silent
    case where every Scala file falls through to ``return []`` (because
    ``extract_elements`` never receives the source_config it needs).
    """

    @pytest.mark.asyncio
    async def test_source_config_reaches_extract_elements(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        import argparse

        from cli.generation import cmd_catalog_sync

        # Set up a fake source path with a scala file present.
        source_dir = tmp_path / 'scalaproject'
        source_dir.mkdir()
        (source_dir / 'X.scala').write_text('class X\n', encoding='utf-8')

        captured: dict = {}

        def spy_extract_elements(path, source_root, *, source_config=None):
            captured['source_config'] = source_config
            captured['path_suffix'] = path.suffix
            return []

        monkeypatch.setattr(
            'docgen.catalog_writer.extract_elements', spy_extract_elements,
        )

        # Wire ariadne.yaml-style source_config via Config.
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

        from unittest.mock import MagicMock
        monkeypatch.setattr(
            'cli.generation.get_library', lambda *_a, **_k: MagicMock(),
        )

        args = argparse.Namespace(
            source='scalaproject',
            allow_degraded=False,
            db=None,
        )
        await cmd_catalog_sync(args)

        # The .scala file must have flowed through extract_elements with
        # the SCIP-aware source_config attached.
        assert captured['path_suffix'] == '.scala'
        assert captured['source_config'] is not None, (
            'source_config did not reach extract_elements — the CLI is '
            'fully decoupled from the SCIP path'
        )
        assert captured['source_config'].repo == 'scalaproject'


# ---------------------------------------------------------------------------
# CLI: cmd_catalog_sync exit code on SCIP error
# ---------------------------------------------------------------------------


class TestCliExitCodeOnScipError:
    @pytest.mark.asyncio
    async def test_exits_2_when_summaries_have_scip_errors(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        """If any returned SyncSummary has scip_error, cmd_catalog_sync
        must signal failure with exit code 2 — the contract for the
        fail-loud path through the CLI.
        """
        import argparse

        from cli.generation import cmd_catalog_sync
        from docgen.catalog_writer import SyncSummary
        from docgen.scip_config import ScipUnavailableError

        # Set up a fake source path under tmp_path that resolves cleanly.
        source_dir = tmp_path / 'scalaproject'
        source_dir.mkdir()

        # Patch sync_source_catalog to return our error-laden summaries.
        async def fake_sync(library, writer, source_name, source_path, *, source_config=None, on_progress=None, concurrency=4, **_kw):
            return [
                SyncSummary(
                    file='src/main/scala/X.scala',
                    skipped=True,
                    scip_error=ScipUnavailableError(
                        repo=source_name, reason='index_missing',
                    ),
                ),
            ]

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_source_catalog', fake_sync,
        )

        # Make get_config return something with the right source.
        from config import Config

        cfg = Config()
        cfg._config['sources'] = {'scalaproject': {'path': str(source_dir)}}
        cfg._config['default_source'] = 'scalaproject'
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)

        # get_library returns a mock.
        from unittest.mock import MagicMock

        monkeypatch.setattr(
            'cli.generation.get_library', lambda *_a, **_k: MagicMock(),
        )

        args = argparse.Namespace(
            source='scalaproject',
            allow_degraded=False,
            db=None,
        )
        rc = await cmd_catalog_sync(args)
        assert rc == 2, (
            "catalog-sync must exit 2 when any SyncSummary carries a "
            "scip_error — that's the fail-loud contract"
        )

    @pytest.mark.asyncio
    async def test_exits_zero_on_clean_sync(
        self, monkeypatch, tmp_path: Path,
    ) -> None:
        """No scip_errors → exit 0 (regression check on the new branch)."""
        import argparse

        from cli.generation import cmd_catalog_sync
        from docgen.catalog_writer import SyncSummary

        source_dir = tmp_path / 'scalaproject'
        source_dir.mkdir()

        async def fake_sync(library, writer, source_name, source_path, *, source_config=None, on_progress=None, concurrency=4, **_kw):
            return [SyncSummary(file='X.scala', added=3)]

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_source_catalog', fake_sync,
        )

        from config import Config

        cfg = Config()
        cfg._config['sources'] = {'scalaproject': {'path': str(source_dir)}}
        cfg._config['default_source'] = 'scalaproject'
        monkeypatch.setattr('cli.generation.get_config', lambda: cfg)

        from unittest.mock import MagicMock

        monkeypatch.setattr(
            'cli.generation.get_library', lambda *_a, **_k: MagicMock(),
        )

        args = argparse.Namespace(
            source='scalaproject',
            allow_degraded=False,
            db=None,
        )
        rc = await cmd_catalog_sync(args)
        assert rc == 0
