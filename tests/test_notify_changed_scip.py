"""Tests for ``notify_changed`` SCIP wiring (SCIP follow-up #6).

Same-shape gap as ``cmd_catalog_sync`` had: ``notify_changed`` reads
each changed file via ``extract_elements`` but currently doesn't accept
or forward a ``source_config``, so incremental syncs of Scala/Java
sources silently extract zero elements.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from docgen.scip_config import (
    ScipUnavailableError,
    SourceScipConfig,
)


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'scalaproject' source."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'scalaproject')


class TestNotifyChangedForwardsSourceConfig:
    @pytest.mark.asyncio
    async def test_source_config_kwarg_accepted(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import catalog_writer

        f = tmp_path / 'X.scala'
        f.write_text('class X\n', encoding='utf-8')

        captured = {}

        def spy(path, source_root, *, source_config=None):
            captured['source_config'] = source_config
            return []

        monkeypatch.setattr(
            'docgen.catalog_writer.extract_elements', spy,
        )

        library = MagicMock()
        library.get_document.return_value = None

        from unittest.mock import AsyncMock
        writer = MagicMock()
        writer.add_document = AsyncMock()

        scip_cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'idx.scip',
            index_kinds={'scala': 'scip'},
        )

        await catalog_writer.notify_changed(
            library, writer, 'scalaproject',
            ['X.scala'],
            source_root=tmp_path,
            source_config=scip_cfg,
        )

        assert captured['source_config'] is scip_cfg


class TestNotifyChangedScipErrorIsTrapped:
    @pytest.mark.asyncio
    async def test_scip_error_does_not_unwind_the_batch(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A SCIP error on one file in a batch must not crash the whole
        notify_changed call. The summary entry for that file should still
        record the failure path.
        """
        from docgen import catalog_writer

        f = tmp_path / 'X.scala'
        f.write_text('class X\n', encoding='utf-8')

        def boom(*args, **kwargs):
            raise ScipUnavailableError(
                repo='scalaproject', reason='index_missing',
            )

        monkeypatch.setattr(
            'docgen.catalog_writer.extract_elements', boom,
        )

        library = MagicMock()
        library.get_document.return_value = None
        from unittest.mock import AsyncMock
        writer = MagicMock()
        writer.add_document = AsyncMock()

        scip_cfg = SourceScipConfig(
            repo='scalaproject',
            artifact_path=tmp_path / 'idx.scip',
            index_kinds={'scala': 'scip'},
        )

        # Must not raise — failures recorded per-file in the summary.
        result = await catalog_writer.notify_changed(
            library, writer, 'scalaproject',
            ['X.scala'],
            source_root=tmp_path,
            source_config=scip_cfg,
        )
        assert 'X.scala' in result
        # The file's entry exists; specific failure shape is implementation-defined,
        # but it should not show as added/modified.
        assert result['X.scala'].get('added', 0) == 0
        assert result['X.scala'].get('modified', 0) == 0
