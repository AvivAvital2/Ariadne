"""Tests for catalog-sync progress callback (#8).

``sync_source_catalog`` accepts ``on_progress(current, total, current_file)``
which the CLI wires to a Rich Progress display. Without this the user
has no feedback during long syncs (scalaproject has thousands of files).
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'src' source."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'src')


class TestProgressCallback:
    @pytest.mark.asyncio
    async def test_progress_invoked_per_file(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import catalog_writer

        # Three catalog files in source_root.
        (tmp_path / 'a.py').write_text('x = 1\n')
        (tmp_path / 'b.py').write_text('y = 2\n')
        (tmp_path / 'c.py').write_text('z = 3\n')

        async def fake_sync_file(*args, **kwargs):
            from docgen.catalog_writer import SyncSummary
            return SyncSummary(file='x')

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_file_catalog', fake_sync_file,
        )

        events: list[tuple[int, int, str | None]] = []

        def cb(current: int, total: int, current_file: str | None) -> None:
            events.append((current, total, current_file))

        await catalog_writer.sync_source_catalog(
            MagicMock(), MagicMock(), 'src', tmp_path, on_progress=cb,
        )

        # At least 3 events (one per file). Total stays constant at 3.
        assert len(events) >= 3, f'expected ≥3 events, got {len(events)}'
        totals = {t for _, t, _ in events}
        assert totals == {3}, f'total should be 3 throughout; got {totals}'

    @pytest.mark.asyncio
    async def test_current_advances_monotonically(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """If `current` ever decreases, the user's progress bar would
        flicker/regress — that's the bug to catch.
        """
        from docgen import catalog_writer

        for n in range(5):
            (tmp_path / f'f{n}.py').write_text(f'x{n} = {n}\n')

        async def fake_sync_file(*args, **kwargs):
            from docgen.catalog_writer import SyncSummary
            return SyncSummary(file='x')

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_file_catalog', fake_sync_file,
        )

        currents: list[int] = []

        def cb(current: int, total: int, current_file: str | None) -> None:
            currents.append(current)

        await catalog_writer.sync_source_catalog(
            MagicMock(), MagicMock(), 'src', tmp_path, on_progress=cb,
        )

        assert currents == sorted(currents), (
            f'current must be monotonic; got: {currents}'
        )
        assert currents[-1] == 5, f'last value must be total=5; got {currents[-1]}'

    @pytest.mark.asyncio
    async def test_current_file_is_a_string_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The third callback arg is the file the sync is *about to*
        process — used by Rich to render the description line. Confirm
        it's a string path, not a Path object (the latter would render
        as the absolute repr including 'PosixPath(...)').
        """
        from docgen import catalog_writer

        f = tmp_path / 'alpha.py'
        f.write_text('x = 1\n')

        async def fake_sync_file(*args, **kwargs):
            from docgen.catalog_writer import SyncSummary
            return SyncSummary(file='x')

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_file_catalog', fake_sync_file,
        )

        files_seen: list[str | None] = []

        def cb(current: int, total: int, current_file: str | None) -> None:
            files_seen.append(current_file)

        await catalog_writer.sync_source_catalog(
            MagicMock(), MagicMock(), 'src', tmp_path, on_progress=cb,
        )

        # At least one event has a non-None current_file referencing alpha.py.
        with_file = [f for f in files_seen if f is not None]
        assert any('alpha.py' in s for s in with_file), (
            f'current_file should reference alpha.py somewhere; got {files_seen}'
        )
        # Each non-None entry is a str.
        for entry in with_file:
            assert isinstance(entry, str)
