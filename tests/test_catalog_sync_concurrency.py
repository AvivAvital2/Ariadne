"""Tests for concurrent catalog-sync.

Sequential per-file processing is too slow for large repos (scalaproject
estimated at ~7 hours). Adding bounded concurrency lets multiple files
process in parallel while honoring SQLite's writer-serialization.
"""
from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture(autouse=True)
def _test_config(tmp_path: Path, monkeypatch):
    """Provide a Config that knows about the 'src' source."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'src')


class TestConcurrentSync:
    @pytest.mark.asyncio
    async def test_concurrency_kwarg_accepted(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """sync_source_catalog must accept ``concurrency`` so the CLI
        can expose ``--concurrency``.
        """
        from docgen import catalog_writer

        async def fake_sync_file(*args, **kwargs):
            from docgen.catalog_writer import SyncSummary
            return SyncSummary(file='x')

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_file_catalog', fake_sync_file,
        )

        await catalog_writer.sync_source_catalog(
            MagicMock(), MagicMock(), 'src', tmp_path, concurrency=4,
        )

    @pytest.mark.asyncio
    async def test_files_process_concurrently(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """With ``concurrency=4``, four files should be in flight
        simultaneously. Pin via a barrier: each task increments a shared
        counter, blocks until all four hit it, then proceeds. If processing
        is sequential, only one task ever blocks on the barrier and the
        test deadlocks → asyncio.wait_for fires.
        """
        from docgen import catalog_writer

        # Five files; concurrency=4 so the first 4 should be in flight.
        for n in range(5):
            (tmp_path / f'f{n}.py').write_text(f'x{n} = {n}\n')

        in_flight = 0
        max_in_flight = 0
        barrier_evt = asyncio.Event()
        lock = asyncio.Lock()

        async def fake_sync_file(*args, **kwargs):
            from docgen.catalog_writer import SyncSummary
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
                if in_flight >= 4:
                    barrier_evt.set()
            # Wait until at least 4 tasks are in flight.
            await asyncio.wait_for(barrier_evt.wait(), timeout=2.0)
            async with lock:
                in_flight -= 1
            return SyncSummary(file=str(args[4]))  # 5th positional = file

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_file_catalog', fake_sync_file,
        )

        await catalog_writer.sync_source_catalog(
            MagicMock(), MagicMock(), 'src', tmp_path, concurrency=4,
        )

        assert max_in_flight >= 4, (
            f'expected at least 4 concurrent in-flight tasks; got '
            f'{max_in_flight} — sync is processing files sequentially'
        )

    @pytest.mark.asyncio
    async def test_concurrency_one_serializes(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """concurrency=1 must process strictly sequentially — important
        for callers that don't want concurrency for safety reasons.
        """
        from docgen import catalog_writer

        for n in range(3):
            (tmp_path / f'g{n}.py').write_text(f'y{n} = {n}\n')

        in_flight = 0
        max_in_flight = 0
        lock = asyncio.Lock()

        async def fake_sync_file(*args, **kwargs):
            from docgen.catalog_writer import SyncSummary
            nonlocal in_flight, max_in_flight
            async with lock:
                in_flight += 1
                max_in_flight = max(max_in_flight, in_flight)
            await asyncio.sleep(0.01)
            async with lock:
                in_flight -= 1
            return SyncSummary(file='x')

        monkeypatch.setattr(
            'docgen.catalog_writer.sync_file_catalog', fake_sync_file,
        )

        await catalog_writer.sync_source_catalog(
            MagicMock(), MagicMock(), 'src', tmp_path, concurrency=1,
        )

        assert max_in_flight == 1


class TestCliConcurrencyFlag:
    def test_argparse_recognizes_concurrency_flag(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['catalog-sync', '--concurrency', '8'])
        assert args.command == 'catalog-sync'
        assert args.concurrency == 8

    def test_default_concurrency_is_set(self) -> None:
        from cli.main import create_parser

        parser = create_parser()
        args = parser.parse_args(['catalog-sync'])
        # Default should be a reasonable parallelism for batch sync.
        assert getattr(args, 'concurrency', 0) >= 4
