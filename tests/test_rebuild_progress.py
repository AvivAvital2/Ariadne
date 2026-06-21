"""Rebuild shows a progress bar with a live ETA, plus an up-front time estimate.

The embedding rebuild knows its total doc count, so instead of printing
``rebuild: N/M docs done`` it drives the shared progress bar (the same
``cli.progress.make_progress`` used by catalog-sync / generate / themes), which
renders done/total, elapsed, and estimated remaining time. The bar is fed by a
callback the writer invokes after each batch.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest

import writer as writer_mod
from cli.progress import format_duration, make_progress
from library import Library
from rich.progress import MofNCompleteColumn, TimeRemainingColumn
from writer import LibraryWriter


class _FakeService:
    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        return [np.ones(8, dtype=np.float32) for _ in texts]

    async def embed(self, text: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)

    async def close(self) -> None:
        pass


def _writer(lib: Library) -> LibraryWriter:
    w = LibraryWriter(lib)
    w._embedding_service = _FakeService()
    return w


class TestFormatDuration:
    @pytest.mark.parametrize('seconds,expected', [
        (0, '0s'), (45, '45s'), (59, '59s'),
        (60, '1m 00s'), (90, '1m 30s'), (605, '10m 05s'),
        (3600, '1h 00m'), (3700, '1h 01m'),
    ])
    def test_compact_human_duration(self, seconds, expected) -> None:
        assert format_duration(seconds) == expected


class TestSharedBarHasEta:
    def test_make_progress_includes_count_elapsed_and_eta(self) -> None:
        # The shared bar must carry the live ETA + done/total the rebuild needs;
        # this is the one definition every countable task reuses.
        progress = make_progress()
        kinds = {type(c) for c in progress.columns}
        assert MofNCompleteColumn in kinds
        assert TimeRemainingColumn in kinds


class TestRebuildDrivesProgressCallback:
    def test_callback_gets_cumulative_count_and_stable_total(
        self, tmp_path, monkeypatch,
    ) -> None:
        # One doc per batch so the callback fires once per doc and we can see
        # the count climb to the (stable) total.
        monkeypatch.setattr(writer_mod, 'EMBED_BATCH_SIZE', 1)
        lib = Library(tmp_path / 't.db')
        for title in ('A', 'B', 'C'):
            lib.add_document(content_type='explanation', title=title, content=f'body {title}')
        try:
            seen: list[tuple[int, int]] = []
            asyncio.run(_writer(lib).rebuild_all_embeddings(
                only_missing=True, on_progress=lambda done, total: seen.append((done, total)),
            ))
        finally:
            lib.close()

        assert seen, 'on_progress must be invoked'
        assert seen[-1] == (3, 3), 'final callback reports all docs complete'
        assert all(total == 3 for _, total in seen), 'total stays the work size'
        done_seq = [done for done, _ in seen]
        assert done_seq == sorted(done_seq), 'completed count is monotonic'
