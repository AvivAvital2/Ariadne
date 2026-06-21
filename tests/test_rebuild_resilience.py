"""Guardrails: a large rebuild is resilient and rate-limit-friendly.

* A batch that still fails after the embedding client's own retries must be
  isolated — the run finishes the other batches, persists their successes,
  and leaves the failed docs NULL so a re-run finishes them. One 429 must
  not abort the whole rebuild (the original crash).
* Concurrent embedding batches are capped, to avoid overrunning the
  provider's per-minute token budget.
"""
from __future__ import annotations

import asyncio

import numpy as np

import writer as writer_mod
from library import Library
from writer import LibraryWriter


class _FakeService:
    def __init__(self, fail_marker: str | None = None, track_peak: bool = False) -> None:
        self.fail_marker = fail_marker
        self.track_peak = track_peak
        self.in_flight = 0
        self.peak = 0

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.in_flight += 1
        self.peak = max(self.peak, self.in_flight)
        try:
            if self.track_peak:
                await asyncio.sleep(0.02)
            if self.fail_marker and any(self.fail_marker in t for t in texts):
                raise RuntimeError('Embedding API HTTP 429: rate limited')
            return [np.ones(8, dtype=np.float32) for _ in texts]
        finally:
            self.in_flight -= 1

    async def embed(self, text: str) -> np.ndarray:
        return np.ones(8, dtype=np.float32)

    async def close(self) -> None:
        pass


def _writer(lib: Library, fake: _FakeService) -> LibraryWriter:
    w = LibraryWriter(lib)
    w._embedding_service = fake
    return w


async def test_one_failing_batch_does_not_abort_the_rest(monkeypatch, tmp_path):
    monkeypatch.setattr(writer_mod, 'EMBED_BATCH_SIZE', 1)  # one doc per batch
    lib = Library(tmp_path / 't.db')
    for title in ('GOOD1', 'FAILME', 'GOOD2'):
        lib.add_document(content_type='explanation', title=title, content=f'body {title}')
    writer = _writer(lib, _FakeService(fail_marker='FAILME'))

    count = await writer.rebuild_all_embeddings(only_missing=True)

    assert count == 2, 'good batches must persist despite one failing batch'
    missing = {d.title for d in lib.list_documents_without_embedding()}
    assert missing == {'FAILME'}, 'only the failed doc stays unembedded'


async def test_concurrency_is_capped(monkeypatch, tmp_path):
    monkeypatch.setattr(writer_mod, 'EMBED_BATCH_SIZE', 1)
    lib = Library(tmp_path / 't.db')
    for i in range(6):
        lib.add_document(content_type='explanation', title=f'D{i}', content=f'body {i}')
    fake = _FakeService(track_peak=True)
    writer = _writer(lib, fake)

    count = await writer.rebuild_all_embeddings(only_missing=True)

    assert count == 6
    assert fake.peak <= writer_mod.EMBED_MAX_CONCURRENT
    assert fake.peak <= 3, 'no more than 3 embedding batches may run at once'


async def test_long_document_gets_chunked_on_rebuild(tmp_path):
    lib = Library(tmp_path / 't.db')
    long_body = 'paragraph text. ' * 60  # ~960 chars > DEFAULT_CHUNK_SIZE (500)
    doc = lib.add_document(content_type='explanation', title='LONG', content=long_body)
    writer = _writer(lib, _FakeService())

    count = await writer.rebuild_all_embeddings(only_missing=True)

    assert count == 1
    assert len(lib.get_chunks(doc.id)) >= 1, 'a long doc must be chunked on rebuild'
