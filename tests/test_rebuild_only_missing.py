"""Guardrail: ``rebuild_all_embeddings(only_missing=True)`` embeds only the
documents whose embedding is NULL, leaving already-embedded docs untouched.
Default (``only_missing=False``) still re-embeds the whole library.

This is the cost/time fix: a post-import embed should touch the delta, not
re-bill the entire corpus.
"""
from __future__ import annotations

import numpy as np

from library import Library
from writer import LibraryWriter


class _FakeEmbeddingService:
    """Records every text it is asked to embed; returns unit vectors offline."""

    def __init__(self) -> None:
        self.embedded_texts: list[str] = []

    async def embed_batch(self, texts: list[str]) -> list[np.ndarray]:
        self.embedded_texts.extend(texts)
        return [np.ones(8, dtype=np.float32) for _ in texts]

    async def embed(self, text: str) -> np.ndarray:
        self.embedded_texts.append(text)
        return np.ones(8, dtype=np.float32)

    async def close(self) -> None:
        pass


def _seed(lib: Library) -> None:
    lib.add_document(
        content_type='explanation', title='HAS', content='already embedded',
        embedding=np.ones(8, dtype=np.float32),
    )
    lib.add_document(content_type='explanation', title='MISS1', content='needs embedding one')
    lib.add_document(content_type='explanation', title='MISS2', content='needs embedding two')


def _writer(lib: Library) -> tuple[LibraryWriter, _FakeEmbeddingService]:
    w = LibraryWriter(lib)
    fake = _FakeEmbeddingService()
    w._embedding_service = fake
    return w, fake


async def test_only_missing_embeds_just_the_null_docs(tmp_path):
    lib = Library(tmp_path / 't.db')
    _seed(lib)
    writer, fake = _writer(lib)

    n = await writer.rebuild_all_embeddings(only_missing=True)

    assert n == 2, 'only the two NULL-embedding docs should be embedded'
    joined = '\n'.join(fake.embedded_texts)
    assert 'MISS1' in joined and 'MISS2' in joined
    assert 'HAS' not in joined, 'already-embedded doc must not be re-embedded'
    assert lib.count_missing_embeddings() == 0


async def test_default_rebuilds_every_document(tmp_path):
    lib = Library(tmp_path / 't.db')
    _seed(lib)
    writer, fake = _writer(lib)

    n = await writer.rebuild_all_embeddings()

    assert n == 3, 'default must re-embed the whole library'
    joined = '\n'.join(fake.embedded_texts)
    assert 'HAS' in joined and 'MISS1' in joined and 'MISS2' in joined
