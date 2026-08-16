"""One provider call per identical text within a service lifetime.

The ask path embeds the same question at several stages (document
search, catalog positioning, clew recall, compact dispatch); the memo
guarantees only the first spends, without any cross-module plumbing.
"""
from __future__ import annotations

import asyncio

import numpy as np

from embedding import EmbeddingConfig, EmbeddingService


class CountingEmbedding(EmbeddingService):
    def __init__(self):
        super().__init__(EmbeddingConfig(api_key="test-key"))
        self.batch_calls = 0

    async def embed_batch(self, texts):
        self.batch_calls += 1
        return [np.zeros(4, dtype=np.float32) for _text in texts]


class TestEmbedMemo:
    def test_identical_text_embeds_once_per_service(self):
        service = CountingEmbedding()

        first = asyncio.run(service.embed("what is the writer?"))
        second = asyncio.run(service.embed("what is the writer?"))

        assert service.batch_calls == 1
        assert (first == second).all()

    def test_distinct_texts_each_spend(self):
        service = CountingEmbedding()

        asyncio.run(service.embed("question one"))
        asyncio.run(service.embed("question two"))

        assert service.batch_calls == 2
