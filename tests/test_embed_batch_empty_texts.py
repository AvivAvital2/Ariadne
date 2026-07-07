"""embed_batch contract for empty/whitespace texts: positional alignment.

``embed_batch`` deliberately never sends empty/whitespace-only texts to the
API (the call would be wasted) and substitutes zero vectors. The contract
these tests pin: the returned list is ALWAYS positionally aligned with the
input — a partially-empty batch must not shift real vectors onto the wrong
positions (callers zip the result against their doc/chunk lists, so a
shift silently assigns wrong embeddings).
"""
from __future__ import annotations

import json

import httpx
import numpy as np

from embedding import EmbeddingConfig, EmbeddingService


def _echo_service() -> tuple[EmbeddingService, dict]:
    """A service whose transport returns one distinct vector per input,
    first component = 1-based input position (so misalignment is visible)."""
    calls = {'n': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls['n'] += 1
        inputs = json.loads(request.content)['input']
        data = [
            {'index': i, 'embedding': [float(i + 1), 0.0, 0.0, 0.0]}
            for i in range(len(inputs))
        ]
        return httpx.Response(200, json={'data': data})

    service = EmbeddingService(EmbeddingConfig(api_key='test-key', dimensions=4))
    service._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url='https://api.openai.com/v1',
    )
    return service, calls


async def test_partially_empty_batch_stays_positionally_aligned():
    service, calls = _echo_service()
    try:
        vectors = await service.embed_batch(['', 'real text', '   '])
    finally:
        await service.close()
    assert calls['n'] == 1
    assert len(vectors) == 3, 'result must match input length, not valid count'
    assert not vectors[0].any(), 'empty text gets a zero vector in place'
    assert vectors[1].any(), 'the real text keeps its (normalized) vector'
    assert not vectors[2].any(), 'whitespace text gets a zero vector in place'


async def test_all_empty_batch_makes_no_api_call():
    service, calls = _echo_service()
    try:
        vectors = await service.embed_batch(['', '   '])
    finally:
        await service.close()
    assert calls['n'] == 0
    assert len(vectors) == 2
    assert not vectors[0].any() and not vectors[1].any()
