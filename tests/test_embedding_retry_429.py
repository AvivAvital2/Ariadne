"""Guardrail: rate-limit (429) and request-timeout (408) responses are
*transient* and must be retried, not treated as permanent client errors.

The original ``embed_batch`` bailed on the first 4xx unconditionally::

    if 400 <= e.response.status_code < 500:
        raise last_error from e

That classified 429 as fatal. During a large rebuild a single
``429 Too Many Requests`` (OpenAI per-minute token cap) aborted the whole
run and orphaned every in-flight batch. These tests pin the contract:

* 429 → retried, and a subsequent 200 succeeds.
* persistent 429 → retried more than once, then surfaces as an error.
* genuine permanent 4xx (400) → still fails fast with no retry (so the
  fix widens *only* the transient cases, not all 4xx).
"""
from __future__ import annotations

import httpx
import numpy as np
import pytest

from embedding import EmbeddingConfig, EmbeddingService


def _embedding_payload() -> dict:
    """A minimal well-formed OpenAI embeddings success body (4-dim vector)."""
    return {'data': [{'index': 0, 'embedding': [0.1, 0.2, 0.3, 0.4]}]}


def _service_with_script(status_sequence: list[int]) -> tuple[EmbeddingService, dict]:
    """Build a service whose mocked transport replays ``status_sequence``.

    Each entry is the HTTP status for that call; 200 returns a valid
    embedding body, anything else returns an OpenAI-shaped error body.
    The last entry repeats if more calls are made than scripted.
    ``calls['n']`` records how many HTTP requests were issued.
    """
    calls = {'n': 0}

    def handler(request: httpx.Request) -> httpx.Response:
        i = calls['n']
        calls['n'] += 1
        status = status_sequence[min(i, len(status_sequence) - 1)]
        if status == 200:
            return httpx.Response(200, json=_embedding_payload())
        return httpx.Response(
            status,
            json={'error': {'message': 'boom', 'type': 'tokens',
                            'code': 'rate_limit_exceeded'}},
        )

    service = EmbeddingService(EmbeddingConfig(api_key='test-key', dimensions=4))
    service._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url='https://api.openai.com/v1',
    )
    return service, calls


@pytest.fixture(autouse=True)
def _no_real_sleep(monkeypatch):
    """Neutralize backoff so retry tests don't wait real seconds."""
    async def _instant(_delay):
        return None

    monkeypatch.setattr('asyncio.sleep', _instant)


async def test_429_then_200_retries_and_succeeds():
    service, calls = _service_with_script([429, 200])
    try:
        embeddings = await service.embed_batch(['hello'])
    finally:
        await service.close()

    assert len(embeddings) == 1
    assert calls['n'] == 2, 'expected one retry after the 429, then success'


async def test_persistent_429_retries_more_than_once_then_raises():
    service, calls = _service_with_script([429])
    try:
        with pytest.raises(RuntimeError, match='429'):
            await service.embed_batch(['hello'])
    finally:
        await service.close()

    assert calls['n'] > 1, 'a 429 must be retried, not raised on first sight'


async def test_400_fails_fast_without_retry():
    service, calls = _service_with_script([400])
    try:
        with pytest.raises(RuntimeError, match='400'):
            await service.embed_batch(['hello'])
    finally:
        await service.close()

    assert calls['n'] == 1, 'permanent 4xx must not be retried'


# --- Broader embed_batch / config coverage (same module Slice 1 modifies) ---


def _service(handler):
    """Build a service whose mocked transport delegates to ``handler(request, call_n)``."""
    calls = {'n': 0}

    def counting(request: httpx.Request) -> httpx.Response:
        calls['n'] += 1
        return handler(request, calls['n'])

    service = EmbeddingService(EmbeddingConfig(api_key='test-key', dimensions=4))
    service._client = httpx.AsyncClient(
        transport=httpx.MockTransport(counting),
        base_url='https://api.openai.com/v1',
    )
    return service, calls


async def test_embed_single_returns_one_normalized_vector():
    service, _ = _service(lambda req, n: httpx.Response(200, json=_embedding_payload()))
    try:
        vec = await service.embed('hello')
    finally:
        await service.close()

    assert vec.shape == (4,)
    assert abs(float(np.linalg.norm(vec)) - 1.0) < 1e-5


async def test_no_texts_returns_empty_without_api_call():
    service, calls = _service(lambda req, n: httpx.Response(200, json=_embedding_payload()))
    try:
        out = await service.embed_batch([])
    finally:
        await service.close()

    assert out == []
    assert calls['n'] == 0


async def test_all_empty_texts_return_zero_vectors_without_api_call():
    service, calls = _service(lambda req, n: httpx.Response(200, json=_embedding_payload()))
    try:
        out = await service.embed_batch(['', '   '])
    finally:
        await service.close()

    assert len(out) == 2
    assert all(not np.any(v) for v in out)
    assert calls['n'] == 0, 'all-empty input must not hit the API'


async def test_zero_vector_response_is_not_normalized():
    service, _ = _service(
        lambda req, n: httpx.Response(
            200, json={'data': [{'index': 0, 'embedding': [0.0, 0.0, 0.0, 0.0]}]}
        )
    )
    try:
        out = await service.embed_batch(['x'])
    finally:
        await service.close()

    assert not np.any(out[0]), 'zero vector must stay zero (no divide-by-zero)'


async def test_generic_request_error_is_retried_then_succeeds():
    def handler(req, n):
        if n == 1:
            raise httpx.ConnectError('transient network blip')
        return httpx.Response(200, json=_embedding_payload())

    service, calls = _service(handler)
    try:
        out = await service.embed_batch(['x'])
    finally:
        await service.close()

    assert len(out) == 1
    assert calls['n'] == 2, 'a transient connection error must be retried'


async def test_dimensions_none_still_embeds():
    service, _ = _service(lambda req, n: httpx.Response(200, json=_embedding_payload()))
    service.config = EmbeddingConfig(api_key='test-key', dimensions=None)
    try:
        out = await service.embed_batch(['x'])
    finally:
        await service.close()

    assert len(out) == 1


def test_config_base_url_and_api_key_resolution(monkeypatch):
    explicit = EmbeddingConfig(api_key='k', base_url='http://example/v1')
    assert explicit.get_base_url() == 'http://example/v1'
    assert explicit.get_api_key() == 'k'

    monkeypatch.setenv('OPENAI_API_KEY', 'env-key')
    monkeypatch.delenv('OPENAI_BASE_URL', raising=False)
    from_env = EmbeddingConfig()
    assert from_env.get_api_key() == 'env-key'
    assert from_env.get_base_url() == 'https://api.openai.com/v1'

    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    with pytest.raises(ValueError, match='API key'):
        EmbeddingConfig().get_api_key()
