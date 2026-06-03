"""Tests for Anthropic Message Batches API wiring.

The provider gains three batch-mode methods that talk to
``/v1/messages/batches`` (POST), ``/v1/messages/batches/{id}`` (GET, status),
and ``/v1/messages/batches/{id}/results`` (GET, JSONL). Exercising real HTTP
is out of scope; these tests verify payload shape, status-polling control
flow, and JSONL result parsing against a fake httpx client.

Why this is structured as a separate test file: batch flow is a different
control path from the synchronous ``call`` (request → poll → fetch). Putting
batch tests here keeps the per-call test file focused on caching / retry.
"""
from __future__ import annotations

import json
from typing import Any

import pytest

from docgen.llm.anthropic_batch import AnthropicBatchStrategy

# ---------------------------------------------------------------------------
# Test doubles — capture the request shape, replay canned responses
# ---------------------------------------------------------------------------


class _FakeResponse:
    def __init__(self, payload: Any, *, status_code: int = 200, text: str | None = None):
        self._payload = payload
        self.status_code = status_code
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload

    def aiter_lines(self):
        async def _gen():
            for line in self.text.splitlines():
                if line.strip():
                    yield line
        return _gen()


class _ScriptedClient:
    """Records every request; replays a queue of pre-scripted responses.

    Each entry in ``responses`` is matched to one request in order. Use
    ``add_post`` / ``add_get`` to script incrementally.
    """

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []  # (method, url, json)
        self.responses: list[_FakeResponse] = []
        self.headers: dict[str, str] = {}

    def add_post(self, response: Any, *, status_code: int = 200) -> None:
        self.responses.append(_FakeResponse(response, status_code=status_code))

    def add_get(self, response: Any, *, status_code: int = 200, text: str | None = None) -> None:
        self.responses.append(_FakeResponse(response, status_code=status_code, text=text))

    async def post(self, url: str, json: dict | None = None):
        self.calls.append(('POST', url, json))
        return self._next()

    async def get(self, url: str):
        self.calls.append(('GET', url, None))
        return self._next()

    def _next(self):
        # A queued ``Exception`` is raised (network errors); a real
        # ``httpx.Response`` raises via ``raise_for_status()`` on 4xx/5xx; a
        # ``_FakeResponse`` succeeds. Lets the retry tests script failures.
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        return None


@pytest.fixture
def scripted(monkeypatch):
    from docgen.llm.anthropic import AnthropicProvider
    sc = _ScriptedClient()
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: sc)
    return sc


# ---------------------------------------------------------------------------
# submit_batch — payload shape
# ---------------------------------------------------------------------------


async def test_submit_batch_posts_one_request_per_input(scripted):
    """submit_batch packs each (custom_id, system, user) tuple into the
    ``requests`` array Anthropic expects.
    """
    from docgen.llm.anthropic import AnthropicProvider, BatchRequest

    scripted.add_post({
        'id': 'batch_abc',
        'processing_status': 'in_progress',
        'request_counts': {'processing': 2, 'succeeded': 0, 'errored': 0},
    })

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    submission = await AnthropicBatchStrategy(provider).submit_batch([
        BatchRequest(custom_id='f1:explanation', system_prompt='S1', user_prompt='U1'),
        BatchRequest(custom_id='f2:architecture', system_prompt='S2', user_prompt='U2'),
    ])

    assert submission.batch_id == 'batch_abc'

    method, url, body = scripted.calls[0]
    assert method == 'POST'
    assert url == '/messages/batches'
    assert isinstance(body, dict)
    requests = body['requests']
    assert len(requests) == 2
    first = requests[0]
    assert first['custom_id'] == 'f1:explanation'
    assert first['params']['model'] == 'claude-opus-4-6'
    assert first['params']['messages'] == [{'role': 'user', 'content': 'U1'}]
    # System should still be cacheable in batch — same shape as sync calls
    assert isinstance(first['params']['system'], list)
    assert first['params']['system'][0]['cache_control'] == {'type': 'ephemeral'}


async def test_submit_batch_emits_message_batches_beta_header():
    """The /messages/batches endpoint requires its own beta header."""
    from docgen.llm.anthropic import (
        ANTHROPIC_MESSAGE_BATCHES_BETA,
        AnthropicProvider,
    )
    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    client = provider._get_client()
    headers = dict(client.headers)
    assert 'anthropic-beta' in headers
    # Both betas should ride the same header value, comma-separated.
    assert ANTHROPIC_MESSAGE_BATCHES_BETA in headers['anthropic-beta']
    assert 'prompt-caching' in headers['anthropic-beta']


# ---------------------------------------------------------------------------
# poll_batch — status loop
# ---------------------------------------------------------------------------


async def test_poll_batch_returns_when_processing_status_ends(scripted):
    """Poll loops until ``processing_status == 'ended'``, then returns the
    final status payload — caller fetches results separately.
    """
    from docgen.llm.anthropic import AnthropicProvider

    scripted.add_get({'processing_status': 'in_progress', 'request_counts': {'processing': 5}})
    scripted.add_get({'processing_status': 'in_progress', 'request_counts': {'processing': 3, 'succeeded': 2}})
    scripted.add_get({
        'processing_status': 'ended',
        'request_counts': {'processing': 0, 'succeeded': 4, 'errored': 1},
    })

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    final = await AnthropicBatchStrategy(provider).poll_batch('batch_abc', poll_interval=0)

    assert final.processing_status == 'ended'
    assert final.succeeded == 4
    assert final.errored == 1
    assert len(scripted.calls) == 3
    for method, url, _ in scripted.calls:
        assert method == 'GET'
        assert url == '/messages/batches/batch_abc'


async def test_poll_batch_invokes_progress_callback(scripted):
    """Each poll fires the optional callback with the latest counts."""
    from docgen.llm.anthropic import AnthropicProvider

    scripted.add_get({'processing_status': 'in_progress', 'request_counts': {'processing': 5}})
    scripted.add_get({
        'processing_status': 'ended',
        'request_counts': {'processing': 0, 'succeeded': 5, 'errored': 0},
    })

    seen: list[tuple[int, int, int]] = []

    def cb(processing: int, succeeded: int, errored: int) -> None:
        seen.append((processing, succeeded, errored))

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    await AnthropicBatchStrategy(provider).poll_batch('batch_abc', poll_interval=0, on_progress=cb)

    assert seen == [(5, 0, 0), (0, 5, 0)]


# ---------------------------------------------------------------------------
# fetch_batch_results — JSONL parsing
# ---------------------------------------------------------------------------


async def test_fetch_batch_results_parses_jsonl_keyed_by_custom_id(scripted):
    """Anthropic returns JSONL; each line carries the custom_id and the
    /messages-shaped response (or an error). The provider returns a
    ``dict[custom_id, str | None]`` of extracted text.
    """
    from docgen.llm.anthropic import AnthropicProvider

    line_ok = json.dumps({
        'custom_id': 'f1:explanation',
        'result': {
            'type': 'succeeded',
            'message': {
                'content': [{'type': 'text', 'text': 'Generated explanation'}],
                'usage': {'input_tokens': 100, 'output_tokens': 50},
            },
        },
    })
    line_err = json.dumps({
        'custom_id': 'f2:architecture',
        'result': {'type': 'errored', 'error': {'type': 'overloaded_error'}},
    })
    jsonl = '\n'.join([line_ok, line_err])

    scripted.add_get({}, text=jsonl)

    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    results = await AnthropicBatchStrategy(provider).fetch_batch_results('batch_abc')

    assert results['f1:explanation'] == 'Generated explanation'
    assert results['f2:architecture'] is None  # errored → None per protocol contract

    method, url, _ = scripted.calls[0]
    assert method == 'GET'
    assert url == '/messages/batches/batch_abc/results'


# ---------------------------------------------------------------------------
# fetch_batch_results — cache_stats extraction (#45.11)
# ---------------------------------------------------------------------------


async def test_fetch_batch_results_records_cache_stats_per_row(scripted):
    """Each succeeded row's ``usage.cache_creation_input_tokens`` /
    ``cache_read_input_tokens`` updates the provider's ``cache_stats``
    so batch runs report cache savings the same way streaming runs do.

    Without this, the CLI's cache-savings telemetry zeroes out on
    every batch run — users would see legit savings in sync mode
    and silently lose the readout in batch mode.
    """
    from docgen.llm.anthropic import AnthropicProvider

    line_cache_write = json.dumps({
        'custom_id': '0',
        'result': {
            'type': 'succeeded',
            'message': {
                'content': [{'type': 'text', 'text': 'doc 1'}],
                'usage': {
                    'input_tokens': 1000,
                    'output_tokens': 50,
                    'cache_creation_input_tokens': 800,
                    'cache_read_input_tokens': 0,
                },
            },
        },
    })
    line_cache_read = json.dumps({
        'custom_id': '1',
        'result': {
            'type': 'succeeded',
            'message': {
                'content': [{'type': 'text', 'text': 'doc 2'}],
                'usage': {
                    'input_tokens': 200,
                    'output_tokens': 50,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 800,
                },
            },
        },
    })
    line_no_cache = json.dumps({
        'custom_id': '2',
        'result': {
            'type': 'succeeded',
            'message': {
                'content': [{'type': 'text', 'text': 'doc 3'}],
                'usage': {
                    'input_tokens': 200,
                    'output_tokens': 50,
                },
            },
        },
    })
    jsonl = '\n'.join([line_cache_write, line_cache_read, line_no_cache])

    scripted.add_get({}, text=jsonl)

    provider = AnthropicProvider(model='claude-3-5-sonnet', api_key='test')
    await AnthropicBatchStrategy(provider).fetch_batch_results('batch_cache')

    # Three succeeded rows → three calls counted.
    assert provider.cache_stats.total_calls == 3
    # Row 0 wrote the cache (create_tokens > 0).
    assert provider.cache_stats.cache_writes == 1
    # Row 1 read the cache (read_tokens > 0, no create).
    assert provider.cache_stats.cache_reads == 1
    # Row 2 had neither — cache miss / not engaged.
    assert provider.cache_stats.cache_misses == 1
    assert provider.cache_stats.total_create_tokens == 800
    assert provider.cache_stats.total_read_tokens == 800


async def test_fetch_batch_results_skips_cache_stats_for_errored(
    scripted,
):
    """Errored rows have no ``message.usage`` block; cache_stats
    counters must not increment for them. Pairs with the per-row
    test above so a fix that runs ``record(0, 0)`` for every line
    (including errored) bumps total_calls beyond what's correct."""
    from docgen.llm.anthropic import AnthropicProvider

    line_ok = json.dumps({
        'custom_id': '0',
        'result': {
            'type': 'succeeded',
            'message': {
                'content': [{'type': 'text', 'text': 'doc 1'}],
                'usage': {
                    'input_tokens': 200,
                    'output_tokens': 50,
                    'cache_creation_input_tokens': 0,
                    'cache_read_input_tokens': 0,
                },
            },
        },
    })
    line_err = json.dumps({
        'custom_id': '1',
        'result': {
            'type': 'errored', 'error': {'type': 'overloaded_error'},
        },
    })
    jsonl = '\n'.join([line_ok, line_err])
    scripted.add_get({}, text=jsonl)

    provider = AnthropicProvider(model='claude-3-5-sonnet', api_key='test')
    await AnthropicBatchStrategy(provider).fetch_batch_results('batch_partial_err')

    # Only the succeeded row should bump total_calls.
    assert provider.cache_stats.total_calls == 1


@pytest.mark.asyncio
async def test_cancel_batch_posts_to_cancel_endpoint(scripted):
    """cancel_batch POSTs to /messages/batches/<id>/cancel so an aborted
    run stops the batch (and its billing) at Anthropic."""
    from docgen.llm.anthropic import AnthropicProvider

    scripted.add_post({'id': 'batch_abc', 'processing_status': 'canceling'})
    provider = AnthropicProvider(model='claude-opus-4-6', api_key='test')
    await AnthropicBatchStrategy(provider).cancel_batch('batch_abc')

    assert ('POST', '/messages/batches/batch_abc/cancel', None) in scripted.calls


# ---------------------------------------------------------------------------
# _batch_request_with_retry — transient-error retry / quota / exhaustion.
# The happy-path scripts above never exercise these branches; without these
# the retry/backoff/quota path was untested (caught via coverage).
# ---------------------------------------------------------------------------


async def test_batch_retries_transient_5xx_then_succeeds(scripted):
    """A 5xx on the batch endpoint is retried; the next success returns."""
    import httpx

    from docgen.llm.anthropic import AnthropicProvider, BatchRequest

    req = httpx.Request('POST', 'https://api.anthropic.com/v1/messages/batches')
    scripted.responses.append(httpx.Response(503, request=req))
    scripted.add_post({
        'id': 'batch_ok', 'processing_status': 'in_progress',
        'request_counts': {},
    })
    provider = AnthropicProvider(model='m', api_key='test', retry_delay=0)
    submission = await AnthropicBatchStrategy(provider).submit_batch([
        BatchRequest(custom_id='c', system_prompt='S', user_prompt='U'),
    ])
    assert submission.batch_id == 'batch_ok'
    assert len(scripted.calls) == 2  # retried once after the 503


async def test_batch_retries_on_network_error_then_succeeds(scripted):
    """A transient httpx network error is retried, not surfaced."""
    import httpx

    from docgen.llm.anthropic import AnthropicProvider, BatchRequest

    scripted.responses.append(httpx.ConnectError('boom'))
    scripted.add_post({
        'id': 'batch_ok', 'processing_status': 'in_progress',
        'request_counts': {},
    })
    provider = AnthropicProvider(model='m', api_key='test', retry_delay=0)
    submission = await AnthropicBatchStrategy(provider).submit_batch([
        BatchRequest(custom_id='c', system_prompt='S', user_prompt='U'),
    ])
    assert submission.batch_id == 'batch_ok'
    assert len(scripted.calls) == 2


async def test_batch_429_quota_raises_without_retry(scripted):
    """A 429 whose body names a quota cap → QuotaExhaustedError, no retry
    (retrying a hard quota just burns attempts; the run must abort)."""
    import httpx

    from docgen.llm.anthropic import (
        AnthropicProvider,
        BatchRequest,
        QuotaExhaustedError,
    )

    req = httpx.Request('POST', 'https://api.anthropic.com/v1/messages/batches')
    scripted.responses.append(httpx.Response(
        429, request=req, json={'error': {'message': 'monthly spend limit'}},
    ))
    provider = AnthropicProvider(model='m', api_key='test', retry_delay=0)
    with pytest.raises(QuotaExhaustedError):
        await AnthropicBatchStrategy(provider).submit_batch([
            BatchRequest(custom_id='c', system_prompt='S', user_prompt='U'),
        ])
    assert len(scripted.calls) == 1  # quota is fatal — no retry


async def test_batch_surfaces_error_after_exhausting_retries(scripted):
    """Every attempt 5xx → the last error is surfaced so the caller's resume
    path (pending_batches) can take over."""
    import httpx

    from docgen.llm.anthropic import AnthropicProvider, BatchRequest

    req = httpx.Request('POST', 'https://api.anthropic.com/v1/messages/batches')
    for _ in range(6):  # _BATCH_MAX_RETRIES
        scripted.responses.append(httpx.Response(503, request=req))
    provider = AnthropicProvider(model='m', api_key='test', retry_delay=0)
    with pytest.raises(httpx.HTTPStatusError):
        await AnthropicBatchStrategy(provider).submit_batch([
            BatchRequest(custom_id='c', system_prompt='S', user_prompt='U'),
        ])
    assert len(scripted.calls) == 6  # exhausted all attempts
