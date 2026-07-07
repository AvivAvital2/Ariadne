"""Evolutionary-TDD walk for the provider-agnostic batch strategy.

Batch dispatch — submit → poll → fetch → cancel — is the same lifecycle for
Anthropic's Message Batches API and OpenAI's Batch API (both offer ~50% off
and a 24h completion window). This file pins the ``BatchStrategy`` seam: the
Anthropic batch lifecycle, historically a set of methods bolted onto
``AnthropicProvider``, now lives in ``AnthropicBatchStrategy`` and is selected
per-config alongside the OpenAI strategy.

The file grows one demand at a time.

Cycle 1 demand: ``AnthropicBatchStrategy`` conforms to the ``BatchStrategy``
protocol and performs the Anthropic lifecycle directly (the logic moved off
the provider), borrowing the provider's authenticated transport. The
provider keeps thin delegating shims so the orchestrator and existing call
sites stay green until Cycle 3 rewires them.
"""
from __future__ import annotations

import json
from typing import Any

import pytest


class _FakeResponse:
    def __init__(self, payload: Any, *, text: str | None = None):
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._payload


class _ScriptedClient:
    """Records requests; replays a queue of canned responses in order."""

    def __init__(self):
        self.calls: list[tuple[str, str, dict | None]] = []
        self.responses: list[_FakeResponse] = []
        self.headers: dict[str, str] = {}

    def add(self, response: Any, *, text: str | None = None) -> None:
        self.responses.append(_FakeResponse(response, text=text))

    async def post(self, url: str, json: dict | None = None):
        self.calls.append(('POST', url, json))
        return self.responses.pop(0)

    async def get(self, url: str):
        self.calls.append(('GET', url, None))
        return self.responses.pop(0)

    async def aclose(self) -> None:
        return None


def _provider_with(monkeypatch, client: _ScriptedClient):
    from docgen.llm.anthropic import AnthropicProvider
    monkeypatch.setattr(AnthropicProvider, '_get_client', lambda self: client)
    return AnthropicProvider(model='claude-opus-4-8', api_key='test')


class TestAnthropicBatchStrategy:
    # ---- C1 ----------------------------------------------------------
    async def test_strategy_conforms_and_runs_anthropic_lifecycle(
        self, monkeypatch,
    ) -> None:
        from docgen.llm.anthropic_batch import AnthropicBatchStrategy
        from docgen.llm.batch import (
            BatchRequest,
            BatchStrategy,
            BatchSubmission,
        )

        client = _ScriptedClient()
        provider = _provider_with(monkeypatch, client)
        strategy = AnthropicBatchStrategy(provider)

        # (a) The strategy satisfies the runtime-checkable protocol — the
        #     four lifecycle methods are present, so make_batch_strategy
        #     can return it behind the BatchStrategy type.
        assert isinstance(strategy, BatchStrategy)

        # (b) submit_batch POSTs each request to /messages/batches with the
        #     Anthropic params shape, and returns the assigned batch_id.
        client.add({
            'id': 'batch_xyz',
            'processing_status': 'in_progress',
            'request_counts': {'processing': 1},
        })
        submission = await strategy.submit_batch([
            BatchRequest(custom_id='d1', system_prompt='S', user_prompt='U'),
        ])
        assert isinstance(submission, BatchSubmission)
        assert submission.batch_id == 'batch_xyz'
        method, url, body = client.calls[0]
        assert (method, url) == ('POST', '/messages/batches')
        first = body['requests'][0]
        assert first['custom_id'] == 'd1'
        assert first['params']['model'] == 'claude-opus-4-8'
        assert first['params']['messages'] == [
            {'role': 'user', 'content': 'U'},
        ]

        # (c) fetch_batch_results parses the JSONL keyed by custom_id and
        #     records cache stats onto the borrowed provider transport, so
        #     batch cache-savings telemetry survives the extraction.
        client.add({}, text=json.dumps({
            'custom_id': 'd1',
            'result': {'type': 'succeeded', 'message': {
                'content': [{'type': 'text', 'text': 'hello'}],
                'usage': {
                    'input_tokens': 10, 'output_tokens': 5,
                    'cache_read_input_tokens': 8,
                },
            }},
        }))
        results = await strategy.fetch_batch_results('batch_xyz')
        assert results == {'d1': 'hello'}
        assert provider.cache_stats.total_calls == 1
        assert provider.cache_stats.cache_reads == 1


class _FakeOpenAIClient:
    """Fakes the OpenAI Batch API surface over httpx — files upload/content
    and batches create/retrieve/cancel — dispatching by (method, url)."""

    def __init__(self, *, polls: list[dict], file_contents: dict[str, str]):
        self.calls: list[tuple[str, str]] = []
        self.headers: dict[str, str] = {}
        self._polls = list(polls)
        self._file_contents = dict(file_contents)
        self.uploads: list[dict] = []
        self.created: list[dict] = []
        self.cancelled: list[str] = []

    async def post(self, url: str, *, json=None, data=None, files=None, **kw):
        self.calls.append(('POST', url))
        if url == '/files':
            self.uploads.append({'data': data, 'files': files})
            return _FakeResponse({'id': 'file_in_1'})
        if url == '/batches':
            self.created.append(json)
            return _FakeResponse({'id': 'batch_oai_1', 'status': 'validating'})
        if url.endswith('/cancel'):
            self.cancelled.append(url)
            return _FakeResponse({'id': 'batch_oai_1', 'status': 'cancelling'})
        raise AssertionError(f'unexpected POST {url}')

    async def get(self, url: str, **kw):
        self.calls.append(('GET', url))
        if url.startswith('/batches/'):
            # Advance through scripted polls; the last one repeats so a
            # later fetch's GET /batches/{id} still sees the terminal state.
            poll = self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]
            return _FakeResponse(poll)
        if url.startswith('/files/') and url.endswith('/content'):
            file_id = url.split('/')[2]
            return _FakeResponse({}, text=self._file_contents.get(file_id, ''))
        raise AssertionError(f'unexpected GET {url}')

    async def aclose(self) -> None:
        return None


def _openai_provider_with(monkeypatch, client: _FakeOpenAIClient):
    from docgen.llm.openai import OpenAIProvider
    monkeypatch.setattr(OpenAIProvider, '_get_client', lambda self: client)
    return OpenAIProvider(model='gpt-5.5', api_key='test')


class TestOpenAIBatchStrategy:
    # ---- C2 ----------------------------------------------------------
    async def test_strategy_conforms_and_runs_openai_lifecycle(
        self, monkeypatch,
    ) -> None:
        from docgen.llm.batch import (
            BatchRequest,
            BatchStatus,
            BatchStrategy,
            BatchSubmission,
        )
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        in_progress = {
            'id': 'batch_oai_1', 'status': 'in_progress',
            'request_counts': {'total': 2, 'completed': 0, 'failed': 0},
        }
        completed = {
            'id': 'batch_oai_1', 'status': 'completed',
            'request_counts': {'total': 2, 'completed': 1, 'failed': 1},
            'output_file_id': 'file_out_1', 'error_file_id': 'file_err_1',
        }
        out_jsonl = json.dumps({
            'custom_id': 'd1',
            'response': {'status_code': 200, 'body': {
                'choices': [{'message': {'content': 'hello'}}],
            }},
            'error': None,
        })
        err_jsonl = json.dumps({
            'custom_id': 'd2',
            'response': {'status_code': 400, 'body': {}},
            'error': {'message': 'bad request'},
        })

        client = _FakeOpenAIClient(
            polls=[in_progress, completed],
            file_contents={'file_out_1': out_jsonl, 'file_err_1': err_jsonl},
        )
        provider = _openai_provider_with(monkeypatch, client)
        strategy = OpenAIBatchStrategy(provider)

        # (a) Same protocol as the Anthropic strategy → interchangeable.
        assert isinstance(strategy, BatchStrategy)

        # (b) submit_batch uploads a JSONL file, then creates a 24h batch
        #     against /v1/chat/completions, returning the batch id.
        submission = await strategy.submit_batch([
            BatchRequest(custom_id='d1', system_prompt='S', user_prompt='U1'),
            BatchRequest(custom_id='d2', system_prompt='S', user_prompt='U2'),
        ])
        assert isinstance(submission, BatchSubmission)
        assert submission.batch_id == 'batch_oai_1'
        assert ('POST', '/files') in client.calls
        assert ('POST', '/batches') in client.calls
        created = client.created[0]
        assert created['input_file_id'] == 'file_in_1'
        assert created['endpoint'] == '/v1/chat/completions'
        assert created['completion_window'] == '24h'
        # The uploaded JSONL: one chat-completions request per BatchRequest.
        raw = client.uploads[0]['files']['file'][1]
        lines = [
            json.loads(line) for line in raw.decode().splitlines() if line.strip()
        ]
        assert {row['custom_id'] for row in lines} == {'d1', 'd2'}
        first = lines[0]
        assert first['method'] == 'POST'
        assert first['url'] == '/v1/chat/completions'
        body = first['body']
        assert body['model'] == 'gpt-5.5'
        assert body['messages'] == [
            {'role': 'system', 'content': 'S'},
            {'role': 'user', 'content': 'U1'},
        ]
        # gpt-5.x → max_completion_tokens, never temperature (mirrors
        # OpenAIProvider.call).
        assert 'max_completion_tokens' in body
        assert 'temperature' not in body

        # (c) poll_batch maps OpenAI statuses/counts onto BatchStatus and
        #     fires on_progress each poll; any terminal status → 'ended'.
        seen: list[tuple[int, int, int]] = []
        status = await strategy.poll_batch(
            'batch_oai_1', poll_interval=0,
            on_progress=lambda p, s, e: seen.append((p, s, e)),
        )
        assert isinstance(status, BatchStatus)
        assert status.processing_status == 'ended'
        assert status.succeeded == 1
        assert status.errored == 1
        assert seen == [(2, 0, 0), (0, 1, 1)]

        # (d) fetch_batch_results merges the output file (succeeded rows →
        #     text) and the error file (failed rows → None).
        results = await strategy.fetch_batch_results('batch_oai_1')
        assert results == {'d1': 'hello', 'd2': None}

    async def test_cancel_posts_to_cancel_endpoint(self, monkeypatch) -> None:
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        client = _FakeOpenAIClient(polls=[{'status': 'cancelling'}], file_contents={})
        provider = _openai_provider_with(monkeypatch, client)
        await OpenAIBatchStrategy(provider).cancel_batch('batch_oai_1')
        assert client.cancelled == ['/batches/batch_oai_1/cancel']

    async def test_poll_maps_all_terminal_statuses_to_ended(
        self, monkeypatch,
    ) -> None:
        """The lifecycle test only covers 'completed'; OpenAI also ends a batch
        on 'failed' / 'expired' / 'cancelled'. Each must terminate the poll and
        normalize to 'ended' (with the counts), or a failed/expired batch spins
        the poll loop forever."""
        from docgen.llm.batch import BatchStatus
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        for terminal in ('failed', 'expired', 'cancelled'):
            client = _FakeOpenAIClient(
                polls=[{
                    'id': 'b', 'status': terminal,
                    'request_counts': {'total': 3, 'completed': 1, 'failed': 2},
                }],
                file_contents={},
            )
            provider = _openai_provider_with(monkeypatch, client)
            status = await OpenAIBatchStrategy(provider).poll_batch(
                'b', poll_interval=0,
            )
            assert isinstance(status, BatchStatus)
            assert status.processing_status == 'ended', (
                f'{terminal} must be treated as terminal → ended'
            )
            assert (status.succeeded, status.errored, status.processing) == (
                1, 2, 0,
            )

    async def test_fetch_skips_bad_output_rows_without_raising(
        self, monkeypatch,
    ) -> None:
        """Output-file rows that are non-200, choice-less, malformed JSON, or
        blank degrade to None / are skipped — fetch never raises on a bad row
        (covers _extract_text's non-200/empty branches + the JSONL skip)."""
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        good = json.dumps({'custom_id': 'g', 'response': {
            'status_code': 200,
            'body': {'choices': [{'message': {'content': 'hi'}}]},
        }})
        non_200 = json.dumps({'custom_id': 'e', 'response': {
            'status_code': 500, 'body': {},
        }})
        no_choices = json.dumps({'custom_id': 'n', 'response': {
            'status_code': 200, 'body': {'choices': []},
        }})
        out = '\n'.join([good, non_200, no_choices, '   ', 'not-json', ''])
        completed = {
            'id': 'b', 'status': 'completed',
            'request_counts': {'total': 3, 'completed': 1, 'failed': 2},
            'output_file_id': 'fout',  # no error_file_id this run
        }
        client = _FakeOpenAIClient(
            polls=[completed], file_contents={'fout': out},
        )
        provider = _openai_provider_with(monkeypatch, client)
        results = await OpenAIBatchStrategy(provider).fetch_batch_results('b')
        assert results == {'g': 'hi', 'e': None, 'n': None}

    async def test_submit_retries_transient_5xx_then_succeeds(
        self, monkeypatch,
    ) -> None:
        """A 5xx at submit (file upload) is retried, not surfaced — a
        transient gateway blip must not abort the whole batch. Mirrors the
        Anthropic strategy's batch retry budget."""
        import httpx

        from docgen.llm.batch import BatchRequest
        from docgen.llm.openai import OpenAIProvider
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        req = httpx.Request('POST', 'https://api.openai.com/v1/files')
        client = _FlakyOpenAIClient([
            httpx.Response(503, request=req),      # /files attempt 1 → 503
            _FakeResponse({'id': 'file_in_1'}),    # /files retry → ok
            _FakeResponse({'id': 'batch_oai_1'}),  # /batches → ok
        ])
        monkeypatch.setattr(OpenAIProvider, '_get_client', lambda self: client)
        provider = OpenAIProvider(model='gpt-5.5', api_key='test', retry_delay=0)

        submission = await OpenAIBatchStrategy(provider).submit_batch([
            BatchRequest(custom_id='d1', system_prompt='S', user_prompt='U'),
        ])
        assert submission.batch_id == 'batch_oai_1'
        assert client.calls.count(('POST', '/files')) == 2  # retried once

    async def test_submit_429_quota_aborts_without_retry(
        self, monkeypatch,
    ) -> None:
        """A 429 whose body names a hard quota → QuotaExhaustedError with no
        retry (retrying a spend cap just burns attempts)."""
        import httpx

        from docgen.llm.anthropic import QuotaExhaustedError
        from docgen.llm.batch import BatchRequest
        from docgen.llm.openai import OpenAIProvider
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        req = httpx.Request('POST', 'https://api.openai.com/v1/files')
        client = _FlakyOpenAIClient([
            httpx.Response(
                429, request=req,
                json={'error': {'message': 'monthly spend limit reached'}},
            ),
        ])
        monkeypatch.setattr(OpenAIProvider, '_get_client', lambda self: client)
        provider = OpenAIProvider(model='gpt-5.5', api_key='test', retry_delay=0)

        with pytest.raises(QuotaExhaustedError):
            await OpenAIBatchStrategy(provider).submit_batch([
                BatchRequest(custom_id='d1', system_prompt='S', user_prompt='U'),
            ])
        assert client.calls.count(('POST', '/files')) == 1  # quota is fatal


class _FlakyOpenAIClient:
    """Replays a queue of responses/exceptions in order, recording calls — so
    a retry test can script a transient failure followed by success. A queued
    Exception is raised (network error); a real httpx.Response with a 4xx/5xx
    status raises via raise_for_status()."""

    def __init__(self, queue: list) -> None:
        self._queue = list(queue)
        self.calls: list[tuple[str, str]] = []
        self.headers: dict[str, str] = {}

    async def post(self, url: str, **kw):
        self.calls.append(('POST', url))
        return self._next()

    async def get(self, url: str, **kw):
        self.calls.append(('GET', url))
        return self._next()

    def _next(self):
        item = self._queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    async def aclose(self) -> None:
        return None


class TestMakeBatchStrategy:
    # ---- C2 selector -------------------------------------------------
    def test_selects_strategy_by_resolved_provider(self) -> None:
        from docgen.llm.anthropic_batch import AnthropicBatchStrategy
        from docgen.llm.batch import BatchStrategy
        from docgen.llm.factory import make_batch_strategy
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        oai = make_batch_strategy('openai', model='gpt-5.5', api_key='k')
        assert isinstance(oai, OpenAIBatchStrategy)
        assert isinstance(oai, BatchStrategy)

        ant = make_batch_strategy(
            'anthropic', model='claude-opus-4-8', api_key='k',
        )
        assert isinstance(ant, AnthropicBatchStrategy)
        assert isinstance(ant, BatchStrategy)

    def test_unknown_provider_raises(self) -> None:
        from docgen.llm.factory import make_batch_strategy

        with pytest.raises(ValueError):
            make_batch_strategy('gemini', model='x', api_key='k')

    def test_batch_strategy_for_wraps_existing_provider_by_type(self) -> None:
        """``batch_strategy_for`` wraps an already-constructed provider in its
        batch strategy (reusing its transport) — the entry point the
        orchestrator uses since it already holds a provider instance."""
        from docgen.llm.anthropic import AnthropicProvider
        from docgen.llm.anthropic_batch import AnthropicBatchStrategy
        from docgen.llm.batch import BatchStrategy
        from docgen.llm.factory import batch_strategy_for
        from docgen.llm.openai import OpenAIProvider
        from docgen.llm.openai_batch import OpenAIBatchStrategy

        oai = batch_strategy_for(OpenAIProvider(model='gpt-5.5', api_key='k'))
        assert isinstance(oai, OpenAIBatchStrategy)
        assert isinstance(oai, BatchStrategy)
        assert oai._provider.model == 'gpt-5.5'  # reuses the same provider

        ant = batch_strategy_for(
            AnthropicProvider(model='claude-opus-4-8', api_key='k'),
        )
        assert isinstance(ant, AnthropicBatchStrategy)

    def test_batch_strategy_for_unknown_type_raises(self) -> None:
        from docgen.llm.factory import batch_strategy_for

        with pytest.raises(ValueError):
            batch_strategy_for(object())


async def test_file_upload_reaches_the_wire_as_multipart(monkeypatch):
    """Regression twin of the embeddings-strategy test: OpenAIProvider's
    client pinned 'Content-Type: application/json' at client level, which
    would turn the chat-batch /files upload into a 415 exactly like the
    embeddings path did on its first live run."""
    import httpx

    from docgen.llm.batch import BatchRequest
    from docgen.llm.openai import OpenAIProvider
    from docgen.llm.openai_batch import OpenAIBatchStrategy

    captured = {}

    def handler(request):
        if request.url.path.endswith('/files'):
            captured['content_type'] = request.headers.get('content-type', '')
            captured['body'] = request.read()
            return httpx.Response(200, json={'id': 'file_in_1'})
        if request.url.path.endswith('/batches'):
            return httpx.Response(
                200, json={'id': 'batch_1', 'status': 'validating'})
        raise AssertionError(f'unexpected {request.url.path}')

    real_async_client = httpx.AsyncClient

    def with_mock_transport(**kwargs):
        return real_async_client(
            transport=httpx.MockTransport(handler), **kwargs)

    monkeypatch.setattr(httpx, 'AsyncClient', with_mock_transport)
    provider = OpenAIProvider(model='gpt-test', api_key='test-key')
    try:
        strategy = OpenAIBatchStrategy(provider)
        await strategy.submit_batch([
            BatchRequest(custom_id='r0', system_prompt='s', user_prompt='u'),
        ])
    finally:
        await provider.aclose()

    assert captured['content_type'].startswith('multipart/form-data')
    assert b'purpose' in captured['body']
    assert b'batch_input.jsonl' in captured['body']
