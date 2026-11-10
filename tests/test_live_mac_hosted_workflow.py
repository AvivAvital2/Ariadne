"""Offline acceptance of the bounded hosted workflow, never model qualification."""

import asyncio
import hashlib
import json
import sqlite3

import httpx
import pytest

from tests import live_mac_hosted_workflow as probe
from tests.live_model_smoke import Budget

PRIVATE = 'PRIVATE_REASONING_WORKFLOW_SENTINEL'
CREDENTIAL = 'test-workflow-credential'
DOCUMENT = (
    '# Arithmetic mean\n\n## Overview\n\n'
    'The mean function accepts values and returns their arithmetic average. '
    'A false-valued input raises ValueError with the message empty input.\n\n'
    '## How It Works\n\n'
    'The source tests values before calculating sum(values) divided by len(values). '
    'It has no assignments to the supplied collection.\n\n'
    '## Question\n\nWhat happens for an empty list?\n\n## Answer\n\n'
    'An empty list raises ValueError before sum or len is evaluated.\n'
)


def model_response(request, *, bad_vector=False, truncated=False):
    payload = json.loads(request.content)
    if request.url.path.endswith('/embeddings'):
        assert payload['model'] == 'Qwen/Qwen3-Embedding-8B'
        assert 'dimensions' not in payload
        width = 4095 if bad_vector else 4096
        vector = [1.0] + [0.0] * (width - 1)
        return httpx.Response(200, json={
            'data': [{'index': index, 'embedding': vector}
                     for index, _ in enumerate(payload['input'])][::-1],
            'usage': {'prompt_tokens': 20, 'total_tokens': 20},
        })
    text = payload['messages'][-1]['content']
    final = ('Declares mean(values); its implementation body is not supplied.'
             if 'Description:' in text else DOCUMENT)
    return httpx.Response(200, json={
        'model': 'Qwen/Qwen3.8-27B', 'debug': PRIVATE,
        'choices': [{'index': 0, 'finish_reason': 'length' if truncated else 'stop',
                     'message': {'role': 'assistant', 'content': final,
                                 'reasoning_content': PRIVATE}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 400,
                  'completion_tokens_details': {'reasoning_tokens': 200}},
    })


def assert_private_data_absent(output):
    for path in output.rglob('*'):
        if path.is_file() and 'provenance' not in path.parts:
            raw = path.read_bytes()
            assert PRIVATE.encode() not in raw, path
            assert CREDENTIAL.encode() not in raw, path


async def test_real_workflow_with_frozen_protocol_and_hosted_identity(tmp_path):
    output, requests, frozen = tmp_path / 'run', [], []

    def respond(request):
        protocol_bytes = (output / 'protocol.json').read_bytes()
        frozen.append(protocol_bytes)
        protocol = json.loads(protocol_bytes)
        assert protocol['maximum_http_attempts'] == 60
        assert protocol['budget_usd'] == 1
        assert protocol['library_provider'] == 'deepinfra'
        assert protocol['max_total_output_tokens'] == 8192
        assert protocol['context_window'] == 32768
        assert protocol['local_q4_qualified'] is protocol['hardware_qualified'] is False
        assert protocol['quality_verdict'] == 'unreviewed'
        assert 'historical 60s' in protocol['timeout_reporting']
        assert 'adapter_calls' in protocol['timeout_reporting']
        for name, digest in protocol['source_hashes'].items():
            assert hashlib.sha256((output / 'provenance' / name).read_bytes()).hexdigest() == digest
        assert hashlib.sha256((output / 'sources' / 'average.py').read_bytes()).hexdigest() == protocol['fixture_sha256']
        assert request.headers['Authorization'] == 'Bearer ' + CREDENTIAL
        assert request.url.host == 'api.deepinfra.com'
        payload = json.loads(request.content)
        if request.url.path.endswith('/chat/completions'):
            actual_call = json.loads((output / 'report.json').read_text())['adapter_calls'][-1]
            assert actual_call['max_tokens'] == payload['max_tokens']
            assert actual_call['effective_timeout_seconds'] == 600
            assert payload['model'] == 'Qwen/Qwen3.8-27B'
            assert payload['reasoning_effort'] == 'xhigh'
            assert payload['chat_template_kwargs'] == {'enable_thinking': True}
            assert {key: payload[key] for key in (
                'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty',
            )} == {'temperature': 1, 'top_p': .95, 'top_k': 20, 'min_p': 0,
                   'presence_penalty': 0, 'repetition_penalty': 1}
            assert 'reasoning_format' not in payload and 'repeat_penalty' not in payload
            assert 0 < payload['max_tokens'] <= 8192 and payload['stream'] is False
            assert set(request.extensions['timeout'].values()) == {600}
        requests.append(payload)
        return model_response(request)

    report = await probe.execute(output, Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(respond), pacing=0)
    assert report['status'] == 'passed_orchestration_pending_quality_review', report
    assert report['quality_verdict'] == 'unreviewed'
    assert report['transport_mode'] == 'simulated'
    assert len(report['documents']) == 2
    assert all(check['passed'] for check in report['checks'])
    assert {check['name'] for check in report['checks']} >= {
        'hosted_identity', 'catalog_sync', 'catalog_describe', 'catalog_idempotence',
        'document_idempotence', 'search_returns_persisted_evidence',
        'ask_returns_answer_and_sources', 'atomic_rollback', 'reopen_preserves_content_and_vectors',
    }
    assert all(value == frozen[0] for value in frozen)
    assert (output / 'protocol.json').read_bytes() == frozen[0]
    assert len(requests) == len(report['requests']) < 60
    assert {row['kind'] for row in report['requests']} == {'chat', 'embeddings'}
    config = json.loads((output / 'config.json').read_text())
    assert config['llm']['provider'] == config['embeddings']['provider'] == 'deepinfra'
    assert 'local_runtime' not in config
    assert (output / 'answer.json').exists() and (output / 'ask-trace.json').exists()
    assert any(row['response_archive']['reasoning_present'] for row in report['requests']
               if row['kind'] == 'chat')
    assert all(row['status'] == 'completed' for row in report['adapter_calls'])
    assert all(row['http_attempts'] for row in report['adapter_calls'])
    assert_private_data_absent(output)


async def test_bad_vector_fails_before_catalog_documents_are_persisted(tmp_path):
    output = tmp_path / 'run'
    report = await probe.execute(output, Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(
                                     lambda request: model_response(request, bad_vector=True)), pacing=0)
    assert report['status'] == 'failed'
    assert len(report['requests']) == 1
    with sqlite3.connect(output / 'library.db') as connection:
        assert connection.execute('SELECT count(*) FROM documents').fetchone()[0] == 0
    assert_private_data_absent(output)


async def test_budget_rejects_before_network_and_cap_is_enforced(tmp_path):
    requests = []
    report = await probe.execute(tmp_path / 'run', Budget(.000000001, max_requests=999),
                                 api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(
                                     requests.append), pacing=0)
    assert report['status'] == 'failed' and requests == []
    protocol = json.loads((tmp_path / 'run' / 'protocol.json').read_text())
    assert protocol['maximum_http_attempts'] == 60
    with pytest.raises(ValueError, match='At most'):
        await probe.execute(tmp_path / 'too-much', Budget(1.01), api_key=CREDENTIAL)
    assert not (tmp_path / 'too-much').exists()


async def test_terminal_embedding_failure_has_no_hidden_production_retry(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(503, json={'error': CREDENTIAL})

    report = await probe.execute(tmp_path / 'run', Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(respond), pacing=0)
    assert report['status'] == 'failed'
    assert len(calls) == len(report['requests']) == 1
    assert_private_data_absent(tmp_path / 'run')


async def test_truncated_chat_fails_without_content_retry_or_reasoning_archive(tmp_path):
    report = await probe.execute(tmp_path / 'run', Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(
                                     lambda request: model_response(request, truncated=True)), pacing=0)
    assert report['status'] == 'failed'
    assert len([row for row in report['requests'] if row['kind'] == 'chat']) == 1
    assert_private_data_absent(tmp_path / 'run')


async def test_cancellation_preserved_without_retry(tmp_path):
    calls = []

    def cancel(request):
        calls.append(request)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await probe.execute(tmp_path / 'run', Budget(1), api_key=CREDENTIAL,
                            transport_factory=lambda: httpx.MockTransport(cancel), pacing=0)
    report = json.loads((tmp_path / 'run' / 'report.json').read_text())
    assert report['status'] == 'cancelled' and len(calls) == 1


async def test_explicit_adapter_timeout_and_output_context_validation(tmp_path):
    output = tmp_path / 'run'
    output.mkdir()
    from tests.live_model_smoke import Run
    run = Run(probe.PROFILE, output, Budget(1), pacing=0)
    calls = []

    def respond(request):
        calls.append(request)
        assert set(request.extensions['timeout'].values()) == {17}
        return model_response(request)

    provider = probe.HostedThinkingProvider(run, CREDENTIAL, lambda: httpx.MockTransport(respond))
    try:
        assert await provider.call('System', 'Source', max_tokens=4096, timeout=17)
        assert run.report['adapter_calls'][0] == {
            'max_tokens': 4096, 'effective_timeout_seconds': 17,
            'http_attempts': [1], 'status': 'completed',
        }
        with pytest.raises(probe.WorkflowAbort):
            await provider.call('System', 'x' * 32768, max_tokens=8192)
        with pytest.raises(probe.WorkflowAbort):
            await provider.call('System', 'Source', max_tokens=8193)
        assert len(calls) == 1
    finally:
        await provider.aclose()


@pytest.mark.parametrize('invalid', ['model', 'missing_usage', 'context_usage'])
async def test_response_identity_and_usage_are_validated_before_storage(tmp_path, invalid):
    def respond(request):
        response = model_response(request)
        if request.url.path.endswith('/chat/completions'):
            body = response.json()
            if invalid == 'model':
                body['model'] = 'another-model'
            elif invalid == 'missing_usage':
                body['usage'] = {}
            else:
                body['usage']['prompt_tokens'] = 32768
            return httpx.Response(200, json=body)
        return response

    report = await probe.execute(tmp_path / 'run', Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(respond), pacing=0)
    assert report['status'] == 'failed'
    assert len([row for row in report['requests'] if row['kind'] == 'chat']) == 1
    assert report['documents'] == []
    assert_private_data_absent(tmp_path / 'run')


async def test_changed_frozen_source_aborts_before_next_paid_request(tmp_path):
    output, calls = tmp_path / 'run', []

    def respond(request):
        calls.append(request)
        (output / 'sources' / 'average.py').write_text('changed after first request')
        return model_response(request)

    report = await probe.execute(output, Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(respond), pacing=0)
    assert report['status'] == 'failed'
    assert len(calls) == len(report['requests']) == 1


async def test_only_recorded_429_retry_is_allowed_and_charged(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            return httpx.Response(429, json={'error': 'Rate limited'}, headers={'retry-after': '0'})
        return model_response(request)

    report = await probe.execute(tmp_path / 'run', Budget(1), api_key=CREDENTIAL,
                                 transport_factory=lambda: httpx.MockTransport(respond), pacing=0)
    assert report['status'] == 'passed_orchestration_pending_quality_review'
    first, second = report['requests'][:2]
    assert first['status'] == 429 and second['status'] == 200
    assert first['attempt'] == 1 and second['attempt'] == 2
    assert 'accounted_usd' not in first
    assert len(calls) == len(report['requests'])
