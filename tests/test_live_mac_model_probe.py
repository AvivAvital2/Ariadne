"""Remote candidate checks use frozen prompts and the existing spending boundary."""

import asyncio
import json

import httpx
import numpy as np
import pytest

from tests.live_mac_model_probe import REMOTE_PROFILE, execute, frozen_requests
from tests.live_model_smoke import Budget


def response(content='## Question\nWhat happens?\n\n## Answer\nA documented result.', **changes):
    return httpx.Response(200, json={
        'model': REMOTE_PROFILE.model,
        'choices': [{'finish_reason': 'stop', 'message': {'content': content, **changes}}],
        'usage': {'prompt_tokens': 20, 'completion_tokens': 100},
    })


def embedding_response(request, *, width=4096):
    from model_config import QUERY_INSTRUCTION

    payload = json.loads(request.content)
    assert payload['model'] == 'Qwen/Qwen3-Embedding-8B'
    assert payload['encoding_format'] == 'float' and 'dimensions' not in payload
    if len(payload['input']) == 1:
        assert payload['input'][0].startswith(f'Instruct: {QUERY_INSTRUCTION}\nQuery:')
    rows = []
    for i in range(len(payload['input'])):
        vector = [0.0] * width
        vector[0], vector[-1] = 3.0, 4.0
        rows.append({'index': i, 'embedding': vector})
    return httpx.Response(200, json={'data': rows[::-1], 'usage': {'prompt_tokens': 20}})


def test_frozen_prompts_target_exact_hosted_candidate_without_oracle_answers():
    from tests.live_model_smoke import PROFILES
    from tests.source_verification_cases import CASES_BY_NAME

    requests = frozen_requests()
    assert list(requests) == ['cache', 'guarded_cache', 'token_expiry']
    for name, payload in requests.items():
        assert payload['model'] == 'Qwen/Qwen3.5-9B'
        assert payload['max_tokens'] == 8192
        assert payload['reasoning_effort'] == 'none' and payload['stream'] is False
        assert payload['temperature'] == 0.7 and payload['top_p'] == 0.8
        assert payload['top_k'] == 20 and payload['min_p'] == 0
        assert payload['presence_penalty'] == 1.5 and payload['repetition_penalty'] == 1
        assert CASES_BY_NAME[name].source.strip() in payload['messages'][1]['content']
        assert CASES_BY_NAME[name].correct not in json.dumps(payload)
    assert REMOTE_PROFILE.model != PROFILES['qwen'].model
    assert REMOTE_PROFILE.embedding_model == PROFILES['qwen'].embedding_model


async def test_generation_records_protocol_before_calls_and_awaits_factual_review(tmp_path):
    output = tmp_path / 'run'
    calls = []

    def serve(request):
        protocol = json.loads((output / 'protocol.json').read_text())
        assert len(protocol['cases']) == 3
        assert protocol['hosted_generation_precision'] == 'bfloat16'
        assert protocol['local_q4_qualified'] is False
        if request.url.path.endswith('/embeddings'):
            return embedding_response(request)
        assert request.url == 'https://api.deepinfra.com/v1/openai/chat/completions'
        assert request.headers['Authorization'] == 'Bearer test-secret'
        calls.append(json.loads(request.content))
        return response()

    report = await execute(output, Budget(1), api_key='test-secret',
                           transport_factory=lambda: httpx.MockTransport(serve), pacing=0)
    assert len(calls) == 6
    assert calls == [payload for payload in frozen_requests().values() for _ in range(2)]
    assert report['status'] == 'completed_pending_review'
    assert report['quality_verdict'] == 'unreviewed'
    assert len(list((output / 'documents').glob('*.md'))) == 6
    assert len(list((output / 'responses').glob('*.json'))) == 10
    assert report['embeddings']['document_vectors'] == 6
    assert report['embeddings']['query_vectors'] == 3
    assert report['embeddings']['stored_dimensions'] == 3072
    assert report['embeddings']['hosted_precision'] == 'undisclosed'
    with np.load(output / 'vectors.npz') as vectors:
        assert vectors['documents'].shape == (6, 3072)
        assert vectors['queries'].shape == (3, 3072)
        assert vectors['documents'].dtype == np.float32
        np.testing.assert_array_equal(vectors['documents'][:, 0], np.ones(6))
        assert not np.any(vectors['documents'][:, 1:])
    assert report['accounted_upper_bound_usd'] < 1
    assert not list(output.rglob('*.db'))


async def test_bad_remote_embedding_width_is_not_recorded_as_success(tmp_path):
    def serve(request):
        if request.url.path.endswith('/embeddings'):
            return embedding_response(request, width=3072)
        return response()

    output = tmp_path / 'run'
    report = await execute(output, Budget(1), api_key='test-secret',
                           transport_factory=lambda: httpx.MockTransport(serve), pacing=0)
    assert report['status'] == 'failed'
    assert '4096' in report['error']
    assert len(list((output / 'documents').glob('*.md'))) == 6
    assert not (output / 'vectors.npz').exists()


async def test_exhausted_budget_sends_nothing_and_preserves_failure(tmp_path):
    calls = []
    report = await execute(tmp_path / 'run', Budget(0.0000001), api_key='test-secret',
                           transport_factory=lambda: httpx.MockTransport(calls.append),
                           pacing=0)
    assert not calls
    assert report['status'] == 'failed'
    assert 'spending limit' in report['error']


async def test_reasoning_response_is_rejected_with_raw_evidence(tmp_path):
    report = await execute(tmp_path / 'run', Budget(1), api_key='test-secret',
                           transport_factory=lambda: httpx.MockTransport(
                               lambda _: response(reasoning_content='Unexpected thinking output')),
                           pacing=0)
    assert report['status'] == 'failed'
    assert 'reasoning' in report['error']
    assert (tmp_path / 'run/responses/001.json').exists()
    assert not list((tmp_path / 'run/documents').glob('*.md'))


async def test_provider_error_redacts_key_from_report_and_response(tmp_path):
    report = await execute(tmp_path / 'run', Budget(1), api_key='test-secret',
                           transport_factory=lambda: httpx.MockTransport(
                               lambda _: httpx.Response(401, json={'error': 'test-secret'})),
                           pacing=0)
    assert report['status'] == 'failed'
    for path in (tmp_path / 'run').rglob('*.json'):
        assert 'test-secret' not in path.read_text()


async def test_cancellation_records_terminal_status_and_retains_spending_reservation(tmp_path):
    calls = []
    budget = Budget(1)

    def cancel(request):
        calls.append(request)
        raise asyncio.CancelledError

    output = tmp_path / 'run'
    with pytest.raises(asyncio.CancelledError):
        await execute(output, budget, api_key='test-secret',
                      transport_factory=lambda: httpx.MockTransport(cancel), pacing=0)
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == 'cancelled'
    assert len(calls) == budget.requests == 1
    assert budget.committed == report['requests'][0]['reserved_usd'] > 0
    assert report['quality_verdict'] == 'unreviewed'
    assert not list((output / 'documents').glob('*.md'))
