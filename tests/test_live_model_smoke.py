"""Run the exact opt-in workflows against deterministic HTTP for every profile."""

import json
import sqlite3

import httpx
import pytest

from tests.live_model_smoke import PROFILES, Budget, RecordedTransport, Run, execute


def fixture_document(name, updated=False):
    facts = {
        'average': 'The mean function computes the arithmetic average with sum and len. '
        + ('Empty input returns 0.0.' if updated else 'Empty input raises ValueError.'),
        'password': 'The consume_reset_token function pops a password reset token before checking expiration. Missing tokens raise KeyError. Expired tokens raise ValueError and remain consumed.',
        'cache': 'The cache_put function moves writes to the end of an OrderedDict. It evicts with popitem(last=False). Negative capacity raises KeyError after the cache empties. Reads do not update recency.',
    }
    return (
        f'# {name}\n\n## Overview\n\n{facts[name]}\n\n## How It Works\n\n'
        f'{facts[name] * 8}\n\n## Design\n\nCallers provide the mutable state.\n\n'
        '## Question\n\nWhat happens at the boundary?\n\n## Answer\n\n'
        f'{facts[name]}\n\n```dot\ndigraph G {{ A [label="Input"]; B [label="Result"]; A -> B; }}\n```\n'
    )


def model_response(request, *, malformed=False):
    payload = json.loads(request.content)
    model = payload['model']
    if request.url.path.endswith('/embeddings'):
        qwen = model.startswith('Qwen/')
        assert ('dimensions' not in payload) if qwen else payload['dimensions'] == 3072
        width = (4096 if qwen else 3072) - int(malformed)
        vectors = []
        for i, input_text in enumerate(payload['input']):
            text = input_text.lower()
            axis = 3
            if any(word in text for word in ('password', 'reset_token', 'recovery', 'expired')):
                axis = 1
            elif any(word in text for word in ('cache', 'evict', 'lru', 'capacity')):
                axis = 2
            elif any(word in text for word in ('mean', 'arithmetic', 'average')):
                axis = 0
            vector = [0.0] * width
            vector[axis] = 1.0
            if axis == 2 and 'negative capacity raises keyerror' in text:
                vector[5] = 0.1  # Rank the short diagram first for cache queries.
            if '0.0' in text:
                vector[4] = 0.1
            vectors.append({'index': i, 'embedding': vector})
        return httpx.Response(
            200, json={'data': vectors[::-1], 'usage': {'prompt_tokens': 20, 'total_tokens': 20}}
        )
    text = json.dumps(payload)
    if 'What exception does cache_put' in text:
        content = 'cache_put raises KeyError once negative capacity has emptied the cache.'
    else:
        name = (
            'average'
            if 'def mean' in text
            else ('password' if 'def consume_reset_token' in text else 'cache')
        )
        content = fixture_document(name, 'return 0.0' in text)
        # A valid diagram has no section headings: MCP returns full content.
        if name == 'cache' and 'technical diagram specialist' in text:
            content = '```dot\ndigraph G { A [label="cache_put"]; B [label="Evict least recently written cache entry"]; A -> B; }\n```'
    if model == 'claude-opus-5-5':
        assert request.url.host == 'api.anthropic.com'
        assert request.headers['x-api-key'] == 'test-anthropic'
        assert payload['max_tokens'] in (1024, 8192)
        return httpx.Response(
            200,
            json={
                'content': [{'type': 'text', 'text': content}],
                'stop_reason': 'end_turn',
                'usage': {'input_tokens': 20, 'output_tokens': 100},
            },
        )
    if model == 'gpt-6-astra':
        assert 'max_completion_tokens' in payload and 'max_tokens' not in payload
        assert request.url.host == 'api.openai.com'
    else:
        assert payload['reasoning_effort'] == 'none'
        assert request.url.host == 'api.deepinfra.com'
    return httpx.Response(
        200,
        json={
            'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 100, 'total_tokens': 120},
        },
    )


@pytest.fixture(autouse=True)
def credentials(monkeypatch):
    for provider in ('deepinfra', 'openai', 'anthropic'):
        monkeypatch.setenv(provider.upper() + '_API_KEY', 'test-' + provider)


@pytest.mark.parametrize('profile', PROFILES.values(), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_same_workflows_for_every_profile(tmp_path, profile):
    report = await execute(
        profile,
        tmp_path / profile.name,
        Budget(9),
        transport_factory=lambda: httpx.MockTransport(model_response),
        pacing=0,
    )
    assert report['status'] == 'passed', report
    assert report['transport_mode'] == 'simulated'
    assert len(report['documents']) == 7
    assert all(check['passed'] for check in report['checks'])
    selection = next(c for c in report['checks'] if c['name'] == 'mcp_sections/cache')
    assert selection['selected_id'] == 'cache--diagram'
    assert selection['full_document_fallback']
    assert {check['name'] for check in report['checks']} >= {
        'mcp_ask_grounded_fact',
        'source_edit_replacement',
        'atomic_rollback',
        'idempotent_replay',
    }
    assert {r['model'] for r in report['requests']} == {profile.model, profile.embedding_model}
    for path in (tmp_path / profile.name).rglob('*.json'):
        assert 'test-openai' not in path.read_text()
        assert 'test-anthropic' not in path.read_text()
        assert 'test-deepinfra' not in path.read_text()


@pytest.mark.parametrize('profile', PROFILES.values(), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_bad_vectors_fail_before_document_write(tmp_path, profile):
    report = await execute(
        profile,
        tmp_path / profile.name,
        Budget(9),
        transport_factory=lambda: httpx.MockTransport(lambda r: model_response(r, malformed=True)),
        pacing=0,
    )
    assert report['status'] == 'failed'
    with sqlite3.connect(tmp_path / profile.name / 'library.db') as conn:
        assert conn.execute('SELECT count(*) FROM documents').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_shared_budget_stops_before_network(tmp_path):
    calls = []

    def respond(request):
        calls.append(request)
        return model_response(request)

    budget = Budget(0.000001)
    for profile in PROFILES.values():
        report = await execute(
            profile,
            tmp_path / profile.name,
            budget,
            transport_factory=lambda: httpx.MockTransport(respond),
            pacing=0,
        )
        assert report['status'] == 'failed'
    assert not calls
    assert budget.committed == 0


@pytest.mark.asyncio
async def test_unknown_usage_keeps_reservation_and_redacts_error(tmp_path):
    profile = PROFILES['openai']
    run = Run(profile, tmp_path, Budget(2))
    transport = RecordedTransport(
        run,
        'chat',
        httpx.MockTransport(lambda _: httpx.Response(401, json={'error': 'test-openai'})),
    )
    async with httpx.AsyncClient(
        transport=transport, headers={'Authorization': 'Bearer test-openai'}
    ) as client:
        await client.post(
            profile.base_url + '/chat/completions',
            json={
                'model': profile.model,
                'messages': [{'role': 'user', 'content': 'hello'}],
                'max_completion_tokens': 1024,
            },
        )
    record = run.report['requests'][0]
    assert run.budget.committed == record['reserved_usd']
    assert 'test-openai' not in (tmp_path / 'responses' / '001.json').read_text()


@pytest.mark.parametrize('value', [0, -1, float('nan'), float('inf')])
def test_invalid_budget(value):
    with pytest.raises(ValueError, match='positive finite'):
        Budget(value)


@pytest.mark.asyncio
async def test_qwen_429_attempts_each_reserve_budget(tmp_path):
    profile = PROFILES['qwen']
    run = Run(profile, tmp_path, Budget(1), pacing=0)
    responses = [429, 429, 200]

    def respond(request):
        status = responses.pop(0)
        if status != 200:
            return httpx.Response(status, json={'error': 'busy'}, headers={'retry-after': '0'})
        return model_response(request)

    async with httpx.AsyncClient(
        transport=RecordedTransport(run, 'embeddings', httpx.MockTransport(respond))
    ) as client:
        response = await client.post(
            profile.embedding_url + '/embeddings',
            json={
                'model': profile.embedding_model,
                'input': ['average'],
                'encoding_format': 'float',
            },
        )
    assert response.status_code == 200
    assert [r['status'] for r in run.report['requests']] == [429, 429, 200]
    assert run.budget.committed == pytest.approx(
        sum(r.get('accounted_usd', r['reserved_usd']) for r in run.report['requests'])
    )


@pytest.mark.parametrize('profile', PROFILES.values(), ids=lambda p: p.name)
@pytest.mark.asyncio
async def test_truncated_generation_is_not_stored(tmp_path, profile):
    def respond(request):
        response = model_response(request)
        data = response.json()
        if profile.provider == 'anthropic':
            data['stop_reason'] = 'max_tokens'
        else:
            data['choices'][0]['finish_reason'] = 'length'
        return httpx.Response(200, json=data)

    report = await execute(
        profile,
        tmp_path / profile.name,
        Budget(2),
        transport_factory=lambda: httpx.MockTransport(respond),
        pacing=0,
    )
    assert report['status'] == 'failed'
    assert not report['documents']
    assert (tmp_path / profile.name / 'responses' / '001.json').exists()


@pytest.mark.asyncio
async def test_unexpected_endpoint_is_rejected_before_sending(tmp_path):
    run = Run(PROFILES['openai'], tmp_path, Budget(1))
    calls = []
    transport = RecordedTransport(run, 'chat', httpx.MockTransport(calls.append))
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(ValueError, match='Unexpected live route'):
            await client.post('https://wrong-endpoint.invalid/chat/completions', json={})
    assert not calls
    assert run.budget.requests == 0
