"""Offline guards for the opt-in hosted 27B thinking diagnostic."""

import asyncio
import copy
import hashlib
import json

import httpx
import pytest

from docgen.generator import DocGenerator
from tests import live_mac_accuracy_probe as probe
from tests.live_model_smoke import Budget
from tests.source_verification_cases import CASES_BY_NAME

PRIVATE = 'PRIVATE_REASONING_MUST_NEVER_REACH_DISK'
FINAL = '## Overview\nThe final source-grounded document.\n\n## Question\nWhat happens?\n\n## Answer\nA result.'


def body():
    return {
        'id': 'test-completion', 'model': 'Qwen/Qwen3.8-27B',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': FINAL, 'reasoning_content': PRIVATE,
            'reasoning_details': [{'text': PRIVATE}],
        }}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 400, 'total_tokens': 500,
                  'completion_tokens_details': {'reasoning_tokens': 200}, 'estimated_cost': 0.001},
        'debug': PRIVATE,
    }


def assert_no_private_text(output):
    for path in output.rglob('*'):
        if path.is_file() and 'provenance' not in path.parts:
            assert PRIVATE not in path.read_text()
            assert 'test-secret' not in path.read_text()


async def test_exact_protocol_and_sources_frozen_before_twelve_calls(tmp_path):
    output, calls, frozen = tmp_path / 'run', [], []
    wire = json.dumps(body()).encode()

    def serve(request):
        data = (output / 'protocol.json').read_bytes()
        frozen.append(data)
        protocol = json.loads(data)
        assert protocol['model'] == 'Qwen/Qwen3.8-27B'
        assert protocol['context_window'] == 32768
        assert protocol['max_total_output_tokens'] == 8192
        assert protocol['maximum_http_attempts'] == 18 and protocol['budget_usd'] == 1
        assert protocol['concurrency'] == 1 and protocol['repetitions'] == 2
        assert protocol['timeout_seconds'] == 600
        assert protocol['local_artifact_qualified'] is False
        assert protocol['review_policy']['production_qualification'] is False
        assert protocol['review_policy']['required'] == [
            'All twelve documents completed and independently reviewed against source',
            'All required source behavior covered correctly',
            'Zero material false claims or contradictions',
            'No loss of previously correct required behavior',
        ]
        assert 'not exact wire parity' in protocol['hosted_mapping']['limit']
        assert 'not exact tokenizer' in protocol['context_admission']
        assert 'final-only' in protocol['response_archive_policy']
        for name, digest in protocol['source_hashes'].items():
            assert hashlib.sha256((output / 'provenance' / name).read_bytes()).hexdigest() == digest
        assert tuple(protocol['cases']) == ('cache', 'guarded_cache', 'token_expiry')
        for name, case in protocol['cases'].items():
            assert (output / 'sources' / f'{name}.py').read_text() == CASES_BY_NAME[name].source
            assert case['oracle'] and case['required_behavior'] == CASES_BY_NAME[name].correct
        assert request.url == 'https://api.deepinfra.com/v1/openai/chat/completions'
        assert request.headers['Authorization'] == 'Bearer test-secret'
        assert set(request.extensions['timeout'].values()) == {600}
        payload = json.loads(request.content)
        assert payload['model'] == 'Qwen/Qwen3.8-27B'
        assert payload['reasoning_effort'] == 'xhigh'
        assert payload['chat_template_kwargs'] == {'enable_thinking': True}
        assert payload['max_tokens'] == 8192 and payload['stream'] is False
        assert {key: payload[key] for key in (
            'temperature', 'top_p', 'top_k', 'min_p', 'presence_penalty', 'repetition_penalty',
        )} == {'temperature': 1, 'top_p': 0.95, 'top_k': 20, 'min_p': 0,
               'presence_penalty': 0, 'repetition_penalty': 1}
        assert 'reasoning_format' not in payload and 'repeat_penalty' not in payload
        for case in CASES_BY_NAME.values():
            assert case.correct not in json.dumps(payload)
            assert case.incorrect not in json.dumps(payload)
        calls.append(payload)
        return httpx.Response(200, content=wire)

    report = await probe.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert report['status'] == 'completed_pending_review', report
    assert len(calls) == len(frozen) == len(report['documents']) == 12
    assert all(item == frozen[0] for item in frozen)
    assert (output / 'protocol.json').read_bytes() == frozen[0]
    protocol = json.loads(frozen[0])
    assert calls == [protocol['cases'][r['case']]['requests'][r['doc_type']] for r in protocol['schedule']]
    assert {(r['case'], r['doc_type'], r['repeat']) for r in report['documents']} == {
        (name, doc_type, repeat) for name in protocol['cases']
        for doc_type in ('explanation', 'qa') for repeat in (1, 2)
    }
    generator = DocGenerator()
    for name, case in protocol['cases'].items():
        metadata = generator.analyzer.analyze_file(output / 'sources' / f'{name}.py')
        for bundle in await generator.build_prompts_for_module(metadata, ('explanation', 'qa')):
            assert case['requests'][bundle.doc_type]['messages'] == [
                {'role': 'system', 'content': bundle.system_prompt},
                {'role': 'user', 'content': bundle.user_prompt},
            ]
    assert report['quality_verdict'] == 'unreviewed'
    assert all(entry['kind'] == 'chat' for entry in report['requests'])
    for entry in report['requests']:
        assert entry['response_archive']['original_sha256'] == hashlib.sha256(wire).hexdigest()
        assert entry['response_archive']['original_bytes'] == len(wire)
        assert entry['response_archive']['reasoning_present'] is True
        assert entry['usage']['completion_tokens_details']['reasoning_tokens'] == 200
        assert entry['usage']['completion_tokens'] == 400
        assert entry['accounted_usd'] == pytest.approx((100 * 0.2 + 400 * 2.5) / 1_000_000)
    saved = json.loads((output / 'responses' / '001.json').read_text())
    assert saved['choices'][0]['message'] == {'role': 'assistant', 'content': FINAL}
    assert (output / 'documents' / report['documents'][0]['file']).read_text() == FINAL
    assert len(list((output / 'responses').glob('*.json'))) == 12
    assert_no_private_text(output)


@pytest.mark.parametrize('kind', [
    'truncated', 'empty', 'mixed_reasoning', 'wrong_model', 'malformed_json',
    'tool_call', 'context_overflow', 'output_reservation_overflow', 'output_overflow', 'missing_usage',
])
async def test_invalid_response_stops_without_retry_or_private_archive(tmp_path, kind):
    output, calls = tmp_path / 'run', []

    def serve(request):
        calls.append(request)
        data = body()
        message = data['choices'][0]['message']
        if kind == 'truncated':
            data['choices'][0]['finish_reason'] = 'length'
        elif kind == 'empty':
            message['content'] = ''
        elif kind == 'mixed_reasoning':
            message['content'] = f'<think>{PRIVATE}</think>\n{FINAL}'
        elif kind == 'wrong_model':
            data['model'] = 'Qwen/Qwen3.5-9B'
        elif kind == 'malformed_json':
            return httpx.Response(200, content=('{"reasoning_content": "' + PRIVATE).encode())
        elif kind == 'tool_call':
            message['tool_calls'] = [{'function': {'arguments': PRIVATE}}]
        elif kind == 'context_overflow':
            data['usage']['prompt_tokens'] = 32700
        elif kind == 'output_reservation_overflow':
            data['usage']['prompt_tokens'] = 25000
        elif kind == 'output_overflow':
            data['usage']['completion_tokens'] = 8193
        elif kind == 'missing_usage':
            del data['usage']
        return httpx.Response(200, json=data)

    report = await probe.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == 1 and report['status'] == 'failed'
    assert report['quality_verdict'] == 'unreviewed' and not report['documents']
    assert (output / 'responses' / '001.json').exists()
    assert report['requests'][0]['response_archive']['original_sha256']
    assert_no_private_text(output)


@pytest.mark.parametrize('change', ['model', 'context', 'output', 'thinking'])
def test_admission_rejects_wrong_model_or_bounds(change):
    payload = probe.request_for('system', 'source')
    if change == 'model':
        payload['model'] = 'other'
    elif change == 'context':
        payload['messages'][1]['content'] = 'x' * 32768
    elif change == 'output':
        payload['max_tokens'] = 8193
    else:
        payload['reasoning_effort'] = 'none'
    with pytest.raises(ValueError, match=r'Request (differs|exceeds)'):
        probe.validate_request(payload)


async def test_insufficient_budget_sends_nothing_and_records_failure(tmp_path):
    calls = []
    report = await probe.execute(tmp_path / 'run', Budget(1e-10), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(calls.append))
    assert not calls and report['status'] == 'failed'
    assert report['quality_verdict'] == 'unreviewed'


async def test_total_attempt_limit_includes_bounded_429_retries(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        if len(calls) % 3:
            return httpx.Response(429, headers={'retry-after': '0'}, json={'error': PRIVATE})
        return httpx.Response(200, json=body())

    report = await probe.execute(tmp_path / 'run', Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == len(report['requests']) == 18
    assert report['status'] == 'failed' and len(report['documents']) == 6
    assert_no_private_text(tmp_path / 'run')


@pytest.mark.parametrize('failure', ['cancel', 'timeout'])
async def test_cancellation_and_timeout_keep_reservation_and_safe_terminal_evidence(tmp_path, failure):
    output, budget = tmp_path / 'run', Budget(1)

    def fail(request):
        assert request.method == 'POST'
        if failure == 'cancel':
            raise asyncio.CancelledError
        raise httpx.ReadTimeout(PRIVATE + ' test-secret')

    operation = probe.execute(output, budget, api_key='test-secret',
                              transport_factory=lambda: httpx.MockTransport(fail))
    if failure == 'cancel':
        with pytest.raises(asyncio.CancelledError):
            await operation
    else:
        await operation
    report = json.loads((output / 'report.json').read_text())
    assert report['status'] == ('cancelled' if failure == 'cancel' else 'failed')
    assert len(report['requests']) == 1
    assert budget.committed == report['requests'][0]['reserved_usd'] > 0
    assert_no_private_text(output)


async def test_budget_cap_and_existing_directory_fail_before_calls(tmp_path):
    calls = []
    with pytest.raises(ValueError, match=r'\$1'):
        await probe.execute(tmp_path / 'new', Budget(1.01), api_key='test-secret',
                            transport_factory=lambda: httpx.MockTransport(calls.append))
    assert not (tmp_path / 'new').exists()
    (tmp_path / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        await probe.execute(tmp_path, Budget(1), api_key='test-secret',
                            transport_factory=lambda: httpx.MockTransport(calls.append))
    assert not calls and (tmp_path / 'sentinel').read_text() == 'keep'


async def test_provider_credentials_redacted_from_final_output(tmp_path):
    data = copy.deepcopy(body())
    data['choices'][0]['message']['content'] = FINAL + ' test-secret'
    output = tmp_path / 'run'
    report = await probe.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(
                                     lambda _: httpx.Response(200, json=data)))
    assert report['status'] == 'completed_pending_review'
    assert_no_private_text(output)
