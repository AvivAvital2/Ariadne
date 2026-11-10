"""Frozen, paired source-scope trials use simulated HTTP in ordinary pytest."""

import asyncio
import copy
import hashlib
import json
import sys

import httpx
import pytest

from tests import live_mac_quality_trial as trial
from tests.live_mac_model_probe import frozen_requests
from tests.live_model_smoke import Budget
from tests.source_verification_cases import CASES_BY_NAME


def response(*, model='Qwen/Qwen3.5-9B', finish='stop', content=None):
    return httpx.Response(200, json={
        'model': model,
        'choices': [{'finish_reason': finish, 'message': {
            'content': content or '## Question\nWhat happens?\n\n## Answer\nA source-backed result.',
        }}],
        'usage': {'prompt_tokens': 30, 'completion_tokens': 80},
    })


def test_trial_changes_only_scope_instruction_and_keeps_confirmations_distinct():
    protocol = trial.frozen_trial()
    original = frozen_requests()
    names = ('cache', 'guarded_cache', 'token_expiry', 'job_claim', 'index_lookup', 'range_pop')
    assert tuple(protocol['cases']) == names
    assert len(protocol['schedule']) == 24
    assert protocol['variant'] == 'source-scope'
    assert protocol['timeout_seconds'] == 180
    assert protocol['maximum_http_attempts'] == 32
    assert protocol['local_q4_qualified'] is False
    assert protocol['quality_verdict'] == 'unreviewed'
    assert 'not independent holdout' in protocol['confirmation_limit']
    for name, case in protocol['cases'].items():
        baseline, candidate = case['requests']['baseline'], case['requests']['candidate']
        if name in original:
            assert baseline == original[name]
            assert case['split'] == 'known_regression'
        else:
            assert case['split'] == 'nine_b_confirmation'
        restored = copy.deepcopy(candidate)
        restored['messages'][0]['content'] = restored['messages'][0]['content'].removesuffix(
            '\n\n' + trial.SCOPE_INSTRUCTION)
        assert restored == baseline
        assert candidate['messages'][1] == baseline['messages'][1]
        assert case['review_target']['question'] == CASES_BY_NAME[name].question
        assert case['review_target']['required_behavior'] == CASES_BY_NAME[name].correct
        assert baseline['model'] == 'Qwen/Qwen3.5-9B'
        assert baseline['max_tokens'] == 8192 and baseline['reasoning_effort'] == 'none'
        assert baseline['min_p'] == 0 and baseline['chat_template_kwargs']['enable_thinking'] is False
        assert CASES_BY_NAME[name].source.strip() in baseline['messages'][1]['content']
        assert f'**Module**: `{name}`' in baseline['messages'][1]['content']
        assert f'**Functions**: {CASES_BY_NAME[name].function}' in baseline['messages'][1]['content']
        for known in CASES_BY_NAME.values():
            assert known.correct not in json.dumps(case['requests'])
            assert known.incorrect not in json.dumps(case['requests'])
        rows = [r for r in protocol['schedule'] if r['case'] == name]
        assert [(r['repeat'], r['arm']) for r in rows][2:] == [
            (2, r['arm']) for r in reversed(rows[:2])]
    assert protocol == trial.frozen_trial()


def test_review_decision_requires_error_reduction_without_coverage_loss():
    decision = trial.frozen_trial()['review_policy']['decision_rule']
    assert decision['promising_requires_all'] == [
        'Fewer documents containing material false claims than baseline',
        'No loss of previously correct required behavior',
        'No reduction in target coverage',
    ]
    assert decision['promotion_requires_all'] == [
        'All required behavior is covered correctly',
        'Zero material errors',
        'Successful evaluation on broader representative cases',
    ]
    assert decision['comparison_requires'] == (
        'All scheduled documents completed and independently reviewed against source')
    assert decision['this_trial_can_grant_production_qualification'] is False


def test_presence_zero_is_exactly_six_original_requests_with_one_sampling_change():
    protocol = trial.frozen_trial('presence-zero')
    original = frozen_requests()
    assert tuple(protocol['cases']) == ('cache', 'guarded_cache', 'token_expiry')
    assert protocol['variant'] == 'presence-zero'
    assert protocol['timeout_seconds'] == 600
    assert protocol['maximum_http_attempts'] == 10
    assert protocol['repetitions'] == 1
    assert protocol['scope_instruction'] is None
    assert protocol['single_changed_factor'] == 'Change presence_penalty from 1.5 to 0 only'
    assert [(r['case'], r['arm'], r['repeat']) for r in protocol['schedule']] == [
        ('cache', 'baseline', 1), ('cache', 'candidate', 1),
        ('guarded_cache', 'candidate', 1), ('guarded_cache', 'baseline', 1),
        ('token_expiry', 'baseline', 1), ('token_expiry', 'candidate', 1),
    ]
    for name, case in protocol['cases'].items():
        baseline, candidate = case['requests']['baseline'], case['requests']['candidate']
        assert baseline == original[name] and baseline['presence_penalty'] == 1.5
        assert candidate['presence_penalty'] == 0
        restored = copy.deepcopy(candidate)
        restored['presence_penalty'] = 1.5
        assert restored == baseline
        assert candidate['messages'] == baseline['messages']
        assert trial.SCOPE_INSTRUCTION not in json.dumps(candidate)
        for known in CASES_BY_NAME.values():
            assert known.correct not in json.dumps(case['requests'])
            assert known.incorrect not in json.dumps(case['requests'])
    decision = protocol['review_policy']['decision_rule']
    assert decision['further_validation_requires_all'] == [
        'All required behavior is covered correctly',
        'Zero material errors',
        'No loss of previously correct required behavior',
    ]
    assert decision['this_trial_can_grant_production_qualification'] is False
    assert 'does not establish causation' in protocol['hypothesis']
    assert protocol == trial.frozen_trial('presence-zero')


@pytest.mark.parametrize(('variant', 'count', 'timeout', 'attempts'), [
    ('source-scope', 24, 180, 32), ('presence-zero', 6, 600, 10),
])
async def test_protocol_and_oracles_frozen_before_any_call_all_outputs_retained(
        tmp_path, variant, count, timeout, attempts):
    output, calls, initial = tmp_path / 'run', [], []

    def serve(request):
        protocol_bytes = (output / 'protocol.json').read_bytes()
        initial.append(protocol_bytes)
        protocol = json.loads(protocol_bytes)
        assert protocol['variant'] == variant
        assert protocol['timeout_seconds'] == timeout
        assert protocol['maximum_http_attempts'] == attempts
        assert set(request.extensions['timeout'].values()) == {timeout}
        for name, case in protocol['cases'].items():
            source = (output / 'sources' / f'{name}.py').read_bytes()
            assert hashlib.sha256(source).hexdigest() == case['source_sha256']
            assert case['oracle']
        for name, digest in protocol['source_hashes'].items():
            assert hashlib.sha256((output / 'provenance' / name).read_bytes()).hexdigest() == digest
        assert request.url == 'https://api.deepinfra.com/v1/openai/chat/completions'
        assert request.headers['Authorization'] == 'Bearer test-secret'
        calls.append(json.loads(request.content))
        return response()

    budget = Budget(1)
    report = await trial.execute(output, budget, api_key='test-secret', variant=variant,
                                 transport_factory=lambda: httpx.MockTransport(serve))
    protocol = json.loads((output / 'protocol.json').read_text())
    assert budget.max_requests == attempts
    assert len(calls) == len(initial) == count
    assert all(value == initial[0] for value in initial)
    assert (output / 'protocol.json').read_bytes() == initial[0]
    assert calls == [protocol['cases'][r['case']]['requests'][r['arm']]
                     for r in protocol['schedule']]
    assert len(report['documents']) == len(report['trials']) == count
    assert all(r['status'] == 'completed' for r in report['trials'])
    assert len(list((output / 'documents').glob('*.md'))) == count
    assert len(list((output / 'responses').glob('*.json'))) == count
    assert report['status'] == 'completed_pending_review'
    assert report['quality_verdict'] == 'unreviewed'
    assert 'quality_passed' not in report
    assert all(r['kind'] == 'chat' for r in report['requests'])
    assert not list(output.rglob('*.db'))


@pytest.mark.parametrize('variant', ['source-scope', 'presence-zero'])
async def test_budget_refuses_first_request_and_records_incomplete_trial(tmp_path, variant):
    calls = []
    report = await trial.execute(tmp_path / 'run', Budget(1e-10), api_key='test-secret', variant=variant,
                                 transport_factory=lambda: httpx.MockTransport(calls.append))
    assert calls == [] and report['status'] == 'failed'
    assert 'spending limit' in report['error']
    assert report['quality_verdict'] == 'unreviewed'
    assert not report['documents']


async def test_model_allowlist_blocks_changed_payload_before_network(tmp_path, monkeypatch):
    frozen = trial.frozen_trial

    def changed(variant='source-scope'):
        protocol = frozen(variant)
        for case in protocol['cases'].values():
            for payload in case['requests'].values():
                payload['model'] = 'Qwen/Qwen3.8-27B'
        return protocol

    monkeypatch.setattr(trial, 'frozen_trial', changed)
    calls = []
    report = await trial.execute(tmp_path / 'run', Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(calls.append))
    assert not calls and report['status'] == 'failed'
    assert 'model' in report['error'].lower()


@pytest.mark.parametrize('kind', ['truncated', 'malformed', 'wrong_model', 'reasoning'])
async def test_bad_response_is_fatal_and_preserves_raw_output_without_retry(tmp_path, kind):
    calls = []

    def serve(request):
        calls.append(request)
        if kind == 'malformed':
            return httpx.Response(200, content=b'{broken json')
        if kind == 'wrong_model':
            return response(model='Qwen/Qwen3.8-27B')
        if kind == 'reasoning':
            return response(content='<think>private reasoning</think>')
        return response(finish='length')

    output = tmp_path / kind
    report = await trial.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == 1 and report['status'] == 'failed'
    assert report['trials'][0]['status'] == 'failed'
    assert (output / 'responses' / '001.json').exists()
    assert not list((output / 'documents').glob('*.md'))
    assert report['quality_verdict'] == 'unreviewed'


async def test_cancel_is_terminal_and_retains_admitted_cost(tmp_path):
    budget, calls, output = Budget(1), [], tmp_path / 'run'

    def cancel(request):
        calls.append(request)
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await trial.execute(output, budget, api_key='test-secret',
                            transport_factory=lambda: httpx.MockTransport(cancel))
    report = json.loads((output / 'report.json').read_text())
    assert len(calls) == 1 and report['status'] == 'cancelled'
    assert report['trials'][0]['status'] == 'cancelled'
    assert report['requests'][0]['reserved_usd'] == budget.committed > 0
    assert report['quality_verdict'] == 'unreviewed'


async def test_provider_error_is_redacted_in_all_saved_evidence(tmp_path):
    output = tmp_path / 'run'
    report = await trial.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(
                                     lambda _: httpx.Response(401, json={'error': 'test-secret'})))
    assert report['status'] == 'failed'
    for path in output.rglob('*.json'):
        assert 'test-secret' not in path.read_text()


async def test_existing_evidence_directory_is_never_overwritten(tmp_path):
    (tmp_path / 'sentinel').write_text('keep')
    with pytest.raises(FileExistsError):
        await trial.execute(tmp_path, Budget(1), api_key='test-secret',
                            transport_factory=lambda: httpx.MockTransport(lambda _: response()))
    assert (tmp_path / 'sentinel').read_text() == 'keep'


async def test_presence_zero_caps_total_http_attempts_including_429_retries(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        if len(calls) % 3:
            return httpx.Response(429, headers={'retry-after': '0'}, json={'error': 'busy'})
        return response()

    report = await trial.execute(
        tmp_path / 'run', Budget(1), api_key='test-secret', variant='presence-zero',
        transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == len(report['requests']) == 10
    assert len(report['documents']) == 3
    assert report['status'] == 'failed' and 'request or spending limit' in report['error']


@pytest.mark.parametrize('variant', ['source-scope', 'presence-zero'])
async def test_dollar_cap_is_enforced_before_preparation(tmp_path, variant):
    calls, output = [], tmp_path / 'run'
    with pytest.raises(ValueError, match=r'at most \$1'):
        await trial.execute(output, Budget(1.01), api_key='test-secret', variant=variant,
                            transport_factory=lambda: httpx.MockTransport(calls.append))
    assert not calls and not output.exists()


def test_cli_selects_presence_zero_variant_without_making_calls(tmp_path, monkeypatch):
    captured = {}

    async def simulate(output, budget, *, api_key, variant):
        assert api_key == 'test-secret'
        captured.update(output=output, budget=budget.limit, variant=variant, max_requests=budget.max_requests)
        return {'status': 'completed_pending_review'}

    monkeypatch.setenv('DEEPINFRA_API_KEY', 'test-secret')
    monkeypatch.setattr(sys, 'argv', ['trial', '--output', str(tmp_path / 'run'), '--variant', 'presence-zero'])
    monkeypatch.setattr(trial, 'execute', simulate)
    assert trial.main() == 0
    assert captured == {'output': tmp_path / 'run', 'budget': 1, 'variant': 'presence-zero', 'max_requests': 10}
