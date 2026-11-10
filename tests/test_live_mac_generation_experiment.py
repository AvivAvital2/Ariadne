"""Offline guards for the bounded, paired 27B documentation prompt experiment."""

import asyncio
import hashlib
import json

import httpx
import pytest

from docgen.generator import DocGenerator
from tests import live_mac_generation_experiment as trial
from tests.live_mac_accuracy_probe import MODEL, SAMPLING, request_for
from tests.live_model_smoke import Budget

PRIVATE = 'PRIVATE_REASONING_NEVER_ARCHIVED'
FINAL = ('## Overview\nSource-backed behavior.\n\n## How It Works\n'
         'This is a synthetic transport fixture used to verify archival, budget and protocol behavior. '
         'Its text is not a factual evaluation of the supplied source and cannot qualify a model.\n\n'
         '## Question\nWhat does this test verify?\n\n## Answer\n'
         'It verifies that a complete final response is preserved, while private reasoning is omitted '
         'and factual correctness remains pending an independent review.')


def response():
    return {
        'model': MODEL,
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': FINAL, 'reasoning_content': PRIVATE,
        }}],
        'usage': {'prompt_tokens': 200, 'completion_tokens': 500, 'total_tokens': 700,
                  'completion_tokens_details': {'reasoning_tokens': 300}},
    }


def assert_private_absent(output):
    for path in output.rglob('*'):
        if path.is_file() and 'provenance' not in path.parts:
            assert PRIVATE not in path.read_text()
            assert 'test-secret' not in path.read_text()


def test_fresh_fixture_execution_oracles_cover_errors_and_state():
    cases = trial.frozen_cases()
    assert tuple(cases) == ('cache', 'guarded_cache', 'token_expiry',
                            'partial_settings', 'generator_groups', 'resource_view')
    assert {case['split'] for case in cases.values()} == {'known_regression', 'fresh_synthetic'}
    assert cases['partial_settings']['oracle'] == [
        {'scenario': 'replace_and_remove', 'returned_same_object': True, 'state': {'b': 2}},
        {'scenario': 'later_invalid_key', 'exception': 'ValueError', 'state': {'a': 2}},
        {'scenario': 'repeated_name', 'returned_same_object': True, 'state': {'a': 3}},
        {'scenario': 'absent_removal', 'returned_same_object': True, 'state': {}},
        {'scenario': 'empty_changes', 'returned_same_object': True, 'state': {'a': 1}},
    ]
    assert cases['generator_groups']['oracle'] == [
        {'scenario': 'lazy_groups', 'consumed_at_creation': 0, 'first': [1, 2],
         'consumed_after_first': 2, 'rest': [[3]], 'consumed_at_end': 3},
        {'scenario': 'invalid_size', 'creation_raised': False, 'exception_on_first_next': 'ValueError',
         'consumed': 0},
        {'scenario': 'empty_input', 'groups': []},
        {'scenario': 'unit_groups', 'groups': [[1], [2]]},
        {'scenario': 'short_input', 'groups': [[1, 2]]},
    ]
    assert cases['resource_view']['oracle'] == [
        {'scenario': 'owned_normal', 'factory_calls': ['data'], 'same_handle': True,
         'closed_inside': False, 'close_calls': 1},
        {'scenario': 'borrowed_normal', 'factory_calls': [], 'same_handle': True, 'close_calls': 0},
        {'scenario': 'owned_body_error', 'exception': 'ValueError', 'close_calls': 1},
        {'scenario': 'factory_error', 'exception': 'LookupError', 'body_entered': False},
        {'scenario': 'close_masks_body_error', 'exception': 'RuntimeError', 'close_calls': 1},
        {'scenario': 'borrowed_body_error', 'exception': 'ValueError', 'close_calls': 0},
        {'scenario': 'owned_close_error', 'exception': 'RuntimeError', 'close_calls': 1},
    ]
    assert all(case['required_behavior'] for case in cases.values())


async def test_protocol_frozen_before_24_paired_calls_and_baseline_unchanged(tmp_path):
    output, calls, protocols = tmp_path / 'trial', [], []

    def serve(request):
        frozen = (output / 'protocol.json').read_bytes()
        protocols.append(frozen)
        protocol = json.loads(frozen)
        assert protocol['model'] == MODEL and protocol['maximum_http_attempts'] == 32
        assert protocol['budget_usd'] == 1 and protocol['timeout_seconds'] == 600
        assert protocol['context_window'] == 32768 and protocol['max_total_output_tokens'] == 8192
        assert protocol['repetitions'] == protocol['concurrency'] == 1
        assert protocol['review_policy']['production_qualification'] is False
        assert protocol['review_policy']['decision_subject'] == (
            'Candidate arm; baseline supplies paired comparisons and previously correct facts')
        assert protocol['review_policy']['coverage_rule'] == 'Omissions and abstentions are coverage misses.'
        assert protocol['review_policy']['non_regression_unit'] == (
            'Each required fact within the same case/document-type pair; aggregate gains cannot conceal a lost fact')
        assert protocol['review_policy']['promising_requires_all'] == [
            'Fewer documents containing material false claims or contradictions than baseline',
            'No loss of correctly documented required behavior',
            'No reduction in required behavior coverage',
        ]
        assert protocol['review_policy']['further_validation_requires_all'] == [
            'Every scheduled document completed and independently reviewed',
            'Zero material false claims or contradictions',
            'All required behavior consistently covered correctly',
            'No loss of previously correct behavior',
        ]
        assert protocol['holdout_limit'].startswith('Fresh to this experiment')
        for path, digest in protocol['source_hashes'].items():
            assert hashlib.sha256((output / 'provenance' / path).read_bytes()).hexdigest() == digest
        oracle_bytes = (output / 'review-oracles.json').read_bytes()
        assert hashlib.sha256(oracle_bytes).hexdigest() == protocol['review_oracles_sha256']
        payload = json.loads(request.content)
        assert request.url == 'https://api.deepinfra.com/v1/openai/chat/completions'
        assert set(request.extensions['timeout'].values()) == {600}
        assert payload['model'] == MODEL and payload['max_tokens'] == 8192
        assert {key: payload[key] for key in SAMPLING} == SAMPLING
        for case in json.loads(oracle_bytes).values():
            assert case['required_behavior'] not in json.dumps(payload)
        assert 'oracle' not in payload and 'review_target' not in payload
        calls.append(payload)
        return httpx.Response(200, json=response())

    report = await trial.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert report['status'] == 'completed_pending_review', report
    assert report['quality_verdict'] == 'unreviewed'
    assert len(calls) == len(report['documents']) == 24
    assert all(item == protocols[0] for item in protocols)
    protocol = json.loads(protocols[0])
    assert calls == [protocol['cases'][s['case']]['requests'][s['doc_type']][s['arm']]
                     for s in protocol['schedule']]
    pairs = [protocol['schedule'][i:i + 2] for i in range(0, 24, 2)]
    assert all(a['case'] == b['case'] and a['doc_type'] == b['doc_type']
               and {a['arm'], b['arm']} == {'baseline', 'candidate'} for a, b in pairs)
    assert [pair[0]['arm'] for pair in pairs] == ['baseline', 'candidate'] * 6
    generator = DocGenerator()
    for name, case in protocol['cases'].items():
        source = (output / 'sources' / f'{name}.py').read_text()
        assert hashlib.sha256(source.encode()).hexdigest() == case['source_sha256']
        metadata = generator.analyzer.analyze_file(output / 'sources' / f'{name}.py')
        for bundle in await generator.build_prompts_for_module(metadata, ('explanation', 'qa')):
            baseline, candidate = (case['requests'][bundle.doc_type][arm]
                                   for arm in ('baseline', 'candidate'))
            assert baseline == request_for(bundle.system_prompt, bundle.user_prompt)
            assert candidate == trial.candidate_request(baseline, bundle.doc_type)
            assert source in baseline['messages'][1]['content']
            assert source in candidate['messages'][1]['content']
            assert {k: v for k, v in baseline.items() if k != 'messages'} == {
                k: v for k, v in candidate.items() if k != 'messages'}
            assert baseline['messages'][1]['content'].split('\n---\n\n')[0] == \
                candidate['messages'][1]['content'].split('\n---\n\n')[0]
    assert all(d['structurally_valid'] for d in report['documents'])
    assert len(list((output / 'responses').glob('*.json'))) == 24
    assert all(r['usage']['completion_tokens'] == 500 for r in report['requests'])
    assert_private_absent(output)


def test_generic_candidate_has_no_fixture_names_or_answers_and_preserves_input():
    recipe = trial.CANDIDATE_SYSTEM + '\n'.join(trial.CANDIDATE_TASKS.values())
    for name, case in trial.frozen_cases().items():
        assert name not in recipe
        assert case['required_behavior'] not in recipe
    baseline = request_for('Original system', 'SOURCE\n---\n\nOriginal task')
    before = json.dumps(baseline, sort_keys=True)
    changed = trial.candidate_request(baseline, 'qa')
    assert json.dumps(baseline, sort_keys=True) == before
    assert '5-10' in changed['messages'][1]['content']
    assert 'SOURCE\n---\n\n' in changed['messages'][1]['content']
    with pytest.raises(ValueError, match='boundary'):
        trial.candidate_request(request_for('system', 'no boundary'), 'qa')
    with pytest.raises(ValueError, match='document type'):
        trial.candidate_request(baseline, 'catalog')


@pytest.mark.parametrize('fault', ['truncation', 'malformed', 'wrong_model', 'timeout'])
async def test_failures_keep_evidence_stop_without_selective_retry(tmp_path, fault):
    calls, output = [], tmp_path / 'trial'

    def serve(request):
        calls.append(request)
        data = response()
        if fault == 'timeout':
            raise httpx.ReadTimeout(PRIVATE + ' test-secret')
        if fault == 'malformed':
            return httpx.Response(200, content=PRIVATE.encode())
        if fault == 'truncation':
            data['choices'][0]['finish_reason'] = 'length'
        elif fault == 'wrong_model':
            data['model'] = 'other-model'
        return httpx.Response(200, json=data)

    report = await trial.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert report['status'] == 'failed' and report['quality_verdict'] == 'unreviewed'
    assert len(calls) == 1 and not report['documents']
    assert report['requests'][0]['reserved_usd'] > 0
    assert_private_absent(output)


async def test_cancellation_preserves_budget_reservation(tmp_path):
    budget = Budget(1)

    def cancel(_request):
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await trial.execute(tmp_path / 'trial', budget, api_key='test-secret',
                            transport_factory=lambda: httpx.MockTransport(cancel))
    report = json.loads((tmp_path / 'trial/report.json').read_text())
    assert report['status'] == 'cancelled'
    assert budget.committed == report['requests'][0]['reserved_usd'] > 0


async def test_retry_budget_counts_every_429_attempt(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        if len(calls) % 3:
            return httpx.Response(429, headers={'retry-after': '0'}, json={'error': PRIVATE})
        return httpx.Response(200, json=response())

    report = await trial.execute(tmp_path / 'trial', Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert report['status'] == 'failed'
    assert len(calls) == len(report['requests']) == 32
    assert len(report['documents']) == 10
    assert_private_absent(tmp_path / 'trial')


@pytest.mark.parametrize('file', ['protocol.json', 'review-oracles.json', 'sources/resource_view.py'])
async def test_changed_frozen_protocol_or_oracles_stops_before_next_call(tmp_path, file):
    calls, output = [], tmp_path / 'trial'

    def serve(request):
        calls.append(request)
        (output / file).write_text('{}')
        return httpx.Response(200, json=response())

    report = await trial.execute(output, Budget(1), api_key='test-secret',
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(calls) == 1 and report['status'] == 'failed'
    assert len(report['documents']) == 1
    assert report['trials'][0]['status'] == 'completed'


async def test_budget_admission_and_existing_directory_preserve_state(tmp_path):
    calls = []
    def factory():
        return httpx.MockTransport(calls.append)
    with pytest.raises(ValueError, match=r'\$1'):
        await trial.execute(tmp_path / 'overspend', Budget(1.01), api_key='test-secret',
                            transport_factory=factory)
    assert not (tmp_path / 'overspend').exists()
    report = await trial.execute(tmp_path / 'tiny', Budget(1e-10), api_key='test-secret',
                                 transport_factory=factory)
    assert report['status'] == 'failed' and not calls
    with pytest.raises(FileExistsError):
        await trial.execute(tmp_path / 'tiny', Budget(1), api_key='test-secret', transport_factory=factory)
    with pytest.raises(ValueError, match='credential'):
        await trial.execute(tmp_path / 'missing', Budget(1), api_key='', transport_factory=factory)
    assert not (tmp_path / 'missing').exists()
