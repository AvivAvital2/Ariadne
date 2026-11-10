"""Guardrails for source-grounded remote generation evaluation."""

import hashlib
import json

import httpx
import pytest

from tests import live_remote_accuracy_generation as probe
from tests.live_model_smoke import Budget

REAL_REGRESSION_CASES = probe.regression_cases


def test_regressions_freeze_individual_facts_before_pair_review():
    for case in REAL_REGRESSION_CASES().values():
        facts = case['required_facts']
        assert len(facts) >= 4
        assert len({fact['id'] for fact in facts}) == len(facts)
        assert all(fact['text'].strip() for fact in facts)


@pytest.fixture(autouse=True)
def cases(monkeypatch):
    data = {
        'sample': {'case_id': 'sample', 'split': 'development',
                   'source': 'def twice(value):\n    return value * 2\n',
                   'doc_types': ['explanation', 'qa'],
                   'required_facts': [{'id': 'double', 'text': 'PRIVATE ORACLE SENTINEL'}],
                   'observations': {'answer': 4}, 'source_segments': []},
        'reserved': {'case_id': 'reserved', 'split': 'confirmation',
                     'source': 'def identity(value):\n    return value\n',
                     'doc_types': ['explanation', 'qa'],
                     'required_facts': [{'id': 'same', 'text': 'PRIVATE HOLDOUT SENTINEL'}],
                     'observations': {}, 'source_segments': []},
    }
    monkeypatch.setattr(probe, 'corpus_cases', lambda: data)
    monkeypatch.setattr(probe, 'regression_cases', lambda: {'sample': data['sample']})
    return data


def response(content='## Overview\nThe function doubles its input.', *, finish='stop'):
    return {'model': probe.MODEL, 'choices': [{'index': 0, 'finish_reason': finish,
            'message': {'role': 'assistant', 'content': content,
                        'reasoning_content': 'PRIVATE REASONING SENTINEL'}}],
            'usage': {'prompt_tokens': 100, 'completion_tokens': 40, 'total_tokens': 140}}


async def test_baseline_freezes_production_prompts_without_sending_oracles(tmp_path):
    run, protocol = await probe.prepare(tmp_path / 'run', Budget(.5), mode='baseline', repeats=2)
    assert len(protocol['schedule']) == 4
    assert {r['case'] for r in protocol['schedule']} == {'sample'}
    assert [r['repeat'] for r in protocol['schedule']] == [1, 1, 2, 2]
    requests = protocol['cases']['sample']['requests']
    assert 'Generate 5-10 Q&A pairs' in requests['qa']['baseline']['messages'][1]['content']
    assert 'Write a comprehensive explanation' in requests['explanation']['baseline']['messages'][1]['content']
    assert 'PRIVATE ORACLE' not in json.dumps(requests)
    assert 'PRIVATE HOLDOUT' not in json.dumps(protocol)
    assert 'PRIVATE ORACLE' in (run.output / 'review-oracles.json').read_text()
    assert protocol['quality_gates']['automatic_factual_pass'] is False
    assert run.report['quality_verdict'] == 'unreviewed'


async def test_candidate_changes_only_qa_task_scope_and_pairs_all_repeats(tmp_path):
    _, protocol = await probe.prepare(tmp_path / 'run', Budget(.5), mode='qa_scope', repeats=3)
    rows = protocol['schedule']
    assert len(rows) == 6 and {r['doc_type'] for r in rows} == {'qa'}
    assert [r['arm'] for r in rows] == ['baseline', 'candidate', 'candidate', 'baseline', 'baseline', 'candidate']
    requests = protocol['cases']['sample']['requests']['qa']
    baseline, candidate = requests['baseline'], requests['candidate']
    assert baseline['messages'][0] == candidate['messages'][0]
    assert baseline['messages'][1]['content'].split('\n---\n\n')[0] == candidate['messages'][1]['content'].split('\n---\n\n')[0]
    assert {k: v for k, v in baseline.items() if k != 'messages'} == {k: v for k, v in candidate.items() if k != 'messages'}
    assert 'Generate 5-10 Q&A pairs' not in candidate['messages'][1]['content']
    assert 'PRIVATE ORACLE' not in json.dumps(candidate)


async def test_confirmation_excludes_development_sources(tmp_path):
    _, protocol = await probe.prepare(tmp_path / 'run', Budget(.5), mode='baseline', split='confirmation', repeats=1)
    assert set(protocol['cases']) == {'reserved'}
    assert all(r['case'] == 'reserved' for r in protocol['schedule'])


async def test_paid_execution_keeps_every_output_and_never_auto_passes_quality(tmp_path):
    seen = []

    def serve(request):
        seen.append(json.loads(request.content))
        assert (tmp_path / 'run/protocol.json').exists()
        assert 'PRIVATE ORACLE' not in request.content.decode()
        assert request.extensions['timeout']['read'] == 600
        return httpx.Response(200, json=response())

    result = await probe.execute(tmp_path / 'run', Budget(.5), api_key='private-key',
                                 mode='baseline', repeats=1,
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert len(seen) == len(result['trials']) == len(result['documents']) == 2
    assert result['status'] == 'completed_pending_review'
    assert result['quality_verdict'] == 'unreviewed'
    for p in (tmp_path / 'run').rglob('*'):
        if p.is_file() and 'provenance' not in p.parts:
            assert 'PRIVATE REASONING SENTINEL' not in p.read_text()
            assert 'private-key' not in p.read_text()


async def test_truncated_output_retained_without_selective_retry(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        return httpx.Response(200, json=response(finish='length'))

    result = await probe.execute(tmp_path / 'run', Budget(.5), api_key='key', repeats=1,
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert result['status'] == 'failed'
    assert len(calls) == 1 and result['trials'][0]['status'] == 'failed'
    assert len(result['documents']) == 0 and result['accounted_upper_bound_usd'] > 0


async def test_frozen_source_tampering_stops_before_next_request(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        (tmp_path / 'run/sources/sample.py').write_text('changed')
        return httpx.Response(200, json=response())

    result = await probe.execute(tmp_path / 'run', Budget(.5), api_key='key', repeats=1,
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert result['status'] == 'failed' and len(calls) == 1
    assert len(result['documents']) == 1


async def test_unknown_usage_timeout_retains_reservation_without_retry(tmp_path):
    calls = []

    def serve(request):
        calls.append(request)
        raise httpx.ReadTimeout('private-key', request=request)

    result = await probe.execute(tmp_path / 'run', Budget(.5), api_key='private-key', repeats=1,
                                 transport_factory=lambda: httpx.MockTransport(serve))
    assert result['status'] == 'failed' and len(calls) == 1
    assert result['accounted_upper_bound_usd'] == result['requests'][0]['reserved_usd']
    assert 'private-key' not in json.dumps(result)


async def test_budget_refuses_network_and_preserves_frozen_protocol(tmp_path):
    result = await probe.execute(tmp_path / 'run', Budget(.00000001), api_key='key', repeats=1,
                                 transport_factory=lambda: httpx.MockTransport(lambda _: pytest.fail('Paid request')))
    assert result['status'] == 'failed' and result['requests'] == []
    assert result['protocol_sha256'] == hashlib.sha256((tmp_path / 'run/protocol.json').read_bytes()).hexdigest()


@pytest.mark.parametrize('kwargs', [{'repeats': 0}, {'repeats': 4}, {'mode': 'unknown'},
                                   {'split': 'unknown'}, {'mode': 'qa_scope', 'split': 'confirmation'}])
async def test_invalid_schedule_refused_before_creating_run(tmp_path, kwargs):
    with pytest.raises(ValueError, match='Invalid bounded experiment'):

        await probe.prepare(tmp_path / 'run', Budget(.5), **kwargs)
    assert not (tmp_path / 'run').exists()
