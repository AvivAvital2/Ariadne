"""Offline checks for the experimental review gate, scoring and actual storage."""

import json
import sqlite3
from unittest.mock import AsyncMock

import httpx
import pytest

from docgen.llm.local import LocalInferenceError
from tests import live_source_verification as experiment
from tests.live_model_smoke import Budget
from tests.source_verification_cases import CASES_BY_NAME, observe_case
from tests.test_live_model_smoke import model_response


@pytest.mark.parametrize('case', CASES_BY_NAME.values(), ids=lambda c: c.name)
def test_execution_oracles(case):
    assert len(observe_case(case)) == len(case.probes)


@pytest.mark.parametrize(
    'raw',
    [
        None,
        'not json',
        '[]',
        '{"verdict":"accept","issues":[{"quote":"claim","reason":"false"}]}',
        '{"verdict":"reject","issues":[]}',
        '{"verdict":"accept","issues":[],"correct":true}',
        '{"verdict":"reject","issues":[{"quote":"invented quotation","reason":"false"}]}',
        '{"verdict":"reject","issues":[{"quote":"claim","reason":""}]}',
    ],
)
def test_malformed_or_inconsistent_reviews_never_admit(raw):
    assert experiment.parse_review(raw, 'claim')['verdict'] == 'invalid'


def test_grounded_rejection_and_uncertainty():
    for verdict in ('reject', 'uncertain'):
        value = {'verdict': verdict, 'issues': [{'quote': 'claim', 'reason': 'source evidence'}]}
        assert experiment.parse_review(json.dumps(value), 'A claim here.') == value


@pytest.mark.asyncio
async def test_reviewer_receives_only_source_and_document():
    provider = AsyncMock()
    provider.call.return_value = '{"verdict":"accept","issues":[]}'
    result = await experiment.review_document(provider, 'source', 'document')
    assert result['verdict'] == 'accept'
    call = provider.call.call_args
    assert json.loads(call.args[1]) == {'source': 'source', 'documentation': 'document'}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'error',
    [LocalInferenceError('Incomplete final output'), ValueError('Incomplete Claude response')],
)
async def test_truncated_review_is_an_invalid_decision_without_stopping_study(error):
    provider = AsyncMock()
    provider.call.side_effect = error
    result = await experiment.review_document(provider, 'source', 'document')
    assert result['verdict'] == 'invalid'
    assert 'error' in result


@pytest.mark.asyncio
async def test_other_provider_errors_are_not_hidden_as_invalid_reviews():
    provider = AsyncMock()
    provider.call.side_effect = ValueError('Invalid model configuration')
    with pytest.raises(ValueError, match='Invalid model configuration'):
        await experiment.review_document(provider, 'source', 'document')


def test_reused_drafts_preserve_text_and_refuse_tampering(tmp_path):
    samples = experiment.frozen_samples()
    experiment.save_json(tmp_path / 'samples.json', samples)
    experiment.save_json(
        tmp_path / 'protocol.json',
        {
            'review_prompt_sha256': experiment.digest(experiment.REVIEW_SYSTEM),
            'cases_sha256': experiment.digest(
                (experiment.ROOT / 'tests/source_verification_cases.py').read_text()
            ),
        },
    )
    drafts = [
        {
            'id': f'{name}-{repeat}',
            'document': 'unchanged',
            'sha256': experiment.digest('unchanged'),
        }
        for name in experiment.FRESH_CASES
        for repeat in (1, 2)
    ]
    experiment.save_json(tmp_path / 'drafts.json', drafts)
    assert experiment.load_reused_drafts(tmp_path, samples) == drafts
    drafts[0]['document'] = 'edited after review'
    experiment.save_json(tmp_path / 'drafts.json', drafts)
    with pytest.raises(ValueError, match='hash mismatch'):
        experiment.load_reused_drafts(tmp_path, samples)


def test_rejecting_everything_is_not_success():
    metrics = experiment.classification_metrics(
        [
            {'correct': True, 'verdict': 'reject'},
            {'correct': False, 'verdict': 'reject'},
        ]
    )
    assert metrics['incorrect_accepted'] == 0
    assert metrics['incorrect_fraction_among_accepted'] is None
    assert metrics['correct_acceptance_rate'] == 0
    assert metrics['acceptance_rate'] == 0
    assert metrics['correct_not_accepted'] == 1


@pytest.mark.asyncio
async def test_baseline_and_gate_store_identical_drafts_without_rejected_content(
    tmp_path, monkeypatch
):
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'test-deepinfra')
    monkeypatch.setattr(experiment, 'WORKFLOW_CASES', ('cache',))
    drafts = [
        {
            'id': 'accepted',
            'case': 'cache',
            'title': 'Cache A',
            'document': '## Question\nHow does cache eviction work?\n\n## Answer\nCache entries are removed. '
            * 4,
            'structural_valid': True,
            'review': {'verdict': 'accept'},
        },
        {
            'id': 'rejected',
            'case': 'cache',
            'title': 'Cache B',
            'document': '## Question\nWhat happens with negative capacity?\n\n## Answer\nThe cache silently succeeds. '
            * 4,
            'structural_valid': True,
            'review': {'verdict': 'reject'},
        },
    ]
    budget, runs = Budget(1), []
    for arm, gated, expected in (
        ('baseline', False, ['accepted', 'rejected']),
        ('verified', True, ['accepted']),
    ):
        result = await experiment.store_and_ask(
            tmp_path / arm,
            drafts,
            budget,
            runs,
            gated=gated,
            transport_factory=lambda: httpx.MockTransport(model_response),
            pacing=0,
        )
        assert result['stored_ids'] == expected
        with sqlite3.connect(tmp_path / arm / 'library.db') as conn:
            stored = dict(conn.execute('SELECT id, content FROM documents'))
        assert stored == {d['id']: d['document'] for d in drafts if d['id'] in expected}
        assert (
            all('Cache B' not in s for s in result['answers']['cache']['sources'])
            if gated
            else True
        )


def test_false_accept_fails_even_when_other_gates_pass(tmp_path, monkeypatch):
    monkeypatch.setattr(experiment, 'WORKFLOW_CASES', ('cache',))
    good = experiment.classification_metrics(
        [
            {'correct': True, 'verdict': 'accept'},
            {'correct': False, 'verdict': 'reject'},
        ]
    )
    summary = {
        'status': 'awaiting_independent_annotations',
        'controls': {
            'known_failure': experiment.classification_metrics(
                [{'correct': False, 'verdict': 'reject'}]
            ),
            'development': good,
            'heldout': good,
        },
        'workflows': {
            arm: {'answers': {'cache': {'answer': 'KeyError', 'sources': ['Cache']}}}
            for arm in ('baseline', 'verified')
        },
    }
    drafts = [
        {
            'id': 'wrong',
            'sha256': experiment.digest('wrong'),
            'review': {'verdict': 'accept'},
            'structural_valid': True,
        }
    ]
    experiment.save_json(tmp_path / 'summary.json', summary)
    experiment.save_json(tmp_path / 'drafts.json', drafts)
    annotations = {
        'drafts': {
            'wrong': {
                'sha256': drafts[0]['sha256'],
                'correct': False,
                'notes': 'False boundary claim.',
            }
        },
        'answers': {
            f'{arm}/cache': {
                'sha256': experiment.digest('KeyError'),
                'result': 'correct',
                'notes': 'Matches executed source.',
            }
            for arm in ('baseline', 'verified')
        },
    }
    result = experiment.score_study(tmp_path, annotations)
    assert result['status'] == 'not_qualified'
    assert not result['gates']['fresh']
    assert result['fresh']['incorrect_accepted'] == 1
    annotations['drafts']['wrong']['sha256'] = 'stale'
    with pytest.raises(ValueError, match='stale'):
        experiment.score_study(tmp_path, annotations)
