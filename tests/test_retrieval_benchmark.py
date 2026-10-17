"""Offline checks for the frozen benchmark, including both real retrieval paths."""

import json

import httpx
import pytest

from tests.live_model_smoke import PROFILES, Budget
from tests.live_retrieval_benchmark import execute, metrics, query_vectors
from tests.retrieval_cases import FIXTURE
from tests.test_live_model_smoke import model_response


def test_frozen_judgments_have_exact_evidence():
    corpus = json.loads(FIXTURE.read_text())
    docs = {d['id']: d for d in corpus['documents']}
    assert len(docs) == 22 and len(corpus['questions']) == 50
    assert len({q['question'] for q in corpus['questions']}) == 50
    for q in corpus['questions']:
        assert all(q['evidence'] in docs[key]['content'] for key in q['documents'])
        for level in ('chunks', 'sections'):
            for key in q[level]:
                doc_id, index = key.rsplit(':', 1)
                text = docs[doc_id][level][int(index)]
                if level == 'sections':
                    text = text[1]
                assert q['evidence'] in text


def test_metrics_count_all_relevant_targets_and_exclude_inapplicable_queries():
    judgments = [{'chunks': ['a', 'b']}, {'chunks': ['z']}, {'chunks': []}]
    rankings = [
        {'chunks': ['x', 'a', 'a']},
        {'chunks': ['a', 'b', 'c', 'd', 'e', 'z']},
        {'chunks': []},
    ]
    result = metrics(rankings, judgments, 'chunks')
    assert result == {
        'eligible_questions': 2,
        'recall_at_5': 0.25,
        'hit_at_5': 0.5,
        'mrr_at_5': 0.25,
    }


@pytest.mark.parametrize('name', ['qwen', 'openai'])
@pytest.mark.asyncio
async def test_frozen_benchmark_uses_embeddings_and_real_storage(tmp_path, monkeypatch, name):
    for key in ('OPENAI_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.setenv(key, 'test-only')
    requests = []

    def respond(request):
        assert request.url.path.endswith('/embeddings'), 'Benchmark must never call a chat model'
        requests.append(json.loads(request.content))
        return model_response(request)

    corpus = json.loads(FIXTURE.read_text())
    result = await execute(
        PROFILES[name],
        tmp_path,
        Budget(0.5),
        corpus,
        transport_factory=lambda: httpx.MockTransport(respond),
        pacing=0,
    )
    assert result['status'] == 'passed', result.get('error') or result['checks']
    # Production Qwen splits inputs into batches of at most 16; OpenAI keeps
    # the benchmark's 22 document requests and two batches of 25 questions.
    assert len(requests) == (27 if name == 'qwen' else 24)
    if name == 'qwen':
        assert all(len(request['input']) <= 16 for request in requests)
    assert result['metrics']['documents']['eligible_questions'] == 50
    assert result['metrics']['chunks']['eligible_questions'] == 46
    assert result['metrics']['sections']['eligible_questions'] == 50
    assert result['metrics']['mcp_evidence_delivery']['questions'] == 50
    assert all(c['passed'] for c in result['checks'])
    # Query batching must preserve the exact production instruction policy.
    from model_config import query_input

    expected = [
        query_input(q['question']) if name == 'qwen' else q['question'] for q in corpus['questions']
    ]
    assert [text for r in requests for text in r['input']][-len(expected):] == expected


@pytest.mark.parametrize('name', ['qwen', 'openai'])
@pytest.mark.asyncio
async def test_batched_queries_match_production_query_method(tmp_path, monkeypatch, name):
    from inference import InferenceContext
    from tests.live_model_smoke import Run, attach_clients, configuration

    for key in ('OPENAI_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.setenv(key, 'test-only')
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return model_response(request)

    profile = PROFILES[name]
    run = Run(profile, tmp_path, Budget(0.5), pacing=0)
    context = InferenceContext.from_config(configuration(profile, tmp_path))
    try:
        await attach_clients(context, run, lambda: httpx.MockTransport(respond))
        await context.embeddings.embed_query('How does retrieval work?')
        await query_vectors(context.embeddings, profile, ['How does retrieval work?'])
        assert requests[0] == requests[1]
    finally:
        await context.aclose()
