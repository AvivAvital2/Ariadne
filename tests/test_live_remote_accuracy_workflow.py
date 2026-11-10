"""Offline guardrails for remote accuracy runs; simulated output is not accuracy evidence."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import pytest

import llm
from tests import live_remote_accuracy_workflow as probe
from tests.live_model_smoke import Budget

KEY = 'remote-accuracy-test-credential'
PRIVATE = 'PRIVATE_REMOTE_ACCURACY_REASONING'


async def test_budget_refuses_before_network_and_bounds_all_attempts(tmp_path):
    requests = []
    report = await probe.execute(tmp_path / 'run', Budget(.000000001, 999), api_key=KEY,
                                 transport_factory=lambda: httpx.MockTransport(requests.append), pacing=0)
    assert report['status'] == 'failed' and requests == []
    protocol = json.loads((tmp_path / 'run/protocol.json').read_text())
    assert protocol['maximum_http_attempts'] == 80
    with pytest.raises(ValueError, match='At most'):
        await probe.execute(tmp_path / 'invalid', Budget(1.01), api_key=KEY)
    assert not (tmp_path / 'invalid').exists()


async def test_terminal_embedding_error_has_no_hidden_sdk_retry(tmp_path):
    requests = []

    def fail(request):
        requests.append(request)
        return httpx.Response(503, json={'error': KEY})

    report = await probe.execute(tmp_path / 'run', Budget(1), api_key=KEY,
                                 transport_factory=lambda: httpx.MockTransport(fail), pacing=0)
    assert report['status'] == 'failed'
    assert len(requests) == len(report['requests']) == 1
    assert report['accounted_upper_bound_usd'] > 0
    assert KEY not in json.dumps(report)


async def test_selector_mapping_preserves_hosted_context_and_restores_original(monkeypatch):
    calls = []
    hosted = SimpleNamespace(local=False, sentinel='storage identity stays hosted')

    async def original(*args, **kwargs):
        calls.append((args, kwargs))
        return 'real upstream completion'

    monkeypatch.setattr(llm, 'chat_complete', original)
    run = SimpleNamespace(report={}, save=lambda: None)
    with probe.selector_mapping(run):
        assert await llm.chat_complete(messages=[], context=hosted, phase='scip-obligation-plan',
                                       max_tokens=384, timeout=17) == 'real upstream completion'
        await llm.chat_complete(messages=[], context=hosted, phase='completion', max_tokens=4096)
    assert llm.chat_complete is original
    assert calls[0][1]['context'] is hosted
    assert calls[0][1]['max_tokens'] == 2432 and calls[0][1]['timeout'] == 17
    assert calls[1][1]['max_tokens'] == 4096
    assert hosted.local is False
    assert run.report['selector_cap_mapping'][0] == {
        'phase': 'scip-obligation-plan', 'requested_max_tokens': 384,
        'effective_max_tokens': 2432,
    }


def test_fixtures_preserve_real_source_extents_and_freeze_oracles(tmp_path):
    run, cfg, fixtures, verify = probe.prepare(tmp_path / 'run', Budget(1),
                                             httpx.AsyncHTTPTransport, 0)
    protocol = json.loads((run.output / 'protocol.json').read_text())
    assert cfg.scope_closure('consumer') == frozenset({'consumer', 'shared'})
    assert cfg.scope_closure('shared') == frozenset({'consumer', 'shared'})
    assert cfg.scope_closure('unrelated') == frozenset({'unrelated'})
    assert 'seeded' in protocol['graph_origin'] and 'not an indexer' in protocol['graph_origin']
    assert protocol['quality_verdict'] == 'unreviewed'
    oracle_data = json.loads((run.output / 'oracle.json').read_text())
    assert 'coverage miss' in oracle_data['coverage_policy']
    for item in fixtures[:2]:
        original = (probe.ROOT / item['origin']).read_text().splitlines()
        excerpt = (run.output / item['path']).read_text().splitlines()
        for start, end in item['ranges']:
            assert excerpt[start - 1:end] == original[start - 1:end]
    verify()
    oracle = run.output / 'oracle.json'
    oracle.write_text('{}')
    with pytest.raises(ValueError, match='Frozen'):
        verify()


def reply(request):
    payload = json.loads(request.content)
    if request.url.path.endswith('/embeddings'):
        vector = [1.] + [0.] * 4095
        return httpx.Response(200, json={
            'data': [{'index': i, 'embedding': vector} for i, _ in enumerate(payload['input'])],
            'usage': {'prompt_tokens': 20, 'total_tokens': 20},
        })
    prompt = payload['messages'][-1]['content']
    if 'You plan proof obligations' in payload['messages'][0]['content']:
        content = ('C1: establish how embedding.doc_embedding_text uses '
                   'embedding._DOC_EMBED_CONTENT_CHARS to bound content and preserve the title')
    elif 'C<n>: K<id>' in prompt:
        content = 'C1: K1'
    elif 'C<n>: S<id>' in prompt:
        content = 'C1: S1 S2'
    elif 'O1:' in prompt and 'F1.' in prompt:
        content = 'O1: F1'
    elif 'IDs only' in prompt or 'IDs, not prose' in prompt:
        content = 'M1 R1 B1'
    elif 'Description:' in prompt:
        content = 'This declaration exposes its supplied signature; omitted implementation behavior is unknown.'
    else:
        content = ('# Embedding input\n\n## Overview\n\n'
                   'The writer refreshes a document embedding. Missing documents return None.\n\n'
                   '## How It Works\n\n'
                   'The unbound branch passes the title and content to doc_embedding_text. '
                   'The helper combines the title, two newline characters and bounded content.\n\n'
                   '## Question\n\nWhat is omitted?\n\n## Answer\n\n'
                   'Network retries and the selected embedding model are not shown.\n')
    return httpx.Response(200, json={
        'model': 'Qwen/Qwen3.8-27B',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {
            'role': 'assistant', 'content': content, 'reasoning_content': PRIVATE}}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 200},
    })


async def test_real_stages_use_frozen_sources_unscripted_pipeline_and_record_evidence(tmp_path):
    output, requests = tmp_path / 'run', []
    frozen = []

    def respond(request):
        frozen.append((output / 'protocol.json').read_bytes())
        protocol = json.loads(frozen[-1])
        for path, digest in protocol['frozen_artifacts'].items():
            assert hashlib.sha256((output / path).read_bytes()).hexdigest() == digest
        payload = json.loads(request.content)
        assert probe.ORACLE_SENTINEL not in json.dumps(payload)
        assert request.headers['Authorization'] == 'Bearer ' + KEY
        if request.url.path.endswith('/chat/completions'):
            assert payload['chat_template_kwargs'] == {'enable_thinking': True}
            assert payload['reasoning_effort'] == 'xhigh'
            assert 0 < payload['max_tokens'] <= 8192
        requests.append(payload)
        return reply(request)

    report = await probe.execute(output, Budget(1), api_key=KEY,
                                 transport_factory=lambda: httpx.MockTransport(respond), pacing=0)
    assert report['status'] == 'completed_pending_quality_review', report
    assert report['quality_verdict'] == 'unreviewed'
    assert len(report['documents']) == 4
    assert len(report['answers']) == 3
    assert len(requests) == len(report['requests']) <= 80
    assert all(item == frozen[0] for item in frozen)
    assert report['selector_cap_mapping']
    assert {row['kind'] for row in report['requests']} == {'chat', 'embeddings'}
    assert {check['name'] for check in report['checks']} >= {
        'hosted_identity', 'source_backed_graph', 'clew_vectors',
        'search_scope/consumer-chain', 'search_scope/shared-bound',
        'search_scope/missing-network-policy', 'reopen_preserves_content_and_vectors',
    }
    for name in ('consumer-chain', 'shared-bound', 'missing-network-policy'):
        answer = json.loads((output / f'answer-{name}.json').read_text())
        assert answer['answer']
        assert (output / f'ask-trace-{name}.json').exists()
        assert (output / f'evidence-{name}.json').exists()
        evidence = json.loads((output / f'evidence-{name}.json').read_text())
        assert evidence['expected_relevant_documents']
        assert 'relevant_document_ranks' in evidence
    compact = json.loads((output / 'evidence-shared-bound.json').read_text())
    assert compact['phases'] == ['scip-obligation-plan', 'scip-route-family-select', 'completion']
    for path in output.rglob('*'):
        if path.is_file() and 'provenance' not in path.parts:
            raw = path.read_bytes()
            assert PRIVATE.encode() not in raw and KEY.encode() not in raw


async def test_global_attempt_cap_also_stops_rate_limit_retries(tmp_path):
    requests = []

    def retry(request):
        requests.append(request)
        return httpx.Response(429, headers={'Retry-After': '0'}, json={'error': 'rate limit'})

    report = await probe.execute(tmp_path / 'run', Budget(1, max_requests=1), api_key=KEY,
                                 transport_factory=lambda: httpx.MockTransport(retry), pacing=0)
    assert report['status'] == 'failed'
    assert len(requests) == len(report['requests']) == 1


def test_out_of_scope_ask_context_is_a_failure_independent_of_search(tmp_path):
    run, _, _, _ = probe.prepare(tmp_path / 'run', Budget(1), httpx.AsyncHTTPTransport, 0)
    answer = SimpleNamespace(model_dump=lambda **_: {'citations': [], 'chain_files': []})
    trace = [{'messages': [{'role': 'user', 'content': 'UNRELATED_CONTROL embedding helper'}]}]
    with pytest.raises(probe.WorkflowAbort, match='ask_scope'):
        probe.check_ask_scope(run, 'negative-control', trace, answer)
    assert run.report['checks'][-1]['passed'] is False


async def test_terminal_ask_failure_preserves_completed_trace_and_restores_wrapper(tmp_path, monkeypatch):
    original = llm.chat_complete

    async def fail_after_completion(_self, *_args, **_kwargs):
        await llm.chat_complete(messages=[{'role': 'user', 'content': 'Bounded offline completion'}])
        raise probe.WorkflowAbort('Deliberate terminal offline Ask failure')

    monkeypatch.setattr(probe.AriadneService, 'ask', fail_after_completion)
    output = tmp_path / 'run'
    report = await probe.execute(output, Budget(1), api_key=KEY,
                                 transport_factory=lambda: httpx.MockTransport(reply), pacing=0)
    assert report['status'] == 'failed'
    assert llm.chat_complete is original
    trace = json.loads((output / 'ask-trace-consumer-chain.json').read_text())
    assert len(trace) == 1 and trace[0]['phase'] == 'completion'
