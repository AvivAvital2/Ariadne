"""Offline checks of the native diagnostic; no actual model inference."""

import json

import httpx
import numpy as np
import pytest

from tests import live_intel_local_workflow as probe


def test_freeze_precedes_requests_and_detects_changed_source(tmp_path):
    run, cfg = probe.prepare(tmp_path / 'run', timeout=14400,
                            chat_url='http://127.0.0.1:18080/v1',
                            embedding_url='http://127.0.0.1:18081/v1')
    protocol = json.loads((run.output / 'protocol.json').read_text())
    assert protocol['facts']['observations'][0]['exception'] == 'ValueError'
    assert protocol['facts']['observations'][0]['state_after'] == {}
    assert protocol['facts']['observations'][1]['return'] == 'u'
    assert protocol['facts']['observations'][2]['exception'] == 'KeyError'
    assert protocol['request_body']['request_timeout_seconds'] == 14400
    assert cfg.sources['app']['depends_on'] == ['shared']
    run.verify()
    (cfg.get_source_path('app') / 'tokens.py').write_text('pass\n')
    with pytest.raises(ValueError, match='Frozen input changed'):
        run.verify()


def test_vector_snapshot_rejects_wrong_width_and_compares_actual_bytes():
    vector = np.zeros(3072, dtype=np.float32)
    vector[0] = 1
    first = probe.vector_record(vector)
    changed = vector.copy()
    changed[0], changed[1] = 0, 1
    assert probe.vector_record(changed)['sha256'] != first['sha256']
    assert first['dimensions'] == 3072
    with pytest.raises(ValueError, match='3072'):
        probe.vector_record(np.ones(4096, dtype=np.float32))
    with pytest.raises(ValueError, match='normalized'):
        probe.vector_record(np.zeros(3072, dtype=np.float32))


def test_persistence_comparison_detects_lost_children():
    before = {'doc': {'content': 'content', 'sections': [{'content': 'section'}]}}
    after = {'doc': {'content': 'content', 'sections': []}}
    with pytest.raises(ValueError, match='Persistence'):
        probe.assert_persisted(before, after)
    probe.assert_persisted(before, json.loads(json.dumps(before)))


def test_fact_review_never_equates_structural_success_with_correctness():
    review = probe.fact_review('The expired token remains in the dictionary.')
    assert review['verdict'] == 'requires_independent_review'
    assert review['automated_correctness_claim'] is False
    assert review['expected_facts'] == probe.CASE.correct


def test_existing_output_or_external_config_is_never_overwritten(tmp_path):
    existing = tmp_path / 'exists'
    existing.mkdir()
    with pytest.raises(FileExistsError):
        probe.prepare(existing, timeout=14400, chat_url='http://127.0.0.1:18080/v1',
                      embedding_url='http://127.0.0.1:18081/v1')
    with pytest.raises(ValueError, match='inside'):
        probe.prepare(tmp_path / 'new', config_path=tmp_path / 'external.yaml', timeout=14400,
                      chat_url='http://127.0.0.1:18080/v1',
                      embedding_url='http://127.0.0.1:18081/v1')


DOCUMENT = (
    '# Token consumption\n\n## Overview\n\n'
    'consume_token removes a token from a caller-owned dictionary and checks its deadline.\n\n'
    '## Question\n\nWhat happens when the current time equals the deadline?\n\n'
    '## Answer\n\nThe function removes the token before comparing now with the deadline. '
    'At equality it raises ValueError("expired") and the token remains removed. '
    'Before the deadline it returns the user and consumes the token. '
    'A missing token raises KeyError during pop before the deadline comparison.\n'
)


@pytest.fixture
def native_servers():
    from tests.llamacpp_server import LlamaServer

    chat = LlamaServer('Qwen/Qwen3.8-27B', thinking=True)
    embed = LlamaServer('Qwen/Qwen3-Embedding-8B', embeddings=True)
    chat.overrides['/native/v1/chat/completions'] = {
        'model': 'Qwen/Qwen3.8-27B',
        'choices': [{'message': {'role': 'assistant', 'content': DOCUMENT,
                                 'reasoning_content': 'test-reasoning'}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 100, 'completion_tokens': 300},
    }
    try:
        yield chat, embed
    finally:
        chat.close()
        embed.close()


async def test_saved_api_native_generation_reopen_and_scoped_ask(tmp_path, native_servers):
    chat, embed = native_servers
    output = tmp_path / 'run'
    report = await probe.execute(output, chat_url=chat.base_url, embedding_url=embed.base_url)
    assert report['structural_verdict'] == 'passed', report
    assert report['factual_verdict'] == 'requires_independent_review'
    assert all(row['passed'] for row in report['checks'])
    project = json.loads((output / 'project.json').read_text())
    from config import Config

    cfg = Config(project['config_path'])
    assert cfg.sources['app']['depends_on'] == ['shared']
    assert cfg.model_settings().chat.timeout_seconds == 14400
    assert cfg.model_settings().embeddings.timeout_seconds == 14400
    assert json.loads((output / 'before-reopen.json').read_text()) == json.loads(
        (output / 'after-reopen.json').read_text())
    completions = [payload for route, payload in chat.requests if route.endswith('/chat/completions')]
    assert [item['max_tokens'] for item in completions] == [2048, 8192, 4096]
    assert all(item['chat_template_kwargs'] == {'enable_thinking': True} for item in completions)
    assert all(item['reasoning_effort'] == 'xhigh' for item in completions)
    assert len(report['requests']) == len(chat.requests) + len(embed.requests)
    assert {row['name'] for row in report['phases']} == {
        'connection_and_saved_project', 'generate_qa', 'persist_vectors', 'reopen', 'search', 'ask'}
    for row in report['requests']:
        assert row['status'] == 'received'
        assert (output / row['response_file']).exists()


async def test_bad_native_embedding_stops_before_project_creation(tmp_path, native_servers):
    chat, embed = native_servers
    embed.overrides['/native/v1/embeddings'] = {
        'data': [{'index': 0, 'embedding': [1.0] * 3072}]}
    output = tmp_path / 'run'
    report = await probe.execute(output, chat_url=chat.base_url, embedding_url=embed.base_url)
    assert report['status'] == 'failed'
    assert not (output / 'project.json').exists()
    assert not (output / 'root' / 'web-projects' / 'model-profiles.json').exists()
    assert sum(route.endswith('/chat/completions') for route, _ in chat.requests) == 1


@pytest.mark.parametrize(('phase', 'cap'), [
    ('connection_and_saved_project', 2049), ('ask', 4097), ('unexpected', 1),
])
async def test_recorder_rejects_phase_cap_drift_before_sending(tmp_path, phase, cap):
    run, _cfg = probe.prepare(tmp_path / 'run', timeout=14400,
                             chat_url='http://127.0.0.1:18080/v1',
                             embedding_url='http://127.0.0.1:18081/v1')
    run.phase_name = phase
    payload = {**probe.mac_accuracy_profile()['llm']['request_options'],
               'model': 'Qwen/Qwen3.8-27B', 'max_tokens': cap}
    request = httpx.Request('POST', 'http://127.0.0.1:18080/v1/chat/completions', json=payload)
    with pytest.raises(ValueError, match='phase'):
        await run.request_started(request)
    assert run.report['requests'] == []


async def test_saved_project_and_registry_are_frozen(tmp_path, native_servers):
    chat, embed = native_servers
    run, cfg = probe.prepare(tmp_path / 'run', timeout=14400,
                            chat_url=chat.base_url, embedding_url=embed.base_url)
    body = json.loads((run.output / 'protocol.json').read_text())['request_body']
    with run.record_native_requests(), run.phase('connection_and_saved_project'):
        saved = await probe.save_through_api(run, cfg, body, 14400)
    from pathlib import Path

    paths = [Path(saved.config_path), Path(cfg.config_path).parent / 'web-projects' / 'model-profiles.json']
    for path in paths:
        assert str(path) in run.frozen
        original = path.read_bytes()
        try:
            path.write_bytes(original + b'\n')
            with pytest.raises(ValueError, match='Frozen input changed'):
                run.verify()
        finally:
            path.write_bytes(original)
    run.verify()


async def test_ask_fallback_is_not_accepted_as_real_synthesis(tmp_path, native_servers, monkeypatch):
    chat, embed = native_servers

    def fail_after_completion(_answer):
        raise ValueError('Deliberate post-completion failure')

    monkeypatch.setattr('library.chain_answer.expand_bare_lines', fail_after_completion)
    output = tmp_path / 'run'
    report = await probe.execute(output, chat_url=chat.base_url, embedding_url=embed.base_url)
    answer = json.loads((output / 'answer.json').read_text())
    assert answer['confidence_reasons'] == ['synthesis failed: ValueError']
    assert answer['llm_calls'] == 0
    assert report['status'] == 'failed'
    assert any(row['name'] == 'ask_real_synthesis' and not row['passed'] for row in report['checks'])
