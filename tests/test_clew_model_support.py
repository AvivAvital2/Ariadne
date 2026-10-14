"""Clew builders obey model binding, writer ownership and the Qwen HTTP contract."""

import asyncio
import json
import sqlite3
import sys

import numpy as np
import pytest

from ariadne_mcp.service import AriadneService
from docgen.llm.local import EmbeddingContractError
from inference import InferenceContext
from library.clew_embeddings import embed_clews
from library.clews import add_clew, nearest_clew_matches
from library.locking import LibraryBusy
from library.model_binding import ModelBindingMismatch, corpus_revision
from tests.clew_workflow import (
    QUESTION,
    ROUTE,
    add_guide,
    build_routes,
    script,
    seed_graph,
    seed_theme,
)
from tests.local_model_server import deployments, trap_inference
from tests.test_local_workflows import fixture_config


@pytest.fixture
def setup(tmp_path, monkeypatch):
    import config

    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        cfg = fixture_config(tmp_path, chat, embed)
        monkeypatch.setattr(config, '_global_config', cfg)
        monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-used')
        monkeypatch.setenv('OPENAI_BASE_URL', 'https://unexpected.invalid/v1')
        context = InferenceContext.from_config(cfg)
        with context.write_session():
            library = context.open_library(operation='write')
            seed_graph(library, tmp_path / 'alpha', 'alpha')
        try:
            yield cfg, context, library, chat, embed
        finally:
            asyncio.run(context.aclose())


def test_builders_honor_config_and_keep_previews_offline(setup, monkeypatch):
    _, context, library, chat, embed = setup
    builder = script('write_clews')['main']
    before = corpus_revision(context.db_path)
    monkeypatch.setattr(sys, 'argv', ['write_clews', '--source', 'alpha', '--dry-run'])
    assert builder() == 0
    assert corpus_revision(context.db_path) == before
    assert not chat.requests and not embed.requests
    monkeypatch.setattr(sys, 'argv', ['write_clews', '--source', 'alpha', '--yes'])
    assert builder() == 0
    with library._conn_provider.acquire() as conn:
        assert conn.execute('SELECT count(*) FROM clews').fetchone()[0] == 2
    before = corpus_revision(context.db_path)
    embedder = script('embed_clews')['run']
    assert asyncio.run(embedder(None, 'alpha', 2, None, False)) == 0
    assert corpus_revision(context.db_path) == before
    assert not chat.requests and not embed.requests
    assert asyncio.run(embedder(None, 'alpha', 2, None, True)) == 0
    requests = [payload for path, payload, _ in embed.requests if path.endswith('/embeddings')]
    assert len(requests) == 1
    assert set(requests[0]['input']) == {
        'accept_payment -> authorize_payment -> record_receipt',
        'start_reset -> validate_token -> consume_token',
    }
    assert requests[0]['model'] == 'Qwen/Qwen3-Embedding-8B'
    assert 'dimensions' not in requests[0]
    with library._conn_provider.acquire() as conn:
        vectors = [np.frombuffer(row[0], dtype=np.float32)
                   for row in conn.execute('SELECT embedding FROM clews')]
    assert all(vector.shape == (3072,) for vector in vectors)
    assert all(np.linalg.norm(vector) == pytest.approx(1) for vector in vectors)
    before_requests = len(embed.requests)
    assert asyncio.run(embedder(None, 'alpha', 2, None, True)) == 0
    assert len(embed.requests) == before_requests


def test_writer_lock_and_incompatible_models_fail_before_inference(setup, monkeypatch):
    cfg, context, library, chat, embed = setup
    builder = script('write_clews')['main']
    monkeypatch.setattr(sys, 'argv', ['write_clews', '--source', 'alpha', '--yes'])
    with context.write_session():
        with pytest.raises(LibraryBusy):
            builder()
    with pytest.raises(LibraryBusy):
        asyncio.run(embed_clews(library, source_name='alpha'))
    cfg._config['embeddings']['model_revision'] = 'c' * 40
    with pytest.raises(ModelBindingMismatch):
        asyncio.run(script('embed_clews')['run'](None, 'alpha', 2, None, True))
    assert not chat.requests and not embed.requests


def test_failed_batch_rolls_back_and_resume_skips_applied_rows(setup):
    _, context, library, _, embed = setup
    with context.write_session():
        build_routes(library, 'alpha')
        with library._conn_provider.acquire() as conn:
            second = conn.execute('SELECT id FROM clews ORDER BY id LIMIT 1 OFFSET 1').fetchone()[0]
            conn.execute('CREATE TRIGGER reject_clew BEFORE UPDATE OF embedding ON clews '
                         f"WHEN NEW.id='{second}' BEGIN SELECT RAISE(ABORT, 'injected'); END")
        with pytest.raises(sqlite3.IntegrityError, match='injected'):
            asyncio.run(embed_clews(library, source_name='alpha', batch_size=2))
        with library._conn_provider.acquire() as conn:
            assert conn.execute('SELECT count(*) FROM clews WHERE embedding IS NOT NULL').fetchone()[0] == 0
        with pytest.raises(sqlite3.IntegrityError, match='injected'):
            asyncio.run(embed_clews(library, source_name='alpha', batch_size=1))
        with library._conn_provider.acquire() as conn:
            assert conn.execute('SELECT count(*) FROM clews WHERE embedding IS NOT NULL').fetchone()[0] == 1
            conn.execute('DROP TRIGGER reject_clew')
        before = len([r for r in embed.requests if r[0].endswith('/embeddings')])
        assert asyncio.run(embed_clews(library, source_name='alpha')) == 1
        assert len([r for r in embed.requests if r[0].endswith('/embeddings')]) == before + 1


def test_malformed_vectors_never_reach_clew_storage(setup):
    _, context, library, _, embed = setup
    with context.write_session():
        build_routes(library, 'alpha')
        embed.embedding_response = {'data': [{'index': 0, 'embedding': [1.0] * 8}]}
        with pytest.raises(EmbeddingContractError):
            asyncio.run(embed_clews(library, source_name='alpha'))
        with library._conn_provider.acquire() as conn:
            assert conn.execute('SELECT count(*) FROM clews WHERE embedding IS NOT NULL').fetchone()[0] == 0


@pytest.mark.parametrize('mode', ['deterministic', 'llm'])
def test_qwen_clew_reaches_real_evidence_walk_and_synthesis(setup, mode):
    cfg, context, library, chat, embed = setup
    cfg._config['ask_selector_mode'] = mode

    def reply():
        payload = chat.requests[-1][1]
        text = json.dumps(payload['messages'])
        if 'C<n>: K<id>' in text:
            content = 'C1: K1'
        elif 'C<n>: S<id>' in text:
            content = 'C1: S1 S2 S3'
        elif 'IDs only' in text or 'IDs, not prose' in text:
            content = 'M1 R1 B1'
        else:
            content = 'payments.accept_payment calls authorize_payment, which calls record_receipt.'
        chat.chat_response = {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
                              'usage': {'prompt_tokens': 32, 'completion_tokens': 20}}

    chat.barrier = reply

    async def run():
        with context.write_session():
            build_routes(library, 'alpha')
            await embed_clews(library, source_name='alpha')
            await add_guide(library, 'alpha')
        service = AriadneService()
        service._config, service._inference_context = cfg, context
        try:
            answer = await service.ask(QUESTION, source='alpha')
            selected = answer.graph_diagnostics['clew_selection']
            assert selected['selected'] >= 1, selected
            assert not selected.get('error'), selected
            assert answer.route_candidates, answer.model_dump()
            assert any(set(ROUTE).issubset(route) for route in answer.route_candidates.values())
            assert answer.llm_calls >= 1
            assert 'synthesis unavailable' not in answer.answer
            assert all(symbol in json.dumps(chat.requests) for symbol in ROUTE)
            assert any("return {'receipt': amount}" in json.dumps(payload)
                       for path, payload, _ in chat.requests if path.endswith('/chat/completions'))
        finally:
            await service.close()

    asyncio.run(run())
    queries = [text for path, payload, _ in embed.requests if path.endswith('/embeddings')
               for text in payload['input'] if text.startswith('Instruct:')]
    assert any(QUESTION in text for text in queries)


def test_clew_text_changes_invalidate_vectors_but_metadata_preserves_them(setup):
    _, context, library, _, _ = setup
    vector = np.zeros(3072, dtype=np.float32)
    vector[0] = 1
    with context.write_session(), library._conn_provider.acquire() as conn:
        values = {'source_name': 'alpha', 'entry_symbol': ROUTE[0], 'steps': ['accept', 'record'],
                  'route': ROUTE, 'files': ['payments.py']}
        cid = add_clew(conn, **values, question='Which payment?', embedding=vector)
        add_clew(conn, **values, strategy='another-strategy')
        assert conn.execute('SELECT embedding FROM clews WHERE id=?', (cid,)).fetchone()[0] == vector.tobytes()
        add_clew(conn, **values, question='Where is the receipt?')
        assert conn.execute('SELECT embedding FROM clews WHERE id=?', (cid,)).fetchone()[0] is None
        add_clew(conn, **values, embedding=vector)
        values['steps'] = ['accept', 'authorize', 'record']
        add_clew(conn, **values)
        assert conn.execute('SELECT embedding FROM clews WHERE id=?', (cid,)).fetchone()[0] is None


def test_bound_clew_cache_observes_vector_replacement_from_another_connection(setup):
    cfg, context, library, _, _ = setup
    first, second = np.zeros((2, 3072), dtype=np.float32)
    first[0] = second[1] = 1
    with context.write_session(), library._conn_provider.acquire() as conn:
        cid = add_clew(conn, source_name='alpha', entry_symbol=ROUTE[0], steps=['pay'],
                       route=ROUTE, files=['payments.py'], embedding=first)
    reader = InferenceContext.from_config(cfg)
    try:
        read_library = reader.open_library()
        with read_library._conn_provider.acquire() as conn:
            assert nearest_clew_matches(conn, first, source_name='alpha', min_similarity=0.9)
        with context.write_session(), library._conn_provider.acquire() as conn:
            conn.execute('UPDATE clews SET embedding=? WHERE id=?', (second.tobytes(), cid))
        with read_library._conn_provider.acquire() as conn:
            assert not nearest_clew_matches(conn, first, source_name='alpha', min_similarity=0.9)
            assert nearest_clew_matches(conn, second, source_name='alpha', min_similarity=0.9)
    finally:
        asyncio.run(reader.aclose())


def test_embedding_limits_and_source_scope_leave_other_routes_untouched(setup):
    _, context, library, _, embed = setup
    with context.write_session():
        build_routes(library, 'alpha')
        with library._conn_provider.acquire() as conn:
            foreign = add_clew(conn, source_name='beta', entry_symbol=ROUTE[0], steps=['pay'],
                               route=ROUTE, files=['payments.py'])
        assert asyncio.run(embed_clews(library, source_name='alpha', limit=1)) == 1
        with library._conn_provider.acquire() as conn:
            assert conn.execute('SELECT count(*) FROM clews WHERE embedding IS NOT NULL').fetchone()[0] == 1
            assert conn.execute('SELECT embedding FROM clews WHERE id=?', (foreign,)).fetchone()[0] is None
        assert asyncio.run(embed_clews(library, source_name='alpha')) == 1
        inputs = [text for path, payload, _ in embed.requests if path.endswith('/embeddings')
                  for text in payload['input']]
        assert len(inputs) == 2


def test_clew_question_generation_uses_qwen_and_invalidates_only_changed_vectors(setup):
    _, context, library, chat, embed = setup
    with context.write_session():
        build_routes(library, 'alpha')
        seed_theme(library, 'alpha')
        asyncio.run(embed_clews(library, source_name='alpha'))
    questioner = script('question_clews')['run']
    requests = len(chat.requests) + len(embed.requests)
    revision = corpus_revision(context.db_path)
    assert asyncio.run(questioner(None, 'alpha', False, None)) == 0
    assert len(chat.requests) + len(embed.requests) == requests
    assert corpus_revision(context.db_path) == revision
    chat.chat_response = {'choices': [{'message': {'content': QUESTION}, 'finish_reason': 'stop'}],
                          'usage': {'prompt_tokens': 32, 'completion_tokens': 20}}
    assert asyncio.run(questioner(None, 'alpha', True, None)) == 0
    with library._conn_provider.acquire() as conn:
        question, vector = conn.execute('SELECT question, embedding FROM clews WHERE entry_symbol=?',
                                       (ROUTE[0],)).fetchone()
        assert question == QUESTION and vector is None
        assert conn.execute('SELECT count(*) FROM clews WHERE embedding IS NOT NULL').fetchone()[0] == 1
    with context.write_session():
        assert asyncio.run(embed_clews(library, source_name='alpha')) == 1
    revision = corpus_revision(context.db_path)
    assert asyncio.run(questioner(None, 'alpha', True, None)) == 0
    assert corpus_revision(context.db_path) == revision
    payloads = [payload for path, payload, _ in chat.requests if path.endswith('/chat/completions')]
    assert len(payloads) == 2 and all(p['model'] == 'Qwen/Qwen3.8-27B' for p in payloads)
    assert any(text.startswith(QUESTION + '\n') for path, payload, _ in embed.requests
               if path.endswith('/embeddings') for text in payload['input'])
    with context.write_session():
        with pytest.raises(LibraryBusy):
            asyncio.run(questioner(None, 'alpha', True, None))
