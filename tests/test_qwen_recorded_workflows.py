"""Replay unedited hosted Qwen responses through the local CLI and durable worker.

These are content/transport/recovery checks, not factuality or vLLM qualification.
Only chat responses are recorded; tokenization and embeddings use local HTTP fixtures.
"""

import asyncio
import hashlib
import json
import sqlite3
from pathlib import Path

import pytest

from ariadne_mcp.service import AriadneService
from docgen.spool import LocalExecutor, SpoolStore
from inference import InferenceContext
from library import Library
from tests.blind_model_cases import CASES
from tests.live_model_smoke import snapshot
from tests.local_model_server import deployments, trap_inference
from tests.test_local_maintenance import invoke
from tests.test_local_spool import document_spec
from tests.test_local_spool_recovery import ProcessCrash
from tests.test_local_workflows import fixture_config

RECORDINGS = Path(__file__).parent / 'fixtures/model_support/qwen-responses-v1.json'


def recorded_response(name, *, answer=False):
    """Return the unedited provider response retained as a regression input."""
    return json.loads(RECORDINGS.read_text())[name]['answer' if answer else 'generation']


@pytest.mark.parametrize('batch', [False, True], ids=['live', 'durable'])
@pytest.mark.parametrize('case', CASES, ids=lambda case: case[0])
def test_recorded_generation_update_and_mcp(tmp_path, monkeypatch, batch, case):
    import config

    name, doc_type, source, question = case
    with deployments() as (chat, embed):
        trap_inference(monkeypatch, [chat.base_url, embed.base_url])
        # Credentials must not cause replay to escape the loopback endpoints.
        monkeypatch.setenv('OPENAI_API_KEY', 'unused-hosted-key')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'unused-hosted-key')
        cfg = fixture_config(tmp_path, chat, embed)
        path = cfg.get_source_path('alpha') / 'core.py'
        path.write_text(source)
        monkeypatch.setattr(config, '_global_config', cfg)
        chat.chat_response = recorded_response(name)
        content = chat.chat_response['choices'][0]['message']['content']
        arguments = (
            'generate',
            '--source',
            'alpha',
            '--path',
            'core.py',
            '--types',
            doc_type,
            '--batch' if batch else '--no-batch',
            '--no-crossrefs',
            '--yes',
        )
        result = invoke(monkeypatch, *arguments)
        if name == 'routing':
            # This historical response uses Mermaid. The public diagram
            # contract now requires DOT; retain the raw response and verify
            # that rejected model output never reaches the document store.
            assert result == 1
            with Library(Path(cfg.db_path)) as library:
                assert library.list_documents(doc_type) == []
            if batch:
                with sqlite3.connect(cfg.db_path) as conn:
                    (job,) = conn.execute(
                        'SELECT state, failure_stage, response_text FROM spool_jobs'
                    ).fetchall()
                assert job == ('failed', 'validation', content)
            assert not any(route.endswith('/embeddings') for route, _, _ in embed.requests)
            return
        assert result == 0
        with Library(Path(cfg.db_path)) as library:
            (doc,) = library.list_documents(doc_type)
            assert doc.content == content
            assert doc.source_name == 'alpha' and doc.source_files == [str(path)]
            before = snapshot(library, doc.id)
        requests = len(chat.requests), len(embed.requests)
        assert invoke(monkeypatch, *arguments) == 0
        assert requests == (len(chat.requests), len(embed.requests))

        async def retrieve():
            service = AriadneService()
            service._config = cfg
            try:
                result = await service.search(query=question, source='alpha', sections_only=True)
                (selected,) = result.documents
                assert selected.id == doc.id
                with Library(Path(cfg.db_path)) as library:
                    sections = library.get_sections(doc.id)
                    if sections:
                        assert selected.sections
                        assert all(s.content in content for s in selected.sections)
                    else:
                        assert selected.content == content
                chat.chat_response = recorded_response(name, answer=True)
                answer = await service.ask(question, source='alpha')
                assert answer.answer == chat.chat_response['choices'][0]['message']['content']
                assert answer.sources == [doc.title]
            finally:
                chat.chat_response = recorded_response(name)
                await service.close()

        asyncio.run(retrieve())
        # A source-only edit must update the processed hash even when the model
        # returns byte-identical prose. This exercises incremental staleness.
        changed_source = source + '\n# Formatting-only revision: café / שלום.\n'
        path.write_text(changed_source)
        assert invoke(monkeypatch, *arguments) == 0
        requests = len(chat.requests), len(embed.requests)
        assert invoke(monkeypatch, *arguments) == 0
        assert requests == (len(chat.requests), len(embed.requests))
        with Library(Path(cfg.db_path)) as library:
            after = snapshot(library, doc.id)
            assert after[0] == before[0] and after[2:] == before[2:]
        with sqlite3.connect(cfg.staleness_db_path) as connection:
            hashes = [row[0] for row in connection.execute('SELECT hash FROM source_records')]
            assert hashes == [hashlib.sha256(changed_source.encode()).hexdigest()]
        assert all('Authorization' not in h for _, _, h in chat.requests + embed.requests)


@pytest.mark.parametrize('name', ['ledger', 'text'])
@pytest.mark.asyncio
async def test_recorded_response_survives_restart_and_embedding_failure(
    local_deployment, monkeypatch, name
):
    context, chat, embed = local_deployment
    chat.chat_response = recorded_response(name)
    content = chat.chat_response['choices'][0]['message']['content']
    with context.write_session():
        library = context.open_library(operation='write')
        store = SpoolStore(context, library)
        run_id = store.create([document_spec()], operation='generate', scope={})
        original = store.update_job

        def interrupt(job_id, **values):
            original(job_id, **values)
            if values.get('state') == 'response_ready':
                raise ProcessCrash

        monkeypatch.setattr(store, 'update_job', interrupt)
        with pytest.raises(ProcessCrash):
            await LocalExecutor(store).drive(run_id)
        assert store.jobs(run_id)[0]['response_text'] == content
        assert library.get_document('doc') is None
    await context.aclose()
    # A new context must use the saved response even while chat is unavailable.
    chat.failures = [400]
    embed.failures = [503]
    resumed = InferenceContext(context.settings, context.db_path, context.staleness_db_path)
    try:
        with resumed.write_session():
            library = resumed.open_library(operation='write')
            store = SpoolStore(resumed, library)
            assert (await LocalExecutor(store).drive(run_id))['state'] == 'completed'
            (job,) = store.jobs(run_id)
            assert job['request_attempts'] == 1 and job['apply_attempts'] == 2
            assert library.get_document('doc').content == content
            assert library.get_chunks('doc') and library.get_sections('doc')
            before, requests = snapshot(library, 'doc'), (len(chat.requests), len(embed.requests))
            await LocalExecutor(store).drive(run_id)
            assert snapshot(library, 'doc') == before
            assert requests == (len(chat.requests), len(embed.requests))
            assert chat.failures == [400]
    finally:
        await resumed.aclose()
