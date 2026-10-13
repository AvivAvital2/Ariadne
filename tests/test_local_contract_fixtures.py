"""Compatibility checks for the design's checked-in, versioned handoff examples."""

import json
import shutil
import sqlite3
from pathlib import Path

import pytest

from docgen.llm.batch import BatchRequest
from docgen.spool import JobKind, LocalExecutor, SpoolJobSpec, SpoolStore
from docgen.staleness import StalenessTracker
from inference import InferenceContext
from library.imports import import_bound_markdown, read_bound_export
from library.model_binding import ModelBindingMismatch
from model_config import canonical_json, resolve_models

FIXTURES = Path(__file__).parent / 'fixtures' / 'model_support'
CONTRACT = json.loads((FIXTURES / 'contract-v1.json').read_text())


def test_frozen_profile_serialization_and_job_payloads(tmp_path):
    settings = resolve_models(CONTRACT['config'], execution=True)
    assert settings.identity == CONTRACT['profile']
    assert canonical_json(settings.identity) == CONTRACT['canonical_profile_utf8']
    assert settings.profile_hash == CONTRACT['profile_hash']
    assert settings.embeddings.embedding_space_id == CONTRACT['embedding_space_id']
    context = InferenceContext(settings, tmp_path / 'library.db', tmp_path / 'staleness.db')
    specs = []
    for example in CONTRACT['jobs']:
        payload = example['payload']
        request = payload['request']
        specs.append(
            SpoolJobSpec(
                example['custom_id'],
                JobKind(payload['kind']),
                payload['target_id'],
                payload['data'],
                request=BatchRequest(
                    example['custom_id'],
                    request['messages'][0]['content'],
                    request['messages'][1]['content'],
                    max_tokens=request['max_tokens'],
                )
                if request
                else None,
                snapshots=payload['snapshots'],
                context_versions=payload['context_versions'],
                expected_version=payload['expected_version'],
                phase=payload['phase'],
            )
        )
    with context.write_session():
        library = context.open_library(operation='write')
        store = SpoolStore(context, library)
        run = store.create(specs, operation='contract', scope={'source_name': 'alpha'})
        executor = LocalExecutor(store)
        for saved, example in zip(store.jobs(run), CONTRACT['jobs'], strict=True):
            assert json.loads(saved['payload_json']) == example['payload']
            assert saved['request_hash'] == example['request_hash']
            if example['response'] is not None:
                assert (
                    executor.parse(example['payload'], example['response'])
                    == example['parsed_result']
                )
            else:
                assert json.loads(saved['parsed_result_json']) == example['parsed_result']
                assert saved['state'] == 'apply_pending'
        library.close()


def test_frozen_namespaced_sidecar_records(tmp_path):
    context = InferenceContext(
        resolve_models(CONTRACT['config']), tmp_path / 'library.db', tmp_path / 'staleness.db'
    )
    with context.write_session():
        for row in CONTRACT['sidecar_records']:
            with StalenessTracker(
                context.staleness_db_path,
                inference_context=context,
                source_name=row['source_name'],
                generation_scope=row['generation_scope'],
            ) as tracker:
                tracker.record_documentation(
                    tmp_path / row['path'], row['doc_ids'], tmp_path, processed_hash=row['hash']
                )
        with sqlite3.connect(context.staleness_db_path) as conn:
            conn.row_factory = sqlite3.Row
            saved = [
                dict(row)
                for row in conn.execute(
                    'SELECT source_name,generation_scope,path,hash,doc_ids FROM source_records'
                )
            ]
        for row in saved:
            row['doc_ids'] = json.loads(row['doc_ids'])
        assert sorted(saved, key=canonical_json) == sorted(
            CONTRACT['sidecar_records'], key=canonical_json
        )


@pytest.mark.asyncio
async def test_frozen_chat_and_embedding_wire_examples(local_deployment):
    context, chat, embeddings = local_deployment
    wire = CONTRACT['requests']
    with context.write_session():
        await context.chat.call('Document the source.', 'Explain arithmetic.', max_tokens=8192)
        await context.embeddings.embed_documents(wire['document_embedding']['input'])
        await context.embeddings.embed_query('How does add work?')
    requests = {route: payload for route, payload, _ in chat.requests}
    assert requests[wire['paths']['chat']] == wire['chat']
    assert requests[wire['paths']['chat_tokenize']] == wire['chat_tokenize']
    embedding_calls = [
        payload for route, payload, _ in embeddings.requests if route.endswith('/embeddings')
    ]
    assert embedding_calls == [wire['document_embedding'], wire['query_embedding']]
    assert embeddings.requests[-2][1] == wire['query_tokenize']
    assert all(
        'Authorization' not in headers for _, _, headers in chat.requests + embeddings.requests
    )
    await context.aclose()


@pytest.mark.parametrize('invalid', ['incompatible', 'unidentified'])
@pytest.mark.asyncio
async def test_frozen_export_provenance_preflights_all_inputs(tmp_path, invalid):
    context = InferenceContext(
        resolve_models(CONTRACT['config']), tmp_path / 'library.db', tmp_path / 'staleness.db'
    )
    incoming = tmp_path / 'incoming'
    incoming.mkdir()
    for name in ('generated', 'authored'):
        source = FIXTURES / 'exports-v1' / (name + '.md')
        doc = read_bound_export(source, context)
        assert doc.metadata['model_provenance']['kind'] == name
        assert doc.metadata['nested'] == {'items': ['one', 'two: value'], 'multiline': 'one\ntwo'}
        shutil.copyfile(source, incoming / (name + '.md'))
    shutil.copyfile(FIXTURES / 'exports-v1' / (invalid + '.md'), incoming / 'z-invalid.md')
    with context.write_session():
        library = context.open_library(operation='write')
        with pytest.raises(ModelBindingMismatch):
            await import_bound_markdown(library, incoming, embed=False)
        assert library.count_documents() == 0
        library.close()
    await context.aclose()
