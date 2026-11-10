"""Native Mac protocol acceptance; deterministic responses do not qualify model accuracy."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from config import Config
from docgen.catalog_enrich import enrich_file
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.validator import ContentValidator
from inference import InferenceContext
from local_models import (
    GEMMA_CHAT_MODEL, MAC_ACCURACY_CHAT_MODEL,
    gemma_model_profile, mac_accuracy_profile, mac_model_profile,
)
from tests.local_model_server import DOCUMENT
from writer import LibraryWriter

DOC_TYPES = ('explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram')
pytestmark = pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)
SOURCE = 'def add(a: int, b: int) -> int:\n    """Add two integers."""\n    return a + b\n'


def chat_response(content):
    return {'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 4, 'completion_tokens': 200}}


def configure(native_pair, tmp_path, monkeypatch, *, sources=None):
    context, chat, embed = native_pair
    data = {GEMMA_CHAT_MODEL: gemma_model_profile,
            MAC_ACCURACY_CHAT_MODEL: mac_accuracy_profile}.get(
                context.settings.chat.model, mac_model_profile)()
    data['llm']['base_url'] = chat.base_url
    data['embeddings']['base_url'] = embed.base_url
    source = tmp_path / 'alpha'
    source.mkdir(exist_ok=True)
    (source / 'core.py').write_text(SOURCE)
    data.update(db_path=str(context.db_path), staleness_db_path=str(context.staleness_db_path),
                default_source='alpha', themes_enabled=False,
                sources=sources or {'alpha': {'path': str(source)}})
    for entry in data['sources'].values():
        Path(entry['path']).mkdir(exist_ok=True)
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(data))
    cfg = Config(path)
    monkeypatch.setattr('config._global_config', cfg)
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-used')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://unexpected.invalid/v1')
    return cfg


def assert_native_vector(vector):
    assert vector.shape == (3072,) and vector.dtype == np.float32
    assert np.linalg.norm(vector.astype(np.float64)) == pytest.approx(1)


@pytest.mark.parametrize('doc_type', DOC_TYPES)
async def test_native_generator_persists_each_supported_document_type(
        native_pair, tmp_path, monkeypatch, doc_type):
    context, chat, _embed = native_pair
    cfg = configure(native_pair, tmp_path, monkeypatch)
    path = cfg.get_source_path('alpha') / 'core.py'
    body = DOCUMENT + '\n## How It Works\n\nInteger addition computes the result.\n'
    if doc_type == 'diagram':
        body += '\n```dot\ndigraph Arithmetic { caller -> add; add -> result; }\n```\n'
    elif doc_type == 'architecture':
        body += '\n## Design\n\nThe caller supplies inputs to a pure arithmetic function.\n'
    elif doc_type == 'qa':
        body += '\n## Question\n\nHow are inputs combined?\n\n## Answer\n\nThe function adds them.\n'
    chat.overrides['/native/v1/chat/completions'] = chat_response(body)
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            generator = DocGenerator(config=GeneratorConfig(
                model=context.settings.chat.model, provider='local', inference_context=context))
            async with generator:
                generated = await generator.generate_from_elements(
                    enrich_file(path, source_root=path.parent), doc_types=(doc_type,))
            assert len(generated) == 1
            result = generated[0]
            assert result.doc_type == doc_type and result.content == body
            assert result.source_snapshot == hashlib.sha256(path.read_bytes()).hexdigest()
            assert ContentValidator().validate(result.content, doc_type).is_valid
            stored = await LibraryWriter(library).add_document(
                result.doc_type, result.title, result.content, list(result.source_files),
                metadata=result.metadata, source_name='alpha')
            assert stored.metadata['model_provenance'] == {
                'version': 1, 'profile_hash': context.settings.profile_hash, 'kind': 'generated'}
            children = [*library.get_chunks(stored.id), *library.get_sections(stored.id)]
            assert children
            for row in [stored, *children]:
                assert_native_vector(row.embedding)
        calls = [payload for route, payload in chat.requests
                 if route == '/native/v1/chat/completions']
        assert len(calls) == 1 and calls[0]['model'] == context.settings.chat.served_model
        assert any('core' in message['content'] and 'add' in message['content']
                   for message in calls[0]['messages'])
    finally:
        await context.aclose()
    reopened = InferenceContext(context.settings, context.db_path, context.staleness_db_path)
    try:
        with reopened.open_library(operation='inspect') as library:
            saved = library.get_document(stored.id)
            assert saved.content == body and saved.content_type == doc_type
            assert saved.source_name == 'alpha' and saved.source_files == [str(path)]
            assert saved.embedding.tobytes() == stored.embedding.tobytes()
    finally:
        await reopened.aclose()


async def test_native_catalog_sync_describe_and_repeat_preserve_elements(
        native_pair, tmp_path, monkeypatch):
    from docgen.catalog_describer import describe_source_elements
    from docgen.catalog_writer import sync_file_catalog

    context, chat, embed = native_pair
    cfg = configure(native_pair, tmp_path, monkeypatch)
    source = cfg.get_source_path('alpha')
    description = 'Adds the two integer arguments and returns their sum.'
    chat.overrides['/native/v1/chat/completions'] = chat_response(description)
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            writer = LibraryWriter(library)
            result = await sync_file_catalog(library, writer, 'alpha', source, source / 'core.py')
            assert result.added == 1
            elements = [doc for doc in library.list_documents('catalog')
                        if doc.metadata.get('kind') == 'element']
            assert len(elements) == 1 and elements[0].metadata['qualified_name'] == 'core.add'
            assert elements[0].source_name == 'alpha'
            assert_native_vector(elements[0].embedding)
            assert chat.requests == []
            summary = await describe_source_elements(library, writer, 'alpha')
            assert summary['described'] == 1 and summary['failed'] == 0
            described = library.get_document(elements[0].id)
            assert described.metadata['description'] == description
            assert described.content.endswith('Description: ' + description)
            before = described.updated_at, described.content, described.embedding.tobytes()
            requests = len(chat.requests), len(embed.requests)
            assert (await sync_file_catalog(
                library, writer, 'alpha', source, source / 'core.py')).unchanged == 1
            assert (await describe_source_elements(library, writer, 'alpha'))['described'] == 0
            saved = library.get_document(described.id)
            assert (saved.updated_at, saved.content, saved.embedding.tobytes()) == before
            assert (len(chat.requests), len(embed.requests)) == requests
    finally:
        await context.aclose()


@pytest.mark.parametrize('coherent', [True, False])
async def test_native_theme_summary_applies_coherence_and_reuses_finished_run(native_pair, coherent):
    from docgen.local_pipeline import plan_themes
    from docgen.spool import LocalExecutor, SpoolStore

    context, chat, embed = native_pair
    response = '# Arithmetic\n\nAdds the supplied integers.' if coherent else 'INCOHERENT'
    chat.overrides['/native/v1/chat/completions'] = chat_response(response)
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            member = await LibraryWriter(library).add_document(
                'catalog', 'core.add', 'Adds two integers.', source_name='alpha',
                metadata={'kind': 'element', 'description': 'Adds integers.'})
            theme = library.add_document(
                'theme', 'Pending theme', '(pending summarization)', metadata={'pending': True})
            library.add_theme(cluster_id='arithmetic', doc_id=theme.id, member_count=1,
                              resolution=1.0, summary_hash='')
            library.set_theme_members('arithmetic', [(member.id, 1.0)])
            store = SpoolStore(context, library)
            run_id = store.create(plan_themes(library), operation='theme-summary', scope={})
            before_embeddings = len(embed.requests)
            executor = LocalExecutor(store)
            assert (await executor.drive(run_id))['state'] == 'completed'
            summary = library.get_theme('arithmetic')
            assert summary.coherent is coherent and not summary.dirty and summary.summary_hash
            stored = library.get_document(theme.id)
            if coherent:
                assert stored.content == response and stored.title == 'Arithmetic'
                assert_native_vector(stored.embedding)
            else:
                assert stored.content == theme.content and stored.updated_at == theme.updated_at
                assert len(embed.requests) == before_embeddings
            before = stored.updated_at, stored.content
            requests = len(chat.requests), len(embed.requests)
            assert (await executor.drive(run_id))['applied'] == 1
            stored = library.get_document(theme.id)
            assert (stored.updated_at, stored.content) == before
            assert (len(chat.requests), len(embed.requests)) == requests
    finally:
        await context.aclose()


@pytest.mark.parametrize(('source', 'expected'), [
    ('alpha', {'alpha', 'shared'}),
    ('shared', {'alpha', 'shared', 'extension'}),
])
async def test_native_search_and_ask_honor_directional_dependency_closure(
        native_pair, tmp_path, monkeypatch, source, expected):
    from ariadne_mcp.service import AriadneService
    from model_config import QUERY_INSTRUCTION

    context, chat, embed = native_pair
    sources = {name: {'path': str(tmp_path / name)}
               for name in ('alpha', 'shared', 'extension', 'unrelated')}
    sources['alpha']['depends_on'] = ['shared']
    sources['extension']['depends_on'] = ['alpha']
    cfg = configure(native_pair, tmp_path, monkeypatch, sources=sources)
    assert cfg.scope_closure(source) == frozenset(expected)
    answer_text = 'The documented arithmetic operation adds its two integer arguments.'
    chat.overrides['/native/v1/chat/completions'] = chat_response(answer_text)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            writer = LibraryWriter(library)
            for name in sources:
                await writer.add_document(
                    'explanation', f'{name} arithmetic guide',
                    DOCUMENT.replace('Core arithmetic', f'{name} arithmetic guide'),
                    source_name=name, doc_id=name)
        result = await service.search(query='How does integer addition work?',
                                      source=source, sections_only=True)
        assert {document.source_name for document in result.documents} == expected
        assert all(document.sections for document in result.documents)
        answer = await service.ask('How does integer addition work?', source=source)
        assert answer.answer == answer_text
        assert set(answer.sources) == {f'{name} arithmetic guide' for name in expected}
        prompt = json.dumps([payload['messages'] for route, payload in chat.requests
                             if route == '/native/v1/chat/completions'])
        assert all(f'{name} arithmetic guide' in prompt for name in expected)
        assert 'unrelated arithmetic guide' not in prompt
        assert any(text.startswith(f'Instruct: {QUERY_INSTRUCTION}\nQuery:')
                   for route, payload in embed.requests if route == '/native/v1/embeddings'
                   for text in payload['input'])
    finally:
        await service.close()
        await context.aclose()


@pytest.mark.parametrize('mode', ['deterministic', 'llm'])
async def test_native_clew_reopen_reaches_evidence_walk_and_synthesis(
        native_pair, tmp_path, monkeypatch, mode):
    from ariadne_mcp.service import AriadneService
    from library.clew_embeddings import embed_clews
    from tests.clew_workflow import QUESTION, ROUTE, add_guide, build_routes, seed_graph

    context, chat, embed = native_pair
    cfg = configure(native_pair, tmp_path, monkeypatch)
    cfg._config['ask_selector_mode'] = mode
    handler_class = chat.server.RequestHandlerClass
    reply = handler_class.reply
    answer_text = 'payments.accept_payment calls authorize_payment, which calls record_receipt.'

    def scripted_reply(handler, status, body):
        if handler.path == '/native/v1/chat/completions' and status == 200:
            prompt = json.dumps(chat.requests[-1][1]['messages'])
            if 'C<n>: K<id>' in prompt:
                content = 'C1: K1'
            elif 'C<n>: S<id>' in prompt:
                content = 'C1: S1 S2 S3'
            elif 'IDs only' in prompt or 'IDs, not prose' in prompt:
                content = 'M1 R1 B1'
            else:
                content = answer_text
            body = chat_response(content)
        reply(handler, status, body)

    monkeypatch.setattr(handler_class, 'reply', scripted_reply)
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            seed_graph(library, cfg.get_source_path('alpha'), 'alpha')
            build_routes(library, 'alpha')
            assert await embed_clews(library, source_name='alpha') == 2
            await add_guide(library, 'alpha')
    finally:
        await context.aclose()

    reopened = InferenceContext(context.settings, context.db_path, context.staleness_db_path)
    service = AriadneService()
    service._config, service._inference_context = cfg, reopened
    try:
        with reopened.write_session(), reopened.open_library(operation='write') as library:
            with library._conn_provider.acquire() as connection:
                rows = connection.execute('SELECT embedding FROM clews').fetchall()
            assert len(rows) == 2
            for (raw,) in rows:
                assert_native_vector(np.frombuffer(raw, dtype=np.float32))
            before = len(embed.requests)
            assert await embed_clews(library, source_name='alpha') == 0
            assert len(embed.requests) == before
        answer = await service.ask(QUESTION, source='alpha')
        selected = answer.graph_diagnostics['clew_selection']
        assert selected['selected'] >= 1 and not selected.get('error'), selected
        assert any(set(ROUTE).issubset(route) for route in answer.route_candidates.values())
        # Compiler-backed Ask may repair/filter narration and append mandatory
        # proof; raw response equality would bypass that existing contract.
        assert answer.llm_calls >= 1 and 'synthesis unavailable' not in answer.answer
        assert all(symbol in answer.answer for symbol in ROUTE)
        assert answer.claims and answer.citations and not answer.unsupported_locations
        prompts = json.dumps([payload['messages'] for route, payload in chat.requests
                              if route == '/native/v1/chat/completions'])
        assert all(symbol in prompts for symbol in ROUTE)
        assert "return {'receipt': amount}" in prompts
        assert any(QUESTION in text and text.startswith('Instruct:')
                   for route, payload in embed.requests if route == '/native/v1/embeddings'
                   for text in payload['input'])
    finally:
        await service.close()
        await reopened.aclose()
