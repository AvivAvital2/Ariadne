import sqlite3
from dataclasses import replace

import numpy as np
import pytest
import yaml

from config import Config
from docgen.llm.local import EmbeddingContractError
from docgen.staleness import StalenessTracker
from inference import InferenceContext
from library.model_binding import ModelBindingMismatch
from local_models import mac_model_profile
from model_config import resolve_models
from tests.local_model_server import DOCUMENT
from web.projects import ProjectStore
from writer import LibraryWriter


@pytest.mark.asyncio
async def test_native_prefix_normalization_and_stored_document_children(native_pair):
    context, chat, embed = native_pair
    try:
        # Deliberately reverse response rows and make discarded components large.
        first = list(range(1, 3073)) + [1000000] * 1024
        second = list(range(3072, 0, -1)) + [2000000] * 1024
        embed.overrides['/native/v1/embeddings'] = {'data': [
            {'index': 1, 'embedding': second}, {'index': 0, 'embedding': first}]}
        vectors = await context.embeddings.embed_documents(['first', 'second'])
        for vector, values in zip(vectors, [first, second], strict=True):
            expected = np.array(values[:3072], dtype=np.float64)
            expected /= np.linalg.norm(expected)
            np.testing.assert_allclose(vector, expected.astype(np.float32))
            assert vector.dtype == np.float32 and vector.nbytes == 12288
        embed.overrides.clear()
        with context.write_session():
            with context.open_library(operation='write') as library:
                document = await LibraryWriter(library).add_document(
                    'explanation', 'Arithmetic', DOCUMENT, ['core.py'], source_name='alpha')
                chunks, sections = library.get_chunks(document.id), library.get_sections(document.id)
                assert len(chunks) > 1 and len(sections) >= 3
                for record in [document, *chunks, *sections]:
                    assert record.embedding.shape == (3072,)
                    assert record.embedding.dtype == np.float32
                    assert np.linalg.norm(record.embedding.astype(np.float64)) == pytest.approx(1)
        with sqlite3.connect(context.db_path) as connection:
            for table in ('documents', 'chunks', 'sections'):
                assert connection.execute(
                    f'SELECT DISTINCT length(embedding) FROM {table}').fetchall() == [(12288,)]
        assert chat.requests == []
    finally:
        await context.aclose()


@pytest.mark.parametrize('invalid', ['wrong_width', 'nonfinite', 'zero_prefix'])
@pytest.mark.asyncio
async def test_bad_native_embedding_preserves_existing_knowledge(native_pair, invalid):
    context, _chat, embed = native_pair
    try:
        with context.write_session():
            with context.open_library(operation='write') as library:
                writer = LibraryWriter(library)
                original = await writer.add_document('explanation', 'Arithmetic', DOCUMENT)
                before = library.get_document(original.id)
                values = [1.0] * 4096
                if invalid == 'wrong_width':
                    values.pop()
                elif invalid == 'nonfinite':
                    values[0] = float('nan')
                else:
                    values[:3072] = [0.0] * 3072
                embed.overrides['/native/v1/embeddings'] = {'data': [{'index': 0, 'embedding': values}]}
                with pytest.raises(EmbeddingContractError):
                    await writer.replace_document(original.id, content='New explanation.')
                after = library.get_document(original.id)
                assert after.content == before.content
                assert after.updated_at == before.updated_at
                assert after.embedding.tobytes() == before.embedding.tobytes()
    finally:
        await context.aclose()


@pytest.mark.asyncio
async def test_mac_library_and_sidecar_reopen_with_exact_pair_and_preserve_knowledge(native_pair, tmp_path):
    context, _chat, _embed = native_pair
    with context.write_session():
        with context.open_library(operation='write') as library:
            document = await LibraryWriter(library).add_document(
                'explanation', 'Arithmetic', DOCUMENT, source_name='alpha')
        with StalenessTracker(context.staleness_db_path, inference_context=context,
                              source_name='alpha', generation_scope='all') as tracker:
            tracker.record_documentation(tmp_path / 'core.py', [document.id], tmp_path,
                                         processed_hash='source-hash')
    await context.aclose()
    reopened = InferenceContext(context.settings, context.db_path, context.staleness_db_path)
    try:
        with reopened.open_library(operation='inspect') as library:
            saved = library.get_document(document.id)
            assert saved.content == document.content
            assert saved.embedding.tobytes() == document.embedding.tobytes()
        with sqlite3.connect(context.staleness_db_path) as connection:
            assert connection.execute('SELECT source_name,hash FROM source_records').fetchall() == [
                ('alpha', 'source-hash')]
        before = context.db_path.read_bytes(), context.staleness_db_path.read_bytes()
        for changed in (
            replace(context.settings, chat=replace(context.settings.chat, model_revision='f' * 40)),
            replace(context.settings, embeddings=replace(context.settings.embeddings, model_revision='f' * 40)),
        ):
            with pytest.raises(ModelBindingMismatch):
                InferenceContext(changed, context.db_path, context.staleness_db_path)
        with pytest.raises(ModelBindingMismatch, match='ownership'):
            InferenceContext(context.settings, tmp_path / 'different.db', context.staleness_db_path)
        assert (context.db_path.read_bytes(), context.staleness_db_path.read_bytes()) == before
    finally:
        await reopened.aclose()


def test_mac_project_uses_fresh_pair_paths_and_retains_dependency_configuration(tmp_path):
    from tests.local_model_server import local_config

    app, shared = tmp_path / 'app', tmp_path / 'shared'
    app.mkdir()
    shared.mkdir()
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump({
        'sources': {'app': {'path': str(app), 'depends_on': ['shared']},
                    'shared': {'path': str(shared)}},
        'onboarding_models': {'mac': mac_model_profile(), 'bf16': local_config()},
    }))
    store = ProjectStore(Config(path))
    native = store.prepare('mac', 'app')
    original = store.prepare('bf16', 'app')
    saved = Config(store.path(native['id']))
    other = Config(store.path(original['id']))
    assert native['id'] != original['id']
    assert saved.db_path != other.db_path
    assert saved.staleness_db_path != other.staleness_db_path
    assert saved.model_settings().identity == resolve_models(mac_model_profile()).identity
    assert saved.sources['app']['depends_on'] == ['shared']
    assert saved.scope_closure('app') == frozenset({'app', 'shared'})
    restarted = ProjectStore(Config(path))
    assert restarted.describe(native['id']) == native
