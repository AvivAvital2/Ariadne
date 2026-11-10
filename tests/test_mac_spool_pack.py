"""Pack portability uses the bound embedding recipe and complete stored records."""
import asyncio
import json
import sqlite3
import zipfile
from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml
from attrs import evolve

from embedding import doc_embedding_text
from inference import InferenceContext
from library import Library
from library.prepared import PreparedDocument
from schema import Section
from spool_pack import build_pack, install_pack
from spools import SpoolError
from tests.local_model_server import DOCUMENT
from writer import LibraryWriter


def _build(library, tmp_path, name='pack.zip'):
    return build_pack(library, environment='corpus', version='1', target_runtime='fixture',
                      source_root=tmp_path, out_path=tmp_path / name)


def _records(library, doc_id):
    return [library.get_document(doc_id), *library.get_chunks(doc_id), *library.get_sections(doc_id)]


def _rewrite_manifest(path, update):
    with zipfile.ZipFile(path) as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    manifest = yaml.safe_load(members['manifest.yaml'])
    update(manifest)
    members['manifest.yaml'] = yaml.safe_dump(manifest).encode()
    with zipfile.ZipFile(path, 'w') as archive:
        for name, content in members.items():
            archive.writestr(name, content)


async def test_bound_pack_records_actual_space_and_preserves_all_vectors_offline(native_pair, tmp_path):
    context, chat, embed = native_pair
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            document = await LibraryWriter(library).add_document(
                'explanation', 'Arithmetic', DOCUMENT, ['core.py'], source_name='corpus')
            expected = _records(library, document.id)
            assert len(library.get_chunks(document.id)) > 1
            assert len(library.get_sections(document.id)) >= 3
            before = len(embed.requests)
            manifest = _build(library, tmp_path)
        assert manifest.embedding_model == context.settings.embeddings.model
        assert manifest.embedding_dim == 3072
        assert manifest.embedding_identity == context.settings.embeddings.identity
        assert len(embed.requests) == before and chat.requests == []
        with zipfile.ZipFile(tmp_path / 'pack.zip') as archive:
            path = tmp_path / 'inspected.db'
            path.write_bytes(archive.read('pack.db'))
        with Library(path, initialize=False, read_only=True) as packed:
            actual = _records(packed, document.id)
        assert len(actual) == len(expected)
        for original, shipped in zip(expected, actual, strict=True):
            assert shipped.content == original.content
            assert shipped.embedding.tobytes() == original.embedding.tobytes()
        with sqlite3.connect(path) as connection:
            assert json.loads(connection.execute(
                'SELECT metadata FROM documents WHERE id=?', (document.id,)).fetchone()[0]) == document.metadata
        np.testing.assert_array_equal(actual[0].embedding, document.embedding)
    finally:
        await context.aclose()


async def test_matching_space_installs_without_inference_and_preserves_children(native_pair, tmp_path):
    context, chat, embed = native_pair
    target = InferenceContext(replace(context.settings, chat=replace(
        context.settings.chat, model_revision='f' * 40)),
        tmp_path / 'consumer.db', tmp_path / 'consumer-staleness.db')
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            document = await LibraryWriter(library).add_document(
                'explanation', 'Arithmetic', DOCUMENT, ['core.py'], source_name='corpus')
            expected = _records(library, document.id)
            _build(library, tmp_path)
        before = len(embed.requests)
        with target.write_session(), target.open_library(operation='write') as library:
            await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip',
                                    cache_dir=tmp_path / 'cache')
            actual = _records(library, document.id)
            assert actual[0].source_name == 'spool:corpus'
            assert actual[0].metadata == document.metadata
            assert len(actual) == len(expected)
            for original, installed in zip(expected, actual, strict=True):
                assert installed.content == original.content
                assert installed.embedding.tobytes() == original.embedding.tobytes()
            before_repeat = actual[0].updated_at
            await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip',
                                    cache_dir=tmp_path / 'cache')
            assert library.get_document(document.id).updated_at == before_repeat
        assert len(embed.requests) == before and chat.requests == []
    finally:
        await context.aclose()
        await target.aclose()


def _snapshot(library):
    with library._conn_provider.acquire() as connection:
        return tuple(connection.iterdump())


@pytest.mark.parametrize('failure', ['metadata', 'cache_copy', 'cache_publish'])
async def test_failed_reinstall_restores_existing_documents_children_metadata_and_cache(
        native_pair, tmp_path, monkeypatch, failure):
    import spool_pack

    context, _chat, _embed = native_pair
    target = InferenceContext(context.settings, tmp_path / 'consumer.db', tmp_path / 'consumer-staleness.db')
    try:
        with context.write_session(), context.open_library(operation='write') as builder:
            writer = LibraryWriter(builder)
            document = await writer.add_document('explanation', 'Arithmetic', DOCUMENT,
                                                 metadata={'edition': 'old'}, source_name='corpus')
            _build(builder, tmp_path)
            await writer.replace_document(document.id, content=DOCUMENT + '\nChanged behavior.',
                                          metadata={'edition': 'new'})
            await writer.add_document('explanation', 'Additional', 'New document.', source_name='corpus')
            _build(builder, tmp_path, 'new-pack.zip')
        cache = tmp_path / 'cache'
        with target.write_session(), target.open_library(operation='write') as library:
            await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip', cache_dir=cache)
            before = _snapshot(library)
            cache_before = {path.name: path.read_bytes() for path in (cache / 'corpus').iterdir()}
            copy_surface_tags = spool_pack._copy_surface_tags

            def fail_after_metadata(source, destination, source_name):
                copy_surface_tags(source, destination, source_name)
                raise RuntimeError('Interrupted after derived metadata')

            if failure == 'metadata':
                monkeypatch.setattr(spool_pack, '_copy_surface_tags', fail_after_metadata)
            elif failure == 'cache_copy':
                copyfile = spool_pack.shutil.copyfile

                def fail_cache_copy(source, destination, **kwargs):
                    if Path(destination).name == 'pack.db' and cache in Path(destination).parents:
                        raise RuntimeError('Interrupted cache copy')
                    return copyfile(source, destination, **kwargs)

                monkeypatch.setattr(spool_pack.shutil, 'copyfile', fail_cache_copy)
            else:
                replace_path = Path.replace

                def fail_publish(path, destination):
                    if path.name == 'new' and Path(destination) == cache / 'corpus':
                        raise RuntimeError('Interrupted cache publish')
                    return replace_path(path, destination)

                monkeypatch.setattr(Path, 'replace', fail_publish)
            with pytest.raises(RuntimeError, match='Interrupted'):
                await asyncio.to_thread(install_pack, library, tmp_path / 'new-pack.zip', cache_dir=cache)
            assert _snapshot(library) == before
            assert {path.name: path.read_bytes() for path in (cache / 'corpus').iterdir()} == cache_before
    finally:
        await context.aclose()
        await target.aclose()


async def test_pack_cannot_overwrite_an_unrelated_source_document(native_pair, tmp_path):
    from spools import SpoolError

    context, _chat, _embed = native_pair
    target = InferenceContext(context.settings, tmp_path / 'consumer.db', tmp_path / 'consumer-staleness.db')
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            document = await LibraryWriter(library).add_document(
                'explanation', 'Arithmetic', DOCUMENT, source_name='corpus')
            _build(library, tmp_path)
        with target.write_session(), target.open_library(operation='write') as library:
            library.apply_prepared_document(PreparedDocument(
                document=evolve(document, source_name='my-project', metadata={'owner': 'user'}),
                embedding_space_id=target.settings.embeddings.embedding_space_id))
            before = _snapshot(library)
            with pytest.raises(SpoolError, match='belongs to another source'):
                await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip',
                                        cache_dir=tmp_path / 'cache')
            assert _snapshot(library) == before
        assert not (tmp_path / 'cache').exists()
    finally:
        await context.aclose()
        await target.aclose()


async def test_legacy_pack_with_only_section_vectors_is_reembedded_for_local_use(native_pair, tmp_path):
    context, _chat, embed = native_pair
    try:
        vector = np.zeros(3072, dtype=np.float32)
        vector[0] = 1
        with Library(tmp_path / 'legacy.db') as builder:
            document = builder.add_document('explanation', 'Legacy', 'Section body.', source_name='corpus')
            builder.store_sections(document.id, [Section(
                document_id=document.id, index=0, heading='Legacy heading', description='Description',
                content='Section body.', embedding=vector)])
            manifest = _build(builder, tmp_path)
        assert manifest.embedding_dim == 3072
        assert manifest.embedding_identity is None
        # Older packs did not inspect child vectors when stamping dimensions.
        _rewrite_manifest(tmp_path / 'pack.zip', lambda data: data.update(embedding_dim=None))
        embed.overrides['/native/v1/embeddings'] = {'data': [
            {'index': 0, 'embedding': [0.0, 1.0] + [0.0] * 4094}]}
        with context.write_session(), context.open_library(operation='write') as library:
            await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip',
                                    cache_dir=tmp_path / 'cache')
            assert library.get_document(document.id).embedding is None
            section = library.get_sections(document.id)[0]
            assert section.embedding[0] == 0 and section.embedding[1] == 1
        sent = [body['input'] for path, body in embed.requests if path == '/native/v1/embeddings']
        assert sent == [['Legacy heading']]
    finally:
        await context.aclose()


@pytest.mark.parametrize('identity', [[], 'not-an-identity', {}, {'version': 1, 'dimensions': 3072}])
async def test_malformed_embedding_identity_fails_before_destination_mutation(
        native_pair, tmp_path, identity):
    context, _chat, embed = native_pair
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            await LibraryWriter(library).add_document('explanation', 'Arithmetic', DOCUMENT, source_name='corpus')
            _build(library, tmp_path)
            _rewrite_manifest(tmp_path / 'pack.zip', lambda data: data.update(embedding_identity=identity))
            before = _snapshot(library)
            before_requests = len(embed.requests)
            with pytest.raises(SpoolError, match='embedding identity'):
                await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip',
                                        cache_dir=tmp_path / 'cache')
            assert _snapshot(library) == before
            assert len(embed.requests) == before_requests
        assert not (tmp_path / 'cache').exists()
    finally:
        await context.aclose()


@pytest.mark.parametrize('changed', ['model_revision', 'tokenizer_revision'])
async def test_identity_mismatch_reembeds_every_record_including_theme_children(
        native_pair, tmp_path, changed):
    context, chat, embed = native_pair
    target = InferenceContext(replace(context.settings, embeddings=replace(
        context.settings.embeddings, **{changed: 'f' * 40}, batch_size=1)),
        tmp_path / 'consumer.db', tmp_path / 'consumer-staleness.db')
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            writer = LibraryWriter(library)
            document = await writer.add_document(
                'explanation', 'Arithmetic', DOCUMENT, source_name='corpus')
            theme = await writer.add_document('theme', 'Arithmetic patterns', DOCUMENT)
            library.add_theme(cluster_id='corpus-theme', doc_id=theme.id, member_count=1,
                              resolution=1, summary_hash='fixture', association='spool:corpus')
            library.set_theme_members('corpus-theme', [(document.id, 1)])
            expected_inputs = []
            for doc in (document, theme):
                expected_inputs.append(doc_embedding_text(doc.title, doc.content))
                expected_inputs.extend(chunk.content for chunk in library.get_chunks(doc.id))
                expected_inputs.extend(section.heading or 'Introduction' for section in library.get_sections(doc.id))
            _build(library, tmp_path)
        embed.requests.clear()
        native = [0.0, 0.0, 1.0] + [0.0] * 4093
        embed.overrides['/native/v1/embeddings'] = {'data': [{'index': 0, 'embedding': native}]}
        with target.write_session(), target.open_library(operation='write') as library:
            await asyncio.to_thread(install_pack, library, tmp_path / 'pack.zip',
                                    cache_dir=tmp_path / 'cache')
            for doc in (document, theme):
                for record in _records(library, doc.id):
                    np.testing.assert_array_equal(record.embedding, np.array(native[:3072], dtype=np.float32))
            assert library.get_document(document.id).source_name == 'spool:corpus'
            assert library.get_document(theme.id).source_name is None
            assert library.get_theme_members('corpus-theme') == [(document.id, 1.0)]
        sent = [text for path, body in embed.requests if path == '/native/v1/embeddings'
                for text in body['input']]
        assert Counter(sent) == Counter(expected_inputs)
        assert not chat.requests
    finally:
        await context.aclose()
        await target.aclose()
