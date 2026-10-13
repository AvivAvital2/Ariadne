import sqlite3

import pytest

from tests.local_model_server import DOCUMENT
from writer import LibraryWriter


@pytest.mark.asyncio
async def test_atomic_bundle_rebuild_and_replay(local_deployment):
    ctx, _, embed = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        writer = LibraryWriter(library)
        doc = await writer.add_document(
            'explanation',
            'Arithmetic',
            DOCUMENT,
            ['core.py'],
            doc_id='fixed-id',
            source_name='alpha',
        )
        assert len(library.get_chunks(doc.id)) > 1
        assert len(library.get_sections(doc.id)) >= 3
        before = len(embed.requests)
        repeated = await writer.add_document(
            'explanation',
            'Arithmetic',
            DOCUMENT,
            ['core.py'],
            doc_id='fixed-id',
            source_name='alpha',
        )
        assert repeated.updated_at == doc.updated_at
        assert len(embed.requests) == before
        await writer.replace_document(doc.id, content='A short arithmetic explanation.')
        assert library.get_sections(doc.id) == []
        assert library.get_chunks(doc.id) == []
        assert await writer.rebuild_all_embeddings() == 1
        assert library.get_document(doc.id).source_name == 'alpha'
    await ctx.aclose()


@pytest.mark.asyncio
async def test_failure_between_child_writes_rolls_back(local_deployment):
    ctx, _, _ = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        writer = LibraryWriter(library)
        doc = await writer.add_document('explanation', 'Arithmetic', DOCUMENT, doc_id='fixed-id')
        before = library.get_document(doc.id)
        with library._conn_provider.acquire() as conn:
            conn.execute(
                "CREATE TRIGGER fail_sections BEFORE INSERT ON sections BEGIN SELECT RAISE(ABORT, 'injected section failure'); END"
            )
        with pytest.raises(sqlite3.IntegrityError, match='injected section failure'):
            await writer.replace_document(doc.id, content=DOCUMENT + '\n## More\nExtra section')
        after = library.get_document(doc.id)
        assert after.updated_at == before.updated_at and after.content == before.content
        assert after.embedding.tobytes() == before.embedding.tobytes()
    await ctx.aclose()
