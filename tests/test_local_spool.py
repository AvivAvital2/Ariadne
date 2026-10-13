from unittest.mock import patch

import pytest

from docgen.llm.batch import BatchRequest
from docgen.llm.local_batch import LocalBatchStrategy
from docgen.spool import JobKind, LocalExecutor, SpoolJobSpec, SpoolStore, UnsupportedSpoolVersion
from model_config import identity_hash


def document_spec(custom_id='doc', target_id='doc', **kwargs):
    return SpoolJobSpec(
        custom_id,
        JobKind.DOCUMENT,
        target_id,
        {
            'content_type': 'explanation',
            'title': 'Arithmetic',
            'source_files': ['core.py'],
            'source_name': 'alpha',
            'metadata': {'generated': True},
        },
        request=BatchRequest(
            custom_id, 'Document the source.', 'Explain arithmetic.', max_tokens=8192
        ),
        **kwargs,
    )


@pytest.mark.asyncio
async def test_batch_fetch_retains_response_until_atomic_application(local_deployment):
    ctx, chat, embed = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        spec = document_spec()
        strategy = LocalBatchStrategy(ctx, library=library, job_specs=[spec])
        run_id = (await strategy.submit_batch([spec.request])).batch_id
        assert run_id.startswith('local:') and chat.requests == []
        status = await strategy.poll_batch(run_id)
        assert status.succeeded == 1 and status.processing_status == 'ended'
        assert library.get_document('doc') is None
        fetched = await strategy.fetch_batch_results(run_id)
        assert fetched['doc']
        assert await strategy.fetch_batch_results(run_id) == fetched
        counts = await LocalExecutor(strategy.store).drive(run_id)
        assert counts['applied'] == 1 and counts['state'] == 'completed'
        doc = library.get_document('doc')
        assert doc and library.get_sections('doc') and library.get_chunks('doc')
        request_count = len(chat.requests) + len(embed.requests)
        await LocalExecutor(strategy.store).drive(run_id)
        assert library.get_document('doc').updated_at == doc.updated_at
        assert len(chat.requests) + len(embed.requests) == request_count
        assert [
            p['max_tokens'] for route, p, _ in chat.requests if route.endswith('/chat/completions')
        ] == [8192]
    await ctx.aclose()


@pytest.mark.asyncio
async def test_embedding_retry_does_not_repeat_chat(local_deployment):
    ctx, chat, embed = local_deployment
    embed.failures = [503]
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([document_spec()], operation='generate', scope={})
        assert (await LocalExecutor(store).drive(run_id))['applied'] == 1
        job = store.jobs(run_id)[0]
        assert job['request_attempts'] == 1 and job['apply_attempts'] == 2
        assert len([p for route, p, _ in chat.requests if route.endswith('/chat/completions')]) == 1
    await ctx.aclose()


@pytest.mark.asyncio
async def test_stale_source_and_exhausted_attempts_are_terminal(local_deployment, tmp_path):
    from docgen.spool import file_fingerprint

    ctx, chat, _ = local_deployment
    source = tmp_path / 'source.py'
    source.write_text('original')
    spec = document_spec(snapshots={str(source): file_fingerprint(source)})
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([spec], operation='generate', scope={})
        source.write_text('newer')
        assert (await LocalExecutor(store).drive(run_id))['obsolete'] == 1
        assert chat.requests == []
        run_id = store.create([document_spec()], operation='generate', scope={})
        chat.failures = [503, 503, 503]
        assert (await LocalExecutor(store).drive(run_id))['failed'] == 1
        assert store.jobs(run_id)[0]['request_attempts'] == 3
        before = len(chat.requests)
        await LocalExecutor(store).drive(run_id)
        assert len(chat.requests) == before
    await ctx.aclose()


def test_unknown_version_refuses_before_modification(local_deployment):
    ctx, _, _ = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([document_spec()], operation='generate', scope={})
        with library._conn_provider.acquire() as conn:
            conn.execute('UPDATE spool_runs SET schema_version=99 WHERE run_id=?', (run_id,))
        with pytest.raises(UnsupportedSpoolVersion):
            store.jobs(run_id)
        with library._conn_provider.acquire() as conn:
            assert conn.execute('SELECT request_attempts FROM spool_jobs').fetchone()[0] == 0


@pytest.mark.asyncio
async def test_later_unsupported_job_refuses_before_recovery(local_deployment):
    ctx, chat, embed = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create(
            [document_spec(), document_spec('later', 'later', parser_version=99)],
            operation='generate',
            scope={},
        )
        with library._conn_provider.acquire() as conn:
            conn.execute("UPDATE spool_jobs SET state='running' WHERE custom_id='doc'")
            before = conn.execute('SELECT * FROM spool_jobs ORDER BY rowid').fetchall()
        with pytest.raises(UnsupportedSpoolVersion):
            await LocalExecutor(store).drive(run_id)
        with library._conn_provider.acquire() as conn:
            assert conn.execute('SELECT * FROM spool_jobs ORDER BY rowid').fetchall() == before
        assert chat.requests == [] and embed.requests == []
    await ctx.aclose()


@pytest.mark.asyncio
async def test_payload_validation_work_is_linear_in_queue_size(local_deployment):
    ctx, chat, _ = local_deployment
    specs = [document_spec(f'doc-{i}', f'doc-{i}') for i in range(12)]
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create(specs, operation='generate', scope={})
        with patch('docgen.spool.identity_hash', wraps=identity_hash) as validate_payload:
            counts = await LocalExecutor(store).drive(run_id)
            # Allow a fixed number of passes, but no full-queue scan at every job stage.
            assert validate_payload.call_count <= 8 * len(specs)
        assert counts['applied'] == len(specs) and counts['state'] == 'completed'
        assert all(library.get_document(spec.target_id) is not None for spec in specs)
        assert len(
            [p for route, p, _ in chat.requests if route.endswith('/chat/completions')]
        ) == len(specs)
    await ctx.aclose()


@pytest.mark.asyncio
async def test_embedding_only_has_no_chat_and_clear_preserves_documents(local_deployment):
    ctx, chat, _ = local_deployment
    spec = SpoolJobSpec(
        'embed',
        JobKind.EMBEDDING,
        'authored',
        {
            'content_type': 'finding',
            'title': 'Authored',
            'content': 'Preserve this exact authored finding',
            'metadata': {},
        },
    )
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        store = SpoolStore(ctx, library)
        run_id = store.create([spec], operation='catalog', scope={})
        with pytest.raises(ValueError, match='terminal'):
            store.clear(run_id)
        assert (await LocalExecutor(store).drive(run_id))['applied'] == 1
        assert chat.requests == []
        store.clear(run_id)
        assert library.get_document('authored').content == 'Preserve this exact authored finding'
        assert store.list_runs() == []
    await ctx.aclose()


@pytest.mark.parametrize('change', ['membership', 'resolution'])
@pytest.mark.asyncio
async def test_changed_theme_inputs_are_obsolete(local_deployment, change):
    from docgen.local_pipeline import plan_themes

    ctx, chat, _ = local_deployment
    with ctx.write_session():
        library = ctx.open_library(operation='write')
        member = library.add_document(
            'catalog',
            'add',
            'Adds integers.',
            metadata={'kind': 'element', 'description': 'Adds integers.'},
            source_name='alpha',
        )
        theme = library.add_document(
            'theme', 'Pending theme', '(pending summarization)', metadata={'pending': True}
        )
        library.add_theme(
            cluster_id='arithmetic',
            doc_id=theme.id,
            member_count=1,
            resolution=1.0,
            summary_hash='',
            dirty=True,
        )
        library.set_theme_members('arithmetic', [(member.id, 1.0)])
        store = SpoolStore(ctx, library)
        run_id = store.create(plan_themes(library), operation='themes', scope={})
        if change == 'membership':
            library.set_theme_members('arithmetic', [])
        else:
            with library._conn_provider.acquire() as conn:
                conn.execute("UPDATE themes SET resolution=2 WHERE cluster_id='arithmetic'")
        assert (await LocalExecutor(store).drive(run_id))['obsolete'] == 1
        assert chat.requests == []
        assert library.get_document(theme.id).content == '(pending summarization)'
    await ctx.aclose()
