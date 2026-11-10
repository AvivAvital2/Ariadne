"""Durable local run controls preserve offline access and corpus ownership."""

import json
import sqlite3

import pytest
import yaml

from ariadne_mcp.service import AriadneService
from config import Config
from docgen.spool import SpoolStore, UnsupportedSpoolVersion
from library.model_binding import ModelBindingMismatch
from local_models import mac_model_profile
from tests.test_local_spool import document_spec


@pytest.fixture
def service(tmp_path):
    data = mac_model_profile()
    data.update(
        db_path='data/library.db',
        staleness_db_path='data/staleness.db',
        default_source='alpha',
        sources={
            'alpha': {'path': str(tmp_path), 'depends_on': ['beta']},
            'beta': {'path': str(tmp_path)},
        },
    )
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(data))
    service = AriadneService()
    service._config = Config(path)
    return service


def create_run(service, *, source='alpha', client_run_id=None, specs=None):
    context = service.inference_context
    with context.write_session(), context.open_library(operation='write') as library:
        store = SpoolStore(context, library)
        run_id = store.create(
            [document_spec()] if specs is None else specs,
            operation='onboard',
            scope={
                'source_name': source,
                'client_run_id': client_run_id,
                'private_path': '/not-for-the-browser',
            },
        )
    return run_id


def prohibit_corpus_access(monkeypatch, service):
    def forbidden(*_args, **_kwargs):
        pytest.fail('Inspection/control must not open a library or acquire its writer')

    context = service.inference_context
    monkeypatch.setattr(context, 'open_library', forbidden)
    monkeypatch.setattr(context, 'write_session', forbidden)


def test_missing_library_lists_offline_without_creating_files(service, monkeypatch):
    context = service.inference_context
    prohibit_corpus_access(monkeypatch, service)
    assert service.local_runs().model_dump() == {'runs': []}
    assert not context.db_path.parent.exists()
    assert context._chat is None and context._embeddings is None


def test_run_inspection_is_offline_scoped_and_contains_only_safe_records(service, monkeypatch):
    selected = create_run(service, client_run_id='abcdef012345')
    create_run(service, source='beta')
    context = service.inference_context
    with sqlite3.connect(context.db_path) as connection:
        connection.execute(
            'UPDATE spool_jobs SET response_text=? WHERE run_id=?',
            ('PRIVATE GENERATED RESPONSE', selected),
        )
    prohibit_corpus_access(monkeypatch, service)
    rows = service.local_runs().model_dump()['runs']
    assert len(rows) == 1 and rows[0]['run_id'] == selected
    assert set(rows[0]) == {
        'run_id',
        'operation_kind',
        'source',
        'phase',
        'state',
        'cancel_requested',
        'created_at',
        'updated_at',
        'counts',
        'client_run_id',
    }
    assert rows[0]['source'] == 'alpha'
    assert rows[0]['counts']['responses'] == 1
    assert rows[0]['client_run_id'] == 'abcdef012345'
    assert 'PRIVATE' not in json.dumps(rows) and 'private_path' not in json.dumps(rows)
    assert service.local_runs(run_id=selected).model_dump()['runs'] == rows
    assert service.local_runs(client_run_id='abcdef012345').model_dump()['runs'] == rows
    assert service.local_runs(client_run_id='012345abcdef').model_dump() == {'runs': []}
    assert context._chat is None and context._embeddings is None


def test_explicit_run_from_another_source_is_rejected_even_for_a_dependency(service):
    foreign = create_run(service, source='beta')
    with pytest.raises(ValueError, match='source'):
        service.local_runs(run_id=foreign, source='alpha')
    with pytest.raises(ValueError, match='source'):
        service.local_runs(source='unknown')


@pytest.mark.parametrize('corruption', ['binding', 'schema', 'payload'])
def test_inspection_rejects_incompatible_runs(service, corruption):
    run_id = create_run(
        service, specs=[document_spec(parser_version=99)] if corruption == 'payload' else None
    )
    with sqlite3.connect(service.inference_context.db_path) as connection:
        if corruption == 'binding':
            connection.execute('UPDATE spool_runs SET profile_hash=?', ('wrong-profile',))
        elif corruption == 'schema':
            connection.execute('UPDATE spool_runs SET schema_version=99')
    error = ModelBindingMismatch if corruption == 'binding' else UnsupportedSpoolVersion
    with pytest.raises(error):
        service.local_runs(run_id=run_id)


def test_cancellation_works_while_writer_is_held_and_preserves_applied_knowledge(
    service, monkeypatch
):
    run_id = create_run(
        service,
        specs=[
            document_spec('saved', 'saved'),
            document_spec('pending', 'pending'),
        ],
    )
    context = service.inference_context
    with context.write_session(), context.open_library(operation='write') as library:
        existing = library.add_document(
            'catalog', 'Saved', 'Preserve this knowledge', doc_id='saved'
        )
        with library._conn_provider.acquire() as connection:
            connection.execute("UPDATE spool_jobs SET state='applied' WHERE target_id='saved'")
        prohibit_corpus_access(monkeypatch, service)
        record = service.local_run_cancel(run_id).model_dump()['run']
        assert record['state'] == 'cancelled' and record['cancel_requested'] is True
        assert record['counts']['applied'] == 1 and record['counts']['cancelled'] == 1
        assert library.get_document(existing.id).content == existing.content
        assert library.get_document(existing.id).updated_at == existing.updated_at
        assert service.local_run_cancel(run_id).model_dump()['run']['state'] == 'cancelled'
    assert context._chat is None and context._embeddings is None


@pytest.mark.parametrize('corruption', ['source', 'binding', 'schema', 'payload'])
def test_cancellation_rejects_foreign_or_incompatible_runs_without_mutation(service, corruption):
    run_id = create_run(
        service,
        source='beta' if corruption == 'source' else 'alpha',
        specs=[document_spec(parser_version=99)] if corruption == 'payload' else None,
    )
    with sqlite3.connect(service.inference_context.db_path) as connection:
        if corruption == 'binding':
            connection.execute('UPDATE spool_runs SET profile_hash=?', ('wrong-profile',))
        elif corruption == 'schema':
            connection.execute('UPDATE spool_runs SET schema_version=99')
    with pytest.raises(ValueError, match=r'source|profile|schema|payload'):
        service.local_run_cancel(run_id, source='alpha')
    with sqlite3.connect(service.inference_context.db_path) as connection:
        assert connection.execute('SELECT state,cancel_requested FROM spool_runs').fetchone() == (
            'running',
            0,
        )


async def test_resume_owns_the_bound_writer_and_uses_the_project_context(service, monkeypatch):
    import config
    from docgen import local_pipeline
    from inference import current_context

    run_id = create_run(service, specs=[])
    context = service.inference_context
    monkeypatch.setattr(config, '_global_config', service.config)
    resume = local_pipeline.resume_run
    calls = []

    async def checked_resume(library, selected):
        assert selected == run_id
        assert library.inference_context is context and current_context() is context
        context.require_writer()
        assert config.get_config() is service.config
        calls.append(selected)
        return await resume(library, selected)

    monkeypatch.setattr(local_pipeline, 'resume_run', checked_resume)
    record = (await service.local_run_resume(run_id)).model_dump()['run']
    assert record['state'] == 'completed' and calls == [run_id]
    assert context.session is None and current_context() is None
    with context.write_session():
        pass  # The operation released the cross-process writer lock.
    assert context._chat is None and context._embeddings is None


async def test_resume_refuses_an_existing_writer_without_reusing_its_session(service):
    from library.locking import LibraryBusy

    run_id = create_run(service, specs=[])
    context = service.inference_context
    with context.write_session():
        with pytest.raises(LibraryBusy):
            await service.local_run_resume(run_id)
    assert SpoolStore(context).get_run(run_id)['state'] == 'running'


@pytest.mark.parametrize('state', ['completed', 'cancelled', 'failed', 'obsolete'])
async def test_terminal_resume_is_offline_and_performs_no_work(service, monkeypatch, state):
    run_id = create_run(service)
    with sqlite3.connect(service.inference_context.db_path) as connection:
        connection.execute('UPDATE spool_runs SET state=?', (state,))
    prohibit_corpus_access(monkeypatch, service)
    record = (await service.local_run_resume(run_id)).model_dump()['run']
    assert record['state'] == state
    assert SpoolStore(service.inference_context).jobs(run_id)[0]['request_attempts'] == 0


@pytest.mark.parametrize('corruption', ['source', 'binding', 'schema', 'payload'])
async def test_resume_rejects_foreign_or_incompatible_runs_before_writer_access(
    service, monkeypatch, corruption
):
    run_id = create_run(
        service,
        source='beta' if corruption == 'source' else 'alpha',
        specs=[document_spec(parser_version=99)] if corruption == 'payload' else None,
    )
    with sqlite3.connect(service.inference_context.db_path) as connection:
        if corruption == 'binding':
            connection.execute('UPDATE spool_runs SET profile_hash=?', ('wrong-profile',))
        elif corruption == 'schema':
            connection.execute('UPDATE spool_runs SET schema_version=99')
    prohibit_corpus_access(monkeypatch, service)
    with pytest.raises(ValueError, match=r'source|profile|schema|payload'):
        await service.local_run_resume(run_id)


async def test_run_tools_are_registered_with_safe_structured_outputs(service, monkeypatch):
    from ariadne_mcp import server

    monkeypatch.setattr(AriadneService, '_instance', service)
    run_id = create_run(service, client_run_id='abcdef012345')
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    names = ('ariadne_local_runs', 'ariadne_local_run_cancel', 'ariadne_local_run_resume')
    assert set(names) <= tools.keys()
    assert tools[names[0]].annotations.readOnlyHint is True
    assert tools[names[1]].annotations.readOnlyHint is False
    assert tools[names[2]].annotations.readOnlyHint is False
    for name in names:
        assert tools[name].outputSchema
        assert 'source' in tools[name].inputSchema['properties']
    _, result = await server.mcp.call_tool(
        names[0],
        {'source': 'alpha', 'client_run_id': 'abcdef012345'},
    )
    assert result['runs'][0]['run_id'] == run_id
    for name in names[1:]:
        _, result = await server.mcp.call_tool(name, {'run_id': run_id, 'source': 'alpha'})
        assert result['run']['state'] == 'cancelled'
        assert 'scope_json' not in result['run'] and 'jobs' not in result['run']


async def test_onboard_correlation_is_durable_and_preserves_saved_dependency_scope(
    service, monkeypatch
):
    from types import SimpleNamespace

    from docgen import local_pipeline
    from tests.local_model_server import trap_inference

    trap_inference(monkeypatch, [])
    captured = {}

    async def finish_without_inference(store, run_id, _orchestrator):
        captured.update(json.loads(store.get_run(run_id)['scope_json']))
        return store.finish(run_id)

    monkeypatch.setattr(local_pipeline, 'complete_onboard', finish_without_inference)
    monkeypatch.setattr(service, 'coverage', lambda _source: SimpleNamespace(coverage_percent=0))
    monkeypatch.setattr(service, '_count_cataloged_files', lambda _source: 0)
    try:
        result = await service.onboard(
            'alpha',
            doc_types=['explanation'],
            client_run_id='abcdef012345',
        )
        assert result.run_id.startswith('local:')
        assert captured['client_run_id'] == 'abcdef012345'
        assert captured['dependencies'] == ['beta']
        assert captured['source_name'] == 'alpha' and captured['doc_types'] == ['explanation']
    finally:
        await service.close()
    reopened = AriadneService()
    reopened._config = Config(service.config.config_path)
    try:
        rows = reopened.local_runs(client_run_id='abcdef012345').model_dump()['runs']
        assert len(rows) == 1 and rows[0]['run_id'] == result.run_id
        assert rows[0]['state'] == 'completed'
    finally:
        await reopened.close()


async def test_onboard_mcp_schema_and_call_forward_the_correlation_id(service, monkeypatch):
    from ariadne_mcp import server, server_admin
    from ariadne_mcp.models import OnboardResponse

    calls = []

    async def onboard(source, **kwargs):
        calls.append((source, kwargs))
        return OnboardResponse(
            source=source,
            model='model',
            mode='live',
            files_indexed=0,
            docs_written=0,
            themes_found=0,
            coverage_percent=0,
            run_id='local:example',
        )

    monkeypatch.setattr(AriadneService, '_instance', service)
    monkeypatch.setattr(service, 'onboard', onboard)
    tools = {tool.name: tool for tool in await server.mcp.list_tools()}
    assert 'client_run_id' in tools['ariadne_onboard'].inputSchema['properties']
    result = await server_admin.ariadne_onboard(source='alpha', client_run_id='abcdef012345')
    assert result.run_id == 'local:example'
    assert calls[0][1]['client_run_id'] == 'abcdef012345'


@pytest.mark.parametrize('operation', ['list', 'cancel', 'resume'])
async def test_unknown_run_in_missing_library_is_rejected_without_writes(
    service, monkeypatch, operation
):
    prohibit_corpus_access(monkeypatch, service)
    with pytest.raises(ValueError, match='Unknown local run'):
        if operation == 'list':
            service.local_runs(run_id='local:missing')
        elif operation == 'cancel':
            service.local_run_cancel('local:missing')
        else:
            await service.local_run_resume('local:missing')
    assert not service.inference_context.db_path.parent.exists()


async def test_interrupted_onboard_remains_discoverable_and_resumable(service, monkeypatch):
    import asyncio

    from docgen import local_pipeline
    from tests.local_model_server import trap_inference

    trap_inference(monkeypatch, [])

    async def interrupt_after_persistence(*_args):
        raise asyncio.CancelledError

    monkeypatch.setattr(local_pipeline, 'complete_onboard', interrupt_after_persistence)
    try:
        with pytest.raises(asyncio.CancelledError):
            await service.onboard('alpha', client_run_id='abcdef012345')
        assert service.inference_context.session is None
    finally:
        await service.close()
    reopened = AriadneService()
    reopened._config = Config(service.config.config_path)
    try:
        record = reopened.local_runs(client_run_id='abcdef012345').model_dump()['runs'][0]
        assert record['state'] == 'running' and record['cancel_requested'] is False
        with reopened.inference_context.write_session():
            pass
    finally:
        await reopened.close()


async def test_hosted_onboarding_ignores_local_correlation_and_creates_no_local_runs(
    service, monkeypatch
):
    from types import SimpleNamespace

    from cli.onboard_pipeline import OnboardResult
    from inference import current_context
    from tests.local_model_server import trap_inference

    trap_inference(monkeypatch, [])
    cfg = service.config
    cfg.config_path.write_text(
        yaml.safe_dump(
            {
                'sources': cfg.sources,
                'default_source': 'alpha',
                'db_path': cfg.db_path,
                'staleness_db_path': cfg.staleness_db_path,
                'llm': {'provider': 'openai', 'model': 'gpt-6-astra'},
            }
        )
    )
    service._config = Config(cfg.config_path)
    seen = []

    async def hosted_pipeline(source, model, types, **kwargs):
        assert current_context().settings.chat.provider == 'openai'
        assert 'client_run_id' not in kwargs
        seen.append((source, model, types, kwargs['mode']))
        return OnboardResult(docs_written=2, themes_found=0)

    monkeypatch.setattr('cli.onboard_pipeline.run_onboard_pipeline', hosted_pipeline)
    monkeypatch.setattr(service, 'coverage', lambda _source: SimpleNamespace(coverage_percent=100))
    monkeypatch.setattr(service, '_count_cataloged_files', lambda _source: 1)
    result = await service.onboard(
        'alpha',
        doc_types=['explanation'],
        client_run_id='abcdef012345',
    )
    assert result.run_id is None and result.docs_written == 2
    assert seen == [('alpha', 'gpt-6-astra', ('explanation',), 'live')]
    with sqlite3.connect(cfg.db_path) as connection:
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='spool_runs'",
            ).fetchall()
            == []
        )
    with pytest.raises(ValueError, match='local model'):
        service.local_runs()
    with pytest.raises(ValueError, match='local model'):
        service.local_run_cancel('local:missing')
    with pytest.raises(ValueError, match='local model'):
        await service.local_run_resume('local:missing')
    await service.close()
