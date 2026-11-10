"""Browser run controls cross a real MCP process and a persisted native-profile library."""
import asyncio
import json
import os
import sqlite3
import threading
from contextlib import suppress
from types import SimpleNamespace

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer
from mcp.client import stdio

from config import Config
from docgen.spool import LocalExecutor, SpoolStore
from inference import InferenceContext
from local_runtime import managed_profile_id
from tests.local_model_server import DOCUMENT
from tests.test_local_spool import document_spec
from tests.test_mac_runtime_api import Runtime
from web.mcp_client import stdio_server_params
from web.projects import ProjectStore
from web.server import make_app

pytestmark = pytest.mark.parametrize('native_pair', ['legacy', 'accuracy', 'gemma'], indirect=True)


def document_snapshot(path, document_id):
    with sqlite3.connect(path) as connection:
        return {
            table: connection.execute(
                f'SELECT * FROM {table} WHERE {column}=? ORDER BY rowid', (document_id,)).fetchall()
            for table, column in [('documents', 'id'), ('chunks', 'document_id'),
                                  ('sections', 'document_id')]
        }


def block_response(server, route, monkeypatch):
    """Delay only the selected local HTTP response; never replace provider logic."""
    entered, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    handler_class = server.server.RequestHandlerClass
    reply = handler_class.reply

    def delayed(handler, status, body):
        if handler.path == route:
            loop.call_soon_threadsafe(entered.set)
            release.wait(30)
        # A terminated MCP request may leave the fixture's response socket closed.
        with suppress(BrokenPipeError, ConnectionResetError):
            reply(handler, status, body)

    monkeypatch.setattr(handler_class, 'reply', delayed)
    return entered, release


@pytest.fixture
async def run_client(native_pair, tmp_path, monkeypatch):
    unused_context, chat, embed = native_pair
    await unused_context.aclose()
    chat.overrides['/native/v1/chat/completions'] = {
        'choices': [{'message': {'content': DOCUMENT}, 'finish_reason': 'stop'}],
        'usage': {'prompt_tokens': 4, 'completion_tokens': 200},
    }
    source = tmp_path / 'source'
    source.mkdir()
    config_path = tmp_path / 'ariadne.yaml'
    config_path.write_text(yaml.safe_dump({'sources': {'alpha': {'path': str(source)}}}))
    projects = ProjectStore(Config(config_path))
    projects.runtime = Runtime()  # Fixture endpoints stand in for the model-serving processes.
    projects.runtime.phase = 'ready'
    projects.runtime.profile_id = managed_profile_id(unused_context.settings)
    project = projects.prepare(projects.runtime.profile_id, 'alpha')
    path = projects.path(project['id'])
    saved = Config(path)
    identity = saved.model_settings().identity
    data = saved._explicit
    data['llm']['base_url'] = chat.base_url
    data['embeddings']['base_url'] = embed.base_url
    path.write_text(yaml.safe_dump(data))
    assert Config(path).model_settings().identity == identity
    context = InferenceContext.from_config(Config(path))
    with context.write_session(), context.open_library(operation='write') as library:
        store = SpoolStore(context, library)
        specs = [document_spec(name, name) for name in ('applied', 'pending')]
        run_id = store.create(specs, operation='generate', scope={'source_name': 'alpha'})
        first = store.jobs(run_id)[0]
        store.update_job(first['job_id'], state='apply_pending', response_text=DOCUMENT,
                         parsed_result_json=json.dumps({**specs[0].data, 'content': DOCUMENT}),
                         request_attempts=1, validation_attempts=1, apply_attempts=1)
        await LocalExecutor(store).apply(store.jobs(run_id)[0], json.loads(first['payload_json']))
    processes = []
    create_process = stdio._create_platform_compatible_process

    async def record_process(*args, **kwargs):
        process = await create_process(*args, **kwargs)
        processes.append(process)
        return process

    def isolated_params(config_path):
        params = stdio_server_params(config_path)
        params.env = {key: value for key, value in params.env.items()
                      if not key.endswith(('_API_KEY', '_TOKEN'))}
        params.env['PYTHON_DOTENV_DISABLED'] = '1'
        return params

    monkeypatch.setattr(stdio, '_create_platform_compatible_process', record_process)
    monkeypatch.setattr('web.projects.stdio_server_params', isolated_params)
    app = make_app(bridge=object(), projects=projects)
    client = TestClient(TestServer(app))
    await client.start_server()
    try:
        yield SimpleNamespace(client=client, projects=projects, project=project,
                              context=context, store=SpoolStore(context), run_id=run_id,
                              chat=chat, embed=embed, processes=processes, app=app)
    finally:
        try:
            await asyncio.wait_for(client.close(), 15)
        finally:
            # A failing assertion must not leave a real MCP worker behind.
            for process in processes:
                if process.returncode is None:
                    process.kill()
                    await asyncio.wait_for(process.wait(), 5)
            await context.aclose()


async def resume(harness):
    response = await asyncio.wait_for(harness.client.post('/api/local-runs', json={
        'project_id': harness.project['id'], 'run_id': harness.run_id, 'action': 'resume'}), 15)
    assert response.status == 202, await response.text()
    return (await response.json())['job_id']


async def test_browser_cancel_persists_before_reaping_pending_generation(run_client, monkeypatch):
    h = run_client
    before = document_snapshot(h.context.db_path, 'applied')
    entered, release = block_response(h.chat, '/native/v1/chat/completions', monkeypatch)
    observed = []
    interrupt = h.projects.interrupt

    async def observe_interrupt(project_id):
        record = h.store.get_run(h.run_id)
        observed.append((record['state'], record['cancel_requested'],
                         h.processes[-1].returncode))
        await interrupt(project_id)

    monkeypatch.setattr(h.projects, 'interrupt', observe_interrupt)
    try:
        job_id = await resume(h)
        await asyncio.wait_for(entered.wait(), 10)
        assert h.store.jobs(h.run_id)[1]['state'] == 'running'
        process = h.processes[-1]
        response = await asyncio.wait_for(h.client.post('/api/local-runs', json={
            'project_id': h.project['id'], 'run_id': h.run_id, 'action': 'cancel'}), 10)
        assert response.status == 200, await response.text()
        assert (await response.json())['run']['state'] == 'cancelled'
        assert observed == [('cancelled', 1, None)]
        assert process.returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        assert h.project['id'] not in h.projects._sessions
        assert h.app['jobs'][job_id]['task'].done()
        assert h.app['jobs'][job_id]['queue'].terminal['type'] == 'cancelled'
        assert [job['state'] for job in h.store.jobs(h.run_id)] == ['applied', 'cancelled']
        assert document_snapshot(h.context.db_path, 'applied') == before
        assert not document_snapshot(h.context.db_path, 'pending')['documents']
    finally:
        release.set()


async def test_browser_stop_restart_resume_reuses_saved_response_once(run_client, monkeypatch):
    h = run_client
    before = document_snapshot(h.context.db_path, 'applied')
    entered, release = block_response(h.embed, '/native/v1/embeddings', monkeypatch)
    try:
        job_id = await resume(h)
        await asyncio.wait_for(entered.wait(), 10)
        saved = h.store.jobs(h.run_id)[1]
        assert saved['state'] == 'apply_pending' and saved['response_text'] == DOCUMENT
        assert saved['request_attempts'] == 1 and saved['apply_attempts'] == 1
        process = h.processes[-1]
        stopped = []
        stop = h.projects.runtime.stop

        async def observe_stop():
            stopped.append((process.returncode, h.app['jobs'][job_id]['task'].done()))
            await stop()

        monkeypatch.setattr(h.projects.runtime, 'stop', observe_stop)
        response = await h.client.post('/api/local-models', json={'action': 'stop'})
        assert response.status == 202
        await asyncio.wait_for(h.app['runtime_actions']['task'], 10)
        assert stopped and stopped[0][0] is not None and stopped[0][1]
        assert h.projects.runtime.phase == 'stopped'
        assert h.app['jobs'][job_id]['queue'].terminal['type'] == 'interrupted'
        assert h.project['id'] not in h.projects._sessions
        with pytest.raises(ProcessLookupError):
            os.kill(process.pid, 0)
        record = h.store.get_run(h.run_id)
        assert record['state'] == 'running' and not record['cancel_requested']
        assert h.store.jobs(h.run_id)[1]['response_text'] == DOCUMENT
        assert document_snapshot(h.context.db_path, 'applied') == before
        assert not document_snapshot(h.context.db_path, 'pending')['documents']
        release.set()

        # Models are stopped; a fresh MCP process can still read durable status.
        requests_before = len(h.chat.requests), len(h.embed.requests)
        response = await h.client.get('/api/local-runs', params={'project_id': h.project['id']})
        assert response.status == 200, await response.text()
        listed = (await response.json())['runs'][0]
        assert listed['run_id'] == h.run_id and listed['counts']['applied'] == 1
        assert not listed['active_here']
        assert h.processes[-1].pid != process.pid
        assert (len(h.chat.requests), len(h.embed.requests)) == requests_before

        assert (await h.client.post('/api/local-models', json={'action': 'start'})).status == 202
        await asyncio.wait_for(h.app['runtime_actions']['task'], 5)
        resumed_id = await resume(h)
        await asyncio.wait_for(h.app['jobs'][resumed_id]['task'], 10)
        terminal = h.app['jobs'][resumed_id]['queue'].terminal
        assert terminal['type'] == 'done', terminal
        assert terminal['result']['run']['state'] == 'completed'
        assert terminal['result']['run']['counts']['applied'] == 2
        jobs = h.store.jobs(h.run_id)
        assert [job['state'] for job in jobs] == ['applied', 'applied']
        assert jobs[1]['request_attempts'] == 1 and jobs[1]['apply_attempts'] == 2
        assert len([route for route, _ in h.chat.requests
                    if route == '/native/v1/chat/completions']) == 1
        assert document_snapshot(h.context.db_path, 'applied') == before
        after = document_snapshot(h.context.db_path, 'pending')
        assert len(after['documents']) == 1 and after['chunks'] and after['sections']

        requests_after = len(h.chat.requests), len(h.embed.requests)
        response = await h.client.post('/api/local-runs', json={
            'project_id': h.project['id'], 'run_id': h.run_id, 'action': 'resume'})
        assert response.status == 400  # Completed work cannot be applied again.
        assert document_snapshot(h.context.db_path, 'pending') == after
        assert (len(h.chat.requests), len(h.embed.requests)) == requests_after
    finally:
        release.set()
