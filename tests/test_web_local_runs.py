"""Browser run control stays project-bound and reuses durable MCP operations."""
import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.test_mac_runtime_api import Runtime, wait_idle
from web.server import make_app

PROJECT = 'a' * 24
RUN = 'local:' + 'b' * 32


class Bridge:
    def __init__(self, order):
        self.calls = []
        self.order = order
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.record = {
            'run_id': RUN, 'operation_kind': 'onboard', 'source': 'demo',
            'phase': 'generate', 'state': 'running', 'cancel_requested': False,
            'counts': {'total': 2, 'applied': 1}, 'client_run_id': 'original-job',
        }

    async def call(self, tool, args, *, progress_callback=None):
        self.calls.append((tool, args))
        if progress_callback is not None:
            await progress_callback(0, None, 'Starting saved work')
        assert args.get('source') == 'demo'
        if args.get('run_id') not in (None, RUN):
            from web.mcp_client import MCPCallError
            raise MCPCallError('Unknown local run')
        if tool in ('ariadne_ask', 'ariadne_search'):
            return {'results': []}
        if tool == 'ariadne_local_runs':
            return {'runs': [self.record.copy()]}
        if tool == 'ariadne_local_run_cancel':
            self.order.append('cancel')
            if self.record['state'] not in {'completed', 'cancelled', 'failed', 'obsolete'}:
                self.record.update(state='cancelled', cancel_requested=True)
            return {'run': self.record.copy()}
        assert tool in ('ariadne_local_run_resume', 'ariadne_onboard')
        if tool == 'ariadne_onboard':
            self.record['client_run_id'] = args['client_run_id']
        self.entered.set()
        await self.release.wait()
        self.record['state'] = 'completed'
        if tool == 'ariadne_onboard':
            return {'run_id': RUN, 'docs_written': 2}
        return {'run': self.record.copy()}


class Store:
    def __init__(self):
        self.order = []
        self.runtime = Runtime()
        self.service = Bridge(self.order)

    def describe(self, project_id):
        if project_id != PROJECT:
            raise ValueError('Unknown project')
        return {'id': PROJECT, 'source': 'demo', 'provider': 'local', 'managed_local': True}

    async def bridge(self, project_id):
        self.describe(project_id)
        return self.service

    def check_runtime_ready(self, project_id):
        self.describe(project_id)
        if self.runtime.phase != 'ready':
            raise ValueError('Start local models before continuing')

    async def interrupt(self, project_id):
        self.describe(project_id)
        self.order.append('interrupt')

    async def interrupt_managed(self):
        self.order.append('interrupt-managed')

    async def close(self):
        pass


@pytest.fixture
async def client():
    store = Store()
    async with TestClient(TestServer(make_app(bridge=store.service, projects=store))) as client:
        yield client, store


async def test_list_durable_runs_with_models_stopped_after_web_restart(client):
    client, store = client
    response = await client.get('/api/local-runs', params={'project_id': PROJECT})
    assert response.status == 200
    data = await response.json()
    assert data['runs'][0]['run_id'] == RUN
    assert data['runs'][0]['counts']['applied'] == 1
    assert store.service.calls == [('ariadne_local_runs', {'source': 'demo'})]
    # The new web application has no browser job registry or remembered SSE state.
    async with TestClient(TestServer(make_app(bridge=store.service, projects=store))) as restarted:
        response = await restarted.get('/api/local-runs', params={'project_id': PROJECT})
        assert response.status == 200
        assert (await response.json())['runs'][0]['run_id'] == RUN


async def test_unknown_project_never_reaches_run_controls(client):
    client, store = client
    assert (await client.get('/api/local-runs', params={'project_id': 'not-a-project'})).status == 400
    response = await client.post('/api/local-runs', json={
        'project_id': 'not-a-project', 'run_id': RUN, 'action': 'cancel'})
    assert response.status == 400
    assert not store.service.calls


async def test_cancel_is_offline_without_interrupting_an_unmatched_execution(client):
    client, store = client
    response = await client.post('/api/local-runs', json={
        'project_id': PROJECT, 'run_id': RUN, 'action': 'cancel'})
    assert response.status == 200
    assert (await response.json())['run']['state'] == 'cancelled'
    assert store.order == ['cancel']
    assert store.service.record['counts']['applied'] == 1


async def test_resume_requires_readiness_and_rejects_duplicate_execution(client):
    client, store = client
    body = {'project_id': PROJECT, 'run_id': RUN, 'action': 'resume'}
    assert (await client.post('/api/local-runs', json=body)).status == 400
    assert not store.service.calls
    store.runtime.phase = 'ready'
    response = await client.post('/api/local-runs', json=body)
    assert response.status == 202
    await asyncio.wait_for(store.service.entered.wait(), 1)
    assert (await client.post('/api/local-runs', json=body)).status == 409
    assert sum(tool == 'ariadne_local_run_resume' for tool, _ in store.service.calls) == 1
    store.service.release.set()


@pytest.mark.parametrize('state', ['completed', 'cancelled', 'failed', 'obsolete'])
async def test_terminal_runs_cannot_be_resubmitted(client, state):
    client, store = client
    store.runtime.phase = 'ready'
    store.service.record['state'] = state
    response = await client.post('/api/local-runs', json={
        'project_id': PROJECT, 'run_id': RUN, 'action': 'resume'})
    assert response.status == 400
    assert not store.service.entered.is_set()


@pytest.mark.parametrize('body', [[], {}, {'project_id': PROJECT, 'run_id': RUN, 'action': 'shell'},
                                 {'project_id': PROJECT, 'run_id': RUN, 'action': 'cancel', 'source': 'other'}])
async def test_control_actions_accept_only_the_saved_project_and_fixed_action(client, body):
    client, store = client
    assert (await client.post('/api/local-runs', json=body)).status == 400
    assert not store.service.calls


async def test_cross_origin_run_mutation_is_rejected(client):
    client, store = client
    response = await client.post('/api/local-runs', json={
        'project_id': PROJECT, 'run_id': RUN, 'action': 'cancel'},
        headers={'Origin': 'https://unrelated.example'})
    assert response.status == 403
    assert not store.service.calls


async def test_onboard_correlates_its_durable_run_and_replays_terminal_events(client):
    client, store = client
    store.runtime.phase = 'ready'
    response = await client.post('/api/onboard', json={'project_id': PROJECT, 'source': 'demo'})
    assert response.status == 200
    job_id = (await response.json())['job_id']
    await asyncio.wait_for(store.service.entered.wait(), 1)
    assert store.service.record['client_run_id'] == job_id
    store.service.release.set()
    first = await client.get('/api/onboard/events', params={'job_id': job_id})
    body = await asyncio.wait_for(first.text(), 2)
    assert '"type": "done"' in body and RUN in body
    # A second tab/reconnect must see the same result without consuming it.
    second = await client.get('/api/onboard/events', params={'job_id': job_id})
    assert second.status == 200
    assert await asyncio.wait_for(second.text(), 2) == body


async def test_runtime_stop_interrupts_owned_execution_before_stopping_servers(client):
    client, store = client

    async def stop():
        store.order.append('stop-servers')
        store.runtime.phase = 'stopped'

    store.runtime.stop = stop
    assert (await client.post('/api/local-models', json={'action': 'stop'})).status == 202
    await wait_idle(client)
    assert store.order == ['interrupt-managed', 'stop-servers']
    assert store.service.record['state'] == 'running'
