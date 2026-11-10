"""The browser owns bounded runtime actions without supplying commands or URLs."""
import asyncio

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config import Config
from local_models import MAC_PROFILE_ID, managed_model_profile
from model_config import resolve_models
from tests.test_web_server import _FakeBridge
from web.projects import ProjectStore
from web.server import make_app


class Runtime:
    def __init__(self):
        self.phase = 'stopped'
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.cancelled = False
        self.failure = None
        self.calls = []
        self.profiles = []
        self.profile_id = MAC_PROFILE_ID

    def status(self):
        return {'phase': self.phase, 'platform_error': None, 'error': None,
                'profile_id': self.profile_id,
                'profile_hash': resolve_models(managed_model_profile(self.profile_id)).profile_hash}

    async def setup(self, *, profile_id=None):
        self.calls.append('download')
        self.profiles.append(profile_id)
        if profile_id:
            self.profile_id = profile_id
        self.phase = 'downloading'
        self.entered.set()
        try:
            await self.release.wait()
        except asyncio.CancelledError:
            self.cancelled = True
            raise
        if self.failure:
            raise ValueError(self.failure)
        self.phase = 'stopped'

    async def start(self, *, profile_id=None):
        self.calls.append('start')
        self.profiles.append(profile_id)
        if profile_id:
            self.profile_id = profile_id
        self.phase = 'ready'

    async def stop(self):
        self.calls.append('stop')
        self.phase = 'stopped'


@pytest.fixture
async def client(tmp_path):
    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    store = ProjectStore(Config(path))
    store.runtime = Runtime()
    async with TestClient(TestServer(make_app(bridge=_FakeBridge(), projects=store))) as client:
        yield client, store.runtime


async def wait_idle(client):
    for _ in range(100):
        state = await (await client.get('/api/local-models')).json()
        if not state['busy']:
            return state
        await asyncio.sleep(0.01)
    pytest.fail('Runtime action did not finish')


async def test_download_status_stop_and_restart(client):
    client, runtime = client
    assert (await client.get('/api/local-models')).status == 200
    response = await client.post('/api/local-models', json={'action': 'download'})
    assert response.status == 202
    await asyncio.wait_for(runtime.entered.wait(), 1)
    state = await (await client.get('/api/local-models')).json()
    assert state['busy'] and state['phase'] == 'downloading'
    assert (await client.post('/api/local-models', json={'action': 'start'})).status == 409
    assert runtime.calls == ['download']
    assert (await client.post('/api/local-models', json={'action': 'stop'})).status == 202
    assert (await wait_idle(client))['phase'] == 'stopped'
    assert runtime.cancelled
    assert (await client.post('/api/local-models', json={'action': 'start'})).status == 202
    assert (await wait_idle(client))['phase'] == 'ready'
    assert runtime.calls == ['download', 'stop', 'start']


async def test_action_failure_is_visible_and_retryable(client):
    client, runtime = client
    runtime.failure = 'Not enough disk space'
    runtime.release.set()
    assert (await client.post('/api/local-models', json={'action': 'download'})).status == 202
    state = await wait_idle(client)
    assert state['phase'] == 'error' and state['error'] == 'Not enough disk space'
    runtime.failure = None
    assert (await client.post('/api/local-models', json={'action': 'download'})).status == 202
    state = await wait_idle(client)
    assert state['phase'] == 'stopped' and not state['error']


@pytest.mark.parametrize('body', [
    [],
    None,
    {'action': 'shell'},
    {'action': 'start', 'url': 'http://elsewhere'},
    {'action': 'start', 'profile_id': 'unregistered'},
    {'action': 'start', 'profile_id': []},
    {'action': 'stop', 'profile_id': MAC_PROFILE_ID},
    {'action': 'start', 'profile_id': MAC_PROFILE_ID, 'model': 'replacement'},
])
async def test_only_fixed_runtime_actions_are_accepted(client, body):
    client, runtime = client
    assert (await client.post('/api/local-models', json=body)).status == 400
    assert runtime.calls == []


async def test_runtime_actions_reject_cross_origin_requests(client):
    client, runtime = client
    response = await client.post('/api/local-models', json={'action': 'start'},
                                 headers={'Origin': 'https://unrelated.example'})
    assert response.status == 403
    assert runtime.calls == []


async def test_shutdown_cancels_download_and_reaps_runtime(client):
    client, runtime = client
    assert (await client.post('/api/local-models', json={'action': 'download'})).status == 202
    await asyncio.wait_for(runtime.entered.wait(), 1)
    await client.close()
    assert runtime.cancelled and runtime.calls == ['download', 'stop']


@pytest.mark.parametrize('profile_id', [MAC_PROFILE_ID, 'qwen-mac-27b-q4-thinking-v1'])
@pytest.mark.parametrize('action', ['download', 'start'])
async def test_runtime_actions_select_the_explicit_registered_profile(client, profile_id, action):
    client, runtime = client
    runtime.release.set()
    response = await client.post('/api/local-models', json={
        'action': action, 'profile_id': profile_id})
    assert response.status == 202
    await wait_idle(client)
    assert runtime.calls == [action]
    assert runtime.profiles == [profile_id]


