"""Barrier tests for browser admission and ownership across asynchronous cleanup."""
import asyncio
import json
from typing import ClassVar

import pytest
import yaml

from config import Config
from local_models import MAC_PROFILE_ID, mac_model_profile
from tests.test_mac_runtime_api import Runtime
from tests.test_web_local_runs import PROJECT, RUN, Store
from tests.test_web_server import _FakeBridge, _FakeReq
from web.jobs import launch_job
from web.local_runs import control_run
from web.projects import ProjectStore
from web.server import (
    _cleanup_projects,
    _local_model_action,
    _make_tool_handler,
    _onboard_start,
    _runtime_status,
    make_app,
)


class Request(_FakeReq):
    headers: ClassVar[dict] = {}
    scheme = 'http'
    host = '127.0.0.1'


class AdmissionStore(Store):
    def __init__(self):
        super().__init__()
        self.runtime.phase = 'ready'
        self.starts = 0

    def project_id_for(self, profile_id, name):
        assert profile_id == MAC_PROFILE_ID and name == 'demo'
        return PROJECT

    async def start(self, profile_id, name, *, themes_enabled=None):
        self.project_id_for(profile_id, name)
        self.starts += 1
        self.themes_enabled = themes_enabled
        return self.describe(PROJECT)


@pytest.fixture
async def state():
    store = AdmissionStore()
    app = make_app(bridge=store.service, projects=store)
    yield app, store
    await _cleanup_projects(app)


async def submit(app, action):
    body = {'project_id': PROJECT, 'source': 'demo'}
    if action == 'resume':
        return await control_run(Request(app=app, body={
            'project_id': PROJECT, 'run_id': RUN, 'action': 'resume'}))
    if action == 'profile':
        body = {'onboarding_model': MAC_PROFILE_ID, 'source': 'demo'}
    if action in ('ask', 'search'):
        return await _make_tool_handler(f'ariadne_{action}')(Request(app=app, body=body))
    return await _onboard_start(Request(app=app, body=body))


@pytest.mark.parametrize('action', ['onboard', 'resume', 'ask', 'search', 'profile'])
async def test_stop_blocks_new_work_before_worker_cleanup_finishes(state, action):
    app, store = state
    entered, release = asyncio.Event(), asyncio.Event()

    async def interrupt_managed():
        entered.set()
        await release.wait()

    store.interrupt_managed = interrupt_managed
    try:
        response = await _local_model_action(Request(app=app, body={'action': 'stop'}))
        assert response.status == 202
        await asyncio.wait_for(entered.wait(), 1)
        assert _runtime_status(app)['phase'] == 'stopping'
        assert (await submit(app, action)).status == 400
        assert not store.service.calls
        assert store.starts == 0
    finally:
        release.set()
        await app['runtime_actions']['task']


@pytest.mark.parametrize(('first', 'second'), [
    ('onboard', 'onboard'), ('onboard', 'resume'), ('resume', 'onboard'),
    ('resume', 'resume'), ('onboard', 'profile'), ('resume', 'profile'),
    ('profile', 'onboard'), ('profile', 'profile'),
])
async def test_one_project_admission_spans_bridge_connection(state, first, second):
    app, store = state
    entered, release = asyncio.Event(), asyncio.Event()
    connections = 0

    async def bridge(project_id):
        nonlocal connections
        store.describe(project_id)
        connections += 1
        if connections == 1:
            entered.set()
            await release.wait()
        return store.service

    store.bridge = bridge
    pending = asyncio.create_task(submit(app, first))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        response = await asyncio.wait_for(submit(app, second), 1)
        assert response.status == 409
        assert store.starts == (1 if first == 'profile' else 0)
        assert connections == 1
    finally:
        release.set()
        await pending


@pytest.mark.parametrize('first', ['onboard', 'resume'])
async def test_profile_start_preserves_an_existing_saved_project_execution(state, first):
    app, store = state
    assert (await submit(app, first)).status in (200, 202)
    await asyncio.wait_for(store.service.entered.wait(), 1)
    assert (await submit(app, 'profile')).status == 409
    assert store.starts == 0
    assert len(app['jobs']) == 1


@pytest.mark.parametrize('action', ['onboard', 'resume', 'ask', 'search'])
async def test_pending_connection_rechecks_readiness_before_dispatch(state, action):
    app, store = state
    entered, release = asyncio.Event(), asyncio.Event()

    async def bridge(project_id):
        store.describe(project_id)
        entered.set()
        await release.wait()
        return store.service

    store.bridge = bridge
    pending = asyncio.create_task(submit(app, action))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        store.runtime.phase = 'stopped'
    finally:
        release.set()
    assert (await pending).status == 400
    assert not store.service.calls


@pytest.mark.parametrize('active', [False, True])
async def test_completed_cancel_never_interrupts_the_project_session(state, active):
    app, store = state
    if active:
        await submit(app, 'resume')
        await asyncio.wait_for(store.service.entered.wait(), 1)
    store.service.record['state'] = 'completed'
    response = await control_run(Request(app=app, body={
        'project_id': PROJECT, 'run_id': RUN, 'action': 'cancel'}))
    assert response.status == 200
    assert json.loads(response.text)['run']['state'] == 'completed'
    assert store.order == ['cancel']
    if active:
        assert all(not job['task'].done() for job in app['jobs'].values())


async def test_cancelled_http_handler_finishes_owned_browser_callers(state):
    app, store = state
    await submit(app, 'resume')
    await asyncio.wait_for(store.service.entered.wait(), 1)
    job = next(iter(app['jobs'].values()))
    entered, release = asyncio.Event(), asyncio.Event()

    async def reap():
        entered.set()
        await release.wait()
        store.order.append('reaped')

    async def interrupt(project_id):
        store.describe(project_id)
        cleanup = asyncio.create_task(reap())
        try:
            await asyncio.shield(cleanup)
        except asyncio.CancelledError:
            await asyncio.shield(cleanup)
            raise

    store.interrupt = interrupt
    pending = asyncio.create_task(control_run(Request(app=app, body={
        'project_id': PROJECT, 'run_id': RUN, 'action': 'cancel'})))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert store.service.record['state'] == 'cancelled'
        pending.cancel()
    finally:
        release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert store.order == ['cancel', 'reaped']
    assert job['task'].done()
    assert job['queue'].terminal['type'] == 'cancelled'


async def test_completed_job_retention_keeps_active_and_latest_completions(state):
    app, store = state
    await submit(app, 'resume')
    await asyncio.wait_for(store.service.entered.wait(), 1)
    active = next(iter(app['jobs']))
    delayed_release = asyncio.Event()

    class DelayedBridge:
        async def call(self, tool, args, *, progress_callback):
            assert tool == 'ariadne_onboard' and args == {'source': 'demo'}
            await progress_callback(0, 1, 'Waiting for completion')
            await delayed_release.wait()
            return {'done': True}

    delayed = launch_job(app, DelayedBridge(), 'ariadne_onboard', {'source': 'demo'})
    completed_ids = []
    for _ in range(70):
        job = launch_job(app, _FakeBridge(), 'ariadne_onboard', {'source': 'demo'})
        await job['task']
        completed_ids.append(job['id'])
    delayed_release.set()
    await delayed['task']
    await asyncio.sleep(0)  # Run completion callbacks before inspecting retention.
    assert set(app['jobs']) == {active, delayed['id'], *completed_ids[-63:]}
    assert app['jobs'][active]['task'].done() is False
    assert app['jobs'][delayed['id']]['queue'].terminal['type'] == 'done'


async def test_cancelling_owned_execution_holds_admission_until_reaped(state):
    app, store = state
    await submit(app, 'resume')
    await asyncio.wait_for(store.service.entered.wait(), 1)
    entered, release = asyncio.Event(), asyncio.Event()

    async def interrupt(project_id):
        store.describe(project_id)
        # The MCP response may finish before its owned session is fully reaped.
        store.service.release.set()
        await next(iter(app['jobs'].values()))['task']
        entered.set()
        await release.wait()

    store.interrupt = interrupt
    pending = asyncio.create_task(control_run(Request(app=app, body={
        'project_id': PROJECT, 'run_id': RUN, 'action': 'cancel'})))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert (await submit(app, 'profile')).status == 409
        assert store.starts == 0
    finally:
        release.set()
        await pending


@pytest.mark.parametrize('profile_id', ['external', 'qwen-hosted', MAC_PROFILE_ID])
def test_admission_identity_is_pure_and_matches_saved_project(tmp_path, monkeypatch, profile_id):
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump({
        'sources': {'demo': {'path': str(tmp_path), 'depends_on': ['dependency']},
                    'dependency': str(tmp_path)},
        'onboarding_models': {'external': mac_model_profile()},
    }))
    for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    store = ProjectStore(Config(path))
    project_id = store.project_id_for(profile_id, 'demo')
    assert not store.root.exists()
    assert not store._sessions
    store.runtime = Runtime()
    store.runtime.phase = 'ready'
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'unused-test-key')
    project = store.prepare(profile_id, 'demo')
    assert project['id'] == project_id
    saved = Config(store.path(project_id))
    assert saved.sources['demo']['depends_on'] == ['dependency']


@pytest.mark.parametrize('profile_id', ['external', 'qwen-hosted'])
@pytest.mark.parametrize('action', ['onboard', 'profile', 'ask', 'search'])
async def test_managed_stop_leaves_external_and_hosted_workflows_available(
        tmp_path, monkeypatch, profile_id, action):
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump({'sources': {'demo': str(tmp_path)},
                                    'onboarding_models': {'external': mac_model_profile()}}))
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'unused-test-key')
    store = ProjectStore(Config(path))
    store.runtime = Runtime()
    project = store.prepare(profile_id, 'demo')
    built = _FakeBridge()

    async def bridge(project_id):
        assert project_id == project['id']
        return built

    monkeypatch.setattr(store, 'bridge', bridge)
    app = make_app(bridge=_FakeBridge(), projects=store)
    app['runtime_actions']['stopping'] = True
    body = {'project_id': project['id'], 'source': 'demo'}
    if action == 'profile':
        body = {'onboarding_model': profile_id, 'source': 'demo'}
    try:
        if action in ('ask', 'search'):
            response = await _make_tool_handler(f'ariadne_{action}')(Request(app=app, body=body))
        else:
            response = await _onboard_start(Request(app=app, body=body))
        assert response.status == 200
        await asyncio.gather(*(job['task'] for job in app['jobs'].values()))
        assert len(built.calls) == 1
    finally:
        await _cleanup_projects(app)
