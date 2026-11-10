"""Managed Mac selection is explicit and cannot start a build before readiness."""
import asyncio
from pathlib import Path

import pytest
import yaml
from aiohttp.test_utils import TestClient, TestServer

from config import Config
from local_models import GEMMA_PROFILE_ID, MAC_PROFILE_ID, mac_model_profile
from local_runtime import MacRuntime
from model_config import resolve_models
from tests.test_web_server import _FakeBridge, _FakeReq
from web.projects import ProjectStore
from web.server import _make_tool_handler, _onboard_start, make_app


@pytest.fixture
def store(tmp_path, monkeypatch):
    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    monkeypatch.setattr('local_runtime.platform_error', lambda: None)
    return legacy_store(Config(path))


def legacy_store(config):
    store = ProjectStore(config)
    store.runtime = MacRuntime(store.runtime.root, settings=resolve_models(mac_model_profile()))
    return store


def test_managed_choice_can_preview_offline_but_build_requires_ready(store, tmp_path):
    choice = next(c for c in store.choices() if c['id'] == MAC_PROFILE_ID)
    assert choice['managed_local'] is True
    assert choice['qualification'] == 'unqualified'
    assert 'factual errors' in choice['quality_note']
    assert 'Experimental' in choice['label']
    assert choice['preview_available'] is True
    assert choice['available'] is False
    assert 'Start' in choice['reason']
    assert not store.root.exists()
    with pytest.raises(ValueError, match='Start'):
        store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    assert store.list_projects() == []
    store.runtime._state.update(phase='ready')
    choice = next(c for c in store.choices() if c['id'] == MAC_PROFILE_ID)
    assert choice['available'] is True
    project = store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    assert project['managed_local'] is True
    saved = Config(store.path(project['id']))
    assert saved.model_settings().identity == resolve_models(mac_model_profile()).identity
    assert not Path(saved.db_path).exists()
    assert ProjectStore(store.config).describe(project['id']) == project


@pytest.mark.parametrize('saved_profile', [MAC_PROFILE_ID, 'qwen-mac-27b-q4-thinking-v1', GEMMA_PROFILE_ID])
@pytest.mark.parametrize('operation', ['ask', 'search', 'rebuild', 'resume'])
async def test_inference_routes_reject_the_other_managed_profile_before_connecting(
        store, tmp_path, monkeypatch, saved_profile, operation):
    from local_models import MAC_PROFILE_IDS, managed_model_profile

    store.runtime = MacRuntime(store.runtime.root, settings=resolve_models(managed_model_profile(saved_profile)))
    store.runtime._state.update(phase='ready')
    project = store.prepare(saved_profile, 'repo', str(tmp_path))
    saved_bytes = store.path(project['id']).read_bytes()
    other = next(profile for profile in MAC_PROFILE_IDS if profile != saved_profile)
    store.runtime = MacRuntime(store.runtime.root, settings=resolve_models(managed_model_profile(other)))
    store.runtime._state.update(phase='ready')
    connected = []

    async def connect(project_id):
        connected.append(project_id)
        return _FakeBridge()

    monkeypatch.setattr(store, 'bridge', connect)
    body = {'project_id': project['id'], 'source': 'repo'}
    route = operation
    if operation == 'ask':
        body['question'] = 'Explain the source'
    elif operation == 'search':
        body['query'] = 'source'
    elif operation == 'rebuild':
        route = 'onboard'
    else:
        route = 'local-runs'
        body = {'project_id': project['id'], 'run_id': 'local:' + 'a' * 32, 'action': 'resume'}
    app = make_app(bridge=_FakeBridge(), projects=store)
    async with TestClient(TestServer(app)) as client:
        response = await client.post('/api/' + route, json=body)
        assert response.status == 400
        assert 'model profile saved for this project' in (await response.json())['error']
        assert not connected and not app['jobs']
    assert store.path(project['id']).read_bytes() == saved_bytes


def test_unsupported_host_has_explanation_and_no_build_side_effects(store, tmp_path, monkeypatch):
    monkeypatch.setattr('local_runtime.platform_error', lambda: 'Native Apple Silicon required')
    choice = next(c for c in store.choices() if c['id'] == MAC_PROFILE_ID)
    assert not choice['available']
    assert choice['reason'] == 'Native Apple Silicon required'
    with pytest.raises(ValueError, match='Apple Silicon'):
        store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    assert not store.root.exists()


async def test_store_close_stops_owned_runtime(store, monkeypatch):
    stopped = []

    async def stop():
        stopped.append(True)

    monkeypatch.setattr(store.runtime, 'stop', stop)
    await store.close()
    assert stopped == [True]


def test_managed_choice_does_not_reuse_a_custom_servers_project(store, tmp_path):
    external = mac_model_profile()
    external['llm']['base_url'] = 'http://127.0.0.1:29000/v1'
    external['embeddings']['base_url'] = 'http://127.0.0.1:29001/v1'
    store.config.config_path.write_text(yaml.safe_dump({'onboarding_models': {'external': external}}))
    store = legacy_store(Config(store.config.config_path))
    custom = store.prepare('external', 'repo', str(tmp_path))
    original = store.path(custom['id']).read_bytes()
    store.runtime._state.update(phase='ready')
    managed = store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    assert managed['id'] != custom['id']
    assert store.path(custom['id']).read_bytes() == original
    settings = Config(store.path(managed['id'])).model_settings()
    assert settings.chat.base_url == store.runtime.settings.chat.base_url
    assert settings.embeddings.base_url == store.runtime.settings.embeddings.base_url
    assert managed['managed_local'] is True
    assert 'managed_local' not in custom


async def test_stopping_models_immediately_blocks_new_builds(store, tmp_path, monkeypatch):
    entered, release = asyncio.Event(), asyncio.Event()

    async def reaping():
        entered.set()
        await release.wait()

    store.runtime._state.update(phase='ready')
    monkeypatch.setattr(store.runtime, '_stop_processes', reaping)
    stopping = asyncio.create_task(store.runtime.stop())
    try:
        await asyncio.wait_for(entered.wait(), 1)
        assert store.runtime.status()['phase'] == 'stopping'
        choice = next(c for c in store.choices() if c['id'] == MAC_PROFILE_ID)
        assert not choice['available']
        with pytest.raises(ValueError, match='Start'):
            store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
        assert store.list_projects() == []
    finally:
        release.set()
        await stopping
    assert store.runtime.status()['phase'] == 'stopped'


@pytest.mark.parametrize('managed', [True, False])
async def test_rebuild_saved_managed_project_requires_runtime_but_external_does_not(
        store, tmp_path, monkeypatch, managed):
    store.config.config_path.write_text(yaml.safe_dump({
        'sources': {'repo': str(tmp_path)},
        'onboarding_models': {'external': mac_model_profile()},
    }))
    store = legacy_store(Config(store.config.config_path))
    store.runtime._state.update(phase='ready')
    project = store.prepare(MAC_PROFILE_ID if managed else 'external', 'repo')
    store.runtime._state.update(phase='stopped')
    connected = []
    built = _FakeBridge()

    async def bridge(project_id):
        connected.append(project_id)
        return built

    monkeypatch.setattr(store, 'bridge', bridge)
    app = make_app(bridge=_FakeBridge(), projects=store)
    response = await _onboard_start(_FakeReq(app=app, body={
        'project_id': project['id'], 'source': 'repo'}))
    if managed:
        assert response.status == 400
        assert b'Start local models' in response.body
        assert connected == [] and not app['jobs']
    else:
        assert response.status == 200 and connected == [project['id']]
        for job in app['jobs'].values():
            await job['task']
        assert built.calls[0][0] == 'ariadne_onboard'


@pytest.mark.parametrize('tool', ['ariadne_ask', 'ariadne_search'])
@pytest.mark.parametrize(('managed', 'phase'), [
    pytest.param(True, 'stopped', id='True-stopped'),
    pytest.param(True, 'starting', id='True-starting'),
    pytest.param(True, 'stopping', id='True-stopping'),
    pytest.param(True, 'error', id='True-error'),
    pytest.param(True, 'ready', id='True-ready'),
    pytest.param(False, 'stopped', id='False-stopped'),
])
async def test_managed_query_admission_preserves_external_projects_and_dependency_scope(
        store, tmp_path, monkeypatch, tool, phase, managed):
    store.config.config_path.write_text(yaml.safe_dump({
        'sources': {'repo': {'path': str(tmp_path), 'depends_on': ['dependency']},
                    'dependency': str(tmp_path)},
        'onboarding_models': {'external': mac_model_profile()},
    }))
    store = legacy_store(Config(store.config.config_path))
    store.runtime._state.update(phase='ready')
    project = store.prepare(MAC_PROFILE_ID if managed else 'external', 'repo')
    store.runtime._state.update(phase=phase)
    if not managed:
        def unexpected_runtime_status():
            pytest.fail('External projects must not inspect managed runtime status')

        monkeypatch.setattr(store.runtime, 'status', unexpected_runtime_status)
    bridge = _FakeBridge()
    connected = []

    async def connect(project_id):
        connected.append(project_id)
        return bridge

    monkeypatch.setattr(store, 'bridge', connect)
    field = 'question' if tool == 'ariadne_ask' else 'query'
    args = {'source': 'repo', field: 'Explain the dependency', 'role': 'developer'}
    response = await _make_tool_handler(tool)(_FakeReq(
        app={'projects': store}, body={'project_id': project['id'], **args}))
    if managed and phase != 'ready':
        assert response.status == 400
        assert not connected and not bridge.calls
        assert b'Start local models' in response.body
    else:
        assert response.status == 200
        assert connected == [project['id']]
        assert bridge.calls == [(tool, args)]
    assert Config(store.path(project['id'])).sources['repo']['depends_on'] == ['dependency']


@pytest.mark.parametrize('mismatch', ['profile', 'identity'])
def test_ready_runtime_for_a_different_pair_cannot_serve_saved_project(store, tmp_path, monkeypatch, mismatch):
    store.runtime._state.update(phase='ready')
    project = store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    saved = store.path(project['id']).read_bytes()
    status = dict(store.runtime.status(), profile_id=MAC_PROFILE_ID,
                  profile_hash=resolve_models(mac_model_profile()).profile_hash)
    status['profile_id' if mismatch == 'profile' else 'profile_hash'] = 'different'
    monkeypatch.setattr(store.runtime, 'status', lambda: status)
    with pytest.raises(ValueError, match=r'profile|model pair'):
        store.check_runtime_ready(project['id'])
    assert store.path(project['id']).read_bytes() == saved


def test_unknown_saved_managed_profile_remains_owned_and_fails_closed(store, tmp_path):
    store.runtime._state.update(phase='ready')
    project = store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    path = store.path(project['id'])
    data = yaml.safe_load(path.read_text())
    data['managed_local_profile'] = 'unknown-managed-profile'
    path.write_text(yaml.safe_dump(data))
    before = path.read_bytes()
    described = store.describe(project['id'])
    assert described['managed_local'] is True
    assert described['managed_local_profile'] == 'unknown-managed-profile'
    with pytest.raises(ValueError, match='Unknown managed'):
        store.check_runtime_ready(project['id'])
    assert path.read_bytes() == before


def test_saved_managed_marker_cannot_cover_different_pinned_weights(store, tmp_path):
    store.runtime._state.update(phase='ready')
    project = store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    path = store.path(project['id'])
    data = yaml.safe_load(path.read_text())
    data['llm']['model_revision'] = 'a' * 40
    path.write_text(yaml.safe_dump(data))
    with pytest.raises(ValueError, match=r'saved model pair|pinned profile'):
        store.check_runtime_ready(project['id'])


def test_accuracy_profile_is_distinct_and_reopening_preserves_both_saved_pairs(store, tmp_path):
    from local_models import MAC_ACCURACY_PROFILE_ID, mac_accuracy_profile

    store.runtime._state.update(phase='ready')
    legacy = store.prepare(MAC_PROFILE_ID, 'repo', str(tmp_path))
    legacy_path = store.path(legacy['id'])
    before = legacy_path.read_bytes()
    # Existing library data is not a target of profile selection or reopening.
    database = legacy_path.parent / 'library.db'
    database.write_bytes(b'existing library is preserved')
    store.runtime = MacRuntime(store.runtime.root, settings=resolve_models(mac_accuracy_profile()))
    store.runtime._state.update(phase='ready')
    accuracy = store.prepare(MAC_ACCURACY_PROFILE_ID, 'repo', str(tmp_path))
    assert accuracy['id'] != legacy['id']
    assert accuracy['managed_local_profile'] == MAC_ACCURACY_PROFILE_ID
    assert '32GB qualification target' in accuracy['memory_note']
    assert '24GB fit is unqualified' in accuracy['memory_note']
    assert legacy['managed_local_profile'] == MAC_PROFILE_ID
    assert legacy_path.read_bytes() == before
    assert database.read_bytes() == b'existing library is preserved'
    reopened = ProjectStore(store.config)
    assert {p['id']: p for p in reopened.list_projects()} == {
        legacy['id']: legacy, accuracy['id']: accuracy}
    assert Config(reopened.path(legacy['id'])).model_settings().identity == resolve_models(mac_model_profile()).identity
    assert Config(reopened.path(accuracy['id'])).model_settings().identity == resolve_models(mac_accuracy_profile()).identity
    choice = next(c for c in reopened.choices() if c['id'] == MAC_ACCURACY_PROFILE_ID)
    assert '27B thinking' in choice['label']
    assert '32GB qualification target' in choice['label']
    assert '24GB fit is unqualified' in choice['memory_note']
    assert 'not passed' in choice['quality_note']
    assert choice['qualification'] == 'unqualified'
