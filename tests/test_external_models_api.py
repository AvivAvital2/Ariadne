"""Explicit endpoint checks save deployments, not source edits or library migrations."""

import asyncio
import json

import pytest
from aiohttp.test_utils import TestClient, TestServer

from config import Config
from tests.test_external_models import endpoints as _endpoints
from tests.test_external_models import request_for
from tests.test_web_server import _FakeBridge
from web.projects import ProjectStore
from web.server import make_app

endpoints = _endpoints


@pytest.fixture
async def external_client(tmp_path):
    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources:\n  consumer:\n    path: .\n    depends_on: [shared]\n'
                    '  shared:\n    path: .\n')
    store = ProjectStore(Config(path))
    async with TestClient(TestServer(make_app(bridge=_FakeBridge(), projects=store))) as client:
        yield client, store


async def test_ui_check_saves_selectable_profile_and_preserves_sources(external_client, endpoints):
    client, store = external_client
    original = store.config.config_path.read_bytes()
    response = await client.post('/api/external-models', json=request_for(endpoints))
    assert response.status == 201
    body = await response.json()
    assert body['qualification'] == 'unqualified'
    choice = next(m for m in body['models'] if m['id'] == body['profile_id'])
    assert choice['available'] and choice['user_managed'] and not choice.get('managed_local')
    assert body['checks']['chat']['completed']
    assert body['checks']['embeddings']['dimensions'] == 3072
    assert store.config.config_path.read_bytes() == original
    assert store.list_projects() == []
    reopened = ProjectStore(Config(store.config.config_path))
    assert reopened.profiles()[body['profile_id']]['llm']['base_url'] == endpoints[0].base_url
    project = reopened.prepare(body['profile_id'], 'consumer')
    saved = Config(reopened.path(project['id']))
    assert saved.scope_closure('consumer') == frozenset({'consumer', 'shared'})
    assert saved.model_settings().chat.base_url == endpoints[0].base_url
    assert saved._explicit['external_model_profile'] == body['profile_id']
    assert 'managed_local_profile' not in saved._explicit


def test_changed_deployment_gets_new_project_without_rebinding_old(tmp_path):
    # Separate synchronous store because this test does not contact endpoints.
    from web.external_models import build_profile

    path = tmp_path / 'standalone.yaml'
    path.write_text('sources: {}\n')
    store = ProjectStore(Config(path))
    ids = []
    saved = []
    for host in ('first', 'second'):
        profile = build_profile({'chat_url': f'https://{host}.example/v1',
                                 'embedding_url': 'https://embed.example/v1'})
        choice = store.save_external_profile(profile)
        project = store.prepare(choice, 'repo', str(tmp_path))
        ids.append(project['id'])
        saved.append((store.path(project['id']), store.path(project['id']).read_bytes()))
    assert ids[0] != ids[1]
    assert all(path.read_bytes() == before for path, before in saved)
    assert len(store.list_projects()) == 2


async def test_failed_probe_saves_nothing(external_client, endpoints):
    client, store = external_client
    before = store.config.config_path.read_bytes()
    endpoints[0].overrides['/native/v1/chat/completions'] = {'choices': []}
    response = await client.post('/api/external-models', json=request_for(endpoints))
    assert response.status == 400
    assert 'Chat' in (await response.json())['error']
    assert store.config.config_path.read_bytes() == before
    assert not store.root.exists()


@pytest.mark.parametrize('body', [[], None, {'chat_url': 'http://host/v1'},
                                {'chat_url': 'http://localhost:bad/v1',
                                 'embedding_url': 'http://host/v1'},
                                {'chat_url': 'http://host/v1', 'embedding_url': 'http://host/v1',
                                 'api_key': 'secret'}])
async def test_invalid_request_cannot_save_or_dispatch(external_client, body, monkeypatch):
    client, store = external_client

    async def unexpected(_profile):
        pytest.fail('Invalid request dispatched inference')

    monkeypatch.setattr('web.external_models.probe_profile', unexpected)
    response = await client.post('/api/external-models', data=json.dumps(body),
                                 headers={'Content-Type': 'application/json'})
    assert response.status == 400
    assert not store.root.exists()


async def test_endpoint_actions_require_same_origin_and_json(external_client, endpoints):
    client, store = external_client
    response = await client.post('/api/external-models', json=request_for(endpoints),
                                 headers={'Origin': 'https://unrelated.example'})
    assert response.status == 403
    response = await client.post('/api/external-models', data=json.dumps(request_for(endpoints)),
                                 headers={'Content-Type': 'text/plain'})
    assert response.status == 400
    assert not store.root.exists() and all(not s.requests for s in endpoints)


async def test_invalid_credential_header_is_not_logged_or_saved(
        external_client, endpoints, monkeypatch, caplog):
    client, store = external_client
    sentinel = 'fixture-private-header-sentinel\n'
    monkeypatch.setenv('ARIADNE_HEADER_TEST', sentinel)
    response = await client.post('/api/external-models', json={
        **request_for(endpoints), 'chat_api_key_env': 'ARIADNE_HEADER_TEST'})
    assert response.status == 400
    assert 'Server readiness' in (await response.json())['error']
    assert sentinel.strip() not in caplog.text
    assert not store.root.exists() and all(not s.requests for s in endpoints)


@pytest.mark.parametrize('endpoint', [0, 1])
async def test_small_server_context_cannot_save_unusable_generation_profile(
        external_client, endpoints, endpoint):
    client, store = external_client
    endpoints[endpoint].context = 4096
    response = await client.post('/api/external-models', json=request_for(endpoints))
    assert response.status == 400
    assert 'context' in (await response.json())['error']
    assert not store.root.exists()
    assert not any(path.endswith('/chat/completions') for path, _ in endpoints[0].requests)


async def test_only_one_explicit_check_runs_at_a_time(external_client, monkeypatch):
    client, _store = external_client
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_probe(_profile):
        entered.set()
        await release.wait()
        return {'chat': {'completed': True}, 'embeddings': {'dimensions': 3072}}

    monkeypatch.setattr('web.external_models.probe_profile', slow_probe)
    body = {'chat_url': 'https://chat.example/v1', 'embedding_url': 'https://embed.example/v1'}
    first = asyncio.create_task(client.post('/api/external-models', json=body))
    try:
        await asyncio.wait_for(entered.wait(), 2)
        second = await client.post('/api/external-models', json=body)
        assert second.status == 409
    finally:
        release.set()
        response = await first
    assert response.status == 201


def test_registry_write_failure_preserves_previous_profiles(tmp_path, monkeypatch):
    from web.external_models import build_profile

    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    store = ProjectStore(Config(path))
    profile = build_profile({'chat_url': 'https://first.example/v1',
                             'embedding_url': 'https://embed.example/v1'})
    first = store.save_external_profile(profile)
    before = store.external_profiles_path.read_bytes()

    def fail_replace(*_args):
        raise OSError('simulated disk failure')

    monkeypatch.setattr('web.external_models.os.replace', fail_replace)
    profile['llm']['base_url'] = 'https://other.example/v1'
    with pytest.raises(OSError, match='simulated disk failure'):
        store.save_external_profile(profile)
    assert store.external_profiles_path.read_bytes() == before
    assert first in store.profiles()
    assert not list(store.root.glob('*.tmp'))


def test_longer_timeout_round_trips_without_changing_existing_profiles_or_projects(tmp_path):
    from web.external_models import build_profile, load_profiles, save_profile

    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    store = ProjectStore(Config(path))
    body = {'chat_url': 'https://chat.example/v1',
            'embedding_url': 'https://embed.example/v1'}
    previous = build_profile(body)
    original_id = store.save_external_profile(previous)
    original_registry = store.external_profiles_path.read_bytes()
    original_project = store.prepare(original_id, 'repo', str(tmp_path))
    original_config = store.path(original_project['id']).read_bytes()

    assert load_profiles(store.external_profiles_path) == {original_id: previous}
    assert save_profile(store.external_profiles_path, previous) == original_id
    assert store.external_profiles_path.read_bytes() == original_registry

    profile = build_profile({**body, 'request_timeout_seconds': 7200})
    selected_id = store.save_external_profile(profile)
    reopened = ProjectStore(Config(path))
    assert reopened.profiles()[original_id] == previous
    assert reopened.profiles()[selected_id] == profile
    assert selected_id != original_id
    selected_project = reopened.prepare(selected_id, 'repo', str(tmp_path))
    assert selected_project['id'] != original_project['id']
    assert store.path(original_project['id']).read_bytes() == original_config
    settings = Config(reopened.path(selected_project['id'])).model_settings()
    assert settings.chat.timeout_seconds == settings.embeddings.timeout_seconds == 7200
    assert settings.identity == Config(store.path(original_project['id'])).model_settings().identity


@pytest.mark.parametrize(('chat_timeout', 'embedding_timeout'), [
    (600, 7200), (7200, 600), (None, 7200), (7200, None),
    (599, 599), (14401, 14401), (True, True), ('7200', '7200'),
])
def test_saved_timeout_tampering_cannot_be_saved_or_loaded(tmp_path, chat_timeout, embedding_timeout):
    from web.external_models import build_profile, load_profiles, profile_id, save_profile

    profile = build_profile({'chat_url': 'https://chat.example/v1',
                             'embedding_url': 'https://embed.example/v1',
                             'request_timeout_seconds': 7200})
    for section, seconds in (('llm', chat_timeout), ('embeddings', embedding_timeout)):
        if seconds is None:
            profile[section].pop('timeout_seconds')
        else:
            profile[section]['timeout_seconds'] = seconds
    path = tmp_path / 'profiles.json'
    message = ('Saved user-managed model profile differs from the supported contract'
               '|Request timeout must be an integer from 600 to 14400 seconds')
    with pytest.raises(ValueError, match=message):
        save_profile(path, profile)
    assert not path.exists()
    path.write_text(json.dumps({'version': 1, 'profiles': {profile_id(profile): profile}}))
    with pytest.raises(ValueError, match=message):
        load_profiles(path)


async def test_api_persists_selected_timeout_for_both_clients(external_client, monkeypatch):
    client, store = external_client
    observed = []

    async def checked(profile):
        observed.append(profile)
        return {'chat': {'completed': True}, 'embeddings': {'dimensions': 3072}}

    monkeypatch.setattr('web.external_models.probe_profile', checked)
    response = await client.post('/api/external-models', json={
        'chat_url': 'https://chat.example/v1', 'embedding_url': 'https://embed.example/v1',
        'request_timeout_seconds': 14400})
    assert response.status == 201
    body = await response.json()
    assert len(observed) == 1
    assert observed[0]['llm']['timeout_seconds'] == 14400
    assert observed[0]['embeddings']['timeout_seconds'] == 14400
    reopened = ProjectStore(Config(store.config.config_path))
    assert reopened.profiles()[body['profile_id']] == observed[0]


@pytest.mark.parametrize('seconds', [None, True, 599, 14401, 600.5, '7200'])
async def test_api_rejects_invalid_timeout_before_probe(external_client, monkeypatch, seconds):
    client, store = external_client

    async def unexpected(_profile):
        pytest.fail('Invalid timeout dispatched inference')

    monkeypatch.setattr('web.external_models.probe_profile', unexpected)
    response = await client.post('/api/external-models', json={
        'chat_url': 'https://chat.example/v1', 'embedding_url': 'https://embed.example/v1',
        'request_timeout_seconds': seconds})
    assert response.status == 400
    assert '600 to 14400 seconds' in (await response.json())['error']
    assert not store.root.exists()
