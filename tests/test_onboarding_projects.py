"""Model choice belongs to onboarding and survives browser/server restarts."""
import asyncio
import contextlib
import json
from pathlib import Path

import anyio
import pytest
import yaml

from config import Config
from model_config import CHAT_MODEL, EMBEDDING_MODEL
from tests.test_web_server import _FakeBridge, _FakeReq
from web.projects import ProjectStore
from web.server import _make_tool_handler, _onboard_start, _project_bridge, make_app


@pytest.fixture
def store(tmp_path, monkeypatch):
    for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.setenv(name, 'test-only')
    path = tmp_path / 'ariadne.yaml'
    path.write_text('sources: {}\n')
    return ProjectStore(Config(path))


def test_saved_choice_is_immutable_and_each_pair_has_its_own_library(store, tmp_path):
    source = tmp_path / 'repo'
    source.mkdir()
    qwen = store.prepare('qwen-hosted', 'repo', str(source))
    claude = store.prepare('claude-opus-5-5', 'repo', str(source))
    gpt = store.prepare('gpt-6-astra', 'repo', str(source))
    assert len({p['id'] for p in (qwen, claude, gpt)}) == 3
    restarted = ProjectStore(store.config)
    assert {p['id'] for p in restarted.list_projects()} == {p['id'] for p in (qwen, claude, gpt)}
    assert restarted.describe(qwen['id']) == qwen
    configs = [Config(store.path(p['id'])) for p in (qwen, claude, gpt)]
    assert len({cfg.db_path for cfg in configs}) == 3
    assert len({cfg.staleness_db_path for cfg in configs}) == 3
    assert configs[0].model_settings().embeddings.model == EMBEDDING_MODEL
    assert configs[1].model_settings().embeddings.provider == 'openai'
    assert store.prepare('qwen-hosted', 'repo', str(source)) == qwen
    assert 'test-only' not in store.path(qwen['id']).read_text()


def test_choices_show_missing_credentials_and_unconfigured_local(store, monkeypatch):
    monkeypatch.delenv('DEEPINFRA_API_KEY')
    choices = {c['id']: c for c in store.choices()}
    assert not choices['qwen-hosted']['available']
    assert choices['qwen-hosted']['preview_available']
    assert choices['qwen-hosted']['reason'] == 'Missing DEEPINFRA_API_KEY'
    assert not choices['qwen-local']['available']
    assert not choices['qwen-local']['preview_available']
    assert choices['claude-opus-5-5']['available']


def test_local_choice_needs_no_cloud_keys_or_running_servers(store, tmp_path, monkeypatch):
    from tests.local_model_server import local_config

    for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(name)
    config = store.config.config_path
    config.write_text(yaml.safe_dump({'onboarding_models': {
        'qwen-local': local_config('http://127.0.0.1:9/v1', 'http://127.0.0.1:9/v1'),
    }}))
    store = ProjectStore(Config(config))
    choice = next(c for c in store.choices() if c['id'] == 'qwen-local')
    assert choice['available']
    project = store.prepare('qwen-local', 'source', str(tmp_path))
    assert project['provider'] == 'local'
    assert not Path(Config(store.path(project['id'])).db_path).exists()


async def test_compare_pairs_without_keys_inference_or_saved_projects(store, tmp_path, monkeypatch):
    import httpx

    from ariadne_mcp.service import AriadneService
    from tests.local_model_server import local_config
    from web import projects

    source = tmp_path / 'repo'
    source.mkdir()
    (source / 'app.py').write_text('def retry(attempts):\n    return max(1, attempts)\n')
    (source / 'ignored.py').write_text('def ignored():\n    return 2\n')
    data = {'sources': {'repo': {'path': str(source), 'exclude': ['ignored.py']}},
            'onboarding_models': {'qwen-local': local_config('http://127.0.0.1:9/v1',
                                                           'http://127.0.0.1:9/v1')}}
    root = store.config.config_path
    root.write_text(yaml.safe_dump(data))
    store = ProjectStore(Config(root))
    before = root.read_bytes()
    for name in ('OPENAI_API_KEY', 'ANTHROPIC_API_KEY', 'DEEPINFRA_API_KEY'):
        monkeypatch.delenv(name)

    def offline(*_args, **_kwargs):
        raise AssertionError('Cost comparison must not call inference endpoints')

    monkeypatch.setattr(httpx.Client, 'send', offline)
    monkeypatch.setattr(httpx.AsyncClient, 'send', offline)
    preview_paths = []

    async def connect(_stack, params):
        cfg = Config(Path(params.env['ARIADNE_CONFIG']))
        preview_paths.append(cfg.config_path)
        service = AriadneService()
        service._config = cfg

        class Estimator:
            async def call(self, tool, args):
                assert tool == 'ariadne_estimate'
                result = service.estimate(**args)
                assert not await anyio.Path(cfg.db_path).exists()
                assert not await anyio.Path(cfg.staleness_db_path).exists()
                return result.model_dump()

        return Estimator()

    monkeypatch.setattr(projects, 'connect_stdio', connect)
    app = make_app(bridge=_FakeBridge(), projects=store)
    handler = _make_tool_handler('ariadne_estimate')
    estimates = {}
    for choice in ('gpt-6-astra', 'claude-opus-5-5', 'qwen-hosted', 'qwen-local'):
        response = await handler(_FakeReq(app=app, body={'source': 'repo', 'onboarding_model': choice}))
        assert response.status == 200
        estimates[choice] = json.loads(response.body)
        assert estimates[choice]['file_count'] == 1
        assert 'project_id' not in estimates[choice]
        assert store.list_projects() == [] and not store.root.exists()
        assert not preview_paths[-1].exists()
        assert root.read_bytes() == before
    assert estimates['gpt-6-astra']['grand_total_cost_usd'] > estimates['claude-opus-5-5']['grand_total_cost_usd'] > estimates['qwen-hosted']['grand_total_cost_usd'] > 0
    assert estimates['gpt-6-astra']['embedding_cost_usd'] > estimates['qwen-hosted']['embedding_cost_usd'] > 0
    assert estimates['qwen-local']['grand_total_cost_usd'] == 0
    assert not app['bridge'].calls


async def test_build_saves_only_final_choice_with_latest_scope(store, tmp_path, monkeypatch):
    source = tmp_path / 'repo'
    source.mkdir()
    # The store was constructed before Connect/discovery/scope edits.
    scope = {'path': str(source), 'exclude': ['ignored.py'], 'index_kinds': ['scip-python'],
             'doc_types_by_language': {'python': ['explanation']}}
    store.config.config_path.write_text(yaml.safe_dump({'sources': {'repo': scope, 'imported': None}}))
    built = _FakeBridge()

    async def bridge(project_id):
        cfg = Config(store.path(project_id))
        assert cfg.provider == 'deepinfra'
        assert cfg._explicit['sources']['repo'] == scope
        assert cfg.get_source_path('imported') is None
        assert not cfg.themes_enabled
        return built

    monkeypatch.setattr(store, 'bridge', bridge)
    app = make_app(bridge=_FakeBridge(), projects=store)
    # Connect cannot bind a model pair, and a saved project's model cannot be overridden.
    connect = await _make_tool_handler('ariadne_source_add')(_FakeReq(app=app, body={
        'name': 'repo', 'path': str(source), 'onboarding_model': 'qwen-hosted'}))
    assert connect.status == 400 and store.list_projects() == []
    response = await _onboard_start(_FakeReq(app=app, body={
        'source': 'repo', 'onboarding_model': 'qwen-hosted', 'types': ['explanation'], 'themes': False}))
    assert response.status == 200
    result = json.loads(response.body)
    assert store.describe(result['project_id'])['provider'] == 'deepinfra'
    assert len(store.list_projects()) == 1
    await app['jobs'][result['job_id']]['task']
    assert built.calls == [('ariadne_onboard', {'source': 'repo', 'doc_types': ['explanation']})]
    assert not app['bridge'].calls


async def test_missing_credentials_block_build_without_saving_choice(store, tmp_path, monkeypatch):
    monkeypatch.delenv('DEEPINFRA_API_KEY')
    store.config.config_path.write_text(yaml.safe_dump({'sources': {'repo': str(tmp_path)}}))
    app = make_app(bridge=_FakeBridge(), projects=store)
    response = await _onboard_start(_FakeReq(app=app, body={
        'source': 'repo', 'onboarding_model': 'qwen-hosted'}))
    assert response.status == 400
    assert 'DEEPINFRA_API_KEY' in json.loads(response.body)['error']
    assert store.list_projects() == [] and not app['jobs']


async def test_requests_and_running_jobs_stay_with_onboarded_pair(store, tmp_path, monkeypatch):
    qwen = store.prepare('qwen-hosted', 'source', str(tmp_path))
    claude = store.prepare('claude-opus-5-5', 'source', str(tmp_path))
    bridges = {p['id']: _FakeBridge() for p in (qwen, claude)}

    async def bridge(project_id):
        return bridges[project_id]

    monkeypatch.setattr(store, 'bridge', bridge)
    app = make_app(bridge=_FakeBridge(), projects=store)
    start = await _onboard_start(_FakeReq(app=app, body={
        'project_id': qwen['id'], 'source': 'source', 'model': CHAT_MODEL,
    }))
    assert start.status == 200
    ask = _make_tool_handler('ariadne_ask')
    response = await ask(_FakeReq(app=app, body={
        'project_id': claude['id'], 'source': 'source', 'question': 'How?',
    }))
    assert response.status == 200
    job = app['jobs'][json.loads(start.body)['job_id']]
    await job['task']
    assert bridges[qwen['id']].calls[0][0] == 'ariadne_onboard'
    assert bridges[claude['id']].calls == [('ariadne_ask', {'source': 'source', 'question': 'How?'})]
    assert not app['bridge'].calls
    for args in ({'model': 'gpt-6-astra'}, {'source': 'another-source'}):
        with pytest.raises(ValueError, match='project'):
            await _project_bridge(app, qwen['id'], args)
    for invalid in ('../../ariadne.yaml', '', 'f' * 24):
        with pytest.raises(ValueError, match='Unknown project'):
            await _project_bridge(app, invalid, {})


async def test_project_mcp_lifecycle_opens_once_and_closes_in_own_task(store, tmp_path, monkeypatch):
    from web import projects

    opened, closed = [], []

    @contextlib.asynccontextmanager
    async def transport():
        async with anyio.create_task_group():
            opened.append(asyncio.current_task())
            try:
                yield
            finally:
                closed.append(asyncio.current_task())

    async def connect(stack, _params):
        await stack.enter_async_context(transport())
        return _FakeBridge()

    monkeypatch.setattr(projects, 'connect_stdio', connect)
    project = store.prepare('qwen-hosted', 'source', str(tmp_path))
    a, b = await asyncio.gather(store.bridge(project['id']), store.bridge(project['id']))
    assert a is b
    await asyncio.create_task(store.close())
    assert len(opened) == 1 and opened == closed
