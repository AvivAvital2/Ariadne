"""Project interruption owns process cleanup, without changing persisted work."""

import asyncio
import os
import sys
import textwrap

import pytest
import yaml
from mcp import StdioServerParameters

from config import Config
from docgen.spool import SpoolStore
from inference import InferenceContext
from local_models import MAC_PROFILE_IDS, mac_model_profile, managed_model_profile
from model_config import resolve_models
from web.projects import ProjectSession, ProjectStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'test-only')
    source = tmp_path / 'source'
    dependency = tmp_path / 'dependency'
    source.mkdir()
    dependency.mkdir()
    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump({'sources': {
        'source': {'path': str(source), 'depends_on': ['dependency']},
        'dependency': {'path': str(dependency)},
    }}))
    return ProjectStore(Config(path))


class ClosingSession:
    def __init__(self):
        self.entered = asyncio.Event()
        self.release = asyncio.Event()
        self.closed = False
        self.close_calls = 0

    async def close(self):
        self.close_calls += 1
        self.entered.set()
        await self.release.wait()
        self.closed = True


async def test_interrupt_waits_for_owned_session_cleanup_and_preserves_project(store):
    project = store.prepare('qwen-hosted', 'source')
    path = store.path(project['id'])
    before = path.read_bytes()
    session = store._sessions[project['id']] = ClosingSession()
    interrupted = asyncio.create_task(store.interrupt(project['id']))
    try:
        await asyncio.wait_for(session.entered.wait(), 1)
        assert not interrupted.done()
        assert not session.closed
    finally:
        session.release.set()
        await interrupted
    assert session.closed
    assert project['id'] not in store._sessions
    assert path.read_bytes() == before
    assert Config(path).sources['source']['depends_on'] == ['dependency']
    assert store.describe(project['id']) == project


async def test_interrupt_saved_project_without_session_does_not_start_one(store):
    project = store.prepare('qwen-hosted', 'source')
    before = store.path(project['id']).read_bytes()
    await store.interrupt(project['id'])
    assert store._sessions == {}
    assert store.path(project['id']).read_bytes() == before


@pytest.mark.parametrize('project_id', [None, '../outside', 'f' * 24])
async def test_interrupt_validates_project_before_touching_owned_sessions(store, project_id):
    project = store.prepare('qwen-hosted', 'source')
    session = store._sessions[project['id']] = ClosingSession()
    with pytest.raises(ValueError, match='Unknown project'):
        await store.interrupt(project_id)
    assert not session.entered.is_set()
    assert store._sessions[project['id']] is session


@pytest.mark.parametrize('profile_id', MAC_PROFILE_IDS)
async def test_interrupt_managed_leaves_external_and_hosted_sessions_running(store, monkeypatch, profile_id):
    data = store.config._explicit
    data['onboarding_models'] = {'external': mac_model_profile()}
    store.config.config_path.write_text(yaml.safe_dump(data))
    store = ProjectStore(Config(store.config.config_path))
    monkeypatch.setattr('local_runtime.platform_error', lambda: None)
    store.runtime.settings = resolve_models(managed_model_profile(profile_id))
    store.runtime._state.update(phase='ready')
    managed = [store.prepare(profile_id, name) for name in ('source', 'dependency')]
    external = store.prepare('external', 'source')
    hosted = store.prepare('qwen-hosted', 'source')
    idle = store.prepare(profile_id, 'idle', str(store.config.config_dir))
    projects = [*managed, external, hosted]
    before = {project['id']: store.path(project['id']).read_bytes() for project in projects}
    sessions = {project['id']: ClosingSession() for project in projects}
    for session in sessions.values():
        session.release.set()
    store._sessions.update(sessions)
    await store.interrupt_managed()
    assert all(sessions[project['id']].closed for project in managed)
    assert not sessions[external['id']].entered.is_set()
    assert not sessions[hosted['id']].entered.is_set()
    assert set(store._sessions) == {external['id'], hosted['id']}
    assert idle['id'] not in store._sessions
    assert store.runtime.status()['phase'] == 'ready'
    assert all(store.path(project_id).read_bytes() == content
               for project_id, content in before.items())


async def test_cancelled_interrupt_still_finishes_cleanup(store):
    project = store.prepare('qwen-hosted', 'source')
    session = store._sessions[project['id']] = ClosingSession()
    interrupted = asyncio.create_task(store.interrupt(project['id']))
    await asyncio.wait_for(session.entered.wait(), 1)
    interrupted.cancel()
    await asyncio.sleep(0)
    session.release.set()
    with pytest.raises(asyncio.CancelledError):
        await interrupted
    assert session.closed
    assert project['id'] not in store._sessions


async def test_concurrent_interruptions_share_one_cleanup(store):
    project = store.prepare('qwen-hosted', 'source')
    session = store._sessions[project['id']] = ClosingSession()
    first = asyncio.create_task(store.interrupt(project['id']))
    await asyncio.wait_for(session.entered.wait(), 1)
    second = asyncio.create_task(store.interrupt(project['id']))
    try:
        await asyncio.sleep(0)
        session.release.set()
        await asyncio.gather(first, second)
        assert session.close_calls == 1
        assert project['id'] not in store._sessions
    finally:
        session.release.set()
        await asyncio.gather(first, second, return_exceptions=True)


@pytest.mark.parametrize('ignore_term', [False, True])
async def test_interrupt_reaps_blocked_mcp_process_and_next_connection_is_fresh(
        store, tmp_path, monkeypatch, ignore_term):
    """Use the installed SDK transport; the child blocks without making inference."""
    script = textwrap.dedent('''
        import json
        import os
        import signal
        import sys
        import time
        from pathlib import Path

        directory = Path(sys.argv[1])
        if sys.argv[2] == 'True':
            signal.signal(signal.SIGTERM, signal.SIG_IGN)
        (directory / 'child.pid').write_text(str(os.getpid()))
        for line in sys.stdin:
            request = json.loads(line)
            method = request['method']
            if 'id' not in request:
                continue
            if method == 'initialize':
                result = {'protocolVersion': request['params']['protocolVersion'],
                          'capabilities': {'tools': {}},
                          'serverInfo': {'name': 'disposable-test', 'version': '1'}}
            elif method == 'tools/list':
                result = {'tools': [{'name': name, 'inputSchema': {'type': 'object'}}
                                    for name in ('block', 'ping')]}
            elif method == 'tools/call' and request['params']['name'] == 'block':
                print(json.dumps({'jsonrpc': '2.0', 'method': 'notifications/progress',
                                  'params': {'progress': 1, 'progressToken':
                                             request['params']['_meta']['progressToken']}}),
                      flush=True)
                time.sleep(3600)
            elif method == 'tools/call':
                result = {'content': [{'type': 'text', 'text': 'alive'}],
                          'structuredContent': {'pid': os.getpid()}}
            else:
                raise AssertionError(method)
            print(json.dumps({'jsonrpc': '2.0', 'id': request['id'], 'result': result}),
                  flush=True)
    ''')
    monkeypatch.setattr('web.projects.stdio_server_params', lambda _path: StdioServerParameters(
        command=sys.executable,
        args=['-I', '-u', '-c', script, str(tmp_path), str(ignore_term)], env={}))
    project = store.prepare('qwen-hosted', 'source')
    bridge = await asyncio.wait_for(store.bridge(project['id']), 10)
    original = store._sessions[project['id']]
    pid = int((tmp_path / 'child.pid').read_text())
    entered = asyncio.Event()

    async def on_progress(*_args):
        entered.set()

    blocked = asyncio.create_task(bridge.call('block', progress_callback=on_progress))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        await asyncio.wait_for(store.interrupt(project['id']), 10)
        assert original.task.done()
        assert project['id'] not in store._sessions
        with pytest.raises(ProcessLookupError):
            os.kill(pid, 0)
        # MCP transport teardown owns the child; the web job separately owns
        # cancellation of its waiting call (the SDK does not finish that waiter).
        blocked.cancel()
        result = await asyncio.wait_for(asyncio.gather(blocked, return_exceptions=True), 2)
        assert isinstance(result[0], BaseException)
        reopened = await asyncio.wait_for(store.bridge(project['id']), 10)
        assert reopened is not bridge
        assert (await reopened.call('ping'))['pid'] != pid
    finally:
        if not blocked.done():
            blocked.cancel()
        await asyncio.gather(blocked, return_exceptions=True)
        await original.close()
        await store.close()


@pytest.mark.parametrize('profile_id', MAC_PROFILE_IDS)
async def test_interrupt_preserves_library_binding_and_uncancelled_durable_run(store, monkeypatch, profile_id):
    monkeypatch.setattr('local_runtime.platform_error', lambda: None)
    store.runtime.settings = resolve_models(managed_model_profile(profile_id))
    store.runtime._state.update(phase='ready')
    project = store.prepare(profile_id, 'source')
    config = Config(store.path(project['id']))
    context = InferenceContext.from_config(config)
    with context.write_session(), context.open_library(operation='write') as library:
        document = library.add_document(content_type='finding', title='Existing knowledge',
                                        content='Prior operator note.', source_name='source')
        run_id = SpoolStore(context, library).create(
            [], operation='generate', scope={'source': 'source', 'dependencies': ['dependency']})
    binding = context.binding
    run_before = SpoolStore(context).get_run(run_id)
    await context.aclose()
    paths = [store.path(project['id']), context.db_path, context.staleness_db_path]
    before = {path: path.read_bytes() for path in paths}
    session = store._sessions[project['id']] = ClosingSession()
    session.release.set()
    await store.interrupt(project['id'])
    assert all(path.read_bytes() == content for path, content in before.items())
    reopened = InferenceContext.from_config(config)
    try:
        assert reopened.binding == binding
        assert SpoolStore(reopened).get_run(run_id) == run_before
        assert run_before['state'] == 'running' and not run_before['cancel_requested']
        with reopened.open_library(operation='inspect') as library:
            assert library.get_document(document.id).content == 'Prior operator note.'
    finally:
        await reopened.aclose()


async def test_interrupt_reaps_owned_process_waiting_for_mcp_initialization(store, monkeypatch):
    from mcp.client import stdio

    created = asyncio.Event()
    processes = []
    create_process = stdio._create_platform_compatible_process

    async def record_process(*args, **kwargs):
        process = await create_process(*args, **kwargs)
        processes.append(process)
        created.set()
        return process

    monkeypatch.setattr(stdio, '_create_platform_compatible_process', record_process)
    monkeypatch.setattr('web.projects.stdio_server_params', lambda _path: StdioServerParameters(
        command=sys.executable, args=['-I', '-c', 'import time; time.sleep(3600)'], env={}))
    project = store.prepare('qwen-hosted', 'source')
    connecting = asyncio.create_task(store.bridge(project['id']))
    await asyncio.wait_for(created.wait(), 5)
    original = store._sessions[project['id']]
    interrupted = asyncio.create_task(store.interrupt(project['id']))
    try:
        done, _pending = await asyncio.wait([interrupted], timeout=5)
        assert interrupted in done, 'Interruption must not wait forever for MCP initialization'
        await interrupted
        assert processes[0].returncode is not None
        with pytest.raises(ProcessLookupError):
            os.kill(processes[0].pid, 0)
        assert project['id'] not in store._sessions
        result = await asyncio.wait_for(asyncio.gather(connecting, return_exceptions=True), 2)
        assert isinstance(result[0], RuntimeError)
    finally:
        if processes[0].returncode is None:
            processes[0].kill()
            await processes[0].wait()
        connecting.cancel()
        original.task.cancel()
        await asyncio.gather(connecting, original.task, interrupted, return_exceptions=True)


async def test_close_before_connection_task_starts_finishes_waiting_connector(tmp_path):
    session = ProjectSession(tmp_path / 'ariadne.yaml')
    connecting = asyncio.create_task(session.connect())
    # connect() queues the owner task, but it has not entered its exit stack yet.
    await asyncio.sleep(0)
    try:
        await session.close()
        done, _pending = await asyncio.wait([connecting], timeout=1)
        assert connecting in done, 'Closing before startup must wake the connection waiter'
        with pytest.raises(RuntimeError, match='interrupted during startup'):
            await connecting
    finally:
        connecting.cancel()
        await asyncio.gather(connecting, return_exceptions=True)
