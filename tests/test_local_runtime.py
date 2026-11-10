import asyncio
import hashlib
import io
import os
import socket
import sys
import tarfile
import threading
from dataclasses import replace

import httpx
import pytest


@pytest.mark.asyncio
async def test_download_verifies_bytes_before_installing_and_reuses_valid_file(tmp_path):
    from local_runtime import download_file

    content = b'a pinned artifact'
    digest = hashlib.sha256(content).hexdigest()
    requests = []
    progress = []
    destination = tmp_path / 'models' / 'model.gguf'

    def serve(request):
        requests.append(request)
        assert not destination.exists()
        return httpx.Response(200, content=content)

    async with httpx.AsyncClient(transport=httpx.MockTransport(serve)) as client:
        await download_file(client, 'https://models.example/pinned', destination,
                            size=len(content), sha256=digest,
                            progress=lambda done, total: progress.append((done, total)))
        assert destination.read_bytes() == content
        assert progress[-1] == (len(content), len(content))
        await download_file(client, 'https://models.example/pinned', destination,
                            size=len(content), sha256=digest)
    assert len(requests) == 1
    assert list(destination.parent.iterdir()) == [destination]


@pytest.mark.parametrize('content', [b'corrupt payload', b'too short', b'too long' * 10])
@pytest.mark.asyncio
async def test_corrupt_download_never_replaces_an_existing_file(tmp_path, content):
    from local_runtime import RuntimeSetupError, download_file

    original = b'previous file'
    destination = tmp_path / 'model.gguf'
    destination.write_bytes(original)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, content=content))) as client:
        with pytest.raises(RuntimeSetupError, match='verification'):
            await download_file(client, 'https://models.example/pinned', destination,
                                size=len(b'correct payload'),
                                sha256=hashlib.sha256(b'correct payload').hexdigest())
    assert destination.read_bytes() == original
    assert list(tmp_path.iterdir()) == [destination]


def runtime_archive(path, members):
    with tarfile.open(path, 'w:gz') as archive:
        for name, content in members.items():
            info = tarfile.TarInfo(name)
            info.mode = 0o755
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))


def test_runtime_install_keeps_native_libraries_beside_executable(tmp_path):
    from local_runtime import unpack_runtime

    archive = tmp_path / 'runtime.tar.gz'
    runtime_archive(archive, {'llama-b11669/llama-server': b'executable',
                              'llama-b11669/libggml-metal.dylib': b'metal library'})
    executable = unpack_runtime(archive, tmp_path / 'installed')
    assert executable.read_bytes() == b'executable'
    assert executable.stat().st_mode & 0o100
    assert (executable.parent / 'libggml-metal.dylib').read_bytes() == b'metal library'


@pytest.mark.parametrize('member', ['../outside', '/outside', 'other/llama-server'])
def test_runtime_install_rejects_files_outside_pinned_directory(tmp_path, member):
    from local_runtime import RuntimeSetupError, unpack_runtime

    archive = tmp_path / 'runtime.tar.gz'
    runtime_archive(archive, {member: b'unexpected'})
    with pytest.raises(RuntimeSetupError):
        unpack_runtime(archive, tmp_path / 'installed')
    assert not (tmp_path / 'outside').exists()


def test_server_commands_use_pinned_files_and_bounded_native_settings(tmp_path):
    from local_models import MAC_ARTIFACTS, mac_model_profile
    from local_runtime import server_command
    from model_config import resolve_models

    settings = resolve_models(mac_model_profile())
    executable = tmp_path / 'llama-server'
    chat = server_command(settings, 'llm', executable, tmp_path)
    embed = server_command(settings, 'embeddings', executable, tmp_path)
    for command, role in [(chat, 'llm'), (embed, 'embeddings')]:
        assert command[0] == str(executable)
        assert command[command.index('--model') + 1] == str(tmp_path / MAC_ARTIFACTS[role].filename)
        assert command[command.index('--host') + 1] == '127.0.0.1'
        assert command[command.index('--parallel') + 1] == '1'
        assert command[command.index('--n-gpu-layers') + 1] == 'all'
        assert command[command.index('--fit') + 1] == 'off'
        assert command[command.index('--cache-ram') + 1] == '0'
    assert chat[chat.index('--ctx-size') + 1] == '32768'
    assert '--no-context-shift' in chat
    assert '--jinja' in chat
    assert embed[embed.index('--pooling') + 1] == 'last'
    assert embed[embed.index('--ubatch-size') + 1] == '8192'
    assert '--embedding' in embed


@pytest.mark.parametrize('mutation', ['chat_pin', 'embedding_pin', 'tokenizer', 'remote', 'same_port'])
def test_managed_server_cannot_misrepresent_artifacts_or_bind_remote_ports(tmp_path, mutation):
    from local_models import mac_model_profile
    from local_runtime import RuntimeSetupError, server_command
    from model_config import resolve_models

    settings = resolve_models(mac_model_profile())
    if mutation == 'chat_pin':
        settings = replace(settings, chat=replace(settings.chat, model_revision='f' * 40))
    elif mutation == 'embedding_pin':
        settings = replace(settings, embeddings=replace(settings.embeddings, model_revision='f' * 40))
    elif mutation == 'tokenizer':
        settings = replace(settings, embeddings=replace(settings.embeddings, tokenizer_revision='f' * 40))
    elif mutation == 'remote':
        settings = replace(settings, chat=replace(settings.chat, base_url='http://example.com:8000/v1'))
    else:
        settings = replace(settings, embeddings=replace(settings.embeddings, base_url=settings.chat.base_url))
    with pytest.raises(RuntimeSetupError):
        server_command(settings, 'llm', tmp_path / 'llama-server', tmp_path)


@pytest.fixture
def managed_runtime(tmp_path, monkeypatch, unused_tcp_port_factory):
    import local_runtime
    from local_models import mac_model_profile
    from model_config import resolve_models

    data = mac_model_profile()
    for role in ('llm', 'embeddings'):
        data[role]['base_url'] = f'http://127.0.0.1:{unused_tcp_port_factory()}/v1'
    runtime = local_runtime.MacRuntime(tmp_path, settings=resolve_models(data))

    async def installed():
        return tmp_path / 'llama-server'

    async def ready(_settings):
        return {'chat': {'server_context': 32768}, 'embeddings': {'server_context': 8192}}

    monkeypatch.setattr(runtime, 'verify_installation', installed)
    monkeypatch.setattr(local_runtime, 'check_endpoints', ready)
    # Real subprocesses exercise ownership and cleanup; they perform no inference.
    monkeypatch.setattr(local_runtime, 'server_command', lambda *_args: [
        sys.executable, '-c', 'import time; time.sleep(60)'])
    return runtime


@pytest.mark.asyncio
async def test_managed_start_and_stop_own_only_their_processes(managed_runtime):
    runtime = managed_runtime
    unrelated = await asyncio.create_subprocess_exec(sys.executable, '-c', 'import time; time.sleep(60)')
    try:
        await runtime.start(timeout=2)
        children = list(runtime.processes.values())
        assert len(children) == 2
        assert all(child.stdin is not None for child in children)
        assert runtime.status()['phase'] == 'ready'
        await runtime.start(timeout=2)
        assert list(runtime.processes.values()) == children
        await runtime.stop()
        await runtime.stop()
        assert all(child.returncode is not None for child in children)
        assert unrelated.returncode is None
        assert runtime.status()['phase'] == 'stopped'
    finally:
        await runtime.stop()
        unrelated.terminate()
        await unrelated.wait()


async def test_runtime_uses_its_own_supervisor_from_an_untrusted_working_directory(
        managed_runtime, tmp_path, monkeypatch):
    import local_runtime

    workspace = tmp_path / 'workspace'
    workspace.mkdir()
    marker = workspace / 'shadow-executed'
    shadow = f'from pathlib import Path; Path({str(marker)!r}).touch(); import time; time.sleep(60)'
    (workspace / 'local_runtime_worker.py').write_text(shadow)
    (workspace / 'threading.py').write_text(shadow)
    monkeypatch.chdir(workspace)
    monkeypatch.setenv('PYTHONPATH', str(workspace))
    pid_files = [tmp_path / f'{role}.pid' for role in ('llm', 'embeddings')]

    def command(_settings, role, _executable, _models):
        pid_file = tmp_path / f'{role}.pid'
        return [sys.executable, '-I', '-c',
                f'import os,time; from pathlib import Path; '
                f'Path({str(pid_file)!r}).write_text(str(os.getpid())); time.sleep(60)']

    monkeypatch.setattr(local_runtime, 'server_command', command)
    try:
        await managed_runtime.start(timeout=5)
        async with asyncio.timeout(5):
            while not all(path.exists() for path in pid_files):
                assert not marker.exists(), 'Runtime executed code from the source repository'
                await asyncio.sleep(0.01)
        assert not marker.exists()
    finally:
        await managed_runtime.stop()
    for path in pid_files:
        with pytest.raises(ProcessLookupError):
            os.kill(int(path.read_text()), 0)


@pytest.mark.asyncio
async def test_start_failure_cleans_up_first_child(managed_runtime, monkeypatch):
    runtime = managed_runtime
    original = asyncio.create_subprocess_exec
    children = []

    async def launch(*args, **kwargs):
        if children:
            raise OSError('second model could not start')
        process = await original(*args, **kwargs)
        children.append(process)
        return process

    monkeypatch.setattr(asyncio, 'create_subprocess_exec', launch)
    with pytest.raises(OSError, match='second model'):
        await runtime.start(timeout=2)
    assert children[0].returncode is not None
    assert runtime.processes == {}
    assert runtime.status()['phase'] == 'error'


@pytest.mark.asyncio
@pytest.mark.parametrize('accuracy', [False, True])
async def test_setup_downloads_and_verifies_runtime_and_models(tmp_path, monkeypatch, accuracy):
    import local_runtime
    from local_models import ModelArtifact, mac_accuracy_profile, mac_model_profile
    from model_config import resolve_models

    archive = tmp_path / 'test.tar.gz'
    runtime_archive(archive, {'llama-b11669/llama-server': b'executable',
                              'llama-b11669/libggml-metal.dylib': b'metal library'})
    archive_bytes = archive.read_bytes()
    monkeypatch.setattr(local_runtime, 'RUNTIME_SIZE', len(archive_bytes))
    monkeypatch.setattr(local_runtime, 'RUNTIME_SHA256', hashlib.sha256(archive_bytes).hexdigest())
    monkeypatch.setattr(local_runtime, 'platform_error', lambda: None)
    artifacts = {role: ModelArtifact('owner/model', 'f' * 40, f'{role}.gguf',
                                   hashlib.sha256(role.encode()).hexdigest(), len(role))
                 for role in ('llm', 'embeddings')}
    monkeypatch.setattr(local_runtime, 'MAC_ACCURACY_ARTIFACTS' if accuracy else 'MAC_ARTIFACTS', artifacts)
    responses = {a.url: role.encode() for role, a in artifacts.items()}
    responses[local_runtime.RUNTIME_URL] = archive_bytes
    settings = resolve_models(mac_accuracy_profile() if accuracy else mac_model_profile())
    runtime = local_runtime.MacRuntime(tmp_path / 'installed', settings=settings)
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, content=responses[str(request.url)]))) as client:
        await runtime.setup(client=client)
    executable = await runtime.verify_installation()
    assert executable.read_bytes() == b'executable'
    assert runtime.status()['phase'] == 'stopped'
    for role in artifacts:
        assert (runtime.root / 'models' / f'{role}.gguf').read_bytes() == role.encode()


@pytest.mark.asyncio
async def test_occupied_port_never_starts_or_kills_a_process(managed_runtime):
    from urllib.parse import urlsplit

    runtime = managed_runtime
    with socket.socket() as owner:
        owner.bind(('127.0.0.1', urlsplit(runtime.settings.chat.base_url).port))
        with pytest.raises(RuntimeError, match='already in use'):
            await runtime.start(timeout=1)
        assert runtime.processes == {}
        assert owner.fileno() >= 0


@pytest.mark.asyncio
async def test_start_timeout_and_cancellation_cleanup_children(managed_runtime, monkeypatch):
    import local_runtime

    children = []
    entered = asyncio.Event()

    async def not_ready(_settings):
        children[:] = managed_runtime.processes.values()
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(local_runtime, 'check_endpoints', not_ready)
    with pytest.raises(TimeoutError):
        await managed_runtime.start(timeout=0.1)
    assert 'timed out' in managed_runtime.status()['error']
    assert all(child.returncode is not None for child in children)
    assert managed_runtime.processes == {}
    entered.clear()
    task = asyncio.create_task(managed_runtime.start(timeout=10))
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert all(child.returncode is not None for child in children)
    assert managed_runtime.status()['phase'] == 'stopped'


@pytest.mark.asyncio
async def test_process_exit_during_readiness_cannot_be_reported_ready(managed_runtime, monkeypatch):
    import local_runtime

    async def exits(_settings):
        for process in managed_runtime.processes.values():
            process.terminate()
            await process.wait()
        return {}

    monkeypatch.setattr(local_runtime, 'check_endpoints', exits)
    with pytest.raises(local_runtime.RuntimeSetupError, match='exited'):
        await managed_runtime.start(timeout=2)
    assert managed_runtime.status()['phase'] == 'error'


@pytest.mark.asyncio
async def test_stop_interrupts_loading_instead_of_waiting_for_startup_timeout(managed_runtime, monkeypatch):
    import local_runtime

    entered = asyncio.Event()

    async def loading(_settings):
        entered.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(local_runtime, 'check_endpoints', loading)
    operation = asyncio.create_task(managed_runtime.start(timeout=60))
    try:
        await entered.wait()
        children = list(managed_runtime.processes.values())
        await asyncio.wait_for(managed_runtime.stop(), timeout=1)
        assert operation.cancelled()
        assert all(child.returncode is not None for child in children)
        assert managed_runtime.status()['phase'] == 'stopped'
    finally:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_stop_cancels_download_and_removes_partial_file(tmp_path, monkeypatch):
    import local_runtime

    entered = asyncio.Event()

    class StreamingArtifact(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'x' * (1024 * 1024)
            entered.set()
            await asyncio.Event().wait()

    monkeypatch.setattr(local_runtime, 'platform_error', lambda: None)
    runtime = local_runtime.MacRuntime(tmp_path / 'models')
    async with httpx.AsyncClient(transport=httpx.MockTransport(
            lambda _request: httpx.Response(200, stream=StreamingArtifact()))) as client:
        operation = asyncio.create_task(runtime.setup(client=client))
        try:
            await entered.wait()
            assert list(runtime.root.glob('*.part'))
            await asyncio.wait_for(runtime.stop(), timeout=1)
            assert operation.cancelled()
            assert list(runtime.root.iterdir()) == []
        finally:
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)


@pytest.mark.asyncio
async def test_cancelled_extraction_retains_ownership_until_thread_finishes(tmp_path, monkeypatch):
    import local_runtime

    entered, release, finished = threading.Event(), threading.Event(), threading.Event()

    def extracting(_archive, _destination):
        entered.set()
        release.wait(timeout=5)
        finished.set()
        return tmp_path / 'llama-server'

    monkeypatch.setattr(local_runtime, 'platform_error', lambda: None)
    monkeypatch.setattr(local_runtime, 'verified_file', lambda *_args: True)
    monkeypatch.setattr(local_runtime, 'unpack_runtime', extracting)
    runtime = local_runtime.MacRuntime(tmp_path)
    operation = asyncio.create_task(runtime.verify_installation())
    try:
        assert await asyncio.to_thread(entered.wait, 2)
        operation.cancel()
        await asyncio.sleep(0.02)
        assert not operation.done()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert finished.is_set()
    finally:
        release.set()
        await asyncio.gather(operation, return_exceptions=True)
        await asyncio.to_thread(finished.wait, 2)


@pytest.mark.asyncio
async def test_cancelling_shutdown_still_terminates_every_owned_child(tmp_path):
    from local_runtime import MacRuntime

    entered, release = asyncio.Event(), asyncio.Event()

    class SlowChild:
        returncode = None
        terminated = False

        def terminate(self):
            self.terminated = True

        async def wait(self):
            entered.set()
            await release.wait()
            self.returncode = -15

    runtime = MacRuntime(tmp_path)
    children = [SlowChild(), SlowChild()]
    runtime.processes = dict(zip(('llm', 'embeddings'), children, strict=True))
    shutdown = asyncio.create_task(runtime.stop())
    try:
        await entered.wait()
        shutdown.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await shutdown
        assert all(child.terminated and child.returncode is not None for child in children)
    finally:
        release.set()
        await asyncio.gather(shutdown, return_exceptions=True)
