"""Managed downloads and native serving for the pinned Mac model candidate."""

from __future__ import annotations

import asyncio
import hashlib
import os
import platform
import socket
import sys
import tarfile
from collections.abc import Callable, Mapping
from contextlib import AsyncExitStack, suppress
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlsplit

import httpx

from docgen.llm.local import TransientInferenceError, check_endpoints
from local_models import (
    GEMMA_ARTIFACTS,
    GEMMA_PROFILE_ID,
    MAC_ACCURACY_ARTIFACTS,
    MAC_ACCURACY_PROFILE_ID,
    MAC_ARTIFACTS,
    MAC_PROFILE_ID,
    MAC_PROFILE_IDS,
    ModelArtifact,
    managed_model_profile,
)
from model_config import ModelSettings, canonical_json, resolve_models

RUNTIME_VERSION = 'b11669'
RUNTIME_SHA256 = 'dbed55493d72bc1ba2babc8770d669c2289b5bc4ee9fb2a4dfc158d45b7d13fd'
RUNTIME_SIZE = 12191582
RUNTIME_FILENAME = f'llama-{RUNTIME_VERSION}-bin-macos-arm64.tar.gz'
RUNTIME_URL = f'https://github.com/ggml-org/llama.cpp/releases/download/{RUNTIME_VERSION}/{RUNTIME_FILENAME}'
WORKER_PATH = Path(__file__).resolve().with_name('local_runtime_worker.py')


class RuntimeSetupError(RuntimeError):
    """A local runtime operation failed with an actionable setup error."""


def verified_file(path: Path, size: int, sha256: str) -> bool:
    if not path.is_file() or path.stat().st_size != size:
        return False
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() == sha256


async def download_file(
    client: httpx.AsyncClient, url: str, destination: Path, *, size: int, sha256: str,
    progress: Callable[[int, int], None] | None = None,
) -> None:
    """Publish only verified bytes; cancellation or errors preserve the prior file."""
    if await asyncio.to_thread(verified_file, destination, size, sha256):
        if progress:
            progress(size, size)
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = None
    try:
        with NamedTemporaryFile(dir=destination.parent, suffix='.part', delete=False) as stream:
            partial = Path(stream.name)
            count, digest = 0, hashlib.sha256()
            async with client.stream('GET', url) as response:
                response.raise_for_status()
                async for chunk in response.aiter_bytes(1024 * 1024):
                    count += len(chunk)
                    if count > size:
                        raise RuntimeSetupError('Download verification failed: unexpected file size')
                    stream.write(chunk)
                    digest.update(chunk)
                    if progress:
                        progress(count, size)
            if count != size or digest.hexdigest() != sha256:
                raise RuntimeSetupError('Download verification failed: size or SHA-256 mismatch')
        partial.replace(destination)
    finally:
        if partial is not None:
            partial.unlink(missing_ok=True)


def unpack_runtime(archive_path: Path, destination: Path) -> Path:
    """Extract a verified archive, retaining its adjacent Metal and support libraries."""
    directory = f'llama-{RUNTIME_VERSION}'
    with tarfile.open(archive_path, 'r:gz') as archive:
        members = archive.getmembers()
        for member in members:
            path = Path(member.name)
            if path.is_absolute() or '..' in path.parts or path.parts[0] != directory:
                raise RuntimeSetupError('Runtime archive contains an unexpected path')
        try:
            archive.extractall(destination, filter='data')
        except tarfile.FilterError as exc:
            raise RuntimeSetupError('Runtime archive contains an unsafe file') from exc
    executable = destination / directory / 'llama-server'
    if not executable.is_file() or not executable.stat().st_mode & 0o100:
        raise RuntimeSetupError('Runtime archive is missing its executable')
    return executable


def server_command(
    settings: ModelSettings, role: str, executable: Path, model_directory: Path
) -> list[str]:
    """Start only the artifacts whose identities are reserved in the library."""
    settings.validate(execution=True)
    profile_id = managed_profile_id(settings)
    if profile_id is None:
        raise RuntimeSetupError('Managed runtime requires the exact pinned Mac model profile')
    artifacts = profile_artifacts(profile_id)
    endpoints = [urlsplit(s.base_url) for s in (settings.chat, settings.embeddings)]
    if any(p.scheme != 'http' or p.hostname != '127.0.0.1' or p.path != '/v1'
           or not p.port for p in endpoints) or endpoints[0].port == endpoints[1].port:
        raise RuntimeSetupError('Managed servers require distinct 127.0.0.1 ports and /v1 endpoints')
    if role not in artifacts:
        raise RuntimeSetupError('Unknown local model role')
    endpoint = settings.chat if role == 'llm' else settings.embeddings
    context = settings.chat.context_window if role == 'llm' else settings.embeddings.max_input_tokens
    command = [str(executable), '--model', str(model_directory / artifacts[role].filename),
               '--alias', endpoint.request_model, '--host', '127.0.0.1',
               '--port', str(urlsplit(endpoint.base_url).port), '--ctx-size', str(context),
               '--parallel', '1', '--n-gpu-layers', 'all', '--fit', 'off',
               '--cache-ram', '0', '--no-webui']
    if role == 'llm':
        options = settings.chat.request_options
        command += ['--jinja', '--chat-template-kwargs', canonical_json(options['chat_template_kwargs']),
                    '--no-context-shift', '--batch-size', '512', '--ubatch-size', '512']
        if 'reasoning_format' in options:
            command += ['--reasoning-format', options['reasoning_format']]
    else:
        command += ['--embedding', '--pooling', 'last',
                    '--batch-size', str(context), '--ubatch-size', str(context)]
    return command


def managed_profile_id(settings: ModelSettings) -> str | None:
    """Identify exact pinned weights and mode, ignoring endpoint/timeout overrides."""
    return next((profile_id for profile_id in MAC_PROFILE_IDS
                 if settings.identity == resolve_models(managed_model_profile(profile_id)).identity), None)


def profile_artifacts(profile_id: str | None) -> Mapping[str, ModelArtifact]:
    if profile_id == MAC_PROFILE_ID:
        return MAC_ARTIFACTS
    if profile_id == MAC_ACCURACY_PROFILE_ID:
        return MAC_ACCURACY_ARTIFACTS
    if profile_id == GEMMA_PROFILE_ID:
        return GEMMA_ARTIFACTS
    raise RuntimeSetupError(f'Unknown managed local model profile: {profile_id!r}')


def platform_error() -> str | None:
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        return 'The managed Mac runtime requires native Apple Silicon macOS; cloud and configured models remain available.'
    return None


def require_platform() -> None:
    if error := platform_error():
        raise RuntimeSetupError(error)


async def finish_owned[T](task: asyncio.Task[T]) -> T:
    """Delay caller cancellation until owned filesystem/process cleanup is finished."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class MacRuntime:
    """Own only this application's model processes, with bounded startup and cleanup."""

    def __init__(self, root: Path, *, settings: ModelSettings | None = None) -> None:
        self.root = root
        self.settings = settings or resolve_models(managed_model_profile(MAC_ACCURACY_PROFILE_ID))
        self.processes: dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()
        self._operation: asyncio.Task | None = None
        self._state: dict = {'phase': 'stopped', 'error': None}

    def status(self) -> dict:
        if self._state['phase'] == 'ready' and any(
            process.returncode is not None for process in self.processes.values()
        ):
            self._state.update(phase='error', error='A local model server exited; inspect its runtime log and restart.')
        return {**self._state, 'platform_error': platform_error(),
                'qualification': 'unqualified', 'log_directory': str(self.root / 'logs'),
                'profile_id': managed_profile_id(self.settings), 'profile_hash': self.settings.profile_hash}

    def _select_profile(self, profile_id: str | None) -> None:
        """Called under the operation lock, before altering files or owned processes."""
        if profile_id is None:
            return
        if profile_id not in MAC_PROFILE_IDS:
            raise RuntimeSetupError(f'Unknown managed local model profile: {profile_id!r}')
        if profile_id == managed_profile_id(self.settings):
            return
        if self.processes:
            raise RuntimeSetupError('Stop local models before selecting a different model profile.')
        self.settings = resolve_models(managed_model_profile(profile_id))
        self._state = {'phase': 'stopped', 'error': None}

    async def setup(self, *, client: httpx.AsyncClient | None = None,
                    profile_id: str | None = None) -> None:
        async with self._lock:
            self._select_profile(profile_id)
            if self.processes:
                raise RuntimeSetupError('Stop local models before downloading or repairing the installation.')
            self._state.update(phase='downloading', error=None)
            self._operation = asyncio.current_task()
            try:
                require_platform()
                server_command(self.settings, 'llm', self.root / 'llama-server', self.root / 'models')
                downloads = [(RUNTIME_URL, self.root / RUNTIME_FILENAME, RUNTIME_SIZE, RUNTIME_SHA256)] + [
                    (artifact.url, self.root / 'models' / artifact.filename, artifact.size, artifact.sha256)
                    for artifact in profile_artifacts(managed_profile_id(self.settings)).values()
                ]
                async with AsyncExitStack() as stack:
                    if client is None:
                        client = await stack.enter_async_context(httpx.AsyncClient(
                            follow_redirects=True, trust_env=False, timeout=120))
                    for url, path, size, sha256 in downloads:
                        self._state.update(file=path.name, downloaded_bytes=0, total_bytes=size)
                        await download_file(client, url, path, size=size, sha256=sha256,
                                            progress=self._download_progress)
                self._state.update(phase='stopped', error=None)
            except BaseException as exc:
                self._state.update(phase='stopped' if isinstance(exc, asyncio.CancelledError) else 'error',
                                   error=None if isinstance(exc, asyncio.CancelledError) else str(exc))
                raise
            finally:
                self._operation = None

    def _download_progress(self, done: int, total: int) -> None:
        self._state.update(downloaded_bytes=done, total_bytes=total)

    async def verify_installation(self) -> Path:
        require_platform()
        archive = self.root / RUNTIME_FILENAME
        files = [(archive, RUNTIME_SIZE, RUNTIME_SHA256)] + [
            (self.root / 'models' / artifact.filename, artifact.size, artifact.sha256)
            for artifact in profile_artifacts(managed_profile_id(self.settings)).values()
        ]
        for path, size, sha256 in files:
            if not await asyncio.to_thread(verified_file, path, size, sha256):
                raise RuntimeSetupError(f'{path.name} is missing or damaged; download the local models again.')
        return await finish_owned(asyncio.create_task(
            asyncio.to_thread(unpack_runtime, archive, self.root / 'runtime')))

    def _check_ports(self) -> None:
        for endpoint in (self.settings.chat, self.settings.embeddings):
            port = urlsplit(endpoint.base_url).port
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
                try:
                    probe.bind(('127.0.0.1', port))
                except OSError as exc:
                    raise RuntimeSetupError(
                        f'Local model port {port} is already in use; stop its owner before starting Ariadne models.'
                    ) from exc

    def _check_processes(self) -> None:
        if any(process.returncode is not None for process in self.processes.values()):
            raise RuntimeSetupError('A model server exited during startup; inspect its runtime log.')

    async def start(self, *, timeout: float = 300, profile_id: str | None = None) -> None:
        async with self._lock:
            self._select_profile(profile_id)
            if self.status()['phase'] == 'ready':
                return
            await self._stop_processes()
            self._state.update(phase='starting', error=None)
            self._operation = asyncio.current_task()
            try:
                # Validate ownership and identities before touching installation or ports.
                server_command(self.settings, 'llm', self.root / 'llama-server', self.root / 'models')
                self._check_ports()
                executable = await self.verify_installation()
                logs = self.root / 'logs'
                logs.mkdir(parents=True, exist_ok=True)
                environment = {key: value for key, value in os.environ.items()
                               if not key.startswith('LLAMA_ARG_')}
                for role in profile_artifacts(managed_profile_id(self.settings)):
                    command = server_command(self.settings, role, executable, self.root / 'models')
                    with (logs / f'{role}.log').open('ab') as log:
                        self.processes[role] = await asyncio.create_subprocess_exec(
                            sys.executable, '-I', str(WORKER_PATH),
                            '--', *command,
                            env=environment, stdin=asyncio.subprocess.PIPE,
                            stdout=log, stderr=asyncio.subprocess.STDOUT,
                        )
                async with asyncio.timeout(timeout):
                    while True:
                        self._check_processes()
                        try:
                            await check_endpoints(self.settings)
                            self._check_processes()
                            break
                        except TransientInferenceError:
                            await asyncio.sleep(0.25)
                self._state.update(phase='ready', error=None)
            except BaseException as exc:
                await self._stop_processes()
                self._state.update(phase='stopped' if isinstance(exc, asyncio.CancelledError) else 'error',
                                   error=None if isinstance(exc, asyncio.CancelledError) else (
                                       'Local model startup timed out; inspect its runtime logs.'
                                       if isinstance(exc, TimeoutError) else str(exc)))
                raise
            finally:
                self._operation = None

    async def _stop_processes(self) -> None:
        for process in self.processes.values():
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    process.terminate()
        try:
            await finish_owned(asyncio.create_task(self._reap_processes()))
        finally:
            self.processes.clear()

    async def _reap_processes(self) -> None:
        for process in self.processes.values():
            if process.returncode is None:
                try:
                    await asyncio.wait_for(process.wait(), timeout=5)
                except TimeoutError:
                    with suppress(ProcessLookupError):
                        process.kill()
                    await process.wait()

    async def stop(self) -> None:
        self._state.update(phase='stopping', error=None)
        operation = self._operation
        if operation is not None and operation is not asyncio.current_task() and not operation.done():
            operation.cancel()
            await asyncio.gather(operation, return_exceptions=True)
        async with self._lock:
            try:
                await self._stop_processes()
            finally:
                self._state.update(phase='stopped', error=None)
