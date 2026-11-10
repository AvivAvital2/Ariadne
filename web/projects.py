"""Persist the model pair chosen at onboarding with its project library."""
from __future__ import annotations

import asyncio
import copy
import os
import re
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.parse import urlsplit

import yaml

from config import Config
from local_models import (
    MAC_ACCURACY_PROFILE_ID,
    MAC_PROFILE_IDS,
    managed_model_profile,
)
from local_runtime import MacRuntime
from model_config import CHAT_MODEL, EMBEDDING_MODEL, credential_env, identity_hash, resolve_models
from web.mcp_client import MCPBridge, connect_stdio, stdio_server_params


class ProjectStore:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.config_dir.resolve() / 'web-projects'
        self.external_profiles_path = self.root / 'model-profiles.json'
        self._sessions = {}
        self._interruptions = {}
        self.runtime = MacRuntime(config.config_dir.resolve() / 'local-models')

    def external_profiles(self) -> dict:
        from web.external_models import load_profiles

        return load_profiles(self.external_profiles_path)

    def save_external_profile(self, profile: dict) -> str:
        from web.external_models import save_profile

        return save_profile(self.external_profiles_path, profile)

    def profiles(self) -> dict:
        """Only configuration and credential presence are read; no inference calls."""
        from docgen.pricing import LLM_PRICING

        profiles = {
            model: {'llm': {'provider': 'anthropic' if model.startswith('claude') else 'openai',
                            'model': model}, 'embeddings': {'provider': 'openai'}}
            for model in LLM_PRICING if model.startswith(('gpt-', 'claude-'))
        }
        profiles['qwen-hosted'] = {
            'llm': {'provider': 'deepinfra', 'model': CHAT_MODEL},
            'embeddings': {'provider': 'deepinfra', 'model': EMBEDDING_MODEL},
            'execution': {'batch_mode': 'never'},
        }
        # The configured pair also exposes local Qwen, including its deployment pins.
        profiles['configured'] = {
            key: copy.deepcopy(value) for key, value in self.config._explicit.items()
            if key in ('llm', 'embeddings', 'execution', 'model', 'provider', 'defaults')
        }
        profiles.update(self.config._explicit.get('onboarding_models', {}) or {})
        profiles.update(self.external_profiles())
        profiles.update({profile_id: managed_model_profile(profile_id) for profile_id in MAC_PROFILE_IDS})
        return profiles

    def choices(self) -> list[dict]:
        choices = []
        external = self.external_profiles()
        for key, profile in self.profiles().items():
            settings = resolve_models(profile)
            credentials = {
                s.api_key_env or credential_env(s.provider)
                for s in (settings.chat, settings.embeddings)
                if (s.provider != 'local' or s.api_key_env)
            }
            missing = sorted(name for name in credentials if not os.environ.get(name))
            choices.append({
                'id': key, 'model': settings.chat.model,
                'provider': settings.chat.provider, 'embedding_model': settings.embeddings.model,
                'label': f'{settings.chat.model} + {settings.embeddings.model} ({settings.chat.provider})',
                'available': not missing,
                'preview_available': True,
                'reason': 'Missing ' + ', '.join(missing) if missing else None,
            })
            if key in external:
                host = urlsplit(settings.chat.base_url).netloc
                choices[-1].update(
                    user_managed=True, qualification='unqualified',
                    label=f'Experimental Qwen 27B Q4 + 8B embeddings — {host} (user managed)',
                    quality_note='Connection tested; generated-answer accuracy remains unqualified.',
                )
            if key in MAC_PROFILE_IDS:
                reason = self._runtime_unavailable(key)
                choices[-1].update(
                    **self._managed_details(key, settings.profile_hash),
                    available=reason is None, reason=reason,
                )
        if not any(choice['provider'] == 'local' and not choice.get('managed_local')
                   for choice in choices):
            choices.append({
                'id': 'qwen-local', 'model': CHAT_MODEL, 'provider': 'local',
                'embedding_model': EMBEDDING_MODEL, 'label': 'Qwen (local)',
                'available': False, 'reason': 'Configure local Qwen endpoints in onboarding_models',
                'preview_available': False,
            })
        return choices

    @staticmethod
    def _managed_details(profile_id: str, profile_hash: str) -> dict:
        details = {'managed_local': True, 'managed_local_profile': profile_id,
                   'profile_hash': profile_hash, 'qualification': 'unqualified'}
        if profile_id not in MAC_PROFILE_IDS:
            return details
        accuracy = profile_id == MAC_ACCURACY_PROFILE_ID
        return {**details,
                'label': ('Experimental Qwen local on Mac (27B thinking + 8B embeddings, Q4; 32GB qualification target)'
                          if accuracy else 'Experimental legacy Qwen local on Mac (9B generation + 8B embeddings, Q4)'),
                'quality_note': ('Experimental: 27B thinking has not passed Ariadne quality qualification.'
                                 if accuracy else 'Experimental: initial generated documents contained factual errors; this candidate is not production qualified.'),
                'memory_note': ('32GB qualification target; 24GB fit is unqualified. Memory use has not been measured on M5.'
                                if accuracy else 'Legacy 9B profile retained for existing projects; M5 memory use remains unqualified.')}

    def _runtime_unavailable(self, profile_id: str) -> str | None:
        if not isinstance(profile_id, str) or profile_id not in MAC_PROFILE_IDS:
            return 'Unknown managed local profile; the saved project has not been changed.'
        status = self.runtime.status()
        if status['platform_error']:
            return status['platform_error']
        if status['phase'] != 'ready':
            return 'Start local models before building this project.'
        expected = resolve_models(managed_model_profile(profile_id)).profile_hash
        if status.get('profile_id') != profile_id or status.get('profile_hash') != expected:
            return 'Stop current local models, then start the model profile saved for this project.'
        return None

    def path(self, project_id: str) -> Path:
        if not isinstance(project_id, str) or not re.fullmatch(r'[a-f0-9]{24}', project_id):
            raise ValueError('Unknown project')
        path = self.root / project_id / 'ariadne.yaml'
        if not path.is_file():
            raise ValueError('Unknown project')
        return path

    def _configuration(self, profile_id: str, name: str, source_path: str | None = None) -> dict:
        # Connect, discovery and scope edits update the root source configuration.
        # Re-read it so both previews and the eventual build use those edits.
        cfg = Config(self.config.config_path) if self.config.config_path else self.config
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Source name is required')
        source = Path(source_path).expanduser().resolve() if source_path else cfg.get_source_path(name)
        if source is None or not source.is_dir():
            raise ValueError(f'Path does not exist or is not a directory: {source}')
        profile = self.profiles().get(profile_id)
        if profile is None:
            raise ValueError('Unknown onboarding model choice')
        data = copy.deepcopy(cfg._explicit)
        for key in ('model', 'provider', 'llm', 'embeddings', 'execution', 'onboarding_models'):
            data.pop(key, None)
        data['defaults'] = {k: v for k, v in data.get('defaults', {}).items()
                            if k not in ('model', 'provider')}
        data.update(copy.deepcopy(profile))
        data.pop('managed_local_profile', None)
        data.pop('external_model_profile', None)
        if profile_id in MAC_PROFILE_IDS:
            data['managed_local_profile'] = profile_id
        elif profile_id in self.external_profiles():
            data['external_model_profile'] = profile_id
        # Keep source settings (including discovery and exclusions) and resolve
        # paths before relocating the configuration beside its own databases.
        sources = {}
        for key in cfg.sources:
            entry = copy.deepcopy(cfg.sources[key])
            if isinstance(entry, str):
                entry = {'path': entry}
            resolved = cfg.get_source_path(key)
            if resolved is not None:
                entry['path'] = str(resolved)
            sources[key] = entry
        sources.setdefault(name, {})['path'] = str(source.resolve())
        data.update(db_path='library.db', staleness_db_path='staleness.db',
                    docs_base='./docs', default_source=name, sources=sources)
        return data

    @asynccontextmanager
    async def preview(self, profile_id: str, name: str) -> AsyncIterator[MCPBridge]:
        """Use the real MCP estimator without saving a project or requiring keys."""
        data = self._configuration(profile_id, name)
        resolve_models(data)  # validate the pair, but never contact model servers
        with TemporaryDirectory(prefix='ariadne-preview-') as directory:
            path = Path(directory) / 'ariadne.yaml'
            path.write_text(yaml.safe_dump(data, sort_keys=False))
            session = ProjectSession(path)
            try:
                yield await session.connect()
            finally:
                await session.close()

    def prepare(self, profile_id: str, name: str, source_path: str | None = None,
                *, themes_enabled: bool | None = None) -> dict:
        """Save the selected pair only when the user starts a build."""
        if profile_id in MAC_PROFILE_IDS and (reason := self._runtime_unavailable(profile_id)):
            raise ValueError(reason)
        data = self._configuration(profile_id, name, source_path)
        settings = resolve_models(data)
        settings.check_credentials()
        project_id = self._project_identity(profile_id, name, data)
        directory = self.root / project_id
        path = directory / 'ariadne.yaml'
        if path.exists():
            saved = Config(path)._explicit
            saved['sources'] = data['sources']
            data = saved
        if themes_enabled is not None:
            data['themes_enabled'] = themes_enabled
        directory.mkdir(parents=True, exist_ok=True)
        temporary = directory / 'ariadne.yaml.tmp'
        temporary.write_text(yaml.safe_dump(data, sort_keys=False))
        temporary.replace(path)
        return self.describe(project_id)

    def project_id_for(self, profile_id: str, name: str) -> str:
        """Resolve admission identity without saving config or closing a session."""
        return self._project_identity(profile_id, name, self._configuration(profile_id, name))

    @staticmethod
    def _project_identity(profile_id: str, name: str, data: dict) -> str:
        project_identity = {'path': data['sources'][name]['path'], 'source': name,
                            'models': resolve_models(data).identity}
        if profile_id in MAC_PROFILE_IDS:
            # Externally managed endpoints must retain their own project config.
            project_identity['managed_local_profile'] = profile_id
        elif data.get('external_model_profile') == profile_id:
            project_identity['external_model_profile'] = profile_id
        return identity_hash(project_identity)[:24]

    async def start(self, profile_id: str, name: str, *, themes_enabled: bool | None = None) -> dict:
        project = self.prepare(profile_id, name, themes_enabled=themes_enabled)
        # A repeated onboarding can change scope; reload the saved source settings.
        previous = self._sessions.pop(project['id'], None)
        if previous:
            await previous.close()
        return project

    def check_runtime_ready(self, project_id: str) -> None:
        cfg = Config(self.path(project_id))
        if 'managed_local_profile' not in cfg._explicit:
            return
        profile_id = cfg._explicit['managed_local_profile']
        if not isinstance(profile_id, str) or profile_id not in MAC_PROFILE_IDS:
            raise ValueError('Unknown managed local profile; the saved project has not been changed.')
        if cfg.model_settings().identity != resolve_models(managed_model_profile(profile_id)).identity:
            raise ValueError('The saved model pair differs from its pinned profile; the project has not been changed.')
        if reason := self._runtime_unavailable(profile_id):
            raise ValueError(reason)

    def describe(self, project_id: str) -> dict:
        cfg = Config(self.path(project_id))
        settings = cfg.model_settings()
        return {'id': project_id, 'source': cfg.default_source,
                'model': settings.chat.model, 'provider': settings.chat.provider,
                'embedding_model': settings.embeddings.model,
                **({'user_managed': True} if 'external_model_profile' in cfg._explicit else {}),
                **(self._managed_details(cfg._explicit['managed_local_profile'], settings.profile_hash)
                   if 'managed_local_profile' in cfg._explicit else {})}

    def list_projects(self) -> list[dict]:
        return [self.describe(path.parent.name)
                for path in sorted(self.root.glob('*/ariadne.yaml'))]

    async def bridge(self, project_id: str) -> MCPBridge:
        path = self.path(project_id)
        session = self._sessions.get(project_id)
        if session is None:
            session = self._sessions[project_id] = ProjectSession(path)
        return await session.connect()

    async def interrupt(self, project_id: str) -> None:
        """Reap this project's owned MCP process without changing persisted work."""
        self.path(project_id)
        session = self._sessions.get(project_id)
        if session is not None:
            cleanup = self._interruptions.get(project_id)
            if cleanup is None:
                cleanup = self._interruptions[project_id] = asyncio.create_task(
                    self._interrupt_session(project_id, session))
            try:
                await asyncio.shield(cleanup)
            except asyncio.CancelledError:
                # A disconnected browser must not strand the owned MCP worker.
                await asyncio.shield(cleanup)
                raise

    async def _interrupt_session(self, project_id: str, session: ProjectSession) -> None:
        try:
            await session.close()
            if self._sessions.get(project_id) is session:
                self._sessions.pop(project_id)
        finally:
            self._interruptions.pop(project_id, None)

    async def interrupt_managed(self) -> None:
        """Stop owned sessions using the managed runtime, preserving other projects."""
        await asyncio.gather(*(self.interrupt(project_id)
                               for project_id in tuple(self._sessions)
                               if self.describe(project_id).get('managed_local')))

    async def close(self) -> None:
        try:
            await asyncio.gather(*(s.close() for s in self._sessions.values()))
        finally:
            await self.runtime.stop()


class ProjectSession:
    """Each MCP connection enters and exits its anyio scopes in the same task."""
    def __init__(self, path: Path):
        self.path = path
        self.ready = asyncio.Event()
        self.stop = asyncio.Event()
        self.task = None
        self.error = None
        self.bridge = None

    async def connect(self) -> MCPBridge:
        if self.task is None:
            self.task = asyncio.create_task(self._run())
        await self.ready.wait()
        if self.error:
            raise self.error
        return self.bridge

    async def _run(self) -> None:
        try:
            async with AsyncExitStack() as stack:
                self.bridge = await connect_stdio(stack, stdio_server_params(str(self.path)))
                self.ready.set()
                await self.stop.wait()
        except Exception as exc:  # noqa: BLE001 — surface child startup/transport errors to callers
            self.error = exc
        finally:
            self.ready.set()

    async def close(self) -> None:
        self.stop.set()
        if self.task:
            if not self.ready.is_set():
                # A child may never answer initialize; stop.wait() is not reached
                # until then. Cancel startup so its own task unwinds the stack.
                self.error = RuntimeError('Project session interrupted during startup')
                self.task.cancel()
                await asyncio.gather(self.task, return_exceptions=True)
                # A task canceled before its first step never runs _run's finally.
                self.ready.set()
            else:
                await self.task
