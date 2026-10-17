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

import yaml

from config import Config
from model_config import CHAT_MODEL, EMBEDDING_MODEL, credential_env, identity_hash, resolve_models
from web.mcp_client import MCPBridge, connect_stdio, stdio_server_params


class ProjectStore:
    def __init__(self, config: Config):
        self.config = config
        self.root = config.config_dir.resolve() / 'web-projects'
        self._sessions = {}

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
        return profiles

    def choices(self) -> list[dict]:
        choices = []
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
        if not any(choice['provider'] == 'local' for choice in choices):
            choices.append({
                'id': 'qwen-local', 'model': CHAT_MODEL, 'provider': 'local',
                'embedding_model': EMBEDDING_MODEL, 'label': 'Qwen (local)',
                'available': False, 'reason': 'Configure local Qwen endpoints in onboarding_models',
                'preview_available': False,
            })
        return choices

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
        data = self._configuration(profile_id, name, source_path)
        settings = resolve_models(data)
        settings.check_credentials()
        project_id = identity_hash({'path': data['sources'][name]['path'], 'source': name,
                                    'models': settings.identity})[:24]
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

    async def start(self, profile_id: str, name: str, *, themes_enabled: bool | None = None) -> dict:
        project = self.prepare(profile_id, name, themes_enabled=themes_enabled)
        # A repeated onboarding can change scope; reload the saved source settings.
        previous = self._sessions.pop(project['id'], None)
        if previous:
            await previous.close()
        return project

    def describe(self, project_id: str) -> dict:
        cfg = Config(self.path(project_id))
        settings = cfg.model_settings()
        return {'id': project_id, 'source': cfg.default_source,
                'model': settings.chat.model, 'provider': settings.chat.provider,
                'embedding_model': settings.embeddings.model}

    def list_projects(self) -> list[dict]:
        return [self.describe(path.parent.name)
                for path in sorted(self.root.glob('*/ariadne.yaml'))]

    async def bridge(self, project_id: str) -> MCPBridge:
        path = self.path(project_id)
        session = self._sessions.get(project_id)
        if session is None:
            session = self._sessions[project_id] = ProjectSession(path)
        return await session.connect()

    async def close(self) -> None:
        await asyncio.gather(*(s.close() for s in self._sessions.values()))


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
            await self.task
