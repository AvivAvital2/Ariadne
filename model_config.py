"""Immutable model settings and versioned identities, independent of CLI code."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

CHAT_MODEL = 'Qwen/Qwen3.8-27B'
EMBEDDING_MODEL = 'Qwen/Qwen3-Embedding-8B'
EMBEDDING_DIM = 3072
NATIVE_DIM = 4096
RECIPE = 'qwen3-embedding-8b-3072-v1'
IDENTITY_VERSION = 1
QUERY_INSTRUCTION = 'Given a question about a codebase, retrieve relevant code documentation that answers the question.'
TOKEN_SAFETY_MARGIN = 64
DEFAULT_REQUEST_OPTIONS = {
    'temperature': 0.7,
    'top_p': 0.8,
    'presence_penalty': 1.5,
    'top_k': 20,
    'chat_template_kwargs': {'enable_thinking': False},
}


class ModelConfigurationError(ValueError):
    """An invalid or unsupported inference configuration."""


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False
    )


def identity_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def infer_provider_from_model(model: str) -> str | None:
    if model.startswith('gpt-'):
        return 'openai'
    if model.startswith('claude-'):
        return 'anthropic'
    return None


def resolve_provider(*, cli_provider: str | None, cfg_provider: str | None, model: str) -> str:
    inferred = infer_provider_from_model(model)
    explicit = cli_provider or cfg_provider
    if explicit and inferred and explicit != inferred:
        raise ModelConfigurationError(
            f'Provider mismatch: model {model!r} is hosted by {inferred!r}, '
            f'but provider was set to {explicit!r}.'
        )
    provider = explicit or inferred or 'openai'
    if provider not in {'openai', 'anthropic', 'local'}:
        raise ModelConfigurationError(f'Unknown provider: {provider!r}')
    return provider


def normalize_url(value: str) -> str:
    parts = urlsplit(value)
    if (
        parts.scheme not in {'http', 'https'}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ModelConfigurationError(
            'base_url must be an explicit HTTP(S) URL without credentials, query or fragment'
        )
    return urlunsplit((parts.scheme, parts.netloc, parts.path.rstrip('/'), '', ''))


def tokenize_url(base_url: str) -> str:
    """Keep any deployment prefix when leaving the OpenAI /v1 route."""
    base = normalize_url(base_url)
    if not base.endswith('/v1'):
        raise ModelConfigurationError('Local base_url must end in /v1')
    return base[:-3] + '/tokenize'


def _positive(name: str, value: Any, *, integer: bool = True) -> None:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value <= 0
    ):
        raise ModelConfigurationError(f'{name} must be positive')
    if integer and not isinstance(value, int):
        raise ModelConfigurationError(f'{name} must be an integer')


def _pin(name: str, value: str | None) -> None:
    if not isinstance(value, str) or not re.fullmatch(r'[0-9a-fA-F]{40}|[0-9a-fA-F]{64}', value):
        raise ModelConfigurationError(
            f'{name} requires an immutable artifact revision (40 or 64 hex digits), not a branch or placeholder'
        )


@dataclass(frozen=True)
class ChatSettings:
    provider: str = 'openai'
    model: str = 'gpt-5.5'
    model_revision: str | None = None
    weight_format: str | None = None
    tokenizer_revision: str | None = None
    served_model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 180.0
    max_concurrency: int = 3
    context_window: int = 32768
    max_output_tokens: int = 8192
    request_options_json: str = '{}'

    @property
    def request_model(self) -> str:
        return self.served_model or self.model

    @property
    def request_options(self) -> dict:
        return json.loads(self.request_options_json)

    @property
    def identity(self) -> dict:
        if self.provider != 'local':
            return {'version': IDENTITY_VERSION, 'provider': self.provider, 'model': self.model}
        return {
            'version': IDENTITY_VERSION,
            'provider': self.provider,
            'artifact': {
                'repository': self.model,
                'revision': self.model_revision,
                'weight_format': self.weight_format,
            },
            'tokenizer_revision': self.tokenizer_revision or self.model_revision,
        }

    def credential(self) -> str | None:
        return resolve_credential(self.provider, self.api_key_env)


@dataclass(frozen=True)
class EmbeddingSettings:
    provider: str = 'openai'
    model: str = 'text-embedding-3-large'
    model_revision: str | None = None
    weight_format: str | None = None
    tokenizer_revision: str | None = None
    served_model: str | None = None
    base_url: str | None = None
    api_key_env: str | None = None
    dimensions: int = EMBEDDING_DIM
    recipe: str = 'openai-3072-v1'
    max_input_tokens: int = 8192
    batch_size: int = 16
    batch_max_tokens: int = 16384
    max_concurrency: int = 1
    timeout_seconds: float = 120.0

    @property
    def request_model(self) -> str:
        return self.served_model or self.model

    @property
    def identity(self) -> dict:
        common = {
            'version': IDENTITY_VERSION,
            'provider': self.provider,
            'model': self.model,
            'dimensions': self.dimensions,
        }
        if self.provider != 'local':
            return {**common, 'recipe': self.recipe}
        return {
            **common,
            'artifact': {
                'repository': self.model,
                'revision': self.model_revision,
                'weight_format': self.weight_format,
            },
            'recipe': {
                'name': self.recipe,
                'tokenizer_revision': self.tokenizer_revision or self.model_revision,
                'pooling': 'last-token',
                'native_dimensions': NATIVE_DIM,
                'stored_dimensions': EMBEDDING_DIM,
                'reduction': 'prefix',
                'normalization': 'l2-float64-then-float32',
                'query_instruction': QUERY_INSTRUCTION,
                'document_input': 'title-newline2-content2000-v1',
                'chunk_input': 'ariadne-chunks-v1',
                'section_input': 'heading-or-introduction-v1',
                'add_special_tokens': True,
            },
        }

    @property
    def embedding_space_id(self) -> str:
        return identity_hash(self.identity)

    def credential(self) -> str | None:
        return resolve_credential(self.provider, self.api_key_env)


def resolve_credential(provider: str, name: str | None) -> str | None:
    if name is None:
        if provider == 'local':
            return None
        name = 'ANTHROPIC_API_KEY' if provider == 'anthropic' else 'OPENAI_API_KEY'
    value = os.environ.get(name)
    if not value:
        raise ModelConfigurationError(
            f'Configured credential environment variable {name!r} is missing or empty'
        )
    return value


@dataclass(frozen=True)
class ExecutionSettings:
    batch_mode: str = 'auto'
    auto_batch_threshold: int = 200
    max_attempts: int = 3


@dataclass(frozen=True)
class ModelSettings:
    chat: ChatSettings = field(default_factory=ChatSettings)
    embeddings: EmbeddingSettings = field(default_factory=EmbeddingSettings)
    execution: ExecutionSettings = field(default_factory=ExecutionSettings)

    @property
    def local(self) -> bool:
        return self.chat.provider == 'local'

    @property
    def identity(self) -> dict:
        return {
            'version': IDENTITY_VERSION,
            'generation': self.chat.identity,
            'embedding': self.embeddings.identity,
        }

    @property
    def profile_hash(self) -> str:
        return identity_hash(self.identity)

    def validate(self, *, execution: bool = False) -> ModelSettings:
        c, e = self.chat, self.embeddings
        if c.provider not in {'openai', 'anthropic', 'local'} or e.provider not in {
            'openai',
            'local',
        }:
            raise ModelConfigurationError('Unsupported inference provider')
        if (c.provider == 'local') != (e.provider == 'local'):
            raise ModelConfigurationError(
                'Local operation requires both llm.provider and embeddings.provider to be local'
            )
        for prefix, settings in [('llm', c), ('embeddings', e)]:
            _positive(f'{prefix}.timeout_seconds', settings.timeout_seconds, integer=False)
            _positive(f'{prefix}.max_concurrency', settings.max_concurrency)
            if settings.api_key_env is not None and (
                not isinstance(settings.api_key_env, str)
                or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', settings.api_key_env)
            ):
                raise ModelConfigurationError(
                    f'{prefix}.api_key_env must name an environment variable or be null'
                )
        for name, value in [
            ('llm.context_window', c.context_window),
            ('llm.max_output_tokens', c.max_output_tokens),
            ('embeddings.max_input_tokens', e.max_input_tokens),
            ('embeddings.batch_size', e.batch_size),
            ('embeddings.batch_max_tokens', e.batch_max_tokens),
            ('execution.max_attempts', self.execution.max_attempts),
            ('execution.auto_batch_threshold', self.execution.auto_batch_threshold),
        ]:
            _positive(name, value)
        if self.execution.batch_mode not in {'auto', 'always', 'never'}:
            raise ModelConfigurationError('execution.batch_mode must be auto, always or never')
        if self.local:
            if c.model != CHAT_MODEL or e.model != EMBEDDING_MODEL:
                raise ModelConfigurationError(
                    f'The supported local pair is {CHAT_MODEL} and {EMBEDDING_MODEL}'
                )
            if e.dimensions != EMBEDDING_DIM or e.recipe != RECIPE:
                raise ModelConfigurationError(
                    f'Local embeddings require {EMBEDDING_DIM} dimensions and recipe {RECIPE}'
                )
            for prefix, settings in [('llm', c), ('embeddings', e)]:
                if settings.max_concurrency != 1:
                    raise ModelConfigurationError(
                        f'{prefix}.max_concurrency must be one for local inference'
                    )
                if not settings.base_url:
                    raise ModelConfigurationError(
                        f'{prefix}.base_url is required for local inference'
                    )
                tokenize_url(settings.base_url)
                # The first supported weight representation. Quantized deployments
                # require separate qualification and a new explicit recipe release.
                if settings.weight_format != 'bf16':
                    raise ModelConfigurationError(f'{prefix}.weight_format must be bf16')
                if execution:
                    _pin(f'{prefix}.model_revision', settings.model_revision)
                    if settings.tokenizer_revision is not None:
                        _pin(f'{prefix}.tokenizer_revision', settings.tokenizer_revision)
                elif not settings.model_revision:
                    raise ModelConfigurationError(f'{prefix}.model_revision is required')
            options = c.request_options
            unknown = set(options) - set(DEFAULT_REQUEST_OPTIONS)
            if unknown or options.get('chat_template_kwargs') != {'enable_thinking': False}:
                raise ModelConfigurationError(
                    'Unsupported local request_options; thinking must be explicitly disabled'
                )
            for key, low, high in [
                ('temperature', 0, 2),
                ('top_p', 0, 1),
                ('presence_penalty', -2, 2),
            ]:
                value = options.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not low <= value <= high
                ):
                    raise ModelConfigurationError(f'Invalid local request option {key}')
            _positive('top_k', options.get('top_k'))
            if c.max_output_tokens + TOKEN_SAFETY_MARGIN >= c.context_window:
                raise ModelConfigurationError(
                    'llm.context_window must leave room for input and the output budget'
                )
        return self

    def check_credentials(self) -> None:
        self.chat.credential()
        self.embeddings.credential()

    def display(self) -> dict:
        """Non-secret operational settings and immutable model manifest."""
        return {
            'profile_hash': self.profile_hash,
            'identity': self.identity,
            'llm_endpoint': self.chat.base_url,
            'embedding_endpoint': self.embeddings.base_url,
            'llm_credential': self.chat.api_key_env,
            'embedding_credential': self.embeddings.api_key_env,
            'backend': 'local spool / live' if self.local else 'cloud batch / live',
            'context_window': self.chat.context_window,
            'max_output_tokens': self.chat.max_output_tokens,
            'concurrency': self.chat.max_concurrency,
        }


def _section(data: dict, name: str) -> dict:
    result = data.get(name, {}) or {}
    if not isinstance(result, dict):
        raise ModelConfigurationError(f'{name} must be a mapping')
    return dict(result)


def resolve_models(
    data: dict,
    *,
    model: str | None = None,
    provider: str | None = None,
    concurrency: int | None = None,
    execution: bool = False,
) -> ModelSettings:
    """Canonical fields agree with explicit aliases; legacy top-level wins."""
    defaults = _section(data, 'defaults')
    chat = _section(data, 'llm')
    embed = _section(data, 'embeddings')
    run = _section(data, 'execution')
    for key in ('model', 'provider'):
        if key in chat:
            for location in (defaults, data):
                if key in location and location[key] != chat[key]:
                    raise ModelConfigurationError(f'llm.{key} conflicts with explicit legacy {key}')
        elif key in data or key in defaults:
            chat[key] = data.get(key, defaults.get(key))
    chat.setdefault('model', 'gpt-5.5')
    chat['provider'] = resolve_provider(
        cli_provider=provider, cfg_provider=chat.get('provider'), model=model or chat['model']
    )
    if model is not None:
        chat['model'] = model
    if concurrency is not None:
        chat['max_concurrency'] = concurrency
    local = chat['provider'] == 'local'
    if local:
        chat.setdefault('max_concurrency', 1)
    options = chat.pop('request_options', DEFAULT_REQUEST_OPTIONS if local else {})
    if not isinstance(options, dict):
        raise ModelConfigurationError('llm.request_options must be a mapping')
    chat['request_options_json'] = canonical_json(options)
    if embed.get('provider') == 'local':
        embed.setdefault('recipe', RECIPE)
    for settings in (chat, embed):
        if settings.get('base_url'):
            settings['base_url'] = normalize_url(settings['base_url'])
    try:
        result = ModelSettings(
            ChatSettings(**chat), EmbeddingSettings(**embed), ExecutionSettings(**run)
        )
    except TypeError as exc:
        raise ModelConfigurationError(f'Unknown model configuration field: {exc}') from exc
    return result.validate(execution=execution)


def resolve_paths(
    config: Any, *, db: Path | None = None, staleness_db: Path | None = None, local: bool = False
) -> tuple[Path, Path]:
    configured = Path(config.db_path).expanduser().resolve()
    main = Path(db).expanduser().resolve() if db is not None else configured
    sidecar = (
        Path(staleness_db).expanduser().resolve()
        if staleness_db is not None
        else Path(config.staleness_db_path).expanduser().resolve()
    )
    if local and main != configured and staleness_db is None:
        sidecar = main.with_name(main.stem + '_staleness.db')
    if main == sidecar:
        raise ModelConfigurationError('Library and staleness databases must be separate paths')
    return main, sidecar


def document_input(title: str, content: str) -> str:
    return f'{title}\n\n{content[:2000]}'


def query_input(query: str) -> str:
    if not query or not query.strip():
        raise ModelConfigurationError('Search query must not be empty')
    return f'Instruct: {QUERY_INSTRUCTION}\nQuery:{query}'
