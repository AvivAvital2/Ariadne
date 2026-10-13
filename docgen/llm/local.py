"""Strict vLLM transport shared by local chat, tokenization and embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference import InferenceContext
    from model_config import EmbeddingSettings, ModelSettings

import asyncio
import time
from http import HTTPStatus

import httpx

from docgen.llm.base import CacheStats
from model_config import TOKEN_SAFETY_MARGIN, ChatSettings, tokenize_url


class LocalInferenceError(RuntimeError):
    """A local endpoint or request contract failed; never fall back to cloud."""

    retryable = False


class TransientInferenceError(LocalInferenceError):
    retryable = True


class EmbeddingContractError(LocalInferenceError):
    pass


class ContextOverflow(LocalInferenceError):  # noqa: N818 - public contract name.
    pass


TRANSIENT_STATUS = {408, 429, 500, 502, 503, 504}
CHAT_MESSAGE_COUNT = 2


def auth_headers(credential: str | None) -> dict[str, str]:
    headers = {'Content-Type': 'application/json'}
    if credential:
        headers['Authorization'] = f'Bearer {credential}'
    return headers


def response_json(response: httpx.Response) -> dict:
    if response.status_code in TRANSIENT_STATUS:
        raise TransientInferenceError(f'Local endpoint returned HTTP {response.status_code}')
    if not HTTPStatus.OK <= response.status_code < HTTPStatus.MULTIPLE_CHOICES:
        raise LocalInferenceError(
            f'Local endpoint returned HTTP {response.status_code}; verify model, credentials and request configuration'
        )
    try:
        data = response.json()
    except ValueError as exc:
        raise LocalInferenceError('Local endpoint returned malformed JSON') from exc
    if not isinstance(data, dict):
        raise LocalInferenceError('Local endpoint must return a JSON object')
    return data


def retry_delay(response: httpx.Response | None, attempt: int) -> float:
    if response is not None:
        try:
            return max(0.0, min(float(response.headers['Retry-After']), 30.0))
        except (KeyError, ValueError):
            pass
    return min(2**attempt, 8.0)


class LocalTransport:
    def __init__(self, settings: ChatSettings | EmbeddingSettings, *, attempts: int = 3) -> None:
        self.settings = settings
        self.attempts = attempts
        self._client: httpx.AsyncClient | None = None
        self._sync_client: httpx.Client | None = None
        self._ready = False

    def _client_options(self) -> dict:
        return {
            'headers': auth_headers(self.settings.credential()),
            'timeout': self.settings.timeout_seconds,
            'trust_env': False,
            'follow_redirects': False,
            'verify': True,
        }

    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(**self._client_options())
        return self._client

    def sync_client(self) -> httpx.Client:
        if self._sync_client is None:
            self._sync_client = httpx.Client(**self._client_options())
        return self._sync_client

    async def request(
        self,
        method: str,
        url: str,
        *,
        payload: dict | None = None,
        attempts: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        for attempt in range(attempts or self.attempts):
            response = None
            try:
                response = await self.client().request(
                    method, url, json=payload, timeout=timeout or self.settings.timeout_seconds
                )
                return response_json(response)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                error = TransientInferenceError('Local endpoint connection failed or timed out')
                error.__cause__ = exc
            except TransientInferenceError as exc:
                error = exc
            if attempt + 1 >= (attempts or self.attempts):
                raise error
            await asyncio.sleep(retry_delay(response, attempt))
        raise RuntimeError('Empty request attempt budget')

    def request_sync(
        self,
        method: str,
        url: str,
        *,
        payload: dict | None = None,
        attempts: int | None = None,
        timeout: float | None = None,
    ) -> dict:
        for attempt in range(attempts or self.attempts):
            response = None
            try:
                response = self.sync_client().request(
                    method, url, json=payload, timeout=timeout or self.settings.timeout_seconds
                )
                return response_json(response)
            except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                error = TransientInferenceError('Local endpoint connection failed or timed out')
                error.__cause__ = exc
            except TransientInferenceError as exc:
                error = exc
            if attempt + 1 >= (attempts or self.attempts):
                raise error
            time.sleep(retry_delay(response, attempt))
        raise RuntimeError('Empty request attempt budget')

    def _check_models(self, data: dict) -> None:
        models = data.get('data')
        if not isinstance(models, list) or self.settings.request_model not in [
            m.get('id') for m in models if isinstance(m, dict)
        ]:
            raise LocalInferenceError(
                f'Endpoint does not advertise configured model {self.settings.request_model!r}'
            )
        self._ready = True

    async def ready(self, *, attempts: int | None = None) -> None:
        if not self._ready:
            self._check_models(
                await self.request('GET', self.settings.base_url + '/models', attempts=attempts)
            )

    def ready_sync(self, *, attempts: int | None = None) -> None:
        if not self._ready:
            self._check_models(
                self.request_sync('GET', self.settings.base_url + '/models', attempts=attempts)
            )

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        if self._sync_client is not None:
            self._sync_client.close()
            self._sync_client = None
        self._ready = False


def build_chat_request(
    settings: ChatSettings, system_prompt: str, user_prompt: str, max_tokens: int
) -> dict:
    if (
        isinstance(max_tokens, bool)
        or not isinstance(max_tokens, int)
        or not 0 < max_tokens <= settings.max_output_tokens
    ):
        raise ContextOverflow(
            f'Requested output cap {max_tokens} exceeds configured maximum {settings.max_output_tokens}'
        )
    if (
        not isinstance(system_prompt, str)
        or not isinstance(user_prompt, str)
        or not user_prompt.strip()
    ):
        raise LocalInferenceError('Local chat requires nonempty text input')
    return {
        **settings.request_options,
        'model': settings.request_model,
        'messages': [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
        ],
        'max_tokens': max_tokens,
        'stream': False,
    }


def chat_token_request(payload: dict) -> dict:
    return {
        'model': payload['model'],
        'messages': payload['messages'],
        'chat_template_kwargs': payload['chat_template_kwargs'],
        'add_generation_prompt': True,
        'add_special_tokens': False,
    }


def token_count(data: dict) -> tuple[int, int]:
    count, limit = data.get('count'), data.get('max_model_len')
    if type(count) is not int or count <= 0 or type(limit) is not int or limit <= 0:
        raise LocalInferenceError(
            'Incompatible tokenization response: expected positive count and max_model_len'
        )
    tokens = data.get('tokens')
    if (
        not isinstance(tokens, list)
        or len(tokens) != count
        or any(type(t) is not int for t in tokens)
    ):
        raise LocalInferenceError('Tokenization response count does not match its token IDs')
    return count, limit


async def check_endpoints(settings: ModelSettings) -> dict:
    """Explicit readiness probe, usable before creating a library."""
    settings.validate(execution=True)
    settings.check_credentials()
    from model_config import query_input

    probes = (
        (
            'chat',
            settings.chat,
            chat_token_request(build_chat_request(settings.chat, '', 'Readiness probe', 1)),
        ),
        (
            'embeddings',
            settings.embeddings,
            {
                'model': settings.embeddings.request_model,
                'prompt': query_input('Readiness probe'),
                'add_special_tokens': True,
            },
        ),
    )
    result = {}
    for name, endpoint, payload in probes:
        transport = LocalTransport(endpoint, attempts=settings.execution.max_attempts)
        try:
            await transport.ready()
            count, limit = token_count(
                await transport.request('POST', tokenize_url(endpoint.base_url), payload=payload)
            )
            result[name] = {
                'advertised_model': endpoint.request_model,
                'probe_tokens': count,
                'server_context': limit,
            }
        finally:
            await transport.aclose()
    return result


def check_chat_context(data: dict, settings: ChatSettings, payload: dict) -> None:
    count, server_limit = token_count(data)
    if count + payload['max_tokens'] + TOKEN_SAFETY_MARGIN > min(
        settings.context_window, server_limit
    ):
        raise ContextOverflow(
            'Input plus requested output exceeds the local context window; split the source using the existing generation controls'
        )


def parse_chat_response(data: dict) -> tuple[str, dict[str, int | None]]:
    choices = data.get('choices')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise LocalInferenceError('Malformed local chat choices')
    choice = choices[0]
    if choice.get('finish_reason') != 'stop':
        raise LocalInferenceError(
            f'Incomplete local final output (finish_reason={choice.get("finish_reason")!r})'
        )
    message = choice.get('message')
    content = message.get('content') if isinstance(message, dict) else None
    if (
        not isinstance(content, str)
        or not content.strip()
        or '<think>' in content
        or '</think>' in content
    ):
        raise LocalInferenceError(
            'Local chat returned empty, malformed or reasoning-only final content'
        )
    raw_usage = data.get('usage') or {}
    usage = {
        key: raw_usage.get(key)
        if isinstance(raw_usage, dict) and type(raw_usage.get(key)) is int
        else None
        for key in ('prompt_tokens', 'completion_tokens')
    }
    return content, usage


class LocalOpenAICompatibleProvider:
    """Fixed Qwen deployment, with no hosted transport inheritance."""

    def __init__(self, context: InferenceContext) -> None:
        self.context = context
        self.settings = context.settings.chat
        self.model = self.settings.request_model
        self.api_key = self.settings.credential() or ''
        self.base_url = self.settings.base_url
        self.transport = LocalTransport(
            self.settings, attempts=context.settings.execution.max_attempts
        )
        self.cache_stats = CacheStats()
        self.last_usage: dict[str, int | None] = {}
        self.last_elapsed: float | None = None

    def _record(self, data: dict, start: float) -> str:
        content, usage = parse_chat_response(data)
        self.last_usage = usage
        self.last_elapsed = time.monotonic() - start
        if all(value is not None for value in usage.values()):
            from docgen.calibration import emit_usage

            emit_usage(
                model=self.model,
                input_tokens=usage['prompt_tokens'],
                output_tokens=usage['completion_tokens'],
            )
        return content

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        cache_system_prompt: bool = True,
        timeout: float | None = None,
        attempts: int | None = None,
    ) -> str:
        del cache_system_prompt  # Protocol compatibility; local prefix caching is server-owned.
        payload = build_chat_request(self.settings, system_prompt, user_prompt, max_tokens)
        return await self.call_payload(payload, attempts=attempts, timeout=timeout)

    async def call_payload(
        self, payload: dict, *, attempts: int | None = None, timeout: float | None = None
    ) -> str:
        """Execute the exact persisted request, validating its frozen options."""
        from dataclasses import replace

        from model_config import canonical_json

        self.context.validate(credentials=True)
        attempts = attempts or self.context.transport_attempts
        options = {
            key: value
            for key, value in payload.items()
            if key not in {'messages', 'model', 'max_tokens', 'stream'}
        }
        settings = replace(self.settings, request_options_json=canonical_json(options))
        replace(self.context.settings, chat=settings).validate(execution=True)
        messages = payload.get('messages', [])
        if (
            len(messages) != CHAT_MESSAGE_COUNT
            or messages[0].get('role') != 'system'
            or messages[1].get('role') != 'user'
        ):
            raise LocalInferenceError('Unsupported persisted message format')
        expected = build_chat_request(
            settings,
            messages[0].get('content'),
            messages[1].get('content'),
            payload.get('max_tokens'),
        )
        if payload != expected:
            raise LocalInferenceError('Persisted request does not match the local request contract')
        start = time.monotonic()
        async with self.context.chat_limit:
            await self.transport.ready(attempts=attempts)
            counted = await self.transport.request(
                'POST',
                tokenize_url(self.base_url),
                payload=chat_token_request(payload),
                attempts=attempts,
                timeout=timeout,
            )
            check_chat_context(counted, self.settings, payload)
            data = await self.transport.request(
                'POST',
                self.base_url + '/chat/completions',
                payload=payload,
                attempts=attempts,
                timeout=timeout,
            )
        return self._record(data, start)

    def call_sync(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        timeout: float | None = None,
    ) -> str:
        self.context.validate(credentials=True)
        payload = build_chat_request(self.settings, system_prompt, user_prompt, max_tokens)
        start = time.monotonic()
        with self.context.chat_limit:
            self.transport.ready_sync()
            counted = self.transport.request_sync(
                'POST',
                tokenize_url(self.base_url),
                payload=chat_token_request(payload),
                timeout=timeout,
            )
            check_chat_context(counted, self.settings, payload)
            data = self.transport.request_sync(
                'POST', self.base_url + '/chat/completions', payload=payload, timeout=timeout
            )
        return self._record(data, start)

    async def aclose(self) -> None:
        await self.transport.aclose()
