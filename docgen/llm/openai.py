"""OpenAI LLM provider.

Talks to OpenAI's ``/chat/completions`` endpoint. Extracted from the
historical ``DocGenerator._call_llm`` so the same logic is now reusable
through the ``LLMProvider`` Protocol. GPT-5.x uses ``max_completion_tokens``;
older models use ``max_tokens``.
"""
from __future__ import annotations

import asyncio
import logging
import time as _time

import httpx
from attrs import define, field

from docgen.llm.anthropic import QuotaExhaustedError, _is_quota_exhausted
from docgen.llm.base import CacheStats

_logger = logging.getLogger(__name__)


def token_limit_field(model: str) -> str:
    """OpenAI's token-limit param name for ``model``: gpt-5.x renamed
    ``max_tokens`` to ``max_completion_tokens``; older models keep
    ``max_tokens`` (and neither sends ``temperature`` — modern models reject
    it). Shared by the unary path, the batch strategy, and the trace-flow
    bridge so the rule lives in one place."""
    return 'max_completion_tokens' if model.startswith('gpt-5') else 'max_tokens'


@define
class OpenAIProvider:
    """LLM provider for OpenAI-compatible ``/chat/completions`` endpoints.

    Works with OpenAI directly, OpenRouter, and any other OpenAI-compatible
    proxy via the ``base_url`` parameter.
    """
    model: str
    api_key: str
    base_url: str = 'https://api.openai.com/v1'
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False)
    # Always-empty stats so callers can read provider.cache_stats uniformly.
    # OpenAI's automatic prompt caching for GPT-4o+ doesn't surface per-call
    # cache token counts in the API response, so we can't populate this here.
    cache_stats: CacheStats = field(factory=CacheStats, init=False)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    'Content-Type': 'application/json',
                    'Authorization': f'Bearer {self.api_key}',
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        cache_system_prompt: bool = True,  # noqa: ARG002 — protocol arg, no OpenAI equivalent
    ) -> str | None:
        client = self._get_client()

        payload: dict = {
            'model': self.model,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
        }
        payload[token_limit_field(self.model)] = max_tokens

        _logger.info(
            'OpenAI request: model=%s prompt_len=%d',
            self.model, len(user_prompt),
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = _time.monotonic()
            try:
                response = await client.post('/chat/completions', json=payload)
                elapsed = _time.monotonic() - t0
                response.raise_for_status()
                data = response.json()
                _logger.info(
                    'OpenAI response: status=%d elapsed=%.1fs',
                    response.status_code, elapsed,
                )
                return data['choices'][0]['message']['content']
            except httpx.HTTPStatusError as e:
                last_error = e
                _logger.warning(
                    'OpenAI request failed (attempt %d/%d): %s',
                    attempt + 1, self.max_retries, e,
                )
                if e.response.status_code == 429:
                    try:
                        err_body = e.response.json()
                    except Exception:
                        err_body = {}
                    if _is_quota_exhausted(err_body):
                        msg = (err_body.get('error') or {}).get('message') or 'quota exhausted'
                        _logger.error('OpenAI quota exhausted; aborting: %s', msg)
                        raise QuotaExhaustedError(msg) from e
                    await asyncio.sleep(self.retry_delay * (2 ** attempt))
                elif e.response.status_code >= 500:
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise  # client errors (4xx) are not retried
            except httpx.RequestError as e:
                last_error = e
                _logger.warning(
                    'OpenAI request error (attempt %d/%d): %s',
                    attempt + 1, self.max_retries, e,
                )
                await asyncio.sleep(self.retry_delay)

        _logger.error('All OpenAI request attempts failed: %s', last_error)
        return None
