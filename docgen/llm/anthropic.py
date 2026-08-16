"""Anthropic LLM provider.

Native ``/v1/messages`` API client. Differs from the OpenAI shape:
the system prompt is a top-level field (not part of ``messages``),
``max_tokens`` is mandatory, auth is via ``x-api-key`` (not Bearer),
and ``anthropic-version`` is a required header.

Response shape: ``content[0].text`` instead of OpenAI's
``choices[0].message.content``.

Also exposes the Message Batches API (``submit_batch``, ``poll_batch``,
``fetch_batch_results``) which trades up to 24h latency for a 50% token
discount. See ``docgen.orchestrator`` for the call-count threshold that
chooses between sync and batch.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time as _time

import httpx
from attrs import define, field

from docgen.llm.base import CacheStats
# Re-exported (not used in this module): their canonical home is
# docgen.llm.batch, but existing `from docgen.llm.anthropic import BatchRequest`
# call sites + tests still import them here. Don't delete as "unused".
from docgen.llm.batch import BatchRequest, BatchStatus, BatchSubmission

_logger = logging.getLogger(__name__)

# Pinned to a known-good API version. Anthropic releases new versions
# infrequently; bump this only after verifying response-shape compatibility.
ANTHROPIC_API_VERSION = '2023-06-01'

# Beta header that opts into ``cache_control`` markers on content blocks. With
# the API version above, prompt caching is gated behind this beta — sending
# the marker without the header yields a 400 from /messages. Drop the header
# once the next API version graduates caching to GA.
ANTHROPIC_PROMPT_CACHING_BETA = 'prompt-caching-2024-07-31'

# Beta header that opts into the Message Batches API. Same lifecycle as
# the prompt-caching beta — drop once GA.
ANTHROPIC_MESSAGE_BATCHES_BETA = 'message-batches-2024-09-24'

# Both betas ride the same header value, comma-separated per Anthropic's spec.
_ANTHROPIC_BETAS = f'{ANTHROPIC_PROMPT_CACHING_BETA},{ANTHROPIC_MESSAGE_BATCHES_BETA}'

# Substrings that indicate a 429 is from monthly/credit quota exhaustion
# (fatal for the run) rather than a per-minute rate limit (transient, retry).
# Matched case-insensitively on the error.message field. Conservative list:
# false negatives just mean we retry uselessly until max_retries; false
# positives would short-circuit a transient error. Lean toward false negatives.
# Substrings that indicate a 429 is from monthly/credit quota exhaustion
# (fatal for the run) rather than a per-minute rate limit (transient, retry).
_QUOTA_EXHAUSTED_KEYWORDS: tuple[str, ...] = (
    'monthly',
    'credit',
    'billing',
    'plan',          # "upgrade your plan", "your plan limit"
    'organization limit',
    'spend limit',
)

# Substrings that mark a 400 ``invalid_request_error`` as a maxed *workspace*
# API usage limit — an admin-set cap that resets on a date, distinct from a
# credit/quota 429. Kept separate so the two caps get distinct messages.
_WORKSPACE_LIMIT_KEYWORDS: tuple[str, ...] = (
    'usage limit',
    'workspace',
)


class QuotaExhaustedError(Exception):
    """Raised on a hard, non-retryable Anthropic account cap.

    Base for both cap shapes so a caller can ``except QuotaExhaustedError`` to
    stop the run cleanly. Distinct from per-minute rate limits, which the
    provider retries with backoff.
    """


class WorkspaceUsageLimitError(QuotaExhaustedError):
    """A maxed *workspace* API usage limit (Anthropic returns this as a 400
    ``invalid_request_error``, not a 429). Subclasses QuotaExhaustedError so
    existing abort handlers catch it, while keeping a distinct type + message
    — which includes Anthropic's "regain access on <date>" note. The remedy
    differs from a credit quota: wait for the reset or raise the workspace cap.
    """


def _is_quota_exhausted(body: dict) -> bool:
    """True iff a 429 error body indicates a credit/monthly quota cap (vs a
    transient per-minute rate limit). Matches ``error.message`` substrings.
    """
    err = (body or {}).get('error') or {}
    msg = (err.get('message') or '').lower()
    if not msg:
        return False
    return any(kw in msg for kw in _QUOTA_EXHAUSTED_KEYWORDS)


def _is_workspace_usage_limit(body: dict) -> bool:
    """True iff a 400 error body indicates a maxed workspace API usage limit."""
    err = (body or {}).get('error') or {}
    msg = (err.get('message') or '').lower()
    if not msg:
        return False
    return any(kw in msg for kw in _WORKSPACE_LIMIT_KEYWORDS)


def raise_if_quota_exhausted(status_code: int, response) -> None:
    """Batch-retry hook: turn a 429 naming a hard quota cap into
    ``QuotaExhaustedError`` so the shared retry loop aborts instead of futilely
    retrying. A 429 that is *not* a quota cap (transient per-minute limit)
    returns None and is retried by the caller; non-429 statuses are ignored.

    Passed as ``on_status`` to :func:`docgen.llm.batch.request_with_retry` by
    both the Anthropic and OpenAI batch strategies (OpenAI surfaces the same
    quota-cap 429 shape).
    """
    if status_code != 429:
        return
    try:
        body = response.json() if hasattr(response, 'json') else {}
    except (ValueError, json.JSONDecodeError):
        body = {}
    if _is_quota_exhausted(body):
        msg = (body.get('error') or {}).get('message') or 'quota exhausted'
        raise QuotaExhaustedError(msg)


@define
class AnthropicProvider:
    """LLM provider for Anthropic's native ``/v1/messages`` endpoint."""
    model: str
    api_key: str
    base_url: str = 'https://api.anthropic.com/v1'
    # Up from 3 → 5 so a 1-2 minute Anthropic outage doesn't fail the
    # run. Combined with exponential backoff capped at 30s, this gives
    # ~60s of total retry wall-time per request.
    max_retries: int = 5
    retry_delay: float = 1.0
    timeout: float = 120.0
    _client: httpx.AsyncClient | None = field(default=None, init=False)
    cache_stats: CacheStats = field(factory=CacheStats, init=False)
    last_usage: dict = field(factory=dict, init=False)

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers={
                    'Content-Type': 'application/json',
                    'x-api-key': self.api_key,
                    'anthropic-version': ANTHROPIC_API_VERSION,
                    'anthropic-beta': _ANTHROPIC_BETAS,
                },
            )
        return self._client

    async def aclose(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def _emit_usage(self, usage: dict) -> None:
        """Feed real token usage to the calibration store (no-op unless a
        run set an observer + context). The unary ``call`` and the
        batch-result paths share the same usage-block shape."""
        from docgen.calibration import emit_usage
        emit_usage(
            model=self.model,
            input_tokens=usage.get('input_tokens', 0) or 0,
            output_tokens=usage.get('output_tokens', 0) or 0,
        )

    async def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int = 4096,
        cache_system_prompt: bool = True,
    ) -> str | None:
        """Run a chat completion against /messages.

        ``cache_system_prompt`` controls whether the system prompt is wrapped
        as a content block with ``cache_control={"type": "ephemeral"}``.
        Default True: every doc-type uses a static system prompt that's
        identical across all files in a generation run, so the second-and-
        later calls in a 5-min window read the cached prefix at ~10% cost.
        Pass False to fall back to the legacy flat-string shape (rollback).

        Anthropic's caching minimum is ~1024 tokens for Opus/Sonnet 4 series;
        shorter prompts are accepted but skip the cache silently.
        """
        client = self._get_client()

        payload: dict = {
            'model': self.model,
            'max_tokens': max_tokens,  # required by Anthropic
            'messages': [
                {'role': 'user', 'content': user_prompt},
            ],
        }
        # Only include ``system`` when there's actually a system prompt.
        # Anthropic rejects ``"system": []`` with 400 Bad Request, so the
        # previous "emit an empty list when no system prompt" branch was
        # broken for callers like catalog-describe that don't use a
        # system role. When we do have one, wrap it as a content block
        # with ``cache_control`` so the 5-min ephemeral cache applies.
        if system_prompt:
            if cache_system_prompt:
                payload['system'] = [
                    {
                        'type': 'text',
                        'text': system_prompt,
                        'cache_control': {'type': 'ephemeral'},
                    },
                ]
            else:
                payload['system'] = system_prompt

        _logger.info(
            'Anthropic request: model=%s prompt_len=%d cache=%s',
            self.model, len(user_prompt),
            'on' if cache_system_prompt and system_prompt else 'off',
        )

        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            t0 = _time.monotonic()
            try:
                response = await client.post('/messages', json=payload)
                elapsed = _time.monotonic() - t0
                response.raise_for_status()
                data = response.json()
                usage = data.get('usage', {}) if isinstance(data, dict) else {}
                self.last_usage = dict(usage)
                if isinstance(data, dict) and data.get('stop_reason') is not None:
                    self.last_usage['stop_reason'] = data.get('stop_reason')
                cache_create = usage.get('cache_creation_input_tokens', 0) or 0
                cache_read = usage.get('cache_read_input_tokens', 0) or 0
                self.cache_stats.record(
                    create_tokens=cache_create, read_tokens=cache_read,
                )
                _logger.info(
                    'Anthropic response: status=%d elapsed=%.1fs '
                    'in=%d out=%d cache_create=%d cache_read=%d',
                    response.status_code, elapsed,
                    usage.get('input_tokens', 0) or 0,
                    usage.get('output_tokens', 0) or 0,
                    cache_create, cache_read,
                )
                # Feed real token usage to the calibration store.
                self._emit_usage(usage)
                # Response shape: { content: [ { type: "text", text: "..." } ] }
                content_blocks = data.get('content', [])
                texts = [
                    b.get('text', '') for b in content_blocks
                    if b.get('type') == 'text'
                ]
                return ''.join(texts) if texts else None
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                try:
                    err_body = e.response.json() if hasattr(e.response, 'json') else {}
                except (ValueError, json.JSONDecodeError):
                    err_body = {}
                # Hard, non-retryable account caps — surface a clean typed
                # error (and skip the payload dump, which is for diagnosing
                # genuine request bugs) so callers fail gracefully. Keep the
                # two distinct: a 429 credit/quota vs a 400 workspace usage
                # cap (each carries its own remedy in the message).
                if status == 429 and _is_quota_exhausted(err_body):
                    msg = (err_body.get('error') or {}).get('message') or 'quota exhausted'
                    _logger.error('Anthropic quota exhausted; aborting: %s', msg)
                    raise QuotaExhaustedError(msg) from e
                if status == 400 and _is_workspace_usage_limit(err_body):
                    msg = (
                        (err_body.get('error') or {}).get('message')
                        or 'workspace API usage limit reached'
                    )
                    _logger.error(
                        'Anthropic workspace usage limit reached; aborting: %s', msg,
                    )
                    raise WorkspaceUsageLimitError(msg) from e
                # Dump the request payload + response body on a genuine 4xx so
                # the user can diagnose without enabling debug logging. The
                # short string str(e) hides what Anthropic actually rejected.
                if 400 <= status < 500:
                    try:
                        resp_text = e.response.text[:500]
                    except Exception:
                        resp_text = '<unavailable>'
                    _logger.error(
                        'Anthropic %d on /messages — payload keys=%s, '
                        'model=%s, max_tokens=%s, msg_count=%d, '
                        'system_present=%s, user_len=%d. Response: %s',
                        status, sorted(payload.keys()), self.model,
                        payload.get('max_tokens'),
                        len(payload.get('messages', [])),
                        'system' in payload, len(user_prompt or ''),
                        resp_text,
                    )
                # Anthropic uses 429 for rate limits and 529 for overloaded.
                if status >= 500 or status in (429, 529):
                    if attempt + 1 >= self.max_retries:
                        _logger.error(
                            'Anthropic request: exhausted %d retries '
                            'on HTTP %d',
                            self.max_retries, status,
                        )
                        break
                    # Exponential backoff on all transient classes
                    # (was linear 1s for vanilla 5xx, which let 502
                    # storms through). Cap at 30s so a single retry
                    # doesn't sit forever.
                    delay = min(self.retry_delay * (2 ** attempt), 30.0)
                    _logger.warning(
                        'Anthropic request: HTTP %d on attempt '
                        '%d/%d, retrying in %.1fs',
                        status, attempt + 1, self.max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                else:
                    raise
            except httpx.RequestError as e:
                last_error = e
                if attempt + 1 >= self.max_retries:
                    _logger.error(
                        'Anthropic request: exhausted %d retries on '
                        'network error',
                        self.max_retries,
                    )
                    break
                delay = min(self.retry_delay * (2 ** attempt), 30.0)
                _logger.warning(
                    'Anthropic network error on attempt %d/%d (%s), '
                    'retrying in %.1fs',
                    attempt + 1, self.max_retries, e, delay,
                )
                await asyncio.sleep(delay)

        _logger.error('All Anthropic request attempts failed: %s', last_error)
        return None
