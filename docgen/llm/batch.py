"""Provider-agnostic batch dispatch abstraction.

Batch dispatch — submit → poll → fetch → cancel — is a lifecycle shared by
Anthropic's Message Batches API and OpenAI's Batch API: both trade up to 24h
of latency for a ~50% token discount. This module holds the neutral request /
response dataclasses plus the ``BatchStrategy`` protocol that both concrete
strategies satisfy. The strategies themselves live next to their providers
(``anthropic_batch.py``, ``openai_batch.py``) and are chosen per-config by
``make_batch_strategy``.
"""
from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

import httpx
from attrs import frozen


@frozen
class BatchRequest:
    """One element of a batch submission.

    ``custom_id`` lets the caller correlate results back to the (file,
    doc_type) pair — or catalog element — that originated the request. The
    provider treats it as opaque.
    """
    custom_id: str
    system_prompt: str
    user_prompt: str
    max_tokens: int = 4096


@frozen
class BatchSubmission:
    """Result of ``submit_batch`` — only the batch_id matters for polling."""
    batch_id: str


@frozen
class BatchStatus:
    """Snapshot of a batch's state. ``processing_status`` is normalized to
    Anthropic's vocabulary (``in_progress`` | ``canceling`` | ``ended``); the
    OpenAI strategy maps its own statuses onto the same three so callers stay
    provider-agnostic.
    """
    batch_id: str
    processing_status: str
    processing: int
    succeeded: int
    errored: int


@runtime_checkable
class BatchStrategy(Protocol):
    """The submit → poll → fetch → cancel lifecycle, independent of provider.

    Implementations: ``AnthropicBatchStrategy`` (Message Batches API) and
    ``OpenAIBatchStrategy`` (Batch API). Both report results as
    ``{custom_id: text | None}`` where ``None`` marks an errored row.
    """

    async def submit_batch(
        self, requests: Sequence[BatchRequest],
    ) -> BatchSubmission: ...

    async def poll_batch(
        self,
        batch_id: str,
        *,
        poll_interval: float = 30.0,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> BatchStatus: ...

    async def fetch_batch_results(
        self, batch_id: str,
    ) -> dict[str, str | None]: ...

    async def cancel_batch(self, batch_id: str) -> None: ...


async def request_with_retry(
    client,
    method: str,
    url: str,
    *,
    retry_delay: float,
    logger: logging.Logger,
    label: str,
    max_retries: int = 6,
    max_delay: float = 30.0,
    on_status: Callable[[int, object], None] | None = None,
    **kwargs,
) -> httpx.Response:
    """Issue an HTTP request to a batch endpoint with transient-error retries.

    Shared by both batch strategies. Provider batch routes (Anthropic Message
    Batches, OpenAI Batch API) hit gateway-level outages — 502/503/504, 429
    rate-limit, 529 overloaded — that routinely last seconds-to-minutes. The
    unary ``call`` budget (≈3 attempts) gives up too soon for batch
    submit/poll, which is critical-path: there's no fallback once the user has
    paid for prompts, and the batch itself runs for minutes-to-hours, so a
    generous exponential backoff (1,2,4,8,16,30s, capped at ``max_delay``) is
    well-amortized.

    Retry policy:
      - 5xx / 429 / 529: exponential backoff, up to ``max_retries`` attempts
      - other 4xx: raised immediately (no retry helps)
      - network errors (``httpx.RequestError``): exponential backoff
      - exhausted: re-raise the last error so the caller's recovery
        (``pending_batches`` row, ``--resume``) can take over

    ``on_status(status, response)`` runs first on every HTTP-status error and
    may raise to abort without retrying — providers use it to turn a
    quota-cap 429 into ``QuotaExhaustedError`` (retrying a hard cap is futile).
    Only GET/POST are used today.
    """
    last_error: Exception | None = None
    method = method.upper()
    for attempt in range(max_retries):
        try:
            # Verb-specific helpers (not .request) so narrow test doubles
            # stay valid; real httpx routes both through .request internally.
            if method == 'GET':
                response = await client.get(url, **kwargs)
            elif method == 'POST':
                response = await client.post(url, **kwargs)
            else:  # pragma: no cover — only GET/POST used today
                response = await client.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            last_error = e
            status = e.response.status_code
            if on_status is not None:
                on_status(status, e.response)  # may raise to abort (quota)
            if status >= 500 or status in (429, 529):
                if attempt + 1 >= max_retries:
                    logger.error(
                        '%s %s %s: exhausted %d retries on %d; surfacing',
                        label, method, url, max_retries, status,
                    )
                    break
                delay = min(retry_delay * (2 ** attempt), max_delay)
                logger.warning(
                    '%s %s %s: HTTP %d on attempt %d/%d, retrying in %.1fs',
                    label, method, url, status, attempt + 1,
                    max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue
            # 4xx (non-429) — no retry helps; surface immediately.
            raise
        except httpx.RequestError as e:
            last_error = e
            if attempt + 1 >= max_retries:
                logger.error(
                    '%s %s %s: exhausted %d retries on network error; '
                    'surfacing', label, method, url, max_retries,
                )
                break
            delay = min(retry_delay * (2 ** attempt), max_delay)
            logger.warning(
                '%s %s %s: network error on attempt %d/%d (%s), '
                'retrying in %.1fs',
                label, method, url, attempt + 1, max_retries, e, delay,
            )
            await asyncio.sleep(delay)
    # Out of retries — surface the last error so the caller's own recovery
    # path (pending_batches, --resume) can react.
    assert last_error is not None
    raise last_error
