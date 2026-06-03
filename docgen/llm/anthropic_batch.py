"""Anthropic Message Batches strategy.

The submit -> poll -> fetch -> cancel lifecycle for Anthropic's
``/v1/messages/batches`` endpoints, extracted from ``AnthropicProvider``. It
composes a provider purely for its authenticated transport (client, retry
budget, ``cache_stats`` accounting, calibration) and owns the batch-specific
request shaping and polling control flow. Selected per-config by
``make_batch_strategy`` alongside the OpenAI strategy.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable, Sequence

import httpx

from docgen.llm.anthropic import (
    AnthropicProvider,
    QuotaExhaustedError,
    _is_quota_exhausted,
)
from docgen.llm.batch import BatchRequest, BatchStatus, BatchSubmission

_logger = logging.getLogger(__name__)


class AnthropicBatchStrategy:
    def __init__(self, provider: AnthropicProvider) -> None:
        """Borrow ``provider`` for its authenticated transport — HTTP client,
        retry budget, ``cache_stats`` accounting, calibration. This strategy
        owns only the batch-specific request shaping and polling control flow.
        """
        self._provider = provider

    # ------------------------------------------------------------------
    # Message Batches API
    # ------------------------------------------------------------------

    def _build_batch_params(self, req: BatchRequest) -> dict:
        """Build the ``params`` block for one batch request.

        Mirrors the sync ``call`` payload exactly — caching markers and
        all — so cached prefixes hit even within batches.
        """
        if req.system_prompt:
            system_field: str | list[dict] = [
                {
                    'type': 'text',
                    'text': req.system_prompt,
                    'cache_control': {'type': 'ephemeral'},
                },
            ]
        else:
            system_field = []
        return {
            'model': self._provider.model,
            'max_tokens': req.max_tokens,
            'system': system_field,
            'messages': [{'role': 'user', 'content': req.user_prompt}],
        }

    # Batch endpoints get a more generous retry budget than the unary
    # ``call()`` path: Anthropic's batch routes hit gateway-level
    # outages (502/503/504) that routinely last 1-2 minutes. The unary
    # default (3 attempts, ~3s total) gives up too soon; batch
    # submission is critical-path (no fallback after the user pays for
    # 404 prompts) and the batch itself runs for minutes-to-hours, so
    # 60-120s of retries is well-amortized.
    _BATCH_MAX_RETRIES = 6
    _BATCH_MAX_DELAY = 30.0  # cap any single backoff

    async def _batch_request_with_retry(
        self, method: str, url: str, **kwargs,
    ) -> httpx.Response:
        """HTTP request to a /messages/batches endpoint with retries.

        Anthropic's batch routes intermittently return 502/503/504 under
        load (CDN/proxy hiccups, not real failures). Without retries,
        a transient gateway error mid-onboard kills hours of work right
        when the user is submitting a 400+ prompt batch.

        Retry policy:
          - 5xx: exponential backoff (1, 2, 4, 8, 16, 30s), 6 attempts
          - 429 non-quota / 529 overloaded: same exponential backoff
          - 429 quota-exhausted: raise QuotaExhaustedError (no retry)
          - other 4xx: raise immediately
          - httpx.RequestError (network): exponential backoff
          - all attempts exhausted: re-raise the last error so the
            caller's recovery (``pending_batches`` row, auto-resume on
            next invocation) can take over.
        """
        client = self._provider._get_client()
        last_error: Exception | None = None
        method = method.upper()
        max_retries = self._BATCH_MAX_RETRIES
        for attempt in range(max_retries):
            try:
                # Use the verb-specific helpers httpx exposes (post/get).
                # Lets test doubles with narrower interfaces stay valid;
                # real httpx routes both through .request() internally.
                if method == 'GET':
                    response = await client.get(url, **kwargs)
                elif method == 'POST':
                    response = await client.post(url, **kwargs)
                else:  # pragma: no cover — only GET/POST used today
                    response = await client.request(
                        method, url, **kwargs,
                    )
                response.raise_for_status()
                return response
            except httpx.HTTPStatusError as e:
                last_error = e
                status = e.response.status_code
                if status == 429:
                    try:
                        err_body = (
                            e.response.json()
                            if hasattr(e.response, 'json') else {}
                        )
                    except (ValueError, json.JSONDecodeError):
                        err_body = {}
                    if _is_quota_exhausted(err_body):
                        msg = (
                            (err_body.get('error') or {}).get('message')
                            or 'quota exhausted'
                        )
                        raise QuotaExhaustedError(msg) from e
                if status >= 500 or status in (429, 529):
                    if attempt + 1 >= max_retries:
                        _logger.error(
                            'Anthropic batch %s %s: exhausted %d '
                            'retries on %d; surfacing error',
                            method, url, max_retries, status,
                        )
                        break
                    delay = min(
                        self._provider.retry_delay * (2 ** attempt),
                        self._BATCH_MAX_DELAY,
                    )
                    _logger.warning(
                        'Anthropic batch %s %s: HTTP %d on attempt '
                        '%d/%d, retrying in %.1fs',
                        method, url, status, attempt + 1,
                        max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                # 4xx (non-429) — no retry helps; surface immediately.
                raise
            except httpx.RequestError as e:
                last_error = e
                if attempt + 1 >= max_retries:
                    _logger.error(
                        'Anthropic batch %s %s: exhausted %d retries '
                        'on network error; surfacing',
                        method, url, max_retries,
                    )
                    break
                delay = min(
                    self._provider.retry_delay * (2 ** attempt),
                    self._BATCH_MAX_DELAY,
                )
                _logger.warning(
                    'Anthropic batch %s %s: network error on attempt '
                    '%d/%d (%s), retrying in %.1fs',
                    method, url, attempt + 1, max_retries, e, delay,
                )
                await asyncio.sleep(delay)
        # Out of retries — surface the last error so the caller's own
        # recovery path (pending_batches, --resume, etc.) can react.
        assert last_error is not None
        raise last_error

    async def submit_batch(
        self, requests: Sequence[BatchRequest],
    ) -> BatchSubmission:
        """POST a batch of message requests; return the assigned batch_id.

        Each ``BatchRequest`` is wrapped into the ``{custom_id, params}``
        shape Anthropic expects. The caller polls via ``poll_batch`` and
        fetches via ``fetch_batch_results``.
        """
        body = {
            'requests': [
                {
                    'custom_id': req.custom_id,
                    'params': self._build_batch_params(req),
                }
                for req in requests
            ],
        }
        _logger.info(
            'Anthropic batch submit: model=%s n=%d',
            self._provider.model, len(body['requests']),
        )
        response = await self._batch_request_with_retry(
            'POST', '/messages/batches', json=body,
        )
        data = response.json()
        batch_id = data['id']
        _logger.info('Anthropic batch submitted: id=%s', batch_id)
        return BatchSubmission(batch_id=batch_id)

    async def cancel_batch(self, batch_id: str) -> None:
        """Cancel an in-flight Message Batch so it stops processing — and
        being billed — at Anthropic. Called when the user aborts a batch
        run. Caller treats failures (e.g. an already-ended batch can't be
        cancelled) as best-effort.
        """
        _logger.info('Anthropic batch cancel: id=%s', batch_id)
        await self._batch_request_with_retry(
            'POST', f'/messages/batches/{batch_id}/cancel',
        )

    async def poll_batch(
        self,
        batch_id: str,
        *,
        poll_interval: float = 30.0,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> BatchStatus:
        """Block (with sleeps) until batch ``processing_status == 'ended'``.

        Each poll fires ``on_progress(processing, succeeded, errored)`` so
        the CLI can render a live count without owning the polling loop.

        Polling cadence: aggressive for the first ~30s (every 3s) so the
        user sees rapid initial feedback once Anthropic begins
        processing, then backs off to ``poll_interval`` for the long
        haul. The cliff between Anthropic's "queued" state (all-zero
        counts) and the first non-zero processing count is usually
        seconds; without the early-aggressive window the progress bar
        looks stuck at 0/N for up to a full ``poll_interval``.
        """
        _FAST_POLL_BUDGET = 30.0  # seconds of aggressive polling at startup
        _FAST_POLL_DELAY = 3.0
        elapsed = 0.0
        while True:
            response = await self._batch_request_with_retry(
                'GET', f'/messages/batches/{batch_id}',
            )
            data = response.json()
            counts = data.get('request_counts', {}) or {}
            processing = int(counts.get('processing', 0) or 0)
            succeeded = int(counts.get('succeeded', 0) or 0)
            errored = int(counts.get('errored', 0) or 0)
            status = data.get('processing_status', 'in_progress')
            if on_progress is not None:
                on_progress(processing, succeeded, errored)
            if status == 'ended':
                _logger.info(
                    'Anthropic batch ended: id=%s succeeded=%d errored=%d',
                    batch_id, succeeded, errored,
                )
                return BatchStatus(
                    batch_id=batch_id,
                    processing_status=status,
                    processing=processing,
                    succeeded=succeeded,
                    errored=errored,
                )
            if poll_interval <= 0:
                continue
            delay = (
                _FAST_POLL_DELAY if elapsed < _FAST_POLL_BUDGET
                else poll_interval
            )
            await asyncio.sleep(delay)
            elapsed += delay

    async def fetch_batch_results(self, batch_id: str) -> dict[str, str | None]:
        """Download the JSONL results and return ``{custom_id: text or None}``.

        ``None`` indicates the request errored — caller decides whether to
        retry synchronously or surface the failure.

        Each succeeded row's ``message.usage`` block is also fed into
        ``self.cache_stats.record(...)`` so batch runs report cache
        savings the same way streaming runs do (#45.11). Errored rows
        skip the update — they have no usage block.
        """
        response = await self._batch_request_with_retry(
            'GET', f'/messages/batches/{batch_id}/results',
        )
        results: dict[str, str | None] = {}
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                _logger.warning('Anthropic batch result: malformed JSONL line, skipping')
                continue
            cid = row.get('custom_id')
            if not cid:
                continue
            result = row.get('result', {}) or {}
            if result.get('type') == 'succeeded':
                msg = result.get('message', {}) or {}
                # Update cache stats from the usage block. Same shape as
                # the per-call ``call`` path uses; errored rows are
                # skipped because they have no usage.
                usage = msg.get('usage', {}) or {}
                create_tokens = int(
                    usage.get('cache_creation_input_tokens', 0) or 0
                )
                read_tokens = int(
                    usage.get('cache_read_input_tokens', 0) or 0
                )
                self._provider.cache_stats.record(
                    create_tokens=create_tokens,
                    read_tokens=read_tokens,
                )
                # Real per-result usage → calibration store.
                self._provider._emit_usage(usage)
                blocks = msg.get('content', []) or []
                texts = [
                    b.get('text', '') for b in blocks
                    if isinstance(b, dict) and b.get('type') == 'text'
                ]
                results[cid] = ''.join(texts) if texts else None
            else:
                results[cid] = None
        _logger.info(
            'Anthropic batch results fetched: id=%s n=%d ok=%d',
            batch_id, len(results), sum(1 for v in results.values() if v),
        )
        return results
