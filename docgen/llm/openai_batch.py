"""OpenAI Batch API strategy.

The submit → poll → fetch → cancel lifecycle over OpenAI's Batch API: upload a
JSONL file of ``/v1/chat/completions`` requests, create a batch with a 24h
completion window (~50% off, the same economics as Anthropic's Message
Batches), poll until terminal, then download the output (and error) files.

Composes an ``OpenAIProvider`` purely for its authenticated httpx transport,
mirroring ``AnthropicBatchStrategy`` so ``make_batch_strategy`` can return
either behind the ``BatchStrategy`` protocol. Statuses and request counts are
normalized onto the shared ``BatchStatus`` vocabulary so callers stay
provider-agnostic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import numpy as np
from collections.abc import Callable, Sequence
from attrs import frozen
from numpy.typing import NDArray

from docgen.llm.anthropic import raise_if_quota_exhausted
from docgen.llm.batch import (
    BatchRequest,
    BatchStatus,
    BatchSubmission,
    request_with_retry,
)
from docgen.llm.openai import OpenAIProvider, token_limit_field
from embedding import EmbeddingService

_logger = logging.getLogger(__name__)

# OpenAI batch statuses that mean "done" (no further polling). Maps onto the
# shared ``ended`` vocabulary regardless of success/failure — the per-request
# outcome is conveyed by the succeeded/errored counts and the results files.
_TERMINAL_STATUSES = {'completed', 'failed', 'expired', 'cancelled'}


def _extract_text(row: dict) -> str | None:
    """Pull the assistant text out of one output-file row, or ``None`` if the
    request didn't succeed (non-200 / empty)."""
    response = row.get('response') or {}
    if response.get('status_code') != 200:
        return None
    body = response.get('body') or {}
    choices = body.get('choices') or []
    if not choices:
        return None
    content = (choices[0].get('message') or {}).get('content')
    return content or None


class OpenAIBatchStrategy:
    """Batch lifecycle over OpenAI's Batch API."""
    # Endpoint recorded at batch creation; the embeddings subclass narrows it.
    _endpoint = '/v1/chat/completions'

    def __init__(self, provider: OpenAIProvider) -> None:
        """Borrow ``provider`` for its authenticated httpx transport. This
        strategy owns only the batch-specific request shaping and polling."""
        self._provider = provider

    @property
    def _model(self) -> str:
        """Model recorded in submit logs; the embeddings subclass overrides."""
        return self._provider.model

    def _build_batch_line(self, req: BatchRequest) -> dict:
        """One JSONL line: a ``/v1/chat/completions`` request keyed by
        custom_id. Mirrors ``OpenAIProvider.call`` — system+user messages, the
        model-appropriate token field, and no ``temperature`` (modern models
        reject it)."""
        body: dict = {
            'model': self._provider.model,
            'messages': [
                {'role': 'system', 'content': req.system_prompt},
                {'role': 'user', 'content': req.user_prompt},
            ],
        }
        body[token_limit_field(self._provider.model)] = req.max_tokens
        return {
            'custom_id': req.custom_id,
            'method': 'POST',
            'url': '/v1/chat/completions',
            'body': body,
        }

    async def _request_with_retry(self, method: str, url: str, **kwargs):
        """OpenAI Batch API request with the shared batch retry budget; a 429
        naming a hard quota cap aborts via ``QuotaExhaustedError``. See
        :func:`docgen.llm.batch.request_with_retry`."""
        return await request_with_retry(
            self._provider._get_client(), method, url,
            retry_delay=self._provider.retry_delay,
            logger=_logger, label='OpenAI batch',
            on_status=raise_if_quota_exhausted,
            **kwargs,
        )

    async def submit_batch(
        self, requests: Sequence[BatchRequest],
    ) -> BatchSubmission:
        """Upload the JSONL input file, then create a 24h batch over it.

        Two calls: ``POST /files`` (multipart, ``purpose=batch``) to register
        the input, then ``POST /batches`` to start processing. The caller
        polls via ``poll_batch`` and fetches via ``fetch_batch_results``.
        """
        jsonl = '\n'.join(
            json.dumps(self._build_batch_line(req)) for req in requests
        )
        upload = await self._request_with_retry(
            'POST', '/files',
            data={'purpose': 'batch'},
            files={
                'file': (
                    'batch_input.jsonl',
                    jsonl.encode('utf-8'),
                    'application/jsonl',
                ),
            },
        )
        input_file_id = upload.json()['id']

        _logger.info(
            'OpenAI batch submit: model=%s n=%d input_file=%s',
            self._model, len(requests), input_file_id,
        )
        response = await self._request_with_retry('POST', '/batches', json={
            'input_file_id': input_file_id,
            'endpoint': self._endpoint,
            'completion_window': '24h',
        })
        batch_id = response.json()['id']
        _logger.info('OpenAI batch submitted: id=%s', batch_id)
        return BatchSubmission(batch_id=batch_id)

    async def cancel_batch(self, batch_id: str) -> None:
        """Cancel an in-flight batch so it stops processing — and being billed
        — at OpenAI. Best-effort, like the Anthropic strategy."""
        _logger.info('OpenAI batch cancel: id=%s', batch_id)
        await self._request_with_retry(
            'POST', f'/batches/{batch_id}/cancel',
        )

    async def poll_batch(
        self,
        batch_id: str,
        *,
        poll_interval: float = 30.0,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> BatchStatus:
        """Block (with sleeps) until the batch reaches a terminal status.

        Each poll fires ``on_progress(processing, succeeded, errored)``.
        OpenAI's ``request_counts`` reports ``total``/``completed``/``failed``;
        we derive ``processing = total - completed - failed`` and map the
        terminal status onto the shared ``ended`` vocabulary.
        """
        while True:
            response = await self._request_with_retry(
                'GET', f'/batches/{batch_id}',
            )
            data = response.json()
            counts = data.get('request_counts', {}) or {}
            total = int(counts.get('total', 0) or 0)
            succeeded = int(counts.get('completed', 0) or 0)
            errored = int(counts.get('failed', 0) or 0)
            processing = max(total - succeeded - errored, 0)
            status = data.get('status', 'in_progress')
            if on_progress is not None:
                on_progress(processing, succeeded, errored)
            if status in _TERMINAL_STATUSES:
                _logger.info(
                    'OpenAI batch ended: id=%s status=%s succeeded=%d errored=%d',
                    batch_id, status, succeeded, errored,
                )
                return BatchStatus(
                    batch_id=batch_id,
                    processing_status='ended',
                    processing=processing,
                    succeeded=succeeded,
                    errored=errored,
                )
            if poll_interval > 0:
                await asyncio.sleep(poll_interval)

    async def fetch_batch_results(
        self, batch_id: str,
    ) -> dict[str, str | None]:
        """Return ``{custom_id: text | None}`` by merging the batch's output
        file (succeeded rows → text) and error file (failed rows → None).

        OpenAI splits results across two files — successes land in
        ``output_file_id``, failures in ``error_file_id`` — unlike Anthropic's
        single results stream, so both are read to reconstruct the full map.
        """
        client = self._provider._get_client()
        meta = await client.get(f'/batches/{batch_id}')
        meta.raise_for_status()
        data = meta.json()

        results: dict[str, str | None] = {}
        output_file_id = data.get('output_file_id')
        if output_file_id:
            for row in await self._download_jsonl(client, output_file_id):
                cid = row.get('custom_id')
                if cid:
                    results[cid] = _extract_text(row)
        error_file_id = data.get('error_file_id')
        if error_file_id:
            for row in await self._download_jsonl(client, error_file_id):
                cid = row.get('custom_id')
                if cid:
                    results[cid] = None

        _logger.info(
            'OpenAI batch results fetched: id=%s n=%d ok=%d',
            batch_id, len(results), sum(1 for v in results.values() if v),
        )
        return results

    @staticmethod
    async def _download_jsonl(client, file_id: str) -> list[dict]:
        response = await client.get(f'/files/{file_id}/content')
        response.raise_for_status()
        rows: list[dict] = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                _logger.warning(
                    'OpenAI batch result: malformed JSONL line, skipping',
                )
        return rows


def _extract_group_vectors(row: dict) -> list[NDArray[np.float32]] | None:
    """Vectors for one embeddings output-file row, ordered by ``index`` and
    unit-normalized (the live ``embed_batch`` contract), or ``None`` if the
    request didn't succeed."""
    response = row.get('response') or {}
    if response.get('status_code') != 200:
        return None
    data = (response.get('body') or {}).get('data') or []
    if not data:
        return None
    vectors: list[NDArray[np.float32]] = []
    for item in sorted(data, key=lambda x: x['index']):
        vector = np.array(item['embedding'], dtype=np.float32)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        vectors.append(vector)
    return vectors


@frozen
class EmbeddingBatchRequest:
    """One JSONL line of an embeddings batch: a GROUP of texts embedded in
    a single request, so a large rebuild stays far below the Batch API's
    per-job request cap. ``custom_id`` correlates the group's vectors back
    to the caller's bookkeeping; the provider treats it as opaque."""
    custom_id: str
    texts: list[str]


class OpenAIEmbeddingsBatchStrategy(OpenAIBatchStrategy):
    """Batch lifecycle over OpenAI's Batch API for ``/v1/embeddings``.

    Inherits the submit/poll/cancel mechanics (same ~50%-off, 24h-window
    economics) and swaps the request shaping and result parsing: lines are
    grouped-text embeddings requests, and results are unit-normalized
    float32 vectors per custom_id (``None`` for failed groups). Borrows
    ``EmbeddingService`` for its authenticated transport and embedding
    config, the way the chat strategy borrows ``OpenAIProvider``.
    """

    _endpoint = '/v1/embeddings'

    def __init__(self, service: EmbeddingService) -> None:
        """Borrow ``service`` for its authenticated httpx transport and
        embedding config (model, dimensions)."""
        self._service = service

    @property
    def _model(self) -> str:
        return self._service.config.model

    def _build_batch_line(self, req: EmbeddingBatchRequest) -> dict:
        body: dict = {
            'model': self._service.config.model,
            'input': list(req.texts),
        }
        if self._service.config.dimensions is not None:
            body['dimensions'] = self._service.config.dimensions
        return {
            'custom_id': req.custom_id,
            'method': 'POST',
            'url': '/v1/embeddings',
            'body': body,
        }

    async def _request_with_retry(self, method: str, url: str, **kwargs):
        """Same retry budget as the chat strategy, over the embedding
        service's client (an async accessor, unlike the provider's)."""
        client = await self._service._get_client()
        return await request_with_retry(
            client, method, url,
            retry_delay=2.0, logger=_logger, label='OpenAI embeddings batch',
            on_status=raise_if_quota_exhausted,
            **kwargs,
        )

    async def fetch_batch_results(
        self, batch_id: str,
    ) -> dict[str, list[NDArray[np.float32]] | None]:
        """``{custom_id: [vectors] | None}`` merged from the batch's output
        file (succeeded groups) and error file (failed groups)."""
        client = await self._service._get_client()
        meta = await client.get(f'/batches/{batch_id}')
        meta.raise_for_status()
        data = meta.json()

        results: dict[str, list[NDArray[np.float32]] | None] = {}
        output_file_id = data.get('output_file_id')
        if output_file_id:
            for row in await self._download_jsonl(client, output_file_id):
                cid = row.get('custom_id')
                if cid:
                    results[cid] = _extract_group_vectors(row)
        error_file_id = data.get('error_file_id')
        if error_file_id:
            for row in await self._download_jsonl(client, error_file_id):
                cid = row.get('custom_id')
                if cid:
                    results[cid] = None

        _logger.info(
            'OpenAI embeddings batch results fetched: id=%s n=%d ok=%d',
            batch_id, len(results),
            sum(1 for v in results.values() if v is not None),
        )
        return results
