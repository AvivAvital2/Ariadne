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

from collections.abc import Callable, Sequence
from typing import Protocol, runtime_checkable

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
