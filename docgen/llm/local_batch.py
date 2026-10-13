"""Local implementation of the neutral batch API; no upload or /batches calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference import InferenceContext
    from library import Library

from collections.abc import Sequence

from docgen.llm.batch import BatchRequest, BatchStatus, BatchSubmission
from docgen.spool import LocalExecutor, SpoolJobSpec, SpoolStore


class LocalBatchStrategy:
    def __init__(
        self,
        context: InferenceContext,
        *,
        library: Library,
        job_specs: list[SpoolJobSpec],
        operation: str = 'generate',
        scope: dict | None = None,
    ) -> None:
        self.context = context
        self.store = SpoolStore(context, library)
        self.specs = job_specs
        self.operation = operation
        self.scope = scope or {}

    async def submit_batch(self, requests: Sequence[BatchRequest]) -> BatchSubmission:
        expected = {spec.custom_id: spec.request for spec in self.specs if spec.request is not None}
        actual = {request.custom_id: request for request in requests}
        if len(actual) != len(requests) or actual != expected:
            raise ValueError('Submitted requests must match the complete local job plan')
        return BatchSubmission(
            self.store.create(self.specs, operation=self.operation, scope=self.scope)
        )

    async def poll_batch(
        self,
        batch_id: str,
        *,
        poll_interval: float = 30.0,
        on_progress: Callable[[int, int, int], None] | None = None,
    ) -> BatchStatus:
        del poll_interval  # The local worker executes in the foreground.
        counts = await LocalExecutor(self.store).drive(
            batch_id, inference_only=True, finalize=False
        )
        failures = counts['failed'] + counts['obsolete'] + counts['cancelled']
        pending = counts['queued'] + counts['running']
        if on_progress:
            on_progress(pending, counts['responses'], failures)
        return BatchStatus(
            batch_id, 'in_progress' if pending else 'ended', pending, counts['responses'], failures
        )

    async def fetch_batch_results(self, batch_id: str) -> dict[str, str | None]:
        return {job['custom_id']: job['response_text'] for job in self.store.jobs(batch_id)}

    async def cancel_batch(self, batch_id: str) -> None:
        self.store.cancel(batch_id)

    async def aclose(self) -> None:
        # The operation context owns the shared transport.
        return None
