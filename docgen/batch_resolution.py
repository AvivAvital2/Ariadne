"""Pure-logic helpers for batch vs sync dispatch resolution.

Used by both:
- ``cli_generate._print_cost_estimate`` (dry-run cost display)
- ``DocGenOrchestrator.run`` (runtime dispatch fork)

Centralizing the resolution rules keeps the dry-run estimate in
lockstep with what actually happens at runtime — a divergence would
let the estimator advertise a 50% batch discount that runtime never
takes (or vice-versa). Lock-step matters because the dry-run is what
users see before paying for the LLM calls.

Two helpers:
- ``resolve_batch_decision`` — pure logic over inputs (provider /
  batch_mode / planned_calls / auto_threshold).
- ``apply_dispatch_gate`` — downgrades batch=True to sync when the
  ``BATCH_DISPATCH_IMPLEMENTED`` feature flag is False, so partial
  rollouts of #45 don't mis-quote the discount.
"""
from __future__ import annotations


# Providers with a wired batch backend (a BatchStrategy in docgen.llm):
# anthropic → Message Batches API, openai → Batch API. Both trade up to 24h
# of latency for ~50% off. cli.generate keeps its batch_eligible precompute in
# sync with this set.
BATCH_ELIGIBLE_PROVIDERS = frozenset({'anthropic', 'openai'})


def resolve_batch_decision(
    *,
    provider: str,
    batch_mode: str,
    planned_calls: int,
    auto_threshold: int,
) -> tuple[bool, str]:
    """Decide batch vs sync from inputs.

    Args:
        provider: LLM provider name. ``'anthropic'`` (Message Batches)
            and ``'openai'`` (Batch API) are batch-eligible; any other
            provider forces sync.
        batch_mode: ``'always'`` / ``'never'`` / ``'auto'``.
        planned_calls: Total planned LLM calls (files × doc_types).
            Only consulted in ``'auto'`` mode.
        auto_threshold: In auto mode, batch iff
            ``planned_calls >= auto_threshold``.

    Returns:
        ``(batch_resolved, reason)``. The reason is shown in dry-run
        output and orchestrator logs so users understand why their
        ``batch_mode`` took (or didn't take) effect.
    """
    if provider not in BATCH_ELIGIBLE_PROVIDERS:
        eligible = ', '.join(sorted(BATCH_ELIGIBLE_PROVIDERS))
        return False, (
            f'cannot batch: provider={provider} '
            f'(no batch backend wired; eligible: {eligible})'
        )

    if batch_mode == 'always':
        return True, '--batch override'
    if batch_mode == 'never':
        return False, '--no-batch override'

    # Auto: compare planned to threshold. ``>=`` not ``>`` so a run
    # right AT the threshold takes the batch path — users who set the
    # threshold to N expect "N or more" to batch.
    if planned_calls >= auto_threshold:
        return True, (
            f'auto: {planned_calls} calls ≥ threshold {auto_threshold}'
        )
    return False, (
        f'auto: {planned_calls} calls < threshold {auto_threshold}'
    )


def apply_dispatch_gate(
    batch_resolved: bool,
    batch_reason: str,
) -> tuple[bool, str]:
    """Downgrade batch=True to sync when ``BATCH_DISPATCH_IMPLEMENTED``
    is False, so the dry-run estimator and runtime stay in lockstep
    while #45 is in partial rollout.

    No-op once the flag flips True (#45.9). Pass-through for sync
    inputs regardless — the gate only acts on batch=True.

    Note: ``BATCH_DISPATCH_IMPLEMENTED`` is imported lazily inside
    the function so:
    1. tests can monkeypatch the symbol on ``docgen.orchestrator``
       and have each call observe the patched value (a top-level
       import here would freeze the value at first import);
    2. this pure-logic module avoids pulling the orchestrator's full
       transitive graph (Library, generator, analyzer) at import
       time.
    """
    from docgen.orchestrator import BATCH_DISPATCH_IMPLEMENTED

    if batch_resolved and not BATCH_DISPATCH_IMPLEMENTED:
        return False, (
            f'{batch_reason} — runtime dispatch not yet implemented; '
            'sync prices apply'
        )
    return batch_resolved, batch_reason
