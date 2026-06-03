"""Contract tests for ``docgen.batch_resolution``.

The dry-run cost estimator (``cli_generate._print_cost_estimate``)
and the runtime dispatch (``DocGenOrchestrator.run``) BOTH consult
these helpers to decide batch-vs-sync. Any divergence between the
two sites would let the dry-run quote a 50% batch discount the
runtime never takes (or vice-versa) — billing surprises the user
literally cannot detect until the invoice arrives.

Each test pins one branch of the resolver. Tests are written against
the stub (which always returns sentinel values), so they must all
fail with assertion errors — NOT with ``ImportError`` or
``NotImplementedError``. Behavioral red → behavioral green.

The ``apply_dispatch_gate`` cases pair "flag-off downgrades batch"
with "flag-off preserves sync" and "flag-on preserves batch", so
a too-broad implementation that always returns False (or always
returns the input) fails one of the three.
"""
from __future__ import annotations

import pytest

from docgen.batch_resolution import (
    apply_dispatch_gate,
    resolve_batch_decision,
)


# ---------------------------------------------------------------------------
# resolve_batch_decision — pure logic, no flag involved
# ---------------------------------------------------------------------------


class TestResolveBatchDecision:
    """Pure-logic resolution: provider eligibility + always/never/auto."""

    def test_openai_is_batch_eligible(self) -> None:
        """OpenAI's Batch API is now wired (OpenAIBatchStrategy), so
        ``batch_mode='always'`` resolves to batch for openai just like
        anthropic — both offer a 24h / ~50%-off batch lane."""
        resolved, reason = resolve_batch_decision(
            provider='openai',
            batch_mode='always',
            planned_calls=1000,
            auto_threshold=200,
        )
        assert resolved is True
        assert 'override' in reason.lower() or 'always' in reason.lower()

    def test_unsupported_provider_forces_sync(self) -> None:
        """A provider with no batch backend (e.g. a hypothetical 'gemini')
        MUST resolve to sync even with ``batch_mode='always'``, with a reason
        that names the provider so the user knows why their flag was ignored."""
        resolved, reason = resolve_batch_decision(
            provider='gemini',
            batch_mode='always',
            planned_calls=1000,
            auto_threshold=200,
        )
        assert resolved is False
        assert 'gemini' in reason.lower()

    def test_always_with_anthropic_resolves_batch(self) -> None:
        """Explicit ``--batch always`` overrides the threshold even
        for tiny runs. User chose this; respect it."""
        resolved, reason = resolve_batch_decision(
            provider='anthropic',
            batch_mode='always',
            planned_calls=1,
            auto_threshold=200,
        )
        assert resolved is True
        # Reason must signal it was an override, not a threshold hit.
        assert 'override' in reason.lower() or 'always' in reason.lower()

    def test_never_with_anthropic_resolves_sync(self) -> None:
        """Explicit ``--no-batch`` overrides the threshold even for
        massive runs. User explicitly opted out; respect it."""
        resolved, reason = resolve_batch_decision(
            provider='anthropic',
            batch_mode='never',
            planned_calls=10000,
            auto_threshold=200,
        )
        assert resolved is False
        assert 'override' in reason.lower() or 'never' in reason.lower()

    def test_auto_above_threshold_resolves_batch(self) -> None:
        """Auto mode flips to batch when planned_calls >= threshold.
        Reason must include both numbers so the user can see why."""
        resolved, reason = resolve_batch_decision(
            provider='anthropic',
            batch_mode='auto',
            planned_calls=300,
            auto_threshold=200,
        )
        assert resolved is True
        # Both numbers visible in the reason — debuggability for the
        # user reading the dry-run output.
        assert '300' in reason and '200' in reason

    def test_auto_below_threshold_resolves_sync(self) -> None:
        """Auto mode stays sync below threshold. Paired with above."""
        resolved, reason = resolve_batch_decision(
            provider='anthropic',
            batch_mode='auto',
            planned_calls=199,
            auto_threshold=200,
        )
        assert resolved is False
        assert '199' in reason and '200' in reason

    def test_auto_at_exact_threshold_resolves_batch(self) -> None:
        """Boundary: planned == threshold counts as ≥ — bites a fix
        that uses ``>`` instead of ``>=``."""
        resolved, _reason = resolve_batch_decision(
            provider='anthropic',
            batch_mode='auto',
            planned_calls=200,
            auto_threshold=200,
        )
        assert resolved is True


# ---------------------------------------------------------------------------
# apply_dispatch_gate — feature-flag downgrade
# ---------------------------------------------------------------------------


class TestApplyDispatchGate:
    """The gate downgrades batch=True to sync when
    ``BATCH_DISPATCH_IMPLEMENTED`` is False, so partial rollouts of
    #45 don't let the dry-run estimator mis-claim the 50% discount.
    """

    def test_flag_off_downgrades_batch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Flag off + batch_resolved=True must downgrade to sync.
        The reason must mention the runtime gap so the user
        understands why their ``--batch always`` didn't take effect.
        """
        import docgen.orchestrator
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', False,
        )
        resolved, reason = apply_dispatch_gate(
            True, 'auto: 500 calls ≥ threshold 200',
        )
        assert resolved is False
        # Reason must signal the gap to the user, not just silently
        # downgrade. 'runtime' or 'not yet implemented' both work.
        lowered = reason.lower()
        assert 'not yet implemented' in lowered or 'runtime' in lowered

    def test_flag_off_preserves_sync(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Sync stays sync regardless of flag — gate only acts on True.
        Bites a fix that always returns False or always rewrites the
        reason."""
        import docgen.orchestrator
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', False,
        )
        resolved, reason = apply_dispatch_gate(False, '--no-batch override')
        assert resolved is False
        # Reason must pass through unchanged — no downgrade message
        # when there was nothing to downgrade.
        assert reason == '--no-batch override'

    def test_flag_on_preserves_batch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Once dispatch is wired, the gate is a no-op on batch=True.
        Pinning this behavior so flipping the flag (#45.9) actually
        unlocks the discount, not silently downgrades it forever."""
        import docgen.orchestrator
        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        resolved, reason = apply_dispatch_gate(
            True, 'auto: 500 calls ≥ threshold 200',
        )
        assert resolved is True
        # Reason must pass through unchanged.
        assert reason == 'auto: 500 calls ≥ threshold 200'
