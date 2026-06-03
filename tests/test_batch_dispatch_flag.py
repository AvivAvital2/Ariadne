"""Contract for the batch-dispatch feature flag (Phase C-fix).

Until ``DocGenOrchestrator.run`` is wired to dispatch via
``provider.submit_batch`` (#45), the dry-run cost estimator must NOT
claim the 50% Anthropic batch discount — otherwise ``--batch always``
promises a saving the user never receives.

``cli_generate._resolve_batch_dispatch`` enforces this by downgrading
``batch_resolved`` to False when ``BATCH_DISPATCH_IMPLEMENTED`` is
off, while preserving the user's intent in the reason string.

These tests pin the contract so that:

- A future refactor that drops the flag check fails one test.
- Flipping the flag (when #45 lands) doesn't silently break the
  pricing path — the paired flag-on test verifies the unguarded
  flow still works.
"""
from __future__ import annotations

import pytest


class TestBatchDispatchFlag:
    def test_flag_off_downgrades_resolved_batch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When batch resolved True via user intent or threshold AND
        the flag is False, the helper forces False and appends the
        runtime-not-wired note. The dollar figure in the dry-run
        therefore reflects sync prices, not the lying 50% discount."""
        import docgen.orchestrator
        from cli.generate import _resolve_batch_dispatch

        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', False,
        )
        resolved, reason = _resolve_batch_dispatch(
            batch_resolved=True, batch_reason='--batch override',
        )
        assert resolved is False
        assert '--batch override' in reason
        assert 'runtime dispatch not yet implemented' in reason
        assert 'sync prices apply' in reason

    def test_flag_on_preserves_resolved_batch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Paired baseline — when the flag is True (post-#45), the
        helper is a pass-through. The reason isn't decorated; the
        dollar figure reflects the real 50% discount.

        Bites a refactor that always downgrades regardless of the
        flag, AND a refactor that always lets the claim through
        regardless of the flag (the paired flag-off test catches that)."""
        import docgen.orchestrator
        from cli.generate import _resolve_batch_dispatch

        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', True,
        )
        resolved, reason = _resolve_batch_dispatch(
            batch_resolved=True, batch_reason='--batch override',
        )
        assert resolved is True
        assert reason == '--batch override'

    def test_flag_off_no_op_when_already_sync(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``batch_resolved`` is already False (e.g., user
        passed ``--no-batch``), the flag has no effect. Ensures the
        helper isn't over-eager — only downgrades when there's
        actually a claim to downgrade.

        Paired with the flag-off-with-claim test above so a helper
        that sets resolved=False on every call would pass that one
        but fail this one's reason-preserved assertion."""
        import docgen.orchestrator
        from cli.generate import _resolve_batch_dispatch

        monkeypatch.setattr(
            docgen.orchestrator, 'BATCH_DISPATCH_IMPLEMENTED', False,
        )
        resolved, reason = _resolve_batch_dispatch(
            batch_resolved=False, batch_reason='--no-batch override',
        )
        assert resolved is False
        # Reason is untouched — no "runtime dispatch" note appended
        # because there was no claim to correct.
        assert reason == '--no-batch override'

    def test_default_flag_value_is_true_post_45_9(self) -> None:
        """After #45.9 the flag is True — the orchestrator's
        ``_run_batch`` is wired and the dry-run estimator legitimately
        claims the batch discount.

        Bites a future emergency rollback that toggles this back to
        False without re-checking that the streaming-only path still
        exists. The gate is the pinch-point: with the flag False, the
        gate downgrades batch=True to sync, which keeps users on the
        streaming path even though the dispatch code is still there."""
        from docgen.orchestrator import BATCH_DISPATCH_IMPLEMENTED

        assert BATCH_DISPATCH_IMPLEMENTED is True
