from __future__ import annotations


class TurnBudgetExceeded(Exception):
    """The bridge cancelled a turn that overran its own hard time budget.

    Distinct from a ``TimeoutError`` raised *inside* a turn: this means WE gave
    up (the turn was still running at ``turn_timeout_seconds``), so an operator
    can tell a too-low budget apart from a genuinely slow dependency.
    """

    def __init__(self, budget_seconds: float):
        self.budget_seconds = budget_seconds
        super().__init__(f'turn exceeded its {budget_seconds:g}s budget')


def to_user_message(exc: BaseException) -> str:
    """Map a turn-time exception to an honest, Slack-friendly message.

    We never mask failures: the source list and the underlying error text are
    surfaced so the user (and the logs) see what actually happened. Teardown
    errors (#890) are handled in the pool and never reach here.
    """
    if isinstance(exc, LookupError):
        # scope_resolution raises this when source= is missing/unresolvable; its
        # message already enumerates the configured sources.
        return (
            f"I'm not sure which service you mean. {exc} "
            'Which one should I look in?'
        )
    if isinstance(exc, TurnBudgetExceeded):
        # OUR hard cap fired (the turn was still running) — name the limit so a
        # too-low budget is visible, and keep it DISTINCT from an inner timeout.
        return (
            f'I ran out of time on this one — my limit is {exc.budget_seconds:g}s. '
            'Try a narrower question, or ask an admin to raise my time budget.'
        )
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError is this on 3.11+
        return 'That took too long — try narrowing the question.'
    return f'Ariadne returned an error: {exc}'
