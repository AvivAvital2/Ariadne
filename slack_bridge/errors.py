from __future__ import annotations


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
    if isinstance(exc, TimeoutError):  # asyncio.TimeoutError is this on 3.11+
        return 'That took too long — try narrowing the question and I will retry.'
    return f'Ariadne returned an error: {exc}'
