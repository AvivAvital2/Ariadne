from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import TypeVar

_logger = logging.getLogger(__name__)

T = TypeVar('T')

# Transient failures worth retrying. NOT a catch-all: a whole-turn timeout is
# deliberately absent — a turn is stateful + non-idempotent (see retry_transient).
DEFAULT_TRANSIENT: tuple[type[BaseException], ...] = (TimeoutError, ConnectionError)


async def retry_transient(
    factory: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    on: tuple[type[BaseException], ...] = DEFAULT_TRANSIENT,
) -> T:
    """Retry a *transient*, *idempotent* async op with exponential backoff.

    ``factory`` must build a FRESH awaitable per call (a coroutine is single-use).
    Only exceptions in ``on`` are retried; anything else propagates at once. The
    final attempt's exception is re-raised unwrapped, so the caller's own error
    handling (and :func:`slack_bridge.errors.to_user_message`) still sees the real
    failure — we never mask it.

    Deliberately NARROW. Wrap one transient call — a Slack web-API request, an
    embedding fetch, a cold MCP connect — NOT a whole agent turn. A turn runs on a
    stateful ``ClaudeSDKClient`` behind a lock; retrying it would re-issue a query
    into a half-finished conversation. Whole-turn give-ups stay fail-fast and
    surface :class:`slack_bridge.errors.TurnBudgetExceeded` instead.
    """
    for attempt in range(1, attempts + 1):
        try:
            return await factory()
        except on as exc:
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            _logger.warning(
                'transient %s on attempt %d/%d; retrying in %.2fs',
                type(exc).__name__, attempt, attempts, delay,
            )
            await asyncio.sleep(delay)
    raise AssertionError('unreachable: the loop returns or raises')
