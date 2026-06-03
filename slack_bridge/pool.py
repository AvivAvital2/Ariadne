from __future__ import annotations

import logging
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

from slack_bridge.session import ThreadSession

_logger = logging.getLogger(__name__)


class SessionPool:
    """Bounded LRU pool of warm per-thread sessions.

    Keyed by Slack ``thread_ts``. A hit reuses the warm ``ClaudeSDKClient`` (and
    its warm Ariadne MCP subprocess); a miss builds a fresh one via
    ``runner_factory(thread_ts, seed)``, evicting the least-recently-used session
    first when at capacity. ``clock`` is injectable for testing.
    """

    def __init__(
        self,
        *,
        runner_factory: Callable[[str, Any], Any],
        max_size: int,
        idle_ttl: float,
        clock: Callable[[], float],
    ):
        self._factory = runner_factory
        self._max_size = max_size
        self._idle_ttl = idle_ttl
        self._clock = clock
        self._sessions: OrderedDict[str, ThreadSession] = OrderedDict()

    def __contains__(self, thread_ts: str) -> bool:
        return thread_ts in self._sessions

    def __len__(self) -> int:
        return len(self._sessions)

    async def get_or_create(self, thread_ts: str, *, seed: Any = None) -> ThreadSession:
        existing = self._sessions.get(thread_ts)
        if existing is not None:
            self._sessions.move_to_end(thread_ts)
            existing.last_activity = self._clock()
            return existing

        while len(self._sessions) >= self._max_size:
            victim_ts = next(
                (ts for ts, session in self._sessions.items() if not session.lock.locked()),
                None,
            )
            if victim_ts is None:
                # Every session has an in-flight turn — tolerate transient
                # over-capacity rather than tear down a live (locked) turn.
                break
            await self._safe_teardown(self._sessions.pop(victim_ts))

        runner = self._factory(thread_ts, seed)
        session = ThreadSession(runner, last_activity=self._clock())
        self._sessions[thread_ts] = session
        return session

    async def evict_idle(self) -> None:
        """Tear down sessions idle past the TTL, but never one mid-turn.

        Run periodically by the app's background task. Skips sessions whose lock
        is held so an in-flight answer isn't torn down underneath itself.
        """
        now = self._clock()
        stale = [
            ts
            for ts, session in self._sessions.items()
            if (now - session.last_activity) > self._idle_ttl and not session.lock.locked()
        ]
        for ts in stale:
            await self._safe_teardown(self._sessions.pop(ts))

    async def _safe_teardown(self, session: ThreadSession) -> None:
        # SDK bug #890: ClaudeSDKClient/MCP teardown can surface CancelledError or
        # a BaseExceptionGroup from the transport's __aexit__. Swallow it (except
        # genuine interrupts) so eviction never propagates a teardown failure.
        try:
            await session.runner.aclose()
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException:  # noqa: BLE001 — see #890 note above
            _logger.debug('ignored error during session teardown', exc_info=True)
