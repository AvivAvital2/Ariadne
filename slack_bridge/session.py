from __future__ import annotations

import asyncio
from typing import Any


class ThreadSession:
    """One Slack thread's conversation: a warm agent runner + a serialization lock.

    A ``ClaudeSDKClient`` is a single stateful conversation and is not safe for
    concurrent ``query``/``receive``. The lock enforces one in-flight turn per
    thread; messages in the same thread queue, while different threads (their own
    sessions) run concurrently on the event loop. ``last_activity`` is maintained
    by the pool for idle eviction.
    """

    def __init__(self, runner: Any, *, last_activity: float = 0.0):
        self.runner = runner
        self.lock = asyncio.Lock()
        self.last_activity = last_activity

    async def ask(self, text: str) -> Any:
        async with self.lock:
            return await self.runner.ask(text)
