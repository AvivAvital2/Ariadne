from __future__ import annotations

import asyncio

from slack_bridge.session import ThreadSession


class _RecordingRunner:
    """Fake AgentRunner that records whether two asks ever overlap."""

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def ask(self, text, images=()):  # noqa: ARG002 — accepts the new kwarg
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        # Yield control twice so a non-serialized impl would interleave here.
        await asyncio.sleep(0)
        await asyncio.sleep(0)
        self.active -= 1
        return f'answer:{text}'


async def test_thread_session_serializes_concurrent_asks():
    runner = _RecordingRunner()
    session = ThreadSession(runner)

    results = await asyncio.gather(session.ask('a'), session.ask('b'))

    # The per-thread lock means only one ask runs at a time.
    assert runner.max_active == 1
    assert set(results) == {'answer:a', 'answer:b'}


class _CapturingRunner:
    def __init__(self):
        self.calls = []

    async def ask(self, text, images=()):
        self.calls.append((text, list(images)))
        return f'answer:{text}'


async def test_thread_session_forwards_images_to_runner():
    runner = _CapturingRunner()
    session = ThreadSession(runner)

    await session.ask('what is this?', images=['IMG'])

    assert runner.calls == [('what is this?', ['IMG'])]
