from __future__ import annotations

import pytest

from slack_bridge.retry import retry_transient


async def test_retry_transient_retries_transient_failures_then_succeeds():
    calls = []

    async def flaky():
        calls.append(1)
        if len(calls) < 3:
            raise ConnectionError('blip')
        return 'ok'

    out = await retry_transient(flaky, attempts=3, base_delay=0.0)
    assert out == 'ok'
    assert len(calls) == 3        # retried past the two transient blips


async def test_retry_transient_reraises_last_failure_and_skips_unlisted():
    # Exhausting the attempts re-raises the real failure unwrapped — never masked,
    # so the caller's own handler (and to_user_message) still sees the truth.
    listed = []

    async def always_down():
        listed.append(1)
        raise TimeoutError('still down')

    with pytest.raises(TimeoutError):
        await retry_transient(always_down, attempts=2, base_delay=0.0, on=(TimeoutError,))
    assert len(listed) == 2

    # An exception NOT in `on` is not transient: propagate immediately, no retry.
    unlisted = []

    async def boom():
        unlisted.append(1)
        raise ValueError('not transient')

    with pytest.raises(ValueError):
        await retry_transient(boom, attempts=5, base_delay=0.0, on=(ConnectionError,))
    assert len(unlisted) == 1     # tried once, gave up — did not retry
