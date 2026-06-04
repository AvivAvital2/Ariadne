"""Exit-code policy for ``cmd_generate`` (``_generate_exit_code``).

A partial failure (some docs generated, some failed) is a *warning*, not
an error — so an ``onboard`` pipeline keeps going to its later phases
instead of halting on the first validation hiccup (which batch mode does
not retry). rc=1 is reserved for a HARD failure: the run aborted, or
every attempted doc failed.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from cli.generate import _generate_exit_code


def _result(*, created, failed, aborted=False):
    return SimpleNamespace(
        docs_created=created, docs_failed=failed, aborted=aborted,
    )


def test_full_success_is_zero():
    assert _generate_exit_code(_result(created=81, failed=0)) == 0


def test_nothing_to_do_is_zero():
    assert _generate_exit_code(_result(created=0, failed=0)) == 0


def test_partial_failure_is_zero():
    # 80 of 81 stored, 1 failed validation → success-with-warning, so an
    # onboard pipeline continues to themes rather than halting.
    assert _generate_exit_code(_result(created=80, failed=1)) == 0


def test_total_failure_is_one():
    # Nothing succeeded — that's a hard failure.
    assert _generate_exit_code(_result(created=0, failed=81)) == 1


def test_aborted_is_one():
    # Aborted (quota / fetch failure / declined) must be non-zero so the
    # pipeline halts and the user resumes — even though docs_failed is 0.
    assert _generate_exit_code(_result(created=0, failed=0, aborted=True)) == 1


# ---------------------------------------------------------------------------
# _progress_heartbeat: keeps the bar refreshing during long async waits
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_progress_heartbeat_ticks_during_wait():
    """While the wrapped phase awaits, the heartbeat must refresh the
    progress display repeatedly (so the elapsed timer/spinner move),
    then stop once the context exits.
    """
    from cli.generate import _progress_heartbeat

    progress = Mock()
    async with _progress_heartbeat(progress, interval=0.01):
        await asyncio.sleep(0.06)
    ticks = progress.refresh.call_count
    assert ticks >= 2, ticks

    # Stops after exit — no further refreshes.
    await asyncio.sleep(0.03)
    assert progress.refresh.call_count == ticks
