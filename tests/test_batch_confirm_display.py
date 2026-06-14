"""The batch SLA confirmation prompt must be readable under the progress bar.

When ``ariadne generate`` resolves to batch dispatch, the orchestrator fires
``confirm_callback`` to get the user's acceptance of the up-to-24h SLA. That
call happens *inside* the Rich ``Progress`` live display, whose auto-refresh
thread — plus the CLI's 0.5s ``_progress_heartbeat`` — repaints the terminal
several times a second. A bare ``input()`` under that live display gets
clobbered: the ``[y/N]:`` suffix is overwritten and the run looks stuck on a
prompt the user can't see.

The fix suspends the display around the prompt. ``_progress_heartbeat`` yields a
``pause()`` context manager that stops the progress bar (halting Rich's
auto-refresh) and silences the heartbeat for its duration; ``_pausing_confirm``
wraps the stdin confirm so the display is paused for exactly the prompt and
resumes after the user answers.
"""
from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from cli.generate import _pausing_confirm, _progress_heartbeat


@pytest.mark.asyncio
async def test_pause_stops_and_restarts_the_progress_bar() -> None:
    """``pause()`` stops the live display on enter and restarts it on exit —
    that ``stop()`` is what frees the terminal for ``input()``."""
    progress = MagicMock()

    async with _progress_heartbeat(progress, interval=0.01) as pause:
        assert not progress.stop.called          # running before we pause
        with pause():
            assert progress.stop.called          # display halted for the prompt
            assert not progress.start.called
        assert progress.start.called             # resumed once the prompt is done


@pytest.mark.asyncio
async def test_heartbeat_is_silent_while_paused() -> None:
    """The 0.5s heartbeat must not repaint while the prompt is up — that
    repaint is exactly what clobbered the prompt. A pause spanning several
    tick intervals must see zero ``refresh()`` calls."""
    progress = MagicMock()

    async with _progress_heartbeat(progress, interval=0.01) as pause:
        await asyncio.sleep(0.03)
        assert progress.refresh.call_count > 0   # ticking while active

        with pause():
            progress.refresh.reset_mock()
            await asyncio.sleep(0.05)             # spans ~5 tick intervals
            paused_refreshes = progress.refresh.call_count

    assert paused_refreshes == 0


@pytest.mark.asyncio
async def test_pausing_confirm_brackets_the_prompt_and_returns_answer() -> None:
    """``_pausing_confirm`` pauses the display *around* the awaited base
    confirm: stop → prompt → start, with the user's answer passed through."""
    progress = MagicMock()
    timeline: list[str] = []
    progress.stop.side_effect = lambda: timeline.append('stop')
    progress.start.side_effect = lambda: timeline.append('start')

    async with _progress_heartbeat(progress, interval=0.01) as pause:
        async def base_confirm(msg: str) -> bool:
            timeline.append('prompt')            # the input() read happens here
            return True

        confirm = _pausing_confirm(base_confirm, pause)
        accepted = await confirm('Continue?')

    assert accepted is True
    assert timeline == ['stop', 'prompt', 'start']
