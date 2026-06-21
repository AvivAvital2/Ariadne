"""Shared rich progress bar for long-running, countable Ariadne tasks.

One definition of the determinate "spinner · bar · M/N · elapsed · eta" layout,
reused by catalog-sync, catalog-describe, generate, themes build, and embedding
rebuild — so they look and behave identically and every countable task gets a
live remaining-time estimate. Before this, each command hand-rolled the same
column tuple; keep new countable tasks on ``make_progress`` rather than
re-listing columns.
"""
from __future__ import annotations

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def progress_columns() -> tuple:
    """The canonical columns for a countable task: spinner, description, bar,
    done/total, elapsed, and a live ETA."""
    return (
        SpinnerColumn(),
        TextColumn('[bold cyan]{task.description}'),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn('·'),
        TimeElapsedColumn(),
        TextColumn('eta'),
        TimeRemainingColumn(),
    )


def make_progress(
    console: Console | None = None, *, transient: bool = False,
) -> Progress:
    """Build a Progress with the shared columns.

    Pass the caller's ``console`` so output is routed consistently;
    ``transient=True`` clears the bar once the block exits.
    """
    return Progress(*progress_columns(), console=console, transient=transient)


def format_duration(seconds: float) -> str:
    """Compact, human duration: ``45s``, ``1m 30s``, ``1h 02m``.

    Used for the up-front time estimate that sits beside the price estimate.
    """
    seconds = int(round(seconds))
    if seconds < 60:
        return f'{seconds}s'
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f'{minutes}m {secs:02d}s'
    hours, minutes = divmod(minutes, 60)
    return f'{hours}h {minutes:02d}m'
