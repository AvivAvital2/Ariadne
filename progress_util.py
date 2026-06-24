"""Wrap an iterable so a progress callback fires as items are consumed.

The DRY core for the persist-phase progress bars: each heavy persist step wraps
the per-item loop it already has (files, sources) with :func:`iter_with_progress`,
and a single renderer in ``cli/index.py`` turns the ``(label, completed, total)``
reports into one rich bar with a live ETA. The step's logic is otherwise
untouched — this only observes how far the loop has advanced.

Repo-root leaf (stdlib only) so both ``docgen.*`` extractors and the persist
pipeline can import it without dragging in a package ``__init__``.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from typing import TypeVar

T = TypeVar('T')

ProgressReport = Callable[[str, int, int], None]


def iter_with_progress(
    items: Iterable[T],
    report: ProgressReport | None,
    label: str,
) -> Iterator[T]:
    """Yield each of ``items`` unchanged. If ``report`` is given, call it with
    ``(label, completed, total)`` before yielding each item (``completed`` =
    items already yielded) and once more with ``(label, total, total)`` after
    the last — so a bar advances 0→total and lands exactly at 100%.

    ``report=None`` is a no-op, so callers that don't render pay nothing and
    behave identically to a plain ``for`` loop.
    """
    seq = items if hasattr(items, '__len__') else list(items)
    total = len(seq)
    completed = 0
    for item in seq:
        if report is not None:
            report(label, completed, total)
        completed += 1
        yield item
    if report is not None:
        report(label, total, total)


def _update_persist_task(progress, state, phase_label, label, completed, total):
    """Drive the single persist-progress task for one ``(label, completed,
    total)`` report. A changed ``label`` (new source) resets the task so its ETA
    recomputes; an unchanged label just advances the bar. An empty ``label``
    renders the phase alone (no source suffix)."""
    if state.get('task_id') is None:
        state['task_id'] = progress.add_task('', total=total, detail='')
        state['label'] = label
    elif label != state['label']:
        progress.reset(state['task_id'], total=total)
        state['label'] = label
    description = f'  {phase_label}: {label}' if label else f'  {phase_label}'
    progress.update(
        state['task_id'], completed=completed, total=total,
        description=description, detail=f'{completed}/{total}',
    )
