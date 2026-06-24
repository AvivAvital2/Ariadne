"""``iter_with_progress`` is the DRY core of the persist-phase progress bars:
it yields items untouched while reporting ``(label, completed, total)`` so a
renderer can drive a bar + ETA. It must advance 0→total and land at 100%, and
be a no-op when no reporter is given.
"""
from __future__ import annotations

from progress_util import iter_with_progress


def test_reports_progress_and_yields_items_unchanged() -> None:
    calls: list[tuple[str, int, int]] = []
    out = list(iter_with_progress(
        ['a', 'b', 'c'], lambda label, done, total: calls.append((label, done, total)),
        'src1',
    ))
    # items pass through unchanged
    assert out == ['a', 'b', 'c']
    # advances 0..n before each item, then lands exactly at total/total
    assert calls == [
        ('src1', 0, 3), ('src1', 1, 3), ('src1', 2, 3), ('src1', 3, 3),
    ]


def test_none_report_is_a_noop() -> None:
    # no reporter → behaves exactly like a plain loop, no error
    assert list(iter_with_progress([1, 2], None, 'x')) == [1, 2]
