"""Contract: ``library_analytics.log_usage`` never inserts NULL into
``usage_events.outcome``.

The schema declares ``outcome TEXT NOT NULL DEFAULT 'call'`` (see
``library.py:152``). ``log_usage`` previously passed ``None`` when
``result_count == 0``, overriding the DEFAULT and raising
``sqlite3.IntegrityError`` on every zero-result tool call. The
correct semantic is ``'call'`` — the tool was called, the outcome
isn't user-confirmed-useful (``'hit'``) and isn't user-confirmed-
miss (``'miss'``) either. ``'call'`` is the neutral
"awaiting-feedback" sentinel.
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture
def library(tmp_path: Path):
    from library import Library
    lib = Library(tmp_path / 'log_usage.db')
    yield lib
    lib.close()


def test_log_usage_with_zero_results_does_not_raise(library):
    """Zero-result search should record a usage event cleanly. Bites
    when log_usage overrides the schema's DEFAULT 'call' with NULL."""
    event_id = library.log_usage(
        tool_name='ariadne_search',
        query='nothing matches',
        result_count=0,
    )
    assert event_id is not None


def test_log_usage_with_zero_results_records_call_outcome(library):
    """The recorded outcome for a zero-result call is 'call' — the
    neutral 'awaiting-feedback' sentinel, matching the schema
    DEFAULT. Not 'miss', because 'miss' is a user-driven label
    written by ariadne_log_miss; conflating them blurs the gap-
    analysis signal."""
    event_id = library.log_usage(
        tool_name='ariadne_search',
        query='nothing matches',
        result_count=0,
    )

    with library._conn_provider.acquire() as conn:
        row = conn.execute(
            'SELECT outcome FROM usage_events WHERE id = ?',
            (event_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == 'call', (
        f"zero-result usage should be 'call' (neutral); got: {row[0]!r}"
    )


def test_log_usage_with_results_still_auto_marks_hit(library):
    """Pre-existing behavior preserved: when results came back,
    auto-mark 'hit' (assumes useful until explicitly logged miss).
    Pairs with the zero-result test so a fix that always-writes-'call'
    fails this half."""
    event_id = library.log_usage(
        tool_name='ariadne_search',
        query='something matched',
        result_count=3,
        document_ids=['a', 'b', 'c'],
    )

    with library._conn_provider.acquire() as conn:
        row = conn.execute(
            'SELECT outcome FROM usage_events WHERE id = ?',
            (event_id,),
        ).fetchone()

    assert row is not None
    assert row[0] == 'hit', (
        f"non-zero result should auto-mark 'hit'; got: {row[0]!r}"
    )
