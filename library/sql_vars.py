"""Shared SQLite bind-variable budgeting for ``WHERE ... IN (...)`` queries.

SQLite caps the number of bind variables per statement
(``SQLITE_MAX_VARIABLE_NUMBER`` — 999 on older builds, 32766 on newer). Any
query that binds a caller-sized id list must chunk it or risk
``sqlite3.OperationalError: too many SQL variables`` on a large corpus (e.g. a
Databricks spool's tens of thousands of scoped ids). This is the single
chunking helper every by-id read / delete / copy shares, so no call site has to
re-derive the budget (and none can drift from it).
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence

# Safely under SQLITE_MAX_VARIABLE_NUMBER on every supported build (999 on the
# oldest). Read at CALL time (tests lower it to force chunking cheaply).
SQL_MAX_VARS = 900


def chunk_ids(
    ids: Sequence, *, copies: int = 1, reserved: int = 0,
) -> Iterator[list]:
    """Yield slices of ``ids`` small enough that one statement binding
    ``copies`` references to each id — plus ``reserved`` other bind params
    outside the ``IN`` clause — stays under ``SQL_MAX_VARS``.

    Use ``copies=2`` for a statement that binds the chunk twice, e.g.
    ``WHERE a IN (?) OR b IN (?)``; ``reserved`` covers fixed params like a
    closure filter. The budget never drops below 1, so progress is guaranteed
    even with a wide ``reserved``. Empty ``ids`` yields nothing.
    """
    budget = max(1, (SQL_MAX_VARS - reserved) // max(1, copies))
    for start in range(0, len(ids), budget):
        yield list(ids[start:start + budget])
