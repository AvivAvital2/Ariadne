"""`query_string_literals_by_value` — the Tier 1 config↔code join primitive.

A config key that appears verbatim as a code string literal is (approximately)
a read site for that key. This query is the code-side half of the join.

Fixtures are synthetic and repo/language-agnostic on purpose: the query must
behave correctly for any key string in any source, regardless of which projects,
languages, files, or keys happen to exist in a real database. See
designs/config-code-bridge/tier1-string-join.md (Feature 1).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the SCIP schema applied."""
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _insert(conn, source, rows):
    """rows: list of (file, line_start, col_start, value, owning_symbol_id)."""
    from docgen.scip_string_literal_index import StringLiteral, persist_string_literals

    persist_string_literals(
        source_name=source,
        literals=[
            StringLiteral(file=Path(f), line_start=ls, col_start=cs, value=v, owning_symbol_id=o)
            for (f, ls, cs, v, o) in rows
        ],
        conn=conn,
    )


def test_returns_exact_matches_source_scoped_and_ordered(conn: sqlite3.Connection) -> None:
    from docgen.scip_string_literal_index import query_string_literals_by_value

    _insert(conn, 'src1', [
        ('reader_b', 99, 4, 'svc.timeout', None),       # same value, later file
        ('reader_a', 53, 10, 'svc.timeout', None),      # same value, earlier file
        ('reader_a', 60, 2, 'svc.timeoutMillis', None), # near-miss → excluded (exact match only)
        ('reader_a', 70, 2, 'unrelated.key', None),     # unrelated → excluded
    ])
    _insert(conn, 'src2', [
        ('reader_z', 1, 0, 'svc.timeout', None),        # same value, other source → excluded
    ])
    out = query_string_literals_by_value(source_name='src1', value='svc.timeout', conn=conn)
    # exact-match only, source-scoped to src1, ordered by (file, line)
    assert [(str(r.file), r.line_start) for r in out] == [('reader_a', 53), ('reader_b', 99)]


def test_no_match_returns_empty(conn: sqlite3.Connection) -> None:
    from docgen.scip_string_literal_index import query_string_literals_by_value

    _insert(conn, 'src1', [('reader_a', 1, 0, 'some.key', None)])
    assert query_string_literals_by_value(
        source_name='src1', value='absent.key', conn=conn,
    ) == []
