"""Slice 1 of designs/fstring-sql-and-dataset-schema-witness.md — capture
f-string SQL so the raw-SQL binder can see it.

Evolving end-to-end test (grows in later slices: constant-folding, the dataset
schema witness, derived->resolved promotion). Here: a Python source with an
f-string SQL query (value interpolation, real table named literally), a plain
string literal, and a NON-SQL f-string. After ingest the SQL f-string reaches
``string_literals`` as ``kind='fstring'`` (the plain literal stays
``kind='plain'``; the non-SQL f-string is gated out), and the raw-SQL binder
recovers its real columns as ``derived`` ``data_access`` — with the interpolation
placeholder never emitted as a column/table.

Supporting tests cover the two other slice-1 criteria: the no-regenerate
migration adds ``kind`` to an existing DB, and reconstructed f-string rows stay
invisible to position-based literal lookups (route/HTTP/resolution isolation).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_string_literal_extractor import (
    ingest_string_literals,
    lookup_literal_at_position,
)
from docgen.sql_access import persist_data_access_rawsql
from library import Library

SRC = '''\
PRIMARY_TABLE = 'primary_table'


def build_query(uid):
    log = "starting build"
    msg = f"built for {uid}"
    feat = f"SELECT age FROM {PRIMARY_TABLE} WHERE id = {uid}"
    return f"SELECT email FROM users WHERE id = {uid}"
'''
FUNC = 'scip-python python svc . svc/build_query().'


def _index(root: Path) -> ScipIndex:
    (root / 'q.py').write_text(SRC)
    # Owning is resolved from the index itself: build_query's def occurrence
    # carries a single-line name-token ``range`` (0-indexed line 3) plus the
    # body span in ``enclosing_range`` — exactly what scip-python emits. No
    # scip_symbols row, no kind, no absolute path: the realistic shape.
    doc = _ScipDoc(
        relative_path='q.py',
        occurrences=(
            _ScipOccurrence(symbol=FUNC, range=(3, 4, 15), is_definition=True,
                            enclosing_range=(3, 0, 7, 0)),
        ),
        symbols=())
    return ScipIndex(documents=(doc,), source_root=root)


def test_fstring_sql_is_captured_and_bound(tmp_path):
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        ingest_string_literals(
            source_name='svc', source_root=tmp_path, conn=conn,
            index_factory=lambda *a, **k: _index(tmp_path))
        conn.commit()

        # CAPTURE + SQL-shape gate: the two SQL f-strings are stored as 'fstring'
        # (the non-SQL f-string is gated out); the plain literal stays 'plain'.
        kinds = dict(conn.execute(
            'SELECT kind, COUNT(*) FROM string_literals WHERE source_name = ? '
            'GROUP BY kind', ('svc',)))
        assert kinds.get('fstring', 0) == 2
        assert kinds.get('plain', 0) >= 1

        # BIND: the binder recovers the real columns of the f-string query.
        persist_data_access_rawsql(conn, 'svc', dialect='duckdb')
        conn.commit()
        access = {(r[0], r[1]) for r in conn.execute(
            "SELECT schema_symbol_id, role FROM data_access WHERE witness = 'rawsql'")}
        cols = {r[0] for r in conn.execute(
            "SELECT canonical_id FROM schema_symbols WHERE source_name = 'svc'")}
    lib.close()

    assert ('data sql svc _._.users#id', 'filter') in access
    assert ('data sql svc _._.users#email', 'project') in access
    # constant-folded: {PRIMARY_TABLE} -> 'primary_table' (a real table node, not
    # the placeholder), so the recipe-style query's column binds to it.
    assert ('data sql svc _._.primary_table#age', 'project') in access
    # the interpolation placeholder is never emitted as a column/table.
    assert not any('__ariadne_ph__' in c for c in cols)


EMBEDDED_SRC = '''\
def make_prompt():
    return """Produce one query.

```sql
SELECT name FROM users WHERE id = 1
```

```sql
```

Return {"label": "x", "sql_query": "SELECT email FROM accounts WHERE active = true"}
"""
'''
PROMPT_FUNC = 'scip-python python svc . svc/make_prompt().'


def _embedded_index(root: Path) -> ScipIndex:
    (root / 'p.py').write_text(EMBEDDED_SRC)
    doc = _ScipDoc(
        relative_path='p.py',
        occurrences=(
            _ScipOccurrence(symbol=PROMPT_FUNC, range=(0, 4, 15), is_definition=True,
                            enclosing_range=(0, 0, 11, 0)),
        ),
        symbols=())
    return ScipIndex(documents=(doc,), source_root=root)


def test_embedded_sql_in_strings_is_captured_and_bound(tmp_path):
    """SQL embedded inside a larger string literal (a prompt/docstring/JSON
    fixture) — markdown ```sql fenced blocks and SQL-shaped JSON values like
    ``"sql_query": "SELECT ..."`` — is extracted as first-class SQL (kind=
    'embedded') and bound. The whole literal isn't SQL (it starts with prose),
    so the standalone extractors miss it; this is the general embedded-SQL path.
    """
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        ingest_string_literals(
            source_name='svc', source_root=tmp_path, conn=conn,
            index_factory=lambda *a, **k: _embedded_index(tmp_path))
        conn.commit()

        # the fenced block + the JSON sql_query value are captured as 'embedded'
        # (the empty fence and the non-SQL "label" value are skipped).
        n_embedded = conn.execute(
            "SELECT COUNT(*) FROM string_literals WHERE source_name = 'svc' "
            "AND kind = 'embedded'").fetchone()[0]
        assert n_embedded == 2

        persist_data_access_rawsql(conn, 'svc', dialect='duckdb')
        conn.commit()
        access = {(r[0], r[1]) for r in conn.execute(
            "SELECT schema_symbol_id, role FROM data_access WHERE witness = 'rawsql'")}
    lib.close()

    # fenced block bound:
    assert ('data sql svc _._.users#name', 'project') in access
    assert ('data sql svc _._.users#id', 'filter') in access
    # JSON sql_query value bound:
    assert ('data sql svc _._.accounts#email', 'project') in access
    assert ('data sql svc _._.accounts#active', 'filter') in access


def test_sql_with_no_owning_symbol_is_surfaced_as_gap(tmp_path):
    """A SQL literal with no enclosing owner (module-level) is RECORDED as a gap
    by the binder, never silently dropped — the don't-mask-errors rule. Without
    a consumer symbol it can't be bound, but it must remain visible to
    coverage/diagnose."""
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT INTO string_literals (source_name, file, line_start, '
            'col_start, value, owning_symbol_id, kind) VALUES (?,?,?,?,?,?,?)',
            ('svc', 'm.py', 1, 0, 'SELECT email FROM users', None, 'plain'))
        conn.commit()
        result = persist_data_access_rawsql(conn, 'svc', dialect='duckdb')
        bound = conn.execute(
            "SELECT COUNT(*) FROM data_access WHERE source_name = 'svc'").fetchone()[0]
    lib.close()
    assert bound == 0  # no owner → nothing to attribute the access to
    assert 'SELECT email FROM users' in result.gaps  # but surfaced, not dropped


def test_kind_column_is_added_by_migration_no_regenerate(tmp_path):
    """An existing DB with the old string_literals schema gains ``kind`` on the
    next open (the idempotent ALTER-TABLE migration), and existing rows backfill
    to 'plain' — so coverage lands without a destructive rebuild."""
    db = tmp_path / 'old.db'
    conn = sqlite3.connect(db)
    conn.execute(
        'CREATE TABLE string_literals ('
        'id INTEGER PRIMARY KEY AUTOINCREMENT, source_name TEXT NOT NULL, '
        'file TEXT NOT NULL, line_start INTEGER NOT NULL, col_start INTEGER NOT NULL, '
        'value TEXT NOT NULL, owning_symbol_id TEXT)')
    conn.execute(
        "INSERT INTO string_literals (source_name, file, line_start, col_start, value) "
        "VALUES ('s', 'f.py', 1, 0, 'SELECT 1')")
    conn.commit()
    conn.close()

    lib = Library(db)  # opening runs the migration
    with lib._conn_provider.acquire() as conn:
        cols = {r[1] for r in conn.execute('PRAGMA table_info(string_literals)')}
        backfilled = conn.execute(
            "SELECT kind FROM string_literals WHERE value = 'SELECT 1'").fetchone()[0]
    lib.close()

    assert 'kind' in cols
    assert backfilled == 'plain'


def test_fstring_rows_are_invisible_to_position_lookups(tmp_path):
    """``lookup_literal_at_position`` (the route/HTTP/resolution API) returns
    plain literals only — a reconstructed f-string row at a position resolves to
    None, so the new rows can't leak into route extraction."""
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        conn.executemany(
            'INSERT INTO string_literals (source_name, file, line_start, '
            'col_start, value, kind) VALUES (?,?,?,?,?,?)',
            [('s', 'f.py', 1, 0, 'plainval', 'plain'),
             ('s', 'f.py', 2, 0, 'SELECT 1', 'fstring')])
        conn.commit()
        plain = lookup_literal_at_position(
            conn, source_name='s', file='f.py', line=1, col=0)
        fstr = lookup_literal_at_position(
            conn, source_name='s', file='f.py', line=2, col=0)
    lib.close()

    assert plain == 'plainval'   # plain literals still resolve
    assert fstr is None          # f-string rows are invisible to position lookups


def test_plain_js_and_scala_literals_keep_kind_plain(tmp_path):
    """The kind addition doesn't disturb non-Python extraction: JS/Scala string
    literals are still captured, tagged kind='plain'."""
    (tmp_path / 'a.js').write_text('const url = "https://x/y";\n')
    (tmp_path / 'b.scala').write_text('object O { val q = "hello scala" }\n')
    index = ScipIndex(documents=(
        _ScipDoc(relative_path='a.js', occurrences=(), symbols=()),
        _ScipDoc(relative_path='b.scala', occurrences=(), symbols=())),
        source_root=tmp_path)
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        ingest_string_literals(
            source_name='poly', source_root=tmp_path, conn=conn,
            index_factory=lambda *a, **k: index)
        conn.commit()
        rows = {(r[0], r[1]) for r in conn.execute(
            "SELECT value, kind FROM string_literals WHERE source_name = 'poly'")}
    lib.close()
    assert ('https://x/y', 'plain') in rows
    assert ('hello scala', 'plain') in rows


def test_missing_index_returns_zero(tmp_path):
    """No ``.ariadne/index.scip`` → ingest is a clean no-op (returns 0)."""
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        n = ingest_string_literals(
            source_name='s', source_root=tmp_path, conn=conn)
    lib.close()
    assert n == 0


def test_corrupt_index_is_skipped_cleanly(tmp_path):
    """A corrupt ``index.scip`` is skipped (returns 0), never a crash."""
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'index.scip').write_bytes(b'not a real scip index')
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        n = ingest_string_literals(
            source_name='s', source_root=tmp_path, conn=conn)
    lib.close()
    assert n == 0


def test_unreadable_file_is_skipped(tmp_path):
    """A doc whose file is missing on disk is skipped (OSError), not fatal."""
    index = ScipIndex(documents=(
        _ScipDoc(relative_path='ghost.py', occurrences=(), symbols=()),),
        source_root=tmp_path)
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        n = ingest_string_literals(
            source_name='s', source_root=tmp_path, conn=conn,
            index_factory=lambda *a, **k: index)
    lib.close()
    assert n == 0
