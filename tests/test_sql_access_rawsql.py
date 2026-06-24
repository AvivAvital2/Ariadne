"""Phase 0 raw-SQL extractor (design §5.7, §5.8, §10): sqlglot over string
literals -> role-typed ``data_access`` rows.

A SELECT yields ``project`` (the SELECT list) + ``filter`` (WHERE); an
UPDATE yields ``write`` (SET) + ``filter`` (WHERE). Value interpolations
(f-string ``{…}``, printf ``%s``) are normalized to placeholders before
parsing (§5.8) so they never become phantom columns. A SQL-shaped literal
that yields no access (unparseable / un-bindable) is recorded as a gap,
never silently dropped (§5.8); ordinary non-SQL strings are not counted.
Idempotent per source+witness. With no declared schema to corroborate
against, rows are ``derived`` (§5.7). Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.sql_access import persist_data_access_rawsql
from library.scip import init_scip_schema

READER = 'scip-python python src1 . src1/get_user().'
WRITER = 'scip-python python src1 . src1/clear_email().'


def _literal(conn, value, owning, file='src1/q.py', line=3):
    conn.execute(
        'INSERT INTO string_literals (source_name, file, line_start, '
        'col_start, value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
        ('src1', file, line, 0, value, owning),
    )


def _rawsql_rows(conn):
    return {
        (r[0], r[1], r[2], r[3])
        for r in conn.execute(
            'SELECT consumer_symbol_id, schema_symbol_id, role, confidence '
            "FROM data_access WHERE witness = 'rawsql'"
        )
    }


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def test_rawsql_emits_role_typed_access_and_is_idempotent(conn):
    # --- given: a read query, a write query, and three that must be
    #            skipped (unparseable / un-bindable / no call site) ------
    _literal(conn, 'SELECT email FROM users WHERE id = ?', READER)
    _literal(conn, 'UPDATE users SET email = ? WHERE id = ?', WRITER)
    _literal(conn, 'not sql at all just prose', READER)     # unparseable
    _literal(conn, 'SELECT x', READER)                       # no table to bind
    _literal(conn, 'SELECT email FROM users', None)          # no owning symbol

    result = persist_data_access_rawsql(conn, 'src1')

    # --- then: only the bound reads/writes are recorded, role-typed and
    #           derived (no schema yet to lift them to resolved) ---------
    expected = {
        (READER, 'data sql src1 _._.users#email', 'project', 'derived'),
        (READER, 'data sql src1 _._.users#id', 'filter', 'derived'),
        (WRITER, 'data sql src1 _._.users#email', 'write', 'derived'),
        (WRITER, 'data sql src1 _._.users#id', 'filter', 'derived'),
    }
    assert _rawsql_rows(conn) == expected
    assert result.rows_written == 4

    # --- and: idempotent — re-running replaces this source's rawsql rows,
    #          it does not duplicate them ------------------------------
    persist_data_access_rawsql(conn, 'src1')
    assert _rawsql_rows(conn) == expected


def test_placeholder_normalization_recovers_interpolated_sql(conn):
    # f-string interpolation ({uid}) and printf %-specifiers (%s, %(key)s)
    # must normalize to placeholders BEFORE parsing (§5.8) — recovering the
    # real columns, and never fabricating one from the interpolation expr.
    _literal(conn, 'SELECT email FROM users WHERE id = {uid}', READER)
    _literal(conn, 'UPDATE users SET email = %s WHERE id = %(key)s', WRITER)

    persist_data_access_rawsql(conn, 'src1')

    assert _rawsql_rows(conn) == {
        (READER, 'data sql src1 _._.users#email', 'project', 'derived'),
        (READER, 'data sql src1 _._.users#id', 'filter', 'derived'),
        (WRITER, 'data sql src1 _._.users#email', 'write', 'derived'),
        (WRITER, 'data sql src1 _._.users#id', 'filter', 'derived'),
    }
    # critically: the interpolation expressions ({uid}, %s, %(key)s) are NOT
    # mistaken for columns — no users#uid / users#key node is fabricated.
    phantom = {
        r[0] for r in conn.execute(
            "SELECT canonical_id FROM schema_symbols "
            "WHERE canonical_id LIKE '%#uid' OR canonical_id LIKE '%#key'"
        )
    }
    assert phantom == set()


def test_unrecovered_sql_is_recorded_as_a_gap_not_silently_dropped(conn):
    # --- given: a recoverable query, a SQL-shaped literal that won't parse
    #            (the dynamically-built SQL §5.8 defers), and a non-SQL
    #            string (a log line) -----------------------------------
    _literal(conn, 'SELECT email FROM users WHERE id = ?', READER)
    _literal(conn, 'SELECT FROM WHERE', WRITER)               # SQL-shaped, unparseable
    _literal(conn, 'just a log message, not a query', READER)  # not SQL

    result = persist_data_access_rawsql(conn, 'src1')

    # the recoverable query still produces its rows
    assert result.rows_written == 2
    # the SQL-shaped literal we could NOT recover is recorded (§5.8: "never
    # silently dropped"); the non-SQL string is not counted as a SQL gap
    assert set(result.gaps) == {'SELECT FROM WHERE'}


def test_select_star_expands_to_per_column_reads():
    """SELECT * is a whole-row read → expand to a read on every KNOWN column of
    the table (§10 Phase 2; fixes the read-recall hole). Columns come from the
    schema (ORM/DDL) already persisted; raw-SQL stays derived."""
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    # known schema columns, as the ORM/DDL pass would have produced earlier
    for col in ('email_addr', 'name'):
        conn.execute(
            'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
            'table_name, column_name, resolution_source, confidence) '
            'VALUES (?,?,?,?,?,?,?)',
            (f'data sql app _._.users#{col}', 'app', 'column', 'users', col,
             'orm:django', 'exact'))
    conn.execute(
        'INSERT INTO string_literals (source_name, file, line_start, col_start, '
        'value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
        ('app', 'q.py', 1, 0, 'SELECT * FROM users',
         'scip-python python app . app/q().'))
    conn.commit()
    persist_data_access_rawsql(conn, 'app')
    reads = {r[0] for r in conn.execute(
        "SELECT schema_symbol_id FROM data_access "
        "WHERE source_name = 'app' AND role = 'project'")}
    conn.close()
    assert reads == {'data sql app _._.users#email_addr', 'data sql app _._.users#name'}


def test_select_star_edge_cases_are_recorded_as_gaps():
    """SELECT * with no known columns (can't expand) or an ambiguous multi-table
    FROM (can't bind) is a recorded gap, never silently dropped (§5.8)."""
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    owner = 'scip-python python app . app/q().'
    for v in ('SELECT * FROM ghost', 'SELECT * FROM a, b'):
        conn.execute(
            'INSERT INTO string_literals (source_name, file, line_start, col_start, '
            'value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
            ('app', 'q.py', 1, 0, v, owner))
    conn.commit()
    result = persist_data_access_rawsql(conn, 'app')
    rows = list(conn.execute("SELECT 1 FROM data_access WHERE source_name = 'app'"))
    conn.close()
    assert rows == []  # nothing bound/expanded
    assert 'SELECT * FROM ghost' in result.gaps   # columns unknown
    assert 'SELECT * FROM a, b' in result.gaps     # ambiguous table


def test_rawsql_completes_clause_role_set_and_resolves_aliases(conn):
    """The full §5.7 clause->role set + alias resolution: INSERT column list ->
    write, ORDER BY -> order, RETURNING -> project (even inside an UPDATE),
    UPDATE SET -> write, WHERE -> filter, CREATE/ALTER -> ddl; and a ``users AS u``
    alias resolves the ``u.col`` qualifier to the real table 'users', not 'u'."""
    INSERTER = 'scip-python python src1 . src1/add_user().'
    SORTER = 'scip-python python src1 . src1/recent().'
    UPDATER = 'scip-python python src1 . src1/rename().'
    ALTERER = 'scip-python python src1 . src1/migrate().'
    CREATOR = 'scip-python python src1 . src1/setup().'
    _literal(conn, 'INSERT INTO users (email, name) VALUES (?, ?) RETURNING id', INSERTER)
    _literal(conn, 'SELECT u.email FROM users AS u WHERE u.id = ? ORDER BY u.name', SORTER)
    _literal(conn, 'UPDATE users SET email = ? WHERE id = ? RETURNING name', UPDATER)
    _literal(conn, 'ALTER TABLE users ADD COLUMN flag BOOLEAN', ALTERER)
    _literal(conn, 'CREATE TABLE audit (id INTEGER, ts TIMESTAMP)', CREATOR)

    persist_data_access_rawsql(conn, 'src1')

    T = 'data sql src1 _._.'
    assert _rawsql_rows(conn) == {
        # INSERT column list -> write; RETURNING -> project
        (INSERTER, T + 'users#email', 'write', 'derived'),
        (INSERTER, T + 'users#name', 'write', 'derived'),
        (INSERTER, T + 'users#id', 'project', 'derived'),
        # alias u -> users; SELECT list -> project, WHERE -> filter, ORDER BY -> order
        (SORTER, T + 'users#email', 'project', 'derived'),
        (SORTER, T + 'users#id', 'filter', 'derived'),
        (SORTER, T + 'users#name', 'order', 'derived'),
        # UPDATE SET -> write, WHERE -> filter, RETURNING -> project (NOT write)
        (UPDATER, T + 'users#email', 'write', 'derived'),
        (UPDATER, T + 'users#id', 'filter', 'derived'),
        (UPDATER, T + 'users#name', 'project', 'derived'),
        # CREATE/ALTER columns -> ddl
        (ALTERER, T + 'users#flag', 'ddl', 'derived'),
        (CREATOR, T + 'audit#id', 'ddl', 'derived'),
        (CREATOR, T + 'audit#ts', 'ddl', 'derived'),
    }
