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

import logging
import sqlite3

import pytest

from docgen.sql_access import (
    _access_for_literal,
    _parse_sql,
    persist_data_access_rawsql,
)
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


def test_binder_does_not_leak_sqlglot_command_fallback_warnings(conn):
    """The binder uses sqlglot as a SQL classifier: a literal that isn't a
    parseable statement is consumed here as 'not SQL' or recorded as a gap.
    sqlglot's per-string ``Command``-fallback WARNING is redundant with that
    handling (it announces the verdict we already act on), so it must not leak to
    the logs during detection — dynamic SQL and prose alike."""
    captured: list[str] = []

    class _Cap(logging.Handler):
        def emit(self, record):
            captured.append(record.getMessage())

    handler = _Cap()
    sqlglot_log = logging.getLogger('sqlglot')
    sqlglot_log.addHandler(handler)
    try:
        _literal(conn, 'CREATE __x__ TABLE __y__ (__z__)', READER)  # dynamic SQL -> Command
        _literal(conn, 'Replace the dataset on mismatch', READER)   # prose -> Command
        persist_data_access_rawsql(conn, 'src1')
    finally:
        sqlglot_log.removeHandler(handler)

    assert not any('Command' in m or 'unsupported syntax' in m for m in captured), captured


def test_access_for_literal_parses_then_extracts_or_empties():
    """``_access_for_literal`` (the audit's per-literal entry point) parses then
    extracts: a SQL statement yields its (table, column, role) triples; a bare
    expression or non-SQL string yields nothing."""
    assert set(_access_for_literal('SELECT email FROM users WHERE id = 1', 'duckdb')) == {
        ('users', 'email', 'project'),
        ('users', 'id', 'filter'),
    }
    assert _access_for_literal('config.json', 'duckdb') == []   # bare expression, not SQL
    assert _access_for_literal('just prose here', 'duckdb') == []


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


def test_bare_expression_strings_are_not_mistaken_for_sql(conn):
    """SQL detection requires a parsed *statement*, not merely a successful
    parse. A dotted string — a filename like ``config.json`` or ``a.b`` — parses
    as a bare column reference (``table.column``), NOT a statement, so it must
    not be read as data access. Only real statements create schema nodes."""
    _literal(conn, 'SELECT email FROM users WHERE id = ?', READER)  # statement -> binds
    _literal(conn, 'config.json', READER)            # bare expr config.json -> NOT SQL
    _literal(conn, 'a.b', READER)                    # bare expr -> NOT SQL
    _literal(conn, 'Create a leaf node.', READER)    # not a statement -> NOT SQL

    persist_data_access_rawsql(conn, 'src1')

    nodes = {
        r[0] for r in conn.execute(
            "SELECT canonical_id FROM schema_symbols WHERE source_name = 'src1'")
    }
    # only the real query produced nodes — no phantom config/json/a/b tables
    assert nodes == {
        'data sql src1 _._.users',
        'data sql src1 _._.users#email',
        'data sql src1 _._.users#id',
    }


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
    # --- given: a recoverable query, a SQL STATEMENT we recognize but can't
    #            turn into a column-level fact (a table-only DELETE), and a
    #            non-SQL string (a log line) -----------------------------
    _literal(conn, 'SELECT email FROM users WHERE id = ?', READER)
    _literal(conn, 'DELETE FROM cache', WRITER)               # statement, no column access
    _literal(conn, 'just a log message, not a query', READER)  # not a SQL statement

    result = persist_data_access_rawsql(conn, 'src1')

    # the recoverable query still produces its rows
    assert result.rows_written == 2
    # the SQL statement we recognized but couldn't bind is recorded (§5.8:
    # "never silently dropped"); the non-SQL string is not counted as a gap
    assert set(result.gaps) == {'DELETE FROM cache'}


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


def test_parse_sql_skips_command_fallbacks_and_returns_none_when_unparsed():
    """_parse_sql treats sqlglot's opaque Command fallback as 'not parsed' and
    keeps trying dialects; when every dialect yields only a Command it returns
    None (a gap), never a Command masquerading as a real statement."""
    assert _parse_sql('CREATE magic thing') is None


def test_rawsql_falls_back_across_dialects_per_query(conn):
    """With NO configured dialect, the binder reads dialect-specific SQL by
    trying supported dialects per query: DuckDB integer-division (``//``) and
    MySQL backtick identifiers both bind, in one source, no config."""
    _literal(conn, 'SELECT a // b FROM t WHERE c = 1', READER)        # // : duckdb only
    _literal(conn, 'SELECT `x` FROM `tbl` WHERE `y` = 1', WRITER)     # backticks: mysql only
    persist_data_access_rawsql(conn, 'src1')                          # no dialect= passed
    rows = _rawsql_rows(conn)
    assert (READER, 'data sql src1 _._.t#c', 'filter', 'derived') in rows
    assert (WRITER, 'data sql src1 _._.tbl#y', 'filter', 'derived') in rows


def test_ddl_placeholder_names_are_not_emitted_as_schema(conn):
    """A CREATE/ALTER whose table or column was an f-string interpolation is
    captured with the placeholder token; the DDL path must drop placeholders
    like every other path, never emitting a phantom schema node (accuracy)."""
    _literal(conn, 'CREATE TABLE __ariadne_ph__ (age INTEGER)', WRITER, line=3)
    _literal(conn, 'CREATE TABLE orders (status TEXT)', WRITER, line=4)
    persist_data_access_rawsql(conn, 'src1', dialect='duckdb')
    nodes = {r[0] for r in conn.execute(
        "SELECT canonical_id FROM schema_symbols WHERE source_name = 'src1'")}
    assert not any('__ariadne_ph__' in n for n in nodes)   # placeholder dropped
    assert 'data sql src1 _._.orders#status' in nodes      # a real DDL still binds
