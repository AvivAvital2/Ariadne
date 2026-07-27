"""Raw-SQL access extraction (design §5.7): parse SQL string literals with
``sqlglot`` and emit role-typed ``data_access`` rows.

The table/column tokens are literal text inside the SQL string, so this
recovers data access at call sites SCIP cannot decode (it sees only
``cursor.execute(<string>)``). Static-only: ``sqlglot`` parses text;
nothing connects to or executes a database (design §2, §7). Unparseable,
pathological, or un-bindable literals are skipped — a recorded gap, never
a crash.
"""
from __future__ import annotations
import logging
import re
from contextlib import contextmanager

from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from attrs import frozen

if TYPE_CHECKING:
    from sqlite3 import Connection

_RAWSQL_WITNESS = 'rawsql'
FSTRING_PLACEHOLDER = '__ariadne_ph__'
_INTERP_RE = re.compile(r"\{[^{}]*\}|%\(\w+\)[a-zA-Z]|%[a-zA-Z]")


def _normalize_placeholders(sql):
    """Replace value interpolations with "?" placeholders (design §5.8)."""
    return _INTERP_RE.sub("?", sql)


def _tables_in(stmt: 'exp.Expression') -> list[str]:
    return [t.name for t in stmt.find_all(exp.Table)]


def _role_for(col: 'exp.Column', stmt: 'exp.Expression') -> str:
    """Classify a column reference by the clause that governs it (§5.7): WHERE /
    JOIN -> ``filter``, ORDER BY -> ``order``, RETURNING -> ``project`` (a read,
    even inside an UPDATE/DELETE), an UPDATE's SET list -> ``write``, else a
    SELECT-list projection -> ``project``. The nearest governing clause wins."""
    node = col.parent
    while node is not None and node is not stmt:
        if isinstance(node, (exp.Where, exp.Join)):
            return 'filter'
        if isinstance(node, exp.Order):
            return 'order'
        if isinstance(node, exp.Returning):
            return 'project'
        node = node.parent
    if isinstance(stmt, exp.Update):
        return 'write'
    return 'project'


def _drop_placeholders(triples):
    """Drop triples whose table or column is the f-string interpolation
    placeholder — a placeholdered interpolation is never a real schema node
    (design: f-string SQL capture). Applied to every emission path, DDL included."""
    return [t for t in triples if FSTRING_PLACEHOLDER not in (t[0], t[1])]


_FALLBACK_DIALECTS = ('duckdb', 'postgres', 'mysql', 'sqlite', None)

# A literal is SQL when sqlglot parses it to one of these top-level STATEMENT
# types. Requiring a statement — not merely "it parsed" — is what separates real
# SQL from filenames/prose that tokenize: a dotted name like ``config.json``
# parses as a bare ``Column`` (table.column), never a statement.
_STATEMENTS = (
    exp.Select, exp.Union, exp.Insert, exp.Update, exp.Delete,
    exp.Merge, exp.Create, exp.Alter, exp.Drop,
)


@contextmanager
def _quiet_sqlglot():
    """Suppress sqlglot's per-string ``Command``-fallback WARNING *for the
    duration of our detection parse only*. We use sqlglot as a SQL classifier: a
    string that doesn't parse to a statement is consumed here as 'not SQL' (or
    recorded as a gap by the binder), so sqlglot logging that same verdict once
    per string is redundant noise — not a hidden error (real unrecovered SQL is
    surfaced via ``RawSqlResult.gaps``). Scoped: it restores the prior level, so
    sqlglot's logging is untouched outside our classification."""
    logger = logging.getLogger('sqlglot')
    prev = logger.level
    logger.setLevel(logging.ERROR)
    try:
        yield
    finally:
        logger.setLevel(prev)


def _parse_sql(sql, dialect=None):
    """Parse SQL to a structured *statement*, trying ``dialect`` first (when set)
    then a fallback of common dialects so dialect-specific syntax reads without
    per-source configuration. Returns the statement, or ``None`` when the literal
    is not SQL — it parses only to a bare expression, sqlglot's ``Command``
    fallback, or nothing.

    The dialect fallback retries only on a parse *error* (dialect-specific
    syntax one dialect rejects and another accepts). A clean parse that isn't a
    statement is a verdict, not a recall miss, so it returns immediately —
    avoiding both wasted re-parses and repeated ``Command`` warnings."""
    with _quiet_sqlglot():
        tried = set()
        for d in (dialect, *_FALLBACK_DIALECTS):
            if d in tried:
                continue
            tried.add(d)
            try:
                stmt = sqlglot.parse_one(sql, read=d)
            except (SqlglotError, RecursionError, ValueError, ArithmeticError):
                # sqlglot can raise beyond SqlglotError on pathological strings,
                # and these escape a SqlglotError-only guard: a plain ValueError
                # ("Cannot convert empty name into var." from an empty
                # identifier), or decimal.InvalidOperation (an ArithmeticError)
                # when a 1-based dialect type-annotates a bracket subscript whose
                # number token isn't a valid Decimal — e.g. the JSONPath literal
                # "$.1.2.3[0]" under 'postgres' → Decimal('1.2.3'). Either way an
                # unparseable string is simply not SQL.
                continue
            return stmt if isinstance(stmt, _STATEMENTS) else None
        return None


def is_sql(value) -> bool:
    """The SQL detector: True when ``value`` parses to a SQL *statement*.
    Replaces the old first-word ``looks_like_sql`` heuristic — sqlglot is the
    authority on whether a string is SQL, and requiring a statement (not a bare
    expression) is what excludes filenames/prose that merely tokenize."""
    return _parse_sql(_normalize_placeholders(value)) is not None


def _triples_from_stmt(stmt):
    """``(table, column, role)`` triples for a parsed SQL *statement* (§5.7).

    DDL (CREATE/ALTER) yields a ``ddl`` role on each defined column; an INSERT's
    column list yields ``write``; other column refs are role-typed by their
    governing clause (``_role_for``). Table aliases (``users AS u``) are resolved
    so ``u.col`` binds to ``users``, not ``u``. Empty when a column cannot be
    bound to a single table. A top-level ``SELECT *`` emits a
    ``(table, '*', 'project')`` whole-row marker that persist expands to a read
    per known column (§10 Phase 2). Interpolation placeholders are dropped from
    every path (``_drop_placeholders``)."""
    # DDL: CREATE/ALTER define columns -> a ddl-role access on the target table.
    if isinstance(stmt, (exp.Create, exp.Alter)):
        target = stmt.find(exp.Table)
        return _drop_placeholders(
            [(target.name, cd.name, 'ddl')
             for cd in stmt.find_all(exp.ColumnDef)]) if target else []
    tables = _tables_in(stmt)
    aliases = {t.alias: t.name for t in stmt.find_all(exp.Table) if t.alias}
    default_table = tables[0] if len(tables) == 1 else None
    out = []
    # INSERT column list (Identifiers, not exp.Column) -> write.
    if isinstance(stmt, exp.Insert) and isinstance(stmt.this, exp.Schema):
        for ident in stmt.this.expressions:
            out.append((stmt.this.this.name, ident.name, 'write'))
    for col in stmt.find_all(exp.Column):
        table = aliases.get(col.table, col.table) or default_table
        if not table:
            continue  # unqualified column, ambiguous/absent table -> gap
        out.append((table, col.name, _role_for(col, stmt)))
    if isinstance(stmt, exp.Select) and any(
            isinstance(e, exp.Star) for e in stmt.expressions):
        # whole-row read; bindable only when the FROM names a single table.
        if len(tables) == 1:
            out.append((tables[0], '*', 'project'))
    return _drop_placeholders(out)


def _access_for_literal(sql, dialect):
    """``(table, column, role)`` triples for one SQL literal — parse (normalizing
    interpolations to placeholders, §5.8) then extract from the statement. Empty
    when the literal is not a SQL statement. Retained for the audit script and
    tests; the binder parses once and calls ``_triples_from_stmt`` directly."""
    stmt = _parse_sql(_normalize_placeholders(sql), dialect)
    return _triples_from_stmt(stmt) if stmt is not None else []


@frozen
class RawSqlResult:
    """Outcome of persist_data_access_rawsql: the number of data_access rows
    written, and the SQL-shaped literals that could NOT be recovered — a
    recorded gap, never silently dropped (design §5.8)."""
    rows_written: int
    gaps: tuple


def persist_data_access_rawsql(conn, source_name, *, dialect=None):
    """(Re)write this source's raw-SQL ``schema_symbols`` + ``data_access``
    rows from its ``string_literals``. Idempotent: clears the source's prior
    ``rawsql`` rows in both tables first (other witnesses untouched).

    With no declared schema to corroborate against, discovered nodes and
    accesses are recorded at confidence ``'derived'`` (§3a/§5.7). A ``SELECT *``
    is expanded to a read per KNOWN column of the table (from schema_symbols
    the ORM/DDL pass already persisted — §10 Phase 2); if no columns are known
    it is a recorded gap. A literal that parses to a SQL *statement* but yields
    no bindable access is recorded as a gap rather than silently dropped (§5.8);
    strings that aren't SQL statements (sqlglot is the detector) are not counted.
    Returns a ``RawSqlResult``.
    """
    conn.execute(
        'DELETE FROM data_access WHERE source_name = ? AND witness = ?',
        (source_name, _RAWSQL_WITNESS),
    )
    conn.execute(
        'DELETE FROM schema_symbols WHERE source_name = ? AND resolution_source = ?',
        (source_name, _RAWSQL_WITNESS),
    )
    nodes = {}
    access = []
    gaps = []
    for value, owning, file, line in conn.execute(
        'SELECT value, owning_symbol_id, file, line_start '
        'FROM string_literals WHERE source_name = ?',
        (source_name,),
    ):
        stmt = _parse_sql(_normalize_placeholders(value), dialect)
        if stmt is None:
            continue  # not a SQL statement -> not SQL (sqlglot is the detector); skip
        if owning is None:
            gaps.append(value)  # SQL with no owning symbol -> gap, not dropped (§5.8)
            continue
        triples = _triples_from_stmt(stmt)
        if not triples:
            gaps.append(value)  # a SQL statement we couldn't bind -> gap (§5.8)
            continue
        for table, column, role in triples:
            table_id = f'data sql {source_name} _._.{table}'
            if column == '*':
                # SELECT * -> a read per known column (schema already persisted).
                known = [r[0] for r in conn.execute(
                    'SELECT column_name FROM schema_symbols WHERE source_name = ? '
                    "AND table_name = ? AND node_type = 'column'",
                    (source_name, table),
                )]
                if not known:
                    gaps.append(value)  # whole-row read, columns unknown -> gap
                    continue
                for known_col in known:
                    access.append((
                        source_name, owning, f'{table_id}#{known_col}', 'project',
                        file, line, _RAWSQL_WITNESS, 'derived',
                    ))
                continue
            col_id = f'{table_id}#{column}'
            nodes.setdefault(table_id, (
                table_id, source_name, 'table', table, None,
                _RAWSQL_WITNESS, 'derived',
            ))
            nodes.setdefault(col_id, (
                col_id, source_name, 'column', table, column,
                _RAWSQL_WITNESS, 'derived',
            ))
            access.append((
                source_name, owning, col_id, role, file, line,
                _RAWSQL_WITNESS, 'derived',
            ))
    conn.executemany(
        'INSERT OR IGNORE INTO schema_symbols '
        '(canonical_id, source_name, node_type, table_name, column_name, '
        ' resolution_source, confidence) VALUES (?, ?, ?, ?, ?, ?, ?)',
        list(nodes.values()),
    )
    conn.executemany(
        'INSERT OR IGNORE INTO data_access '
        '(source_name, consumer_symbol_id, schema_symbol_id, role, '
        ' call_site_file, call_site_line, witness, confidence) '
        'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
        access,
    )
    return RawSqlResult(rows_written=len(access), gaps=tuple(gaps))
