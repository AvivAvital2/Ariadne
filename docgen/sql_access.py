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
import re

from typing import TYPE_CHECKING

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError
from attrs import frozen

if TYPE_CHECKING:
    from sqlite3 import Connection

_RAWSQL_WITNESS = 'rawsql'
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


def _access_for_literal(sql, dialect):
    """``(table, column, role)`` triples for one SQL literal (§5.7).

    Value interpolations are normalized to placeholders first (§5.8). DDL
    (CREATE/ALTER) yields a ``ddl`` role on each defined column; an INSERT's
    column list yields ``write``; other column refs are role-typed by their
    governing clause (``_role_for``). Table aliases (``users AS u``) are resolved
    so ``u.col`` binds to ``users``, not ``u``. Empty when the literal does not
    parse, or when a column cannot be bound to a single table. A top-level
    ``SELECT *`` emits a ``(table, '*', 'project')`` whole-row marker that persist
    expands to a read per known column (§10 Phase 2)."""
    try:
        stmt = sqlglot.parse_one(_normalize_placeholders(sql), read=dialect)
    except (SqlglotError, RecursionError):
        return []  # unparseable / pathological -> a gap, not a crash (§7)
    # DDL: CREATE/ALTER define columns -> a ddl-role access on the target table.
    if isinstance(stmt, (exp.Create, exp.Alter)):
        target = stmt.find(exp.Table)
        return [(target.name, cd.name, 'ddl')
                for cd in stmt.find_all(exp.ColumnDef)] if target else []
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
    return out


@frozen
class RawSqlResult:
    """Outcome of persist_data_access_rawsql: the number of data_access rows
    written, and the SQL-shaped literals that could NOT be recovered — a
    recorded gap, never silently dropped (design §5.8)."""
    rows_written: int
    gaps: tuple


_SQL_PREFIX_RE = re.compile(
    r"^\s*(SELECT|INSERT|UPDATE|DELETE|CREATE|ALTER|WITH)\b", re.IGNORECASE
)


def _looks_like_sql(value):
    """True if a literal begins with a SQL DML/DDL keyword. An unrecovered
    SQL-shaped literal is a gap (§5.8); ordinary prose (log lines, messages)
    is not, so the gap counter is not drowned in non-SQL noise."""
    return bool(_SQL_PREFIX_RE.match(value))


def persist_data_access_rawsql(conn, source_name, *, dialect=None):
    """(Re)write this source's raw-SQL ``schema_symbols`` + ``data_access``
    rows from its ``string_literals``. Idempotent: clears the source's prior
    ``rawsql`` rows in both tables first (other witnesses untouched).

    With no declared schema to corroborate against, discovered nodes and
    accesses are recorded at confidence ``'derived'`` (§3a/§5.7). A ``SELECT *``
    is expanded to a read per KNOWN column of the table (from schema_symbols
    the ORM/DDL pass already persisted — §10 Phase 2); if no columns are known
    it is a recorded gap. A SQL-shaped literal that yields no access is recorded
    as a gap rather than silently dropped (§5.8); ordinary non-SQL strings are
    not counted. Returns a ``RawSqlResult``.
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
        if owning is None:
            continue  # no call site to attribute the access to
        triples = _access_for_literal(value, dialect)
        if not triples:
            if _looks_like_sql(value):
                gaps.append(value)  # SQL-shaped but unrecovered -> gap (§5.8)
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
