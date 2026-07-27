"""Schema-as-data witnesses (design §4): parse a schema into ``schema_symbols``
and cross-check it against the ORM/raw-SQL rows, promoting a ``derived`` name
the schema confirms up to ``resolved`` (the recall-saver, §3a line 230; §4 line
289-294 — the schema is the name authority, the ORM keeps ``producer_symbol_id``,
disagreement is flagged as drift).

Two witnesses feed the one merge engine (``_merge_tables``):

- ``parse_schema_ddl`` — generic ``CREATE TABLE`` via ``sqlglot`` (Postgres to
  start; the dialect is a parameter).
- ``parse_django_migrations`` — Django's *own committed output* (``CreateModel``
  in migration files), the design-faithful per-ORM witness (§5.0.1 bullet 2),
  reusing ``DjangoStrategy``'s column naming so the names match the model-derived
  rows and thus promote them.

Static parse only — no DB execution (§2/§7).
"""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING

import sqlglot
from attrs import field, frozen
from sqlglot import exp
from sqlglot.errors import SqlglotError

from docgen.scip_cross_source import _CONFIDENCE_RANK
from ast_utils import safe_ast_parse

if TYPE_CHECKING:
    from sqlite3 import Connection

_DDL_WITNESS = 'schema-sql'
_MIGRATION_WITNESS = 'migration'
_ALEMBIC_WITNESS = 'alembic'  # distinct from Django's so the two never clobber
_RESOLVED_RANK = _CONFIDENCE_RANK['resolved']


@frozen
class SchemaDdlResult:
    """Outcome of merging a parsed schema into ``schema_symbols``."""
    promoted: int = 0          # derived rows the schema corroborated → resolved
    declared: int = 0          # exact nodes the schema declares (incl. new ones)
    gaps: tuple = field(factory=tuple)  # drift / observed-but-undeclared


def parse_schema_ddl(sql, *, dialect='postgres'):
    """Parse ``CREATE TABLE`` statements into ``[{table, columns: [{name, type,
    nullable, primary_key}]}]``. Non-``CREATE TABLE`` statements (indexes,
    inserts) and table-level constraints are skipped — a dump carries much
    ``sqlglot`` needn't model. Postgres by default; the dialect is a parameter
    so other databases plug in later."""
    tables = []
    try:
        statements = sqlglot.parse(sql, read=dialect)
    except (SqlglotError, ValueError, ArithmeticError):
        # sqlglot may raise beyond SqlglotError on pathological input: a plain
        # ValueError (empty-identifier "Cannot convert empty name into var.") or
        # decimal.InvalidOperation (an ArithmeticError) when a 1-based dialect
        # annotates a bracket subscript with a non-Decimal number token.
        return tables
    for stmt in statements:
        if not isinstance(stmt, exp.Create) or not isinstance(stmt.this, exp.Schema):
            continue  # not a CREATE TABLE (CREATE INDEX/VIEW/AS → this isn't a Schema)
        schema = stmt.this  # exp.Schema; schema.this is the exp.Table
        columns = []
        for cd in schema.expressions:
            if not isinstance(cd, exp.ColumnDef):
                continue  # a table-level constraint, e.g. PRIMARY KEY (id)
            kind = cd.args.get('kind')
            cons = {type(c.kind).__name__ for c in cd.constraints}
            columns.append({
                'name': cd.name,
                'type': kind.sql() if kind else None,
                'nullable': 'NotNullColumnConstraint' not in cons,
                'primary_key': 'PrimaryKeyColumnConstraint' in cons,
            })
        if columns:
            tables.append({'table': schema.this.name, 'columns': columns})
    return tables


def persist_schema_ddl(conn, source_name, sql, *, dialect='postgres'):
    """Merge a parsed ``CREATE TABLE`` schema into ``schema_symbols`` — the
    generic DDL witness. See ``_merge_tables`` for the cross-check semantics."""
    return _merge_tables(
        conn, source_name, parse_schema_ddl(sql, dialect=dialect),
        witness=_DDL_WITNESS)


def parse_django_migrations(migrations):
    """Parse Django ``CreateModel`` operations into the same ``[{table,
    columns}]`` shape as ``parse_schema_ddl``. ``migrations`` is an iterable of
    ``(app_label, source_code)``. Django's own committed migrations are a
    name-authority witness (§5.0.1 bullet 2, the design-faithful promotion
    source); reuses ``DjangoStrategy._column`` so the names MATCH the
    model-derived rows and promote them. Only ``CreateModel`` is read here —
    ``AddField``/``AlterField``/``RunSQL`` are a documented future refinement."""
    from docgen.orm_bindings.django import _column

    tables = []
    for app_label, code in migrations:
        try:
            tree = safe_ast_parse(code)
        except SyntaxError:
            continue  # unparsable migration -> skip (never crash)
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and _attr_name(node.func) == 'CreateModel'):
                continue
            kwargs = {kw.arg: kw.value for kw in node.keywords}
            model = _str_const(kwargs.get('name'))
            if model is None:
                continue
            columns = [
                {'name': _column(fname, fcall, None).column_name,
                 'type': None, 'nullable': None, 'primary_key': None}
                for fname, fcall in _create_model_fields(kwargs.get('fields'))
            ]
            if columns:
                tables.append({
                    'table': _migration_table(kwargs.get('options'), app_label, model),
                    'columns': columns,
                })
    return tables


def persist_schema_from_migrations(conn, source_name, migrations):
    """Merge Django migrations (Django's committed output) into
    ``schema_symbols`` — the design-faithful per-ORM witness. Same cross-check
    as the DDL path; promotes the model-derived names the migrations confirm."""
    return _merge_tables(
        conn, source_name, parse_django_migrations(migrations),
        witness=_MIGRATION_WITNESS)


def _merge_tables(conn, source_name, tables, *, witness):
    """The shared cross-check engine (design §4 line 289-294): the schema is the
    name/type authority. A pre-existing ``derived`` row the schema confirms is
    promoted to ``resolved`` (keeping ``producer_symbol_id``); a column the
    schema declares but no row covers is inserted ``exact``; a model column on a
    declared table the schema lacks is held ``derived`` and surfaced as drift; a
    query touching an undeclared column is surfaced as observed-but-undeclared
    (§6). Idempotent: clears this witness's prior rows first."""
    conn.execute(
        'DELETE FROM schema_symbols WHERE source_name = ? AND resolution_source = ?',
        (source_name, witness),
    )
    promoted = declared = 0
    gaps = []
    for t in tables:
        table = t['table']
        table_id = f'data sql {source_name} _._.{table}'
        p, d = _merge(conn, source_name, table_id, 'table', table, None, None,
                      witness=witness)
        promoted += p
        declared += d
        declared_ids = {table_id}
        for col in t['columns']:
            cid = f'data sql {source_name} _._.{table}#{col["name"]}'
            declared_ids.add(cid)
            p, d = _merge(conn, source_name, cid, 'column', table, col['name'], col,
                          witness=witness)
            promoted += p
            declared += d
        # Drift: a row on a declared table whose column the schema does not
        # declare — the model has a column the DB doesn't. Held, never promoted.
        for cid, colname in conn.execute(
            'SELECT canonical_id, column_name FROM schema_symbols WHERE '
            "source_name = ? AND table_name = ? AND node_type = 'column'",
            (source_name, table),
        ).fetchall():
            if cid not in declared_ids:
                gaps.append(f'{table}.{colname}: in model, not in schema (drift)')
        # Query-side cross-check (§6 line 376): a query touching a column this
        # table does not declare → observed-but-undeclared (typo/drift).
        for _cid, _colname, _consumer in conn.execute(
            'SELECT s.canonical_id, s.column_name, da.consumer_symbol_id '
            'FROM data_access da JOIN schema_symbols s '
            'ON da.schema_symbol_id = s.canonical_id WHERE s.source_name = ? '
            "AND s.table_name = ? AND s.node_type = 'column'",
            (source_name, table),
        ).fetchall():
            if _cid not in declared_ids:
                gaps.append(
                    f'{table}.{_colname} queried by {_consumer} but not in schema (typo/drift)')
    return SchemaDdlResult(promoted=promoted, declared=declared, gaps=tuple(gaps))


def _merge(conn, source_name, cid, node_type, table, column, col, *, witness):
    """Confirm-or-declare one node; returns ``(promoted, declared)`` deltas."""
    row = conn.execute(
        'SELECT confidence FROM schema_symbols WHERE canonical_id = ?', (cid,),
    ).fetchone()
    ctype = col['type'] if col else None
    nullable = (1 if col['nullable'] else 0) if col else None
    pk = (1 if col['primary_key'] else 0) if col else None
    if row is None:
        conn.execute(
            'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
            'table_name, column_name, column_type, is_nullable, is_primary_key, '
            'resolution_source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (cid, source_name, node_type, table, column, ctype, nullable, pk,
             witness, 'exact'),
        )
        return 0, 1
    # The schema is the name/type authority; promote a below-floor row to
    # resolved and keep its producer binding. An asserted row keeps confidence.
    promoted = 1 if _CONFIDENCE_RANK.get(row[0], -1) < _RESOLVED_RANK else 0
    new_conf = 'resolved' if promoted else row[0]
    conn.execute(
        'UPDATE schema_symbols SET confidence = ?, column_type = ?, '
        'is_nullable = ?, is_primary_key = ? WHERE canonical_id = ?',
        (new_conf, ctype, nullable, pk, cid),
    )
    return promoted, 0


def _attr_name(func):
    return (func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None)


def _str_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _create_model_fields(fields_node):
    """``[(field_name, field_call)]`` from a ``CreateModel`` ``fields=[...]`` —
    a list of ``(name, models.X(...))`` tuples."""
    if not isinstance(fields_node, ast.List):
        return []
    out = []
    for elt in fields_node.elts:
        if isinstance(elt, (ast.Tuple, ast.List)) and len(elt.elts) == 2:
            name = _str_const(elt.elts[0])
            if name is not None and isinstance(elt.elts[1], ast.Call):
                out.append((name, elt.elts[1]))
    return out


def _migration_table(options_node, app_label, model):
    """``options={'db_table': 'x'}`` wins; else Django's ``app_label_model``."""
    if isinstance(options_node, ast.Dict):
        for k, v in zip(options_node.keys, options_node.values):
            if _str_const(k) == 'db_table':
                explicit = _str_const(v)
                if explicit:
                    return explicit
    return f'{app_label}_{model.lower()}'


def parse_alembic_migrations(migrations):
    """Parse Alembic ``op.create_table``/``op.add_column``/``op.alter_column``
    operations into the same ``[{table, columns}]`` shape as the other witnesses.
    ``migrations`` is an iterable of migration source strings. Names only (like the
    Django witness — column type/nullable/pk are a future refinement). create_table
    and add_column declare columns; an ``alter_column`` carrying ``new_column_name``
    declares the renamed-to column (the old name lingers — full ordered drop/replay
    is a documented future step, §10 P4)."""
    by_table: dict = {}
    for code in migrations:
        try:
            tree = safe_ast_parse(code)
        except SyntaxError:
            continue  # unparsable migration -> skip (never crash)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for table, name in _op_columns(node):
                    by_table.setdefault(table, []).append(
                        {'name': name, 'type': None, 'nullable': None,
                         'primary_key': None})
    return [{'table': t, 'columns': c} for t, c in by_table.items()]
def _op_columns(call):
    """``(table, column_name)`` pairs an Alembic op declares — ``create_table``'s
    ``sa.Column`` args, ``add_column``'s single column, or ``alter_column``'s
    ``new_column_name`` (a rename declares the post-rename column). Empty for
    anything else (another verb, a non-literal table, a non-``op`` receiver, an
    ``alter_column`` that is not a rename)."""
    method = _op_method(call)
    if method == 'create_table':
        table = _str_const(call.args[0]) if call.args else None
        if table is None:
            return []
        return [(table, n) for a in call.args[1:]
                if (n := _column_name(a)) is not None]
    if method == 'add_column':
        table = _str_const(call.args[0]) if call.args else None
        col = call.args[1] if len(call.args) > 1 else None
        name = _column_name(col) if col is not None else None
        if table is not None and name is not None:
            return [(table, name)]
    if method == 'alter_column':
        table = _str_const(call.args[0]) if call.args else None
        new = next((_str_const(k.value) for k in call.keywords
                    if k.arg == 'new_column_name'), None)
        if table is not None and new is not None:
            return [(table, new)]
    return []


def _op_method(call):
    """The method name of an ``op.<method>(...)`` call, else None."""
    func = call.func
    if (isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name)
            and func.value.id == 'op'):
        return func.attr
    return None


def _column_name(arg):
    """The column name from a ``sa.Column("name", …)`` arg, or None for a
    constraint / a Column with no string-literal name."""
    if not (isinstance(arg, ast.Call) and _attr_name(arg.func) == 'Column'):
        return None
    return _first_str(arg.args)


def _first_str(args):
    """The first positional string-literal arg, or None."""
    return next((s for a in args if (s := _str_const(a)) is not None), None)


def persist_schema_from_alembic(conn, source_name, migrations):
    """Merge Alembic migrations (``op.create_table``/``op.add_column``) into
    ``schema_symbols`` — a distinct witness from Django's so the two never
    clobber. Same cross-check; promotes the model-derived names the migration
    confirms (§4/§5.0.1 bullet 2)."""
    return _merge_tables(
        conn, source_name, parse_alembic_migrations(migrations),
        witness=_ALEMBIC_WITNESS)
