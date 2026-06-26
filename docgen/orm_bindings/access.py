"""Layer-2 access binding engine (design §5.0, §5.1, §5.2) — the per-ORM query
recognizer dispatcher.

Owns the invariant Layer-2 pipeline, written once: build the ``(file, line) ->
enclosing symbol`` owning map and each detected strategy's ``model -> (table,
{field: column})`` schema map, dispatch to the strategy's ``recognize`` for the
per-ORM query decode, then (re)write the uniform role-typed ``data_access`` rows
(idempotent per source+strategy) and aggregate the strategies' surfaced gaps —
never silent (§5.0, §5.8). A field declared in the schema binds at ``resolved``
(§3a). Each ORM's recognizer lives in its own strategy module (Django in
``django.py``, SQLAlchemy in ``sqlalchemy.py``); only the pipeline is here.

Static-only: ``ast`` over source already on disk; nothing is imported or run.
"""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from attrs import frozen
from docgen.orm_bindings.engine import _parse, _symbol_at
from docgen.scip_owning import build_owning_resolver

if TYPE_CHECKING:
    from sqlite3 import Connection

    from docgen.scip_extractor import ScipIndex


@frozen
class AccessResult:
    """Outcome of ``persist_data_access_orm``: ``data_access`` rows written,
    and query forms that could not be decoded/resolved — recorded gaps, never
    silently dropped (§5.0, §5.8)."""
    rows_written: int
    gaps: tuple


def _import_aliases(scip_index: 'ScipIndex', root) -> dict:
    """``relative_path -> {local_name: imported_name}`` for every aliased
    ``from module import Name as Local`` in each source file — so a query that
    names a model under an import alias resolves to the declared model name
    before the Layer-2 schema lookup (§5.0). Built once and threaded into every
    strategy's ``recognize``. A file that does not parse contributes nothing (its
    parse gap is already recorded by Layer-1 discovery). A model aliased via a
    re-export aggregator (the query file imports the already-renamed name) is not
    resolved here and remains a surfaced gap, never a silent wrong answer."""
    out: dict = {}
    for doc in scip_index.documents:
        tree = _parse(root, doc.relative_path)
        if tree is None:
            continue
        file_aliases = {
            name.asname: name.name
            for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
            for name in node.names if name.asname
        }
        if file_aliases:
            out[doc.relative_path] = file_aliases
    return out


def _schema_map(strategy, scip_index, root) -> dict:
    """``model_name -> (table_name, {field_name: column_name})`` from the
    strategy's Layer-1 discovery — so a query's field names resolve to columns."""
    tables, _gaps = strategy.discover(scip_index, root, _symbol_at(scip_index))
    return {
        t.model_name: (t.table_name, {c.field_name: c.column_name for c in t.columns})
        for t in tables
    }


def persist_data_access_orm(conn, source_name, scip_index, *, strategies):
    """Recognize ORM query call sites and (re)write their role-typed
    ``data_access`` rows (idempotent per source+strategy). Returns an
    ``AccessResult`` carrying the rows written and the undecodable/unresolved
    query forms as surfaced gaps (§5.0, §5.8). Recognized columns are declared
    in the schema, so rows bind at ``resolved`` (§3a)."""
    owning = build_owning_resolver(scip_index)
    root = scip_index.source_root
    aliases = _import_aliases(scip_index, root)
    written = 0
    gaps: list[str] = []
    for strategy in strategies:
        if not strategy.detect(scip_index, root):
            continue
        schema = _schema_map(strategy, scip_index, root)
        rows, strat_gaps = strategy.recognize(
            scip_index, root, owning, schema, aliases)
        gaps.extend(strat_gaps)
        conn.execute(
            'DELETE FROM data_access WHERE source_name = ? AND witness = ?',
            (source_name, strategy.name))
        out = [
            (source_name, consumer,
             f'data sql {source_name} _._.{table}#{column}', role, file, line,
             strategy.name, 'resolved')
            for consumer, table, column, role, file, line in rows
        ]
        conn.executemany(
            'INSERT OR IGNORE INTO data_access '
            '(source_name, consumer_symbol_id, schema_symbol_id, role, '
            ' call_site_file, call_site_line, witness, confidence) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            out)
        written += len(out)
    return AccessResult(rows_written=written, gaps=tuple(gaps))
