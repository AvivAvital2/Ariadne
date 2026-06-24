"""Query views over the SQL data model (design §6).

The cheap, graph-free read path: answer "who writes / reads this table or
column" with a plain SELECT over ``data_access``. This is what the
``ariadne_data`` MCP tool wraps. It applies the shared confidence floor
(the read boundary, §3a/§6a), so the query view and the graph projection
(``CrossSourceGraph.add_data_layer``) assert exactly the same facts.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from docgen.scip_cross_source import (
    _floor_rank, _CONFIDENCE_RANK)

if TYPE_CHECKING:
    from sqlite3 import Connection

# Role taxonomy (design §5): the SENT 'write' mutates a column; the rest
# observe it. 'maps_to'/'ddl' are structural, not application accesses.
_WRITE_ROLES = {'write'}
_READ_ROLES = {'filter', 'project', 'order'}


def data_access_for(
    conn: 'Connection',
    schema_symbol_id: str,
    *,
    min_confidence: str | None = None,
) -> dict[str, list[str]]:
    """Consumer symbols that access ``schema_symbol_id`` (a table/column
    canonical id), split into ``'writes'`` and ``'reads'`` and filtered to
    facts at/above ``min_confidence`` on the shared ladder.

    Non-access roles (``'maps_to'``, ``'ddl'``) are ignored — they are not
    application data accesses. Results are deterministically ordered.
    """
    floor = _floor_rank(min_confidence)
    writes: list[str] = []
    reads: list[str] = []
    for consumer, role, confidence in conn.execute(
        'SELECT consumer_symbol_id, role, confidence FROM data_access '
        'WHERE schema_symbol_id = ? ORDER BY consumer_symbol_id, role',
        (schema_symbol_id,),
    ):
        if _CONFIDENCE_RANK.get(confidence, -1) < floor:
            continue
        if role in _WRITE_ROLES:
            writes.append(consumer)
        elif role in _READ_ROLES:
            reads.append(consumer)
        # else: maps_to / ddl / unknown — not an app data access; skip
    return {'writes': writes, 'reads': reads}


def data_touched_by(conn, symbol_id, *, min_confidence=None):
    """Tables/columns a code symbol touches — forward trace-flow's terminal
    annotation (§6): the accesses it makes (``data_access`` by consumer,
    role-typed) plus the table/column it defines (``schema_symbols`` by
    producer -> ``maps_to``). Filtered to facts at/above the floor;
    deterministically ordered. Never recursed through — annotation only."""
    floor = _floor_rank(min_confidence)
    touches = []
    for schema_id, role, confidence in conn.execute(
        'SELECT schema_symbol_id, role, confidence FROM data_access '
        'WHERE consumer_symbol_id = ? ORDER BY schema_symbol_id, role',
        (symbol_id,),
    ):
        if _CONFIDENCE_RANK.get(confidence, -1) >= floor:
            touches.append((schema_id, role))
    for canonical_id, confidence in conn.execute(
        'SELECT canonical_id, confidence FROM schema_symbols '
        'WHERE producer_symbol_id = ? ORDER BY canonical_id',
        (symbol_id,),
    ):
        if _CONFIDENCE_RANK.get(confidence, -1) >= floor:
            touches.append((canonical_id, 'maps_to'))
    return tuple(touches)
def accesses_to_table(conn, table_name, *, source=None,
                      min_confidence=None):
    """All application data-access sites of ``table_name`` — every consumer
    that writes/reads the table or any of its columns (design §6, the
    ``ariadne_data(table)`` query view). Plain SQL over ``data_access`` joined
    to ``schema_symbols``; filtered to facts at/above the confidence floor.
    Non-access roles (``'ddl'``) are ignored. Optionally scoped to one source
    (table names can collide across sources). Deterministically ordered."""
    floor = _floor_rank(min_confidence)
    writes = []
    reads = []
    sql = (
        'SELECT da.consumer_symbol_id, s.column_name, da.role, da.confidence '
        'FROM data_access da JOIN schema_symbols s '
        'ON da.schema_symbol_id = s.canonical_id WHERE s.table_name = ?'
    )
    params = [table_name]
    if source is not None:
        sql += ' AND s.source_name = ?'
        params.append(source)
    sql += ' ORDER BY da.consumer_symbol_id, s.column_name, da.role'
    for consumer, column, role, confidence in conn.execute(sql, params):
        if _CONFIDENCE_RANK.get(confidence, -1) < floor:
            continue
        entry = {'consumer': consumer, 'column': column}
        if role in _WRITE_ROLES:
            writes.append(entry)
        elif role in _READ_ROLES:
            reads.append(entry)
    return {'table': table_name, 'writes': writes, 'reads': reads}


def schema_of_table(conn, table_name, *, source=None,
                    min_confidence=None):
    """The columns of ``table_name`` and their types (design §6, the
    ``ariadne_schema(table)`` query view): name, type, nullability, primary-key
    flag, and FK target, from ``schema_symbols``. Filtered to facts at/above
    the confidence floor (a below-floor column name is held back, not
    asserted). Optionally scoped to one source. Deterministically ordered."""
    floor = _floor_rank(min_confidence)
    columns = []
    sql = (
        'SELECT column_name, column_type, is_nullable, is_primary_key, '
        'references_id, confidence FROM schema_symbols '
        "WHERE table_name = ? AND node_type = 'column'"
    )
    params = [table_name]
    if source is not None:
        sql += ' AND source_name = ?'
        params.append(source)
    sql += ' ORDER BY column_name'
    for name, ctype, nullable, pk, ref, confidence in conn.execute(sql, params):
        if _CONFIDENCE_RANK.get(confidence, -1) < floor:
            continue
        columns.append({
            'name': name,
            'type': ctype,
            'nullable': None if nullable is None else bool(nullable),
            'primary_key': None if pk is None else bool(pk),
            'references': ref,
            'confidence': confidence,
        })
    return {'table': table_name, 'columns': columns}


def dead_columns(conn, source_name, *, min_confidence=None):
    """Declared columns (``schema_symbols`` at/above the floor) that no code
    symbol reads or writes — no ``data_access`` row references them (design §10
    Phase 2, dead-column detection). Returns ``[(table, column)]``, ordered."""
    floor = _floor_rank(min_confidence)
    dead = []
    rows = conn.execute(
        'SELECT canonical_id, table_name, column_name, confidence '
        'FROM schema_symbols WHERE source_name = ? AND node_type = ? '
        'ORDER BY canonical_id',
        (source_name, 'column'),
    ).fetchall()
    for cid, table, column, confidence in rows:
        if _CONFIDENCE_RANK.get(confidence, -1) < floor:
            continue  # below the read boundary — not an asserted declaration
        accessed = conn.execute(
            'SELECT 1 FROM data_access WHERE schema_symbol_id = ? LIMIT 1', (cid,),
        ).fetchone()
        if accessed is None:
            dead.append((table, column))
    return dead


def data_model_gaps(conn, source_name):
    """The gaps recorded for a source during indexing — undecodable query forms,
    schema drift/typo (§3a/§5.0 "surface, don't guess"). Read from the
    ``data_model_gaps`` store that ``persist_data_model`` fills; ordered."""
    return [r[0] for r in conn.execute(
        'SELECT detail FROM data_model_gaps WHERE source_name = ? ORDER BY id',
        (source_name,))]
