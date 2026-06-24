"""Behavioral tests for the SQL data layer projection (design §6, §8) and
the read-boundary confidence threshold (design §3a, §6a).

The design's central claim is that table/column nodes are *ordinary*
cross-source graph nodes: once ``add_data_layer`` projects the
``schema_symbols`` / ``data_access`` rows into the graph, the EXISTING
traversal (``callers_of`` / ``compute_impact_radius``) reaches them with
no change to those functions. The second claim is the safety valve: a
fact below the assert threshold (default ``resolved``) is NEVER asserted
in the graph — it is held as a gap. Entirely synthetic fixtures.

Synthetic only: source ``src1``, table ``users``, column ``email``.
"""
from __future__ import annotations

import sqlite3

import pytest

from cli.callers import compute_impact_radius
from docgen.scip_cross_source import CrossSourceGraph
from library.scip import init_scip_schema

# Data nodes use the 'data sql' canonical scheme (design §4); '_' is the
# database/schema sentinel for dialects/codebases that don't use them.
TABLE_ID = 'data sql src1 _._.users'
COLUMN_EMAIL_ID = 'data sql src1 _._.users#email'

# App-side SCIP symbols (synthetic scip-python-style wire ids — the graph
# treats canonical_id as an opaque string, so the exact shape is immaterial).
DEACTIVATE_ID = 'scip-python python src1 . src1/deactivate().'
NOTIFY_ID = 'scip-python python src1 . src1/notify().'
USER_EMAIL_ATTR_ID = 'scip-python python src1 . src1/User#email.'
# Deliberately NEVER inserted into scip_symbols — exercises orphan-skip.
ORPHAN_ID = 'scip-python python src1 . src1/orphan().'
# A writer/table whose binding is only a derived guess (below 'resolved').
DERIVED_WRITER_ID = 'scip-python python src1 . src1/legacy_update().'
DERIVED_TABLE_ID = 'data sql src1 _._.audit'


def _insert_symbol(conn, canonical_id, qn, kind, file='src1/models.py', line=1):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, '
        'file, line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (canonical_id, 'src1', 'python', file, line, line, kind,
         qn.rsplit('.', 1)[-1], qn, None),
    )


def _insert_column(conn, canonical_id, table, column, *, producer=None,
                   confidence='resolved'):
    conn.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'database, db_schema, table_name, column_name, producer_symbol_id, '
        'resolution_source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (canonical_id, 'src1', 'column', '_', '_', table, column, producer,
         'orm:django', confidence),
    )


def _insert_access(conn, consumer, schema_symbol, role, *, confidence='resolved',
                   file='src1/models.py', line=11):
    conn.execute(
        'INSERT INTO data_access (source_name, consumer_symbol_id, '
        'schema_symbol_id, role, call_site_file, call_site_line, witness, '
        'confidence) VALUES (?,?,?,?,?,?,?,?)',
        ('src1', consumer, schema_symbol, role, file, line, 'orm:django',
         confidence),
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def test_data_nodes_traverse_through_existing_graph(conn):
    # --- given: the §8 world -------------------------------------------
    # app symbols: a writer (query API), a reader (attribute ref), the
    # ORM attribute that *is* the column.
    _insert_symbol(conn, DEACTIVATE_ID, 'src1.deactivate', 'Function', line=10)
    _insert_symbol(conn, NOTIFY_ID, 'src1.notify', 'Function', line=4)
    _insert_symbol(conn, USER_EMAIL_ATTR_ID, 'src1.User.email', 'Field', line=2)

    # (A) a GENUINE SCIP reference: notify() reads User.email
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
        (NOTIFY_ID, USER_EMAIL_ATTR_ID, 'reference', 'src1/models.py', 5, 'exact'),
    )

    # A SQL-first TABLE node — no producer symbol (schema came from DDL,
    # not an ORM model). And the column node, bound to its producer
    # attribute (Layer 1 / maps_to).
    conn.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, resolution_source, confidence) VALUES (?,?,?,?,?,?)',
        (TABLE_ID, 'src1', 'table', 'users', 'ddl', 'exact'),
    )
    _insert_column(conn, COLUMN_EMAIL_ID, 'users', 'email',
                   producer=USER_EMAIL_ATTR_ID, confidence='exact')

    # (B) the query-API WRITE: deactivate() does .update(email=…) — a kwarg,
    # NOT a SCIP reference, so only a data_access row captures it.
    _insert_access(conn, DEACTIVATE_ID, COLUMN_EMAIL_ID, 'write',
                   confidence='resolved')
    # An ORPHAN access: the consumer symbol was never indexed. It must be
    # skipped (mirrors load_from's orphan-edge handling), never asserted.
    _insert_access(conn, ORPHAN_ID, COLUMN_EMAIL_ID, 'filter',
                   confidence='resolved', file='gone.py', line=1)

    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)

    # --- then: the column is a first-class node ------------------------
    assert COLUMN_EMAIL_ID in graph._symbols
    col = graph._symbols[COLUMN_EMAIL_ID]
    assert (col.kind, col.language, col.display_name) == ('Column', 'sql', 'email')
    assert col.parent_qualified_name == '_._.users'

    # --- and: the SQL-first table node has no producer line (sentinel) -
    table = graph._symbols[TABLE_ID]
    assert (table.kind, table.parent_qualified_name) == ('Table', None)
    assert (table.file, table.line_start) == ('', 0)  # no producer -> sentinel

    # --- and: callers_of finds BOTH the write edge and the maps_to link,
    #          the access role preserved as edge_type, and the orphan is
    #          absent (skipped, not asserted) --------------------------
    by_caller = {
        e.caller.canonical_id: e.edge_type
        for e in graph.callers_of(COLUMN_EMAIL_ID)
    }
    assert by_caller == {
        DEACTIVATE_ID: 'write',          # (B) data_access role
        USER_EMAIL_ATTR_ID: 'maps_to',   # (Layer 1) producer binding
    }
    assert ORPHAN_ID not in by_caller    # orphan consumer skipped

    # --- and: impact_radius reaches the writer directly AND the reader
    #          transitively (maps_to -> existing SCIP ref) — the §8 payoff
    affected = {s.canonical_id for s in
                compute_impact_radius(graph, COLUMN_EMAIL_ID, depth=5).affected_symbols}
    assert DEACTIVATE_ID in affected       # writes it (data_access)
    assert USER_EMAIL_ATTR_ID in affected  # is it (maps_to)
    assert NOTIFY_ID in affected           # reads it (maps_to then SCIP ref)


def test_below_threshold_facts_are_not_asserted(conn):
    # --- given: one solidly-resolved write, plus a *derived* (guessed)
    #            write and a *derived* table node ------------------------
    _insert_symbol(conn, DEACTIVATE_ID, 'src1.deactivate', 'Function', line=10)
    _insert_symbol(conn, DERIVED_WRITER_ID, 'src1.legacy_update', 'Function', line=20)

    _insert_column(conn, COLUMN_EMAIL_ID, 'users', 'email', confidence='resolved')
    conn.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, resolution_source, confidence) VALUES (?,?,?,?,?,?)',
        (DERIVED_TABLE_ID, 'src1', 'table', 'audit', 'orm:django', 'derived'),
    )
    _insert_access(conn, DEACTIVATE_ID, COLUMN_EMAIL_ID, 'write',
                   confidence='resolved')
    _insert_access(conn, DERIVED_WRITER_ID, COLUMN_EMAIL_ID, 'filter',
                   confidence='derived')

    # --- then: at the default floor ('resolved'), derived facts are gaps,
    #           not assertions — a confidently-wrong fact is worse than an
    #           honest gap (§6a) ----------------------------------------
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)

    callers = {e.caller.canonical_id for e in graph.callers_of(COLUMN_EMAIL_ID)}
    assert DEACTIVATE_ID in callers              # resolved -> asserted
    assert DERIVED_WRITER_ID not in callers      # derived  -> held as a gap
    assert DERIVED_TABLE_ID not in graph._symbols  # derived node not asserted

    # --- and: the threshold IS the gate — lowering it to 'derived'
    #           asserts exactly what was held back -----------------------
    relaxed = CrossSourceGraph()
    relaxed.load_from(conn)
    relaxed.add_data_layer(conn, min_confidence='derived')

    callers_relaxed = {e.caller.canonical_id
                       for e in relaxed.callers_of(COLUMN_EMAIL_ID)}
    assert DERIVED_WRITER_ID in callers_relaxed
    assert DERIVED_TABLE_ID in relaxed._symbols
