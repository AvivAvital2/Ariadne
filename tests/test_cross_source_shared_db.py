"""Cross-source column identity — opt-in, default-off (design §6, "Cross-source
column identity is opt-in").

The canonical id embeds ``<source>`` (§4), so by default two services that each
own a ``accounts.balance`` are **distinct** nodes and no cross-source coupling is
inferred — fusing them on a name match would assert a coupling that may not exist
(the confidently-wrong fact §6a forbids). Cross-source identity is emitted **only**
behind an explicit ``shared_database`` declaration naming the sources (and,
optionally, the database/schema) that share one physical schema. Under that gate
the matching column in every member source collapses to ONE node, so an
A-writes / B-reads coupling is traversable "in one walk"; absent it, the query
returns nothing, not a guess.

Synthetic sources only.
"""
from __future__ import annotations

import sqlite3

import pytest

from cli.callers import compute_impact_radius
from docgen.scip_cross_source import (
    CrossSourceGraph,
    SharedDatabase,
    shared_databases_from_config,
)
from library.scip import init_scip_schema

# code consumers (one per source)
A_WRITER = 'scip-python python svc_a . svc_a/writer().'
B_READER = 'scip-python python svc_b . svc_b/reader().'
C_OTHER = 'scip-python python svc_c . svc_c/other().'
# model attrs — each "defines" (produces) its source's balance column
A_ATTR = 'scip-python python svc_a . svc_a/Account#balance.'
B_ATTR = 'scip-python python svc_b . svc_b/Account#balance.'
C_ATTR = 'scip-python python svc_c . svc_c/Account#balance.'
# model classes — each produces its source's accounts table node
A_CLS = 'scip-python python svc_a . svc_a/Account#.'
B_CLS = 'scip-python python svc_b . svc_b/Account#.'
# per-source data nodes (the default: distinct, source-keyed)
A_COL = 'data sql svc_a main.public.accounts#balance'
B_COL = 'data sql svc_b main.public.accounts#balance'
C_COL = 'data sql svc_c main.public.accounts#balance'
A_TBL = 'data sql svc_a main.public.accounts'
B_TBL = 'data sql svc_b main.public.accounts'
# the fused, source-independent ids the gate must mint for group {svc_a, svc_b}
SHARED_COL = 'data sql @shared:svc_a+svc_b main.public.accounts#balance'
SHARED_TBL = 'data sql @shared:svc_a+svc_b main.public.accounts'

_AB = [SharedDatabase(sources=frozenset({'svc_a', 'svc_b'}))]


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    for sym, src in [
        (A_WRITER, 'svc_a'), (B_READER, 'svc_b'), (C_OTHER, 'svc_c'),
        (A_ATTR, 'svc_a'), (B_ATTR, 'svc_b'), (C_ATTR, 'svc_c'),
        (A_CLS, 'svc_a'), (B_CLS, 'svc_b'),
    ]:
        c.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, src, 'python', f'{src}.py', 1, 1, '', sym, sym, None),
        )
    # column nodes: one per source, same (db, schema, table, column)
    for cid, src, producer in [
        (A_COL, 'svc_a', A_ATTR), (B_COL, 'svc_b', B_ATTR), (C_COL, 'svc_c', C_ATTR),
    ]:
        c.execute(
            'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
            'database, db_schema, table_name, column_name, producer_symbol_id, '
            'resolution_source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (cid, src, 'column', 'main', 'public', 'accounts', 'balance',
             producer, 'orm:django', 'exact'),
        )
    # table nodes (column_name NULL) for two of the sources
    for cid, src, producer in [(A_TBL, 'svc_a', A_CLS), (B_TBL, 'svc_b', B_CLS)]:
        c.execute(
            'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
            'database, db_schema, table_name, column_name, producer_symbol_id, '
            'resolution_source, confidence) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (cid, src, 'table', 'main', 'public', 'accounts', None,
             producer, 'orm:django', 'exact'),
        )
    # accesses: A writes, B filters, C filters — each its own source's column
    for src, consumer, schema_id, role in [
        ('svc_a', A_WRITER, A_COL, 'write'),
        ('svc_b', B_READER, B_COL, 'filter'),
        ('svc_c', C_OTHER, C_COL, 'filter'),
    ]:
        c.execute(
            'INSERT INTO data_access (source_name, consumer_symbol_id, '
            'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
            (src, consumer, schema_id, role, 'orm:django', 'resolved'),
        )
    yield c
    c.close()


def _graph(conn, shared_database=None):
    g = CrossSourceGraph()
    g.load_from(conn)
    g.add_data_layer(conn, shared_database=shared_database)
    return g


def _callers(g, cid):
    return {(e.caller.canonical_id, e.edge_type) for e in g.callers_of(cid)}


def test_default_off_columns_stay_per_source(conn):
    # No declaration: each source's balance is a DISTINCT, source-keyed node.
    g = _graph(conn)
    assert SHARED_COL not in g._symbols  # nothing fused
    assert {A_COL, B_COL, C_COL} <= set(g._symbols)
    # svc_a's column is reachable only from svc_a's own writer + producer —
    # no cross-source coupling is inferred.
    assert _callers(g, A_COL) == {(A_WRITER, 'write'), (A_ATTR, 'maps_to')}
    assert _callers(g, B_COL) == {(B_READER, 'filter'), (B_ATTR, 'maps_to')}


def test_shared_declaration_fuses_columns_across_sources(conn):
    # Declaring svc_a + svc_b share a database collapses their balance column
    # into ONE node carrying BOTH sources' accesses and BOTH producers' maps_to.
    g = _graph(conn, _AB)
    assert SHARED_COL in g._symbols
    assert A_COL not in g._symbols and B_COL not in g._symbols  # collapsed
    assert _callers(g, SHARED_COL) == {
        (A_WRITER, 'write'), (A_ATTR, 'maps_to'),
        (B_READER, 'filter'), (B_ATTR, 'maps_to'),
    }


def test_unlisted_source_is_never_fused(conn):
    # svc_c is not named in the declaration: even with an identically-named
    # column it stays a distinct per-source node — never a name match.
    g = _graph(conn, _AB)
    assert C_COL in g._symbols
    assert _callers(g, C_COL) == {(C_OTHER, 'filter'), (C_ATTR, 'maps_to')}
    assert (C_OTHER, 'filter') not in _callers(g, SHARED_COL)


def test_table_node_fuses_too(conn):
    # The gate applies to table nodes (column_name NULL), not only columns.
    g = _graph(conn, _AB)
    assert SHARED_TBL in g._symbols
    assert _callers(g, SHARED_TBL) == {(A_CLS, 'maps_to'), (B_CLS, 'maps_to')}


def test_database_scope_narrows_the_gate(conn):
    # database='other' excludes rows physically in 'main' → no fusion.
    g_miss = _graph(conn, [SharedDatabase(
        sources=frozenset({'svc_a', 'svc_b'}), database='other')])
    assert SHARED_COL not in g_miss._symbols and A_COL in g_miss._symbols
    # database='main' matches the rows → fuses.
    g_hit = _graph(conn, [SharedDatabase(
        sources=frozenset({'svc_a', 'svc_b'}), database='main')])
    assert SHARED_COL in g_hit._symbols


def test_schema_scope_narrows_the_gate(conn):
    # db_schema='other' excludes rows in schema 'public' → no fusion.
    g_miss = _graph(conn, [SharedDatabase(
        sources=frozenset({'svc_a', 'svc_b'}), db_schema='other')])
    assert SHARED_COL not in g_miss._symbols and A_COL in g_miss._symbols
    # db_schema='public' matches → fuses.
    g_hit = _graph(conn, [SharedDatabase(
        sources=frozenset({'svc_a', 'svc_b'}), db_schema='public')])
    assert SHARED_COL in g_hit._symbols


def test_impact_radius_couples_writer_and_reader_in_one_walk(conn):
    # The §6 headline: changing the shared column reaches code in BOTH sources.
    g = _graph(conn, _AB)
    report = compute_impact_radius(g, SHARED_COL, depth=3)
    reached = {s.canonical_id for s in report.affected_symbols}
    assert A_WRITER in reached and B_READER in reached
    # ...whereas with no declaration the same start reaches nothing (no guess).
    g_off = _graph(conn)
    empty = compute_impact_radius(g_off, SHARED_COL, depth=3)
    assert empty.affected_symbols == []


def test_from_config_parses_declarations():
    assert shared_databases_from_config(None) == []
    assert shared_databases_from_config([]) == []
    decls = shared_databases_from_config([
        {'sources': ['svc_a', 'svc_b'], 'database': 'main', 'schema': 'public'},
        {'sources': ['svc_c', 'svc_d']},  # optional keys absent
    ])
    assert decls == [
        SharedDatabase(frozenset({'svc_a', 'svc_b'}), 'main', 'public'),
        SharedDatabase(frozenset({'svc_c', 'svc_d'}), None, None),
    ]
