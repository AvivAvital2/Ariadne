"""End-to-end integration for the SQL data model (design §5.7 -> §3a -> §6).

The unit slices were each tested with hand-injected fixtures, so none
exercised the real composition — which is how the orphaned-node break
slipped through. This composes the ACTUAL pipeline: a raw SQL string
literal -> persist_data_access_rawsql -> add_data_layer -> traversal, and
asserts a raw-SQL access becomes a traversable graph fact. Synthetic only.
"""
from __future__ import annotations

import sqlite3

import pytest

from cli.callers import compute_impact_radius
from docgen.scip_cross_source import CrossSourceGraph
from docgen.sql_access import persist_data_access_rawsql
from library.scip import init_scip_schema

CLEAR = 'scip-python python src1 . src1/clear().'
TABLE = 'data sql src1 _._.users'
COL_EMAIL = 'data sql src1 _._.users#email'
COL_ID = 'data sql src1 _._.users#id'


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    c.execute(
        'INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)',
        (CLEAR, 'src1', 'python', 'q.py', 1, 1, 'Function', 'clear',
         'src1.clear', None),
    )
    c.execute(
        'INSERT INTO string_literals (source_name, file, line_start, '
        'col_start, value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
        ('src1', 'q.py', 1, 0, 'UPDATE users SET email = ? WHERE id = ?', CLEAR),
    )
    yield c
    c.close()


def test_rawsql_access_becomes_a_traversable_graph_fact(conn):
    persist_data_access_rawsql(conn, 'src1')

    # Raw SQL with no declared schema -> derived table + column nodes (§3a),
    # so the data_access edges are no longer orphaned.
    nodes = dict(conn.execute(
        'SELECT canonical_id, confidence FROM schema_symbols '
        "WHERE resolution_source = 'rawsql'"
    ))
    assert nodes == {
        TABLE: 'derived',
        COL_EMAIL: 'derived',
        COL_ID: 'derived',
    }

    # Project into the graph (at the derived floor, since nothing corroborates
    # these yet) and traverse: "what writes users.email?" -> clear(), end to end.
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn, min_confidence='derived')

    assert COL_EMAIL in graph._symbols
    assert CLEAR in {e.caller.canonical_id for e in graph.callers_of(COL_EMAIL)}
    affected = {s.canonical_id for s in
                compute_impact_radius(graph, COL_EMAIL, depth=5).affected_symbols}
    assert CLEAR in affected
