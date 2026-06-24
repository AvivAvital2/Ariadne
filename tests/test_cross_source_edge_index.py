"""Slice 3c — endpoint-indexed edges (design §6: "register data edges in a
dict[canonical_id -> edges] index instead of appending to the scanned list —
O(1) per hop, no traversal blow-up. The index helps the existing code edges
too.").

A performance refactor: ``callers_of``/``callees_of`` must return EXACTLY what
the naive O(E) scan returns (correctness invariant), over a graph carrying both
code edges (SCIP) and data edges (``add_data_layer``), and stay consistent
after re-projection (the index can't drift from ``_edges``). Synthetic only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.scip_cross_source import (
    CrossSourceEdge,
    CrossSourceGraph,
    CrossSourceSymbol,
)
from library.scip import init_scip_schema

A = 'scip-python python s . s/a().'
B = 'scip-python python s . s/b().'
C = 'scip-python python s . s/c().'
COL = 'data sql s _._.t#col'


@pytest.fixture
def graph():
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym in (A, B, C):
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 's', 'python', 'x.py', 1, 1, '', sym, sym, None),
        )
    for caller, callee in [(A, B), (A, C), (B, C)]:  # code edges
        conn.execute(
            'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
            'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
            (caller, callee, 'call', 'x.py', 2, 'exact'),
        )
    # a data node + two data-access edges
    conn.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, column_name, producer_symbol_id, resolution_source, '
        'confidence) VALUES (?,?,?,?,?,?,?,?)',
        (COL, 's', 'column', 't', 'col', B, 'orm:django', 'exact'),
    )
    for consumer, role in [(A, 'write'), (C, 'filter')]:
        conn.execute(
            'INSERT INTO data_access (source_name, consumer_symbol_id, '
            'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
            ('s', consumer, COL, role, 'orm:django', 'resolved'),
        )
    g = CrossSourceGraph()
    g.load_from(conn)
    g.add_data_layer(conn)
    yield g
    conn.close()


def _scan_callers(g, sym):
    return [e for e in g._edges if e.callee.canonical_id == sym]


def _scan_callees(g, sym):
    return [e for e in g._edges if e.caller.canonical_id == sym]


def test_endpoint_index_matches_the_naive_scan(graph):
    # every symbol (code + data nodes) — the indexed lookup equals the scan
    for sym in graph._symbols:
        assert graph.callers_of(sym) == _scan_callers(graph, sym)
        assert graph.callees_of(sym) == _scan_callees(graph, sym)
    # spot-check the data node is reachable both ways (B maps_to COL; A/C access)
    assert {e.caller.canonical_id for e in graph.callers_of(COL)} == {A, B, C}
    # an unknown symbol yields no edges (dict miss, not a crash)
    assert graph.callers_of('nope') == []
    assert graph.callees_of('nope') == []


def test_callers_of_answers_a_directly_assigned_edge_list():
    """A graph built by assigning ``_symbols``/``_edges`` directly — the
    pattern ``catalog_enrich`` uses — must answer ``callers_of`` / ``callees_of``
    WITHOUT a prior ``materialize`` / ``load_from`` / ``add_data_layer`` call.

    Regression guard (slice 3c, c5f3e7c): the lazy endpoint-index defaulted
    ``_edge_index_dirty=False``, so a direct ``_edges`` assignment left the index
    stale-empty and ``callers_of``/``callees_of`` wrongly returned ``[]``.
    """
    def _sym(cid):
        return CrossSourceSymbol(
            canonical_id=cid, source_name='s', language='python', file='x.py',
            line_start=1, line_end=1, kind='', display_name=cid,
            qualified_name=cid, parent_qualified_name=None)

    a, b = _sym('A'), _sym('B')
    edge = CrossSourceEdge(caller=a, callee=b, edge_type='call', file='x.py', line=1)
    g = CrossSourceGraph()
    g._symbols = {'A': a, 'B': b}
    g._edges = [edge]  # direct — no materialize/load_from/add_data_layer
    assert g.callers_of('B') == [edge]
    assert g.callees_of('A') == [edge]
