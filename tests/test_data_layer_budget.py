"""The §9a in-memory data-edge budget. ``add_data_layer`` holds the projected
data layer in ``self._edges`` for ``impact_radius``; a measured ~262 B/edge means
a runaway repo could resident-balloon, so projection is capped at ``max_data_edges``
and, when it caps, warns loudly — never a silent truncation (design §9a)."""
from __future__ import annotations

import logging
import sqlite3

from docgen.scip_cross_source import CrossSourceGraph, CrossSourceSymbol
from library.scip import init_scip_schema


def _conn_with_accesses(n: int) -> sqlite3.Connection:
    """One column accessed by ``n`` distinct consumers (n data_access rows)."""
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    conn.execute(
        "INSERT INTO schema_symbols (canonical_id, source_name, node_type, "
        "table_name, column_name, resolution_source, confidence) "
        "VALUES ('data sql s _._.t#c', 's', 'column', 't', 'c', 'ddl', 'exact')")
    conn.executemany(
        "INSERT INTO data_access (source_name, consumer_symbol_id, "
        "schema_symbol_id, role, witness, confidence) VALUES (?, ?, ?, ?, ?, ?)",
        [('s', f'fn{i}', 'data sql s _._.t#c', 'filter', 'rawsql', 'exact')
         for i in range(n)])
    conn.commit()
    return conn


def _graph_with_consumers(n: int) -> CrossSourceGraph:
    graph = CrossSourceGraph()
    for i in range(n):
        graph._symbols[f'fn{i}'] = CrossSourceSymbol(
            canonical_id=f'fn{i}', source_name='s', language='python', file='m.py',
            line_start=i, line_end=i, kind='Function', display_name=f'fn{i}',
            qualified_name=f'fn{i}', parent_qualified_name=None)
    return graph


def _data_edges(graph: CrossSourceGraph) -> list:
    return [e for e in graph._edges if e.edge_type == 'filter']


def test_add_data_layer_caps_data_edges_and_warns(caplog) -> None:
    """3 accesses, budget 2 → only 2 data edges projected; the 3rd is dropped
    with a warning naming the cap (the §9a safety valve, surfaced not silent)."""
    conn = _conn_with_accesses(3)
    graph = _graph_with_consumers(3)
    with caplog.at_level(logging.WARNING):
        projected = graph.add_data_layer(
            conn, min_confidence='recovered', max_data_edges=2)
    conn.close()
    assert projected == 2
    assert len(_data_edges(graph)) == 2
    assert any('2' in r.getMessage() and 'budget' in r.getMessage().lower()
               for r in caplog.records)


def test_add_data_layer_under_budget_projects_all_silently(caplog) -> None:
    """Accesses under the budget all project; the loop completes with no warning
    (the common path — the cap never trips for realistic sizes)."""
    conn = _conn_with_accesses(2)
    graph = _graph_with_consumers(2)
    with caplog.at_level(logging.WARNING):
        projected = graph.add_data_layer(
            conn, min_confidence='recovered', max_data_edges=5)
    conn.close()
    assert projected == 2
    assert len(_data_edges(graph)) == 2
    assert not [r for r in caplog.records if 'budget' in r.getMessage().lower()]


def test_default_budget_is_generous_enough_for_realistic_sizes() -> None:
    """The default budget (no ``max_data_edges`` argument) does not cap a realistic
    handful of accesses — it is a runaway guard, not a normal limiter (a real
    source projects on the order of 100 edges)."""
    conn = _conn_with_accesses(5)
    graph = _graph_with_consumers(5)
    projected = graph.add_data_layer(conn, min_confidence='recovered')
    conn.close()
    assert projected == 5
