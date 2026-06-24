"""Slice 4 — the data-aware MCP surface (design §6 "Resolution": cheap
query-view tools ``ariadne_data(symbol|table)`` and ``ariadne_schema(table)``,
answered by plain SQL over ``schema_symbols`` + ``data_access`` — no graph —
"so the feature is reachable by agents from day one"; §10 Phase-1 *Surface*).

These close the agent-visibility gap: the doc-graph ``impact_radius`` can't see
data nodes, and agents reach Ariadne only over MCP. Direct questions answer here
over plain SQL; transitive/cross-source answers stay on the SCIP-graph path.

The query views honor the shared confidence floor (the read boundary, §3a/§6a,
line 222) — below-floor facts are held back, never asserted. Synthetic only.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from ariadne_mcp.server_knowledge import (
    ariadne_data,
    ariadne_data_health,
    ariadne_schema,
)
from docgen.sql_query_views import accesses_to_table, schema_of_table
from library.scip import init_scip_schema

# shop source
PLACE = 'scip-python python shop . shop/place_order().'   # writes total
REPORT = 'scip-python python shop . shop/report().'       # filters total
AUDIT = 'scip-python python shop . shop/audit().'         # orders by total
MIGRATE = 'scip-python python shop . shop/migrate().'     # ddl on the table
ORDER_TOTAL = 'scip-python python shop . shop/Order#total.'  # producer
T_ORDERS = 'data sql shop _._.orders'              # table node
C_TOTAL = 'data sql shop _._.orders#total'         # column (exact)
C_NOTE = 'data sql shop _._.orders#note'           # column (derived/below floor)
C_MEMO = 'data sql shop _._.orders#memo'           # column with unknown null/pk
# a second source that happens to own a table of the same name
WH_SYNC = 'scip-python python warehouse . warehouse/sync().'
WH_TOTAL = 'data sql warehouse _._.orders#total'


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / 'views.db'
    c = sqlite3.connect(path)
    init_scip_schema(c)
    # schema_symbols: a table node + two columns (one below the floor) + the
    # same-named table in a second source.
    rows = [
        # (cid, source, node_type, table, column, type, null, pk, ref, producer, conf)
        (T_ORDERS, 'shop', 'table', 'orders', None, None, None, None, None, None, 'exact'),
        (C_TOTAL, 'shop', 'column', 'orders', 'total', 'INTEGER', 0, 1, None, ORDER_TOTAL, 'exact'),
        (C_NOTE, 'shop', 'column', 'orders', 'note', 'TEXT', 1, 0, None, None, 'derived'),
        (C_MEMO, 'shop', 'column', 'orders', 'memo', 'TEXT', None, None, None, None, 'exact'),
        (WH_TOTAL, 'warehouse', 'column', 'orders', 'total', 'BIGINT', 0, 1, None, None, 'exact'),
    ]
    for r in rows:
        c.execute(
            'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
            'table_name, column_name, column_type, is_nullable, is_primary_key, '
            'references_id, producer_symbol_id, resolution_source, confidence) '
            'VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            (*r[:-1], 'orm:django', r[-1]),
        )
    # data_access: a write, two reads (filter/order), a ddl (skipped — not an
    # app access), a below-floor read, and a cross-source write.
    accesses = [
        ('shop', PLACE, C_TOTAL, 'write', 'resolved'),
        ('shop', REPORT, C_TOTAL, 'filter', 'resolved'),
        ('shop', AUDIT, C_TOTAL, 'order', 'resolved'),
        ('shop', MIGRATE, T_ORDERS, 'ddl', 'exact'),
        ('shop', REPORT, C_NOTE, 'filter', 'derived'),
        ('warehouse', WH_SYNC, WH_TOTAL, 'write', 'resolved'),
    ]
    for src, consumer, schema_id, role, conf in accesses:
        c.execute(
            'INSERT INTO data_access (source_name, consumer_symbol_id, '
            'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
            (src, consumer, schema_id, role, 'orm:django', conf),
        )
    c.commit()
    c.close()
    monkeypatch.setattr('config.get_config', lambda: SimpleNamespace(db_path=str(path)))
    return path


# -- the MCP tools (end-to-end: tool -> query view -> SQL, db_path from config) --

def test_ariadne_data_symbol_lists_touches(db):
    # a consumer: the column it writes, role-typed
    out = ariadne_data(symbol=PLACE)
    assert out['symbol'] == PLACE
    assert {'schema_id': C_TOTAL, 'role': 'write'} in out['touches']
    # a producer: the column it defines (maps_to), via schema_symbols
    out2 = ariadne_data(symbol=ORDER_TOTAL)
    assert {'schema_id': C_TOTAL, 'role': 'maps_to'} in out2['touches']


def test_ariadne_data_table_lists_accessors(db):
    out = ariadne_data(table='orders', source='shop')
    writers = {(w['consumer'], w['column']) for w in out['writes']}
    readers = {(r['consumer'], r['column']) for r in out['reads']}
    assert writers == {(PLACE, 'total')}
    assert readers == {(REPORT, 'total'), (AUDIT, 'total')}
    # the ddl row is not an application access; the below-floor read is held back
    assert MIGRATE not in {w['consumer'] for w in out['writes']}
    assert C_NOTE not in {r['column'] for r in out['reads']}  # 'note' excluded


def test_ariadne_schema_lists_columns(db):
    out = ariadne_schema(table='orders', source='shop')
    cols = {c['name']: c for c in out['columns']}
    assert cols['total'] == {
        'name': 'total', 'type': 'INTEGER', 'nullable': False,
        'primary_key': True, 'references': None, 'confidence': 'exact',
    }
    assert 'note' not in cols  # derived → below the floor → not asserted
    # unknown nullable/pk (NULL in the DB) surface as None, not a coerced False
    assert cols['memo']['nullable'] is None
    assert cols['memo']['primary_key'] is None


def test_ariadne_data_requires_exactly_one_selector(db):
    assert 'error' in ariadne_data()                       # neither
    assert 'error' in ariadne_data(symbol=PLACE, table='orders')  # both


# -- the query views directly (focused: confidence floor + source scoping) --

def test_read_boundary_surfaces_below_floor_only_when_lowered(db):
    conn = sqlite3.connect(db)
    try:
        # default floor (resolved): the derived 'note' read/column is held back
        assert all(r['column'] != 'note'
                   for r in accesses_to_table(conn, 'orders', source='shop')['reads'])
        assert all(c['name'] != 'note'
                   for c in schema_of_table(conn, 'orders', source='shop')['columns'])
        # lower the floor to 'derived' → the same facts surface
        lowered = accesses_to_table(conn, 'orders', source='shop', min_confidence='derived')
        assert ('note' in {r['column'] for r in lowered['reads']})
        cols = schema_of_table(conn, 'orders', source='shop', min_confidence='derived')
        assert 'note' in {c['name'] for c in cols['columns']}
    finally:
        conn.close()


def test_source_scoping_isolates_same_named_tables(db):
    conn = sqlite3.connect(db)
    try:
        # no source → both sources' 'orders.total' writers surface
        allsrc = accesses_to_table(conn, 'orders')
        assert {w['consumer'] for w in allsrc['writes']} == {PLACE, WH_SYNC}
        # source='shop' → only shop's
        shop = accesses_to_table(conn, 'orders', source='shop')
        assert {w['consumer'] for w in shop['writes']} == {PLACE}
        # schema_of_table is source-scoped the same way
        assert {c['type'] for c in schema_of_table(conn, 'orders', source='warehouse')['columns']} == {'BIGINT'}
        # ...and unscoped (source=None) spans both sources' columns
        allcols = schema_of_table(conn, 'orders')['columns']
        assert {(c['name'], c['type']) for c in allcols} == {
            ('total', 'INTEGER'), ('memo', 'TEXT'), ('total', 'BIGINT'),
        }
    finally:
        conn.close()


async def test_tools_are_exposed_on_the_live_server():
    # The design's payoff is "reachable by agents from day one" — agents reach
    # Ariadne only over MCP. Asserting against the *assembled* FastMCP server
    # (not a fake mcp) is what actually proves register_tools is wired into
    # server.py and the tools survive onto the live tool list.
    import ariadne_mcp.server as server

    tools = {t.name: t for t in await server.mcp.list_tools()}
    assert 'ariadne_data' in tools, 'ariadne_data not exposed on the live server'
    assert 'ariadne_schema' in tools, 'ariadne_schema not exposed on the live server'
    # the agent-facing selectors survive onto the published input schema
    assert {'symbol', 'table'} <= set(tools['ariadne_data'].inputSchema['properties'])
    assert 'table' in tools['ariadne_schema'].inputSchema['properties']


def test_ariadne_data_health_surfaces_gaps_and_dead_columns(tmp_path, monkeypatch):
    """B3+B4 reachability: ariadne_data_health surfaces the recorded gaps
    (data_model_gaps store) and dead columns (declared, unaccessed). Without
    this tool those diagnostics were computed and discarded."""
    db = tmp_path / 'health.db'
    c = sqlite3.connect(db)
    init_scip_schema(c)
    # a recorded gap (as persist_data_model would store) + a declared,
    # unaccessed column (dead).
    c.execute('INSERT INTO data_model_gaps (source_name, detail) VALUES (?, ?)',
              ('shop', 'orders.legacy: in model, not in schema (drift)'))
    c.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, column_name, resolution_source, confidence) '
        'VALUES (?,?,?,?,?,?,?)',
        ('data sql shop _._.orders#dead', 'shop', 'column', 'orders', 'dead',
         'schema-sql', 'exact'))
    c.commit()
    c.close()
    monkeypatch.setattr('config.get_config', lambda: SimpleNamespace(db_path=str(db)))

    out = ariadne_data_health(source='shop')
    assert any('legacy' in g for g in out['gaps'])
    assert {'table': 'orders', 'column': 'dead'} in out['dead_columns']


async def test_ariadne_data_health_exposed_on_the_live_server():
    import ariadne_mcp.server as server
    tools = {t.name for t in await server.mcp.list_tools()}
    assert 'ariadne_data_health' in tools
