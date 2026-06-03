"""Producer-internal schema invariants for SCIP + Layer C tables.

``init_scip_schema`` materializes the tables that ``ariadne_search``,
``ariadne_callers``, ``ariadne_trace_flow``, and the cross-source
graph builder all read. If a column those queries depend on gets
dropped or renamed in ``library_scip.py``, the dependent queries
silently return wrong data — these tests catch that at the schema
boundary instead of at the query boundary.

Salvaged from ``tests/contract/test_scip_schema_contract.py`` (the
slim-consumer fork). With the slim consumer gone, these tests remain
useful as producer-side regression guards.
"""
from __future__ import annotations

import sqlite3

import pytest

from library.scip import init_scip_schema


# Columns each downstream query path reads. Inline because the slim's
# schema_contract.py is gone and these constants live close to the
# tests that exercise them — exactly one consumer.

SCIP_SYMBOLS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'canonical_id', 'source_name', 'language', 'file',
    'line_start', 'line_end', 'kind', 'display_name',
    'qualified_name', 'parent_qualified_name',
})

SCIP_EDGES_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'caller_canonical_id', 'callee_canonical_id', 'edge_type',
    'file', 'line', 'confidence',
})

API_ENDPOINTS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'endpoint_id', 'source_name', 'http_method', 'path_template',
    'producer_symbol_id', 'resolution_source',
})

API_CALLS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'consumer_symbol_id', 'endpoint_id', 'call_site_file',
    'call_site_line', 'resolution_source', 'confidence',
})

STRING_LITERALS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'source_name', 'file', 'line_start', 'col_start',
    'value', 'owning_symbol_id',
})

CONFIG_VALUES_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'source_name', 'file', 'key', 'value', 'line_start',
})

PROCESS_INVOCATIONS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'source_name', 'caller_symbol_id', 'target_path',
    'target_symbol_id', 'confidence', 'file', 'line_start', 'line_end',
})

HTTP_CLIENT_CALLS_REQUIRED_COLUMNS: frozenset[str] = frozenset({
    'source_name', 'consumer_symbol_id', 'raw_url', 'http_method',
    'call_site_file', 'call_site_line', 'sink_name', 'confidence',
})


@pytest.fixture
def fresh_scip_db():
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    yield conn
    conn.close()


def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    rows = conn.execute(f'PRAGMA table_info({table})').fetchall()
    return {row[1] for row in rows}


@pytest.mark.parametrize(
    ('table', 'required'),
    [
        ('scip_symbols', SCIP_SYMBOLS_REQUIRED_COLUMNS),
        ('scip_edges', SCIP_EDGES_REQUIRED_COLUMNS),
        ('api_endpoints', API_ENDPOINTS_REQUIRED_COLUMNS),
        ('api_calls', API_CALLS_REQUIRED_COLUMNS),
        ('string_literals', STRING_LITERALS_REQUIRED_COLUMNS),
        ('config_values', CONFIG_VALUES_REQUIRED_COLUMNS),
        ('process_invocations', PROCESS_INVOCATIONS_REQUIRED_COLUMNS),
        ('http_client_calls', HTTP_CLIENT_CALLS_REQUIRED_COLUMNS),
    ],
)
def test_required_columns_present(
    fresh_scip_db: sqlite3.Connection,
    table: str,
    required: frozenset[str],
) -> None:
    """``init_scip_schema`` materializes ``table`` with every column the
    downstream query path reads. Producer additions are fine; removals
    break callers (``CrossSourceGraph.load_from``, the trace-flow
    query, etc.) and this test bites first."""
    actual = _columns(fresh_scip_db, table)
    missing = required - actual
    assert missing == set(), (
        f'{table} missing required columns: {sorted(missing)}. '
        f'Either restore them in library_scip.py or update the '
        f'downstream readers + this test.'
    )


def test_double_init_preserves_all_required_columns() -> None:
    """Calling ``init_scip_schema`` twice on the same connection (e.g.
    after Library reopens an existing DB) must not drop any column.
    Bites a ``CREATE TABLE`` that drops ``IF NOT EXISTS`` or a
    destructive migration that silently rewrites the schema."""
    conn = sqlite3.connect(':memory:')
    try:
        init_scip_schema(conn)
        init_scip_schema(conn)
        checks = [
            ('scip_symbols', SCIP_SYMBOLS_REQUIRED_COLUMNS),
            ('scip_edges', SCIP_EDGES_REQUIRED_COLUMNS),
            ('api_endpoints', API_ENDPOINTS_REQUIRED_COLUMNS),
            ('api_calls', API_CALLS_REQUIRED_COLUMNS),
            ('string_literals', STRING_LITERALS_REQUIRED_COLUMNS),
            ('config_values', CONFIG_VALUES_REQUIRED_COLUMNS),
            ('process_invocations', PROCESS_INVOCATIONS_REQUIRED_COLUMNS),
            ('http_client_calls', HTTP_CLIENT_CALLS_REQUIRED_COLUMNS),
        ]
        for table, required in checks:
            actual = _columns(conn, table)
            missing = required - actual
            assert missing == set(), (
                f'after double init, {table} missing: {sorted(missing)}'
            )
    finally:
        conn.close()
