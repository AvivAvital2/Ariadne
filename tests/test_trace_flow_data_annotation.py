"""Phase 1 / slice 3b — forward trace-flow's terminal data annotation
(design §6: "a terminal annotation … always shown, never recursed through").

Forward ``trace-flow`` answers "where does this go"; reaching the database
*ends* the flow. So at each symbol the trace visits, annotate the tables/
columns it touches — the accesses it makes (``data_access`` by consumer,
role-typed) plus the table/column it defines (``schema_symbols`` by producer)
— but NEVER walk out of a data node into its other readers. Synthetic only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.sql_query_views import data_touched_by
from docgen.trace_flow import trace_flow
from library.scip import init_scip_schema

HANDLER = 'scip-python python app . app/views/handler().'
DEACT = 'scip-python python app . app/svc/deactivate().'
U_EMAIL = 'scip-python python app . app/models/User#email.'
COL = 'data sql app _._.users#email_addr'
SECRET = 'data sql app _._.users#secret'


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    for sym, file in [(HANDLER, 'app/views.py'), (DEACT, 'app/svc.py'),
                      (U_EMAIL, 'app/models.py')]:
        c.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'app', 'python', file, 1, 1, '', sym, sym, None),
        )
    # handler() calls deactivate()  (a real SCIP call edge)
    c.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
        (HANDLER, DEACT, 'call', 'app/views.py', 5, 'exact'),
    )
    # deactivate writes users.email (resolved) and filters a derived column
    for schema, role, witness, conf in [
        (COL, 'write', 'orm:django', 'resolved'),
        (SECRET, 'filter', 'rawsql', 'derived'),
    ]:
        c.execute(
            'INSERT INTO data_access (source_name, consumer_symbol_id, '
            'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
            ('app', DEACT, schema, role, witness, conf),
        )
    # the column is defined by the User.email field (producer)
    c.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, column_name, producer_symbol_id, resolution_source, '
        'confidence) VALUES (?,?,?,?,?,?,?,?)',
        (COL, 'app', 'column', 'users', 'email_addr', U_EMAIL, 'orm:django', 'exact'),
    )
    # a derived column the same field also "defines" — held below the floor
    c.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, column_name, producer_symbol_id, resolution_source, '
        'confidence) VALUES (?,?,?,?,?,?,?,?)',
        ('data sql app _._.users#draft', 'app', 'column', 'users', 'draft',
         U_EMAIL, 'orm:django', 'derived'),
    )
    yield c
    c.close()


def test_data_touched_by_reports_accesses_and_defines(conn):
    # a consumer: the resolved access shows; the derived one is held back
    assert data_touched_by(conn, DEACT) == ((COL, 'write'),)
    # a producer: the table/column it defines
    assert data_touched_by(conn, U_EMAIL) == ((COL, 'maps_to'),)
    # lowering the floor surfaces the derived access too (sorted)
    assert data_touched_by(conn, DEACT, min_confidence='derived') == (
        (COL, 'write'), (SECRET, 'filter'),
    )


def test_trace_flow_annotates_data_touches_without_recursing(conn):
    result = trace_flow(start_symbol=HANDLER, conn=conn)

    # the trace hops through the SCIP call, NOT into the data node
    assert [(h.tier, h.callee_symbol_id) for h in result.hops] == [('scip', DEACT)]
    assert all(h.callee_symbol_id != COL for h in result.hops)  # never recursed

    # ...and deactivate is annotated with the column it writes (terminal)
    assert result.data_touches[DEACT] == ((COL, 'write'),)
    assert COL not in result.data_touches  # a data node is annotation, not a hop
