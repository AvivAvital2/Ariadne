"""trace_flow walks CALL flow, not type references.

scip-python records a function's outgoing references indiscriminately: real
calls (``foo().``), class/type refs (``Bar#``), attribute/variable refs
(``x.``), and anonymous locals (``local N``) all land in ``scip_edges`` with
edge_type 'call'. Walking the non-call ones buries the actual flow and — since
a widely-referenced class points all over the graph — fans out until the trace
times out. Only ``foo().``-shaped callees (and non-scip synthetic ids used in
tests) are genuine flow hops.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.trace_flow import trace_flow
from library.scip import init_scip_schema

CALLER = 'scip-python python app . `app.svc`/route().'
CALL = 'scip-python python app . `app.svc`/handler().'
CLS = 'scip-python python app . `app.models`/User#'
FIELD = 'scip-python python app . `app.models`/User#id.'
LOCAL = 'local 14'


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    for sym in (CALLER, CALL, CLS, FIELD):
        c.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'app', 'python', 'app/svc.py', 1, 1, '', sym, sym, None),
        )
    # route() "references" a real call + a class + a field + a local — all
    # land in scip_edges, but only the call is flow.
    for callee in (CALL, CLS, FIELD, LOCAL):
        c.execute(
            'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
            'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
            (CALLER, callee, 'call', 'app/svc.py', 5, 'exact'),
        )
    yield c
    c.close()


def test_trace_flow_follows_calls_not_type_refs(conn) -> None:
    result = trace_flow(start_symbol=CALLER, depth=3, conn=conn)
    callees = {h.callee_symbol_id for h in result.hops}
    assert CALL in callees, 'the real call must be followed'
    assert CLS not in callees, 'class/type reference must be skipped'
    assert FIELD not in callees, 'attribute reference must be skipped'
    assert LOCAL not in callees, 'anonymous local must be skipped'
def test_trace_flow_excludes_contains_edges_even_when_member_looks_callable(conn):
    nested = "scip-python python app . `app.svc`/Owner#nested()."
    conn.execute(
        "INSERT INTO scip_symbols (canonical_id, source_name, language, file, "
        "line_start, line_end, kind, display_name, qualified_name, "
        "parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (nested, "app", "python", "app/svc.py", 20, 25, "Method",
         "nested", nested, CALLER),
    )
    conn.execute(
        "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, "
        "edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)",
        (CALLER, nested, "contains", "app/svc.py", 20, "exact"),
    )

    result = trace_flow(start_symbol=CALLER, depth=3, conn=conn)

    assert nested not in {hop.callee_symbol_id for hop in result.hops}
