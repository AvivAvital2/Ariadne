"""The wiring gate: SCIP ingest invariants, checked loudly.

Every SCIP defect found in this codebase succeeded silently. `build_graph` was never on
the onboard path; `resolve_external_to` was gated on an id production never produces;
staleness was computed and never passed; body extents were computed and thrown away;
`local N` ids were never namespaced, so one row fused 4,446 files. In each case the
pipeline reported success.

So the ingest has invariants, and they are checked. A store that violates one is not
"degraded" — it is wrong, and the gate says which.

Synthetic fixtures only: sources ``src1``/``src2``.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.scip_wiring import wiring_report
from library.scip import init_scip_schema


def _symbol(conn, cid, *, source, file, qn, line_start=1, line_end=1):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (cid, source, 'python', file, line_start, line_end, 'Function', 'f', qn, ''),
    )
def _edge(conn, caller, callee, *, edge_type='call', file='a.py', line=1,
          confidence='exact'):
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
        (caller, callee, edge_type, file, line, confidence),
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _healthy(conn):
    """What a correctly wired store looks like: namespaced ids, real extents,
    in-source edges, and the relationship SCIP supplies."""
    a = 'scip-python python src1 0.1 `m`/a().'
    b = 'scip-python python src1 0.1 `m`/b().'
    iface = 'scip-python python src1 0.1 `m`/Proto#'
    _symbol(conn, a, source='src1', file='m.py', qn='m.a', line_start=1, line_end=8)
    _symbol(conn, b, source='src1', file='m.py', qn='m.b', line_start=10, line_end=14)
    _symbol(conn, iface, source='src1', file='m.py', qn='m.Proto',
            line_start=20, line_end=24)
    # a local scoped to its document, in the shape ingest now produces
    _symbol(conn, 'local src1:m.py:1', source='src1',
            file='m.py', qn='local 1', line_start=3, line_end=3)
    _edge(conn, a, b)
    _edge(conn, b, iface, edge_type='implements')
    conn.commit()
    return conn
def test_a_correctly_wired_store_passes_every_check(conn):
    report = wiring_report(_healthy(conn))

    assert report.ok, [c.name for c in report.checks if not c.ok]
    assert {c.name for c in report.checks} == {
        'local_ids_namespaced', 'locals_never_cross_sources',
        'definition_extents_present', 'implements_edges_present',
    }


def test_a_bare_local_id_fails_the_namespacing_check(conn):
    """One `local 1` row is shared by every document that emitted that index."""
    _healthy(conn)
    _symbol(conn, 'local 1', source='src1', file='other.py', qn='local 1')
    conn.commit()

    report = wiring_report(conn)
    check = next(c for c in report.checks if c.name == 'local_ids_namespaced')

    assert not report.ok
    assert not check.ok
    assert check.measured['bare_local_ids'] == 1
def test_a_cross_source_edge_through_a_local_fails(conn):
    """A local binding is private to its document, so no other source can refer to it.

    An edge that crosses sources with a local at one end therefore means two documents
    were given one identity -- the shape that welded 4,446 files together and put 19.5%
    of the live call graph through fused nodes.
    """
    _healthy(conn)
    foreign = 'scip-python python src2 0.1 `n`/z().'
    _symbol(conn, foreign, source='src2', file='n.py', qn='n.z',
            line_start=1, line_end=6)
    _edge(conn, 'local src1:m.py:1', foreign)
    conn.commit()

    report = wiring_report(conn)
    check = next(c for c in report.checks if c.name == 'locals_never_cross_sources')

    assert not report.ok
    assert not check.ok
    assert check.measured['fused_cross_source_edges'] == 1
def test_a_deliberately_resolved_cross_source_edge_is_legitimate(conn):
    """A cross-repository edge is the product of the cross-source graph, not a defect.

    Stage 1's travel index crosses repositories on these edges and ``ariadne callers``
    is documented as a cross-source caller tree. ``_resolve_external`` creates them
    deliberately, matching a dropped reference to a UNIQUE definition in another
    source, and ``persist_all_sources`` switches that on for every loaded source. A
    gate that forbids the result can only be satisfied by turning the feature off.
    """
    _healthy(conn)
    foreign = 'scip-python python src2 0.1 `n`/z().'
    _symbol(conn, foreign, source='src2', file='n.py', qn='n.z',
            line_start=1, line_end=6)
    _edge(conn, 'scip-python python src1 0.1 `m`/a().', foreign,
          confidence='resolved')
    conn.commit()

    report = wiring_report(conn)

    assert report.ok, [c.name for c in report.checks if not c.ok]


def test_a_store_where_no_definition_has_a_body_fails(conn):
    """The live store's state before the extent fix: 0 of 306,473 multi-line."""
    only = 'scip-python python src1 0.1 `m`/a().'
    _symbol(conn, only, source='src1', file='m.py', qn='m.a',
            line_start=5, line_end=5)
    conn.commit()

    report = wiring_report(conn)
    check = next(c for c in report.checks if c.name == 'definition_extents_present')

    assert not check.ok
    assert check.measured['multi_line'] == 0


def test_a_store_with_no_implements_edges_fails(conn):
    """SCIP supplies `is_implementation`; a store without it discarded the relation."""
    a = 'scip-python python src1 0.1 `m`/a().'
    b = 'scip-python python src1 0.1 `m`/b().'
    _symbol(conn, a, source='src1', file='m.py', qn='m.a', line_start=1, line_end=8)
    _symbol(conn, b, source='src1', file='m.py', qn='m.b', line_start=10, line_end=14)
    _edge(conn, a, b)
    conn.commit()

    report = wiring_report(conn)
    check = next(c for c in report.checks if c.name == 'implements_edges_present')

    assert not check.ok
    assert check.measured['implements_edges'] == 0


def test_an_empty_store_is_reported_as_unwired_not_as_healthy(conn):
    """Nothing indexed is not the same as correctly indexed — the silent-success trap."""
    report = wiring_report(conn)

    assert not report.ok
    assert all(not c.ok for c in report.checks)
