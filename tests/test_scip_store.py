"""Writing the rebuilt graph into the store, and reading it back.

This is the step the wiring gate inspects, so its correctness is what makes the gate mean
anything. Two failures in this codebase's history happened right here:

* a **global** ``DELETE`` while re-inserting one source, so a multi-source store kept only
  the last build (fixed once for ``doc_graph`` in ``92b1d40``; the same shape applies here);
* ``scip_edges`` has no ``source_name``, so clearing one source's edges means joining
  through ``scip_symbols`` — which has to happen **before** those symbols are deleted, or
  the join finds nothing and the edges are orphaned.

Synthetic fixtures only: sources ``src1``/``src2``.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.scip_graph import build_rows
from docgen.scip_index import ScipDocument, ScipIndex, ScipOccurrence, ScipRelationship, ScipSymbolInfo
from docgen.scip_store import load_rows, save_rows
from docgen.scip_wiring import wiring_report
from library.scip import init_scip_schema

RUN = 'scip-python python {src} 0.1 `m`/run().'
HELPER = 'scip-python python {src} 0.1 `m`/helper().'
IMPL = 'scip-python python {src} 0.1 `m`/Impl#'
PROTO = 'scip-python python {src} 0.1 `m`/Proto#'


def _occ(symbol, line, *, definition=False, enclosing=None):
    return ScipOccurrence(symbol=symbol, range=(line, 4, line, 12),
                          is_definition=definition, enclosing_range=enclosing or ())


def _rows_for(source):
    """A small but complete source: a call, an implements relation, and a local."""
    run, helper = RUN.format(src=source), HELPER.format(src=source)
    impl, proto = IMPL.format(src=source), PROTO.format(src=source)
    doc = ScipDocument(f'{source}/m.py', occurrences=(
        _occ(run, 4, definition=True, enclosing=(4, 0, 20, 0)),
        _occ(helper, 30, definition=True, enclosing=(30, 0, 34, 0)),
        _occ(impl, 40, definition=True, enclosing=(40, 0, 48, 0)),
        _occ(proto, 50, definition=True, enclosing=(50, 0, 54, 0)),
        _occ('local 1', 6, definition=True),
        _occ(helper, 10),
    ), symbols=(
        ScipSymbolInfo(symbol=impl, kind='Class', display_name='Impl',
                       relationships=(ScipRelationship(symbol=proto,
                                                       is_implementation=True),)),
    ))
    index = ScipIndex(documents=(doc,)).scoped_to(source)
    return build_rows(index, source_name=source, language='python')


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def test_symbols_and_edges_round_trip(conn):
    save_rows(conn, _rows_for('src1'), source_name='src1')
    back = load_rows(conn)

    assert RUN.format(src='src1') in back.symbols
    assert [(e.caller.canonical_id, e.callee.canonical_id, e.edge_type)
            for e in back.edges if e.edge_type == 'call'] == [
        (RUN.format(src='src1'), HELPER.format(src='src1'), 'call')]


def test_body_extents_survive_the_round_trip(conn):
    """Without this a cited hop cannot be quoted — the store held 0 of 306,473."""
    save_rows(conn, _rows_for('src1'), source_name='src1')
    back = load_rows(conn)

    run = back.symbols[RUN.format(src='src1')]
    assert (run.line_start, run.line_end) == (5, 21)


def test_implements_edges_reach_the_store(conn):
    """The relation the store never held."""
    save_rows(conn, _rows_for('src1'), source_name='src1')

    kinds = dict(conn.execute(
        'SELECT edge_type, COUNT(*) FROM scip_edges GROUP BY 1').fetchall())
    assert kinds.get('implements') == 1


def test_resaving_one_source_leaves_another_untouched(conn):
    """`92b1d40` in this table's shape: a global delete kept only the last build."""
    save_rows(conn, _rows_for('src1'), source_name='src1')
    save_rows(conn, _rows_for('src2'), source_name='src2')
    save_rows(conn, _rows_for('src1'), source_name='src1')

    per_source = dict(conn.execute(
        'SELECT source_name, COUNT(*) FROM scip_symbols GROUP BY 1').fetchall())
    assert set(per_source) == {'src1', 'src2'}
    assert per_source['src1'] == per_source['src2']


def test_resaving_replaces_rather_than_duplicates(conn):
    save_rows(conn, _rows_for('src1'), source_name='src1')
    first = conn.execute('SELECT COUNT(*) FROM scip_edges').fetchone()[0]
    save_rows(conn, _rows_for('src1'), source_name='src1')

    assert conn.execute('SELECT COUNT(*) FROM scip_edges').fetchone()[0] == first


def test_a_sources_edges_are_cleared_before_its_symbols(conn):
    """Edges carry no source, so clearing them means joining through the symbols.

    Do it in the wrong order and the join finds nothing: the symbols are already gone and
    their edges are left behind, pointing at rows that no longer exist.
    """
    save_rows(conn, _rows_for('src1'), source_name='src1')
    save_rows(conn, _rows_for('src2'), source_name='src2')
    save_rows(conn, _rows_for('src1'), source_name='src1')

    orphans = conn.execute('''
        SELECT COUNT(*) FROM scip_edges e
        LEFT JOIN scip_symbols c ON c.canonical_id = e.caller_canonical_id
        LEFT JOIN scip_symbols d ON d.canonical_id = e.callee_canonical_id
        WHERE c.canonical_id IS NULL OR d.canonical_id IS NULL
    ''').fetchone()[0]
    assert orphans == 0


def test_a_store_built_this_way_passes_the_wiring_gate(conn):
    """The whole point: the gate reported the old store BROKEN on all four checks."""
    save_rows(conn, _rows_for('src1'), source_name='src1')
    save_rows(conn, _rows_for('src2'), source_name='src2')

    report = wiring_report(conn)

    assert report.ok, [(c.name, c.detail) for c in report.failures()]
