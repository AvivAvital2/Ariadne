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
from docgen import scip_store
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
def _ownership_rows(source):
    owner_type = f"semanticdb maven {source} . pkg/Owner#"
    owner_term = f"semanticdb maven {source} . pkg/Owner."
    member = f"semanticdb maven {source} . pkg/Owner#run()."
    document = ScipDocument(f"{source}/Owner.scala", occurrences=(
        _occ(owner_type, 1, definition=True, enclosing=(1, 0, 10, 0)),
        _occ(member, 4, definition=True, enclosing=(4, 0, 8, 0)),
        _occ(owner_term, 12, definition=True, enclosing=(12, 0, 16, 0)),
    ))
    index = ScipIndex(documents=(document,)).scoped_to(source)
    return build_rows(index, source_name=source, language="scala"), (
        owner_type, owner_term, member)


def test_canonical_contains_edges_reach_the_store(conn):
    rows, (owner_type, _owner_term, member) = _ownership_rows("src1")

    save_rows(conn, rows, source_name="src1")

    assert conn.execute(
        "SELECT caller_canonical_id,callee_canonical_id FROM scip_edges "
        "WHERE edge_type='contains'"
    ).fetchall() == [(owner_type, member)]


def test_canonical_ownership_replaces_a_stale_companion_edge(conn):
    rows, (owner_type, owner_term, member) = _ownership_rows("src1")
    save_rows(conn, rows, source_name="src1")
    conn.execute(
        "INSERT OR REPLACE INTO scip_edges VALUES (?,?,?,?,?,?)",
        (owner_term, member, "contains", "src1/Owner.scala", 4, "stale"))

    save_rows(conn, rows, source_name="src1")

    assert conn.execute(
        "SELECT caller_canonical_id,callee_canonical_id FROM scip_edges "
        "WHERE edge_type='contains'"
    ).fetchall() == [(owner_type, member)]


def _insert_spool_symbol(
    conn, canonical_id, source_name, *, file="Owner.scala", line=1,
    kind="Method", display_name="symbol",
):
    conn.execute(
        "INSERT INTO scip_symbols (canonical_id, source_name, language, file, "
        "line_start, line_end, kind, display_name, qualified_name, "
        "parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (
            canonical_id, source_name, "scala", file, line, line + 1, kind,
            display_name, display_name, None,
        ),
    )


def test_canonical_ownership_backfill_repairs_a_spool_without_reindexing(conn):
    source = "spool:fixture"
    package = "semanticdb maven fixture . pkg/"
    owner_type = package + "Owner#"
    owner_term = package + "Owner."
    member = owner_type + "run()."
    for symbol, line, kind in (
        (package, 1, "Package"),
        (owner_type, 2, "Class"),
        (owner_term, 12, "Object"),
        (member, 4, "Method"),
    ):
        _insert_spool_symbol(conn, symbol, source, line=line, kind=kind)
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        (owner_term, member, "contains", "Owner.scala", 4, "stale"),
    )

    result = scip_store.backfill_canonical_ownership(conn, source)

    assert result.scanned_symbols == 4
    assert result.candidate_edges == 1
    assert result.removed_edges == 1
    assert result.inserted_edges == 1
    assert conn.execute(
        "SELECT caller_canonical_id, callee_canonical_id, edge_type, "
        "file, line, confidence FROM scip_edges"
    ).fetchall() == [
        (owner_type, member, "contains", "Owner.scala", 4, "exact"),
    ]

    repeated = scip_store.backfill_canonical_ownership(conn, source)
    assert repeated.removed_edges == 0
    assert repeated.inserted_edges == 0


def test_canonical_ownership_backfill_is_source_scoped(conn):
    first_source = "spool:first"
    second_source = "spool:second"
    first_owner = "semanticdb maven first . pkg/Owner#"
    first_member = first_owner + "run()."
    second_owner = "semanticdb maven second . pkg/Owner#"
    second_member = second_owner + "run()."
    for symbol, source, line, kind in (
        (first_owner, first_source, 1, "Class"),
        (first_member, first_source, 2, "Method"),
        (second_owner, second_source, 1, "Class"),
        (second_member, second_source, 2, "Method"),
    ):
        _insert_spool_symbol(conn, symbol, source, line=line, kind=kind)
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        (second_owner, second_member, "contains", "Owner.scala", 2, "kept"),
    )

    scip_store.backfill_canonical_ownership(conn, first_source)

    assert conn.execute(
        "SELECT caller_canonical_id, callee_canonical_id, confidence "
        "FROM scip_edges ORDER BY caller_canonical_id"
    ).fetchall() == [
        (first_owner, first_member, "exact"),
        (second_owner, second_member, "kept"),
    ]


def test_canonical_ownership_backfill_rejects_an_unbounded_batch(conn):
    with pytest.raises(ValueError, match="batch_size must be positive"):
        scip_store.backfill_canonical_ownership(conn, "spool:fixture", batch_size=0)


def test_canonical_ownership_backfill_rolls_back_a_partial_repair(conn):
    source = "spool:fixture"
    owner = "semanticdb maven fixture . pkg/Owner#"
    member = owner + "run()."
    _insert_spool_symbol(conn, owner, source, line=1, kind="Class")
    _insert_spool_symbol(conn, member, source, line=2, kind="Method")

    class FailingConnection:
        def execute(self, sql, parameters=()):
            if sql.startswith("INSERT OR IGNORE INTO scip_edges"):
                raise RuntimeError("injected write failure")
            return conn.execute(sql, parameters)

        def executemany(self, sql, rows):
            return conn.executemany(sql, rows)

    with pytest.raises(RuntimeError, match="injected write failure"):
        scip_store.backfill_canonical_ownership(FailingConnection(), source)

    assert conn.execute("SELECT * FROM scip_edges").fetchall() == []
    assert conn.execute(
        "SELECT name FROM sqlite_temp_master "
        "WHERE name = '_scip_canonical_ownership'"
    ).fetchall() == []


def _insert_colliding_edge_types(conn):
    edge = ("owner", "member", "Owner.scala", 4)
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        (edge[0], edge[1], "call", edge[2], edge[3], "exact"),
    )
    conn.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        (edge[0], edge[1], "contains", edge[2], edge[3], "exact"),
    )


def test_scip_edge_identity_includes_edge_type(conn):
    _insert_colliding_edge_types(conn)

    assert conn.execute(
        "SELECT edge_type FROM scip_edges ORDER BY edge_type"
    ).fetchall() == [("call",), ("contains",)]


def test_scip_schema_migrates_the_legacy_edge_primary_key():
    legacy = sqlite3.connect(":memory:")
    legacy.execute(
        "CREATE TABLE scip_edges ("
        "caller_canonical_id TEXT NOT NULL, "
        "callee_canonical_id TEXT NOT NULL, "
        "edge_type TEXT NOT NULL, file TEXT NOT NULL, "
        "line INTEGER NOT NULL, confidence TEXT NOT NULL, "
        "PRIMARY KEY (caller_canonical_id, callee_canonical_id, file, line))"
    )
    legacy.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("owner", "member", "call", "Owner.scala", 4, "exact"),
    )

    init_scip_schema(legacy)
    legacy.execute(
        "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
        ("owner", "member", "contains", "Owner.scala", 4, "exact"),
    )

    primary_key = [
        row[1] for row in sorted(
            (
                row for row in legacy.execute("PRAGMA table_info(scip_edges)")
                if row[5]
            ),
            key=lambda row: row[5],
        )
    ]
    assert primary_key == [
        "caller_canonical_id", "callee_canonical_id", "edge_type", "file", "line",
    ]
    assert legacy.execute("SELECT COUNT(*) FROM scip_edges").fetchone()[0] == 2
    legacy.close()


def test_scip_schema_refuses_an_unknown_edge_primary_key():
    unknown = sqlite3.connect(":memory:")
    unknown.execute(
        "CREATE TABLE scip_edges ("
        "caller_canonical_id TEXT NOT NULL PRIMARY KEY, "
        "callee_canonical_id TEXT NOT NULL, edge_type TEXT NOT NULL, "
        "file TEXT NOT NULL, line INTEGER NOT NULL, confidence TEXT NOT NULL)"
    )

    with pytest.raises(RuntimeError, match="unsupported scip_edges primary key"):
        init_scip_schema(unknown)

    unknown.close()
