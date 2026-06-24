"""C6c — Slick (Scala) Layer-1 structural binding, the §5.4 extraction
(design §5.4, §5.0.1, §3a, §6).

Slick declares the schema with explicit string literals: a table is a class
``extends Table[Row](tag, "name")`` (optionally ``(tag, Some("schema"), "name")``),
each column a ``def c = column[T]("name")``. Both names are explicit literals —
no naming rule to derive — so every binding is ``exact``. The shared engine
discovers the Table subclasses from the SCIP index (anchored on the class /
column def occurrences, §5.0.1 #1), reads the literal names from the Scala source
with ast-grep, and emits ``schema_symbols``; the projection ``def *`` and other
non-``column`` defs are skipped. Un-anchored / un-named definitions are surfaced
as gaps, never silently bound (§5.0, §5.0.1 #5).

End-to-end over the real engine: synthesized SCIP + real Scala source ->
persist_schema_symbols -> add_data_layer. No injected schema_symbols. Synthetic
fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import SlickStrategy, persist_schema_symbols
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
)
from library.scip import init_scip_schema

P = 'scala shop Tables.scala '
USERS = P + 'Users#'
U_ID = P + 'Users#id().'
U_EMAIL = P + 'Users#email().'
ORGS = P + 'Orgs#'
O_SLUG = P + 'Orgs#slug().'

# 0-indexed def occurrence lines must match MODEL_SRC below.
DEFS = {USERS: 0, U_ID: 1, U_EMAIL: 2, ORGS: 5, O_SLUG: 6}

MODEL_SRC = '''\
class Users(tag: Tag) extends Table[UserRow](tag, "users") {
  def id    = column[Int]("id", O.PrimaryKey)
  def email = column[String]("email_addr")
  def * = (id, email)
}
class Orgs(tag: Tag) extends Table[OrgRow](tag, Some("acct"), "orgs") {
  def slug = column[String]("slug")
}
'''


@pytest.fixture
def setup(tmp_path):
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'Tables.scala').write_text(MODEL_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym, line in DEFS.items():
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'shop', 'jvm', 'shop/Tables.scala', line + 1, line + 1,
             '', sym, sym, None),
        )
    doc = _ScipDoc(
        relative_path='shop/Tables.scala',
        occurrences=tuple(
            _ScipOccurrence(symbol=sym, range=(line, 6, line, 12), is_definition=True)
            for sym, line in DEFS.items()
        ),
        symbols=tuple(
            _ScipSymbol(symbol=sym, kind='', display_name=sym) for sym in DEFS
        ),
    )
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    yield conn, index
    conn.close()


def _schema_rows(conn):
    return {
        r[0]: r[1:] for r in conn.execute(
            'SELECT canonical_id, node_type, column_name, producer_symbol_id, '
            'references_id, confidence FROM schema_symbols'
        )
    }


def test_slick_full_structural_binding(setup):
    conn, index = setup

    persist_schema_symbols(conn, 'shop', index, strategies=[SlickStrategy()])
    rows = _schema_rows(conn)

    T = 'data sql shop _._.'
    # Table[...](tag, "users") -> exact table; column[T]("name") -> exact column.
    # Explicit string literals throughout, so every binding is exact (§5.4).
    assert rows[T + 'users'] == ('table', None, USERS, None, 'exact')
    assert rows[T + 'users#id'] == ('column', 'id', U_ID, None, 'exact')
    assert rows[T + 'users#email_addr'] == ('column', 'email_addr', U_EMAIL, None, 'exact')
    # (tag, Some("acct"), "orgs") -> table name is the bare string literal "orgs"
    assert rows[T + 'orgs'] == ('table', None, ORGS, None, 'exact')
    assert rows[T + 'orgs#slug'] == ('column', 'slug', O_SLUG, None, 'exact')
    # the projection `def * = (id, email)` is not a column[...] def -> not bound
    assert (T + 'users#*') not in rows
    assert len([k for k in rows if k.startswith(T + 'users#')]) == 2

    # add_data_layer: exact asserts at the default floor; maps_to reaches the col
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)
    assert (T + 'users#email_addr') in graph._symbols
    assert (U_EMAIL, 'maps_to') in {
        (e.caller.canonical_id, e.edge_type) for e in graph.callers_of(T + 'users#email_addr')
    }


EP = 'scala s edge.scala '
E_NONAME, E_EMPTY, E_THINGS, E_NAME = (
    EP + 'NoName#', EP + 'Empty#', EP + 'Things#', EP + 'Things#name().')
# Ghost (L3) and Things.ua (L12) are deliberately UN-anchored.
EDGE_DEFS = {E_NONAME: 4, E_EMPTY: 7, E_THINGS: 8, E_NAME: 9}

EDGE_SRC = '''\
class Plain {}
class Helper extends Object {}
class Other(tag: Tag) extends Base[X](tag) {}
class Ghost(tag: Tag) extends Table[G](tag, "ghosts")
class NoName(tag: Tag) extends Table[N] {
  def nx = column[Int]("nx")
}
class Empty(tag: Tag) extends Table[E](tag, "empty")
class Things(tag: Tag) extends Table[T](tag, "things") {
  def name = column[String]("thing_name")
  def p = plain("x")
  def bad = column[Int]()
  def ua = column[Int]("ua")
  def * = (name)
}
'''


def test_slick_edge_and_error_paths_are_surfaced(tmp_path):
    # non-table classes (no extends / extends a non-generic / a non-Table generic)
    # are not tables; a no-args Table -> no name literal; an un-anchored table /
    # column, a column with no name literal, a non-`column` def, and `def *` are
    # each handled — surfaced as a gap or skipped, never silently bound.
    (tmp_path / 's').mkdir()
    (tmp_path / 's' / 'edge.scala').write_text(EDGE_SRC)
    (tmp_path / 's' / 'other.py').write_text('x = 1\n')  # non-Scala -> skipped
    # 's/missing.scala' is referenced by the index but never written -> unreadable
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    occ = tuple(
        _ScipOccurrence(symbol=sym, range=(line, 6, line, 12), is_definition=True)
        for sym, line in EDGE_DEFS.items()
    )
    # non-Scala + unreadable docs FIRST so detect() exercises its skip path
    index = ScipIndex(
        documents=(
            _ScipDoc(relative_path='s/other.py', occurrences=(), symbols=()),
            _ScipDoc(relative_path='s/missing.scala', occurrences=(), symbols=()),
            _ScipDoc(relative_path='s/edge.scala', occurrences=occ, symbols=()),
        ),
        source_root=tmp_path,
    )

    result = persist_schema_symbols(conn, 's', index, strategies=[SlickStrategy()])
    rows = _schema_rows(conn)
    T = 'data sql s _._.'

    # only the anchored, named tables bind; Empty has no body -> zero columns
    tables = {r[0] for r in conn.execute(
        "SELECT canonical_id FROM schema_symbols WHERE node_type = 'table'")}
    assert tables == {T + 'things', T + 'empty'}
    assert rows[T + 'things#thing_name'] == ('column', 'thing_name', E_NAME, None, 'exact')
    # un-anchored table/column, no-name table/column -> NOT bound
    assert (T + 'ghosts') not in rows
    assert (T + 'things#ua') not in rows        # column un-anchored
    # every undecodable/unresolved form is surfaced as a gap, never silently dropped
    assert any('Ghost' in g and 'anchor' in g for g in result.gaps)
    assert any('NoName' in g and 'name literal' in g for g in result.gaps)
    assert any('bad' in g and 'name literal' in g for g in result.gaps)
    assert any('ua' in g and 'anchor' in g for g in result.gaps)
    conn.close()

    # detect() is False on a source with no Slick tables
    (tmp_path / 's' / 'plain.scala').write_text('class Plain {}\n')
    empty_index = ScipIndex(
        documents=(_ScipDoc(relative_path='s/plain.scala', occurrences=(), symbols=()),),
        source_root=tmp_path,
    )
    assert SlickStrategy().detect(empty_index, tmp_path) is False
