"""C6c — Slick (Scala) Layer-2 access binding (design §5.4 Access API, §5.0, §3a).

Slick is the Layer-1-strong case: its DSL references columns as real ``def``
symbols (``users.filter(_.email === …)``, ``users.map(_.email)``), so the
recognizer's job is to add the *role*. It resolves the query receiver to its
Table class (a ``val x = TableQuery[Y]`` binding, or inline ``TableQuery[Y]``,
walked back through chained verbs), reads the ``_.col`` lambda fields against the
Layer-1 schema, and emits role-typed ``data_access`` rows: ``filter`` -> filter,
``map`` -> project (or write when it feeds ``.update``), ``sortBy`` -> order,
``+=`` -> a whole-row write of every column. Resolved columns bind ``resolved``
(§3a); receivers that don't resolve are left alone, never guessed.

End-to-end over the real engine: synthesized SCIP + real Scala model & DAO
source -> persist_data_access_orm. No injected data_access. Synthetic fixtures.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import SlickStrategy
from docgen.orm_bindings.access import persist_data_access_orm
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from library.scip import init_scip_schema

MP = 'scala shop Tables.scala '
USERS, U_ID, U_EMAIL, U_NAME = (
    MP + 'Users#', MP + 'Users#id().', MP + 'Users#email().', MP + 'Users#name().')
SP = 'scala shop Dao.scala '
BYEMAIL, EMAILS, SORTED, ADD, SETNAME, INLINE = (
    SP + 'byEmail().', SP + 'emails().', SP + 'sorted().', SP + 'add().',
    SP + 'setName().', SP + 'inline().')
ALIASED = SP + 'aliased().'

MODEL_SRC = '''\
class Users(tag: Tag) extends Table[UserRow](tag, "users") {
  def id    = column[Int]("id")
  def email = column[String]("email_addr")
  def name  = column[String]("name")
}
'''
MODEL_OCC = tuple(
    _ScipOccurrence(symbol=s, range=(ln, 6, ln, 12), is_definition=True)
    for s, ln in ((USERS, 0), (U_ID, 1), (U_EMAIL, 2), (U_NAME, 3))
)

DAO_SRC = '''\
import shop.tables.{Users => U}
import shop.tables.{Secret => _}
object Dao {
  val users = TableQuery[Users]
  def byEmail(x: String) = users.filter(_.email === x)
  def emails = users.map(_.email)
  def sorted = users.sortBy(_.name)
  def add(u: UserRow) = users += u
  def setName(x: Int) = users.filter(_.id === x).map(_.name).update("n")
  def inline = TableQuery[Users].filter(_.id === 1)
  def aliased(x: String) = TableQuery[U].filter(_.email === x)
}
'''
DAO_OCC = tuple(
    _ScipOccurrence(symbol=s, range=(ln, 0, ln, 80), is_definition=True)
    for s, ln in ((BYEMAIL, 4), (EMAILS, 5), (SORTED, 6), (ADD, 7),
                  (SETNAME, 8), (INLINE, 9), (ALIASED, 10))
)


def _access(conn):
    return {
        tuple(r) for r in conn.execute(
            'SELECT consumer_symbol_id, schema_symbol_id, role, confidence '
            "FROM data_access WHERE witness = 'orm:slick'"
        )
    }


def _index(tmp_path):
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'Tables.scala').write_text(MODEL_SRC)
    (tmp_path / 'shop' / 'Dao.scala').write_text(DAO_SRC)
    return ScipIndex(
        documents=(
            _ScipDoc(relative_path='shop/Tables.scala', occurrences=MODEL_OCC, symbols=()),
            _ScipDoc(relative_path='shop/Dao.scala', occurrences=DAO_OCC, symbols=()),
        ),
        source_root=tmp_path,
    )


def test_slick_query_call_sites_bind_role_typed_access(tmp_path):
    index = _index(tmp_path)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)

    persist_data_access_orm(conn, 'shop', index, strategies=[SlickStrategy()])

    T = 'data sql shop _._.'
    EMAIL, NAME, ID = T + 'users#email_addr', T + 'users#name', T + 'users#id'
    # _.col lambda fields resolve to COLUMNS; role per verb; the receiver resolves
    # via the `val users = TableQuery[Users]` binding (and inline TableQuery[Users]);
    # map feeding .update is a write; += is a whole-row write of every column; a
    # Table class named under a Scala import rename (TableQuery[U]) resolves too.
    assert _access(conn) == {
        (BYEMAIL, EMAIL, 'filter', 'resolved'),    # users.filter(_.email === x)
        (EMAILS, EMAIL, 'project', 'resolved'),     # users.map(_.email)
        (SORTED, NAME, 'order', 'resolved'),        # users.sortBy(_.name)
        (ADD, ID, 'write', 'resolved'),             # users += u  (whole-row)
        (ADD, EMAIL, 'write', 'resolved'),
        (ADD, NAME, 'write', 'resolved'),
        (SETNAME, ID, 'filter', 'resolved'),        # .filter(_.id === x)
        (SETNAME, NAME, 'write', 'resolved'),        # .map(_.name).update(...)
        (INLINE, ID, 'filter', 'resolved'),         # TableQuery[Users].filter(_.id === 1)
        # TableQuery[U]: class under import rename `{Users => U}` resolves via
        # _scala_import_aliases; the `{Secret => _}` hide selector binds no alias.
        (ALIASED, EMAIL, 'filter', 'resolved'),
    }


SE = 'scala s EdgeDao.scala '
UNDECLARED, MIXED = SE + 'undeclared().', SE + 'mixedField().'

EDGE_DAO_SRC = '''\
object EdgeDao {
  val users = TableQuery[Users]
  val notQuery = 5
  def unresolved = unknownVar.filter(_.id === 1)
  def chained = obj.users.filter(_.id === 1)
  def inlineOther = OtherQuery[Z].filter(_.id === 1)
  def noFields = users.filter(externalCond)
  def undeclared = users.map(_.bogus)
  def mixedField = users.filter(_.id === obj.field)
  def insertBad(u: R) = unknownVar += u
  def noOwn = users.map(_.id)
}
'''


def test_slick_access_edge_and_error_paths_are_surfaced(tmp_path):
    # a non-TableQuery val, an unresolved receiver, an inline non-TableQuery
    # generic, a chained field-expression receiver, a verb naming no column, an
    # undeclared field, a += with an unresolved receiver, and a resolved query
    # with no owning symbol — each surfaced as a gap or left alone, never guessed.
    (tmp_path / 's').mkdir()
    (tmp_path / 's' / 'Tables.scala').write_text(MODEL_SRC)
    (tmp_path / 's' / 'EdgeDao.scala').write_text(EDGE_DAO_SRC)
    (tmp_path / 's' / 'other.py').write_text('x = 1\n')  # non-Scala -> skipped
    # 's/missing.scala' referenced but never written -> unreadable -> skipped
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    # only undeclared (L7) + mixedField (L8) get owning; noOwn (L10) deliberately
    # has none, so a resolved query surfaces a no-owning gap.
    edge_occ = tuple(
        _ScipOccurrence(symbol=s, range=(ln, 0, ln, 80), is_definition=True)
        for s, ln in ((UNDECLARED, 7), (MIXED, 8))
    )
    index = ScipIndex(
        documents=(
            _ScipDoc(relative_path='s/other.py', occurrences=(), symbols=()),
            _ScipDoc(relative_path='s/missing.scala', occurrences=(), symbols=()),
            _ScipDoc(relative_path='s/Tables.scala', occurrences=MODEL_OCC, symbols=()),
            _ScipDoc(relative_path='s/EdgeDao.scala', occurrences=edge_occ, symbols=()),
        ),
        source_root=tmp_path,
    )

    result = persist_data_access_orm(conn, 's', index, strategies=[SlickStrategy()])

    T = 'data sql s _._.'
    # only the resolvable, anchored, declared-field query yields a row;
    # the non-wildcard `obj.field` is not a column ref, so only `id` binds
    assert _access(conn) == {(MIXED, T + 'users#id', 'filter', 'resolved')}
    # undecodable/unresolved forms surface as gaps, never silently dropped
    assert any('bogus' in g and 'undeclared' in g for g in result.gaps)
    assert any('owning' in g for g in result.gaps)        # noOwn: resolved but unanchored
