"""C6b — SQLAlchemy Layer-2 access binding (design §5.1 Access API, §5.0, §3a,
§6).

SQLAlchemy query call sites name their columns as attribute refs (``User.email``)
or kwargs (``filter_by(email=…)``), so SCIP emits no reference to the column the
recognizer must resolve (§5.0). The engine walks the query chains in the app
source — ``select(...)``, ``Session.query(...)``, ``.where()/.filter()/
.filter_by()/.with_entities()/.values()/.order_by()``, ``insert/update/delete``,
and ``session.add(Model(...))`` — resolves the root model + the attribute/kwarg
names against the Layer-1 schema, and emits role-typed ``data_access`` rows at
each call site's enclosing symbol. Resolved columns -> ``resolved`` (§3a).

Forms that cannot be decoded/resolved are surfaced as gaps, never silently
dropped (§5.0, §5.8): an undeclared field, an unknown model, a call site with no
owning symbol. A chain not rooted at a query constructor, and ``session.add`` of
a non-model / bare variable (needs dataflow), are left alone — not guessed.

End-to-end over the real pipeline: synthesized SCIP + real model & service
source -> persist_schema_symbols (Layer 1) -> persist_data_access_orm (Layer 2).
No injected data_access. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import SQLAlchemyStrategy, persist_schema_symbols
from docgen.orm_bindings.access import persist_data_access_orm
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from library.scip import init_scip_schema

MP = 'scip-python python shop . shop/models/'
SP = 'scip-python python shop . shop/services/'
USER = MP + 'User#'
U_ID, U_EMAIL, U_NAME, U_ORG = (MP + 'User#id.', MP + 'User#email.',
                                MP + 'User#name.', MP + 'User#org_id.')
BY_EMAIL, EMAILS, BUMP, ADD_USER = (SP + 'by_email().', SP + 'emails().',
                                    SP + 'bump().', SP + 'add_user().')
BY_ALIAS = SP + 'by_alias().'

MODELS_SRC = '''\
class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column("email_addr")
    name: Mapped[str] = mapped_column()
    org_id: Mapped[int] = mapped_column()
'''
# 0-indexed model def lines: User 0, id 2, email 3, name 4, org_id 5
MODEL_OCC = (
    _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
    _ScipOccurrence(symbol=U_ID, range=(2, 4, 2, 6), is_definition=True),
    _ScipOccurrence(symbol=U_EMAIL, range=(3, 4, 3, 9), is_definition=True),
    _ScipOccurrence(symbol=U_NAME, range=(4, 4, 4, 8), is_definition=True),
    _ScipOccurrence(symbol=U_ORG, range=(5, 4, 5, 10), is_definition=True),
)

SERVICES_SRC = '''\
from sqlalchemy import select
from shop.models import User as DBUser
def by_email(session, x):
    return session.query(User).filter_by(email=x)
def emails(session):
    return select(User.email).where(User.org_id == 1)
def bump(session):
    update(User).values(name="n").where(User.id == 5)
def add_user(session):
    session.add(User(email="e"))
def by_alias(session, x):
    return session.query(DBUser).filter_by(name=x)
'''


def _models_doc():
    return _ScipDoc(relative_path='shop/models.py', occurrences=MODEL_OCC, symbols=())


def _access(conn):
    return {
        tuple(r) for r in conn.execute(
            'SELECT consumer_symbol_id, schema_symbol_id, role, confidence '
            "FROM data_access WHERE witness = 'orm:sqlalchemy'"
        )
    }


@pytest.fixture
def setup(tmp_path):
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'shop' / 'services.py').write_text(SERVICES_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    # service defs carry BODY-spanning ranges so the call sites resolve to their
    # enclosing (owning) symbol by containment.
    services_doc = _ScipDoc(
        relative_path='shop/services.py',
        occurrences=(
            _ScipOccurrence(symbol=BY_EMAIL, range=(2, 0, 3, 60), is_definition=True),
            _ScipOccurrence(symbol=EMAILS, range=(4, 0, 5, 60), is_definition=True),
            _ScipOccurrence(symbol=BUMP, range=(6, 0, 7, 60), is_definition=True),
            _ScipOccurrence(symbol=ADD_USER, range=(8, 0, 9, 60), is_definition=True),
            _ScipOccurrence(symbol=BY_ALIAS, range=(10, 0, 11, 60), is_definition=True),
        ),
        symbols=(),
    )
    index = ScipIndex(documents=(_models_doc(), services_doc), source_root=tmp_path)
    yield conn, index
    conn.close()


def test_sqlalchemy_query_call_sites_bind_role_typed_access(setup):
    conn, index = setup
    persist_schema_symbols(conn, 'shop', index, strategies=[SQLAlchemyStrategy()])

    persist_data_access_orm(conn, 'shop', index, strategies=[SQLAlchemyStrategy()])

    T = 'data sql shop _._.'
    EMAIL, NAME, ID, ORG = (T + 'users#email_addr', T + 'users#name',
                            T + 'users#id', T + 'users#org_id')
    # attribute (User.email) + kwarg (filter_by(email=)) names resolve to COLUMNS;
    # role per verb; chained .values().where() yields a row per verb; select(M.col)
    # projects; session.add(M(...)) expands to a whole-row write of every column;
    # a model named under an import alias (query(DBUser)) resolves to its columns.
    assert _access(conn) == {
        (BY_EMAIL, EMAIL, 'filter', 'resolved'),    # session.query(User).filter_by(email=)
        (EMAILS, EMAIL, 'project', 'resolved'),      # select(User.email)
        (EMAILS, ORG, 'filter', 'resolved'),         # .where(User.org_id == 1)
        (BUMP, NAME, 'write', 'resolved'),           # update(User).values(name=)
        (BUMP, ID, 'filter', 'resolved'),            # .where(User.id == 5)
        (ADD_USER, ID, 'write', 'resolved'),         # session.add(User(...)) whole-row
        (ADD_USER, EMAIL, 'write', 'resolved'),
        (ADD_USER, NAME, 'write', 'resolved'),
        (ADD_USER, ORG, 'write', 'resolved'),
        # query(DBUser): model named via `from shop.models import User as DBUser`
        # resolves through the shared engine's import-alias map before the schema
        # lookup — not a false 'unknown model' gap (§5.0).
        (BY_ALIAS, NAME, 'filter', 'resolved'),
    }


U2, MAKEQ = SP + 'undeclared().', SP + 'factory_rooted().'
COMPLEX = SP + 'complex_root().'
UNK, ADDVAR, ADDUNK, NOTROOT, EMPTY = (
    SP + 'unknown_model().', SP + 'add_var().', SP + 'add_unknown().',
    SP + 'not_rooted().', SP + 'empty_and_plain().')

EDGE_SRC = '''\
def undeclared(session):
    return select(User.bogus)
def unknown_model(session):
    return session.query(Ghost).filter_by(x=1)
def add_var(session, u):
    session.add(u)
def add_unknown(session):
    session.add(Ghost(x=1))
def not_rooted(session, data):
    return data.where(User.id == 1)
def factory_rooted(session):
    return make_q(User).filter(User.id == 1)
def complex_root(session):
    return select(func(User.id))
def empty_and_plain(session):
    select()
    helper()
select(User.email)
session.add(User(name="n"))
'''


def test_sqlalchemy_access_edge_and_error_paths_are_surfaced(tmp_path):
    # undeclared field, unknown model, no-owning module call sites -> gaps;
    # add(var)/add(non-model), a chain not rooted at a query ctor, select(),
    # a plain call -> left alone (never guessed). Every edge/error path §5.1
    # recognition handles.
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'shop' / 'edge.py').write_text(EDGE_SRC)
    (tmp_path / 'shop' / 'broken.py').write_text('not valid python {{{')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    edge_funcs = [U2, UNK, ADDVAR, ADDUNK, NOTROOT, MAKEQ, COMPLEX]
    occ = [
        _ScipOccurrence(symbol=sym, range=(2 * i, 0, 2 * i + 1, 60), is_definition=True)
        for i, sym in enumerate(edge_funcs)
    ]
    occ.append(_ScipOccurrence(symbol=EMPTY, range=(14, 0, 16, 60), is_definition=True))
    edge_doc = _ScipDoc(relative_path='shop/edge.py', occurrences=tuple(occ), symbols=())
    broken_doc = _ScipDoc(relative_path='shop/broken.py', occurrences=(), symbols=())
    index = ScipIndex(documents=(_models_doc(), edge_doc, broken_doc), source_root=tmp_path)

    persist_schema_symbols(conn, 's', index, strategies=[SQLAlchemyStrategy()])
    result = persist_data_access_orm(conn, 's', index, strategies=[SQLAlchemyStrategy()])
    gaps = result.gaps

    # every undecodable/unresolved form is SURFACED, never silently dropped
    assert any('bogus' in g and 'undeclared' in g for g in gaps)        # select(User.bogus)
    assert any('Ghost' in g and 'unknown model' in g for g in gaps)     # query(Ghost)
    assert sum('no owning symbol' in g for g in gaps) >= 2              # module select + add
    # add(var)/add(Ghost(...)), data.where(...), make_q(User).filter(...), select(),
    # helper() are NOT guessed — no resolved rows come out of the edge fixture
    assert _access(conn) == set()
    conn.close()
