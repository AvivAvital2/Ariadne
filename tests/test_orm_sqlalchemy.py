"""C6a — SQLAlchemy Layer-1 structural binding, the §5.1 extraction
(design §5.1, §5.0.1, §3a, §6).

The shared engine discovers SQLAlchemy declarative models from the SCIP index —
anchored on the class/attribute definition symbols (§5.0.1 #1), metadata read
from the model source — and binds, per §5.1:

  table  = __tablename__ literal (exact)
  column = first positional string to mapped_column/Column (exact)
           else the attribute name (derived)
           FK from ForeignKey("table.col") -> references_id -> target table

producer_symbol_id is the model class / attribute symbol; unresolved bindings
(no SCIP anchor, unparsable source, FK target not a discovered table) are
RECORDED as surfaced gaps, never silently dropped (§5.0, §5.0.1 #5).
add_data_layer then asserts exact at the default floor and holds derived below
it (§3a/§6).

End-to-end over the real engine: synthesized SCIP + real source ->
persist_schema_symbols -> add_data_layer. No injected schema_symbols. Synthetic
fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import SQLAlchemyStrategy, persist_schema_symbols
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
)
from library.scip import init_scip_schema

# package 'shop', file shop/models.py
P = 'scip-python python shop . shop/'
ORG = P + 'Org#'
ORG_ID = P + 'Org#id.'
ORG_NAME = P + 'Org#name.'
USER = P + 'User#'
USER_ID = P + 'User#id.'
USER_EMAIL = P + 'User#email.'
USER_ORG = P + 'User#org_id.'
USER_EXTERNAL = P + 'User#external_id.'

# 0-indexed definition lines must match MODEL_SRC below.
DEFS = {
    ORG: 0, ORG_ID: 2, ORG_NAME: 3,
    USER: 5, USER_ID: 7, USER_EMAIL: 8, USER_ORG: 9, USER_EXTERNAL: 10,
}

MODEL_SRC = '''\
class Org(Base):
    __tablename__ = "orgs"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column()

class User(Base):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column("email_addr")
    org_id: Mapped[int] = mapped_column(ForeignKey("orgs.id"))
    external_id: Mapped[int] = mapped_column(ForeignKey("external.id"))
'''


@pytest.fixture
def setup(tmp_path):
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(MODEL_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym, line in DEFS.items():
        display = sym.rstrip('#.').rsplit('/', 1)[-1].split('#')[-1] or sym
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'shop', 'python', 'shop/models.py', line + 1, line + 1,
             '', display, display, None),
        )
    doc = _ScipDoc(
        relative_path='shop/models.py',
        occurrences=tuple(
            _ScipOccurrence(symbol=sym, range=(line, 4, line, 8), is_definition=True)
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


def test_sqlalchemy_full_structural_binding(setup):
    conn, index = setup

    result = persist_schema_symbols(conn, 'shop', index, strategies=[SQLAlchemyStrategy()])
    rows = _schema_rows(conn)

    T = 'data sql shop _._.'
    # __tablename__ literal -> exact table
    assert rows[T + 'orgs'] == ('table', None, ORG, None, 'exact')
    assert rows[T + 'users'] == ('table', None, USER, None, 'exact')
    # first positional string to mapped_column -> exact column name override
    assert rows[T + 'users#email_addr'] == ('column', 'email_addr', USER_EMAIL, None, 'exact')
    # no positional string -> derived attribute-name column
    assert rows[T + 'users#id'] == ('column', 'id', USER_ID, None, 'derived')
    assert rows[T + 'orgs#id'] == ('column', 'id', ORG_ID, None, 'derived')
    assert rows[T + 'orgs#name'] == ('column', 'name', ORG_NAME, None, 'derived')
    # FK -> derived attr-name column + references_id to the resolved target table
    assert rows[T + 'users#org_id'] == ('column', 'org_id', USER_ORG, T + 'orgs', 'derived')

    # unresolved FK target (no model has __tablename__ "external") -> a surfaced
    # gap, references_id left unbound — never a guessed link.
    assert rows[T + 'users#external_id'][3] is None  # references_id
    assert any('external' in g for g in result.gaps)

    # add_data_layer: exact asserts at the default floor; derived held below
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)
    assert (T + 'users#email_addr') in graph._symbols      # exact -> asserted
    assert (T + 'users#id') not in graph._symbols          # derived -> gap
    assert (USER_EMAIL, 'maps_to') in {
        (e.caller.canonical_id, e.edge_type) for e in graph.callers_of(T + 'users#email_addr')
    }


EP = 'scip-python python s . edge_models/'
E_THING = EP + 'Thing#'
E_NAME = EP + 'Thing#name.'
E_REL = EP + 'Thing#rel.'
E_SYMREF = EP + 'Thing#symref.'
E_TWEAKED = EP + 'Tweaked#'
# Ghost (line 3) and Thing.missing (line 12) are deliberately UN-anchored.
EDGE_DEFS = {E_THING: 6, E_NAME: 9, E_REL: 13, E_SYMREF: 14, E_TWEAKED: 15}

EDGE_SRC = '''\
NOT_A_CLASS = 1
class Helper:
    pass
class Ghost(Base):
    __tablename__ = "ghosts"
    gid = mapped_column(primary_key=True)
class Thing(Base):
    """Things table."""
    __tablename__ = "things"
    name = Column("thing_name", String)
    plain = 5
    typed: int = 5
    missing = mapped_column()
    rel = mapped_column(ForeignKey("things.id"))
    symref = mapped_column(ForeignKey(Other.id))
class Tweaked(Base):
    __tablename__ = SOME_CONST
    x = mapped_column()
class Plain:
    y = mapped_column()
'''


def test_sqlalchemy_edge_and_error_paths_are_surfaced(tmp_path):
    # a non-class node, a non-model class, a docstring, plain/typed non-column
    # assignments, an un-anchored model/field, a Column() positional override, a
    # self-referential string FK, a symbol-form FK (no string target), and a
    # non-literal __tablename__ — every edge/error path the §5.1 binding handles.
    (tmp_path / 'edge_models.py').write_text(EDGE_SRC)
    (tmp_path / 'broken.py').write_text('this is not valid python {{{')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    occ = [
        _ScipOccurrence(symbol=sym, range=(line, 4, line, 8), is_definition=True)
        for sym, line in EDGE_DEFS.items()
    ]
    index = ScipIndex(
        documents=(
            _ScipDoc(relative_path='edge_models.py', occurrences=tuple(occ),
                     symbols=tuple(_ScipSymbol(symbol=s, kind='', display_name=s)
                                   for s in EDGE_DEFS)),
            _ScipDoc(relative_path='broken.py', occurrences=(), symbols=()),
        ),
        source_root=tmp_path,
    )

    result = persist_schema_symbols(conn, 's', index, strategies=[SQLAlchemyStrategy()])
    rows = _schema_rows(conn)
    T = 'data sql s _._.'

    # __tablename__ literal -> exact table; Column positional string -> exact column
    assert rows[T + 'things'] == ('table', None, E_THING, None, 'exact')
    assert rows[T + 'things#thing_name'] == ('column', 'thing_name', E_NAME, None, 'exact')
    # FK string target resolves to the owning model's table (self-ref here)
    assert rows[T + 'things#rel'] == ('column', 'rel', E_REL, T + 'things', 'derived')
    # symbol-form ForeignKey(Other.id) -> no string target -> no guessed link
    assert rows[T + 'things#symref'] == ('column', 'symref', E_SYMREF, None, 'derived')

    # non-column attrs (plain = 5, typed: int = 5) and the docstring are skipped
    assert (T + 'things#plain') not in rows
    assert (T + 'things#typed') not in rows
    # un-anchored model/field SURFACED as gaps, never silently bound; unparsable
    # doc recorded; non-literal __tablename__ surfaced — not crashed (§5.0.1 #5, §7)
    assert (T + 'ghosts') not in rows
    assert (T + 'things#missing') not in rows
    assert any('Ghost' in g and 'anchor' in g for g in result.gaps)
    assert any('missing' in g and 'anchor' in g for g in result.gaps)
    assert any('broken.py' in g and 'parse' in g for g in result.gaps)
    assert any('Tweaked' in g and 'literal' in g for g in result.gaps)
    # only Thing -> things was bound; Ghost/Tweaked/Plain/Helper are all excluded
    tables = {r[0] for r in conn.execute(
        "SELECT canonical_id FROM schema_symbols WHERE node_type = 'table'")}
    assert tables == {T + 'things'}
    conn.close()

    # detect() is False on a source with no SQLAlchemy models — even past an
    # unparsable doc, which must be skipped (not crash) before the verdict.
    (tmp_path / 'no_models.py').write_text('x = 1\n')
    empty = ScipIndex(
        documents=(
            _ScipDoc(relative_path='broken.py', occurrences=(), symbols=()),
            _ScipDoc(relative_path='no_models.py', occurrences=(), symbols=()),
        ),
        source_root=tmp_path,
    )
    assert SQLAlchemyStrategy().detect(empty, tmp_path) is False


IS_P = 'scip-python python studio . studio/'
FEATURE = IS_P + 'Feature#'
F_ID, F_NAME, F_STATE = IS_P + 'Feature#id.', IS_P + 'Feature#name.', IS_P + 'Feature#state.'
IS_DEFS = {FEATURE: 0, F_ID: 1, F_NAME: 2, F_STATE: 3}

IS_SRC = '''\
class Feature(Base):
    id = Column(String, primary_key=True)
    name = Column(String(255), nullable=True)
    state = Column(Enum("a", "b", name="state"))
class Abstract(Base):
    pass
'''


def test_sqlalchemy_tablename_from_class_name_when_absent(tmp_path):
    """A declarative model with NO __tablename__ (the common declarative_base /
    CustomBase convention) binds with the table
    name derived from the lowercased class name at confidence 'derived', so it is
    held below the assert floor until a migration/DDL witness promotes it (§3a).
    A base subclass with neither __tablename__ nor columns is not a model."""
    (tmp_path / 'studio').mkdir()
    (tmp_path / 'studio' / 'models.py').write_text(IS_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym, line in IS_DEFS.items():
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'studio', 'python', 'studio/models.py', line + 1, line + 1,
             '', sym, sym, None),
        )
    doc = _ScipDoc(
        relative_path='studio/models.py',
        occurrences=tuple(
            _ScipOccurrence(symbol=sym, range=(line, 4, line, 8), is_definition=True)
            for sym, line in IS_DEFS.items()),
        symbols=tuple(_ScipSymbol(symbol=s, kind='', display_name=s) for s in IS_DEFS),
    )
    index = ScipIndex(documents=(doc,), source_root=tmp_path)

    persist_schema_symbols(conn, 'studio', index, strategies=[SQLAlchemyStrategy()])
    rows = _schema_rows(conn)
    T = 'data sql studio _._.'
    # NO __tablename__ -> table name = lowercased class name, confidence derived
    assert rows[T + 'feature'] == ('table', None, FEATURE, None, 'derived')
    assert rows[T + 'feature#id'] == ('column', 'id', F_ID, None, 'derived')
    assert rows[T + 'feature#name'] == ('column', 'name', F_NAME, None, 'derived')
    assert rows[T + 'feature#state'] == ('column', 'state', F_STATE, None, 'derived')
    # a base subclass with neither __tablename__ nor columns is not a model
    assert (T + 'abstract') not in rows

    # derived -> held below the default assert floor (a gap until a witness promotes)
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)
    assert (T + 'feature#id') not in graph._symbols
