"""Owning-symbol attribution via SCIP ``enclosing_range`` (design §5.0, §5.1).

A query inside a function *body* must bind to that function. A SCIP definition
occurrence's plain ``range`` is only the **name token** — one line for
scip-python (measured: 100% single-line on real indexes). The body span lives
in SCIP's separate ``enclosing_range``. So ``_owning_lookup`` must:

  - use ``enclosing_range`` (the body span) for containment when present, and
    fall back to the name-token ``range`` only when it is absent; and
  - never offer a ``local N`` symbol as an owning symbol — otherwise a query
    assigned to a local variable on the same line binds to that *variable*
    (the anonymous ``local 148`` consumers seen on real data) or, with no
    same-line def at all, is dropped as "no owning symbol" so its columns look
    falsely dead.

Covered end-to-end over the real binding path (SQLAlchemy strategy ->
``persist_data_access_orm``) plus the supporting plumbing that carries the body
span off the wire (``_proto_to_doc``) and through the ``.vue`` remap
(``apply_vue_mapping``). SCIP is synthesized exactly as scip-python emits it:
single-line name-token ``range`` + multi-line ``enclosing_range``.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import SQLAlchemyStrategy
from docgen.orm_bindings.access import _owning_lookup, persist_data_access_orm
from docgen.scip_extractor import (
    ScipIndex,
    _proto_to_doc,
    _ScipDoc,
    _ScipOccurrence,
    apply_vue_mapping,
)
from library.scip import init_scip_schema

# scip-python-style symbols (neutral synthetic fixtures).
P = 'scip-python python src1 . app/models/'
WIDGET = P + 'Widget#'
WIDGET_LABEL = P + 'Widget#label.'
LOAD = P + 'load_widgets().'

# 1-indexed source; the query sits on line 6, inside load_widgets (def line 5).
MODEL_SRC = '''\
class Widget(Base):
    __tablename__ = "widget"
    label = mapped_column("label")

def load_widgets():
    rows = select(Widget.label).where(Widget.label == "x")
    return rows
'''


def _index(tmp_path, *, enclosing):
    """SCIP index for MODEL_SRC. ``load_widgets`` carries a single-line
    name-token ``range`` (line 4, 0-indexed); ``enclosing`` is its body span
    (lines 4-6) when given, else ``()`` — the with/without-body-span contrast.
    A same-line ``local 3`` (the ``rows`` assignment on line 5) is always
    present: the fix must skip it rather than let it win the query's owner."""
    (tmp_path / 'models.py').write_text(MODEL_SRC)
    doc = _ScipDoc(
        relative_path='models.py',
        occurrences=(
            _ScipOccurrence(symbol=WIDGET, range=(0, 6, 12), is_definition=True),
            _ScipOccurrence(symbol=WIDGET_LABEL, range=(2, 4, 9), is_definition=True),
            _ScipOccurrence(symbol=LOAD, range=(4, 4, 16), is_definition=True,
                            enclosing_range=enclosing),
            _ScipOccurrence(symbol='local 3', range=(5, 4, 8), is_definition=True),
        ),
        symbols=(),
    )
    return ScipIndex(documents=(doc,), source_root=tmp_path)


def _rows(conn):
    return list(conn.execute(
        'SELECT consumer_symbol_id, role FROM data_access WHERE source_name = ?',
        ('src1',)))


def test_body_query_binds_to_enclosing_function_not_local(tmp_path):
    """With the body span present, the body query binds to ``load_widgets`` —
    not to the same-line ``local 3``, and never dropped. Strip the body span
    (only the name-token ``range`` left) and the query has no containing named
    def → it is SURFACED as a gap, no row. The sole difference between the two
    runs is ``enclosing_range``, so it is exactly what carries the fix."""
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    try:
        # With the body span: attributed to the enclosing function.
        result = persist_data_access_orm(
            conn, 'src1', _index(tmp_path, enclosing=(4, 0, 6, 15)),
            strategies=[SQLAlchemyStrategy()])
        rows = _rows(conn)
        assert rows, 'a body query must produce data_access rows, not be dropped'
        assert {c for c, _ in rows} == {LOAD}, (
            f'consumer must be the function, never the local: {rows}')
        assert {role for _, role in rows} == {'project', 'filter'}
        assert not any('owning symbol' in g for g in result.gaps)

        # Without the body span: no containing named def → surfaced gap, no row.
        result2 = persist_data_access_orm(
            conn, 'src1', _index(tmp_path, enclosing=()),
            strategies=[SQLAlchemyStrategy()])
        assert _rows(conn) == [], 'no body span → no owner to bind, not a local'
        assert any('has no owning symbol' in g for g in result2.gaps)
    finally:
        conn.close()


def test_owning_lookup_uses_body_span_skips_locals_falls_back_to_range():
    """``_owning_lookup`` contract: body-span containment, ``local`` excluded,
    fall back to ``range`` when no span, tightest *named* def wins, and a
    non-definition occurrence is not an owner."""
    doc = _ScipDoc(
        relative_path='m.py',
        occurrences=(
            # named def: single-line name-token range + multi-line body span
            _ScipOccurrence(symbol='pkg/f().', range=(5, 4, 15),
                            is_definition=True, enclosing_range=(5, 0, 20, 0)),
            # same-line local inside f's body — must be skipped, not chosen
            _ScipOccurrence(symbol='local 7', range=(10, 4, 8), is_definition=True),
            # named def with NO body span: falls back to its (4-tuple) range
            _ScipOccurrence(symbol='pkg/g().', range=(25, 0, 30, 0),
                            is_definition=True),
            # single-line named def, 3-tuple range — exercises the else span[0]
            _ScipOccurrence(symbol='pkg/C.', range=(2, 0, 6), is_definition=True),
            # nested: outer class body + inner method body — tightest wins
            _ScipOccurrence(symbol='pkg/Outer#', range=(40, 6, 11),
                            is_definition=True, enclosing_range=(40, 0, 60, 0)),
            _ScipOccurrence(symbol='pkg/Outer#m().', range=(45, 8, 9),
                            is_definition=True, enclosing_range=(45, 0, 50, 0)),
            # a non-definition occurrence — excluded from the owning map
            _ScipOccurrence(symbol='pkg/ref().', range=(11, 0, 4),
                            is_definition=False),
        ),
        symbols=(),
    )
    owning = _owning_lookup(ScipIndex(documents=(doc,)))
    # inside f's body, on the local's own line: binds to f, never local 7
    assert owning('m.py', 10) == 'pkg/f().'
    # inside g's body via the name-token-range fallback (no body span)
    assert owning('m.py', 27) == 'pkg/g().'
    # on a single-line def's own line (3-tuple range, else-branch)
    assert owning('m.py', 2) == 'pkg/C.'
    # nested: the tighter inner method wins over the enclosing class
    assert owning('m.py', 47) == 'pkg/Outer#m().'
    # outside every def → no owner
    assert owning('m.py', 100) is None


def test_proto_to_doc_parses_enclosing_range():
    """The real protobuf->intermediate step must carry ``enclosing_range`` (the
    field the loader previously dropped on the floor)."""
    scip_pb2 = pytest.importorskip('docgen.scip.scip_pb2')
    pb_doc = scip_pb2.Document(relative_path='m.py')
    occ = pb_doc.occurrences.add()
    occ.symbol = 'pkg/f().'
    occ.range[:] = [5, 4, 17]               # name token, single line
    occ.enclosing_range[:] = [5, 0, 20, 13]  # body span, multi-line
    occ.symbol_roles = scip_pb2.SymbolRole.Definition

    doc = _proto_to_doc(pb_doc)
    assert doc.occurrences[0].range == (5, 4, 17)
    assert doc.occurrences[0].enclosing_range == (5, 0, 20, 13)
    assert doc.occurrences[0].is_definition is True


def test_apply_vue_mapping_carries_and_offsets_enclosing_range():
    """The ``.vue`` remap must carry ``enclosing_range`` through, with the line
    components offset like ``range`` (else the body span is lost for Vue)."""
    doc = _ScipDoc(
        relative_path='Foo.vue.script.ts',
        occurrences=(
            _ScipOccurrence(symbol='pkg/f().', range=(2, 0, 9),
                            is_definition=True, enclosing_range=(2, 0, 8, 0)),
            # a ref with no body span — the absent-span case stays absent
            _ScipOccurrence(symbol='pkg/g().', range=(4, 0, 5),
                            is_definition=False),
        ),
        symbols=(),
    )
    out = apply_vue_mapping(
        ScipIndex(documents=(doc,)),
        {'Foo.vue.script.ts': {'original': 'Foo.vue', 'line_offset': 10}})
    occ0, occ1 = out.documents[0].occurrences
    assert out.documents[0].relative_path == 'Foo.vue'
    assert occ0.range == (12, 0, 9)               # 3-tuple: line + 10
    assert occ0.enclosing_range == (12, 0, 18, 0)  # 4-tuple: both lines + 10
    assert occ1.enclosing_range == ()              # absent span stays absent
