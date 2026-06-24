"""Phase 1 / slice 3 — the join (design §6, §10 Phase 1 "Join", and the §8
end-to-end example).

``impact_radius(column)`` must reach a dependent code symbol via BOTH paths
at once:
  - Layer 1: a genuine attribute reference ``u.email`` is a SCIP reference to
    the field symbol, which ``maps_to`` the column — so ``maps_to`` then
    ``callers_of(field)`` reaches the reader (free propagation, no data_access).
  - Layer 2: ``.update(email=…)`` names the column as a kwarg (no SCIP ref) —
    reached through the ``data_access`` write edge.

Composed from the REAL pipeline: synthesized SCIP (model + service defs, and
the ``u.email`` reference edge) + real source -> persist_schema_symbols (L1)
-> persist_data_access_orm (L2) -> add_data_layer -> compute_impact_radius.
No injected schema_symbols/data_access. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from cli.callers import compute_impact_radius
from docgen.orm_bindings import DjangoStrategy, persist_schema_symbols
from docgen.orm_bindings.access import persist_data_access_orm
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
)
from library.scip import init_scip_schema

P = 'scip-python python app . app/'
USER, U_EMAIL = P + 'models/User#', P + 'models/User#email.'
NOTIFY, DEACT = P + 'svc/notify().', P + 'svc/deactivate().'
COL = 'data sql app _._.users#email_addr'

MODELS_SRC = '''\
class User(models.Model):
    email = models.EmailField(db_column="email_addr")
    class Meta:
        db_table = "users"
'''
SVC_SRC = '''\
def notify(u):
    send(u.email)
def deactivate(uid):
    User.objects.update(email="")
'''

# scip_symbols: (canonical_id, file, line_start, line_end)
SYMS = [
    (USER, 'app/models.py', 1, 4), (U_EMAIL, 'app/models.py', 2, 2),
    (NOTIFY, 'app/svc.py', 1, 2), (DEACT, 'app/svc.py', 3, 4),
]


@pytest.fixture
def setup(tmp_path):
    (tmp_path / 'app').mkdir()
    (tmp_path / 'app' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'app' / 'svc.py').write_text(SVC_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym, file, ls, le in SYMS:
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'app', 'python', file, ls, le, '', sym, sym, None),
        )
    # the genuine attribute reference u.email -> User.email (Layer-1 path)
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
        (NOTIFY, U_EMAIL, 'reference', 'app/svc.py', 2, 'exact'),
    )
    models_doc = _ScipDoc(
        relative_path='app/models.py',
        occurrences=(
            _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
            _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
        ),
        symbols=(),
    )
    svc_doc = _ScipDoc(
        relative_path='app/svc.py',
        occurrences=(
            _ScipOccurrence(symbol=NOTIFY, range=(0, 0, 1, 40), is_definition=True),
            _ScipOccurrence(symbol=DEACT, range=(2, 0, 3, 40), is_definition=True),
        ),
        symbols=(),
    )
    index = ScipIndex(documents=(models_doc, svc_doc), source_root=tmp_path)
    yield conn, index
    conn.close()


def test_impact_radius_joins_layer1_and_layer2_paths(setup):
    conn, index = setup
    persist_schema_symbols(conn, 'app', index, strategies=[DjangoStrategy()])
    persist_data_access_orm(conn, 'app', index, strategies=[DjangoStrategy()])

    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)

    # the column's direct callers: the maps_to producer (Layer 1) + the write
    # (Layer 2) — NOTE notify is NOT a direct caller; it reaches via the field.
    direct = {(e.caller.canonical_id, e.edge_type) for e in graph.callers_of(COL)}
    assert direct == {(U_EMAIL, 'maps_to'), (DEACT, 'write')}

    affected = {s.canonical_id for s in
                compute_impact_radius(graph, COL, depth=5).affected_symbols}
    # Layer-2: the query writer is reached directly through data_access
    assert DEACT in affected
    # Layer-1: the reader is reached transitively — maps_to -> the field symbol
    # -> callers_of(field) via the genuine u.email SCIP reference
    assert U_EMAIL in affected
    assert NOTIFY in affected
