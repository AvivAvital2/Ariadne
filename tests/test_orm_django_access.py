"""Phase 1 / slice 2 — Django Layer-2 access binding (design §5.2 Access API,
§5.0, §3a, §6, §10 Phase 1).

ORM query call sites name their columns as kwargs/strings, so SCIP emits no
reference to the field symbol (§5.0) — only a recognizer captures them. The
engine walks ``<Model>.objects.<verb>(...)`` call sites in the app source,
resolves the receiver model + the field names against the Layer-1 schema, and
emits role-typed ``data_access`` rows at each call site's enclosing symbol:
filter/exclude/get -> filter, create/update -> write, values/only/defer ->
project, order_by -> order. Resolved columns -> ``resolved`` (§3a).

End-to-end over the real pipeline: synthesized SCIP + real model & service
source -> persist_schema_symbols (Layer 1) -> persist_data_access_orm
(Layer 2) -> add_data_layer. No injected data_access. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

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

MP = 'scip-python python billing . billing/models/'
SP = 'scip-python python billing . billing/services/'
USER, U_EMAIL, U_NAME = MP + 'User#', MP + 'User#email.', MP + 'User#name.'
FIND, DEACT, RECENT, MAKE = (SP + 'find_user().', SP + 'deactivate().',
                             SP + 'recent().', SP + 'make().')

MODELS_SRC = '''\
class User(models.Model):
    email = models.EmailField(db_column="email_addr")
    name = models.CharField()
    class Meta:
        db_table = "users"
'''

SERVICES_SRC = '''\
def find_user(uid):
    return User.objects.filter(email=uid).values("name")
def deactivate(uid):
    User.objects.filter(email=uid).update(name="")
def recent():
    return User.objects.order_by("name")
def make(email):
    User.objects.create(email=email)
'''

# scip_symbols rows: (canonical_id, file, line_start, line_end)
SYMBOLS = [
    (USER, 'billing/models.py', 1, 5), (U_EMAIL, 'billing/models.py', 2, 2),
    (U_NAME, 'billing/models.py', 3, 3),
    (FIND, 'billing/services.py', 1, 2), (DEACT, 'billing/services.py', 3, 4),
    (RECENT, 'billing/services.py', 5, 6), (MAKE, 'billing/services.py', 7, 8),
]


@pytest.fixture
def setup(tmp_path):
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'billing' / 'services.py').write_text(SERVICES_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym, file, ls, le in SYMBOLS:
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'billing', 'python', file, ls, le, '', sym, sym, None),
        )
    models_doc = _ScipDoc(
        relative_path='billing/models.py',
        occurrences=(
            _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
            _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
            _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True),
        ),
        symbols=(),
    )
    # service defs carry BODY-spanning ranges so the call sites resolve to
    # their enclosing (owning) symbol by containment.
    services_doc = _ScipDoc(
        relative_path='billing/services.py',
        occurrences=(
            _ScipOccurrence(symbol=FIND, range=(0, 0, 1, 60), is_definition=True),
            _ScipOccurrence(symbol=DEACT, range=(2, 0, 3, 60), is_definition=True),
            _ScipOccurrence(symbol=RECENT, range=(4, 0, 5, 60), is_definition=True),
            _ScipOccurrence(symbol=MAKE, range=(6, 0, 7, 60), is_definition=True),
        ),
        symbols=(),
    )
    index = ScipIndex(documents=(models_doc, services_doc), source_root=tmp_path)
    yield conn, index
    conn.close()


def _access(conn):
    return {
        tuple(r) for r in conn.execute(
            'SELECT consumer_symbol_id, schema_symbol_id, role, confidence '
            "FROM data_access WHERE witness = 'orm:django'"
        )
    }


def test_django_query_call_sites_bind_role_typed_access(setup):
    conn, index = setup
    persist_schema_symbols(conn, 'billing', index, strategies=[DjangoStrategy()])

    persist_data_access_orm(conn, 'billing', index, strategies=[DjangoStrategy()])

    T = 'data sql billing _._.'
    EMAIL, NAME = T + 'users#email_addr', T + 'users#name'
    # field NAMES (email/name) resolve to their COLUMNS; roles per verb;
    # chained .filter().values() yields a row per verb (§5.2)
    assert _access(conn) == {
        (FIND, EMAIL, 'filter', 'resolved'),    # .filter(email=…)
        (FIND, NAME, 'project', 'resolved'),     # .values("name")
        (DEACT, EMAIL, 'filter', 'resolved'),    # .filter(email=…)
        (DEACT, NAME, 'write', 'resolved'),      # .update(name=…)
        (RECENT, NAME, 'order', 'resolved'),     # .order_by("name")
        (MAKE, EMAIL, 'write', 'resolved'),      # .create(email=…)
    }

    # end-to-end: rows project (resolved asserts at the default floor) and the
    # column's callers are the query sites, role preserved — alongside the
    # Layer-1 maps_to from the field symbol.
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)
    by_caller = {(e.caller.canonical_id, e.edge_type)
                 for e in graph.callers_of(EMAIL)}
    assert {(FIND, 'filter'), (DEACT, 'filter'), (MAKE, 'write'),
            (U_EMAIL, 'maps_to')} <= by_caller


def test_django_access_resolves_model_imported_under_alias(tmp_path):
    """A query that names its model under an import alias (``from .models import
    User as DBUser``) resolves to the declared model before the schema lookup —
    the SAME shared-engine import-alias map the SQLAlchemy recognizer uses — so
    ``DBUser.objects.filter(...)`` binds its column instead of surfacing a false
    'unknown model' gap (§5.0)."""
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'billing' / 'svc.py').write_text(
        'from billing.models import User as DBUser\n'
        'def by_alias(uid):\n'
        '    return DBUser.objects.filter(email=uid)\n')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    MP = 'scip-python python billing . billing/models/'
    SP = 'scip-python python billing . billing/svc/'
    USER, U_EMAIL, U_NAME, BY_ALIAS = (
        MP + 'User#', MP + 'User#email.', MP + 'User#name.', SP + 'by_alias().')
    models_doc = _ScipDoc(relative_path='billing/models.py', occurrences=(
        _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
        _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
        _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True)), symbols=())
    # by_alias() body spans lines 1..2; the import on line 0 is module-level.
    svc_doc = _ScipDoc(relative_path='billing/svc.py', occurrences=(
        _ScipOccurrence(symbol=BY_ALIAS, range=(1, 0, 2, 60), is_definition=True),), symbols=())
    index = ScipIndex(documents=(models_doc, svc_doc), source_root=tmp_path)
    persist_schema_symbols(conn, 'billing', index, strategies=[DjangoStrategy()])
    result = persist_data_access_orm(conn, 'billing', index, strategies=[DjangoStrategy()])
    rows = {tuple(r) for r in conn.execute(
        "SELECT consumer_symbol_id, schema_symbol_id, role, confidence FROM data_access "
        "WHERE witness = 'orm:django'")}
    assert rows == {(BY_ALIAS, 'data sql billing _._.users#email_addr', 'filter', 'resolved')}
    # the aliased model resolved -> NO false 'unknown model' gap surfaced
    assert not any('unknown model' in g for g in result.gaps)
    conn.close()


class _NonDetectingStrategy:
    name = 'orm:dummy'

    def detect(self, scip_index, root):
        return False

    def discover(self, scip_index, root, symbol_at):
        raise AssertionError('a non-detecting strategy must never run')


PROBE = SP + 'probe().'
EDGE_SRC = '''\
User.objects.filter(email="x")
def probe():
    Ghost.objects.filter(x=1)
    User.objects.annotate(n=1)
    User.objects.all()
    User.objects.filter(**kw)
    User.objects.filter(org__name="x")
    User.objects.filter(bogus=1)
'''


def test_django_access_surfaces_every_undecodable_form_as_a_gap(tmp_path):
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'billing' / 'edge.py').write_text(EDGE_SRC)
    (tmp_path / 'billing' / 'broken.py').write_text('def (((')  # unparsable -> skipped
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)

    models_doc = _ScipDoc(
        relative_path='billing/models.py',
        occurrences=(
            _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
            _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
            _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True),
        ),
        symbols=(),
    )
    edge_doc = _ScipDoc(
        relative_path='billing/edge.py',
        occurrences=(
            # probe() body spans lines 1..8; the module-level query on line 0
            # is deliberately OUTSIDE any def -> no owning symbol.
            _ScipOccurrence(symbol=PROBE, range=(1, 0, 8, 60), is_definition=True),
            # a non-definition occurrence is skipped by the owning index.
            _ScipOccurrence(symbol=USER, range=(0, 0, 0, 4), is_definition=False),
        ),
        symbols=(),
    )
    broken_doc = _ScipDoc(relative_path='billing/broken.py', occurrences=(), symbols=())
    index = ScipIndex(documents=(models_doc, edge_doc, broken_doc), source_root=tmp_path)

    result = persist_data_access_orm(
        conn, 'billing', index,
        strategies=[_NonDetectingStrategy(), DjangoStrategy()],
    )

    # nothing bindable here -> no rows; every form is surfaced as a gap (§5.0)
    assert result.rows_written == 0
    blob = ' | '.join(result.gaps)
    for needle in ('no owning symbol', 'unknown model',
                   'computed', 'non-literal', 'spanning lookup', 'undeclared field'):
        assert needle in blob, needle
    # the non-detecting strategy wrote nothing
    assert not list(conn.execute(
        "SELECT 1 FROM data_access WHERE witness = 'orm:dummy'"))


QSP = 'scip-python python billing . billing/q_svc/'
WQ, WQC, WAGG, WQV = (QSP + 'with_q().', QSP + 'with_q_combined().',
                      QSP + 'with_agg().', QSP + 'with_qs_var().')
Q_SRC = '''\
def with_q(x):
    User.objects.filter(Q(email=x))
def with_q_combined(x):
    User.objects.filter(Q(email=x) | Q(name=x))
def with_agg():
    User.objects.aggregate(Sum("name"))
def with_qs_var(qs, x):
    qs.filter(email=x)
'''


def test_django_access_decodes_q_computed_and_surfaces_queryset_var(tmp_path):
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'billing' / 'q_svc.py').write_text(Q_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    models_doc = _ScipDoc(
        relative_path='billing/models.py',
        occurrences=(
            _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
            _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
            _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True),
        ),
        symbols=(),
    )
    q_doc = _ScipDoc(
        relative_path='billing/q_svc.py',
        occurrences=(
            _ScipOccurrence(symbol=WQ, range=(0, 0, 1, 40), is_definition=True),
            _ScipOccurrence(symbol=WQC, range=(2, 0, 3, 40), is_definition=True),
            _ScipOccurrence(symbol=WAGG, range=(4, 0, 5, 40), is_definition=True),
            _ScipOccurrence(symbol=WQV, range=(6, 0, 7, 40), is_definition=True),
        ),
        symbols=(),
    )
    index = ScipIndex(documents=(models_doc, q_doc), source_root=tmp_path)

    result = persist_data_access_orm(conn, 'billing', index, strategies=[DjangoStrategy()])

    T = 'data sql billing _._.'
    rows = {tuple(r) for r in conn.execute(
        "SELECT consumer_symbol_id, schema_symbol_id, role FROM data_access "
        "WHERE witness = 'orm:django'")}
    assert rows == {
        # Q()-wrapped filter is decoded (not silently dropped), incl. combined Q|Q
        (WQ, T + 'users#email_addr', 'filter'),
        (WQC, T + 'users#email_addr', 'filter'),
        (WQC, T + 'users#name', 'filter'),
        # aggregate(Sum("name")) -> the declared field read
        (WAGG, T + 'users#name', 'project'),
    }
    # queryset-variable call (qs.filter) is SURFACED as a gap, never silent (§5.0)
    assert any('queryset variable' in g for g in result.gaps)


FSP = 'scip-python python billing . billing/f_svc/'
FF, GG = FSP + 'ff().', FSP + 'gg().'
F_SRC = '''\
def ff():
    User.objects.filter(email=F("name"))
def gg():
    User.objects.update(email=F("name"))
'''


def test_django_access_decodes_f_in_value_position(tmp_path):
    # F("name") references a field's current value -> a read of that field,
    # alongside the kwarg key's own role. Neither must be silently dropped.
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODELS_SRC)
    (tmp_path / 'billing' / 'f_svc.py').write_text(F_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    models_doc = _ScipDoc(
        relative_path='billing/models.py',
        occurrences=(
            _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
            _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
            _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True),
        ),
        symbols=(),
    )
    f_doc = _ScipDoc(
        relative_path='billing/f_svc.py',
        occurrences=(
            _ScipOccurrence(symbol=FF, range=(0, 0, 1, 40), is_definition=True),
            _ScipOccurrence(symbol=GG, range=(2, 0, 3, 40), is_definition=True),
        ),
        symbols=(),
    )
    index = ScipIndex(documents=(models_doc, f_doc), source_root=tmp_path)

    persist_data_access_orm(conn, 'billing', index, strategies=[DjangoStrategy()])

    T = 'data sql billing _._.'
    rows = {tuple(r) for r in conn.execute(
        "SELECT consumer_symbol_id, schema_symbol_id, role FROM data_access "
        "WHERE witness = 'orm:django'")}
    assert rows == {
        (FF, T + 'users#email_addr', 'filter'),   # the filter key
        (FF, T + 'users#name', 'project'),          # F("name") read
        (GG, T + 'users#email_addr', 'write'),     # the update key
        (GG, T + 'users#name', 'project'),          # F("name") read
    }


def test_bulk_create_expands_to_per_column_writes(tmp_path):
    """A whole-row write (bulk_create) expands to a write on EVERY column of
    the model's table — the §10 Phase-2 fix for the write-recall hole (so
    impact_radius on any column reaches the bulk writer). Was a surfaced gap."""
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(
        'class User(models.Model):\n'
        '    email = models.EmailField(db_column="email_addr")\n'
        '    name = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "users"\n')
    (tmp_path / 'billing' / 'svc.py').write_text(
        'def seed(rows):\n'
        '    User.objects.bulk_create(rows)\n')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    MP = 'scip-python python billing . billing/models/'
    SP = 'scip-python python billing . billing/svc/'
    USER, U_EMAIL, U_NAME, SEED = (
        MP + 'User#', MP + 'User#email.', MP + 'User#name.', SP + 'seed().')
    models_doc = _ScipDoc(relative_path='billing/models.py', occurrences=(
        _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
        _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
        _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True)), symbols=())
    svc_doc = _ScipDoc(relative_path='billing/svc.py', occurrences=(
        _ScipOccurrence(symbol=SEED, range=(0, 0, 1, 40), is_definition=True),), symbols=())
    index = ScipIndex(documents=(models_doc, svc_doc), source_root=tmp_path)
    persist_schema_symbols(conn, 'billing', index, strategies=[DjangoStrategy()])
    persist_data_access_orm(conn, 'billing', index, strategies=[DjangoStrategy()])
    writes = {tuple(r) for r in conn.execute(
        "SELECT consumer_symbol_id, schema_symbol_id, role FROM data_access "
        "WHERE role = 'write'")}
    assert (SEED, 'data sql billing _._.users#email_addr', 'write') in writes
    assert (SEED, 'data sql billing _._.users#name', 'write') in writes
    conn.close()


def test_save_is_surfaced_as_a_gap(tmp_path):
    """save() is a whole-row instance write whose receiver model is not
    statically bound without dataflow (§5 residue). It must be SURFACED as a
    gap, never silently dropped (§5.0 'surface, don't guess') — the resolvable
    Model(...).save() expansion is a documented future refinement."""
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(
        'class User(models.Model):\n'
        '    email = models.EmailField(db_column="email_addr")\n'
        '    name = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "users"\n')
    (tmp_path / 'billing' / 'svc.py').write_text(
        'def make():\n'
        '    User(email="x").save()\n'
        'def touch(obj):\n'
        '    obj.save()\n')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    MP = 'scip-python python billing . billing/models/'
    SP = 'scip-python python billing . billing/svc/'
    USER, U_EMAIL, U_NAME = MP + 'User#', MP + 'User#email.', MP + 'User#name.'
    MAKE, TOUCH = SP + 'make().', SP + 'touch().'
    models_doc = _ScipDoc(relative_path='billing/models.py', occurrences=(
        _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
        _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
        _ScipOccurrence(symbol=U_NAME, range=(2, 4, 2, 8), is_definition=True)), symbols=())
    svc_doc = _ScipDoc(relative_path='billing/svc.py', occurrences=(
        _ScipOccurrence(symbol=MAKE, range=(0, 0, 1, 40), is_definition=True),
        _ScipOccurrence(symbol=TOUCH, range=(2, 0, 3, 40), is_definition=True)), symbols=())
    index = ScipIndex(documents=(models_doc, svc_doc), source_root=tmp_path)
    persist_schema_symbols(conn, 'billing', index, strategies=[DjangoStrategy()])
    result = persist_data_access_orm(conn, 'billing', index, strategies=[DjangoStrategy()])
    writes = {tuple(r) for r in conn.execute(
        "SELECT consumer_symbol_id, schema_symbol_id FROM data_access WHERE role = 'write'")}
    conn.close()
    # neither save() is silently dropped — both are surfaced as gaps, no rows.
    assert writes == set()
    assert sum('.save()' in g for g in result.gaps) >= 2
