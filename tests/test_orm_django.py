"""Phase 1 / slice 1 — Django Layer-1 structural binding, the FULL §5.2
extraction (design §5.2, §5.0.1, §3a, §6, §10 Phase 1).

The engine discovers Django models from the SCIP index — anchored on the
class/field definition symbols (§5.0.1 #1), metadata read from the model
source — and binds, per §5.2:

  table  = Meta.db_table (exact)  | else app_label_model (derived)
  column = db_column (exact)      | else field-name (derived)
           FK/OneToOne -> {field}_id (derived) + references_id -> target table

producer_symbol_id is the model/field symbol; unresolved bindings (no SCIP
anchor, unparsable source, FK target not found) are RECORDED as surfaced
gaps, never silently dropped (§5.0, §5.0.1 #5). add_data_layer then asserts
exact at the default floor and holds derived below it (§3a/§6).

End-to-end over the real pipeline: synthesized SCIP (realistic — kind='',
no relationships, per §9a) + real source -> persist_schema_symbols ->
add_data_layer. No injected schema_symbols. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import DjangoStrategy, persist_schema_symbols
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
)
from library.scip import init_scip_schema

# app 'billing', file billing/models.py (so app_label derives to 'billing')
P = 'scip-python python billing . billing/'
ORG = P + 'Org#'
ORG_SLUG = P + 'Org#slug.'
USER = P + 'User#'
USER_EMAIL = P + 'User#email.'
USER_NAME = P + 'User#name.'
USER_ORG = P + 'User#org.'
USER_EXTERNAL = P + 'User#external.'
AUDIT = P + 'Audit#'
AUDIT_NOTE = P + 'Audit#note.'

# 0-indexed definition lines must match the source below.
DEFS = {
    ORG: 0, ORG_SLUG: 3,
    USER: 5, USER_EMAIL: 6, USER_NAME: 7, USER_ORG: 8, USER_EXTERNAL: 9,
    AUDIT: 13, AUDIT_NOTE: 14,
}

MODEL_SRC = '''\
class Org(models.Model):
    class Meta:
        db_table = "orgs"
    slug = models.SlugField()

class User(models.Model):
    email = models.EmailField(db_column="email_addr")
    name = models.CharField()
    org = models.ForeignKey(Org, on_delete=models.CASCADE)
    external = models.ForeignKey(ExternalThing, on_delete=models.CASCADE)
    class Meta:
        db_table = "users"

class Audit(models.Model):
    note = models.TextField()
'''


@pytest.fixture
def setup(tmp_path):
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODEL_SRC)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    for sym, line in DEFS.items():
        display = sym.rstrip('#.').rsplit('/', 1)[-1].split('#')[-1] or sym
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'billing', 'python', 'billing/models.py', line + 1, line + 1,
             '', display, display, None),
        )
    doc = _ScipDoc(
        relative_path='billing/models.py',
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


def test_django_full_structural_binding(setup):
    conn, index = setup

    result = persist_schema_symbols(conn, 'billing', index, strategies=[DjangoStrategy()])
    rows = _schema_rows(conn)

    T = 'data sql billing _._.'
    # explicit Meta.db_table -> exact table; explicit db_column -> exact column
    assert rows[T + 'orgs'] == ('table', None, ORG, None, 'exact')
    assert rows[T + 'users'] == ('table', None, USER, None, 'exact')
    assert rows[T + 'users#email_addr'] == ('column', 'email_addr', USER_EMAIL, None, 'exact')
    # no db_column -> derived field-name column
    assert rows[T + 'users#name'] == ('column', 'name', USER_NAME, None, 'derived')
    assert rows[T + 'orgs#slug'] == ('column', 'slug', ORG_SLUG, None, 'derived')
    # FK -> derived {field}_id column + references_id to the resolved target
    assert rows[T + 'users#org_id'] == ('column', 'org_id', USER_ORG, T + 'orgs', 'derived')
    # no Meta.db_table -> derived app_label_model table name
    assert rows[T + 'billing_audit'] == ('table', None, AUDIT, None, 'derived')
    assert rows[T + 'billing_audit#note'] == ('column', 'note', AUDIT_NOTE, None, 'derived')

    # unresolved FK target (ExternalThing is not a discovered model) -> a
    # surfaced gap, and references_id left unbound — never a guessed link.
    assert rows[T + 'users#external_id'][3] is None  # references_id
    assert any('ExternalThing' in g for g in result.gaps)

    # add_data_layer: exact asserts at the default floor; derived held below
    graph = CrossSourceGraph()
    graph.load_from(conn)
    graph.add_data_layer(conn)
    assert (T + 'users#email_addr') in graph._symbols      # exact -> asserted
    assert (T + 'users#name') not in graph._symbols        # derived -> gap
    assert (USER_EMAIL, 'maps_to') in {
        (e.caller.canonical_id, e.edge_type) for e in graph.callers_of(T + 'users#email_addr')
    }


class _NonDetectingStrategy:
    name = 'orm:dummy'

    def detect(self, scip_index, root):
        return False

    def discover(self, scip_index, root, symbol_at):
        raise AssertionError('a non-detecting strategy must never run')


EP = 'scip-python python s . edge_models/'
T_, TN, TR, TB, TD = (EP + 'Thing#', EP + 'Thing#name.', EP + 'Thing#rel.',
                      EP + 'Thing#blank.', EP + 'Thing#direct.')
TW, TWX = EP + 'Tweaked#', EP + 'Tweaked#x.'
PLN = EP + 'Plain#'
# Ghost (line 3) and Thing.missing (line 7) are deliberately UN-anchored.
EDGE_DEFS = {T_: 5, TN: 6, TR: 8, TB: 9, TD: 10, TW: 11, TWX: 15, PLN: 16}

EDGE_SRC = '''\
NOT_A_CLASS = 1
class Helper:
    pass
class Ghost(models.Model):
    pass
class Thing(models.Model):
    name = models.CharField(max_length=5)
    missing = models.CharField()
    rel = models.ForeignKey("pkg.Thing", on_delete=models.CASCADE)
    blank = models.ForeignKey()
    direct = EmailField(db_column="d")
class Tweaked(models.Model):
    class Meta:
        ordering = ["name"]
        db_table = SOME_CONST
    x = models.CharField()
class Plain(models.Model):
    class Meta:
        ordering = ["x"]
'''


def test_django_edge_and_error_paths_are_surfaced(tmp_path):
    # root-level source (no package dir -> app_label ambiguous -> recovered);
    # plus an unparsable doc, a non-model class, un-anchored model/field, a
    # string FK target, an arg-less FK, a non-literal Meta.db_table, and a
    # non-db_column kwarg — every edge/error path the §5.2 binding must handle.
    (tmp_path / 'edge_models.py').write_text(EDGE_SRC)
    (tmp_path / 'broken.py').write_text('this is not valid python {{{')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    occ = [
        _ScipOccurrence(symbol=sym, range=(line, 4, line, 8), is_definition=True)
        for sym, line in EDGE_DEFS.items()
    ]
    occ.append(_ScipOccurrence(symbol=T_, range=(5, 0, 5, 5), is_definition=False))
    index = ScipIndex(
        documents=(
            _ScipDoc(relative_path='edge_models.py', occurrences=tuple(occ),
                     symbols=tuple(_ScipSymbol(symbol=s, kind='', display_name=s)
                                   for s in EDGE_DEFS)),
            _ScipDoc(relative_path='broken.py', occurrences=(), symbols=()),
        ),
        source_root=tmp_path,
    )

    result = persist_schema_symbols(
        conn, 's', index, strategies=[_NonDetectingStrategy(), DjangoStrategy()],
    )
    rows = _schema_rows(conn)
    T = 'data sql s _._.'

    # ambiguous app_label -> recovered table name (§5.2 note)
    assert rows[T + 'thing'] == ('table', None, T_, None, 'recovered')
    assert rows[T + 'tweaked'] == ('table', None, TW, None, 'recovered')  # db_table non-literal
    # derived field-name column (kwargs present but no db_column)
    assert rows[T + 'thing#name'] == ('column', 'name', TN, None, 'derived')
    # field class imported directly (Name, not models.X) still binds
    assert rows[T + 'thing#d'] == ('column', 'd', TD, None, 'exact')
    # string FK target resolves; arg-less FK has no target to link
    assert rows[T + 'thing#rel_id'] == ('column', 'rel_id', TR, T + 'thing', 'derived')
    assert rows[T + 'thing#blank_id'] == ('column', 'blank_id', TB, None, 'derived')

    # un-anchored model/field are SURFACED as gaps, never silently bound,
    # and the unparsable doc is recorded — not crashed (§5.0.1 #5, §7)
    assert (T + 'ghost') not in rows
    assert (T + 'thing#missing') not in rows
    assert any('Ghost' in g for g in result.gaps)
    assert any('missing' in g for g in result.gaps)
    assert any('broken.py' in g and 'parse' in g for g in result.gaps)
    # the non-detecting strategy wrote nothing
    assert not any('orm:dummy' == r[0] for r in conn.execute(
        'SELECT resolution_source FROM schema_symbols'))

    # a Meta with no db_table at all -> the inner scan exhausts -> recovered
    assert rows[T + 'plain'] == ('table', None, PLN, None, 'recovered')

    # detect() returns False on a source with no Django models
    (tmp_path / 'no_models.py').write_text('x = 1\n')
    empty = ScipIndex(
        documents=(_ScipDoc(relative_path='no_models.py', occurrences=(), symbols=()),),
        source_root=tmp_path,
    )
    assert DjangoStrategy().detect(empty, tmp_path) is False
