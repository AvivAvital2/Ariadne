"""§10 final wiring — make the SQL data model live on a real ``ariadne index``
(design §10, lines 1138-1139: "Wire the new persist_* steps into the chain ...;
register the data nodes/edges in add_data_layer within load_from").

Two wiring points, tested behaviorally end-to-end:

1. ``persist_data_model`` runs the ORM + raw-SQL binders over every source and
   populates ``schema_symbols`` / ``data_access`` (the persist side, called from
   ``cli/index.py`` after ``persist_string_literals``).
2. ``cli.callers._load_graph`` applies ``add_data_layer`` after ``load_from`` so
   the CLI ``impact_radius`` / ``callers`` graph carries the data tier (design
   §6 wiring correction: that graph "gets the data layer ✓").

Synthetic fixtures only.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import config
from cli.callers import _load_graph
from docgen.orm_bindings import DjangoStrategy
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.scip_persist import persist_data_model
from library import Library
from library.scip import init_scip_schema

# --- point 2: _load_graph applies the data layer -------------------------

HANDLER = 'scip-python python app . app/svc/handler().'
COL = 'data sql app _._.accounts#balance'


def test_load_graph_applies_the_data_layer(tmp_path):
    db = tmp_path / 'ariadne.db'
    conn = sqlite3.connect(db)
    init_scip_schema(conn)
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (HANDLER, 'app', 'python', 'app/svc.py', 1, 1, '', 'handler', HANDLER, None),
    )
    conn.execute(
        'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
        'table_name, column_name, producer_symbol_id, resolution_source, '
        'confidence) VALUES (?,?,?,?,?,?,?,?)',
        (COL, 'app', 'column', 'accounts', 'balance', None, 'orm:django', 'exact'),
    )
    conn.execute(
        'INSERT INTO data_access (source_name, consumer_symbol_id, '
        'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
        ('app', HANDLER, COL, 'write', 'orm:django', 'resolved'),
    )
    conn.commit()
    conn.close()

    graph = _load_graph(str(db))
    # The data node exists and its access edge is reachable — load_from alone
    # would NOT register it; this proves add_data_layer ran in _load_graph.
    assert COL in graph._symbols
    assert (HANDLER, 'write') in {
        (e.caller.canonical_id, e.edge_type) for e in graph.callers_of(COL)
    }


# --- point 1: persist_data_model populates the tables --------------------

MODELS_SRC = '''\
class User(models.Model):
    email = models.EmailField(db_column="email_addr")
    class Meta:
        db_table = "users"
'''
SERVICES_SRC = '''\
def find_user(uid):
    return User.objects.filter(email=uid).values("email")
'''
MP = 'scip-python python billing . billing/'
USER, U_EMAIL, FIND = MP + 'User#', MP + 'User#email.', MP + 'find_user().'


def _django_index(root: Path) -> ScipIndex:
    (root / 'billing').mkdir()
    (root / 'billing' / 'models.py').write_text(MODELS_SRC)
    (root / 'billing' / 'services.py').write_text(SERVICES_SRC)
    (root / '.ariadne').mkdir()
    (root / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}),
    )
    models_doc = _ScipDoc(
        relative_path='billing/models.py',
        occurrences=(
            _ScipOccurrence(symbol=USER, range=(0, 6, 0, 10), is_definition=True),
            _ScipOccurrence(symbol=U_EMAIL, range=(1, 4, 1, 9), is_definition=True),
        ),
        symbols=(),
    )
    services_doc = _ScipDoc(
        relative_path='billing/services.py',
        occurrences=(
            _ScipOccurrence(symbol=FIND, range=(0, 0, 1, 60), is_definition=True),
        ),
        symbols=(),
    )
    return ScipIndex(documents=(models_doc, services_doc), source_root=root)


def test_persist_data_model_writes_schema_and_orm_access(tmp_path):
    index = _django_index(tmp_path)
    db = tmp_path / 'ariadne.db'

    # strategies passed explicitly (exercises the non-default branch)
    total = persist_data_model(
        db, [('billing', tmp_path)],
        index_factory=lambda *a, **k: index, strategies=[DjangoStrategy()],
    )

    conn = sqlite3.connect(db)
    schema = {r[0] for r in conn.execute('SELECT canonical_id FROM schema_symbols')}
    access = {tuple(r) for r in conn.execute(
        'SELECT consumer_symbol_id, schema_symbol_id, role FROM data_access')}
    conn.close()

    # Layer 1 (persist_schema_symbols) ran:
    assert 'data sql billing _._.users' in schema
    assert 'data sql billing _._.users#email_addr' in schema
    # Layer 2 (persist_data_access_orm) ran: .filter(email=) -> filter on the col
    assert (FIND, 'data sql billing _._.users#email_addr', 'filter') in access
    assert total > 0


def test_persist_data_model_runs_rawsql_and_skips_sources_without_manifest(tmp_path):
    db = tmp_path / 'ariadne.db'
    # Seed a SQL literal as persist_string_literals would have, earlier in the
    # chain. persist_data_model's raw-SQL binder reads string_literals.
    lib = Library(db)
    with lib._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT INTO string_literals (source_name, file, line_start, '
            'col_start, value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
            ('billing', 'q.py', 10, 4, 'SELECT email_addr FROM users',
             'scip-python python billing . billing/q().'),
        )
        conn.commit()
    lib.close()

    # No manifest under tmp_path/src -> ORM load is skipped optimistically, but
    # raw-SQL still runs off the persisted literals (index-independent).
    src = tmp_path / 'src'
    src.mkdir()
    total = persist_data_model(db, [('billing', src)], index_factory=lambda *a, **k: None)

    conn = sqlite3.connect(db)
    rawsql = [tuple(r) for r in conn.execute(
        "SELECT schema_symbol_id, role, confidence FROM data_access "
        "WHERE witness = 'rawsql'")]
    conn.close()
    # the literal was parsed into derived raw-SQL access rows (never asserted at
    # the default floor, but recorded — design §5.7/§5.8)
    assert rawsql
    assert all(c == 'derived' for *_, c in rawsql)
    assert total > 0


# --- the cross-source gate, reachable from ariadne.yaml (design §6) -------

def test_load_graph_fuses_cross_source_from_shared_database_config(tmp_path, monkeypatch):
    # Two services that each own accounts.balance, plus a consumer of each.
    db = tmp_path / 'ariadne.db'
    conn = sqlite3.connect(db)
    init_scip_schema(conn)
    A_W = 'scip-python python svc_a . svc_a/w().'
    B_R = 'scip-python python svc_b . svc_b/r().'
    A_COL = 'data sql svc_a _._.accounts#balance'
    B_COL = 'data sql svc_b _._.accounts#balance'
    SHARED = 'data sql @shared:svc_a+svc_b _._.accounts#balance'
    for sym, src in [(A_W, 'svc_a'), (B_R, 'svc_b')]:
        conn.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, src, 'python', 'x.py', 1, 1, '', sym, sym, None),
        )
    for cid, src in [(A_COL, 'svc_a'), (B_COL, 'svc_b')]:
        conn.execute(
            'INSERT INTO schema_symbols (canonical_id, source_name, node_type, '
            'table_name, column_name, producer_symbol_id, resolution_source, '
            'confidence) VALUES (?,?,?,?,?,?,?,?)',
            (cid, src, 'column', 'accounts', 'balance', None, 'orm:django', 'exact'),
        )
    for src, consumer, cid, role in [
        ('svc_a', A_W, A_COL, 'write'), ('svc_b', B_R, B_COL, 'filter'),
    ]:
        conn.execute(
            'INSERT INTO data_access (source_name, consumer_symbol_id, '
            'schema_symbol_id, role, witness, confidence) VALUES (?,?,?,?,?,?)',
            (src, consumer, cid, role, 'orm:django', 'resolved'),
        )
    conn.commit()
    conn.close()

    # ariadne.yaml declaring the two sources share one physical database.
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'db_path: {db}\n'
        f'shared_database:\n'
        f'  - sources: [svc_a, svc_b]\n'
        f'sources:\n'
        f'  svc_a:\n    path: {tmp_path}\n'
        f'  svc_b:\n    path: {tmp_path}\n',
        encoding='utf-8',
    )
    monkeypatch.setattr(config, '_global_config', config.Config(config_path=yaml_path))

    graph = _load_graph(None)  # reads cfg.db_path AND cfg.shared_database

    # the declaration fused the two sources' balance column into one node, so
    # impact_radius couples a writer in A with a reader in B in a single walk.
    assert SHARED in graph._symbols
    assert {e.caller.canonical_id for e in graph.callers_of(SHARED)} >= {A_W, B_R}
    assert A_COL not in graph._symbols and B_COL not in graph._symbols


# --- A: the schema-promotion wiring (config schema_sql → persist_data_model) ---

def test_source_config_parses_schema_sql(tmp_path):
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n  shop:\n    path: {tmp_path}\n'
        f'    schema_sql: [db/schema.sql]\n')
    cfg = config.Config(config_path=yaml_path)
    assert cfg.get_source_config('shop').schema_sql == ('db/schema.sql',)


def test_persist_data_model_promotes_against_a_configured_schema(tmp_path):
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(
        'class Order(models.Model):\n'
        '    status = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "orders"\n')
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}))
    MP = 'scip-python python shop . shop/'
    ORDER, O_STATUS = MP + 'Order#', MP + 'Order#status.'
    doc = _ScipDoc(
        relative_path='shop/models.py',
        occurrences=(
            _ScipOccurrence(symbol=ORDER, range=(0, 6, 0, 11), is_definition=True),
            _ScipOccurrence(symbol=O_STATUS, range=(1, 4, 1, 10), is_definition=True),
        ),
        symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    # a checked-in Postgres schema dump confirming the derived 'status' column
    (tmp_path / 'schema.sql').write_text('CREATE TABLE orders (status TEXT NOT NULL);')
    db = tmp_path / 'ariadne.db'

    persist_data_model(
        db, [('shop', tmp_path)],
        index_factory=lambda *a, **k: index,
        # a missing path is tolerated alongside the real dump.
        schema_paths_by_source={'shop': ['schema.sql', 'missing.sql']})

    conn = sqlite3.connect(db)
    conf = dict(conn.execute(
        "SELECT canonical_id, confidence FROM schema_symbols WHERE column_name='status'"))
    conn.close()
    # ORM-derived 'status', confirmed by the configured schema → resolved.
    assert conf.get('data sql shop _._.orders#status') == 'resolved'


def test_persist_data_model_stores_gaps_for_diagnosis(tmp_path):
    """persist_data_model must COLLECT the persist functions' gaps and store
    them (data_model_gaps), not discard them — so 'surface, don't guess' (§3a/
    §5.0) is reachable. Here a model column the schema lacks → a drift gap."""
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(
        'class Order(models.Model):\n'
        '    status = models.CharField()\n'
        '    legacy = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "orders"\n')
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}))
    MP = 'scip-python python shop . shop/'
    ORDER, O_STATUS, O_LEGACY = MP + 'Order#', MP + 'Order#status.', MP + 'Order#legacy.'
    doc = _ScipDoc(relative_path='shop/models.py', occurrences=(
        _ScipOccurrence(symbol=ORDER, range=(0, 6, 0, 11), is_definition=True),
        _ScipOccurrence(symbol=O_STATUS, range=(1, 4, 1, 10), is_definition=True),
        _ScipOccurrence(symbol=O_LEGACY, range=(2, 4, 2, 10), is_definition=True)), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    (tmp_path / 'schema.sql').write_text('CREATE TABLE orders (status TEXT);')
    db = tmp_path / 'ariadne.db'
    persist_data_model(db, [('shop', tmp_path)], index_factory=lambda *a, **k: index,
                       schema_paths_by_source={'shop': ['schema.sql']})
    conn = sqlite3.connect(db)
    gaps = [r[0] for r in conn.execute(
        "SELECT detail FROM data_model_gaps WHERE source_name = 'shop'")]
    conn.close()
    assert any('legacy' in g for g in gaps)  # stored, not discarded


def test_persist_data_model_promotes_from_discovered_migrations(tmp_path):
    """persist_data_model auto-discovers Django migrations (<app>/migrations/
    *.py) and uses them as the design-faithful promotion witness — no config,
    unlike the .sql dump. A derived model column the migration confirms → resolved."""
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(
        'class Order(models.Model):\n'
        '    status = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "orders"\n')
    (tmp_path / 'shop' / 'migrations').mkdir()
    (tmp_path / 'shop' / 'migrations' / '__init__.py').write_text('')
    (tmp_path / 'shop' / 'migrations' / '0001_initial.py').write_text(
        "class Migration:\n"
        "    operations = [migrations.CreateModel(name='Order',\n"
        "        fields=[('status', models.CharField())],\n"
        "        options={'db_table': 'orders'})]\n")
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}))
    MP = 'scip-python python shop . shop/'
    ORDER, O_STATUS = MP + 'Order#', MP + 'Order#status.'
    doc = _ScipDoc(relative_path='shop/models.py', occurrences=(
        _ScipOccurrence(symbol=ORDER, range=(0, 6, 0, 11), is_definition=True),
        _ScipOccurrence(symbol=O_STATUS, range=(1, 4, 1, 10), is_definition=True)), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    db = tmp_path / 'ariadne.db'
    persist_data_model(db, [('shop', tmp_path)], index_factory=lambda *a, **k: index)
    conn = sqlite3.connect(db)
    conf = dict(conn.execute(
        "SELECT canonical_id, confidence FROM schema_symbols WHERE column_name = 'status'"))
    conn.close()
    assert conf.get('data sql shop _._.orders#status') == 'resolved'


def test_source_config_parses_sql_dialect(tmp_path):
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n  shop:\n    path: {tmp_path}\n    sql_dialect: mysql\n')
    cfg = config.Config(config_path=yaml_path)
    assert cfg.get_source_config('shop').sql_dialect == 'mysql'


def test_persist_data_model_parses_schema_in_the_configured_dialect(tmp_path):
    """A MySQL schema dump (backtick identifiers) only parses under the mysql
    dialect; persist_data_model must use the per-source sql_dialect so it
    promotes (vs the default postgres, which can't parse it). C7 multi-dialect."""
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(
        'class Order(models.Model):\n'
        '    status = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "orders"\n')
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}))
    (tmp_path / 'schema.sql').write_text('CREATE TABLE `orders` (`status` VARCHAR(20))')
    MP = 'scip-python python shop . shop/'
    ORDER, O_STATUS = MP + 'Order#', MP + 'Order#status.'
    doc = _ScipDoc(relative_path='shop/models.py', occurrences=(
        _ScipOccurrence(symbol=ORDER, range=(0, 6, 0, 11), is_definition=True),
        _ScipOccurrence(symbol=O_STATUS, range=(1, 4, 1, 10), is_definition=True)), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    db = tmp_path / 'ariadne.db'
    persist_data_model(db, [('shop', tmp_path)], index_factory=lambda *a, **k: index,
                       schema_paths_by_source={'shop': ['schema.sql']},
                       dialect_by_source={'shop': 'mysql'})
    conn = sqlite3.connect(db)
    conf = dict(conn.execute(
        "SELECT canonical_id, confidence FROM schema_symbols WHERE column_name = 'status'"))
    conn.close()
    assert conf.get('data sql shop _._.orders#status') == 'resolved'


def test_persist_data_model_composes_all_witnesses(tmp_path):
    """Full-chain composition gap-check: ORM Layer-1/2 + raw-SQL + the migration
    witness all run in one persist_data_model without interfering — the migration
    promotes the derived ORM column, the raw-SQL access is recorded."""
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(
        'class Order(models.Model):\n'
        '    status = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "orders"\n')
    (tmp_path / 'shop' / 'migrations').mkdir()
    (tmp_path / 'shop' / 'migrations' / '0001.py').write_text(
        "class Migration:\n"
        "    operations = [migrations.CreateModel(name='Order',\n"
        "        fields=[('status', models.CharField())], options={'db_table': 'orders'})]\n")
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}))
    MP = 'scip-python python shop . shop/'
    ORDER, O_STATUS = MP + 'Order#', MP + 'Order#status.'
    doc = _ScipDoc(relative_path='shop/models.py', occurrences=(
        _ScipOccurrence(symbol=ORDER, range=(0, 6, 0, 11), is_definition=True),
        _ScipOccurrence(symbol=O_STATUS, range=(1, 4, 1, 10), is_definition=True)), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    db = tmp_path / 'ariadne.db'
    lib = Library(db)
    with lib._conn_provider.acquire() as c:
        c.execute('INSERT INTO string_literals (source_name, file, line_start, '
                  'col_start, value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
                  ('shop', 'q.py', 1, 0, 'SELECT status FROM orders', MP + 'q().'))
        c.commit()
    lib.close()
    persist_data_model(db, [('shop', tmp_path)], index_factory=lambda *a, **k: index)
    conn = sqlite3.connect(db)
    status_conf = dict(conn.execute(
        "SELECT canonical_id, confidence FROM schema_symbols WHERE column_name = 'status'"))
    rawsql_access = list(conn.execute(
        "SELECT 1 FROM data_access WHERE witness = 'rawsql' AND source_name = 'shop'"))
    conn.close()
    assert status_conf.get('data sql shop _._.orders#status') == 'resolved'  # migration promoted
    assert rawsql_access  # raw-SQL ran + composed


def test_persist_data_model_discovers_alembic_migrations(tmp_path):
    """persist_data_model auto-discovers Alembic migrations at ``versions/*.py``
    and promotes the __tablename__-less SQLAlchemy columns they confirm — the
    SQLAlchemy-models + Alembic, no-__tablename__ path."""
    (tmp_path / 'studio').mkdir()
    (tmp_path / 'studio' / 'models.py').write_text(
        'class Feature(Base):\n'
        '    id = Column(String, primary_key=True)\n'
        '    name = Column(String)\n')
    (tmp_path / 'studio' / 'alembic' / 'versions').mkdir(parents=True)
    (tmp_path / 'studio' / 'alembic' / 'versions' / '0001_init.py').write_text(
        'def upgrade():\n'
        '    op.create_table("feature",\n'
        '        sa.Column("id", sa.String()),\n'
        '        sa.Column("name", sa.String()))\n')
    (tmp_path / 'studio' / 'alembic' / 'versions' / '__init__.py').write_text('')
    (tmp_path / '.ariadne').mkdir()
    (tmp_path / '.ariadne' / 'manifest.json').write_text(
        json.dumps({'indexers': [{'kind': 'python', 'scip_path': 'index.scip'}]}))
    MP = 'scip-python python studio . studio/'
    FEAT, F_ID, F_NAME = MP + 'Feature#', MP + 'Feature#id.', MP + 'Feature#name.'
    doc = _ScipDoc(relative_path='studio/models.py', occurrences=(
        _ScipOccurrence(symbol=FEAT, range=(0, 6, 0, 13), is_definition=True),
        _ScipOccurrence(symbol=F_ID, range=(1, 4, 1, 6), is_definition=True),
        _ScipOccurrence(symbol=F_NAME, range=(2, 4, 2, 8), is_definition=True)), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    db = tmp_path / 'ariadne.db'
    persist_data_model(db, [('studio', tmp_path)], index_factory=lambda *a, **k: index)
    conn = sqlite3.connect(db)
    conf = dict(conn.execute(
        'SELECT canonical_id, confidence FROM schema_symbols '
        "WHERE column_name IN ('id', 'name')"))
    conn.close()
    # the auto-discovered Alembic migration promoted the derived SQLAlchemy columns
    assert conf['data sql studio _._.feature#id'] == 'resolved'
    assert conf['data sql studio _._.feature#name'] == 'resolved'


def test_persist_data_model_honors_staleness_exemption(tmp_path):
    """A staleness-exempt source (ignore_staleness: true) loads its SCIP index
    with the age gate DISABLED — persist_data_model threads max_staleness_days=None
    through to the loader, so an unchanged-but-old index is reused, not refused."""
    (tmp_path / '.ariadne' / 'intermediate').mkdir(parents=True)
    (tmp_path / '.ariadne' / 'manifest.json').write_text(json.dumps(
        {'indexers': [{'kind': 'python', 'cwd': '.', 'scip_path': 'intermediate/x.scip'}]}))
    seen = {}

    def factory(scip_path, repo=None, max_staleness_days='UNSET'):
        seen['max_staleness_days'] = max_staleness_days
        return ScipIndex(documents=(), source_root=scip_path.parent)

    persist_data_model(tmp_path / 'ariadne.db', [('src', tmp_path)],
                       index_factory=factory, max_staleness_by_source={'src': None})
    # the exempt source's index is loaded with the age gate off
    assert seen['max_staleness_days'] is None


def test_persist_data_model_surfaces_gap_when_manifest_declares_unbuilt_indexes(
    tmp_path,
):
    """A manifest that declares indexers but has NONE built (no ``scip_path`` on
    any entry — e.g. ``discover`` ran, ``ariadne index`` did not) must SURFACE a
    ``data_model_gap``, not silently write 0 rows. The gap reports the shortfall
    and the fix command — 'surface, don't guess' (§3a/§5.0), so a stale/unbuilt
    index is distinguishable from a source that genuinely has no data model."""
    (tmp_path / '.ariadne').mkdir(parents=True)
    (tmp_path / '.ariadne' / 'manifest.json').write_text(json.dumps({
        'indexers': [
            {'kind': 'scala', 'cwd': '.', 'markers': ['build.sbt']},
            {'kind': 'python', 'cwd': 'be', 'markers': ['be/__init__.py']},
        ],
    }), encoding='utf-8')
    db = tmp_path / 'ariadne.db'

    rows = persist_data_model(db, [('src1', tmp_path)])

    assert rows == 0  # nothing bound — no index artifact was built
    conn = sqlite3.connect(db)
    try:
        gaps = [r[0] for r in conn.execute(
            'SELECT detail FROM data_model_gaps WHERE source_name = ?',
            ('src1',))]
    finally:
        conn.close()
    # the unbuilt-index gap, naming the shortfall (0 of 2) and the fix command
    assert any('0 of 2' in g and 'ariadne index' in g for g in gaps), gaps
