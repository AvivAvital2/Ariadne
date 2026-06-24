"""Phase 2 — the derived→resolved promotion via a parsed Postgres schema
(design §3a line 230, §4 line 289-294, §5.0.1 bullet 2).

A parsed ``CREATE TABLE`` is ``exact`` ground truth (§3a line 202). Cross-checking
it against the ORM-derived ``schema_symbols`` is the recall-saver: a ``derived``
field-name column the schema confirms is promoted to ``resolved`` (and so it
surfaces at the default floor), the ORM's ``producer_symbol_id`` is preserved
(DDL is the name authority, the ORM keeps the binding), a column the schema
declares but no model covers is inserted ``exact``, and a model column the schema
lacks is held ``derived`` and flagged as drift.

End-to-end over two real parsers: ``persist_schema_symbols`` (ORM) →
``persist_schema_ddl`` (DDL) → ``add_data_layer``. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.orm_bindings import (
    DjangoStrategy,
    SQLAlchemyStrategy,
    persist_schema_symbols,
)
from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence
from docgen.sql_access import persist_data_access_rawsql
from docgen.sql_query_views import dead_columns
from docgen.sql_schema import (
    parse_alembic_migrations,
    parse_django_migrations,
    parse_schema_ddl,
    persist_schema_ddl,
    persist_schema_from_alembic,
    persist_schema_from_migrations,
)
from library.scip import init_scip_schema

P = 'scip-python python billing . billing/'
USER, U_EMAIL, U_NAME, U_LEGACY = (
    P + 'User#', P + 'User#email.', P + 'User#name.', P + 'User#legacy.')
DEFS = {USER: 0, U_EMAIL: 1, U_NAME: 2, U_LEGACY: 3}

MODELS_SRC = '''\
class User(models.Model):
    email = models.EmailField(db_column="email_addr")
    name = models.CharField()
    legacy = models.CharField()
    class Meta:
        db_table = "users"
'''

# Postgres schema: confirms email_addr + name, declares created_at (no model),
# and LACKS 'legacy' (the model has a column the DB doesn't → drift).
SCHEMA_SQL = '''
CREATE TABLE users (
    email_addr VARCHAR(255) NOT NULL,
    name TEXT,
    created_at TIMESTAMP
);
'''

T = 'data sql billing _._.'


@pytest.fixture
def conn(tmp_path):
    (tmp_path / 'billing').mkdir()
    (tmp_path / 'billing' / 'models.py').write_text(MODELS_SRC)
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    for sym, line in DEFS.items():
        c.execute(
            'INSERT INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
            (sym, 'billing', 'python', 'billing/models.py', line + 1, line + 1,
             '', sym.rstrip('#.').rsplit('#', 1)[-1] or sym, sym, None),
        )
    doc = _ScipDoc(
        relative_path='billing/models.py',
        occurrences=tuple(
            _ScipOccurrence(symbol=s, range=(ln, 4, ln, 8), is_definition=True)
            for s, ln in DEFS.items()
        ),
        symbols=(),
    )
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    # Layer 1: ORM-derived schema_symbols.
    persist_schema_symbols(c, 'billing', index, strategies=[DjangoStrategy()])
    yield c
    c.close()


def _rows(conn):
    return {
        r[0]: {'node_type': r[1], 'column_type': r[2], 'producer': r[3],
               'confidence': r[4]}
        for r in conn.execute(
            'SELECT canonical_id, node_type, column_type, producer_symbol_id, '
            'confidence FROM schema_symbols')
    }


def test_parse_schema_ddl_extracts_tables_and_columns():
    tables = {t['table']: t for t in parse_schema_ddl(SCHEMA_SQL)}
    assert set(tables) == {'users'}
    cols = {c['name']: c for c in tables['users']['columns']}
    assert set(cols) == {'email_addr', 'name', 'created_at'}
    assert cols['email_addr']['nullable'] is False  # NOT NULL
    assert cols['name']['nullable'] is True
    assert 'VARCHAR' in cols['email_addr']['type']


def test_parse_schema_ddl_handles_mixed_statements_and_bad_input():
    sql = '''
    CREATE TABLE accounts (id SERIAL, owner TEXT, PRIMARY KEY (id));
    CREATE INDEX idx_owner ON accounts (owner);
    INSERT INTO accounts VALUES (1, 'x');
    CREATE TABLE t_empty ();
    '''
    tables = {t['table']: t for t in parse_schema_ddl(sql)}
    # only the real CREATE TABLE with columns is captured; the index, the
    # insert, and the empty table are skipped.
    assert set(tables) == {'accounts'}
    # the table-level PRIMARY KEY(id) constraint is not mistaken for a column.
    assert {c['name'] for c in tables['accounts']['columns']} == {'id', 'owner'}
    # malformed input is swallowed (returns nothing), never raised.
    assert parse_schema_ddl('@@@ not ; valid %%%') == []


def test_schema_promotes_confirmed_derived_columns(conn):
    # Baseline (ORM only): name + legacy are derived, email_addr is exact.
    before = _rows(conn)
    assert before[T + 'users#name']['confidence'] == 'derived'
    assert before[T + 'users#legacy']['confidence'] == 'derived'
    assert before[T + 'users#email_addr']['confidence'] == 'exact'

    result = persist_schema_ddl(conn, 'billing', SCHEMA_SQL)
    after = _rows(conn)

    # the schema CONFIRMS the derived field-name column → promoted to resolved,
    # and the ORM's producer binding is preserved (DDL is the name authority,
    # the ORM keeps producer_symbol_id).
    assert after[T + 'users#name']['confidence'] == 'resolved'
    assert after[T + 'users#name']['producer'] == U_NAME
    assert after[T + 'users#name']['column_type'] == 'TEXT'

    # already-exact column stays exact (both agree); type comes from the DDL.
    assert after[T + 'users#email_addr']['confidence'] == 'exact'

    # a column the schema declares but no model covers → inserted exact.
    assert after[T + 'users#created_at']['confidence'] == 'exact'
    assert after[T + 'users#created_at']['producer'] is None

    # a model column the schema LACKS → held derived + flagged as drift.
    assert after[T + 'users#legacy']['confidence'] == 'derived'
    assert any('legacy' in g for g in result.gaps)
    assert result.promoted >= 1 and result.declared >= 1


def test_promoted_column_surfaces_at_the_default_floor(conn):
    # Before promotion the derived 'name' is held below the resolved floor.
    g0 = CrossSourceGraph()
    g0.load_from(conn)
    g0.add_data_layer(conn)
    assert T + 'users#name' not in g0._symbols

    persist_schema_ddl(conn, 'billing', SCHEMA_SQL)

    g1 = CrossSourceGraph()
    g1.load_from(conn)
    g1.add_data_layer(conn)
    # promoted → resolved → now asserted at the default floor.
    assert T + 'users#name' in g1._symbols
    # the unconfirmed (drift) column stays held back.
    assert T + 'users#legacy' not in g1._symbols


def test_query_side_cross_check_flags_observed_undeclared_columns():
    """A query touching a column the schema doesn't declare = observed-but-
    undeclared (typo/drift), §6 line 376. Flagged with the accessing consumer.
    End-to-end: a raw-SQL query (real persist) touching a real column + a typo,
    cross-checked against a parsed schema."""
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    conn.execute(
        'INSERT INTO string_literals (source_name, file, line_start, col_start, '
        'value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
        ('app', 'q.py', 5, 4, 'SELECT email, emial FROM users',
         'scip-python python app . app/q().'))
    conn.commit()
    persist_data_access_rawsql(conn, 'app')

    result = persist_schema_ddl(conn, 'app', 'CREATE TABLE users (email TEXT);')

    # the typo column is queried but not declared → flagged WITH the consumer.
    assert any('emial' in g and 'q().' in g and 'not in schema' in g
               for g in result.gaps)
    # the real, declared column is NOT flagged on the query side.
    assert not any('users.email ' in g and 'queried' in g for g in result.gaps)
    conn.close()


def test_dead_columns_reports_declared_columns_no_one_accesses():
    """A declared column (schema, at/above the floor) that no code reads or
    writes — no data_access row — is a dead column (§10 Phase 2)."""
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    conn.execute(
        'INSERT INTO string_literals (source_name, file, line_start, col_start, '
        'value, owning_symbol_id) VALUES (?,?,?,?,?,?)',
        ('app', 'q.py', 1, 0, 'SELECT email, ghost FROM users',
         'scip-python python app . app/q().'))
    conn.commit()
    persist_data_access_rawsql(conn, 'app')
    persist_schema_ddl(conn, 'app', 'CREATE TABLE users (email TEXT, unused TEXT);')
    # 'unused' is declared but accessed by nobody → dead; 'email' is accessed;
    # 'ghost' is derived (below the floor) → not a declared column.
    assert dead_columns(conn, 'app') == [('users', 'unused')]
    conn.close()


def test_django_migration_witness_promotes_confirmed_columns(tmp_path):
    """Django migrations (committed output) are a name-authority witness: a
    derived model column the migration confirms is promoted to resolved — the
    design-faithful per-ORM promotion (§5.0.1 bullet 2), no Postgres dump."""
    (tmp_path / 'shop').mkdir()
    (tmp_path / 'shop' / 'models.py').write_text(
        'class Order(models.Model):\n'
        '    status = models.CharField()\n'
        '    legacy = models.CharField()\n'
        '    class Meta:\n'
        '        db_table = "orders"\n')
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    MP = 'scip-python python shop . shop/'
    ORDER, O_STATUS, O_LEGACY = MP + 'Order#', MP + 'Order#status.', MP + 'Order#legacy.'
    doc = _ScipDoc(relative_path='shop/models.py', occurrences=(
        _ScipOccurrence(symbol=ORDER, range=(0, 6, 0, 11), is_definition=True),
        _ScipOccurrence(symbol=O_STATUS, range=(1, 4, 1, 10), is_definition=True),
        _ScipOccurrence(symbol=O_LEGACY, range=(2, 4, 2, 10), is_definition=True)), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    persist_schema_symbols(conn, 'shop', index, strategies=[DjangoStrategy()])
    assert {r[0]: r[1] for r in conn.execute(
        "SELECT canonical_id, confidence FROM schema_symbols WHERE column_name='status'"
    )}['data sql shop _._.orders#status'] == 'derived'

    migration = (
        "class Migration:\n"
        "    operations = [\n"
        "        migrations.CreateModel(name='Order',\n"
        "            fields=[('status', models.CharField())],\n"
        "            options={'db_table': 'orders'})]\n")
    result = persist_schema_from_migrations(conn, 'shop', [('shop', migration)])

    after = {r[0]: r[1] for r in conn.execute(
        "SELECT canonical_id, confidence FROM schema_symbols WHERE node_type='column'")}
    # the migration confirms 'status' -> promoted; 'legacy' (not in it) -> drift
    assert after['data sql shop _._.orders#status'] == 'resolved'
    assert after['data sql shop _._.orders#legacy'] == 'derived'
    assert any('legacy' in g for g in result.gaps)
    assert result.promoted >= 1


def test_parse_django_migrations_edge_cases():
    """Branch coverage of the migration parser: db_table absent -> derive
    app_label_model; malformed field entries skipped; a CreateModel with no
    name or no fields skipped; unparsable code skipped."""
    code = (
        "migrations.CreateModel(name='Item', fields=[\n"
        "    ('sku', models.CharField()),\n"      # kept
        "    'bad',\n"                              # not a tuple -> skip
        "    ('x', 1),\n"                           # value not a Call -> skip
        "    (123, models.TextField()),\n"         # name not a string -> skip
        "    ('a', 'b', 'c'),\n"                    # not a 2-tuple -> skip
        "    ('qty', models.IntegerField())])\n"   # kept
        "migrations.CreateModel(fields=[('z', models.CharField())])\n"  # no name -> skip
        "migrations.CreateModel(name='Empty')\n"   # no fields -> skip
    )
    tables = {t['table']: t for t in
              parse_django_migrations([('shop', code), ('app', 'def (((')])}
    assert set(tables) == {'shop_item'}  # no db_table -> derived app_label_model
    assert {c['name'] for c in tables['shop_item']['columns']} == {'sku', 'qty'}


def test_migration_table_derivation_when_options_lack_usable_db_table():
    """options dict present but without a usable db_table (a non-db_table key,
    or a non-string db_table value) -> derive Django's app_label_model."""
    code = (
        "migrations.CreateModel(name='Opt', fields=[('a', models.CharField())],\n"
        "    options={'ordering': ['a'], 'db_table': 123})\n")
    tables = parse_django_migrations([('shop', code)])
    assert tables == [{'table': 'shop_opt', 'columns': [
        {'name': 'a', 'type': None, 'nullable': None, 'primary_key': None}]}]


SA_P = 'scip-python python studio . studio/'
SAF, SAF_ID, SAF_NAME, SAF_LEGACY = (
    SA_P + 'Feature#', SA_P + 'Feature#id.', SA_P + 'Feature#name.', SA_P + 'Feature#legacy.')

SA_MODEL = '''\
class Feature(Base):
    id = Column(String, primary_key=True)
    name = Column(String(255))
    legacy = Column(String)
'''

ALEMBIC_MIG = '''\
revision = "01"
down_revision = None


def upgrade():
    op.create_table(
        "feature",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
'''


def test_alembic_migration_witness_promotes_derived_sqlalchemy_columns(tmp_path):
    """Alembic ``op.create_table`` is a name-authority witness: a derived,
    __tablename__-less SQLAlchemy column the migration confirms is promoted to
    resolved — the path that makes a real source's derived columns assert. A distinct
    witness from Django's, so the two never clobber each other (§4/§5.0.1)."""
    (tmp_path / 'studio').mkdir()
    (tmp_path / 'studio' / 'models.py').write_text(SA_MODEL)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    defs = {SAF: 0, SAF_ID: 1, SAF_NAME: 2, SAF_LEGACY: 3}
    doc = _ScipDoc(relative_path='studio/models.py', occurrences=tuple(
        _ScipOccurrence(symbol=s, range=(ln, 6, ln, 12), is_definition=True)
        for s, ln in defs.items()), symbols=())
    index = ScipIndex(documents=(doc,), source_root=tmp_path)
    persist_schema_symbols(conn, 'studio', index, strategies=[SQLAlchemyStrategy()])
    T2 = 'data sql studio _._.'
    before = {r[0]: r[1] for r in conn.execute(
        'SELECT canonical_id, confidence FROM schema_symbols')}
    assert before[T2 + 'feature'] == 'derived'        # __tablename__-less table
    assert before[T2 + 'feature#name'] == 'derived'

    result = persist_schema_from_alembic(conn, 'studio', [ALEMBIC_MIG])

    after = {r[0]: r[1] for r in conn.execute(
        'SELECT canonical_id, confidence FROM schema_symbols')}
    # the migration confirms the table + id + name -> promoted to resolved
    assert after[T2 + 'feature'] == 'resolved'
    assert after[T2 + 'feature#id'] == 'resolved'
    assert after[T2 + 'feature#name'] == 'resolved'
    # 'legacy' (not in the migration) -> held derived + drift gap
    assert after[T2 + 'feature#legacy'] == 'derived'
    assert any('legacy' in g for g in result.gaps)
    assert result.promoted >= 1


def test_parse_alembic_migrations_edge_cases():
    """Branch coverage: create_table columns (sa.Column) vs constraints (skipped),
    add_column, a non-literal table name skipped, a Column with no string name
    skipped, a 1-arg add_column skipped, an empty create_table, drop_column / a
    non-op call ignored, unparsable code skipped. alter_column with
    new_column_name declares the renamed-to column; a non-rename alter_column
    (no new_column_name) and a non-literal-table rename are skipped."""
    code = '''\
def upgrade():
    op.create_table(
        "feature",
        sa.Column("id", sa.Integer()),
        sa.Column("name", sa.String()),
        sa.Column(dynamic),
        sa.UniqueConstraint("id"),
    )
    op.add_column("feature", sa.Column("flag", sa.Boolean()))
    op.add_column("feature")
    op.create_table(dynamic, sa.Column("z"))
    op.create_table("empty")
    op.drop_column("feature", "old")
    op.alter_column("feature", "name", new_column_name="title")
    op.alter_column("feature", "flag", nullable=False)
    op.alter_column(dynamic, "x", new_column_name="y")
    other.create_table("x", sa.Column("y"))
'''
    tables = parse_alembic_migrations([code, 'def ((('])
    assert {t['table'] for t in tables} == {'feature'}   # empty/x/dynamic excluded
    cols = {c['name'] for t in tables for c in t['columns']}
    # 'title' added by the rename; non-rename + non-literal-table alter skipped
    assert cols == {'id', 'name', 'flag', 'title'}


def test_alembic_alter_column_rename_declares_post_rename_column():
    """``op.alter_column(table, old, new_column_name=new)`` is a rename witness:
    the post-rename column is declared so code referencing the new name resolves.
    Without full ordered replay the old name lingers — a documented §10 P4
    tradeoff (declare the new, don't drop the old)."""
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    migrations = [
        'def upgrade():\n'
        '    op.create_table("widget",\n'
        '        sa.Column("id", sa.Integer()),\n'
        '        sa.Column("old_label", sa.String()))\n',
        'def upgrade():\n'
        '    op.alter_column("widget", "old_label", new_column_name="label")\n',
    ]
    persist_schema_from_alembic(c, 'shop', migrations)
    cols = {r[0] for r in c.execute(
        "SELECT canonical_id FROM schema_symbols "
        "WHERE canonical_id LIKE '%widget#%'")}
    c.close()
    base = 'data sql shop _._.widget#'
    assert base + 'label' in cols      # renamed-to column declared → resolves
    assert base + 'old_label' in cols  # old name lingers (no replay, §10 P4)
