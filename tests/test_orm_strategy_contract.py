"""The ORM strategy contract (design §5.0): a typed ``OrmStrategy`` Protocol +
the ``DEFAULT_STRATEGIES`` registry. Adding an ORM is implementing the Protocol's
four members and registering one line — the shared engine does the rest, with no
engine change. This test pins that contract: the registered strategies conform,
and a minimal hand-rolled strategy plugs straight into the engine.
"""
from __future__ import annotations

import sqlite3

from docgen.orm_bindings import (
    Col,
    DEFAULT_STRATEGIES,
    OrmStrategy,
    Table,
    persist_schema_symbols,
)
from docgen.scip_extractor import ScipIndex
from library.scip import init_scip_schema


def test_default_strategies_conform_to_the_protocol():
    assert DEFAULT_STRATEGIES  # the registry is non-empty
    for strategy in DEFAULT_STRATEGIES:
        assert isinstance(strategy, OrmStrategy)  # runtime_checkable: has the members
    assert {'orm:django', 'orm:sqlalchemy', 'orm:slick'} <= {
        s.name for s in DEFAULT_STRATEGIES}


class _MinimalStrategy:
    """A whole new ORM is just these four members — nothing else."""

    name = 'orm:minimal'

    def detect(self, scip_index, root):
        return True

    def discover(self, scip_index, root, symbol_at):
        return [Table('Widget', 'widgets', 'sym:Widget#', 'exact',
                      (Col('sku', 'sym:Widget#sku.', 'exact', field_name='sku'),))], ()

    def recognize(self, scip_index, root, owning, schema, aliases):
        return [], ()


def test_a_protocol_conforming_strategy_plugs_into_the_engine(tmp_path):
    """The contract is sufficient: a strategy implementing ``OrmStrategy`` and
    returning ``Table``/``Col`` plugs into the shared engine with ZERO engine
    changes — the 'add an ORM' extensibility the Protocol guarantees."""
    assert isinstance(_MinimalStrategy(), OrmStrategy)
    conn = sqlite3.connect(':memory:')
    init_scip_schema(conn)
    index = ScipIndex(documents=(), source_root=tmp_path)

    result = persist_schema_symbols(
        conn, 'src1', index, strategies=[_MinimalStrategy()])

    rows = {r[0]: r[1] for r in conn.execute(
        'SELECT canonical_id, confidence FROM schema_symbols')}
    assert rows['data sql src1 _._.widgets'] == 'exact'
    assert rows['data sql src1 _._.widgets#sku'] == 'exact'
    assert result.nodes_written == 2
    conn.close()
