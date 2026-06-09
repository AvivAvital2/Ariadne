"""persist_config_values — Tier 2 Feature 1.

Wires ingest_config_values into the persist layer so the config_values table is
actually populated during indexing (it's currently empty because nothing calls
the extractor). Mirrors persist_string_literals. Synthetic, agnostic fixtures.
See designs/config-code-bridge/tier2-resolution.md (Feature 1).
"""
from __future__ import annotations

import pytest


@pytest.fixture
def db(tmp_path):
    from library import Library
    from library.scip import init_scip_schema

    p = tmp_path / 'cfg.db'
    lib = Library(p)
    with lib._conn_provider.acquire() as conn:
        init_scip_schema(conn)
        conn.commit()
    lib.close()
    return p


def test_persists_hocon_config_values(db, tmp_path) -> None:
    from docgen.scip_config_index import query_config_values_by_key
    from docgen.scip_persist import persist_config_values
    from library import Library

    src = tmp_path / 'src1'
    src.mkdir()
    (src / 'reference.conf').write_text('svc {\n  timeout = 30\n}\n', encoding='utf-8')

    n = persist_config_values(db, [('src1', src)])
    assert n >= 1

    lib = Library(db)
    try:
        with lib._conn_provider.acquire() as conn:
            rows = query_config_values_by_key(source_name='src1', key='svc.timeout', conn=conn)
        assert any(r.value == '30' for r in rows)
    finally:
        lib.close()


def test_source_without_config_contributes_zero(db, tmp_path) -> None:
    from docgen.scip_persist import persist_config_values

    empty = tmp_path / 'empty'
    empty.mkdir()
    assert persist_config_values(db, [('src1', empty)]) == 0
