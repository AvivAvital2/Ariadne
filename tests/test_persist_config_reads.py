"""persist_config_reads (index-loop wrapper) — Tier 2 Feature 5.

Wraps extract_config_reads + the config_reads persistence into one
per-source step for cmd_index, mirroring persist_config_values. It reads
the already-populated string_literals + config_values (seeded here as the
earlier persist steps would) and the source files off disk. Synthetic
fixtures only. See designs/config-code-bridge/tier2-resolution.md (Feature 5).
"""
from __future__ import annotations

from pathlib import Path

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


def _seed(db, source, literals, config_values) -> None:
    """literals: (file, line, col, value); config_values: (file, key, value, line)."""
    from docgen.scip_config_index import persist_config_values
    from docgen.scip_config_scanners import ConfigValue
    from docgen.scip_string_literal_index import (
        StringLiteral,
        persist_string_literals,
    )
    from library import Library

    lib = Library(db)
    try:
        with lib._conn_provider.acquire() as conn:
            persist_string_literals(
                source_name=source,
                literals=[
                    StringLiteral(
                        file=Path(f), line_start=ln, col_start=c,
                        value=v, owning_symbol_id=None,
                    )
                    for (f, ln, c, v) in literals
                ],
                conn=conn,
            )
            persist_config_values(
                source_name=source,
                config_values=[
                    ConfigValue(file=Path(f), key=k, value=v, line_start=ln)
                    for (f, k, v, ln) in config_values
                ],
                conn=conn,
            )
            conn.commit()
    finally:
        lib.close()


def test_persists_config_reads(db, tmp_path) -> None:
    from docgen.scip_config_index import query_config_reads_by_key
    from docgen.scip_persist import persist_config_reads
    from library import Library

    src = tmp_path / 'src1'
    src.mkdir()
    reader = src / 'reader.scala'
    reader.write_text(
        'val ttl = cfg.getString("svc.cache.ttl")\n', encoding='utf-8',
    )
    _seed(
        db, 'src1',
        [(str(reader), 1, 24, 'svc.cache.ttl')],
        [('app.conf', 'svc.cache.ttl', '30', 5)],
    )

    assert persist_config_reads(db, [('src1', src)]) == 1

    lib = Library(db)
    try:
        with lib._conn_provider.acquire() as conn:
            got = query_config_reads_by_key(
                source_name='src1', key='svc.cache.ttl', conn=conn,
            )
        assert len(got) == 1
        assert got[0].value == '30'
        assert got[0].confidence == 'config-resolved'
    finally:
        lib.close()


def test_source_without_reads_contributes_zero(db, tmp_path) -> None:
    from docgen.scip_persist import persist_config_reads

    empty = tmp_path / 'empty'
    empty.mkdir()
    assert persist_config_reads(db, [('src1', empty)]) == 0
