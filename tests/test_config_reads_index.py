"""Contract for the config_reads index (Tier 2 Feature 4).

Persists the config-getter read sites enumerated by Feature 3
(``extract_config_reads`` → :class:`ConfigRead`) to the ``config_reads``
table, queryable by key. Mirrors the ``config_values`` index
(persist clears the source's prior rows; query-by-key returns all matches
since one key is read from many sites). Synthetic fixtures only.

See designs/config-code-bridge/tier2-resolution.md (Feature 4).
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _read(file, line, col, key, value, confidence):
    from docgen.scip_config_usage_extractor import ConfigRead

    return ConfigRead(
        file=Path(file), line=line, col=col, key=key,
        value=value, confidence=confidence,
    )


def test_schema_table_and_index_created(conn) -> None:
    # init_scip_schema (run by the fixture) creates config_reads + an
    # index covering (source_name, key) for the query path.
    table = conn.execute(
        "SELECT name FROM sqlite_master "
        "WHERE type='table' AND name='config_reads'"
    ).fetchone()
    assert table is not None
    indexes = {
        row[0] for row in conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='config_reads'"
        ).fetchall()
    }
    assert any('source' in n or 'key' in n for n in indexes)


def test_persist_and_query_by_key(conn) -> None:
    from docgen.scip_config_index import (
        persist_config_reads,
        query_config_reads_by_key,
    )

    reads = [
        _read('a.scala', 1, 24, 'svc.cache.ttl', '30', 'config-resolved'),
        _read('b.scala', 9, 12, 'svc.cache.ttl', '30', 'config-resolved'),
    ]
    assert persist_config_reads(
        source_name='src1', config_reads=reads, conn=conn,
    ) == 2

    got = query_config_reads_by_key(
        source_name='src1', key='svc.cache.ttl', conn=conn,
    )
    assert len(got) == 2  # one key, many read sites
    assert {str(r.file) for r in got} == {'a.scala', 'b.scala'}
    assert got[0].confidence == 'config-resolved'


def test_undeclared_value_round_trips_as_null(conn) -> None:
    from docgen.scip_config_index import (
        persist_config_reads,
        query_config_reads_by_key,
    )

    persist_config_reads(
        source_name='src1',
        config_reads=[
            _read('c.scala', 3, 8, 'svc.cache.size', None, 'config-resolved'),
        ],
        conn=conn,
    )
    got = query_config_reads_by_key(
        source_name='src1', key='svc.cache.size', conn=conn,
    )
    assert len(got) == 1
    assert got[0].value is None


def test_unknown_key_returns_empty(conn) -> None:
    from docgen.scip_config_index import query_config_reads_by_key

    assert query_config_reads_by_key(
        source_name='src1', key='absent.key', conn=conn,
    ) == []


def test_re_ingest_replaces_source_rows(conn) -> None:
    from docgen.scip_config_index import (
        persist_config_reads,
        query_config_reads_by_key,
    )

    persist_config_reads(
        source_name='src1',
        config_reads=[_read('a.scala', 1, 0, 'k1', 'v', 'string-match')],
        conn=conn,
    )
    persist_config_reads(
        source_name='src1',
        config_reads=[_read('b.scala', 1, 0, 'k2', 'w', 'config-resolved')],
        conn=conn,
    )
    assert query_config_reads_by_key(
        source_name='src1', key='k1', conn=conn,
    ) == []
    assert len(query_config_reads_by_key(
        source_name='src1', key='k2', conn=conn,
    )) == 1


def test_isolated_per_source(conn) -> None:
    from docgen.scip_config_index import (
        persist_config_reads,
        query_config_reads_by_key,
    )

    persist_config_reads(
        source_name='A',
        config_reads=[_read('a.scala', 1, 0, 'k', 'va', 'config-resolved')],
        conn=conn,
    )
    persist_config_reads(
        source_name='B',
        config_reads=[_read('b.scala', 1, 0, 'k', 'vb', 'config-resolved')],
        conn=conn,
    )
    assert query_config_reads_by_key(
        source_name='A', key='k', conn=conn,
    )[0].value == 'va'
    assert query_config_reads_by_key(
        source_name='B', key='k', conn=conn,
    )[0].value == 'vb'


def test_empty_input_clears_source(conn) -> None:
    from docgen.scip_config_index import (
        persist_config_reads,
        query_config_reads_by_key,
    )

    persist_config_reads(
        source_name='src1',
        config_reads=[_read('a.scala', 1, 0, 'k', 'v', 'config-resolved')],
        conn=conn,
    )
    assert persist_config_reads(
        source_name='src1', config_reads=[], conn=conn,
    ) == 0
    assert query_config_reads_by_key(
        source_name='src1', key='k', conn=conn,
    ) == []
