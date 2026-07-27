"""SQL that binds a caller-sized id list must chunk it, including the paths
that bind each id MORE than once.

Regression for the databricks spool build that crashed twice with
``sqlite3.OperationalError: too many SQL variables``:

  - the themes phase — ``graph_builder._delete_semantic_for_ids`` /
    ``_delete_semantic_within_ids`` bind ``ids + ids`` (source_id + target_id);
  - the pack build — ``spool_pack._copy_source_scip`` binds ``ids + ids``
    (caller + callee) when copying the source's SCIP edges.

The df70c85 fix chunked the read paths (``library.core``) but missed these
double-bind writes. All by-id SQL now shares ``library.sql_vars.chunk_ids``.

Each test caps SQLite's per-statement variable limit low (via ``setlimit`` on
every connection) so a few dozen ids reproduce the crash without needing tens
of thousands of rows.
"""
from __future__ import annotations

import sqlite3

import pytest

from library import Library
from library.scip import init_scip_schema
from library.sql_vars import SQL_MAX_VARS, chunk_ids

_VAR_CAP = 20   # SQLite per-statement variable ceiling for this test's conns
_CHUNK = 8      # chunk budget; copies=2 → 4 ids/chunk → 8 binds ≤ _VAR_CAP
_N = 40         # > _VAR_CAP → an unchunked ids+ids query (80 binds) blows it


@pytest.fixture
def capped(monkeypatch):
    """Cap SQLite's variable limit low on every connection and shrink the
    chunk budget so a modest id count exercises chunking."""
    real_connect = sqlite3.connect

    def _capped(*args, **kwargs):
        conn = real_connect(*args, **kwargs)
        conn.setlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER, _VAR_CAP)
        return conn

    monkeypatch.setattr(sqlite3, 'connect', _capped)
    monkeypatch.setattr('library.sql_vars.SQL_MAX_VARS', _CHUNK)


def test_chunk_ids_budget_copies_and_reserved():
    ids = list(range(100))
    # Round-trips every id, in order, never an empty chunk.
    assert [x for c in chunk_ids(ids) for x in c] == ids
    assert max(len(c) for c in chunk_ids(ids)) <= SQL_MAX_VARS
    # copies=2 halves the per-chunk budget (a chunk bound twice stays < limit).
    assert max(len(c) for c in chunk_ids(ids, copies=2)) <= SQL_MAX_VARS // 2
    # reserved is subtracted before chunking; budget never drops below 1.
    big_reserved = list(chunk_ids(ids, reserved=SQL_MAX_VARS - 3))
    assert all(len(c) >= 1 for c in big_reserved)
    assert [x for c in big_reserved for x in c] == ids
    assert list(chunk_ids([])) == []


def _semantic_edges(conn) -> set[tuple[str, str]]:
    return {
        (r[0], r[1]) for r in conn.execute(
            "SELECT source_id, target_id FROM doc_graph "
            "WHERE edge_type = 'semantic_neighbor'").fetchall()
    }


def _add_edge(conn, source: str, target: str) -> None:
    conn.execute(
        "INSERT INTO doc_graph (source_id, target_id, edge_type, weight) "
        "VALUES (?, ?, 'semantic_neighbor', 1.0)", (source, target))


def test_delete_semantic_for_ids_chunks_under_var_limit(capped, tmp_path):
    """OR-semantics: any edge TOUCHING an id is cleared (single-source refresh)."""
    from docgen.graph_builder import _delete_semantic_for_ids

    lib = Library(tmp_path / 'g1.db')
    try:
        with lib._conn_provider.acquire() as conn:
            _add_edge(conn, 'a0', 'out')        # source in ids → cleared
            _add_edge(conn, 'keep0', 'keep1')   # neither in ids → preserved
            conn.commit()
            ids = [f'a{i}' for i in range(_N)]   # 40 ids → 80 binds unchunked
            _delete_semantic_for_ids(conn, ids)  # must not raise
            edges = _semantic_edges(conn)
        assert ('a0', 'out') not in edges
        assert ('keep0', 'keep1') in edges
    finally:
        lib.close()


def test_delete_semantic_within_ids_stays_correct_across_chunks(capped, tmp_path):
    """Both-endpoints-in-set: an edge whose endpoints fall in DIFFERENT chunks
    must still be cleared (naive per-chunk AND would miss it), while an edge
    with only one endpoint in the set must survive (scoped spool rebuild)."""
    from docgen.graph_builder import _delete_semantic_within_ids

    lib = Library(tmp_path / 'g2.db')
    try:
        with lib._conn_provider.acquire() as conn:
            _add_edge(conn, 'a0', 'a30')        # both in ids, far-apart chunks
            _add_edge(conn, 'a1', 'out')        # only source in ids → preserved
            _add_edge(conn, 'keep0', 'keep1')   # neither in ids → preserved
            conn.commit()
            ids = [f'a{i}' for i in range(_N)]
            _delete_semantic_within_ids(conn, ids)  # must not raise
            edges = _semantic_edges(conn)
        assert ('a0', 'a30') not in edges       # both-in-set (cross-chunk) → cleared
        assert ('a1', 'out') in edges           # only one endpoint in set → kept
        assert ('keep0', 'keep1') in edges
    finally:
        lib.close()


def test_spool_pack_scip_edge_copy_chunks(capped, tmp_path):
    from spool_pack import _copy_source_scip

    src = Library(tmp_path / 'src.db')
    dest = Library(tmp_path / 'dest.db')
    try:
        syms = [f'sym{i}' for i in range(_N)]  # 40 symbols → 80 binds unchunked
        with src._conn_provider.acquire() as conn:
            init_scip_schema(conn)
            conn.executemany(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, '
                'file, line_start, line_end, kind, display_name, '
                'qualified_name, parent_qualified_name) '
                'VALUES (?,?,?,?,?,?,?,?,?,?)',
                [(s, 'corpus', 'go', 'f.go', 1, 2, 'Function', s, s, None)
                 for s in syms],
            )
            # An edge between two of the source's symbols → must be copied.
            conn.execute(
                'INSERT INTO scip_edges (caller_canonical_id, '
                'callee_canonical_id, edge_type, file, line, confidence) '
                'VALUES (?,?,?,?,?,?)',
                ('sym0', 'sym1', 'call', 'f.go', 5, 'exact'),
            )
            conn.commit()
        with dest._conn_provider.acquire() as conn:
            init_scip_schema(conn)

        _copy_source_scip(src, dest, 'corpus')  # must not raise

        with dest._conn_provider.acquire() as conn:
            edge_count = conn.execute(
                'SELECT COUNT(*) FROM scip_edges').fetchone()[0]
        assert edge_count == 1, 'the source SCIP edge must survive the copy'
    finally:
        src.close()
        dest.close()
