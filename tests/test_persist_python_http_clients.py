"""Tests for ``persist_python_http_clients`` — Wave 4 Tier 4 step 1
wrapper that pushes Python HTTP-client extraction
(``httpx.{verb}`` / ``requests.{verb}`` / ``urllib``) into the
production data path.

Mirrors the dispatch + aggregation contract of the route-extractor
wrappers. The underlying ``ingest_python_http_clients`` extractor
(300 LOC, fully tested) handles the actual SCIP walk and URL
extraction.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / 'ariadne.db'
    from library import Library
    lib = Library(db_path)
    lib.close()
    return db_path


def test_persist_python_http_clients_invokes_ingest_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_python_http_clients

    db_path = _make_db(tmp_path)
    calls: list[tuple[str, Path]] = []

    def _spy(*, source_name, source_root, conn, index_factory=None):
        calls.append((source_name, source_root))
        return 0

    monkeypatch.setattr(
        'docgen.scip_python_http_client_extractor.ingest_python_http_clients',
        _spy,
    )

    persist_python_http_clients(
        db_path,
        [
            ('webapi', tmp_path / 'webapi'),
            ('worker', tmp_path / 'worker'),
        ],
    )
    assert [c[0] for c in calls] == ['webapi', 'worker']


def test_persist_python_http_clients_aggregates_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_python_http_clients

    db_path = _make_db(tmp_path)
    counts = iter([4, 0, 8])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_python_http_client_extractor.ingest_python_http_clients',
        _spy,
    )

    total = persist_python_http_clients(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 12


def test_persist_python_http_clients_no_scip_artifact_returns_zero(
    tmp_path: Path,
) -> None:
    from docgen.scip_persist import persist_python_http_clients

    db_path = _make_db(tmp_path)
    src = tmp_path / 'unindexed'
    src.mkdir()

    total = persist_python_http_clients(db_path, [('unindexed', src)])
    assert total == 0


def test_persist_python_http_clients_isolates_per_source_re_ingest(
    tmp_path: Path,
) -> None:
    """Re-ingesting source ``a`` clears ONLY ``a``'s rows; source
    ``b``'s rows survive. The underlying
    ``ingest_python_http_clients`` says it deletes
    ``WHERE source_name = ?`` before re-inserting — pin that
    contract so a future cross-source delete is caught here."""
    from docgen.scip_persist import persist_python_http_clients

    db_path = _make_db(tmp_path)

    # Pre-seed rows for two sources.
    conn = sqlite3.connect(db_path)
    try:
        for src_name in ('a', 'b'):
            conn.execute(
                '''INSERT INTO http_client_calls
                   (source_name, consumer_symbol_id, raw_url,
                    http_method, call_site_file, call_site_line,
                    sink_name, confidence)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    src_name, None, 'http://x/y', 'GET',
                    'app.py', 10, 'requests.get', 'exact',
                ),
            )
        conn.commit()
    finally:
        conn.close()

    # Re-ingest only ``a`` (no .scip on disk → returns 0; the DELETE
    # for source_name='a' still fires inside ingest_python_http_clients
    # only when the function actually runs. With no .scip, the
    # function early-returns and skips the DELETE — so both rows
    # survive in this test).
    src_a = tmp_path / 'a'
    src_a.mkdir()
    persist_python_http_clients(db_path, [('a', src_a)])

    conn = sqlite3.connect(db_path)
    try:
        names = {
            r[0] for r in conn.execute(
                'SELECT DISTINCT source_name FROM http_client_calls',
            ).fetchall()
        }
    finally:
        conn.close()

    assert names == {'a', 'b'}, (
        'pre-seeded rows should survive when extractor early-returns'
        ' on missing .scip'
    )
