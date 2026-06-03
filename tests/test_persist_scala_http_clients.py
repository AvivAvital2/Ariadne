"""Tests for ``persist_scala_http_clients`` — Wave 4 Tier 4 step 3
wrapper for Scala HTTP-client extraction (Akka-HTTP
``Http().singleRequest`` / sttp ``basicRequest.<verb>``).

Mirrors the dispatch contract of the Python and JS wrappers.
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


def test_persist_scala_http_clients_invokes_ingest_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_scala_http_clients

    db_path = _make_db(tmp_path)
    calls: list[tuple[str, Path]] = []

    def _spy(*, source_name, source_root, conn, index_factory=None):
        calls.append((source_name, source_root))
        return 0

    monkeypatch.setattr(
        'docgen.scip_scala_http_client_extractor.ingest_scala_http_clients',
        _spy,
    )

    persist_scala_http_clients(
        db_path,
        [
            ('scalaproject', tmp_path / 'scalaproject'),
            ('biggerproject', tmp_path / 'biggerproject'),
        ],
    )
    assert [c[0] for c in calls] == ['scalaproject', 'biggerproject']


def test_persist_scala_http_clients_aggregates_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_scala_http_clients

    db_path = _make_db(tmp_path)
    counts = iter([2, 0, 13])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_scala_http_client_extractor.ingest_scala_http_clients',
        _spy,
    )

    total = persist_scala_http_clients(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 15


def test_persist_scala_http_clients_no_scip_artifact_returns_zero(
    tmp_path: Path,
) -> None:
    from docgen.scip_persist import persist_scala_http_clients

    db_path = _make_db(tmp_path)
    src = tmp_path / 'unindexed'
    src.mkdir()

    total = persist_scala_http_clients(db_path, [('unindexed', src)])
    assert total == 0


def test_persist_scala_http_clients_does_not_clobber_other_sources(
    tmp_path: Path,
) -> None:
    """Pre-seeded http_client_calls rows for source ``a`` survive a
    persist run scoped to source ``b``. Per-source isolation pinned
    independently per wrapper."""
    from docgen.scip_persist import persist_scala_http_clients

    db_path = _make_db(tmp_path)

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            '''INSERT INTO http_client_calls
               (source_name, consumer_symbol_id, raw_url,
                http_method, call_site_file, call_site_line,
                sink_name, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                'a', None, '/api/x', 'GET',
                'a.scala', 1, 'singleRequest', 'exact',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    src_b = tmp_path / 'b'
    src_b.mkdir()
    persist_scala_http_clients(db_path, [('b', src_b)])

    conn = sqlite3.connect(db_path)
    try:
        names = {
            r[0] for r in conn.execute(
                'SELECT DISTINCT source_name FROM http_client_calls',
            ).fetchall()
        }
    finally:
        conn.close()

    assert 'a' in names
