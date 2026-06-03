"""Tests for ``persist_python_routes`` — Wave 4 Tier 2 step 3 wrapper
that pushes Flask / FastAPI route extraction into the production
data path.

Mirrors the test shape of ``test_persist_akka_http_endpoints.py``.
The wrapper is a thin composition over ``ingest_python_routes``;
these tests pin the dispatch + aggregation contract, not the
extractor's internal logic (covered in
``tests/test_scip_python_web_extractor.py``).
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


def test_persist_python_routes_invokes_ingest_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_python_routes

    db_path = _make_db(tmp_path)
    calls: list[tuple[str, Path]] = []

    def _spy(*, source_name, source_root, conn, index_factory=None):
        calls.append((source_name, source_root))
        return 0

    monkeypatch.setattr(
        'docgen.scip_python_web_extractor.ingest_python_routes', _spy,
    )

    persist_python_routes(
        db_path,
        [
            ('webapi', tmp_path / 'webapi'),
            ('worker', tmp_path / 'worker'),
        ],
    )
    assert [c[0] for c in calls] == ['webapi', 'worker']


def test_persist_python_routes_aggregates_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_python_routes

    db_path = _make_db(tmp_path)
    counts = iter([2, 0, 6])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_python_web_extractor.ingest_python_routes', _spy,
    )

    total = persist_python_routes(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 8


def test_persist_python_routes_no_scip_artifact_returns_zero(
    tmp_path: Path,
) -> None:
    """A source without ``.scip`` yields 0 — no exception. Matches
    the fail-soft contract of every other persist_* wrapper."""
    from docgen.scip_persist import persist_python_routes

    db_path = _make_db(tmp_path)
    src = tmp_path / 'unindexed'
    src.mkdir()

    total = persist_python_routes(db_path, [('unindexed', src)])
    assert total == 0


def test_persist_python_routes_preserves_swagger_rows(
    tmp_path: Path,
) -> None:
    """Pre-existing Swagger row for source X (resolution_source=
    'swagger') survives a subsequent persist_python_routes
    invocation — same coexistence contract as the Akka wrapper.
    Pinned per-wrapper because each wrapper independently delegates
    to its own ingest function; a regression in one wrapper's
    extractor wouldn't be caught by Akka's test."""
    from docgen.scip_persist import persist_python_routes

    db_path = _make_db(tmp_path)
    src = tmp_path / 'svc'
    src.mkdir()

    conn = sqlite3.connect(db_path)
    try:
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            (
                'preseed-id', 'svc', 'GET', '/health',
                None, 'swagger',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    persist_python_routes(db_path, [('svc', src)])

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT path_template, resolution_source FROM api_endpoints '
            "WHERE source_name = 'svc'",
        ).fetchall()
    finally:
        conn.close()

    assert ('/health', 'swagger') in rows
