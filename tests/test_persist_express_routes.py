"""Tests for ``persist_express_routes`` — Wave 4 Tier 2 step 4
wrapper that pushes Express / Koa route extraction into the
production data path.

Mirrors the test shape of ``test_persist_akka_http_endpoints.py``
and ``test_persist_python_routes.py``. Same wrapper-level dispatch
+ aggregation contract; extractor logic is covered in
``tests/test_scip_express_route_extractor.py``.
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


def test_persist_express_routes_invokes_ingest_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_express_routes

    db_path = _make_db(tmp_path)
    calls: list[tuple[str, Path]] = []

    def _spy(*, source_name, source_root, conn, index_factory=None):
        calls.append((source_name, source_root))
        return 0

    monkeypatch.setattr(
        'docgen.scip_express_route_extractor.ingest_express_routes',
        _spy,
    )

    persist_express_routes(
        db_path,
        [
            ('webapp', tmp_path / 'webapp'),
            ('admin', tmp_path / 'admin'),
        ],
    )
    assert [c[0] for c in calls] == ['webapp', 'admin']


def test_persist_express_routes_aggregates_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_express_routes

    db_path = _make_db(tmp_path)
    counts = iter([3, 0, 9])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_express_route_extractor.ingest_express_routes',
        _spy,
    )

    total = persist_express_routes(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 12


def test_persist_express_routes_no_scip_artifact_returns_zero(
    tmp_path: Path,
) -> None:
    """A source without ``.scip`` yields 0 — no exception. Matches
    the fail-soft contract of every other persist_* wrapper."""
    from docgen.scip_persist import persist_express_routes

    db_path = _make_db(tmp_path)
    src = tmp_path / 'unindexed'
    src.mkdir()

    total = persist_express_routes(db_path, [('unindexed', src)])
    assert total == 0


def test_persist_express_routes_preserves_swagger_rows(
    tmp_path: Path,
) -> None:
    """Pre-existing Swagger row survives a subsequent
    persist_express_routes invocation. Pinned per-wrapper because each
    wrapper independently delegates to its own ingest function — a
    regression in one extractor's resolution_source filter wouldn't
    be caught by Akka's or python_routes' tests."""
    from docgen.scip_persist import persist_express_routes

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

    persist_express_routes(db_path, [('svc', src)])

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT path_template, resolution_source FROM api_endpoints '
            "WHERE source_name = 'svc'",
        ).fetchall()
    finally:
        conn.close()

    assert ('/health', 'swagger') in rows
