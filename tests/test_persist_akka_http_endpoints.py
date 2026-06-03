"""Tests for ``persist_string_literals`` + ``persist_akka_http_endpoints``
— the wrappers that close the data path from ``.scip`` artifacts to
``string_literals`` and ``api_endpoints`` (Wave 4 Tier 2 step 2 +
Layer C string_literals prereq).

Both wrappers are thin compositions over the existing
``ingest_string_literals`` / ``ingest_akka_http_routes`` extractor
entry points, which have their own deep test suites. These tests pin
the wrapper contract: dispatches once per source, aggregates count,
handles missing-scip cleanly, propagates the conn it owns.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / 'ariadne.db'
    # Initialize the schema by opening a Library — same path the wrappers use.
    from library import Library
    lib = Library(db_path)
    lib.close()
    return db_path


# ---------------------------------------------------------------------------
# persist_string_literals
# ---------------------------------------------------------------------------


def test_persist_string_literals_invokes_ingest_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    """The wrapper calls ``ingest_string_literals`` once per source
    pair, passing the source_name + source_root through unchanged."""
    from docgen.scip_persist import persist_string_literals

    db_path = _make_db(tmp_path)
    calls: list[tuple[str, Path]] = []

    def _spy(*, source_name, source_root, conn, index_factory=None):
        calls.append((source_name, source_root))
        return 0

    monkeypatch.setattr(
        'docgen.scip_string_literal_extractor.ingest_string_literals',
        _spy,
    )

    persist_string_literals(
        db_path,
        [
            ('webapp', tmp_path / 'webapp'),
            ('service', tmp_path / 'service'),
        ],
    )
    assert [c[0] for c in calls] == ['webapp', 'service']
    assert all(isinstance(c[1], Path) for c in calls)


def test_persist_string_literals_returns_total_count(
    tmp_path: Path, monkeypatch,
) -> None:
    """Returns the sum of per-source insertion counts. Bites if a
    future change forgets to aggregate."""
    from docgen.scip_persist import persist_string_literals

    db_path = _make_db(tmp_path)
    counts = iter([3, 7, 11])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_string_literal_extractor.ingest_string_literals',
        _spy,
    )

    total = persist_string_literals(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 21


def test_persist_string_literals_no_scip_artifact_returns_zero(
    tmp_path: Path,
) -> None:
    """A source whose ``<root>/.ariadne/index.scip`` doesn't exist
    yields zero rows — no exception, no partial state. The underlying
    ``ingest_string_literals`` returns 0 in that case."""
    from docgen.scip_persist import persist_string_literals

    db_path = _make_db(tmp_path)
    src = tmp_path / 'unindexed'
    src.mkdir()
    # Note: no .ariadne/index.scip on disk.

    total = persist_string_literals(db_path, [('unindexed', src)])
    assert total == 0


# ---------------------------------------------------------------------------
# persist_akka_http_endpoints
# ---------------------------------------------------------------------------


def test_persist_akka_http_endpoints_invokes_ingest_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    """One call per source; signature matches what cmd_index plumbs."""
    from docgen.scip_persist import persist_akka_http_endpoints

    db_path = _make_db(tmp_path)
    calls: list[tuple[str, Path]] = []

    def _spy(*, source_name, source_root, conn, index_factory=None):
        calls.append((source_name, source_root))
        return 0

    monkeypatch.setattr(
        'docgen.scip_akka_http_extractor.ingest_akka_http_routes', _spy,
    )

    persist_akka_http_endpoints(
        db_path,
        [
            ('scalaproject', tmp_path / 'scalaproject'),
            ('biggerproject', tmp_path / 'biggerproject'),
        ],
    )
    assert [c[0] for c in calls] == ['scalaproject', 'biggerproject']


def test_persist_akka_http_endpoints_aggregates_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_akka_http_endpoints

    db_path = _make_db(tmp_path)
    counts = iter([5, 0, 4])

    def _spy(*, source_name, source_root, conn, index_factory=None):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_akka_http_extractor.ingest_akka_http_routes', _spy,
    )

    total = persist_akka_http_endpoints(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 9


def test_persist_akka_http_endpoints_no_scip_artifact_returns_zero(
    tmp_path: Path,
) -> None:
    """A source without .scip yields 0 — no crash on missing
    artifact. Same fail-soft contract as persist_string_literals."""
    from docgen.scip_persist import persist_akka_http_endpoints

    db_path = _make_db(tmp_path)
    src = tmp_path / 'unindexed'
    src.mkdir()

    total = persist_akka_http_endpoints(db_path, [('unindexed', src)])
    assert total == 0


def test_persist_akka_http_endpoints_preserves_swagger_rows_for_same_source(
    tmp_path: Path,
) -> None:
    """A pre-existing Swagger row for source X (resolution_source=
    'swagger') survives a subsequent persist_akka_http_endpoints
    invocation. The Akka extractor only deletes
    resolution_source='pattern' rows before re-inserting.

    Pins the coexistence contract: a single source can have endpoints
    from multiple resolution paths simultaneously.
    """
    from docgen.scip_persist import persist_akka_http_endpoints

    db_path = _make_db(tmp_path)
    src = tmp_path / 'svc'
    src.mkdir()  # no .scip present → akka extraction is a no-op

    # Pre-seed a swagger-resolved row for this source.
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

    persist_akka_http_endpoints(db_path, [('svc', src)])

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT path_template, resolution_source FROM api_endpoints '
            "WHERE source_name = 'svc'",
        ).fetchall()
    finally:
        conn.close()

    assert ('/health', 'swagger') in rows, (
        'swagger row must survive akka extractor run; '
        f'got rows={rows}'
    )
