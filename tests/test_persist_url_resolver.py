"""Tests for ``persist_url_resolver`` — Wave 4 Phase 8c wrapper that
resolves ``http_client_calls`` URLs against ``api_endpoints``
templates and writes matched edges to ``api_calls``.

This is the final step that makes ``ariadne_trace_flow``'s HTTP-tier
hop return real cross-language chains. Without this resolver,
client and server data sit in separate tables that nothing joins.
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


def test_persist_url_resolver_invokes_resolve_for_each_source(
    tmp_path: Path, monkeypatch,
) -> None:
    """Wrapper calls ``resolve_urls_to_endpoints`` once per source
    pair, ignoring source_root (resolution is DB-only)."""
    from docgen.scip_persist import persist_url_resolver

    db_path = _make_db(tmp_path)
    calls: list[str] = []

    def _spy(*, conn, source_name):
        calls.append(source_name)
        return 0

    monkeypatch.setattr(
        'docgen.scip_url_resolver.resolve_urls_to_endpoints', _spy,
    )

    persist_url_resolver(
        db_path,
        [
            ('webapp', tmp_path / 'webapp'),
            ('admin', tmp_path / 'admin'),
        ],
    )
    assert calls == ['webapp', 'admin']


def test_persist_url_resolver_aggregates_counts(
    tmp_path: Path, monkeypatch,
) -> None:
    from docgen.scip_persist import persist_url_resolver

    db_path = _make_db(tmp_path)
    counts = iter([3, 0, 5])

    def _spy(*, conn, source_name):
        return next(counts)

    monkeypatch.setattr(
        'docgen.scip_url_resolver.resolve_urls_to_endpoints', _spy,
    )

    total = persist_url_resolver(
        db_path,
        [
            ('a', tmp_path / 'a'),
            ('b', tmp_path / 'b'),
            ('c', tmp_path / 'c'),
        ],
    )
    assert total == 8


def test_persist_url_resolver_returns_zero_on_empty_tables(
    tmp_path: Path,
) -> None:
    """Calling against empty ``http_client_calls`` /
    ``api_endpoints`` writes nothing and returns 0 cleanly. Pinned
    so callers can run this opportunistically without checking the
    table state first."""
    from docgen.scip_persist import persist_url_resolver

    db_path = _make_db(tmp_path)
    src = tmp_path / 'svc'
    src.mkdir()

    total = persist_url_resolver(db_path, [('svc', src)])
    assert total == 0


def test_persist_url_resolver_writes_api_calls_for_matched_url(
    tmp_path: Path,
) -> None:
    """End-to-end smoke: pre-seed one client URL + one matching
    endpoint, verify the resolver writes the join row.

    This is the integration receipt that ``ariadne_trace_flow``'s
    HTTP-tier join finally has data to walk.
    """
    from docgen.scip_persist import persist_url_resolver

    db_path = _make_db(tmp_path)
    src = tmp_path / 'svc'
    src.mkdir()

    conn = sqlite3.connect(db_path)
    try:
        # Pre-seed: one client call to /api/login from a known consumer
        conn.execute(
            '''INSERT INTO http_client_calls
               (source_name, consumer_symbol_id, raw_url,
                http_method, call_site_file, call_site_line,
                sink_name, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                'svc', 'consumer-sym-id', '/api/login', 'POST',
                'app.py', 42, 'requests.post', 'exact',
            ),
        )
        # Pre-seed: one matching endpoint
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            (
                'endpoint-id-1', 'backend', 'POST', '/api/login',
                'producer-sym-id', 'pattern',
            ),
        )
        conn.commit()
    finally:
        conn.close()

    persist_url_resolver(db_path, [('svc', src)])

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT consumer_symbol_id, endpoint_id FROM api_calls',
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 1
    consumer_sym, endpoint_id = rows[0]
    assert consumer_sym == 'consumer-sym-id'
    assert endpoint_id == 'endpoint-id-1', (
        'resolver should bind the client URL to the matching endpoint'
    )
