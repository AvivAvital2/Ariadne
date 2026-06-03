"""Tests for ``persist_api_endpoints`` — wires Wave 4 Tier 2 (HTTP API
surface).

Without this helper, ``api_endpoints`` is the same kind of
``_SCHEMA``-without-INSERT gap that ``scip_symbols``/``scip_edges`` /
``scip_index_state`` were before being wired. Once populated,
``ariadne_trace_flow``'s HTTP-tier join (``api_calls``-to-
``api_endpoints``) starts returning rows once the client side
(``api_calls``) is wired too.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from textwrap import dedent

import pytest


def _write_swagger_spec(path: Path, *, paths: dict) -> None:
    """Write a minimal OpenAPI 2.0 / 3.0-compatible spec.

    ``paths`` is a dict of ``{path: {method: {operationId: ...}}}``.
    """
    spec = {
        'swagger': '2.0',
        'info': {'title': 'test', 'version': '1.0'},
        'paths': paths,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec), encoding='utf-8')


def test_persist_api_endpoints_writes_rows_from_swagger_spec(
    tmp_path: Path,
) -> None:
    """A source with ``swagger_paths`` declared has its OpenAPI spec
    parsed and rows land in ``api_endpoints`` keyed by source_name."""
    from docgen.scip_persist import persist_api_endpoints

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'webapp'
    spec_path = source_root / 'api' / 'openapi.json'
    _write_swagger_spec(spec_path, paths={
        '/users': {'get': {'operationId': 'listUsers'}},
        '/users/{id}': {'post': {'operationId': 'createUser'}},
    })

    persisted = persist_api_endpoints(
        db_path,
        [('webapp', source_root, ['api/openapi.json'])],
    )
    assert persisted == 2

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            'SELECT source_name, http_method, path_template, '
            'resolution_source FROM api_endpoints '
            "WHERE source_name = 'webapp' "
            'ORDER BY http_method',
        ).fetchall()
    finally:
        conn.close()

    assert len(rows) == 2
    methods = {r[1] for r in rows}
    paths = {r[2] for r in rows}
    assert methods == {'GET', 'POST'}
    assert paths == {'/users', '/users/{id}'}
    assert {r[3] for r in rows} == {'swagger'}


def test_persist_api_endpoints_skips_source_with_no_swagger(
    tmp_path: Path,
) -> None:
    """A source passed with empty ``swagger_paths`` writes no rows.
    Bites if anyone changes the helper to do something invasive when
    the list is empty."""
    from docgen.scip_persist import persist_api_endpoints

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'pyproject'
    source_root.mkdir()

    persisted = persist_api_endpoints(
        db_path, [('pyproject', source_root, [])],
    )
    assert persisted == 0

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute(
            'SELECT COUNT(*) FROM api_endpoints',
        ).fetchone()[0]
    finally:
        conn.close()
    assert count == 0


def test_persist_api_endpoints_resolves_swagger_paths_relative_to_source_root(
    tmp_path: Path,
) -> None:
    """``swagger_paths`` in the YAML are relative paths; the helper
    resolves them under ``source_root`` before opening."""
    from docgen.scip_persist import persist_api_endpoints

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'service'
    spec_path = source_root / 'docs' / 'api.json'
    _write_swagger_spec(spec_path, paths={
        '/health': {'get': {'operationId': 'health'}},
    })

    # Note the swagger_path is 'docs/api.json' — relative.
    persisted = persist_api_endpoints(
        db_path,
        [('service', source_root, ['docs/api.json'])],
    )
    assert persisted == 1


def test_persist_api_endpoints_idempotent_re_run_replaces_rows(
    tmp_path: Path,
) -> None:
    """``ingest_swagger_for_source`` clears prior rows for the source
    before re-inserting. Re-running with a smaller spec overwrites,
    not appends."""
    from docgen.scip_persist import persist_api_endpoints

    db_path = tmp_path / 'ariadne.db'
    source_root = tmp_path / 'webapp'
    spec_path = source_root / 'api.json'

    # First ingest — three endpoints.
    _write_swagger_spec(spec_path, paths={
        '/a': {'get': {}},
        '/b': {'get': {}},
        '/c': {'get': {}},
    })
    persist_api_endpoints(
        db_path,
        [('webapp', source_root, ['api.json'])],
    )

    # Spec shrinks — only one endpoint now.
    _write_swagger_spec(spec_path, paths={
        '/a': {'get': {}},
    })
    persist_api_endpoints(
        db_path,
        [('webapp', source_root, ['api.json'])],
    )

    conn = sqlite3.connect(db_path)
    try:
        rows = conn.execute(
            "SELECT path_template FROM api_endpoints "
            "WHERE source_name = 'webapp'",
        ).fetchall()
    finally:
        conn.close()
    paths = {r[0] for r in rows}
    assert paths == {'/a'}, (
        f'second ingest must replace prior rows; got paths={paths}'
    )
