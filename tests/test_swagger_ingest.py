"""Contract for Swagger 2.0 ingestion — Phase 7c.

Parses a Swagger 2.0 spec (JSON or YAML), extracts each ``(method,
path, operationId)`` endpoint declaration, attempts to bind
operationIds back to scip_symbols via three strategies in order:

1. **Convention**: operationId matches a symbol's ``display_name``
   exactly. Common in Akka HTTP / Spring with ``@Operation(operationId
   = "X")``.
2. **Annotation hint**: a SCIP symbol's ``signature_documentation``
   (or ``documentation``) contains an explicit ``@operationId X``
   annotation. Less common, but precise where it exists.
3. **Manual override**: ``ariadne.yaml`` per-source
   ``swagger_overrides`` map (TBD; out of scope for this slice).

Endpoints whose operationId can't be bound are still persisted —
their ``producer_symbol_id`` stays NULL. The agent layer can ask the
user to fill them in or surface them as gaps.

This slice supports Swagger 2.0 ONLY (not OpenAPI 3.x). Per the
user's scoping decision: Swagger covers what scalaproject actually has;
OpenAPI 3 support can be a future slice if needed.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _make_swagger_spec(*, paths: dict, info_title: str = 'API') -> dict:
    """Build a minimal Swagger 2.0 spec dict for fixture purposes."""
    return {
        'swagger': '2.0',
        'info': {'title': info_title, 'version': '1.0'},
        'paths': paths,
    }


def _write_yaml_spec(path: Path, spec: dict) -> None:
    """Write a Swagger spec as YAML."""
    import yaml
    path.write_text(yaml.safe_dump(spec), encoding='utf-8')


def _write_json_spec(path: Path, spec: dict) -> None:
    path.write_text(json.dumps(spec), encoding='utf-8')


# ---------------------------------------------------------------------------
# parse_swagger_spec — extracts endpoint declarations
# ---------------------------------------------------------------------------


class TestParseSwaggerSpec:
    def test_parses_yaml_spec(self, tmp_path: Path) -> None:
        from docgen.swagger_ingest import parse_swagger_spec

        spec = _make_swagger_spec(paths={
            '/api/login': {
                'post': {
                    'operationId': 'login',
                    'summary': 'Authenticate a user',
                },
            },
        })
        spec_path = tmp_path / 'swagger.yaml'
        _write_yaml_spec(spec_path, spec)

        endpoints = parse_swagger_spec(spec_path)
        assert len(endpoints) == 1
        ep = endpoints[0]
        assert ep.method == 'POST'
        assert ep.path == '/api/login'
        assert ep.operation_id == 'login'

    def test_parses_json_spec(self, tmp_path: Path) -> None:
        from docgen.swagger_ingest import parse_swagger_spec

        spec = _make_swagger_spec(paths={
            '/api/license/validate': {
                'get': {'operationId': 'validateToken'},
            },
        })
        spec_path = tmp_path / 'swagger.json'
        _write_json_spec(spec_path, spec)

        endpoints = parse_swagger_spec(spec_path)
        assert len(endpoints) == 1
        assert endpoints[0].method == 'GET'
        assert endpoints[0].path == '/api/license/validate'
        assert endpoints[0].operation_id == 'validateToken'

    def test_handles_multiple_methods_per_path(self, tmp_path: Path) -> None:
        """A single path can declare GET, POST, PUT, etc. — each is a
        distinct endpoint."""
        from docgen.swagger_ingest import parse_swagger_spec

        spec = _make_swagger_spec(paths={
            '/api/users/{id}': {
                'get': {'operationId': 'getUser'},
                'put': {'operationId': 'updateUser'},
                'delete': {'operationId': 'deleteUser'},
            },
        })
        spec_path = tmp_path / 'swagger.yaml'
        _write_yaml_spec(spec_path, spec)

        endpoints = parse_swagger_spec(spec_path)
        methods = sorted(e.method for e in endpoints)
        assert methods == ['DELETE', 'GET', 'PUT']

    def test_endpoint_without_operation_id_is_still_emitted(
        self, tmp_path: Path,
    ) -> None:
        """Some Swagger specs omit operationId. The endpoint is still
        a valid declaration; binding will just fail and producer_symbol_id
        stays None."""
        from docgen.swagger_ingest import parse_swagger_spec

        spec = _make_swagger_spec(paths={
            '/api/health': {'get': {'summary': 'health check'}},
        })
        spec_path = tmp_path / 'swagger.yaml'
        _write_yaml_spec(spec_path, spec)

        endpoints = parse_swagger_spec(spec_path)
        assert len(endpoints) == 1
        assert endpoints[0].operation_id is None

    def test_missing_file_raises_file_not_found(
        self, tmp_path: Path,
    ) -> None:
        from docgen.swagger_ingest import parse_swagger_spec

        with pytest.raises(FileNotFoundError):
            parse_swagger_spec(tmp_path / 'nonexistent.yaml')

    def test_malformed_spec_raises(self, tmp_path: Path) -> None:
        """A file that's neither valid YAML nor JSON should raise so
        the caller doesn't silently treat it as zero-endpoint."""
        from docgen.swagger_ingest import parse_swagger_spec

        spec_path = tmp_path / 'broken.yaml'
        spec_path.write_text(
            '{not: [valid: yaml: at: all',
            encoding='utf-8',
        )
        with pytest.raises(Exception):  # noqa: B017 — yaml/json error
            parse_swagger_spec(spec_path)


# ---------------------------------------------------------------------------
# bind_operation_id_to_symbol — convention strategy
# ---------------------------------------------------------------------------


class TestOperationIdBinding:
    def test_convention_match_returns_symbol_canonical_id(self) -> None:
        """operationId='validateToken' matches a CrossSourceSymbol
        whose display_name is 'validateToken' or 'validate_token' (we
        try snake_case as fallback)."""
        from docgen.swagger_ingest import bind_operation_id

        # Build a minimal symbol-by-display-name index
        symbols_by_display: dict[str, str] = {
            'validateToken': 'scip-java maven g a 1 com/scalaproject/LicenseService#validateToken().',
        }
        canonical = bind_operation_id(
            'validateToken',
            symbols_by_display=symbols_by_display,
        )
        assert canonical == (
            'scip-java maven g a 1 com/scalaproject/LicenseService#validateToken().'
        )

    def test_no_match_returns_none(self) -> None:
        from docgen.swagger_ingest import bind_operation_id

        result = bind_operation_id(
            'nonexistent',
            symbols_by_display={},
        )
        assert result is None

    def test_snake_case_fallback(self) -> None:
        """If operationId is camelCase (validateToken) but the
        display_name is snake_case (validate_token), the binder tries
        the conversion — common when Scala/Java declares operationIds
        but the symbol is in a Python consumer's namespace."""
        from docgen.swagger_ingest import bind_operation_id

        symbols_by_display: dict[str, str] = {
            'validate_token': 'scip-python python scalaproject 0.1 lib/foo#validate_token().',
        }
        result = bind_operation_id(
            'validateToken',
            symbols_by_display=symbols_by_display,
        )
        assert result is not None
        assert 'validate_token' in result


# ---------------------------------------------------------------------------
# ingest_swagger_for_source — top-level
# ---------------------------------------------------------------------------


class TestIngestSwaggerForSource:
    def test_persists_endpoints_to_api_endpoints_table(
        self, tmp_path: Path,
    ) -> None:
        from docgen.swagger_ingest import ingest_swagger_for_source
        from library import Library

        spec = _make_swagger_spec(paths={
            '/api/login': {'post': {'operationId': 'login'}},
        })
        spec_path = tmp_path / 'swagger.yaml'
        _write_yaml_spec(spec_path, spec)

        db_path = tmp_path / 'a.db'
        lib = Library(db_path)
        # No symbols pre-loaded → operationId won't bind, but endpoint
        # is still persisted.
        with lib._conn_provider.acquire() as conn:
            count = ingest_swagger_for_source(
                source_name='scalaproject',
                swagger_paths=[spec_path],
                conn=conn,
            )
            row = conn.execute(
                'SELECT http_method, path_template, producer_symbol_id, '
                'resolution_source FROM api_endpoints '
                "WHERE source_name='scalaproject'"
            ).fetchone()
        lib.close()

        assert count == 1
        method, path_t, producer_sym, resolution = row
        assert method == 'POST'
        assert path_t == '/api/login'
        assert producer_sym is None  # no symbol matched
        assert resolution == 'swagger'

    def test_binds_via_convention_when_symbol_exists(
        self, tmp_path: Path,
    ) -> None:
        """If a scip_symbols row's display_name matches the
        operationId, the endpoint's producer_symbol_id is set."""
        from docgen.swagger_ingest import ingest_swagger_for_source
        from library import Library

        spec = _make_swagger_spec(paths={
            '/api/login': {'post': {'operationId': 'login'}},
        })
        spec_path = tmp_path / 'swagger.yaml'
        _write_yaml_spec(spec_path, spec)

        db_path = tmp_path / 'a.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            # Pre-populate a matching symbol
            conn.execute(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, file, line_start, line_end, kind, display_name, qualified_name, parent_qualified_name) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    'scip-java maven g a 1 com/scalaproject/Login#login().',
                    'scalaproject', 'scala', 'Login.scala',
                    1, 5, 'Method', 'login', 'com.scalaproject.Login.login',
                    'com.scalaproject.Login',
                ),
            )
            ingest_swagger_for_source(
                source_name='scalaproject',
                swagger_paths=[spec_path],
                conn=conn,
            )
            producer_sym = conn.execute(
                "SELECT producer_symbol_id FROM api_endpoints "
                "WHERE source_name='scalaproject'"
            ).fetchone()[0]
        lib.close()

        assert producer_sym is not None
        assert 'login' in producer_sym

    def test_processes_multiple_swagger_files(
        self, tmp_path: Path,
    ) -> None:
        """A source can declare multiple swagger_paths (e.g., one per
        microservice). All endpoints get ingested under the same
        source_name."""
        from docgen.swagger_ingest import ingest_swagger_for_source
        from library import Library

        spec_a = _make_swagger_spec(paths={
            '/api/a': {'get': {'operationId': 'opA'}},
        })
        spec_b = _make_swagger_spec(paths={
            '/api/b': {'get': {'operationId': 'opB'}},
        })
        a_path = tmp_path / 'a.yaml'
        b_path = tmp_path / 'b.yaml'
        _write_yaml_spec(a_path, spec_a)
        _write_yaml_spec(b_path, spec_b)

        lib = Library(tmp_path / 'db.db')
        with lib._conn_provider.acquire() as conn:
            count = ingest_swagger_for_source(
                source_name='multi',
                swagger_paths=[a_path, b_path],
                conn=conn,
            )
            paths = sorted(
                row[0] for row in conn.execute(
                    "SELECT path_template FROM api_endpoints "
                    "WHERE source_name='multi'"
                )
            )
        lib.close()

        assert count == 2
        assert paths == ['/api/a', '/api/b']

    def test_re_ingest_replaces_prior_endpoints(
        self, tmp_path: Path,
    ) -> None:
        """A re-ingest after the spec changed shouldn't accumulate
        ghost endpoints. Old rows for the source are cleared, fresh
        ones inserted."""
        from docgen.swagger_ingest import ingest_swagger_for_source
        from library import Library

        v1 = _make_swagger_spec(paths={
            '/api/old': {'get': {'operationId': 'oldOp'}},
        })
        spec_path = tmp_path / 'swagger.yaml'
        _write_yaml_spec(spec_path, v1)

        lib = Library(tmp_path / 'db.db')
        with lib._conn_provider.acquire() as conn:
            ingest_swagger_for_source(
                source_name='svc', swagger_paths=[spec_path], conn=conn,
            )
        lib.close()

        # Now spec is updated — /api/old is gone, /api/new replaces it
        v2 = _make_swagger_spec(paths={
            '/api/new': {'get': {'operationId': 'newOp'}},
        })
        _write_yaml_spec(spec_path, v2)

        lib = Library(tmp_path / 'db.db')
        with lib._conn_provider.acquire() as conn:
            ingest_swagger_for_source(
                source_name='svc', swagger_paths=[spec_path], conn=conn,
            )
            paths = sorted(
                row[0] for row in conn.execute(
                    "SELECT path_template FROM api_endpoints "
                    "WHERE source_name='svc'"
                )
            )
        lib.close()

        assert paths == ['/api/new']
