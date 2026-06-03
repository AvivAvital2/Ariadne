"""Tests for docgen.catalog_lookup.get_element_body."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from docgen.catalog_lookup import get_element_body


def _make_found_info(file_path, line_start, line_end):
    return {
        'found': True,
        'language': 'python',
        'subtype': 'function',
        'qualified_name': 'x.foo',
        'signature': 'def foo():',
        'location': {
            'line_start': line_start,
            'line_end': line_end,
            'col_start': 0,
            'col_end': 0,
        },
        'parent_qualified_name': None,
        'description': None,
        'file': str(file_path) if file_path else None,
    }


class TestGetElementBodyNotFound:
    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_passes_through_suggestions(self, mock_lookup):
        mock_lookup.return_value = {
            'found': False,
            'error': 'not_in_catalog',
            'qualified_name': 'x.missing',
            'suggestions_in_file': ['x.foo'],
            'suggestions_in_source': ['x.bar'],
        }
        result = get_element_body(MagicMock(), 'x', None, 'x.missing')
        assert result['found'] is False
        assert result['error'] == 'not_in_catalog'
        assert result['suggestions_in_file'] == ['x.foo']
        assert 'body' not in result
        assert 'body_error' not in result


class TestGetElementBodyFound:
    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_multi_line_slice(self, mock_lookup, tmp_path):
        fp = tmp_path / 'sample.py'
        fp.write_text('line1\nline2\nline3\nline4\n')
        mock_lookup.return_value = _make_found_info(fp, 2, 3)
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['found'] is True
        assert result['body'] == 'line2\nline3'
        assert result['body_line_count'] == 2

    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_single_line(self, mock_lookup, tmp_path):
        fp = tmp_path / 'sample.py'
        fp.write_text('alpha\nbeta\ngamma\n')
        mock_lookup.return_value = _make_found_info(fp, 2, 2)
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['body'] == 'beta'
        assert result['body_line_count'] == 1

    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_full_file(self, mock_lookup, tmp_path):
        fp = tmp_path / 'sample.py'
        fp.write_text('one\ntwo\nthree\n')
        mock_lookup.return_value = _make_found_info(fp, 1, 3)
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['body'] == 'one\ntwo\nthree'
        assert result['body_line_count'] == 3

    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_preserves_original_metadata(self, mock_lookup, tmp_path):
        fp = tmp_path / 'sample.py'
        fp.write_text('x\ny\nz\n')
        mock_lookup.return_value = _make_found_info(fp, 1, 1)
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['qualified_name'] == 'x.foo'
        assert result['signature'] == 'def foo():'
        assert result['language'] == 'python'
        assert result['subtype'] == 'function'


class TestGetElementBodyErrors:
    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_missing_line_start(self, mock_lookup, tmp_path):
        info = _make_found_info(tmp_path / 'x.py', None, 5)
        mock_lookup.return_value = info
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['found'] is True
        assert result['body_error'] == 'missing_file_or_location'
        assert 'body' not in result

    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_missing_line_end(self, mock_lookup, tmp_path):
        info = _make_found_info(tmp_path / 'x.py', 1, None)
        mock_lookup.return_value = info
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['body_error'] == 'missing_file_or_location'

    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_missing_file_path(self, mock_lookup):
        info = _make_found_info(None, 1, 2)
        mock_lookup.return_value = info
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['body_error'] == 'missing_file_or_location'

    @patch('docgen.catalog_lookup.lookup_symbol')
    def test_unreadable_file(self, mock_lookup):
        info = _make_found_info('/nonexistent/path/nofile_xyz.py', 1, 2)
        mock_lookup.return_value = info
        result = get_element_body(MagicMock(), 'x', None, 'x.foo')
        assert result['found'] is True
        assert 'body_error' in result
        assert 'read_failed' in result['body_error']
        assert 'body' not in result


# --- Added coverage: CLI, MCP tool registration, real-Library integration ---

import json
import subprocess
from pathlib import Path

ARIADNE_ROOT = str(Path(__file__).resolve().parent.parent)


class TestCmdBodyCli:
    """Smoke test for `ariadne body` CLI via subprocess.

    Validates: subparser is registered, cmd_body is wired to HANDLERS,
    cmd_body returns JSON with expected shape for a known catalog element.
    Does NOT validate: error branches, flag parsing edge cases.
    """

    def test_known_symbol_returns_body(self):
        # Uses a stable symbol that exists in the live catalog.
        # If catalog is stale or symbol moves, this test is a signal to re-sync.
        result = subprocess.run(
            ['uv', 'run', 'ariadne', 'body',
             '--source', 'ariadne',
             '--name', 'embedding.EMBEDDING_DIM'],
            cwd=ARIADNE_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, f'CLI failed: {result.stderr}'
        data = json.loads(result.stdout)
        assert data.get('found') is True, f'expected found=True, got {data}'
        assert 'body' in data, f'expected body field, got keys {list(data)}'
        assert 'EMBEDDING_DIM' in data['body']
        assert data['body_line_count'] >= 1

    def test_missing_symbol_returns_not_found(self):
        result = subprocess.run(
            ['uv', 'run', 'ariadne', 'body',
             '--source', 'ariadne',
             '--name', 'nonexistent.symbol.xyzzy'],
            cwd=ARIADNE_ROOT,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0
        data = json.loads(result.stdout)
        assert data.get('found') is False
        assert 'body' not in data


class TestMcpToolRegistration:
    """Validates ariadne_body is importable from mcp_server and has expected signature.

    Does NOT validate: FastMCP registration actually exposes it via MCP protocol.
    A live MCP client test would be needed for full validation.
    """

    def test_function_exists(self):
        import ariadne_mcp.server as mcp_server
        assert hasattr(mcp_server, 'ariadne_body'), (
            'ariadne_body not defined in mcp_server.py'
        )

    def test_function_signature(self):
        import inspect

        import ariadne_mcp.server as mcp_server
        sig = inspect.signature(mcp_server.ariadne_body)
        params = list(sig.parameters.keys())
        assert 'qualified_name' in params
        assert 'source' in params
        assert 'file' in params

    def test_is_coroutine(self):
        import inspect

        import ariadne_mcp.server as mcp_server
        assert inspect.iscoroutinefunction(mcp_server.ariadne_body), (
            'ariadne_body should be async for consistency with other MCP tools'
        )


class TestRealLibraryIntegration:
    """Integration test: real Library against live ariadne.db.

    Validates: get_element_body correctly reads lookup_symbol response shape
    (catches breaking changes in lookup_symbol that unit tests miss).
    Does NOT validate: isolated/empty DB scenarios (we use the live catalog).
    """

    def test_real_library_known_symbol(self):
        from config import get_config
        from library import Library
        cfg = get_config()
        lib = Library(cfg.db_path)
        try:
            result = get_element_body(
                lib,
                'ariadne',
                None,
                'embedding.EMBEDDING_DIM',
            )
        finally:
            lib.close()
        assert result.get('found') is True, (
            f'Expected EMBEDDING_DIM in catalog. Got: {result}'
        )
        assert 'body' in result
        assert 'EMBEDDING_DIM' in result['body']
        # body should be the one-line constant declaration
        assert result['body_line_count'] == 1

    def test_real_library_missing_symbol(self):
        from config import get_config
        from library import Library
        cfg = get_config()
        lib = Library(cfg.db_path)
        try:
            result = get_element_body(
                lib,
                'ariadne',
                None,
                'totally.made.up.symbol.xyzzy',
            )
        finally:
            lib.close()
        assert result.get('found') is False
        assert 'body' not in result
        # lookup_symbol should supply suggestions structure
        assert 'suggestions_in_source' in result
