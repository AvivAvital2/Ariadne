"""Contract for SCIP-driven JS/TS HTTP client extractor (Phase 8b.2).

Architecture mirrors Phase 8b.1 (Python HTTP clients) and Phase 8a.3
(Express routes):

- **SCIP** filters which call sites are real JS/TS HTTP-client
  primitives via the Phase 2r ``DEFAULT_SINK_REGISTRY`` filtered to
  ``language='typescript'`` (covers ``.js``/``.ts``/``.jsx``/``.tsx``/
  ``.mjs``/``.cjs`` per the ``LANGUAGES`` registry).
- **ast-grep / tree-sitter-javascript** walks the call STRUCTURE: the
  ``call_expression`` and its first ``arguments`` child.
- **Phase 2p ``string_literals``** supplies the URL value at the
  first arg's position. Template literals with ``${}`` interpolation
  aren't indexed by Phase 2p, so the lookup naturally returns
  ``None`` for those.
- **Phase 2d ``scip_symbols``** supplies ``consumer_symbol_id`` —
  the enclosing function/method.

Output: rows in ``http_client_calls``. Phase 8c will read these and
match URLs against ``api_endpoints.path_template``.

Precondition: ``ingest_string_literals`` must run for the same
``(source_name, source_root)`` first. The ``_run_pipeline`` helper
bundles them.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
)
from docgen.scip_string_literal_extractor import ingest_string_literals


# Synthetic SCIP symbols. Suffix patterns the registry recognizes.
_FETCH_SYM = 'scip-typescript . . . lib.dom.d.ts/fetch.'
_AXIOS_GET_SYM = (
    'scip-typescript . . . axios/index.d.ts/AxiosInstance#get().'
)
_AXIOS_POST_SYM = (
    'scip-typescript . . . axios/index.d.ts/AxiosInstance#post().'
)
_AXIOS_PUT_SYM = (
    'scip-typescript . . . axios/index.d.ts/AxiosInstance#put().'
)
_AXIOS_DELETE_SYM = (
    'scip-typescript . . . axios/index.d.ts/AxiosInstance#delete().'
)
_GOT_GET_SYM = (
    'scip-typescript . . . got/dist/source/index.d.ts/Got#get.'
)
_NODE_FETCH_SYM = (
    'scip-typescript . . . node-fetch/lib/index.d.ts/fetch.'
)


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _occ_at(
    text: str, marker: str, symbol: str, *, nth: int = 0,
) -> _ScipOccurrence:
    """Word-boundary aware nth-occurrence finder; returns a SCIP
    occurrence at that position (0-indexed line/col)."""
    found = 0
    pos = -1
    n = len(text)
    i = 0
    while i <= n - len(marker):
        if text.startswith(marker, i):
            before_ok = (
                i == 0
                or not (text[i - 1].isalnum() or text[i - 1] == '_')
            )
            j = i + len(marker)
            after_ok = (
                j >= n
                or not (text[j].isalnum() or text[j] == '_')
            )
            if before_ok and after_ok:
                if found == nth:
                    pos = i
                    break
                found += 1
                i = j
                continue
        i += 1
    if pos < 0:
        raise ValueError(
            f'marker {marker!r} (nth={nth}) not found as a word',
        )
    line = text.count('\n', 0, pos)
    line_start = text.rfind('\n', 0, pos) + 1
    col = pos - line_start
    return _ScipOccurrence(
        symbol=symbol,
        range=(line, col, line, col + len(marker)),
        is_definition=False,
    )


def _make_index(
    src_file: Path,
    source_root: Path,
    occurrences: list[_ScipOccurrence],
) -> ScipIndex:
    rel = src_file.relative_to(source_root)
    return ScipIndex(
        documents=(_ScipDoc(
            relative_path=str(rel),
            occurrences=tuple(occurrences),
            symbols=(),
        ),),
        source_root=source_root,
    )


def _add_scip_symbol(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    source_name: str,
    file: str,
    line_start: int,
    line_end: int,
    kind: str = 'Function',
    qualified_name: str = '',
    parent_qualified_name: str | None = None,
    language: str = 'typescript',
) -> None:
    conn.execute(
        '''INSERT INTO scip_symbols
           (canonical_id, source_name, language, file,
            line_start, line_end, kind, display_name,
            qualified_name, parent_qualified_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (canonical_id, source_name, language, file,
         line_start, line_end, kind,
         qualified_name.rsplit('.', 1)[-1] or canonical_id,
         qualified_name, parent_qualified_name),
    )
    conn.commit()


def _query_client_calls(
    conn: sqlite3.Connection, source_name: str,
) -> list[dict]:
    cur = conn.execute(
        '''SELECT consumer_symbol_id, raw_url, http_method,
                  call_site_file, call_site_line, sink_name, confidence
           FROM http_client_calls WHERE source_name = ?
           ORDER BY call_site_file, call_site_line''',
        (source_name,),
    )
    cols = [
        'consumer_symbol_id', 'raw_url', 'http_method',
        'call_site_file', 'call_site_line', 'sink_name',
        'confidence',
    ]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _run_pipeline(
    *, source_name: str, source_root: Path,
    conn: sqlite3.Connection, index: ScipIndex,
) -> int:
    """Phase 2p literal indexing, then JS HTTP client extraction."""
    from docgen.scip_js_http_client_extractor import (
        ingest_js_http_clients,
    )

    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_js_http_clients(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# ---------------------------------------------------------------------------
# fetch — browser/node global
# ---------------------------------------------------------------------------


class TestFetch:
    def test_fetch_with_literal_url(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "const r = fetch('http://api/users');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['http_method'] == 'GET'
        assert rows[0]['sink_name'] == 'fetch'

    def test_await_fetch(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'async function load() {\n'
            "  const r = await fetch('http://api/data');\n"
            '  return r.json();\n'
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/data'

    def test_fetch_template_literal_no_interpolation(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "fetch(`/api/health`);\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == '/api/health'

    def test_fetch_template_with_interpolation_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "const id = 'abc';\n"
            "fetch(`/api/users/${id}`);\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # Phase 2p doesn't index interpolated templates, so the lookup
        # at the call's first-arg position returns None → no row
        for r in rows:
            assert '${' not in r['raw_url']
            assert '/api/users' != r['raw_url']  # not bare without id

    def test_fetch_variable_url_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "const URL = 'http://api/x';\n"
            'fetch(URL);\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # Variable URL — call-site position holds an identifier, not a
        # string literal. Phase 2s territory; no row.
        assert rows == []


# ---------------------------------------------------------------------------
# axios verbs
# ---------------------------------------------------------------------------


class TestAxios:
    def test_axios_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "axios.get('http://api/users');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _AXIOS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['http_method'] == 'GET'
        assert rows[0]['sink_name'] == 'axios.get'

    def test_axios_post(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "axios.post('http://api/login', { user });\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _AXIOS_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['http_method'] == 'POST'

    def test_axios_put_and_delete(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "axios.put('http://api/items/1', { name: 'x' });\n"
            "axios.delete('http://api/items/1');\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'put', _AXIOS_PUT_SYM),
            _occ_at(text, 'delete', _AXIOS_DELETE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        methods = {r['http_method'] for r in rows}
        assert {'PUT', 'DELETE'} == methods

    def test_axios_instance_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``const api = axios.create(...); api.get(...)`` — the
        instance call resolves via SCIP to ``AxiosInstance#get``,
        same suffix as direct ``axios.get``. Same registry entry
        matches both."""
        text = (
            'const api = axios.create({ baseURL: "/" });\n'
            "api.get('/users');\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        # The 'get' that matters is the second one (axios.create's
        # own type signature doesn't generate a 'get' identifier
        # token here).
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _AXIOS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == '/users'


# ---------------------------------------------------------------------------
# got + node-fetch
# ---------------------------------------------------------------------------


class TestOtherClients:
    def test_got_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "const r = got.get('http://api/v1/data');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _GOT_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/v1/data'
        assert rows[0]['sink_name'] == 'got.get'

    def test_node_fetch(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        # Use a single ``fetch(...)`` call without an import line —
        # ``'node-fetch'`` inside an import string would match the
        # word-boundary fetch finder and offset ``nth`` indices. The
        # SCIP fixture (``_NODE_FETCH_SYM``) is what disambiguates
        # node-fetch from the global ``fetch``, not the import.
        text = "const r = fetch('http://api/legacy');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _NODE_FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/legacy'
        assert rows[0]['sink_name'] == 'node-fetch'


# ---------------------------------------------------------------------------
# Consumer symbol resolution
# ---------------------------------------------------------------------------


class TestConsumerSymbolResolution:
    def test_call_inside_function_resolves_consumer(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'function loadUsers() {\n'
            "  return fetch('http://api/users').then(r => r.json());\n"
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        fn_id = 'scip-typescript . . . app.js/loadUsers().'
        _add_scip_symbol(
            conn, canonical_id=fn_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=3,
            kind='Function', qualified_name='loadUsers',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] == fn_id

    def test_call_at_module_level_consumer_is_null(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "fetch('http://api/x');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] is None

    def test_call_inside_class_method_picks_method(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'class UserService {\n'
            '  async fetchUsers() {\n'
            "    return fetch('http://api/users');\n"
            '  }\n'
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        # Class spans 1-5, Method spans 2-4 — the Method must be the
        # consumer (kind filter rejects Class regardless of range).
        _add_scip_symbol(
            conn, canonical_id='scip:UserService#',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=5,
            kind='Class', qualified_name='UserService',
        )
        method_id = 'scip:UserService#fetchUsers().'
        _add_scip_symbol(
            conn, canonical_id=method_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=4,
            kind='Method', qualified_name='UserService.fetchUsers',
            parent_qualified_name='UserService',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] == method_id


# ---------------------------------------------------------------------------
# Adversarial — SCIP filtering, malformed input
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unrelated_get_call_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``cache.get('key')`` where ``cache`` is some unrelated
        library. SCIP symbol points at ``cachelib/Cache#get.`` — not
        in the registry. No row."""
        text = "const v = cache.get('key');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'get',
                'scip-typescript . . . cachelib/Cache#get.',
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert rows == []

    def test_no_scip_occurrence_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "fetch('http://api/x');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert rows == []

    def test_malformed_js_doesnt_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        broken_text = 'function broken( {\n'
        good_text = "fetch('http://api/ok');\n"
        broken = tmp_path / 'broken.js'
        good = tmp_path / 'good.js'
        broken.write_text(broken_text)
        good.write_text(good_text)
        index = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='broken.js',
                    occurrences=(),
                    symbols=(),
                ),
                _ScipDoc(
                    relative_path='good.js',
                    occurrences=(
                        _occ_at(good_text, 'fetch', _FETCH_SYM),
                    ),
                    symbols=(),
                ),
            ),
            source_root=tmp_path,
        )
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        urls = [r['raw_url'] for r in rows]
        assert 'http://api/ok' in urls

    def test_missing_index_file_no_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_js_http_client_extractor import (
            ingest_js_http_clients,
        )
        rc = ingest_js_http_clients(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Production-realistic SCIP integration
# ---------------------------------------------------------------------------


class TestScipIntegrationChallenges:
    def test_realistic_scip_typescript_symbol(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "axios.get('http://api/realistic');\n"
        src = tmp_path / 'app.ts'
        src.write_text(text)
        realistic_sym = (
            'scip-typescript npm axios 1.6.0 '
            'index.d.ts/AxiosInstance#get().'
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', realistic_sym),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert any(
            r['raw_url'] == 'http://api/realistic'
            and r['http_method'] == 'GET'
            for r in rows
        )

    def test_typescript_file_indexed(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``.ts`` source uses the same ``javascript`` ast-grep
        grammar as ``.js`` (matches catalog convention). Pure-JS
        syntax inside a ``.ts`` file should still be indexed —
        TypeScript-specific syntax (generic type args, type
        annotations) isn't covered by tree-sitter-javascript and is
        out of scope for v1."""
        text = "axios.get('http://api/typed');\n"
        src = tmp_path / 'app.ts'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _AXIOS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert any(
            r['raw_url'] == 'http://api/typed' for r in rows
        )

    def test_mixed_real_and_unrelated_occurrences(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'function listItems() {\n'
            "  const cached = cache.get('items');\n"
            "  return axios.get('http://api/items');\n"
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'get',
                'scip-typescript . . . cachelib/Cache#get.',
                nth=0,
            ),
            _occ_at(text, 'get', _AXIOS_GET_SYM, nth=1),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        urls = [r['raw_url'] for r in rows]
        assert 'http://api/items' in urls
        assert 'items' not in urls

    def test_multi_line_call_args(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'axios.post(\n'
            "  'http://api/multi-line',\n"
            '  { x: 1 },\n'
            ');\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _AXIOS_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/multi-line'


# ---------------------------------------------------------------------------
# Phase 2s wiring — variable URLs resolve via scip_symbols + string_literals
# ---------------------------------------------------------------------------


class TestPhase2sVariableResolution:
    """Phase 2s shipped as a library; this test asserts Phase 8b.2
    consumes it. Variable URL identifiers resolve through
    ``resolve_arg_value`` rather than being silently skipped."""

    def test_variable_url_resolves_via_scip_symbols(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "const API_URL = 'http://api/users';\n"
            'function fetchUsers() {\n'
            '  return fetch(API_URL);\n'
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        fn_id = 'scip-typescript . . . app.js/fetchUsers().'
        _add_scip_symbol(
            conn, canonical_id=fn_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=4,
            kind='Function', qualified_name='fetchUsers',
            language='typescript',
        )
        # The variable's def — Phase 2s looks for this
        _add_scip_symbol(
            conn, canonical_id='scip:app.API_URL',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=1,
            kind='Variable', qualified_name='app.API_URL',
            language='typescript',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['confidence'] == 'resolved-constant'
        assert rows[0]['consumer_symbol_id'] == fn_id

    def test_variable_url_with_no_def_still_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired negative — variable without a scip_symbol is
        skipped; the literal sibling still emits."""
        text = (
            'function fetchTwo() {\n'
            "  fetch('http://api/literal');\n"
            '  fetch(UNDEFINED_URL);\n'
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        fn_id = 'scip-typescript . . . app.js/fetchTwo().'
        _add_scip_symbol(
            conn, canonical_id=fn_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=4,
            kind='Function', qualified_name='fetchTwo',
            language='typescript',
        )
        # No scip_symbol for UNDEFINED_URL
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM, nth=0),
            _occ_at(text, 'fetch', _FETCH_SYM, nth=1),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        urls = [r['raw_url'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal call; got {urls}'
        )
        assert rows[0]['raw_url'] == 'http://api/literal'


# ---------------------------------------------------------------------------
# Re-ingest semantics
# ---------------------------------------------------------------------------


class TestReIngest:
    def test_replaces_same_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text1 = "fetch('http://api/old');\n"
        src = tmp_path / 'app.js'
        src.write_text(text1)
        index1 = _make_index(src, tmp_path, [
            _occ_at(text1, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index1,
        )
        text2 = "axios.post('http://api/new');\n"
        src.write_text(text2)
        index2 = _make_index(src, tmp_path, [
            _occ_at(text2, 'post', _AXIOS_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index2,
        )
        rows = _query_client_calls(conn, 'myapi')
        urls = [r['raw_url'] for r in rows]
        assert 'http://api/old' not in urls
        assert 'http://api/new' in urls

    def test_preserves_other_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        conn.execute(
            '''INSERT INTO http_client_calls
               (source_name, raw_url, http_method,
                call_site_file, call_site_line, sink_name, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            ('other', 'http://other/x', 'GET',
             '/x.js', 1, 'fetch', 'literal'),
        )
        conn.commit()
        text = "fetch('http://api/mine');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'fetch', _FETCH_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        my = [r['raw_url'] for r in _query_client_calls(conn, 'myapi')]
        other = [
            r['raw_url'] for r in _query_client_calls(conn, 'other')
        ]
        assert 'http://api/mine' in my
        assert other == ['http://other/x']
