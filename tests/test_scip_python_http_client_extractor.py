"""Contract for SCIP-driven Python HTTP client extractor (Phase 8b.1).

Architecture mirrors Phase 8a — same shape as the route extractors:

- **SCIP** filters which call sites are real Python HTTP-client
  primitives. The match goes through Phase 2r's
  ``DEFAULT_SINK_REGISTRY`` so adding a new client library is a
  registry edit, not a new code branch in the extractor.
- **ast.parse** walks the call STRUCTURE: which call_expression sits
  at each SCIP-matched position; what's the position of the first
  argument expression.
- **Phase 2p ``string_literals``** supplies the literal URL VALUE.
  Quote handling, f-string exclusion, escape unfolding — all in the
  Phase 2p pre-pass.
- **Phase 2d ``scip_symbols``** supplies the ``consumer_symbol_id`` —
  the enclosing function/method that contains the call site, NULL
  when the call sits at module level.

Output: rows in ``http_client_calls`` (Phase 8b schema). Phase 8c will
read these and match URLs against ``api_endpoints.path_template`` to
populate ``api_calls``.

Precondition: ``ingest_string_literals`` must run for the same
``(source_name, source_root)`` first. The route extractors enforce
this; the test helper bundles them in a single ``_run_pipeline`` call.

These tests are RED until ``docgen/scip_python_http_client_extractor``
implements ``ingest_python_http_clients``.
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


# Synthetic SCIP symbols mirroring the suffix-pattern matcher's
# expectations. Real scip-python prepends ``scip-python python <pkg>
# <version>`` — the suffix matcher ignores that, so tests can use
# minimal strings.
_REQUESTS_GET_SYM = (
    'scip-python . . . app.py/requests/api/get.'
)
_REQUESTS_POST_SYM = (
    'scip-python . . . app.py/requests/api/post.'
)
_REQUESTS_PUT_SYM = (
    'scip-python . . . app.py/requests/api/put.'
)
_REQUESTS_DELETE_SYM = (
    'scip-python . . . app.py/requests/api/delete.'
)
_REQUESTS_SESSION_GET_SYM = (
    'scip-python . . . app.py/requests/sessions/Session#get.'
)
_HTTPX_GET_SYM = (
    'scip-python . . . app.py/httpx/_api/get.'
)
_HTTPX_CLIENT_GET_SYM = (
    'scip-python . . . app.py/httpx/_client/Client#get.'
)
_HTTPX_ASYNC_CLIENT_GET_SYM = (
    'scip-python . . . app.py/httpx/_client/AsyncClient#get.'
)
_AIOHTTP_GET_SYM = (
    'scip-python . . . app.py/aiohttp/client/ClientSession#get.'
)
_URLLIB_URLOPEN_SYM = (
    'scip-python . . . app.py/urllib/request/urlopen.'
)


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the SCIP schema applied."""
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _occ_at(
    text: str, marker: str, symbol: str, *, nth: int = 0,
) -> _ScipOccurrence:
    """Find the nth occurrence of ``marker`` AS A WORD in ``text``;
    return a SCIP occurrence at that position (0-indexed line/col)."""
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
    language: str = 'python',
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
    """Return list of dicts so tests can assert on field names without
    indexing into tuples."""
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
    """Run Phase 2p literal indexing first, then Python HTTP client
    extraction. The extractor reads URL values from
    ``string_literals`` — Phase 2p must populate that table for the
    source first."""
    from docgen.scip_python_http_client_extractor import (
        ingest_python_http_clients,
    )

    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_python_http_clients(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# ---------------------------------------------------------------------------
# requests library
# ---------------------------------------------------------------------------


class TestRequestsLibrary:
    def test_requests_get_with_literal_url(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'import requests\n'
            "resp = requests.get('http://api/users')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['http_method'] == 'GET'
        assert rows[0]['sink_name'] == 'requests.get'

    def test_requests_post_method(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "requests.post('http://api/login', data={})\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _REQUESTS_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['http_method'] == 'POST'
        assert rows[0]['raw_url'] == 'http://api/login'

    def test_requests_put_and_delete(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "requests.put('http://api/items/1', data={})\n"
            "requests.delete('http://api/items/1')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'put', _REQUESTS_PUT_SYM),
            _occ_at(text, 'delete', _REQUESTS_DELETE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        methods = {r['http_method'] for r in rows}
        assert {'PUT', 'DELETE'} == methods

    def test_requests_session_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'session = requests.Session()\n'
            "resp = session.get('http://api/users')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # SCIP would resolve the session.get call to
        # requests/sessions/Session#get. — that's the second 'get' as
        # a word (first 'get' would be requests.get if any; here
        # only session.get exists).
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_SESSION_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['http_method'] == 'GET'
        assert rows[0]['sink_name'] == 'requests.Session.get'


# ---------------------------------------------------------------------------
# httpx library
# ---------------------------------------------------------------------------


class TestHttpxLibrary:
    def test_httpx_module_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "resp = httpx.get('http://api/data')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _HTTPX_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/data'
        assert rows[0]['http_method'] == 'GET'

    def test_httpx_client_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'with httpx.Client() as c:\n'
            "    r = c.get('http://api/x')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # Two 'get' tokens? No — only one (c.get). SCIP attributes
        # it to httpx.Client#get.
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _HTTPX_CLIENT_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['sink_name'] == 'httpx.Client.get'

    def test_httpx_async_client_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'async def fetch():\n'
            '    async with httpx.AsyncClient() as c:\n'
            "        r = await c.get('http://api/async')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _HTTPX_ASYNC_CLIENT_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/async'
        assert rows[0]['sink_name'] == 'httpx.AsyncClient.get'


# ---------------------------------------------------------------------------
# aiohttp + urllib
# ---------------------------------------------------------------------------


class TestOtherLibraries:
    def test_aiohttp_client_session_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'async def fetch():\n'
            '    async with aiohttp.ClientSession() as s:\n'
            "        async with s.get('http://api/v1') as r:\n"
            '            return await r.text()\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _AIOHTTP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/v1'
        assert rows[0]['sink_name'] == 'aiohttp.ClientSession.get'

    def test_urllib_urlopen(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'from urllib.request import urlopen\n'
            "resp = urlopen('http://api/data')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'urlopen', _URLLIB_URLOPEN_SYM, nth=1),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/data'
        # Default method GET; data= kwarg detection is Phase 2s territory
        assert rows[0]['http_method'] == 'GET'


# ---------------------------------------------------------------------------
# Consumer symbol resolution
# ---------------------------------------------------------------------------


class TestConsumerSymbolResolution:
    def test_call_inside_function_resolves_consumer(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            'def fetch_users():\n'
            "    return requests.get('http://api/users').json()\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        fn_id = 'scip-python . . . app.py/fetch_users().'
        _add_scip_symbol(
            conn, canonical_id=fn_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=2,
            kind='Function', qualified_name='app.fetch_users',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
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
        text = "data = requests.get('http://api/x')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] is None

    def test_call_inside_method_picks_method_not_class(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """When the call is inside a method of a class, the consumer
        is the METHOD's symbol, not the class. Same kind-filter
        semantics as Phase 2p ownership lookup."""
        text = (
            'class UserClient:\n'
            '    def fetch(self):\n'
            "        return requests.get('http://api/users')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # Both Class and Method line ranges contain the call
        _add_scip_symbol(
            conn, canonical_id='scip:UserClient#',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=3,
            kind='Class', qualified_name='app.UserClient',
        )
        method_id = 'scip:UserClient#fetch().'
        _add_scip_symbol(
            conn, canonical_id=method_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=3,
            kind='Method', qualified_name='app.UserClient.fetch',
            parent_qualified_name='app.UserClient',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['consumer_symbol_id'] == method_id


# ---------------------------------------------------------------------------
# Adversarial — SCIP filters non-HTTP calls; non-literal URLs skipped
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unrelated_get_call_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``cache.get('key')`` where ``cache`` resolves via SCIP to a
        cache library, NOT a Python HTTP client — the suffix matcher
        rejects, no row emitted."""
        text = "value = cache.get('key')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'get',
                'scip-python . . . app.py/cachelib/Cache#get.',
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert rows == []

    def test_variable_url_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``requests.get(URL)`` with URL as a variable — Phase 2p
        only stores literals at the variable's *definition* site, not
        at the call site. The lookup at the call site's first arg
        position yields ``None``; no row emitted."""
        text = (
            "URL = 'http://api/x'\n"
            'resp = requests.get(URL)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # No row — variable resolution is Phase 2s territory
        assert rows == []

    def test_fstring_url_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``requests.get(f'http://api/{user_id}')`` — Phase 2p
        explicitly skips f-strings, so the lookup returns None."""
        text = (
            'def fetch(user_id):\n'
            "    return requests.get(f'http://api/{user_id}')\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
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
        """A call with no SCIP occurrence at the method's position
        means SCIP didn't classify the symbol. Skip cleanly even
        though the source contains ``requests.get(...)`` syntactically."""
        text = "resp = requests.get('http://api/x')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        # Empty SCIP — no occurrences at all
        index = _make_index(src, tmp_path, [])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert rows == []

    def test_malformed_python_doesnt_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Syntax error in one file → that file is skipped, others
        still emit."""
        broken_text = 'def oops(\n  # missing close paren\n'
        good_text = "resp = requests.get('http://api/ok')\n"
        broken = tmp_path / 'broken.py'
        good = tmp_path / 'good.py'
        broken.write_text(broken_text)
        good.write_text(good_text)
        index = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='broken.py',
                    occurrences=(),
                    symbols=(),
                ),
                _ScipDoc(
                    relative_path='good.py',
                    occurrences=(
                        _occ_at(good_text, 'get', _REQUESTS_GET_SYM),
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
        from docgen.scip_python_http_client_extractor import (
            ingest_python_http_clients,
        )
        rc = ingest_python_http_clients(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Production-realistic SCIP integration
# ---------------------------------------------------------------------------


class TestScipIntegrationChallenges:
    def test_realistic_scip_python_symbol_with_full_preamble(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "resp = requests.get('http://api/realistic')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        realistic_sym = (
            'scip-python python pypi-requests 2.31.0 '
            'requests/api.py/get.'
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', realistic_sym),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # Suffix matcher should still classify as requests.get even
        # with the realistic preamble
        assert any(
            r['raw_url'] == 'http://api/realistic'
            and r['http_method'] == 'GET'
            for r in rows
        )

    def test_mixed_real_and_unrelated_occurrences(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """File contains BOTH a real ``requests.get(...)`` and an
        unrelated ``cache.get(...)``. SCIP classifies both — the
        registry filters to real HTTP-client sinks, no row for the
        unrelated cache call."""
        text = (
            'def list_items():\n'
            "    cached = cache.get('items')\n"
            "    return requests.get('http://api/items').json()\n"
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'get',
                'scip-python . . . cachelib/Cache#get.',
                nth=0,
            ),
            _occ_at(text, 'get', _REQUESTS_GET_SYM, nth=1),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        # Only the real HTTP call is recorded
        urls = [r['raw_url'] for r in rows]
        assert 'http://api/items' in urls
        assert 'items' not in urls

    def test_multi_line_call_args(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Call args formatted across multiple lines — common in
        Python codebases for readability."""
        text = (
            'resp = requests.get(\n'
            "    'http://api/multi-line',\n"
            '    timeout=30,\n'
            ')\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/multi-line'

    def test_url_kwarg_form(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``requests.get(url='http://api/x')`` — passing the URL as
        ``url=`` kwarg instead of positional. Common pattern.

        ``arg_index=0`` from the SinkSpec normally points at the
        first positional, but should also accept ``url=`` since
        that's the documented parameter name."""
        text = "resp = requests.get(url='http://api/kwarg')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/kwarg'


# ---------------------------------------------------------------------------
# Phase 2s wiring — variable URLs resolve via scip_symbols + string_literals
# ---------------------------------------------------------------------------


class TestPhase2sVariableResolution:
    """Phase 2s shipped as a library; this test asserts Phase 8b.1
    consumes it. Variable URL identifiers resolve through
    ``resolve_arg_value`` rather than being silently skipped."""

    def test_variable_url_resolves_via_scip_symbols(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Module-level constant referenced in ``requests.get`` —
        Phase 2s walks ``scip_symbols`` to the def line, finds the
        single literal there, returns ``'resolved-constant'``."""
        text = (
            'BASE_URL = "http://api/users"\n'
            'def fetch_users():\n'
            '    return requests.get(BASE_URL)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        fn_id = 'scip-python . . . app.py/fetch_users().'
        _add_scip_symbol(
            conn, canonical_id=fn_id,
            source_name='myapi', file=str(src.resolve()),
            line_start=2, line_end=3,
            kind='Function', qualified_name='app.fetch_users',
        )
        # The variable's def — Phase 2s looks for this
        _add_scip_symbol(
            conn, canonical_id='scip:app.BASE_URL',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=1,
            kind='Variable', qualified_name='app.BASE_URL',
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_client_calls(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['raw_url'] == 'http://api/users'
        assert rows[0]['http_method'] == 'GET'
        assert rows[0]['confidence'] == 'resolved-constant'

    def test_variable_url_with_no_def_still_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired negative — when ``scip_symbols`` doesn't carry the
        variable's definition, Phase 2s returns unresolved and the
        extractor skips. Paired with a literal call so the test bites
        both stub (no rows) and over-eager resolution (variable
        captured without a def)."""
        text = (
            'def fetch():\n'
            '    requests.get("http://api/literal")\n'
            '    requests.get(UNDEFINED_URL)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        _add_scip_symbol(
            conn, canonical_id='scip:app.fetch',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=3,
            kind='Function', qualified_name='app.fetch',
        )
        # NO scip_symbol for UNDEFINED_URL
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM, nth=0),
            _occ_at(text, 'get', _REQUESTS_GET_SYM, nth=1),
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
        text1 = "requests.get('http://api/old')\n"
        src = tmp_path / 'app.py'
        src.write_text(text1)
        index1 = _make_index(src, tmp_path, [
            _occ_at(text1, 'get', _REQUESTS_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index1,
        )
        # Replace
        text2 = "requests.post('http://api/new')\n"
        src.write_text(text2)
        index2 = _make_index(src, tmp_path, [
            _occ_at(text2, 'post', _REQUESTS_POST_SYM),
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
        # Pre-existing row from a different source
        conn.execute(
            '''INSERT INTO http_client_calls
               (source_name, raw_url, http_method,
                call_site_file, call_site_line, sink_name, confidence)
               VALUES (?, ?, ?, ?, ?, ?, ?)''',
            ('other', 'http://other/x', 'GET',
             '/x.py', 1, 'requests.get', 'literal'),
        )
        conn.commit()

        text = "requests.get('http://api/mine')\n"
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _REQUESTS_GET_SYM),
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
