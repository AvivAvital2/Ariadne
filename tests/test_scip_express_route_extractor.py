"""Contract for SCIP-driven Express/Koa route extractor (Phase 8a.3).

Architecture matches Phase 8a.1 (Akka HTTP) and 8a.2 (Flask/FastAPI):

- **SCIP** filters which call sites are real Express/Koa router methods.
  No false positives from any local function called ``get`` or ``post``.
- **ast-grep / tree-sitter-javascript** parses the call structure: the
  ``call_expression`` and its ``arguments`` child, the first-argument
  string literal that holds the route path.

Symbol-suffix classification:

- ``...Application#get().`` / ``...Application.get.`` — Express ``app``
- ``...Router#get().`` / ``...Router.get.`` — Express/Koa Router instance
- ``...IRouter#get().`` / ``...IRouterMatcher#get().`` — Express type
  defs (callable interfaces from ``@types/express``)

Path normalization to unified template form:

- Express ``:id`` → ``{id}``
- Express ``:id?`` (optional) → ``{id}`` (we drop the optionality marker)
- Multiple params per path: ``/users/:userId/posts/:postId`` →
  ``/users/{userId}/posts/{postId}``

These tests are RED until ``docgen/scip_express_route_extractor.py``
implements ``ingest_express_routes``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_express_route_extractor import ingest_express_routes
from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
)
from docgen.scip_string_literal_extractor import ingest_string_literals


def _run_pipeline(
    *, source_name: str, source_root: Path,
    conn: sqlite3.Connection, index: ScipIndex,
) -> int:
    """Run Phase 2p literal indexing first, then Express route
    extraction. The route extractor reads the path string from
    ``string_literals`` — Phase 2p must populate that table for the
    source first."""
    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_express_routes(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# Synthetic SCIP symbols matching the suffix-pattern matcher.
_APP_GET_SYM = (
    'scip-typescript . . . src/app.ts/express/Application#get().'
)
_APP_POST_SYM = (
    'scip-typescript . . . src/app.ts/express/Application#post().'
)
_APP_PUT_SYM = (
    'scip-typescript . . . src/app.ts/express/Application#put().'
)
_APP_DELETE_SYM = (
    'scip-typescript . . . src/app.ts/express/Application#delete().'
)
_ROUTER_GET_SYM = (
    'scip-typescript . . . src/app.ts/express/Router#get().'
)
_ROUTER_POST_SYM = (
    'scip-typescript . . . src/app.ts/express/Router#post().'
)
_KOA_ROUTER_GET_SYM = (
    'scip-typescript . . . src/app.ts/@koa/router/Router#get().'
)
_KOA_ROUTER_POST_SYM = (
    'scip-typescript . . . src/app.ts/@koa/router/Router#post().'
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
    return a SCIP occurrence at that position (0-indexed line/col).

    Word-boundary matching prevents false matches when one keyword is a
    substring of another (e.g., ``get`` inside ``getCachedUser`` or
    ``post`` inside ``compose``)."""
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
            f'marker {marker!r} (nth={nth}) not found as a word in text',
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


def _query_endpoints(
    conn: sqlite3.Connection, source_name: str,
) -> set[tuple[str, str]]:
    cur = conn.execute(
        'SELECT http_method, path_template FROM api_endpoints '
        'WHERE source_name = ?',
        (source_name,),
    )
    return {(row[0], row[1]) for row in cur.fetchall()}


# ---------------------------------------------------------------------------
# Tree-sitter-javascript grammar contract — pin the AST shape we depend on
# ---------------------------------------------------------------------------


class TestJavaScriptGrammarAssumptions:
    """Pin the tree-sitter-javascript AST shape that the extractor relies
    on. If the grammar evolves and renames or restructures these nodes,
    the assertions here pinpoint the assumption that broke instead of
    leaving the extractor mysteriously empty.
    """

    def test_javascript_language_is_supported(self) -> None:
        from ast_grep_py import SgRoot

        root = SgRoot('const x = 1;', 'javascript').root()
        assert root.kind() == 'program'

    def test_member_call_has_member_expression_callee(self) -> None:
        """``app.get(...)`` parses as a ``call_expression`` whose first
        child is a ``member_expression`` whose final
        ``property_identifier`` is the method name. The extractor
        depends on that final identifier's position to match SCIP
        occurrences."""
        from ast_grep_py import SgRoot

        text = "app.get('/x', h);\n"
        root = SgRoot(text, 'javascript').root()

        calls = list(root.find_all(kind='call_expression'))
        assert calls, 'expected at least one call_expression'
        call = calls[0]
        children = list(call.children())
        kinds = [c.kind() for c in children]
        assert kinds[0] == 'member_expression', (
            f'expected member_expression as first child, got {kinds}'
        )
        assert 'arguments' in kinds, (
            f'expected arguments as a child, got {kinds}'
        )

        # The property_identifier inside the member_expression is the
        # method name — that's what scip-typescript points at.
        member = children[0]
        prop_ids = [
            c for c in member.children()
            if c.kind() == 'property_identifier'
        ]
        assert prop_ids, (
            f'expected property_identifier inside member_expression; '
            f'children={[c.kind() for c in member.children()]}'
        )
        assert prop_ids[-1].text() == 'get'

    def test_string_literal_text_includes_quotes(self) -> None:
        """``'/path'`` / ``"/path"`` parse as ``string`` nodes whose
        ``.text()`` includes the surrounding quote characters; the
        extractor strips them to recover the value."""
        from ast_grep_py import SgRoot

        text = "app.get('/x', h);\n"
        root = SgRoot(text, 'javascript').root()
        strings = list(root.find_all(kind='string'))
        assert strings
        assert strings[0].text() in ("'/x'", '"/x"')


# ---------------------------------------------------------------------------
# Express app — app.<verb>('/path', handler)
# ---------------------------------------------------------------------------


class TestExpressApp:
    def test_app_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "app.get('/status', (req, res) => res.send('ok'));\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])

        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/status') in _query_endpoints(conn, 'myapi')

    def test_app_post(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "app.post('/login', (req, res) => res.send('ok'));\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _APP_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('POST', '/login') in _query_endpoints(conn, 'myapi')

    def test_app_put_and_delete(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "app.put('/items/:id', updateItem);\n"
            "app.delete('/items/:id', deleteItem);\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'put', _APP_PUT_SYM),
            _occ_at(text, 'delete', _APP_DELETE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        assert ('PUT', '/items/{id}') in rows
        assert ('DELETE', '/items/{id}') in rows


# ---------------------------------------------------------------------------
# Express Router — router.<verb>('/path', handler)
# ---------------------------------------------------------------------------


class TestExpressRouter:
    def test_router_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "const router = express.Router();\n"
            "router.get('/users', listUsers);\n"
        )
        src = tmp_path / 'routes.js'
        src.write_text(text)
        # Only one `get` in the source — SCIP places its occurrence on
        # the method name in the router.get(...) call.
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _ROUTER_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users') in _query_endpoints(conn, 'myapi')

    def test_router_post(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "router.post('/users', createUser);\n"
        src = tmp_path / 'routes.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _ROUTER_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('POST', '/users') in _query_endpoints(conn, 'myapi')


# ---------------------------------------------------------------------------
# Koa Router — same call shape as Express Router but @koa/router package
# ---------------------------------------------------------------------------


class TestKoaRouter:
    def test_koa_router_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "const Router = require('@koa/router');\n"
            "const router = new Router();\n"
            "router.get('/health', ctx => { ctx.body = 'ok'; });\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _KOA_ROUTER_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/health') in _query_endpoints(conn, 'myapi')

    def test_koa_router_post(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "router.post('/login', login);\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _KOA_ROUTER_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('POST', '/login') in _query_endpoints(conn, 'myapi')


# ---------------------------------------------------------------------------
# Path-parameter normalization
# ---------------------------------------------------------------------------


class TestPathNormalization:
    def test_single_param(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "app.get('/users/:id', show);\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users/{id}') in _query_endpoints(conn, 'myapi')

    def test_multiple_params(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "app.get('/users/:userId/posts/:postId', show);\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/users/{userId}/posts/{postId}') in rows

    def test_optional_param_strips_question_mark(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = "app.get('/users/:id?', show);\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users/{id}') in _query_endpoints(conn, 'myapi')


# ---------------------------------------------------------------------------
# Multiple routes / files
# ---------------------------------------------------------------------------


class TestMultiple:
    def test_two_routes_in_one_file(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "app.get('/a', h1);\n"
            "app.post('/b', h2);\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
            _occ_at(text, 'post', _APP_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/a') in rows
        assert ('POST', '/b') in rows

    def test_routes_from_multiple_files(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text_a = "app.get('/from-a', h);\n"
        text_b = "router.post('/from-b', h);\n"
        a = tmp_path / 'a.js'
        b = tmp_path / 'sub' / 'b.js'
        a.write_text(text_a)
        b.parent.mkdir()
        b.write_text(text_b)

        index = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='a.js',
                    occurrences=(
                        _occ_at(text_a, 'get', _APP_GET_SYM),
                    ),
                    symbols=(),
                ),
                _ScipDoc(
                    relative_path='sub/b.js',
                    occurrences=(
                        _occ_at(text_b, 'post', _ROUTER_POST_SYM),
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
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/from-a') in rows
        assert ('POST', '/from-b') in rows


# ---------------------------------------------------------------------------
# Adversarial — SCIP filters out non-Express symbols, plus other edge cases
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unrelated_get_call_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``cache.get('key')`` where ``cache`` is some unrelated lib —
        SCIP symbol points to ``some/Cache#get.``, NOT a web framework.
        Filtered out by the symbol-suffix matcher."""
        text = "const v = cache.get('key');\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'get',
                'scip-typescript . . . app.js/some/Cache#get.',
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()

    def test_call_with_no_scip_occurrence_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A call with no matching SCIP occurrence at all → skip cleanly."""
        text = "app.get('/x', h);\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        # Empty SCIP — no Express/Koa occurrences for the call site
        index = _make_index(src, tmp_path, [])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()

    def test_malformed_js_doesnt_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Syntax-error file → that file is skipped, others still emit."""
        broken_text = 'function broken( {\n'
        good_text = "app.get('/ok', h);\n"
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
                        _occ_at(good_text, 'get', _APP_GET_SYM),
                    ),
                    symbols=(),
                ),
            ),
            source_root=tmp_path,
        )
        # Should not raise
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/ok') in _query_endpoints(conn, 'myapi')

    def test_call_with_variable_path_arg_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``app.get(MY_PATH, handler)`` with a variable arg — skip
        cleanly. Phase 8c will resolve constants via SCIP refs; for
        now, only literal string paths are accepted."""
        text = (
            "const MY_PATH = '/login';\n"
            "app.get(MY_PATH, login);\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/login') not in rows

    def test_template_string_with_interpolation_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``app.get(`/api/${prefix}/items`, handler)`` — template literal
        with interpolation. Static path can't be resolved at index
        time; skip without crashing."""
        text = (
            "const prefix = 'v1';\n"
            "app.get(`/api/${prefix}/items`, h);\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        # Don't emit a literal endpoint with the interpolation visible
        for _method, path in rows:
            assert '${' not in path, (
                f'Interpolation leaked into path_template: {path!r}'
            )

    def test_missing_index_file_no_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``index_factory=None`` with no .scip on disk → return 0
        cleanly."""
        rc = ingest_express_routes(
            source_name='myapi',
            source_root=tmp_path,
            conn=conn,
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Production-realistic SCIP integration challenges
# ---------------------------------------------------------------------------


class TestScipIntegrationChallenges:
    """Adversarial tests probing SCIP integration edge cases.

    Each test describes a production-realistic scenario the extractor
    SHOULD handle. Failures here surface real impl gaps that synthetic
    happy-path tests don't expose.
    """

    def test_realistic_scip_typescript_symbol_with_full_preamble(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Real scip-typescript canonical_ids carry the verbose
        ``scip-typescript npm <pkg> <version> <descriptors>`` preamble.
        The suffix-based matcher should still classify them
        correctly."""
        text = "app.get('/realistic', h);\n"
        src = tmp_path / 'app.ts'
        src.write_text(text)
        realistic_sym = (
            'scip-typescript npm @types/express 4.17.21 '
            'lib/application.d.ts/Application#get().'
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', realistic_sym),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/realistic') in _query_endpoints(conn, 'myapi')

    def test_async_arrow_handler(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``app.get('/x', async (req, res) => { ... })`` — handler is
        an async arrow function. The path-arg extraction shouldn't
        depend on the handler shape at all."""
        text = (
            "app.get('/users', async (req, res) => {\n"
            "  const u = await fetchUsers();\n"
            "  res.json(u);\n"
            "});\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users') in _query_endpoints(conn, 'myapi')

    def test_mixed_real_and_unrelated_scip_occurrences(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """File contains BOTH a real ``app.get(...)`` and an unrelated
        ``cache.get(...)`` call. SCIP has occurrences for both — the
        matcher must filter to web-only and not emit a row for the
        unrelated one."""
        text = (
            "app.get('/api/items', (req, res) => {\n"
            "  const cached = cache.get('items');\n"
            "  res.json(cached || []);\n"
            "});\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM, nth=0),
            _occ_at(
                text, 'get',
                'scip-typescript . . . app.js/cachelib/Cache#get.',
                nth=1,
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/api/items') in rows
        # cache.get('items') did NOT pollute api_endpoints
        assert ('GET', 'items') not in rows

    def test_typescript_file_with_express_route(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``.ts`` file with type-annotated handler — ast-grep parses
        TypeScript-light using the JavaScript grammar (matches the
        catalog extractor's convention). The route extraction should
        still work as long as the call_expression structure matches."""
        text = (
            "import { Request, Response } from 'express';\n"
            "app.get('/typed', (req: Request, res: Response) => {\n"
            "  res.send('ok');\n"
            "});\n"
        )
        src = tmp_path / 'app.ts'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/typed') in _query_endpoints(conn, 'myapi')


# ---------------------------------------------------------------------------
# Re-ingest semantics — same as Akka / Python web extractors
# ---------------------------------------------------------------------------


class TestReIngest:
    def test_re_ingest_replaces_pattern_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        # Pre-existing swagger row — must NOT be cleared
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('swag1', 'myapi', 'GET', '/swagger', 'sym', 'swagger'),
        )
        conn.commit()

        text1 = "app.get('/old', h);\n"
        src = tmp_path / 'app.js'
        src.write_text(text1)
        index1 = _make_index(src, tmp_path, [
            _occ_at(text1, 'get', _APP_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index1,
        )
        assert ('GET', '/old') in _query_endpoints(conn, 'myapi')

        # Re-ingest with new content
        text2 = "app.post('/new', h);\n"
        src.write_text(text2)
        index2 = _make_index(src, tmp_path, [
            _occ_at(text2, 'post', _APP_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index2,
        )

        rows = _query_endpoints(conn, 'myapi')
        # Old pattern row gone
        assert ('GET', '/old') not in rows
        # New pattern row present
        assert ('POST', '/new') in rows
        # Swagger row preserved
        assert ('GET', '/swagger') in rows


def test_non_js_documents_are_not_read(
    tmp_path: Path, conn: sqlite3.Connection, monkeypatch,
) -> None:
    """Only JS/TS-family docs are read for routes. A Python/Scala doc —
    e.g. a Databricks spool's Spark/SDK source — must be skipped WITHOUT
    being read/parsed, so the extractor doesn't walk the whole corpus file
    by file for routes it cannot contain. The sibling HTTP-client
    extractors already guard this way; the route extractor now matches.

    Asserted on the read itself (not route output) because non-JS text is
    already inert to extraction — the defect this guards is the wasted
    per-file read/parse over a large non-JS corpus.
    """
    keep = "app.get('/keep', (req, res) => res.send('ok'));\n"
    (tmp_path / 'api.js').write_text(keep)
    (tmp_path / 'service.py').write_text("app.get('/drop', h);\n")
    (tmp_path / 'Engine.scala').write_text("app.get('/nope', h)\n")

    read_paths: list[str] = []
    real_read_text = Path.read_text

    def _spy_read_text(self, *a, **k):
        read_paths.append(self.name)
        return real_read_text(self, *a, **k)
    monkeypatch.setattr(Path, 'read_text', _spy_read_text)

    index = ScipIndex(
        documents=tuple(
            _ScipDoc(relative_path=rel, occurrences=(), symbols=())
            for rel in ('api.js', 'service.py', 'Engine.scala')
        ),
        source_root=tmp_path,
    )
    ingest_express_routes(
        source_name='myapi', source_root=tmp_path,
        conn=conn, index_factory=lambda: index,
    )
    assert 'api.js' in read_paths          # JS doc is read
    assert 'service.py' not in read_paths  # non-JS docs are skipped, not read
    assert 'Engine.scala' not in read_paths
