"""Contract for SCIP-driven Akka HTTP route extractor (Phase 8a.1).

Architecture: matches the project's SCIP-everywhere convention (per
``docgen/scip_extractor.py`` and the SCIP integration plan). Scala
parsing in this codebase routes through SCIP indexes — the compiler-
aware path that knows resolved symbols, imports, and aliases. ast-grep
was deliberately rejected because tree-sitter doesn't see implicits,
type-resolved overloads, etc.

Tests synthesize the SCIP intermediates directly via the Phase 2a
pattern (``_ScipDoc`` / ``_ScipOccurrence``) and pair them with real
source files. The extractor uses occurrence positions as precise
anchors, then reads source text at those positions to extract the
literal path arguments and compose templates by brace nesting.

Akka HTTP directives recognized by symbol-suffix pattern:
- ``...Directives#path().`` — path
- ``...Directives#pathPrefix().`` — pathPrefix
- ``...Directives#get.`` / ``.post.`` / ``.put.`` / ``.delete.`` /
  ``.patch.`` / ``.head.`` / ``.options.`` — verbs

These tests are RED until ``docgen/scip_akka_http_extractor.py`` is
rewritten against this SCIP-driven contract.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_akka_http_extractor import ingest_akka_http_routes
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
    """Run Phase 2p literal indexing first, then the Akka route
    extractor. The extractor's contract requires string_literals to be
    populated for the source before it runs — this helper enforces that
    ordering once per test instead of duplicating two ingest calls."""
    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_akka_http_routes(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# Symbol templates — SCIP canonical_id format. The extractor matches by
# trailing-suffix pattern so the leading project/version preamble is
# free.
_PATH_SYM = 'akka/http/scaladsl/server/Directives#path().'
_PATH_PREFIX_SYM = 'akka/http/scaladsl/server/Directives#pathPrefix().'
_GET_SYM = 'akka/http/scaladsl/server/Directives#get.'
_POST_SYM = 'akka/http/scaladsl/server/Directives#post.'
_PUT_SYM = 'akka/http/scaladsl/server/Directives#put.'
_DELETE_SYM = 'akka/http/scaladsl/server/Directives#delete.'


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
    """Find the nth occurrence of ``marker`` AS A WORD in ``text`` and
    return a SCIP occurrence at that position (0-indexed line/col).

    Word-boundary matching matters here because the marker ``'path'`` is
    a substring of ``'pathPrefix'`` — naive ``text.index`` would point
    SCIP at the wrong identifier and the extractor would parse the
    wrong argument. Real SCIP indexes don't have this problem (each
    occurrence is precisely positioned by the compiler) but our test
    fixture has to emulate that precision.
    """
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
# Simple routes
# ---------------------------------------------------------------------------


class TestScalaGrammarAssumptions:
    """Pin the tree-sitter-scala AST shape that the extractor relies on.

    These tests exist so that if tree-sitter-scala's grammar evolves
    (node renames, restructuring), failures here pinpoint the
    assumption that broke instead of leaving the extractor mysteriously
    silent. Each assertion mirrors a structural assumption made in
    ``docgen/scip_akka_http_extractor.py``.
    """

    def test_scala_language_is_supported(self) -> None:
        """``ast-grep-py`` in this environment ships tree-sitter-scala
        — the extractor depends on this."""
        from ast_grep_py import SgRoot

        root = SgRoot('val x = 1', 'scala').root()
        assert root.kind() == 'compilation_unit'

    def test_apply_with_block_is_outer_call_with_block_child(
        self,
    ) -> None:
        """The Scala apply-with-block syntax ``f(args) { block }``
        parses as an OUTER ``call_expression`` whose children are an
        INNER ``call_expression`` (containing the function and
        arguments) and a ``block`` (the body). This is the linchpin
        of the extractor's scope-detection — if it changes, the
        whole pipeline breaks silently. Pin it explicitly."""
        from ast_grep_py import SgRoot

        text = 'val r = path("login") { post { ok } }\n'
        root = SgRoot(text, 'scala').root()
        # Find the outer call_expression — the one whose text starts
        # with "path(" and contains the block.
        outer = None
        for call in root.find_all(kind='call_expression'):
            if call.text().startswith('path("login") {'):
                outer = call
                break
        assert outer is not None, (
            'Expected outer call_expression spanning the whole '
            'apply-with-block'
        )

        children = list(outer.children())
        kinds = [c.kind() for c in children]
        # First child must be the inner call_expression (function +
        # arguments); second must be the block.
        assert 'call_expression' in kinds, (
            f'Expected inner call_expression as outer child; got {kinds}'
        )
        assert 'block' in kinds, (
            f'Expected block as outer child; got {kinds}'
        )

    def test_arguments_node_wraps_punctuation_and_expression(
        self,
    ) -> None:
        """An ``arguments`` node has children ``[ '(', <expr>, ')' ]``
        — the extractor filters out the parens to get the expression
        list."""
        from ast_grep_py import SgRoot

        text = 'val r = f("x")\n'
        root = SgRoot(text, 'scala').root()

        args_node = None
        for n in root.find_all(kind='arguments'):
            args_node = n
            break
        assert args_node is not None
        kinds = [c.kind() for c in args_node.children()]
        assert kinds[0] == '('
        assert kinds[-1] == ')'
        # Between the parens, at least one meaningful expression.
        meaningful = kinds[1:-1]
        assert meaningful, 'Expected an expression inside arguments'

    def test_string_literal_is_quoted_string_node(self) -> None:
        """String literals in Scala parse as ``string`` nodes whose
        text includes the surrounding quotes; the extractor strips
        them to get the value."""
        from ast_grep_py import SgRoot

        text = 'val r = f("hello")\n'
        root = SgRoot(text, 'scala').root()

        strings = list(root.find_all(kind='string'))
        assert strings, 'Expected at least one string node'
        assert strings[0].text() == '"hello"'

    def test_inner_and_outer_call_share_callee_position(
        self,
    ) -> None:
        """The inner and outer ``call_expression`` of an
        apply-with-block share the same callee identifier position —
        this is why the call index buckets calls by position
        (multiple per key) rather than overwriting."""
        from ast_grep_py import SgRoot

        text = 'val r = path("x") { y }\n'
        root = SgRoot(text, 'scala').root()

        calls_by_pos: dict[tuple[int, int], int] = {}
        for call in root.find_all(kind='call_expression'):
            children = list(call.children())
            if not children:
                continue
            first = children[0]
            # Walk down to identifier
            current = first
            while current.kind() == 'call_expression':
                sub = list(current.children())
                if not sub:
                    break
                current = sub[0]
            if current.kind() not in ('identifier', 'simple_identifier'):
                continue
            r = current.range()
            key = (r.start.line, r.start.column)
            calls_by_pos[key] = calls_by_pos.get(key, 0) + 1

        # The position of `path` (line 0, somewhere) should have
        # at least 2 calls bucketed (inner f(x) and outer f(x){y}).
        max_count = max(calls_by_pos.values()) if calls_by_pos else 0
        assert max_count >= 2, (
            'Expected inner+outer call_expressions to share callee '
            f'position; got counts {calls_by_pos}'
        )


class TestSimpleRoute:
    def test_path_post_creates_endpoint(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``path("login") { post { ... } }`` with the SCIP occurrences
        positioned at the directive keywords → POST /login row in
        api_endpoints."""
        text = (
            'val routes =\n'
            '  path("login") {\n'
            '    post { complete("ok") }\n'
            '  }\n'
        )
        src = tmp_path / 'Routes.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'post', _POST_SYM),
        ])

        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )

        rows = _query_endpoints(conn, 'myapi')
        assert ('POST', '/login') in rows

    def test_get_creates_get_endpoint(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Same shape with ``get`` instead of ``post``."""
        text = 'val r = path("status") { get { complete("ok") } }\n'
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'get', _GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/status') in _query_endpoints(conn, 'myapi')


# ---------------------------------------------------------------------------
# Nested path composition
# ---------------------------------------------------------------------------


class TestNestedPaths:
    def test_pathprefix_then_path_composes(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``pathPrefix("api") { path("login") { post { ... } } }`` →
        POST /api/login. Composition tracks nested SCIP-anchored
        directive scopes."""
        text = (
            'val routes =\n'
            '  pathPrefix("api") {\n'
            '    path("login") {\n'
            '      post { complete("ok") }\n'
            '    }\n'
            '  }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'pathPrefix', _PATH_PREFIX_SYM),
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'post', _POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )

        assert ('POST', '/api/login') in _query_endpoints(conn, 'myapi')

    def test_three_level_nesting(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Three nested prefix levels compose: /api/v1/users."""
        text = (
            'val r =\n'
            '  pathPrefix("api") {\n'
            '    pathPrefix("v1") {\n'
            '      path("users") {\n'
            '        get { complete("[]") }\n'
            '      }\n'
            '    }\n'
            '  }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'pathPrefix', _PATH_PREFIX_SYM, nth=0),
            _occ_at(text, 'pathPrefix', _PATH_PREFIX_SYM, nth=1),
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'get', _GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/api/v1/users') in _query_endpoints(conn, 'myapi')


# ---------------------------------------------------------------------------
# Multiple verbs / routes
# ---------------------------------------------------------------------------


class TestMultipleRoutes:
    def test_concat_with_tilde_creates_two_endpoints(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``path("items") { get {} ~ post {} }`` → both GET /items and
        POST /items as separate rows."""
        text = (
            'val r = path("items") {\n'
            '  get { complete("list") } ~\n'
            '  post { complete("create") }\n'
            '}\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'get', _GET_SYM),
            _occ_at(text, 'post', _POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )

        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/items') in rows
        assert ('POST', '/items') in rows

    def test_two_unrelated_routes_in_one_file(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two ``val routes`` declarations in the same file produce
        independent endpoints."""
        text = (
            'val r1 = path("a") { get { complete("a") } }\n'
            'val r2 = path("b") { post { complete("b") } }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM, nth=0),
            _occ_at(text, 'get', _GET_SYM),
            _occ_at(text, 'path', _PATH_SYM, nth=1),
            _occ_at(text, 'post', _POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )

        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/a') in rows
        assert ('POST', '/b') in rows


class TestMultipleFiles:
    def test_routes_from_multiple_files(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A SCIP index with multiple ``_ScipDoc`` entries → one row
        per route across all files."""
        text_a = 'val r = path("from-a") { get { complete("ok") } }\n'
        text_b = 'val r = path("from-b") { post { complete("ok") } }\n'
        a = tmp_path / 'A.scala'
        b = tmp_path / 'sub' / 'B.scala'
        a.write_text(text_a)
        b.parent.mkdir()
        b.write_text(text_b)

        # Build occurrences per doc
        index = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='A.scala',
                    occurrences=(
                        _occ_at(text_a, 'path', _PATH_SYM),
                        _occ_at(text_a, 'get', _GET_SYM),
                    ),
                    symbols=(),
                ),
                _ScipDoc(
                    relative_path='sub/B.scala',
                    occurrences=(
                        _occ_at(text_b, 'path', _PATH_SYM),
                        _occ_at(text_b, 'post', _POST_SYM),
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
# Path parameters
# ---------------------------------------------------------------------------


class TestPathParameters:
    def test_segment_matcher_becomes_placeholder(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``path("users" / Segment) { get { ... } }`` →
        path_template contains a ``{...}`` placeholder where Segment was."""
        import re
        text = (
            'val r = path("users" / Segment) { get { complete("ok") } }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'get', _GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        get_paths = [p for m, p in rows if m == 'GET']
        assert any(
            re.match(r'^/users/\{[^/]+\}$', p) for p in get_paths
        ), f'Expected /users/{{...}}, got {get_paths!r}'


# ---------------------------------------------------------------------------
# Adversarial / edge cases
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_no_akka_occurrences_no_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """SCIP index with no Akka HTTP occurrences → no api_endpoints
        rows. The extractor only fires for symbols matching the
        registry."""
        # Source has a function called `path` but it's NOT Akka HTTP —
        # SCIP would resolve it to a different symbol. Test by giving
        # NO occurrences for that name (the SCIP perspective: it's
        # not an Akka call).
        text = (
            'object Foo { def path(s: String): Int = 0 }\n'
            'val x = Foo.path("notroute")\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [])

        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()

    def test_unrelated_symbol_with_path_in_name_ignored(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A SCIP occurrence whose symbol contains 'path' but isn't an
        Akka HTTP directive (different package) is filtered out by the
        suffix matcher."""
        text = 'val x = somepkg.path("X")\n'
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'path',
                'com/other/somepkg.path().',  # not Akka HTTP
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()

    def test_missing_index_file_no_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """When ``index_factory=None`` and no .scip exists at the
        expected path, function returns 0 cleanly (matches the
        ScipUnavailableError-degraded-fallback pattern)."""
        # No .ariadne/index.scip in tmp_path
        rc = ingest_akka_http_routes(
            source_name='myapi',
            source_root=tmp_path,
            conn=conn,
        )
        assert rc == 0
        assert _query_endpoints(conn, 'myapi') == set()


# ---------------------------------------------------------------------------
# Re-ingest semantics — same as Swagger / config / string-literal
# ---------------------------------------------------------------------------


class TestScipIntegrationChallenges:
    """Adversarial tests probing SCIP integration edge cases.

    Each test describes a production-realistic scenario the extractor
    SHOULD handle. Failures here surface real impl gaps that synthetic
    happy-path tests didn't expose.
    """

    def test_realistic_scip_java_symbol_format(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Real scip-java canonical_ids carry a verbose preamble:
        ``scip-java semanticdb maven <repo> <version>
        <descriptors>``. Suffix matcher should still classify."""
        text = (
            'val routes = path("realistic") { '
            'get { complete("ok") } }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        realistic_path_sym = (
            'scip-java semanticdb maven com.typesafe.akka '
            'akka-http-core_2.13 10.5.0 '
            'akka/http/scaladsl/server/Directives#path().'
        )
        realistic_get_sym = (
            'scip-java semanticdb maven com.typesafe.akka '
            'akka-http-core_2.13 10.5.0 '
            'akka/http/scaladsl/server/Directives#get.'
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', realistic_path_sym),
            _occ_at(text, 'get', realistic_get_sym),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/realistic') in _query_endpoints(conn, 'myapi')

    def test_routes_concatenated_at_top_level(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two top-level path scopes joined with ``~`` (Scala route
        concat) — both should emit independent endpoints with their
        respective paths, NOT a composed /a/b path."""
        text = (
            'val routes =\n'
            '  path("a") {\n'
            '    get { complete("a") }\n'
            '  } ~\n'
            '  path("b") {\n'
            '    post { complete("b") }\n'
            '  }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM, nth=0),
            _occ_at(text, 'get', _GET_SYM),
            _occ_at(text, 'path', _PATH_SYM, nth=1),
            _occ_at(text, 'post', _POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        # Each top-level path has only its OWN segment, not composed
        assert ('GET', '/a') in rows
        assert ('POST', '/b') in rows
        # Critically: no composed /a/b — they're siblings, not nested
        assert ('GET', '/a/b') not in rows
        assert ('POST', '/a/b') not in rows

    def test_path_with_string_interpolation_skipped_or_handled(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``path(s"$prefix/login")`` — Scala string interpolation.
        We can't statically resolve ``$prefix`` without SCIP value
        propagation (Phase 8c territory). The extractor must NOT
        crash AND must NOT emit a literal endpoint with the
        interpolation syntax visible in path_template."""
        text = (
            'val prefix = "api"\n'
            'val routes =\n'
            '  path(s"$prefix/login") {\n'
            '    post { complete("ok") }\n'
            '  }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'post', _POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        # Crucially: don't emit `/$prefix/login` as a literal path
        for method, path in rows:
            assert '$' not in path, (
                f'Interpolation leaked into path_template: {path!r}'
            )

    def test_unrelated_directives_member_not_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A SCIP occurrence with a symbol that mentions Directives
        but isn't a recognized HTTP directive (e.g.,
        ``Directives#extractRequestContext.``) should NOT emit any
        endpoint row."""
        text = (
            'val routes =\n'
            '  extractRequestContext { ctx =>\n'
            '    complete("ok")\n'
            '  }\n'
        )
        src = tmp_path / 'R.scala'
        src.write_text(text)
        # Akka's extractRequestContext is a directive but NOT one we
        # care about for endpoint extraction (it doesn't introduce a
        # path or HTTP method).
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'extractRequestContext',
                'akka/http/scaladsl/server/Directives#'
                'extractRequestContext.',
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()


class TestReIngest:
    def test_re_ingest_replaces_pattern_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Re-running clears prior pattern-resolution rows for this
        source. Swagger-resolution rows (different ``resolution_source``)
        stay put."""
        # Pre-existing swagger row
        conn.execute(
            'INSERT INTO api_endpoints VALUES '
            '(?, ?, ?, ?, ?, ?)',
            ('swag1', 'myapi', 'GET', '/swagger', 'sym',
             'swagger'),
        )
        conn.commit()

        text = 'val r = path("akka") { get { complete("ok") } }\n'
        src = tmp_path / 'R.scala'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'path', _PATH_SYM),
            _occ_at(text, 'get', _GET_SYM),
        ])

        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )

        rows = _query_endpoints(conn, 'myapi')
        # Akka pattern row added
        assert ('GET', '/akka') in rows
        # Swagger row preserved
        assert ('GET', '/swagger') in rows

        # Now re-ingest with new content
        text2 = 'val r = path("changed") { post { complete("ok") } }\n'
        src.write_text(text2)
        index2 = _make_index(src, tmp_path, [
            _occ_at(text2, 'path', _PATH_SYM),
            _occ_at(text2, 'post', _POST_SYM),
        ])

        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index2,
        )

        rows = _query_endpoints(conn, 'myapi')
        # Old Akka pattern row gone
        assert ('GET', '/akka') not in rows
        # New Akka pattern row present
        assert ('POST', '/changed') in rows
        # Swagger row STILL preserved
        assert ('GET', '/swagger') in rows
