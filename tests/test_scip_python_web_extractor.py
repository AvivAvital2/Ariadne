"""Contract for SCIP-driven Python web framework route extractor (Phase 8a.2).

Architecture matches Phase 8a.1: tests synthesize ``_ScipDoc`` /
``_ScipOccurrence`` (Phase 2a pattern) plus real Python source files.
The extractor uses SCIP occurrence positions to identify *real*
Flask/FastAPI decorators (no false positives from any
``@x.route(...)`` decorator), then parses the decorator AST via
``ast.parse`` to extract paths/methods cleanly.

Symbol-suffix classification:

- ``...Flask#route().`` / ``...flask.Flask.route()`` — Flask classic
  ``@app.route(...)``
- ``...Flask#get.`` / ``...FastAPI#get.`` / ``...APIRouter#get.`` (and
  other verbs) — Flask 2 / FastAPI verb decorators

Path normalization to unified template form:

- Flask ``<id>`` / ``<int:id>`` → ``{id}``
- FastAPI ``{id}`` / ``{id:int}`` → ``{id}``

Tests pin behavior — what's in ``api_endpoints`` after
``ingest_python_routes`` runs.
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
from docgen.scip_python_web_extractor import ingest_python_routes
from docgen.scip_string_literal_extractor import ingest_string_literals


def _run_pipeline(
    *, source_name: str, source_root: Path,
    conn: sqlite3.Connection, index: ScipIndex,
) -> int:
    """Run Phase 2p literal indexing first, then Python-web route
    extraction. The route extractor reads literal values from
    ``string_literals`` — Phase 2p must populate that table for the
    source before extraction can resolve any path arg."""
    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_python_routes(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# Synthetic SCIP symbols matching the suffix-pattern matcher.
_FLASK_ROUTE_SYM = 'scip-python . . . app.py/flask/Flask#route().'
_FLASK_POST_SYM = 'scip-python . . . app.py/flask/Flask#post.'
_FASTAPI_GET_SYM = (
    'scip-python . . . app.py/fastapi/FastAPI#get.'
)
_FASTAPI_POST_SYM = (
    'scip-python . . . app.py/fastapi/FastAPI#post.'
)
_ROUTER_POST_SYM = (
    'scip-python . . . app.py/fastapi/APIRouter#post.'
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
    """Find the nth occurrence of ``marker`` AS A WORD in ``text``;
    return a SCIP occurrence at that position (0-indexed line/col).

    Word-boundary matching prevents false matches when one keyword is
    a substring of another (e.g., ``post`` inside ``compose``).
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
# Flask classic — @app.route('...', methods=[...])
# ---------------------------------------------------------------------------


class TestFlaskClassicRoute:
    def test_post_route(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.route('/login', methods=['POST'])\n"
            'def login():\n'
            '    return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', _FLASK_ROUTE_SYM),
        ])

        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('POST', '/login') in _query_endpoints(conn, 'myapi')

    def test_default_method_is_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.route('/status')\n"
            'def status(): return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', _FLASK_ROUTE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/status') in _query_endpoints(conn, 'myapi')

    def test_method_tuple_emits_multiple(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.route('/items', methods=['GET', 'POST'])\n"
            'def items(): return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', _FLASK_ROUTE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/items') in rows
        assert ('POST', '/items') in rows


# ---------------------------------------------------------------------------
# Verb decorators — @app.<verb>(...) / @router.<verb>(...)
# ---------------------------------------------------------------------------


class TestVerbDecorators:
    def test_flask_post_short_decorator(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.post('/login')\n"
            'def login(): return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _FLASK_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('POST', '/login') in _query_endpoints(conn, 'myapi')

    def test_fastapi_get(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.get('/users/{id}')\n"
            'def get_user(id: int): return {}\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _FASTAPI_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users/{id}') in _query_endpoints(conn, 'myapi')

    def test_fastapi_router_post(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@router.post('/login')\n"
            'def login(): return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _ROUTER_POST_SYM),
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
    def test_flask_typed_param(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.route('/users/<int:id>')\n"
            'def get_user(id): return {}\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', _FLASK_ROUTE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users/{id}') in _query_endpoints(conn, 'myapi')

    def test_flask_untyped_param(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.route('/users/<id>')\n"
            'def get_user(id): return {}\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', _FLASK_ROUTE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users/{id}') in _query_endpoints(conn, 'myapi')

    def test_fastapi_typed_param(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text = (
            "@app.get('/users/{id:int}')\n"
            'def get_user(id: int): return {}\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _FASTAPI_GET_SYM),
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
            "@app.get('/a')\n"
            'def a(): pass\n'
            '\n'
            "@app.post('/b')\n"
            'def b(): pass\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _FASTAPI_GET_SYM),
            _occ_at(text, 'post', _FASTAPI_POST_SYM),
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
        text_a = (
            "@app.get('/from-a')\n"
            'def a(): pass\n'
        )
        text_b = (
            "@router.post('/from-b')\n"
            'def b(): pass\n'
        )
        a = tmp_path / 'a.py'
        b = tmp_path / 'sub' / 'b.py'
        a.write_text(text_a)
        b.parent.mkdir()
        b.write_text(text_b)

        index = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='a.py',
                    occurrences=(
                        _occ_at(text_a, 'get', _FASTAPI_GET_SYM),
                    ),
                    symbols=(),
                ),
                _ScipDoc(
                    relative_path='sub/b.py',
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
# Adversarial — SCIP filters out non-web symbols, plus other edge cases
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unrelated_decorator_with_get_attr_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``@cache.get('key')`` where ``cache`` is some unrelated
        object (NOT FastAPI) — the SCIP symbol points to
        ``some/other/Cache#get.``, NOT a web framework. Filtered out
        by the symbol-suffix matcher even though the syntax looks
        identical. This is the value-add over pure ast.parse: SCIP
        disambiguates."""
        text = (
            "@cache.get('key')\n"
            'def cached(): return 42\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # SCIP says this symbol is unrelated to web
        index = _make_index(src, tmp_path, [
            _occ_at(
                text, 'get',
                'scip-python . . . app.py/some/Cache#get.',
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()

    def test_decorator_with_no_scip_occurrence_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A decorator with no matching SCIP occurrence at all (e.g.,
        a local function definition) — skip cleanly."""
        text = (
            'def my_dec(f): return f\n'
            '\n'
            "@my_dec\n"
            'def login(): return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # Empty SCIP — no Flask/FastAPI occurrences
        index = _make_index(src, tmp_path, [])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert _query_endpoints(conn, 'myapi') == set()

    def test_malformed_python_doesnt_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Syntax error in one file → that file is skipped, others
        still emit."""
        broken_text = 'def oops(\n  # missing close paren\n'
        good_text = (
            "@app.get('/ok')\n"
            'def ok(): pass\n'
        )
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
                        _occ_at(good_text, 'get', _FASTAPI_GET_SYM),
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

    def test_decorator_with_variable_path_arg_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``@app.get(MY_PATH)`` with a variable arg — skip cleanly.
        Phase 8c will resolve constants via SCIP refs; for now, only
        literal path strings are accepted."""
        text = (
            "MY_PATH = '/login'\n"
            '@app.get(MY_PATH)\n'
            'def login(): pass\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', _FASTAPI_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        # Don't emit — we can't be sure of the path
        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/login') not in rows

    def test_missing_index_file_no_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``index_factory=None`` with no .scip on disk → return 0
        cleanly."""
        rc = ingest_python_routes(
            source_name='myapi',
            source_root=tmp_path,
            conn=conn,
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Re-ingest semantics — same as Akka extractor
# ---------------------------------------------------------------------------


class TestScipIntegrationChallenges:
    """Adversarial tests probing SCIP integration edge cases.

    Each test describes a production-realistic scenario the extractor
    SHOULD handle. Failures here surface real impl gaps that synthetic
    happy-path tests didn't expose.
    """

    def test_realistic_scip_python_symbol_with_full_preamble(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Real scip-python canonical_ids carry a verbose preamble:
        ``scip-python python <repo> <version> <path>/<descriptors>``.
        The suffix-based matcher should still classify them correctly."""
        text = (
            "@app.get('/realistic')\n"
            'def handler(): return {}\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)

        # Realistic full scip-python emission format (representative
        # of what 'scip-python index' actually writes for a Flask app).
        realistic_sym = (
            'scip-python python pypi-fastapi 0.110.0 '
            'fastapi/applications.py/FastAPI#get.'
        )
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'get', realistic_sym),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/realistic') in _query_endpoints(conn, 'myapi')

    def test_decorator_on_class_method(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``class MyView: @app.route('/x') def get(self): ...`` —
        decorators on class methods should still register routes
        (Flask MethodView and FastAPI dependency-injection class
        styles both use this pattern)."""
        text = (
            'class MyView:\n'
            "    @app.route('/users/{id}', methods=['GET'])\n"
            '    def get_user(self, id):\n'
            '        return {}\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', _FLASK_ROUTE_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/users/{id}') in _query_endpoints(conn, 'myapi')

    def test_multi_line_decorator_args(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Decorator args formatted across multiple lines — common in
        FastAPI codebases for readability:

        .. code-block:: python

            @app.post(
                '/items',
                response_model=Item,
                status_code=201,
            )
            def create_item(): ...
        """
        text = (
            '@app.post(\n'
            "    '/items',\n"
            '    response_model=dict,\n'
            ')\n'
            'def create_item(): pass\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'post', _FASTAPI_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('POST', '/items') in _query_endpoints(conn, 'myapi')

    def test_flask_blueprint_route(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``@bp.route('/path')`` where ``bp`` is a Flask Blueprint —
        an extremely common Flask pattern. Symbol resolves to
        ``flask.Blueprint#route``, NOT ``flask.Flask#route``. Matcher
        should still classify it as a route."""
        text = (
            "@bp.route('/blueprint-route')\n"
            'def bp_handler(): return "ok"\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # Real scip-python emits Blueprint#route for Blueprint instances
        bp_sym = 'scip-python . . . app.py/flask/Blueprint#route().'
        index = _make_index(src, tmp_path, [
            _occ_at(text, 'route', bp_sym),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        assert ('GET', '/blueprint-route') in _query_endpoints(
            conn, 'myapi',
        )

    def test_mixed_web_and_unrelated_scip_occurrences(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """File contains BOTH a real ``@app.get(...)`` and an unrelated
        ``cache.get(...)`` call. SCIP has occurrences for both — the
        matcher must filter to web-only and not emit a row for the
        unrelated one."""
        text = (
            "@app.get('/api/items')\n"
            'def list_items():\n'
            "    cached = cache.get('items')\n"
            '    return cached or []\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index(src, tmp_path, [
            # Real FastAPI decorator
            _occ_at(text, 'get', _FASTAPI_GET_SYM, nth=0),
            # Unrelated cache.get() call
            _occ_at(
                text, 'get',
                'scip-python . . . app.py/cachelib/Cache#get.',
                nth=1,
            ),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_endpoints(conn, 'myapi')
        # Real route present
        assert ('GET', '/api/items') in rows
        # Cache call did NOT pollute api_endpoints
        assert ('GET', 'items') not in rows


class TestReIngest:
    def test_re_ingest_replaces_pattern_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        # Pre-existing swagger row
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('s1', 'myapi', 'GET', '/swagger', 'sym', 'swagger'),
        )
        conn.commit()

        text1 = (
            "@app.get('/old')\n"
            'def old(): pass\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text1)
        index1 = _make_index(src, tmp_path, [
            _occ_at(text1, 'get', _FASTAPI_GET_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index1,
        )

        # Replace
        text2 = (
            "@app.post('/new')\n"
            'def new(): pass\n'
        )
        src.write_text(text2)
        index2 = _make_index(src, tmp_path, [
            _occ_at(text2, 'post', _FASTAPI_POST_SYM),
        ])
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index2,
        )

        rows = _query_endpoints(conn, 'myapi')
        assert ('GET', '/old') not in rows
        assert ('POST', '/new') in rows
        assert ('GET', '/swagger') in rows  # swagger preserved
