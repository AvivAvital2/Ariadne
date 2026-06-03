"""Contract for Phase 2s — sink-site argument resolution.

The resolver answers: "given an arg expression at (file, line, col)
in source ``S``, what literal string does it actually resolve to?"

Returns ``(value, confidence)``:

- ``('literal', value)`` — direct hit on Phase 2p ``string_literals``
- ``('resolved-constant', value)`` — variable identifier whose
  definition's line carries a single literal in ``string_literals``
- ``('config-resolved', value)`` — variable whose def is a call to a
  config getter (``config.getString("k")``); the key is resolved
  against Phase 2q ``config_values``. (Phase 2s.b)
- ``('unresolved', None)`` — couldn't resolve

v1 covered the first two branches; Phase 2s.b adds the third by
consulting ``inspect_definition_rhs`` to classify the RHS at the
def line. Transitive chains (``A = B; B = "..."``) are still out
of scope.

These tests follow the lesson from Phase 2t/2q — stub returns
``(None, 'unresolved')`` always, every test fails behaviorally
rather than on missing imports, and every "should be unresolved"
case is paired with a positive baseline so a stub can't pass it
trivially.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _add_string_literal(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    file: str,
    line: int,
    col: int,
    value: str,
    owning_symbol_id: str | None = None,
) -> None:
    conn.execute(
        '''INSERT INTO string_literals
           (source_name, file, line_start, col_start, value,
            owning_symbol_id)
           VALUES (?, ?, ?, ?, ?, ?)''',
        (source_name, file, line, col, value, owning_symbol_id),
    )
    conn.commit()


def _add_scip_symbol(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    source_name: str,
    file: str,
    line_start: int,
    line_end: int,
    qualified_name: str,
    kind: str = 'Variable',
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
         qualified_name.rsplit('.', 1)[-1],
         qualified_name, None),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# Direct literal at position
# ---------------------------------------------------------------------------


class TestDirectLiteral:
    def test_literal_at_position_resolves(
        self, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_resolution import resolve_arg_value

        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=2, col=19, value='/api/users',
        )
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=2, col=19,
        )
        assert value == '/api/users'
        assert confidence == 'literal'

    def test_no_literal_at_position_returns_unresolved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Paired baseline: a different position WITH a literal
        (positive) and the queried position with no literal
        (negative). Bites a stub that always returns the same
        result."""
        from docgen.scip_resolution import resolve_arg_value

        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=10, col=4, value='hello',
        )
        # Positive — confirms infrastructure works
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=4,
        )
        assert v_pos == 'hello'
        assert c_pos == 'literal'
        # Negative — different position, same fixture
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=99, col=99,
        )
        assert v_neg is None
        assert c_neg == 'unresolved'


# ---------------------------------------------------------------------------
# Variable resolution — identifier → its def line's literal
# ---------------------------------------------------------------------------


class TestVariableResolution:
    def test_variable_resolves_via_scip_symbol(
        self, conn: sqlite3.Connection,
    ) -> None:
        """``URL`` identifier at the call site, defined as a
        module-level constant on line 1 with a literal value."""
        from docgen.scip_resolution import resolve_arg_value

        # Module-level: URL = "/api/users" on line 1
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file='/app.py',
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=1, col=6, value='/api/users',
        )
        # The call passes URL at line 5 (some position in the source)
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=5, col=18, identifier_name='URL',
        )
        assert value == '/api/users'
        assert confidence == 'resolved-constant'

    def test_unknown_identifier_unresolved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Paired — known var resolves; unknown var doesn't.
        Bites a stub (no resolution) AND a too-eager impl that
        returns the file's only literal regardless of name match."""
        from docgen.scip_resolution import resolve_arg_value

        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file='/app.py',
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=1, col=6, value='/api/users',
        )
        # Positive — URL resolves
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=5, col=18, identifier_name='URL',
        )
        assert v_pos == '/api/users'
        assert c_pos == 'resolved-constant'
        # Negative — unknown identifier
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=5, col=18, identifier_name='UNKNOWN',
        )
        assert v_neg is None
        assert c_neg == 'unresolved'

    def test_ambiguous_identifier_unresolved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Two scip_symbols share the trailing identifier name
        ``URL``. The resolver can't safely pick — unresolved.
        Paired with a separately-resolvable identifier (``OTHER``)
        with one match."""
        from docgen.scip_resolution import resolve_arg_value

        # Two URLs in different scopes
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file='/app.py',
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_scip_symbol(
            conn, canonical_id='scip:helpers.URL',
            source_name='myapi', file='/helpers.py',
            line_start=1, line_end=1,
            qualified_name='helpers.URL',
        )
        # Single OTHER for the positive baseline
        _add_scip_symbol(
            conn, canonical_id='scip:app.OTHER',
            source_name='myapi', file='/app.py',
            line_start=2, line_end=2,
            qualified_name='app.OTHER',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=2, col=8, value='/api/other',
        )
        # Negative — two URL definitions, ambiguous
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=20, identifier_name='URL',
        )
        assert v_neg is None
        assert c_neg == 'unresolved'
        # Positive — OTHER has one definition
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=20, identifier_name='OTHER',
        )
        assert v_pos == '/api/other'
        assert c_pos == 'resolved-constant'

    def test_def_with_no_literal_unresolved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """The variable is defined but its def line has NO literal
        in ``string_literals`` (e.g., it was assigned the result
        of a computation). The resolver returns unresolved.
        Paired with a sibling var that does have a literal at its
        def line."""
        from docgen.scip_resolution import resolve_arg_value

        # NO_LITERAL is defined at line 3 but no string_literal there
        _add_scip_symbol(
            conn, canonical_id='scip:app.NO_LITERAL',
            source_name='myapi', file='/app.py',
            line_start=3, line_end=3,
            qualified_name='app.NO_LITERAL',
        )
        # WITH_LITERAL at line 5 with a literal there
        _add_scip_symbol(
            conn, canonical_id='scip:app.WITH_LITERAL',
            source_name='myapi', file='/app.py',
            line_start=5, line_end=5,
            qualified_name='app.WITH_LITERAL',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=5, col=14, value='/api/with-lit',
        )
        # Negative
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=10, identifier_name='NO_LITERAL',
        )
        assert v_neg is None
        assert c_neg == 'unresolved'
        # Positive
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=10, identifier_name='WITH_LITERAL',
        )
        assert v_pos == '/api/with-lit'
        assert c_pos == 'resolved-constant'

    def test_multiple_literals_on_def_line_unresolved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """The def line has more than one literal — resolver can't
        pick safely. Paired with a sibling whose def line has a
        single literal."""
        from docgen.scip_resolution import resolve_arg_value

        # MULTI's def line has two literals — ambiguous
        _add_scip_symbol(
            conn, canonical_id='scip:app.MULTI',
            source_name='myapi', file='/app.py',
            line_start=1, line_end=1,
            qualified_name='app.MULTI',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=1, col=10, value='/api/x',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=1, col=20, value='/api/y',
        )
        # SOLO has one literal on its def line
        _add_scip_symbol(
            conn, canonical_id='scip:app.SOLO',
            source_name='myapi', file='/app.py',
            line_start=2, line_end=2,
            qualified_name='app.SOLO',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=2, col=10, value='/api/solo',
        )
        # Negative
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=10, identifier_name='MULTI',
        )
        assert v_neg is None
        assert c_neg == 'unresolved'
        # Positive
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=10, identifier_name='SOLO',
        )
        assert v_pos == '/api/solo'
        assert c_pos == 'resolved-constant'


# ---------------------------------------------------------------------------
# Resolution priority
# ---------------------------------------------------------------------------


class TestResolutionPriority:
    def test_direct_literal_wins_over_variable(
        self, conn: sqlite3.Connection,
    ) -> None:
        """The position has a literal AND the caller passes
        ``identifier_name``. The direct-literal branch should win
        (more confident than name resolution).

        Paired with the same fixture missing the literal so the
        variable branch fires — same identifier_name, different
        outcome based on whether the literal exists."""
        from docgen.scip_resolution import resolve_arg_value

        # Variable URL defined elsewhere
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file='/app.py',
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=1, col=6, value='/from-var',
        )
        # ALSO a literal at the call's position
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=10, col=20, value='/from-direct',
        )
        # When position has a literal, return it (literal beats var)
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=20, identifier_name='URL',
        )
        assert value == '/from-direct'
        assert confidence == 'literal'
        # Confirm the variable branch DOES work in isolation by
        # querying a position with no direct literal
        v_var, c_var = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=99, col=99, identifier_name='URL',
        )
        assert v_var == '/from-var'
        assert c_var == 'resolved-constant'

    def test_position_with_no_literal_no_identifier_unresolved(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Paired baseline — same fixture, same source — but with
        and without ``identifier_name`` provided. Without it, no
        variable branch fires, so unresolved. With it, resolves."""
        from docgen.scip_resolution import resolve_arg_value

        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file='/app.py',
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/app.py',
            line=1, col=6, value='/api/u',
        )
        # No identifier_name — even though the var exists, can't
        # resolve without knowing what name to look up
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=10,
        )
        assert v_neg is None
        assert c_neg == 'unresolved'
        # WITH identifier_name — resolves
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file='/app.py',
            line=10, col=10, identifier_name='URL',
        )
        assert v_pos == '/api/u'
        assert c_pos == 'resolved-constant'


# ---------------------------------------------------------------------------
# Source isolation — resolver must respect source_name
# ---------------------------------------------------------------------------


class TestSourceIsolation:
    def test_other_source_data_does_not_leak(
        self, conn: sqlite3.Connection,
    ) -> None:
        """A scip_symbol + string_literal from a DIFFERENT source
        with the same identifier name must not be returned. Paired
        with a same-source resolution to confirm the resolver
        actually reads from string_literals when the source matches."""
        from docgen.scip_resolution import resolve_arg_value

        # Other source has URL
        _add_scip_symbol(
            conn, canonical_id='scip:other.URL',
            source_name='other', file='/o.py',
            line_start=1, line_end=1,
            qualified_name='other.URL',
        )
        _add_string_literal(
            conn, source_name='other', file='/o.py',
            line=1, col=6, value='/from-other',
        )
        # My source has its own URL
        _add_scip_symbol(
            conn, canonical_id='scip:mine.URL',
            source_name='myapi', file='/m.py',
            line_start=1, line_end=1,
            qualified_name='mine.URL',
        )
        _add_string_literal(
            conn, source_name='myapi', file='/m.py',
            line=1, col=6, value='/from-mine',
        )
        # Resolving against myapi must return /from-mine, not
        # /from-other (different source)
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file='/m.py',
            line=10, col=10, identifier_name='URL',
        )
        assert value == '/from-mine'
        assert confidence == 'resolved-constant'


# ---------------------------------------------------------------------------
# Phase 2s.b — config-getter resolution via definition inspector
# ---------------------------------------------------------------------------


def _add_config_value(
    conn: sqlite3.Connection,
    *,
    source_name: str,
    file: str,
    key: str,
    value: str,
    line: int = 1,
) -> None:
    conn.execute(
        '''INSERT INTO config_values
           (source_name, file, key, value, line_start)
           VALUES (?, ?, ?, ?, ?)''',
        (source_name, file, key, value, line),
    )
    conn.commit()


class TestConfigGetterResolution:
    """Phase 2s.b — when a variable's def line is a call to a config
    getter (``config.getString("k")`` / ``config["k"]`` / etc.), the
    resolver must look up ``k`` in Phase 2q ``config_values`` and
    return that value with confidence ``'config-resolved'``.

    These tests use real on-disk files via ``tmp_path`` because the
    inspector parses source text. Files that don't exist on disk
    fall back to v1's literal-in-range behavior gracefully.
    """

    def test_python_getter_call_routes_through_config_values(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Positive — ``URL = config.getString("api.url")`` and
        ``api.url`` is in ``config_values``: resolver returns the
        looked-up value with ``'config-resolved'`` confidence."""
        from docgen.scip_resolution import resolve_arg_value

        py_file = tmp_path / 'app.py'
        py_file.write_text(
            'URL = config.getString("api.url")\n',
        )
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file=str(py_file),
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_config_value(
            conn, source_name='myapi',
            file='/conf/reference.conf',
            key='api.url', value='/api/v1/users',
        )
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(py_file),
            line=5, col=10, identifier_name='URL',
        )
        assert value == '/api/v1/users'
        assert confidence == 'config-resolved'

    def test_python_getter_call_unknown_key_unresolved(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Paired — same getter-call RHS but the key isn't in
        ``config_values``. Critically: the literal key string IS in
        ``string_literals`` at line 1 (Phase 2p indexed it), so a
        broken impl that ignores the inspector and returns the
        line's only literal would WRONGLY return ``"missing.key"``
        with ``'resolved-constant'`` confidence — this test bites
        that, asserting the value is None and the confidence is
        ``'unresolved'``."""
        from docgen.scip_resolution import resolve_arg_value

        py_file = tmp_path / 'app.py'
        py_file.write_text(
            'URL = config.getString("missing.key")\n',
        )
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file=str(py_file),
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_config_value(
            conn, source_name='myapi',
            file='/conf/reference.conf',
            key='other.key', value='/api/other',
        )
        # The literal IS in string_literals (Phase 2p indexes it).
        # v1 would have returned ('missing.key', 'resolved-constant').
        # Phase 2s.b must NOT — the inspector says "this is a getter
        # call", so the literal key isn't the variable's value.
        _add_string_literal(
            conn, source_name='myapi', file=str(py_file),
            line=1, col=23, value='missing.key',
        )
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(py_file),
            line=5, col=10, identifier_name='URL',
        )
        assert value is None
        assert confidence == 'unresolved'

    def test_python_subscript_routes_through_config_values(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """``URL = config["api.url"]`` is a getter pattern too.
        Paired with the call-form test above so the inspector's
        subscript branch is exercised independently."""
        from docgen.scip_resolution import resolve_arg_value

        py_file = tmp_path / 'app.py'
        py_file.write_text('URL = config["api.url"]\n')
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file=str(py_file),
            line_start=1, line_end=1,
            qualified_name='app.URL',
        )
        _add_config_value(
            conn, source_name='myapi',
            file='/conf/reference.conf',
            key='api.url', value='/from-cfg-subscript',
        )
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(py_file),
            line=5, col=10, identifier_name='URL',
        )
        assert value == '/from-cfg-subscript'
        assert confidence == 'config-resolved'

    def test_other_rhs_returns_unresolved(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """When the RHS is neither a literal nor a recognized getter
        (``OPAQUE = build_url("/api/inner")``), the resolver returns
        unresolved — NOT the inner literal. Paired with a literal-RHS
        sibling so a stub returning unresolved on every call fails
        the literal half.

        v1 returned the line's only literal regardless of expression
        shape; Phase 2s.b distinguishes 'other' from 'literal'."""
        from docgen.scip_resolution import resolve_arg_value

        py_file = tmp_path / 'app.py'
        py_file.write_text(
            'OPAQUE = build_url("/api/inner")\n'
            'URL = "/api/literal"\n'
        )
        _add_scip_symbol(
            conn, canonical_id='scip:app.OPAQUE',
            source_name='myapi', file=str(py_file),
            line_start=1, line_end=1,
            qualified_name='app.OPAQUE',
        )
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file=str(py_file),
            line_start=2, line_end=2,
            qualified_name='app.URL',
        )
        _add_string_literal(
            conn, source_name='myapi', file=str(py_file),
            line=1, col=19, value='/api/inner',
        )
        _add_string_literal(
            conn, source_name='myapi', file=str(py_file),
            line=2, col=6, value='/api/literal',
        )
        # Negative: opaque function call → unresolved
        v_neg, c_neg = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(py_file),
            line=10, col=10, identifier_name='OPAQUE',
        )
        assert v_neg is None
        assert c_neg == 'unresolved'
        # Positive: real literal RHS → resolved-constant
        v_pos, c_pos = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(py_file),
            line=10, col=10, identifier_name='URL',
        )
        assert v_pos == '/api/literal'
        assert c_pos == 'resolved-constant'

    def test_scala_getter_call_routes_through_config_values(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Same flow for Scala — Typesafe Config ``getString`` is
        the canonical getter. Without a per-language inspector
        dispatch, Python AST would fail on Scala source and the
        resolver would silently fall back to v1 behavior — this
        test bites that."""
        from docgen.scip_resolution import resolve_arg_value

        scala_file = tmp_path / 'App.scala'
        scala_file.write_text(
            'val URL = config.getString("api.url")\n',
        )
        _add_scip_symbol(
            conn, canonical_id='scip:App.URL',
            source_name='myapi', file=str(scala_file),
            line_start=1, line_end=1,
            qualified_name='App.URL',
            language='scala',
        )
        _add_config_value(
            conn, source_name='myapi',
            file='/conf/reference.conf',
            key='api.url', value='/scala-cfg-value',
        )
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(scala_file),
            line=5, col=10, identifier_name='URL',
        )
        assert value == '/scala-cfg-value'
        assert confidence == 'config-resolved'

    def test_javascript_getter_call_routes_through_config_values(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Same flow for JavaScript — the JS inspector dispatch
        path is exercised here independently of Scala/Python."""
        from docgen.scip_resolution import resolve_arg_value

        js_file = tmp_path / 'app.js'
        js_file.write_text(
            "const URL = config.get('api.url');\n",
        )
        _add_scip_symbol(
            conn, canonical_id='scip:app.URL',
            source_name='myapi', file=str(js_file),
            line_start=1, line_end=1,
            qualified_name='app.URL',
            language='javascript',
        )
        _add_config_value(
            conn, source_name='myapi',
            file='/conf/reference.conf',
            key='api.url', value='/js-cfg-value',
        )
        value, confidence = resolve_arg_value(
            conn=conn, source_name='myapi', file=str(js_file),
            line=5, col=10, identifier_name='URL',
        )
        assert value == '/js-cfg-value'
        assert confidence == 'config-resolved'
