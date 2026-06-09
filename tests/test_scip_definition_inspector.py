"""Contract for Phase 2s.b — definition-site RHS inspector.

Given source text, a line number, and a language tag, returns what
kind of expression sits on the RHS of the assignment at that line:

- ``'literal'`` — RHS is a single string literal.
- ``'getter_call'`` — RHS is a call to a known config getter
  (``config.getString("K")``, ``config.get("K")``, ``config["K"]``,
  etc.); ``config_key`` carries the literal first-argument string.
- ``'other'`` — anything else (function call to non-getter, complex
  expression, no assignment at the line).

Each test pairs a positive case (literal OR getter) with a contrast
case in the same fixture so a stub returning ``'other'`` for every
input fails on every test, and an over-broad impl that always
returns ``'getter_call'`` fails on the literal half.

These tests are RED until ``inspect_definition_rhs`` actually
inspects per-language ASTs.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


class TestPython:
    def test_distinguishes_literal_from_method_getter(self) -> None:
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            'lit = "literal_value"\n'
            'got = config.get("KEY_A")\n'
        )
        # Literal half
        r1 = inspect_definition_rhs(
            source_text=text, line=1, language='python',
        )
        assert r1.kind == 'literal'
        # Getter half
        r2 = inspect_definition_rhs(
            source_text=text, line=2, language='python',
        )
        assert r2.kind == 'getter_call'
        assert r2.config_key == 'KEY_A'

    def test_subscript_is_getter(self) -> None:
        """``config["K"]`` is a common Python config-access idiom —
        treated as a getter pattern. Paired with a non-getter call
        so the test bites both stub (always 'other') and over-broad
        impl (always 'getter_call')."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            'sub = config["KEY_B"]\n'
            'val = compute(42)\n'
        )
        r1 = inspect_definition_rhs(
            source_text=text, line=1, language='python',
        )
        assert r1.kind == 'getter_call'
        assert r1.config_key == 'KEY_B'
        # Non-getter call → 'other'
        r2 = inspect_definition_rhs(
            source_text=text, line=2, language='python',
        )
        assert r2.kind == 'other'

    def test_known_getter_method_names(self) -> None:
        """Multiple recognized getter methods. Pin the set so an
        impl that handles only one method (``get``) fails on the
        others."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            'a = config.get("A")\n'
            'b = config.getString("B")\n'
            'c = config.getInt("C")\n'
        )
        for line, key in [(1, 'A'), (2, 'B'), (3, 'C')]:
            r = inspect_definition_rhs(
                source_text=text, line=line, language='python',
            )
            assert r.kind == 'getter_call', (
                f'line {line}: expected getter_call, got {r.kind}'
            )
            assert r.config_key == key


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------


class TestJavaScript:
    def test_distinguishes_literal_from_getter(self) -> None:
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            "const lit = 'literal_value';\n"
            "const got = config.get('KEY_A');\n"
        )
        r1 = inspect_definition_rhs(
            source_text=text, line=1, language='javascript',
        )
        assert r1.kind == 'literal'
        r2 = inspect_definition_rhs(
            source_text=text, line=2, language='javascript',
        )
        assert r2.kind == 'getter_call'
        assert r2.config_key == 'KEY_A'

    def test_subscript_is_getter(self) -> None:
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            "const sub = config['KEY_B'];\n"
            'const computed = compute(42);\n'
        )
        r1 = inspect_definition_rhs(
            source_text=text, line=1, language='javascript',
        )
        assert r1.kind == 'getter_call'
        assert r1.config_key == 'KEY_B'
        r2 = inspect_definition_rhs(
            source_text=text, line=2, language='javascript',
        )
        assert r2.kind == 'other'


# ---------------------------------------------------------------------------
# Scala
# ---------------------------------------------------------------------------


class TestScala:
    def test_distinguishes_val_literal_from_getter(self) -> None:
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            'val lit = "literal_value"\n'
            'val got = config.getString("KEY_A")\n'
        )
        r1 = inspect_definition_rhs(
            source_text=text, line=1, language='scala',
        )
        assert r1.kind == 'literal'
        r2 = inspect_definition_rhs(
            source_text=text, line=2, language='scala',
        )
        assert r2.kind == 'getter_call'
        assert r2.config_key == 'KEY_A'

    def test_typesafe_config_getter_methods(self) -> None:
        """Pin the canonical Typesafe Config method set Scalaproject
        uses: ``getString`` / ``getInt`` / ``getBoolean``."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            'val s = config.getString("S")\n'
            'val i = config.getInt("I")\n'
            'val b = config.getBoolean("B")\n'
        )
        for line, key in [(1, 'S'), (2, 'I'), (3, 'B')]:
            r = inspect_definition_rhs(
                source_text=text, line=line, language='scala',
            )
            assert r.kind == 'getter_call', (
                f'line {line}: expected getter_call, got {r.kind}'
            )
            assert r.config_key == key

    def test_non_getter_call_returns_other(self) -> None:
        """Paired — getter and non-getter at adjacent lines.
        Bites stub (everything 'other') and over-broad impls
        (everything 'getter_call')."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            'val got = config.getString("REAL")\n'
            'val sum = compute(1, 2)\n'
        )
        r1 = inspect_definition_rhs(
            source_text=text, line=1, language='scala',
        )
        assert r1.kind == 'getter_call'
        assert r1.config_key == 'REAL'
        r2 = inspect_definition_rhs(
            source_text=text, line=2, language='scala',
        )
        assert r2.kind == 'other'


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_malformed_source_returns_other(self) -> None:
        """Paired — well-formed source returns the right kind;
        broken source returns 'other'. Without the positive baseline
        a stub passes trivially."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        # Positive baseline
        good = 'x = "value"\n'
        r_good = inspect_definition_rhs(
            source_text=good, line=1, language='python',
        )
        assert r_good.kind == 'literal'
        # Broken
        broken = 'def oops(\n'  # incomplete function def
        r_broken = inspect_definition_rhs(
            source_text=broken, line=1, language='python',
        )
        assert r_broken.kind == 'other'

    def test_line_at_no_assignment_returns_other(self) -> None:
        """Paired — assignment line returns its kind; non-assignment
        line returns 'other'."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = (
            '# top-level comment\n'
            'x = "value"\n'
        )
        r_comment = inspect_definition_rhs(
            source_text=text, line=1, language='python',
        )
        assert r_comment.kind == 'other'
        r_assign = inspect_definition_rhs(
            source_text=text, line=2, language='python',
        )
        assert r_assign.kind == 'literal'

    def test_unknown_language_returns_other(self) -> None:
        """Paired — known language returns its kind; unknown
        returns 'other'. Pins that the dispatch table covers
        exactly the supported languages."""
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
        )

        text = 'x = "value"\n'
        r_known = inspect_definition_rhs(
            source_text=text, line=1, language='python',
        )
        assert r_known.kind == 'literal'
        r_unknown = inspect_definition_rhs(
            source_text=text, line=1, language='cobol',
        )
        assert r_unknown.kind == 'other'


class TestReceiverChains:
    """``getConfig("a").getString("b")`` reconstructs the dotted key
    ``a.b`` across scala/python/js. Split one-branch-per-test so a
    regression isolates: each case exercises a distinct arc of the
    receiver-chain walk (chain hit, deeper nesting, plain-identifier
    receiver, bare-name getter, a non-``getConfig`` call in the chain,
    a ``getConfig`` with a non-literal / empty arg, and a value getter
    whose own arg is dynamic)."""

    def _key(self, text: str, language: str) -> str:
        from docgen.scip_definition_inspector import inspect_definition_rhs

        r = inspect_definition_rhs(
            source_text=text, line=1, language=language,
        )
        assert r.kind == 'getter_call', f'{language}: {text!r} -> {r.kind}'
        return r.config_key

    def _kind(self, text: str, language: str) -> str:
        from docgen.scip_definition_inspector import inspect_definition_rhs

        return inspect_definition_rhs(
            source_text=text, line=1, language=language,
        ).kind

    # ---- Scala ----
    def test_scala_two_level_chain(self) -> None:
        assert self._key(
            'val f = cfg.getConfig("featureflags").getBoolean("enabled")\n',
            'scala',
        ) == 'featureflags.enabled'

    def test_scala_nested_chain(self) -> None:
        assert self._key(
            'val t = cfg.getConfig("svc").getConfig("cache").getInt("ttl")\n',
            'scala',
        ) == 'svc.cache.ttl'

    def test_scala_plain_identifier_receiver_stays_bare(self) -> None:
        assert self._key('val s = cfg.getString("plain")\n', 'scala') == 'plain'

    def test_scala_bare_name_getter_stays_bare(self) -> None:
        # No receiver object at all — callee is a plain identifier.
        assert self._key('val s = getString("plain")\n', 'scala') == 'plain'

    def test_scala_non_getconfig_call_in_chain_adds_no_segment(self) -> None:
        assert self._key(
            'val b = cfg.lookup("a").getString("b")\n', 'scala',
        ) == 'b'

    def test_scala_getconfig_dynamic_arg_adds_no_segment(self) -> None:
        assert self._key(
            'val b = cfg.getConfig(section).getString("b")\n', 'scala',
        ) == 'b'

    def test_scala_value_getter_with_dynamic_arg_is_other(self) -> None:
        assert self._kind('val s = cfg.getString(name)\n', 'scala') == 'other'

    # ---- Python ----
    def test_python_two_level_chain(self) -> None:
        assert self._key(
            'f = cfg.getConfig("featureflags").getBoolean("enabled")\n',
            'python',
        ) == 'featureflags.enabled'

    def test_python_plain_receiver_stays_bare(self) -> None:
        assert self._key('s = cfg.getString("plain")\n', 'python') == 'plain'

    def test_python_value_getter_with_dynamic_arg_is_other(self) -> None:
        assert self._kind('s = cfg.getString(name)\n', 'python') == 'other'

    # ---- JavaScript ----
    def test_js_two_level_chain(self) -> None:
        assert self._key(
            "const f = cfg.getConfig('featureflags').getBoolean('enabled');\n",
            'javascript',
        ) == 'featureflags.enabled'

    def test_js_plain_receiver_stays_bare(self) -> None:
        assert self._key(
            "const s = cfg.getString('plain');\n", 'javascript',
        ) == 'plain'

    def test_js_bare_name_getter_is_other(self) -> None:
        # Bare-name call (identifier callee) is not a recognized getter
        # in JS — only member-expression callees are.
        assert self._kind("const s = getString('plain');\n", 'javascript') == 'other'

    def test_js_non_getconfig_call_in_chain_adds_no_segment(self) -> None:
        assert self._key(
            "const b = cfg.lookup('a').getBoolean('b');\n", 'javascript',
        ) == 'b'

    def test_js_getconfig_dynamic_arg_adds_no_segment(self) -> None:
        assert self._key(
            "const b = cfg.getConfig(section).getBoolean('b');\n", 'javascript',
        ) == 'b'

    def test_js_getconfig_empty_args_adds_no_segment(self) -> None:
        assert self._key(
            "const b = cfg.getConfig().getBoolean('b');\n", 'javascript',
        ) == 'b'

    def test_js_value_getter_with_dynamic_arg_is_other(self) -> None:
        assert self._kind("const s = cfg.getString(name);\n", 'javascript') == 'other'


class TestRhsClassificationArcs:
    """Completes branch coverage of the per-language RHS classifiers
    that the chain feature reworked: non-getter RHS shapes (dynamic
    subscript, template strings, non-expression values) must classify
    as ``other`` / ``literal`` so the getter-call path stays precise."""

    def _kind(self, text: str, language: str) -> str:
        from docgen.scip_definition_inspector import inspect_definition_rhs

        return inspect_definition_rhs(
            source_text=text, line=1, language=language,
        ).kind

    def test_python_non_string_subscript_is_other(self) -> None:
        assert self._kind('x = obj[123]\n', 'python') == 'other'

    def test_python_non_expression_rhs_is_other(self) -> None:
        assert self._kind('x = 1 + 2\n', 'python') == 'other'

    def test_js_plain_template_literal_is_literal(self) -> None:
        assert self._kind('const x = `plain`;\n', 'javascript') == 'literal'

    def test_js_interpolated_template_is_other(self) -> None:
        assert self._kind('const x = `v=${y}`;\n', 'javascript') == 'other'

    def test_js_dynamic_subscript_is_other(self) -> None:
        assert self._kind('const x = config[idx];\n', 'javascript') == 'other'

    def test_js_non_expression_rhs_is_other(self) -> None:
        assert self._kind('const x = 1 + 2;\n', 'javascript') == 'other'

    def test_scala_non_call_non_string_rhs_is_other(self) -> None:
        assert self._kind('val x = 42\n', 'scala') == 'other'


class TestBatchInspection:
    """``inspect_definitions_at_lines`` classifies many lines from a
    SINGLE parse — the linear-time scaffold behind config-read
    extraction. Parse-count assertions pin the performance contract so a
    regression to per-line re-parsing fails the suite; the equivalence
    test pins that it cannot diverge from the single-line API."""

    def _count(self, monkeypatch, attr):
        import docgen.scip_definition_inspector as insp
        calls = {'n': 0}
        if attr == 'SgRoot':
            real = insp.SgRoot

            def counting(text, lang):
                calls['n'] += 1
                return real(text, lang)

            monkeypatch.setattr(insp, 'SgRoot', counting)
        else:
            real = insp.ast.parse

            def counting(*a, **k):
                calls['n'] += 1
                return real(*a, **k)

            monkeypatch.setattr(insp.ast, 'parse', counting)
        return calls

    def test_scala_multiple_lines_from_one_parse(self, monkeypatch) -> None:
        from docgen.scip_definition_inspector import inspect_definitions_at_lines

        calls = self._count(monkeypatch, 'SgRoot')
        text = (
            'val a = cfg.getString("k.one")\n'
            'val b = "plain"\n'
            'val c = cfg.getConfig("grp").getInt("two")\n'
        )
        out = inspect_definitions_at_lines(
            source_text=text, lines=(1, 2, 3), language='scala',
        )
        assert calls['n'] == 1  # one parse for three lines, not three
        assert out[1].kind == 'getter_call' and out[1].config_key == 'k.one'
        assert out[2].kind == 'literal'
        assert out[3].kind == 'getter_call' and out[3].config_key == 'grp.two'

    def test_python_multiple_lines_from_one_parse(self, monkeypatch) -> None:
        from docgen.scip_definition_inspector import inspect_definitions_at_lines

        calls = self._count(monkeypatch, 'ast')
        text = 'a = config.get("A")\nb = "lit"\nc = compute(1)\n'
        out = inspect_definitions_at_lines(
            source_text=text, lines=(1, 2, 3), language='python',
        )
        assert calls['n'] == 1
        assert out[1].kind == 'getter_call' and out[1].config_key == 'A'
        assert out[2].kind == 'literal'
        assert out[3].kind == 'other'

    def test_line_without_assignment_is_omitted(self, monkeypatch) -> None:
        from docgen.scip_definition_inspector import inspect_definitions_at_lines

        text = '# just a comment\nx = config.get("K")\n'
        out = inspect_definitions_at_lines(
            source_text=text, lines=(1, 2), language='python',
        )
        assert 1 not in out  # no assignment on the comment line
        assert out[2].kind == 'getter_call'

    def test_unknown_language_returns_empty_map(self) -> None:
        from docgen.scip_definition_inspector import inspect_definitions_at_lines

        out = inspect_definitions_at_lines(
            source_text='x = "v"\n', lines=(1,), language='cobol',
        )
        assert out == {}

    def test_matches_single_line_inspection(self) -> None:
        # The scaffold must never diverge from the single-line API.
        from docgen.scip_definition_inspector import (
            inspect_definition_rhs,
            inspect_definitions_at_lines,
        )

        text = (
            'val a = cfg.getString("one")\n'
            'val b = "lit"\n'
            'val c = other(1)\n'
            'val d = cfg.getConfig("x").getInt("y")\n'
        )
        lines = range(1, 5)
        batch = inspect_definitions_at_lines(
            source_text=text, lines=lines, language='scala',
        )
        for ln in lines:
            single = inspect_definition_rhs(
                source_text=text, line=ln, language='scala',
            )
            assert batch.get(
                ln, single.__class__(kind='other'),
            ) == single, f'divergence at line {ln}'

    def test_python_annassign_without_initializer_is_other(self) -> None:
        # AnnAssign with no value -> rhs is None -> 'other'.
        from docgen.scip_definition_inspector import inspect_definition_rhs

        r = inspect_definition_rhs(
            source_text='x: int\n', line=1, language='python',
        )
        assert r.kind == 'other'

    def test_js_declaration_without_initializer_is_other(self) -> None:
        # `let x;` -> variable_declarator with no RHS -> 'other'.
        from docgen.scip_definition_inspector import inspect_definition_rhs

        r = inspect_definition_rhs(
            source_text='let x;\n', line=1, language='javascript',
        )
        assert r.kind == 'other'
