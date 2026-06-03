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
