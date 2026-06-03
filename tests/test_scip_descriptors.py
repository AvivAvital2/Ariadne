"""Tests for SCIP descriptor parsing (SCIP plan, Phase A.1).

Pure-Python unit tests for the symbol-string parser. No protobuf dependency.
These tests target plausible bugs in the regex grammar, JVM-type decoding,
and qualified-name composition — all the parts where SCIP semantics meet
Ariadne's qualified_name contract.

Tests must FAIL until ``docgen.scip_descriptors`` exists.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# _parse_descriptors — raw descriptor string → list of (name, kind, disambig)
# ---------------------------------------------------------------------------


class TestParseDescriptors:
    def test_class_descriptor(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('com/example/Foo#')
        assert out == [
            ('com', 'package', ''),
            ('example', 'package', ''),
            ('Foo', 'type', ''),
        ]

    def test_method_descriptor_with_disambiguator(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('com/example/Foo#bar().')
        assert out[-1] == ('bar', 'method', '')
        assert out[-2] == ('Foo', 'type', '')

    def test_method_with_jvm_disambiguator(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('com/example/Foo#bar(I).')
        # The disambiguator string is preserved untransformed at the parse
        # stage; decoding to "int" happens in _decode_java_descriptor.
        last = out[-1]
        assert last[0] == 'bar'
        assert last[1] == 'method'
        assert last[2] == 'I'

    def test_term_descriptor_for_companion_object(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('com/example/Foo.')
        assert out[-1] == ('Foo', 'term', '')

    def test_typealias_descriptor(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('com/example/MyAlias:')
        assert out[-1] == ('MyAlias', 'typealias', '')

    def test_type_parameter_is_skipped(self) -> None:
        """`[T]` is a type parameter, not part of the qualified name."""
        from docgen.scip_descriptors import _parse_descriptors

        # `Container#[T]bar().` — the [T] should be skipped, not appear in output.
        out = _parse_descriptors('com/example/Container#[T]bar().')
        names = [name for name, _, _ in out]
        assert 'T' not in names
        assert 'bar' in names

    def test_backtick_escaped_name(self) -> None:
        """`<init>` is the canonical SCIP encoding of a constructor."""
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('com/example/Foo#`<init>`().')
        # The backtick-escaped name preserves the inner text as the descriptor name.
        assert out[-1][0] == '<init>'
        assert out[-1][1] == 'method'

    def test_empty_string_returns_empty(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        assert _parse_descriptors('') == []


# ---------------------------------------------------------------------------
# Parameter descriptors — scip-python emits one SymbolInformation per
# parameter of a method, with descriptors like ``Foo#bar().(self)``.
#
# Evolutionary-TDD walk: the parser must recognize these so that the
# parameter's qualified_name is distinct from its enclosing method's.
# Before this work, all three of (method, self-param, other-param)
# collapsed to the same parsed structure → same qualified_name → the
# graph resolver reported "ambiguous" with N identical-looking
# candidates that the user couldn't tell apart.
# ---------------------------------------------------------------------------


class TestParameterDescriptors:
    # ---- T1 -----------------------------------------------------------
    # The smallest demand: a parameter descriptor ``(self)`` after a
    # method's ``().`` is recognized and emitted as a ``parameter`` kind
    # entry with the parameter's name.
    def test_t1_method_with_one_parameter_is_parsed(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('Foo#bar().(self)')
        # The method itself is still parsed as before.
        assert ('Foo', 'type', '') in out
        assert ('bar', 'method', '') in out
        # The parameter is now also a parse result with a 'parameter'
        # kind. Before this fix the parser bailed at '(' after ')' and
        # silently dropped the param, leaving the parse result identical
        # to the method's own.
        assert ('self', 'parameter', '') in out

    # ---- T2 -----------------------------------------------------------
    # The qualified name composed from the parsed result must include
    # the parameter as a trailing segment. That's what makes the
    # method's QN distinct from each parameter's QN — the parameter's
    # QN is longer (parent_method.param) and so doesn't collide.
    def test_t2_parameter_extends_qualified_name(self) -> None:
        from docgen.scip_descriptors import _qualified_name_from_symbol

        method_sym = (
            'scip-python python myproject 0.1 mod/Klass#frob().'
        )
        param_sym = (
            'scip-python python myproject 0.1 mod/Klass#frob().(arg)'
        )

        method_qn, method_parent = _qualified_name_from_symbol(
            method_sym, 'python',
        )
        param_qn, param_parent = _qualified_name_from_symbol(
            param_sym, 'python',
        )

        # The method's QN is unchanged from before the parameter fix.
        assert method_qn == 'mod.Klass.frob'
        assert method_parent == 'mod.Klass'

        # The parameter's QN extends the method's by one segment, so
        # the two are now distinct.
        assert param_qn == 'mod.Klass.frob.arg'
        assert param_qn != method_qn
        # The parameter's parent is its enclosing method (one fewer
        # segment than the parameter's own QN).
        assert param_parent == 'mod.Klass.frob'

        # T1 still holds — both symbols parse correctly.
        from docgen.scip_descriptors import _parse_descriptors
        method_desc = _parse_descriptors('mod/Klass#frob().')
        param_desc = _parse_descriptors('mod/Klass#frob().(arg)')
        assert ('frob', 'method', '') in method_desc
        assert ('arg', 'parameter', '') in param_desc

    # ---- T2.5 (regression) --------------------------------------------
    # Earlier the parser would silently bail if any descriptor followed
    # a parameter — ``foo().(self).bar().`` would return only
    # ``[(foo, method, ''), (self, parameter, '')]``, dropping bar.
    # The fix lets the loop skip a single ``.`` separator after a
    # parameter so subsequent descriptors continue to parse.
    def test_t2_5_parser_continues_past_parameter(self) -> None:
        from docgen.scip_descriptors import _parse_descriptors

        out = _parse_descriptors('foo().(self).bar().')
        names = [(name, kind) for name, kind, _ in out]
        assert ('foo', 'method') in names
        assert ('self', 'parameter') in names
        assert ('bar', 'method') in names, (
            f'parser bailed before bar — got {names}'
        )

    # ---- T3 -----------------------------------------------------------
    # End-to-end at the resolver layer: a graph populated with a method
    # AND its parameter symbols returns a UNIQUE match for the method's
    # qualified name. Before this fix, all three (method + self-param +
    # other-param) had the same QN, so the resolver reported "3
    # candidates" indistinguishable from each other.
    def test_t3_resolver_returns_unique_method_when_params_present(
        self,
    ) -> None:
        from docgen.scip_cross_source import (
            CrossSourceGraph, CrossSourceSymbol,
        )
        from docgen.scip_descriptors import _qualified_name_from_symbol

        graph = CrossSourceGraph()
        # Three SCIP symbols for the same method: the method itself
        # and two parameter symbols. scip-python emits all three.
        canonical_ids = [
            'scip-python python myproject 0.1 mod/Klass#frob().',
            'scip-python python myproject 0.1 mod/Klass#frob().(self)',
            'scip-python python myproject 0.1 mod/Klass#frob().(arg)',
        ]
        for cid in canonical_ids:
            qn, parent = _qualified_name_from_symbol(cid, 'python')
            graph._symbols[cid] = CrossSourceSymbol(
                canonical_id=cid,
                source_name='myproject',
                language='python',
                file='mod.py',
                line_start=10,
                line_end=20,
                kind='Method',
                display_name='frob',
                qualified_name=qn,
                parent_qualified_name=parent,
            )

        # The resolver should return a UNIQUE match for the method.
        result = graph.resolve_symbol('Klass.frob')
        assert result.symbol is not None, (
            'expected unique match, got candidates: '
            f'{[c.qualified_name for c in result.candidates]}'
        )
        assert result.symbol.canonical_id.endswith('frob().')
        # Tier should be 'suffix' since 'Klass.frob' is a suffix of
        # 'mod.Klass.frob' but isn't a complete qualified_name.
        assert result.match_tier == 'suffix'

        # And explicitly querying for the parameter's QN finds only
        # that parameter — the resolver tiers are clean.
        param_result = graph.resolve_symbol('Klass.frob.arg')
        assert param_result.symbol is not None
        assert param_result.symbol.canonical_id.endswith('(arg)')


# ---------------------------------------------------------------------------
# _decode_java_descriptor — JVM type encoding → human dotted form
# ---------------------------------------------------------------------------


class TestDecodeJavaDescriptor:
    def test_void(self) -> None:
        from docgen.scip_descriptors import _decode_java_descriptor

        assert _decode_java_descriptor('V') == '(void)'

    def test_primitives(self) -> None:
        from docgen.scip_descriptors import _decode_java_descriptor

        assert _decode_java_descriptor('I') == '(int)'
        assert _decode_java_descriptor('J') == '(long)'
        assert _decode_java_descriptor('D') == '(double)'
        assert _decode_java_descriptor('F') == '(float)'
        assert _decode_java_descriptor('Z') == '(boolean)'
        assert _decode_java_descriptor('B') == '(byte)'
        assert _decode_java_descriptor('S') == '(short)'
        assert _decode_java_descriptor('C') == '(char)'

    def test_object_descriptor(self) -> None:
        from docgen.scip_descriptors import _decode_java_descriptor

        assert _decode_java_descriptor('Ljava/lang/String;') == '(java.lang.String)'

    def test_array_descriptor(self) -> None:
        from docgen.scip_descriptors import _decode_java_descriptor

        assert _decode_java_descriptor('[I') == '(int[])'
        assert _decode_java_descriptor('[[I') == '(int[][])'
        # Array of objects.
        assert _decode_java_descriptor(
            '[Ljava/lang/String;',
        ) == '(java.lang.String[])'

    def test_multi_argument_descriptor(self) -> None:
        from docgen.scip_descriptors import _decode_java_descriptor

        # Foo.bar(int, java.lang.String, long)
        assert _decode_java_descriptor('ILjava/lang/String;J') == (
            '(int,java.lang.String,long)'
        )

    def test_array_of_object_followed_by_map(self) -> None:
        """Mixed: int[], java.util.Map (object after array)."""
        from docgen.scip_descriptors import _decode_java_descriptor

        result = _decode_java_descriptor('[ILjava/util/Map;')
        assert result == '(int[],java.util.Map)'

    def test_empty_descriptor_yields_empty_paren(self) -> None:
        from docgen.scip_descriptors import _decode_java_descriptor

        assert _decode_java_descriptor('') == '()'


# ---------------------------------------------------------------------------
# _qualified_name_from_symbol
# ---------------------------------------------------------------------------


class TestQualifiedNameFromSymbol:
    def test_class_symbol_scala(self) -> None:
        """Class-only symbol → qn is package + class, parent is package."""
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#',
            language='scala',
        )
        assert qn == 'com.example.Foo'
        assert parent == 'com.example'

    def test_class_symbol_java(self) -> None:
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#',
            language='java',
        )
        assert qn == 'com.example.Foo'
        assert parent == 'com.example'

    def test_method_symbol_scala_no_overload_decoding(self) -> None:
        """Scala does not decode the JVM disambiguator into the qualified name."""
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#bar(I).',
            language='scala',
        )
        # Plan §2: Java overloads include (params); Scala does not.
        assert qn == 'com.example.Foo.bar'
        assert parent == 'com.example.Foo'

    def test_method_symbol_java_with_overload_decoding(self) -> None:
        """Java methods carry the decoded JVM signature for overload disambiguation."""
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#bar(I).',
            language='java',
        )
        assert qn == 'com.example.Foo.bar(int)'
        assert parent == 'com.example.Foo'

    def test_overloads_yield_distinct_qns(self) -> None:
        """Two Java methods with the same name but different params must
        produce different qualified_names — that's the whole point of
        decoding the disambiguator.
        """
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn_a, _ = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#bar(I).',
            language='java',
        )
        qn_b, _ = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#bar(Ljava/lang/String;).',
            language='java',
        )
        assert qn_a != qn_b

    def test_local_symbol_returns_as_is(self) -> None:
        """`local <id>` symbols are intra-document; no qualified name."""
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol('local 0', language='scala')
        assert qn == 'local 0'
        assert parent is None

    def test_nested_class_qn(self) -> None:
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Outer#Inner#',
            language='scala',
        )
        assert qn == 'com.example.Outer.Inner'
        assert parent == 'com.example.Outer'

    def test_package_only_symbol(self) -> None:
        """A symbol that's just a package — no `Foo#` etc — has no parent."""
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/',
            language='scala',
        )
        assert qn == 'com.example'
        assert parent is None

    def test_companion_class_and_object_collapse_in_scala(self) -> None:
        """Plan §2.1: class Foo and companion object Foo$ both map to
        qualified_name = 'com.example.Foo' (disambiguated by subtype, which
        is the caller's responsibility, not the parser's).
        """
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn_class, _ = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo#',
            language='scala',
        )
        qn_object, _ = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 com/example/Foo.',
            language='scala',
        )
        # Both map to the same qualified_name; the type/term distinction
        # carries onto subtype but not into qn.
        assert qn_class == qn_object == 'com.example.Foo'


# ---------------------------------------------------------------------------
# Adversarial — these target plausible bugs that would otherwise pass
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_double_decoding_does_not_explode_array_brackets(self) -> None:
        """A bug like ``"[I" → "[(int)"`` (extra paren) would silently
        corrupt every Java array signature. Pin the exact bracket shape.
        """
        from docgen.scip_descriptors import _decode_java_descriptor

        # Both arrays should look canonical, no stray characters.
        result = _decode_java_descriptor('[Ljava/lang/Object;')
        assert '(' + 'java.lang.Object[]' + ')' == result, (
            f'unexpected shape: {result!r}'
        )

    def test_disambiguator_with_unknown_primitive_letter(self) -> None:
        """Defensive: an unknown letter should not crash; preserve as-is."""
        from docgen.scip_descriptors import _decode_java_descriptor

        # Q is not a valid JVM primitive marker. Don't crash.
        result = _decode_java_descriptor('Q')
        # We accept either "(Q)" preserve or "(unknown)" — the contract is
        # "no exception". A KeyError would be a real bug.
        assert isinstance(result, str)
        assert result.startswith('(') and result.endswith(')')

    def test_qualified_name_with_no_descriptors_falls_back_to_symbol(self) -> None:
        """A symbol the parser can't decode shouldn't crash — it should
        return the raw symbol so the caller can still log it sensibly.
        """
        from docgen.scip_descriptors import _qualified_name_from_symbol

        qn, parent = _qualified_name_from_symbol(
            'scip-java maven org.example my-lib 1.0 ', language='scala',
        )
        # Either the raw input or an empty string is acceptable — the
        # contract is "no exception, no None for qn".
        assert isinstance(qn, str)
        assert parent is None
