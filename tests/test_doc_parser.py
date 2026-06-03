"""Tests for Scaladoc/Javadoc parsing (SCIP plan, Phase A.2).

The doc parser turns raw `/** ... */` blocks into ``StructuredDoc`` —
fields a downstream prompt or UI can render directly. This module is
pure-string (no protobuf) and shared between Scala and Java.

Tests target the grammar — `@param`/`@return`/`@throws`/`@see`/
`@deprecated`/`@since`, summary-from-first-sentence, link preservation,
malformed-input safety, code-fence handling — plus the Java-specific
`@exception` alias and `{@link}` handling.

These tests must FAIL until ``docgen.doc_parser`` exists.
"""
from __future__ import annotations

from textwrap import dedent

import pytest

# ---------------------------------------------------------------------------
# StructuredDoc shape
# ---------------------------------------------------------------------------


class TestStructuredDoc:
    def test_default_values(self) -> None:
        from docgen.doc_parser import StructuredDoc

        d = StructuredDoc()
        assert d.summary == ''
        assert d.body == ''
        assert d.params == {}
        assert d.returns is None
        assert d.throws == {}
        assert d.see_also == ()
        assert d.deprecated is None
        assert d.since is None

    def test_is_frozen(self) -> None:
        from docgen.doc_parser import StructuredDoc

        d = StructuredDoc()
        with pytest.raises(Exception):  # noqa: B017
            d.summary = 'mutated'  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Scaladoc grammar
# ---------------------------------------------------------------------------


class TestScaladocGrammar:
    def test_summary_is_first_sentence(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(
            'Computes the alpha. The rest of the description follows.'
        )
        assert d.summary == 'Computes the alpha.'

    def test_param_tag(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute the sum.

            @param x The first operand.
            @param y The second operand.
        '''))
        assert d.params == {'x': 'The first operand.', 'y': 'The second operand.'}

    def test_return_tag(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute the sum.

            @return The total of x + y.
        '''))
        assert d.returns == 'The total of x + y.'

    def test_throws_tag(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute.

            @throws IllegalArgumentException If args are negative.
            @throws IOException If reading fails.
        '''))
        assert d.throws == {
            'IllegalArgumentException': 'If args are negative.',
            'IOException': 'If reading fails.',
        }

    def test_see_tag(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute.

            @see [[com.example.Other]]
            @see [[https://example.com docs]]
        '''))
        assert '[[com.example.Other]]' in d.see_also
        assert any('example.com' in s for s in d.see_also)

    def test_deprecated_tag(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Old API.

            @deprecated Use newApi instead. Will be removed in 2.0.
        '''))
        assert d.deprecated == 'Use newApi instead. Will be removed in 2.0.'

    def test_since_tag(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute.

            @since 1.4.0
        '''))
        assert d.since == '1.4.0'

    def test_multi_line_param_description(self) -> None:
        """A `@param` description continues until the next `@tag` or EOF."""
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute.

            @param x The first operand. This is a longer description
              that spans multiple lines and explains the role of x.
            @param y The second operand.
        '''))
        assert 'longer description' in d.params['x']
        assert 'spans multiple lines' in d.params['x']
        assert d.params['y'] == 'The second operand.'

    def test_link_syntax_preserved_in_body(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(
            'References [[com.example.Other]] for details.'
        )
        assert '[[com.example.Other]]' in d.body

    def test_code_fence_preserved_in_body(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        raw = dedent('''\
            Computes:
            {{{
            val x = 1
            val y = 2
            }}}
            and returns x + y.
        ''')
        d = parse_scaladoc(raw)
        assert '{{{' in d.body
        assert 'val x = 1' in d.body
        assert '}}}' in d.body

    def test_body_excludes_tag_lines(self) -> None:
        """Body should be the prose, not contaminated by @tag lines."""
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Computes the sum.

            @param x The first operand.
            @return The sum.
        '''))
        assert 'Computes the sum.' in d.body
        # Tag lines should not appear in the body verbatim.
        assert '@param x' not in d.body
        assert '@return' not in d.body

    def test_malformed_param_tag_does_not_crash(self) -> None:
        """`@param` with no name must not raise — defensive parsing."""
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc('Compute.\n\n@param\n@return result.\n')
        # Whatever we do with the malformed @param, no exception is the contract.
        assert d.returns == 'result.'

    def test_empty_input(self) -> None:
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc('')
        assert d.summary == ''
        assert d.body == ''

    def test_strips_leading_asterisks(self) -> None:
        """Real Scaladoc comments come with `*` line prefixes from the
        `/** ... */` syntax. The parser must strip those.
        """
        from docgen.doc_parser import parse_scaladoc

        raw = dedent('''\
            * Computes the sum.
            *
            * @param x The first operand.
            * @return The total.
        ''')
        d = parse_scaladoc(raw)
        assert d.summary == 'Computes the sum.'
        assert d.params == {'x': 'The first operand.'}
        assert d.returns == 'The total.'


# ---------------------------------------------------------------------------
# Javadoc grammar
# ---------------------------------------------------------------------------


class TestJavadocGrammar:
    def test_param_return_throws(self) -> None:
        from docgen.doc_parser import parse_javadoc

        d = parse_javadoc(dedent('''\
            Compute the sum.

            @param x The first operand.
            @param y The second operand.
            @return The total.
            @throws IllegalArgumentException If args are negative.
        '''))
        assert d.params == {'x': 'The first operand.', 'y': 'The second operand.'}
        assert d.returns == 'The total.'
        assert d.throws == {'IllegalArgumentException': 'If args are negative.'}

    def test_exception_aliases_throws(self) -> None:
        """Javadoc accepts `@exception` as an alias for `@throws`."""
        from docgen.doc_parser import parse_javadoc

        d = parse_javadoc(dedent('''\
            Compute.

            @exception IOException If reading fails.
        '''))
        assert d.throws == {'IOException': 'If reading fails.'}

    def test_inline_link_preserved_in_body(self) -> None:
        """Javadoc uses `{@link Foo}` for references; preserve in body."""
        from docgen.doc_parser import parse_javadoc

        d = parse_javadoc(
            'References {@link com.example.Other} for details.'
        )
        assert '{@link com.example.Other}' in d.body

    def test_summary_is_first_sentence(self) -> None:
        from docgen.doc_parser import parse_javadoc

        d = parse_javadoc(
            'Compute the alpha. Then the beta. Then the gamma.'
        )
        assert d.summary == 'Compute the alpha.'


# ---------------------------------------------------------------------------
# JSDoc grammar — shares the @tag machinery but adds two dialect rules:
#   * `@returns` is an alias for `@return`
#   * `@param`/`@returns`/`@throws` may carry a leading `{type}` annotation
#     that must be stripped (params/returns) or used as the key (throws)
# ---------------------------------------------------------------------------


class TestJsdocGrammar:
    def test_summary_is_first_sentence(self) -> None:
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc('Logs the user in. Everything after is body.')
        assert d.summary == 'Logs the user in.'

    def test_param_with_brace_type_strips_type(self) -> None:
        """`@param {string} name desc` → name→desc, the {type} is dropped
        (StructuredDoc.params is name→description)."""
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc(dedent('''\
            Build a greeting.

            @param {string} name The person to greet.
            @param {number} times How many times.
        '''))
        assert d.params == {
            'name': 'The person to greet.',
            'times': 'How many times.',
        }

    def test_param_without_brace_type(self) -> None:
        """Untyped `@param name desc` still works (type is optional)."""
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc(dedent('''\
            Build a greeting.

            @param name The person to greet.
        '''))
        assert d.params == {'name': 'The person to greet.'}

    def test_returns_alias_and_brace_type(self) -> None:
        """JSDoc uses `@returns` (with an s); the {type} is stripped."""
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc(dedent('''\
            Fetch the user.

            @returns {Promise<User>} The resolved user.
        '''))
        assert d.returns == 'The resolved user.'

    def test_return_singular_also_supported(self) -> None:
        """`@return` (Javadoc spelling) is accepted too."""
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc(dedent('''\
            Add two numbers.

            @return The sum.
        '''))
        assert d.returns == 'The sum.'

    def test_throws_brace_type_is_the_key(self) -> None:
        """`@throws {RangeError} desc` → the braced type is the throws key."""
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc(dedent('''\
            Parse a port number.

            @throws {RangeError} If the value is out of range.
        '''))
        assert d.throws == {'RangeError': 'If the value is out of range.'}

    def test_deprecated_since_see(self) -> None:
        from docgen.doc_parser import parse_jsdoc

        d = parse_jsdoc(dedent('''\
            Old helper.

            @deprecated Use newHelper instead.
            @since 2.1.0
            @see https://example.com/docs
        '''))
        assert d.deprecated == 'Use newHelper instead.'
        assert d.since == '2.1.0'
        assert any('example.com' in s for s in d.see_also)


# ---------------------------------------------------------------------------
# Round-trip — JSON-serializable output
# ---------------------------------------------------------------------------


class TestStructuredDocRoundtrip:
    def test_asdict_is_json_serializable(self) -> None:
        """The whole point of structuring the doc is to flow it through
        ``ElementInfo.documentation`` (a dict) and onto a prompt — that
        requires ``json.dumps(asdict(structured_doc))`` to succeed.
        """
        import json

        from attrs import asdict

        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc(dedent('''\
            Compute.

            @param x First.
            @return Total.
            @throws E If bad.
            @see [[com.example.Other]]
            @deprecated Use foo.
            @since 1.0
        '''))
        body = json.dumps(asdict(d))
        # Round-trip through JSON to prove no non-serializable types snuck in.
        loaded = json.loads(body)
        assert loaded['params'] == {'x': 'First.'}
        assert loaded['returns'] == 'Total.'
        assert loaded['throws'] == {'E': 'If bad.'}
        assert loaded['deprecated'] == 'Use foo.'
        assert loaded['since'] == '1.0'


# ---------------------------------------------------------------------------
# Adversarial — bugs that would still pass weaker tests
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_tag_inside_code_fence_is_not_parsed_as_a_tag(self) -> None:
        """A `@param` inside a `{{{ ... }}}` code block is part of the
        example, not a real tag. A naive line-based parser will eat it
        as a real `@param` — that's the bug to catch.
        """
        from docgen.doc_parser import parse_scaladoc

        raw = dedent('''\
            Demonstrate scaladoc.

            {{{
            @param fakey not a real tag
            }}}

            @param real The actual parameter.
        ''')
        d = parse_scaladoc(raw)
        # The fake @param must not be in the params dict.
        assert 'fakey' not in d.params
        # The real one is.
        assert d.params == {'real': 'The actual parameter.'}

    def test_return_with_leading_tag_text_does_not_swallow_summary(self) -> None:
        """If the doc starts directly with `@return`, summary should be empty
        (not eat the @return content).
        """
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc('@return The total.')
        assert d.summary == ''
        assert d.returns == 'The total.'

    def test_param_without_description_is_empty_string(self) -> None:
        """`@param x` with no description must yield an entry, not crash."""
        from docgen.doc_parser import parse_scaladoc

        d = parse_scaladoc('@param x\n@param y A description.')
        # x is recorded with empty description.
        assert 'x' in d.params
        assert d.params['x'] == ''
        assert d.params['y'] == 'A description.'
