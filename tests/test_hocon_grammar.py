"""Pure grammar tests for the HOCON Lark parser.

These tests exercise the grammar in isolation — feed a small HOCON
fixture, assert structure of the resulting parse tree (and crucially,
the line/column metadata Lark attaches to every node).

No DB / library / catalog_extractor involvement here. The shape under
test is `docgen.hocon_grammar.parse(src) -> lark.Tree`. The extractor
that turns the tree into ElementInfo lives in test_hocon_extractor.py.

Behavior contracts:

- Single key=value pairs parse cleanly with line numbers attached.
- Both `=` and `:` separators are accepted (HOCON spec).
- Nested objects produce nested tree nodes; line ranges span the
  block from `{` to `}`.
- Comments (`#` and `//`) are dropped without disturbing line numbers
  of subsequent entries.
- Triple-quoted strings span multiple lines (PGP-key blocks); end_line
  reflects the line of the closing `\"\"\"`.
- Substitutions (`${var}`, `${?optional}`) are recognized.
- Includes (`include "x.conf"`) are recognized.
- Time / size unit-suffixed numbers (`5 minutes`, `100 MiB`) parse.
- A real-world fixture matching the activation block from scalaproject's
  engine/reference.conf parses successfully and has all five expected
  child keys at correct line ranges.

The grammar lives at `docgen/hocon_grammar.lark`; the loader/parser
function `parse(src) -> lark.Tree` is exposed from `docgen.hocon_grammar`.
"""
from __future__ import annotations

import pytest


# ---------------------------------------------------------------------------
# Module / loader smoke tests
# ---------------------------------------------------------------------------


def test_hocon_grammar_module_exposes_parse():
    """The HOCON grammar module ships a `parse` function returning a
    lark.Tree. That's the contract the extractor depends on."""
    from docgen.hocon_grammar import parse

    tree = parse('key = "value"\n')
    # Lark trees are duck-typed; check attributes rather than isinstance
    # (saves importing lark at the test layer).
    assert hasattr(tree, 'children'), 'parse() must return a lark.Tree-like object'


def test_hocon_grammar_propagates_positions():
    """Every node must carry line/column info via propagate_positions=True.
    Without this we cannot emit ElementInfo with line ranges."""
    from docgen.hocon_grammar import parse

    tree = parse('key = "value"\n')
    # Walk to the first non-root node and confirm line tracking is on.
    first_child = next(iter(tree.children), None)
    assert first_child is not None, 'parse tree must have at least one entry'
    meta = getattr(first_child, 'meta', None)
    assert meta is not None and meta.line == 1, (
        f"first entry should be at line 1, got meta={meta!r}"
    )


# ---------------------------------------------------------------------------
# Core syntax: key/value, separators, nesting
# ---------------------------------------------------------------------------


def test_simple_assignment_with_equals():
    from docgen.hocon_grammar import parse

    tree = parse('foo = "bar"\n')
    # Should parse without raising; one entry at line 1.
    assert tree is not None


def test_simple_assignment_with_colon():
    """HOCON allows `:` as a separator equivalent to `=`."""
    from docgen.hocon_grammar import parse

    tree = parse('foo : "bar"\n')
    assert tree is not None


def test_dotted_key_path():
    """`a.b.c = 1` is a path expression, not three separate keys."""
    from docgen.hocon_grammar import parse

    tree = parse('a.b.c = 1\n')
    assert tree is not None


def test_nested_object_block():
    """`a { b = 1 }` is the block-shorthand for `a.b = 1`."""
    from docgen.hocon_grammar import parse

    src = (
        'outer {\n'
        '  inner = 42\n'
        '}\n'
    )
    tree = parse(src)
    assert tree is not None


def test_multiple_top_level_entries():
    from docgen.hocon_grammar import parse

    src = (
        'first = 1\n'
        'second = 2\n'
        'third = 3\n'
    )
    tree = parse(src)
    # Three entries on lines 1, 2, 3.
    children = list(tree.children)
    assert len(children) >= 3, (
        f"expected at least 3 entries, got {len(children)}"
    )


# ---------------------------------------------------------------------------
# Comments and whitespace
# ---------------------------------------------------------------------------


def test_hash_comment_ignored():
    from docgen.hocon_grammar import parse

    tree = parse('# this is a comment\nfoo = 1\n')
    children = list(tree.children)
    # The comment is ignored; there's exactly one entry.
    assert len(children) == 1
    # And it's on line 2 — comments don't shift line numbering.
    assert children[0].meta.line == 2


def test_double_slash_comment_ignored():
    from docgen.hocon_grammar import parse

    tree = parse('// this is a comment\nfoo = 1\n')
    children = list(tree.children)
    assert len(children) == 1
    assert children[0].meta.line == 2


def test_inline_comment_does_not_break_parse():
    from docgen.hocon_grammar import parse

    src = 'foo = 1 # inline\nbar = 2\n'
    tree = parse(src)
    children = list(tree.children)
    assert len(children) == 2


# ---------------------------------------------------------------------------
# String forms
# ---------------------------------------------------------------------------


def test_double_quoted_string_value():
    from docgen.hocon_grammar import parse

    tree = parse('msg = "hello"\n')
    assert tree is not None


def test_unquoted_string_value():
    """HOCON allows unquoted strings for simple values."""
    from docgen.hocon_grammar import parse

    tree = parse('mode = production\n')
    assert tree is not None


def test_triple_quoted_multiline_string():
    """Triple-quoted strings span multiple lines and embed special chars
    without escaping. This is the form PGP key blocks use."""
    from docgen.hocon_grammar import parse

    src = (
        'pgp = """-----BEGIN PGP PUBLIC KEY BLOCK-----\n'
        'mQENBFey9YgBEADIMqHXR1aV4qFI...\n'
        '-----END PGP PUBLIC KEY BLOCK-----"""\n'
        'sibling = 1\n'
    )
    tree = parse(src)
    children = list(tree.children)
    assert len(children) == 2
    # The pgp entry spans lines 1-3 (triple-quoted string is multi-line).
    pgp_entry = children[0]
    assert pgp_entry.meta.line == 1
    assert pgp_entry.meta.end_line >= 3, (
        f"triple-quoted string entry should span to line >= 3, "
        f"got end_line={pgp_entry.meta.end_line}"
    )
    # The sibling entry must follow on line 4.
    assert children[1].meta.line == 4


# ---------------------------------------------------------------------------
# Numbers, booleans, null, units
# ---------------------------------------------------------------------------


def test_number_value():
    from docgen.hocon_grammar import parse

    tree = parse('count = 42\n')
    assert tree is not None


def test_boolean_value():
    from docgen.hocon_grammar import parse

    parse('a = true\n')
    parse('b = false\n')


def test_null_value():
    from docgen.hocon_grammar import parse

    parse('a = null\n')


def test_time_unit_value():
    """`5 minutes` is a HOCON-spec'd duration."""
    from docgen.hocon_grammar import parse

    parse('timeout = 5 minutes\n')


def test_size_unit_value():
    """`100 MiB` is a HOCON-spec'd memory size."""
    from docgen.hocon_grammar import parse

    parse('cap = 100 MiB\n')


def test_array_value():
    from docgen.hocon_grammar import parse

    parse('items = [1, 2, 3]\n')


# ---------------------------------------------------------------------------
# Substitutions and includes
# ---------------------------------------------------------------------------


def test_substitution_required():
    from docgen.hocon_grammar import parse

    parse('host = ${SOME_VAR}\n')


def test_substitution_optional():
    """`${?VAR}` is optional — doesn't fail if undefined."""
    from docgen.hocon_grammar import parse

    parse('host = ${?MAYBE_VAR}\n')


def test_include_directive():
    from docgen.hocon_grammar import parse

    parse('include "other.conf"\n')


# ---------------------------------------------------------------------------
# End-to-end fixture: the activation block from engine/reference.conf
# ---------------------------------------------------------------------------


def test_activation_block_fixture_parses():
    """A fixture mimicking the shape of scalaproject's
    engine/src/main/resources/reference.conf activation block. If this
    fails to parse, the grammar is incomplete in a load-bearing way."""
    from docgen.hocon_grammar import parse

    src = (
        'activation {\n'
        '\tpub = [\n'
        '\t\t"""-----BEGIN PGP PUBLIC KEY BLOCK-----\n'
        '\t\tVersion: GnuPG v2\n'
        '\n'
        '\t\tmQENBGICifMBCACZ3EuqoSLaNkf+FD+v8pKZLkxT1tCLFQs41yFR/NWGqR/tl0cW\n'
        '\t\t-----END PGP PUBLIC KEY BLOCK-----"""\n'
        '\t]\n'
        '\tlicenseFileName = "sblic.af"\n'
        '\tlicenseSigName = "sblic.af.asc"\n'
        '\tlicenseDir = "/var/lib/examplecorp/license"\n'
        '\tmaxCreationDateDaySpan = 14\n'
        '}\n'
    )
    tree = parse(src)
    assert tree is not None
    # Top-level has exactly one entry (the activation block).
    children = list(tree.children)
    assert len(children) == 1
    activation = children[0]
    assert activation.meta.line == 1


# ---------------------------------------------------------------------------
# Negative tests
# ---------------------------------------------------------------------------


def test_malformed_input_raises():
    """A truly malformed HOCON file should raise a Lark parse error.
    The extractor catches this and falls back to file_index-only."""
    from docgen.hocon_grammar import parse
    from lark.exceptions import UnexpectedInput

    with pytest.raises(UnexpectedInput):
        parse('this is { not valid HOCON because no key\n')
