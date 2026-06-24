"""``safe_ast_parse`` must silence the SyntaxWarning flood from parsing arbitrary
target source (invalid escapes like ``"\\c"``, ``"\\["``) — the noise that
buried ``ariadne index`` output — while still raising real SyntaxErrors.
"""
from __future__ import annotations

import ast
import warnings

import pytest

from ast_utils import safe_ast_parse
from docgen.scip_string_literal_extractor import _extract_python_literals


def test_suppresses_syntaxwarning_from_invalid_escapes() -> None:
    """Target source with invalid escape sequences must parse without leaking a
    SyntaxWarning. Promoting SyntaxWarning to an error makes the assertion
    strict: if any escapes, the parse raises and the test fails."""
    source = 'PATTERN = "\\c\\[\\]"\nPATH = "a\\/b"\n'
    with warnings.catch_warnings():
        warnings.simplefilter('error', SyntaxWarning)
        tree = safe_ast_parse(source, filename='target.py')
    assert isinstance(tree, ast.Module)


def test_still_raises_on_real_syntax_error() -> None:
    """Suppression is scoped to warnings — a genuine SyntaxError still raises."""
    with pytest.raises(SyntaxError):
        safe_ast_parse('def oops(:\n', filename='target.py')


def test_string_literal_extractor_does_not_leak_syntaxwarning() -> None:
    """The string-literal extractor (the "Indexed N string literals" flood
    source) now parses target source through safe_ast_parse, so invalid escapes
    no longer spray SyntaxWarning across ``ariadne index`` output — while the
    literals are still extracted."""
    source = 'BAD = "\\c\\["\nOK = "fine"\n'
    with warnings.catch_warnings():
        warnings.simplefilter('error', SyntaxWarning)
        literals = _extract_python_literals(source)
    assert 'fine' in [value for _line, _col, value in literals]
