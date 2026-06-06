"""The 'diagram' doc type now carries Graphviz DOT, not Mermaid.

DOT renders to PNG with a tiny native binary (`dot`); Mermaid would need headless
Chromium. So the validator must require a ```dot block containing a real graph,
and no longer accept Mermaid as a diagram.
"""
from __future__ import annotations

from docgen.validator import ContentValidator


def _issues(content: str):
    return ContentValidator().validate(content, doc_type='diagram', title='Export Flow').issues


def test_valid_dot_diagram_has_no_missing_or_invalid_diagram_error() -> None:
    content = (
        '# Export Flow\n\nThe export pipeline, end to end:\n\n'
        '```dot\n'
        'digraph export {\n'
        '  rankdir=LR;\n'
        '  source -> render -> upload;\n'
        '}\n'
        '```\n'
    )
    codes = {i.code for i in _issues(content)}
    assert 'NO_DIAGRAM' not in codes
    assert 'INVALID_DIAGRAM' not in codes
    assert 'NO_MERMAID' not in codes   # the old Mermaid-only check must be gone


def test_diagram_without_a_dot_block_is_flagged() -> None:
    content = '# Export Flow\n\nProse describing the flow, but no graph block at all.\n'
    issues = _issues(content)
    assert any(i.level == 'error' and i.code == 'NO_DIAGRAM' for i in issues)


def test_leftover_mermaid_block_does_not_satisfy_a_diagram() -> None:
    # Diagrams migrated to DOT; a stray Mermaid block is no longer a valid diagram.
    content = '# Export Flow\n\n```mermaid\nflowchart TD\n  A --> B\n```\n'
    issues = _issues(content)
    assert any(i.code == 'NO_DIAGRAM' for i in issues)


def test_dot_block_without_a_graph_keyword_is_invalid() -> None:
    content = '# Export Flow\n\n```dot\njust some text, not a graph\n```\n'
    issues = _issues(content)
    assert any(i.code == 'INVALID_DIAGRAM' for i in issues)


def test_dot_with_unbalanced_braces_warns() -> None:
    """A DOT block that lost its closing brace (truncated output) is warned on."""
    content = '# Export Flow\n\n```dot\ndigraph G {\n  source -> sink;\n```\n'  # no }
    assert any(i.code == 'UNBALANCED_BRACKETS' for i in _issues(content))


def test_dot_with_a_very_long_line_warns() -> None:
    """A >200-char line usually means truncated/garbled DOT — warn on it."""
    long_chain = ' -> '.join(f'node_{i}' for i in range(60))  # well over 200 chars
    content = f'# Export Flow\n\n```dot\ndigraph G {{\n  {long_chain};\n}}\n```\n'
    assert any(i.code == 'LONG_DIAGRAM_LINE' for i in _issues(content))
