from __future__ import annotations

from diagram_format import extract_dot_blocks, fence_dot, split_description_and_dot


def test_fence_extract_round_trip() -> None:
    """The single contract: what fence_dot writes, extract_dot_blocks reads back."""
    dot = 'digraph G { a -> b }'
    assert extract_dot_blocks(fence_dot(dot)) == [dot]


def test_extract_all_blocks_in_order() -> None:
    text = 'x\n```dot\ndigraph A {}\n```\ny\n```graphviz\ndigraph B {}\n```\n'
    assert extract_dot_blocks(text) == ['digraph A {}', 'digraph B {}']


def test_extract_none() -> None:
    assert extract_dot_blocks('no diagrams') == []
    assert extract_dot_blocks('') == []


def test_split_description_and_dot() -> None:
    content = 'How it flows.\n\n```dot\ndigraph G { a -> b }\n```\n'
    description, dot = split_description_and_dot(content)
    assert description == 'How it flows.'
    assert dot == 'digraph G { a -> b }'


def test_split_no_dot_returns_empty_dot() -> None:
    description, dot = split_description_and_dot('just prose, no graph')
    assert description == 'just prose, no graph'
    assert dot == ''
