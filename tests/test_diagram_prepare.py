from __future__ import annotations

from slack_bridge.diagram import (
    DiagramRenderError,
    DotUnavailableError,
    prepare_diagrams,
)


def test_renders_to_image_and_strips_the_source() -> None:
    """On success the DOT source is replaced by a marker and the PNG is queued
    for upload — the user sees the image, not raw code."""
    out = prepare_diagrams('Flow:\n\n```dot\ndigraph G { a -> b }\n```\n', render=lambda _d: b'PNGBYTES')
    assert out.images == [b'PNGBYTES']
    assert '```dot' not in out.text
    assert 'diagram' in out.text.lower()


def test_two_blocks_yield_two_images() -> None:
    text = '```dot\ndigraph A {}\n```\n```dot\ndigraph B {}\n```'
    out = prepare_diagrams(text, render=lambda _d: b'IMG')
    assert out.images == [b'IMG', b'IMG']


def test_dot_unavailable_warns_in_reply_and_keeps_source() -> None:
    """No `dot` → a user-facing warning in the reply (not just a server log) and
    the DOT source is kept so they can render it elsewhere."""
    def _missing(_d):
        raise DotUnavailableError('no dot')

    out = prepare_diagrams('```dot\ndigraph G { a -> b }\n```', render=_missing)
    assert out.images == []
    assert '```dot' in out.text                      # source preserved
    assert 'graphviz' in out.text.lower()            # explains why, in the reply


def test_render_error_warns_in_reply_and_keeps_source() -> None:
    def _bad(_d):
        raise DiagramRenderError('syntax error')

    out = prepare_diagrams('```dot\nnot valid\n```', render=_bad)
    assert out.images == []
    assert '```dot' in out.text
    assert 'render' in out.text.lower()


def test_no_diagrams_passes_through_untouched() -> None:
    out = prepare_diagrams('just prose, no diagram', render=lambda _d: b'X')
    assert out.text == 'just prose, no diagram'
    assert out.images == []
