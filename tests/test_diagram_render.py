from __future__ import annotations

import subprocess

import pytest

import slack_bridge.diagram as diagram
from slack_bridge.diagram import (
    DiagramRenderError,
    DotUnavailableError,
    dot_available,
    render_dot_to_png,
)

_PNG_MAGIC = b'\x89PNG\r\n\x1a\n'


def test_renders_valid_dot_to_png_bytes() -> None:
    png = render_dot_to_png('digraph G { source -> render -> upload }')
    assert png[:8] == _PNG_MAGIC
    assert len(png) > 100


def test_invalid_dot_raises_render_error() -> None:
    with pytest.raises(DiagramRenderError):
        render_dot_to_png('this is not valid dot at all')


def test_missing_dot_binary_is_detected_and_degrades(monkeypatch) -> None:
    """`dot` is a SOFT dependency: when absent, detection reports it and the
    renderer raises a clear DotUnavailableError so the bridge can fall back."""
    monkeypatch.setattr(diagram.shutil, 'which', lambda _name: None)
    assert dot_available() is False
    with pytest.raises(DotUnavailableError):
        render_dot_to_png('digraph G { a -> b }')


def test_dot_timeout_is_surfaced_as_render_error(monkeypatch) -> None:
    """A `dot` that hangs is surfaced as a render error, never left to block."""
    def _timeout(*_a, **_k):
        raise subprocess.TimeoutExpired(cmd='dot', timeout=0.01)

    monkeypatch.setattr(diagram.subprocess, 'run', _timeout)
    with pytest.raises(DiagramRenderError):
        render_dot_to_png('digraph G { a -> b }')
