"""Render Graphviz DOT diagrams to PNG via the `dot` binary.

`dot` (graphviz) is an **optional** dependency, needed only here — at the bridge —
to turn a stored DOT diagram into an image for Slack. Generation and validation
are pure-Python and never call it. If `dot` isn't installed, callers should catch
:class:`DotUnavailableError` and degrade gracefully (e.g. post the DOT source as
text instead of an image).

The DOT fence grammar itself lives in :mod:`diagram_format` (shared with the
generator and validator); this module is only the bridge-side render/upload prep.
"""
from __future__ import annotations

import shutil
import subprocess
from typing import Callable, NamedTuple

from diagram_format import DOT_BLOCK_RE

_DOT = 'dot'


class DotUnavailableError(RuntimeError):
    """The Graphviz ``dot`` binary isn't installed/on PATH."""


class DiagramRenderError(RuntimeError):
    """``dot`` ran but failed to render the given source (e.g. invalid DOT)."""


def dot_available() -> bool:
    """True iff the Graphviz ``dot`` binary is on PATH."""
    return shutil.which(_DOT) is not None


def render_dot_to_png(dot_source: str, *, timeout: float = 15.0) -> bytes:
    """Render Graphviz DOT source to PNG bytes.

    Raises :class:`DotUnavailableError` if ``dot`` isn't installed, and
    :class:`DiagramRenderError` if ``dot`` rejects the source or times out.
    """
    if not dot_available():
        raise DotUnavailableError(
            "Graphviz 'dot' is not installed — install graphviz to render "
            'diagrams (e.g. `apt install graphviz`).',
        )
    try:
        proc = subprocess.run(
            [_DOT, '-Tpng'],
            input=dot_source.encode('utf-8'),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as exc:
        raise DiagramRenderError(f'dot timed out after {timeout}s') from exc

    if proc.returncode != 0:
        detail = proc.stderr.decode('utf-8', 'replace').strip()
        raise DiagramRenderError(detail or 'dot exited non-zero')
    return proc.stdout


class PreparedReply(NamedTuple):
    """A reply ready to post: text (diagram blocks handled) + PNGs to upload."""

    text: str
    images: list[bytes]


_DIAGRAM_UNAVAILABLE_NOTE = (
    '⚠️ _Diagram rendering is not set up on this bridge (Graphviz not installed) - '
    'showing the DOT source below; paste it into a Graphviz viewer to see it._\n\n'
)


def prepare_diagrams(
    answer: str,
    *,
    render: Callable[[str], bytes] = render_dot_to_png,
) -> PreparedReply:
    """Turn ```dot blocks in a reply into uploadable PNGs, degrading gracefully.

    On success the block is replaced by a short marker and its PNG is queued for
    upload. If ``dot`` is unavailable the source is kept and a single user-facing
    warning is prepended (and once detected, later blocks skip the render rather
    than re-probing); an invalid block keeps its source with a per-block warning.
    Never a silent drop.
    """
    images: list[bytes] = []
    unavailable = False

    def _replace(match) -> str:
        nonlocal unavailable
        if unavailable:
            return match.group(0)  # dot already known missing — don't re-probe
        dot = match.group(1).strip()
        try:
            images.append(render(dot))
            return '_📊 (diagram rendered below)_'
        except DotUnavailableError:
            unavailable = True
            return match.group(0)
        except DiagramRenderError as exc:
            return f'⚠️ _could not render this diagram ({exc}); DOT source -_\n\n{match.group(0)}'

    text = DOT_BLOCK_RE.sub(_replace, answer or '')
    if unavailable:
        text = _DIAGRAM_UNAVAILABLE_NOTE + text
    return PreparedReply(text=text, images=images)
