"""Fail-closed source resolution.

When the source can't be determined, resolution must RAISE rather than
silently fall back to ``default_source`` — a silent default answers from
the wrong repo (the exact failure mode: a projecta question scoped to
``ariadne``). ``default_source`` no longer auto-applies anywhere, and the
MCP path additionally ignores the process cwd (the server's cwd is the
Ariadne install, not the user's project, so it's meaningless for scope).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from config import Config
from library import Library


def _cfg(tmp_path: Path, body: str) -> Config:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body)
    return Config(p)


def test_undetermined_source_raises_even_with_default_source(
    tmp_path: Path,
) -> None:
    """A configured ``default_source`` must NOT silently scope an
    otherwise-undetermined request. The MCP path (``use_cwd=False``,
    no explicit source) fails closed so the caller resolves the source
    explicitly instead of receiving the wrong repo's docs."""
    from scope_resolution import make_scoped_library

    src = tmp_path / 'mylib'
    src.mkdir()
    cfg = _cfg(tmp_path, f'''\
default_source: mylib
sources:
  mylib:
    path: {src}
''')
    with Library(tmp_path / 'lib.db') as library:
        with pytest.raises(LookupError) as exc:
            make_scoped_library(cfg, library, None, use_cwd=False)
        # The error guides the caller: it lists the configured sources.
        assert 'mylib' in str(exc.value)
