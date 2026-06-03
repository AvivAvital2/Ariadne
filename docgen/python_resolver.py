"""Python interpreter resolution for SCIP indexing (Phase 2n / Layer A).

`PythonIndexerAdapter` previously required ``env_hints['python_path']``
from the user. That contradicts Ariadne's "informs the unknowledgeable
user" purpose — projects like scalaproject deploy multiple Python envs via
Dockerfile and reference them from HOCON config; the user can't be
expected to map those dots before running indexing.

This module replaces that precondition with auto-resolution per cwd.
The probe order, highest priority first:

1. ``env_hints['python_path']`` — explicit user override; only honored
   if the file exists AND is executable. Garbage in (wrong path saved
   in config) silently falls through rather than failing the run.
2. ``<cwd>/.venv/bin/python`` — modern convention used by uv, poetry,
   and recent manual workflows.
3. ``<cwd>/venv/bin/python`` — older manual convention.
4. Walk up to ancestor's ``.venv`` or ``venv`` (bounded depth, default
   5 levels) — services the monorepo case where one root venv is
   shared across multiple sub-package cwds.
5. ``shutil.which('python3')`` → ``shutil.which('python')`` — system
   fallback. Last-resort, but guarantees we don't surprise users with
   a hard failure when *some* python is on PATH.
6. Unresolved — return a result with ``source='unresolved'`` and a
   user-actionable error_message. The caller (the adapter) is
   responsible for surfacing this rather than crashing.

Each resolution carries a ``source`` provenance label
(``'env_hints'`` / ``'cwd-venv'`` / ``'walk-up'`` / ``'system'`` /
``'unresolved'``) so the adapter can log what was picked and the user
can spot a wrong choice without reading code.

The ``which`` callable is dependency-injected (default
``shutil.which``) so tests can stub system-python lookup
OS-independently — that pattern matches the rest of the SCIP
indexer-adapter family.
"""
from __future__ import annotations

import os
import shutil
from collections.abc import Callable
from pathlib import Path

from attrs import frozen


@frozen
class InterpreterResolution:
    """Result of resolving a Python interpreter for SCIP indexing.

    ``path`` is the resolved interpreter (or ``None`` when
    ``source='unresolved'``). ``source`` is the provenance label so
    callers can log + users can spot wrong picks. ``error_message``
    carries a user-actionable diagnostic for the unresolved case;
    empty otherwise.
    """
    path: Path | None
    source: str
    error_message: str = ''


# Modern (.venv) first so we prefer it when both styles coexist.
_VENV_NAMES: tuple[str, ...] = ('.venv', 'venv')


def _is_usable_python(path: Path) -> bool:
    """Return True if ``path`` is a regular file and is executable.

    Used to validate every interpreter candidate before claiming a
    resolution. Without this, an empty ``.venv/`` (e.g. partial poetry
    install) would falsely satisfy ``cwd-venv``, and the indexer would
    fail much later with a confusing error.
    """
    if not path.is_file():
        return False
    return os.access(str(path), os.X_OK)


def _find_venv_python(directory: Path) -> Path | None:
    """Return the first usable bin/python under directory's venv dirs.

    Checks ``.venv`` then ``venv``, in that order. Returns ``None`` if
    neither yields a usable interpreter (missing dir, missing
    bin/python, or bin/python not executable).
    """
    for name in _VENV_NAMES:
        candidate = directory / name / 'bin' / 'python'
        if _is_usable_python(candidate):
            return candidate
    return None


def _system_fallback(
    which: Callable[[str], str | None],
) -> InterpreterResolution | None:
    """Try ``python3`` then ``python`` on PATH.

    Returns ``None`` if neither resolves — the caller falls through
    to the unresolved result.
    """
    for name in ('python3', 'python'):
        found = which(name)
        if found:
            return InterpreterResolution(
                path=Path(found), source='system',
            )
    return None


def _unresolved() -> InterpreterResolution:
    """Build the terminal unresolved result with a user-actionable
    error message. The wording is intentionally specific: it names
    every probe location so the user knows exactly which signals
    failed."""
    return InterpreterResolution(
        path=None,
        source='unresolved',
        error_message=(
            'No Python interpreter found — checked env_hints, .venv/, '
            'venv/, ancestor venvs, and PATH (python3, python). Install '
            'Python or create a virtual environment in the project '
            'directory.'
        ),
    )


def resolve_python_interpreter(
    cwd: Path,
    *,
    env_hints: dict[str, str] | None = None,
    max_walk_up: int = 5,
    which: Callable[[str], str | None] | None = None,
) -> InterpreterResolution:
    """Resolve a Python interpreter for SCIP indexing in ``cwd``.

    See module docstring for the full probe order and rationale.
    """
    if which is None:
        which = shutil.which

    # Priority 1: explicit override (env_hints['python_path']).
    # Only honored if the file exists AND is executable — silently
    # fall through if not, so a stale config value doesn't kill the
    # run.
    if env_hints:
        explicit = env_hints.get('python_path')
        if explicit:
            explicit_path = Path(explicit)
            if _is_usable_python(explicit_path):
                return InterpreterResolution(
                    path=explicit_path, source='env_hints',
                )
            # Fall through if explicit path is unusable.

    # Defensive: cwd must be a directory for venv probes. If a caller
    # passes a file by mistake, skip to system fallback rather than
    # crashing on the iterdir-equivalent operations below.
    if not cwd.is_dir():
        sys_result = _system_fallback(which)
        return sys_result if sys_result is not None else _unresolved()

    # Priority 2: cwd's own .venv or venv.
    local = _find_venv_python(cwd)
    if local is not None:
        return InterpreterResolution(path=local, source='cwd-venv')

    # Priority 3: walk up to ancestor venv (bounded depth).
    current = cwd
    for _ in range(max_walk_up):
        parent = current.parent
        if parent == current:
            break  # filesystem root reached; terminate.
        ancestor = _find_venv_python(parent)
        if ancestor is not None:
            return InterpreterResolution(
                path=ancestor, source='walk-up',
            )
        current = parent

    # Priority 4: system fallback (which('python3') → which('python')).
    sys_result = _system_fallback(which)
    if sys_result is not None:
        return sys_result

    # Priority 5: nothing found.
    return _unresolved()


__all__ = [
    'InterpreterResolution',
    'resolve_python_interpreter',
]
