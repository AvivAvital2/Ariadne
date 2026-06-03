"""Venv auto-materialization for the Python indexer (Phase 2n.b).

When :class:`PythonIndexerAdapter` resolves to system Python AND the
project has a deps file, this module materializes a local ``.venv``
on demand. The user gets high-quality SCIP indexing without manually
running ``uv sync`` / ``poetry install``.

Detection priority:

1. ``uv.lock`` OR ``pyproject.toml`` with ``[tool.uv]`` → ``'uv'``
2. ``pyproject.toml`` with ``[tool.poetry]`` → ``'poetry'``
3. ``requirements.txt`` → ``'pip'``
4. otherwise → ``None`` (no materialization possible)

Why not just use ``pyproject.toml`` presence as a generic signal? Many
projects ship ``[project]`` metadata for hatchling / setuptools / flit
without indicating which tool runs. We only claim a tool when the
markers are unambiguous — guessing wrong wastes time on a doomed
``poetry install`` that errors immediately.

The subprocess runner is dependency-injected so tests can substitute
fakes; production uses ``subprocess.run``.
"""
from __future__ import annotations

import subprocess
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Literal

from attrs import frozen


_Tool = Literal['uv', 'poetry', 'pip', 'none']


@frozen
class MaterializationResult:
    """Outcome of a materialization attempt.

    On failure, ``error_message`` carries a user-actionable diagnostic;
    ``tool`` indicates which tool was attempted (or ``'none'`` if no
    deps file was detected).
    """
    success: bool
    tool: _Tool
    error_message: str = ''


def detect_deps_tool(cwd: Path) -> str | None:
    """Inspect ``cwd`` to determine which deps-tool the project uses.

    Returns the tool name (``'uv'`` / ``'poetry'`` / ``'pip'``) or
    ``None`` if no recognized signal is present.
    """
    # Priority 1: uv lockfile is the strongest signal — even overrides
    # legacy [tool.poetry] in the same pyproject (the lockfile is the
    # active tool's; project is migrating).
    if (cwd / 'uv.lock').exists():
        return 'uv'

    pyproject = cwd / 'pyproject.toml'
    if pyproject.exists():
        try:
            with pyproject.open('rb') as f:
                data = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError):
            data = {}
        tool = data.get('tool', {}) or {}
        # Priority 2 (no uv.lock): [tool.uv] in pyproject
        if 'uv' in tool:
            return 'uv'
        # Priority 3: [tool.poetry]
        if 'poetry' in tool:
            return 'poetry'

    # Priority 4: requirements.txt → pip path (only when no pyproject
    # tool section already claimed the project)
    if (cwd / 'requirements.txt').exists():
        return 'pip'

    return None


def _run_single(
    runner: Callable,
    cmd: list[str],
    cwd: Path,
    tool: _Tool,
) -> MaterializationResult:
    """Run a single materialization subprocess; map exceptions to a
    clean MaterializationResult."""
    try:
        result = runner(cmd, cwd=cwd, capture_output=True)
    except FileNotFoundError as e:
        return MaterializationResult(
            success=False,
            tool=tool,
            error_message=f'{tool} binary not found: {e}',
        )
    if result.returncode != 0:
        stderr = result.stderr
        if isinstance(stderr, bytes):
            stderr = stderr.decode('utf-8', errors='replace')
        return MaterializationResult(
            success=False,
            tool=tool,
            error_message=(
                stderr.strip() if stderr
                else f'{tool} exited nonzero'
            ),
        )
    return MaterializationResult(success=True, tool=tool)


def materialize_venv(
    cwd: Path,
    *,
    runner: Callable | None = None,
    system_python: str | None = None,
) -> MaterializationResult:
    """Materialize a venv in ``cwd`` according to the detected tool.

    - uv → ``uv sync`` (one subprocess; uv handles venv creation +
      install in one step)
    - poetry → ``poetry install --no-root`` (deps only; don't try to
      install the project itself which we may not have built)
    - pip → ``<system_python> -m venv .venv`` then
      ``./.venv/bin/pip install -r requirements.txt``. Requires
      ``system_python`` to be provided since pip needs a Python
      to drive ``-m venv``.

    Failures are caught and reported in ``MaterializationResult`` —
    callers (specifically ``PythonIndexerAdapter``) treat a failed
    materialization as "fall through with original resolution" rather
    than a hard error.
    """
    if runner is None:
        runner = subprocess.run

    tool = detect_deps_tool(cwd)
    if tool is None:
        return MaterializationResult(
            success=False,
            tool='none',
            error_message='No deps file found in cwd',
        )

    if tool == 'uv':
        return _run_single(runner, ['uv', 'sync'], cwd, 'uv')

    if tool == 'poetry':
        return _run_single(
            runner, ['poetry', 'install', '--no-root'], cwd, 'poetry',
        )

    # tool == 'pip'
    if system_python is None:
        return MaterializationResult(
            success=False,
            tool='pip',
            error_message=(
                'No system python available to drive `python -m venv`. '
                'Resolver must find a system Python before pip-path '
                'materialization can run.'
            ),
        )
    # Step 1: create venv with system python
    create = _run_single(
        runner,
        [system_python, '-m', 'venv', '.venv'],
        cwd,
        'pip',
    )
    if not create.success:
        return create
    # Step 2: install requirements with the new venv's pip
    pip = str(cwd / '.venv' / 'bin' / 'pip')
    return _run_single(
        runner,
        [pip, 'install', '-r', 'requirements.txt'],
        cwd,
        'pip',
    )


__all__ = [
    'MaterializationResult',
    'detect_deps_tool',
    'materialize_venv',
]
