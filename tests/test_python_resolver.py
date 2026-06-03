"""Contract for Python interpreter resolution (SCIP-everywhere Phase 2n).

PythonIndexerAdapter shouldn't require ``env_hints['python_path']`` from
the user — that contradicts Ariadne's "informs the unknowledgeable user"
purpose. For projects like scalaproject, multiple Python environments are
deployed via Dockerfile and referenced from HOCON; the user can't be
expected to map these dots. The resolver auto-resolves the interpreter
per ``cwd`` by probing conventional venv locations and falling back to
system python.

Probe order (highest priority first):

1. ``env_hints['python_path']`` (explicit override; only honored if file exists)
2. ``<cwd>/.venv/bin/python`` (uv, poetry, manual venv — modern convention)
3. ``<cwd>/venv/bin/python``
4. Walk up to ancestor's ``.venv/bin/python`` (bounded depth)
5. ``shutil.which('python3')`` → ``shutil.which('python')``
6. Unresolved (with clear error message)

Each :class:`InterpreterResolution` carries a ``source`` label so
callers can log what was picked and users can spot wrong picks.

These tests are the **compass + contract** for the resolver. They run
RED first (module does not exist yet); the implementation in the next
slice satisfies them.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _make_venv(parent: Path, name: str = '.venv') -> Path:
    """Create a fake venv layout under ``parent``.

    Returns the path to ``bin/python`` so tests can assert on it.
    """
    bin_dir = parent / name / 'bin'
    bin_dir.mkdir(parents=True)
    python = bin_dir / 'python'
    python.write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
    python.chmod(0o755)
    return python


def _no_system_python(name: str) -> str | None:
    """Stand-in for ``shutil.which`` that always returns None.

    Used when the test is asserting venv-based resolution and we want
    system fallback to be unambiguously off the table.
    """
    return None


# ---------------------------------------------------------------------------
# Probe priority (which signal wins when multiple are available)
# ---------------------------------------------------------------------------


class TestProbePriority:
    def test_env_hints_takes_priority_over_local_venv(
        self, tmp_path: Path,
    ) -> None:
        """env_hints['python_path'] wins over a local .venv. The user's
        explicit override is the highest-priority signal."""
        from docgen.python_resolver import resolve_python_interpreter

        _make_venv(tmp_path)
        explicit = tmp_path / 'explicit-bin' / 'python'
        explicit.parent.mkdir()
        explicit.write_text('#!/bin/sh\n')
        explicit.chmod(0o755)

        result = resolve_python_interpreter(
            tmp_path,
            env_hints={'python_path': str(explicit)},
            which=_no_system_python,
        )
        assert result.path == explicit
        assert result.source == 'env_hints'

    def test_cwd_dot_venv_preferred_over_walk_up(
        self, tmp_path: Path,
    ) -> None:
        """A local .venv beats an ancestor .venv. Closer wins."""
        from docgen.python_resolver import resolve_python_interpreter

        # Ancestor venv (further away, should NOT be picked)
        _make_venv(tmp_path)
        # Local cwd has its own venv (should be picked)
        cwd = tmp_path / 'pkg'
        cwd.mkdir()
        local_python = _make_venv(cwd)

        result = resolve_python_interpreter(
            cwd, which=_no_system_python,
        )
        assert result.path == local_python
        assert result.source == 'cwd-venv'

    def test_dot_venv_preferred_over_bare_venv(
        self, tmp_path: Path,
    ) -> None:
        """When both .venv/ and venv/ exist in cwd, .venv wins (modern
        convention used by uv, poetry, and recent manual workflows)."""
        from docgen.python_resolver import resolve_python_interpreter

        dot_venv_python = _make_venv(tmp_path, '.venv')
        _make_venv(tmp_path, 'venv')

        result = resolve_python_interpreter(
            tmp_path, which=_no_system_python,
        )
        assert result.path == dot_venv_python

    def test_bare_venv_used_when_dot_venv_absent(
        self, tmp_path: Path,
    ) -> None:
        """Falls through to venv/ when .venv/ is absent."""
        from docgen.python_resolver import resolve_python_interpreter

        bare = _make_venv(tmp_path, 'venv')
        result = resolve_python_interpreter(
            tmp_path, which=_no_system_python,
        )
        assert result.path == bare
        assert result.source == 'cwd-venv'


# ---------------------------------------------------------------------------
# Walk-up to ancestor venv
# ---------------------------------------------------------------------------


class TestWalkUp:
    def test_walks_up_to_ancestor_venv(self, tmp_path: Path) -> None:
        """If cwd has no venv but an ancestor does, find it via walk-up.

        Covers the monorepo case where a single .venv at the project
        root services multiple sub-package cwds."""
        from docgen.python_resolver import resolve_python_interpreter

        ancestor_python = _make_venv(tmp_path)
        deep = tmp_path / 'a' / 'b' / 'c'
        deep.mkdir(parents=True)

        result = resolve_python_interpreter(
            deep, which=_no_system_python,
        )
        assert result.path == ancestor_python
        assert result.source == 'walk-up'

    def test_walk_up_respects_max_depth(self, tmp_path: Path) -> None:
        """Walk-up stops at a bounded depth — must not search half the
        disk on a deeply-nested cwd."""
        from docgen.python_resolver import resolve_python_interpreter

        # Venv at the very top
        _make_venv(tmp_path)
        # Cwd is deep enough to exceed max_walk_up
        deep = tmp_path / 'a' / 'b' / 'c' / 'd' / 'e'
        deep.mkdir(parents=True)

        # max_walk_up=2 → only check parent + grandparent of cwd, not all
        # the way up to the root venv
        result = resolve_python_interpreter(
            deep, max_walk_up=2, which=_no_system_python,
        )
        assert result.source != 'walk-up'

    def test_walk_up_picks_nearest_venv(self, tmp_path: Path) -> None:
        """When two ancestors have venvs, the nearest one wins.

        This matches user intuition: the venv closest to the code is
        most likely the one that owns the code's dependencies."""
        from docgen.python_resolver import resolve_python_interpreter

        # Far ancestor venv (root)
        _make_venv(tmp_path)
        # Closer ancestor venv (mid-tree)
        mid = tmp_path / 'a' / 'b'
        mid.mkdir(parents=True)
        nearer_python = _make_venv(mid)
        # Cwd below mid
        deep = mid / 'c' / 'd'
        deep.mkdir(parents=True)

        result = resolve_python_interpreter(
            deep, which=_no_system_python,
        )
        assert result.path == nearer_python


# ---------------------------------------------------------------------------
# System fallback
# ---------------------------------------------------------------------------


class TestSystemFallback:
    def test_uses_system_python3_when_no_venv(
        self, tmp_path: Path,
    ) -> None:
        """No venv anywhere → fall back to ``which('python3')``."""
        from docgen.python_resolver import resolve_python_interpreter

        def fake_which(name: str) -> str | None:
            return '/usr/bin/python3' if name == 'python3' else None

        result = resolve_python_interpreter(tmp_path, which=fake_which)
        assert result.path == Path('/usr/bin/python3')
        assert result.source == 'system'

    def test_falls_back_to_python_when_python3_missing(
        self, tmp_path: Path,
    ) -> None:
        """``which('python3')`` returns None → try ``which('python')``.

        Covers older systems where only the unversioned binary exists."""
        from docgen.python_resolver import resolve_python_interpreter

        def fake_which(name: str) -> str | None:
            return '/usr/local/bin/python' if name == 'python' else None

        result = resolve_python_interpreter(tmp_path, which=fake_which)
        assert result.path == Path('/usr/local/bin/python')
        assert result.source == 'system'

    def test_unresolved_when_no_venv_and_no_system_python(
        self, tmp_path: Path,
    ) -> None:
        """Nothing found anywhere → unresolved with a clear message that
        a user can act on."""
        from docgen.python_resolver import resolve_python_interpreter

        result = resolve_python_interpreter(
            tmp_path, which=_no_system_python,
        )
        assert result.path is None
        assert result.source == 'unresolved'
        # User-actionable message
        assert 'python' in result.error_message.lower()


# ---------------------------------------------------------------------------
# Provenance / logging
# ---------------------------------------------------------------------------


class TestProvenance:
    def test_resolution_carries_source_label(
        self, tmp_path: Path,
    ) -> None:
        """Every resolution result carries a ``source`` label so the
        adapter can log what was picked. Helps users spot wrong picks
        before committing to a long index run."""
        from docgen.python_resolver import resolve_python_interpreter

        _make_venv(tmp_path)
        result = resolve_python_interpreter(
            tmp_path, which=_no_system_python,
        )
        assert result.source in {
            'env_hints', 'cwd-venv', 'walk-up', 'system', 'unresolved',
        }

    def test_env_hints_with_nonexistent_path_falls_through(
        self, tmp_path: Path,
    ) -> None:
        """Garbage in ``env_hints['python_path']`` (file doesn't exist)
        → skip and continue probing. Trust-but-verify: don't blindly
        accept a stale config value."""
        from docgen.python_resolver import resolve_python_interpreter

        local_venv_python = _make_venv(tmp_path)
        result = resolve_python_interpreter(
            tmp_path,
            env_hints={'python_path': '/nonexistent/garbage/python'},
            which=_no_system_python,
        )
        # Should fall through to cwd-venv, not blindly accept env_hints
        assert result.path == local_venv_python
        assert result.source == 'cwd-venv'


# ---------------------------------------------------------------------------
# Adversarial cases (per "include adversarial cases in the FIRST batch")
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_empty_env_hints_dict_is_fine(self, tmp_path: Path) -> None:
        """``{}`` should behave as no override — fall through to probing.
        Don't crash on a defaulted-empty dict from the caller."""
        from docgen.python_resolver import resolve_python_interpreter

        local = _make_venv(tmp_path)
        result = resolve_python_interpreter(
            tmp_path, env_hints={}, which=_no_system_python,
        )
        assert result.path == local

    def test_none_env_hints_is_fine(self, tmp_path: Path) -> None:
        """``None`` should behave as ``{}`` — fall through to probing."""
        from docgen.python_resolver import resolve_python_interpreter

        local = _make_venv(tmp_path)
        result = resolve_python_interpreter(
            tmp_path, env_hints=None, which=_no_system_python,
        )
        assert result.path == local

    def test_cwd_is_a_file_returns_unresolved(
        self, tmp_path: Path,
    ) -> None:
        """Defensive: if cwd is somehow a file (caller bug), don't crash —
        return unresolved with a clear message."""
        from docgen.python_resolver import resolve_python_interpreter

        not_a_dir = tmp_path / 'notadir.txt'
        not_a_dir.write_text('hi')

        result = resolve_python_interpreter(
            not_a_dir, which=_no_system_python,
        )
        assert result.path is None
        assert result.source == 'unresolved'

    def test_filesystem_root_walk_up_terminates(
        self, tmp_path: Path,
    ) -> None:
        """Walking up from cwd must terminate at the filesystem root,
        not loop. Unbounded walks are a DOS waiting to happen."""
        from docgen.python_resolver import resolve_python_interpreter

        deep = tmp_path / 'a' / 'b'
        deep.mkdir(parents=True)

        def fake_which(name: str) -> str | None:
            return '/usr/bin/python3' if name == 'python3' else None

        # No venv anywhere; should reach root, fall back to system,
        # NOT loop indefinitely
        result = resolve_python_interpreter(deep, which=fake_which)
        assert result.source == 'system'

    def test_venv_directory_without_python_binary_is_skipped(
        self, tmp_path: Path,
    ) -> None:
        """If ``.venv/`` exists but has no ``bin/python`` (corrupted or
        partial venv), skip it and continue probing. Don't claim
        ``cwd-venv`` for an unusable venv."""
        from docgen.python_resolver import resolve_python_interpreter

        # Empty .venv directory (no bin/python)
        (tmp_path / '.venv').mkdir()

        def fake_which(name: str) -> str | None:
            return '/usr/bin/python3' if name == 'python3' else None

        result = resolve_python_interpreter(tmp_path, which=fake_which)
        assert result.source != 'cwd-venv'
        assert result.source == 'system'

    def test_env_hints_path_is_not_executable_falls_through(
        self, tmp_path: Path,
    ) -> None:
        """File at env_hints path exists but is not executable (e.g.,
        wrong path saved into config) → skip and continue probing."""
        from docgen.python_resolver import resolve_python_interpreter

        # Plain text file, not chmod +x'd
        not_executable = tmp_path / 'not-executable'
        not_executable.write_text('not a real python\n')

        local = _make_venv(tmp_path)
        result = resolve_python_interpreter(
            tmp_path,
            env_hints={'python_path': str(not_executable)},
            which=_no_system_python,
        )
        assert result.path == local
        assert result.source == 'cwd-venv'


# ---------------------------------------------------------------------------
# API shape
# ---------------------------------------------------------------------------


class TestAPIShape:
    def test_resolution_is_attrs_frozen(self, tmp_path: Path) -> None:
        """Resolutions are immutable value objects (attrs @frozen) so
        callers can pass them around without defensive copies and tests
        can assert on equality."""
        from docgen.python_resolver import (
            InterpreterResolution,
            resolve_python_interpreter,
        )

        _make_venv(tmp_path)
        result = resolve_python_interpreter(
            tmp_path, which=_no_system_python,
        )
        assert isinstance(result, InterpreterResolution)
        # @frozen raises FrozenInstanceError on attribute assignment
        try:
            result.path = Path('/etc/passwd')  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError(
            'InterpreterResolution should be frozen (attrs @frozen)',
        )

    def test_which_callable_is_injectable(self, tmp_path: Path) -> None:
        """Tests must be able to inject a fake ``which`` to keep them
        OS-independent. Production passes None and gets ``shutil.which``."""
        from docgen.python_resolver import resolve_python_interpreter

        called_with: list[str] = []

        def fake_which(name: str) -> str | None:
            called_with.append(name)
            return None

        resolve_python_interpreter(tmp_path, which=fake_which)
        # which was called for at least one python name
        assert called_with
        assert all(n.startswith('python') for n in called_with)
