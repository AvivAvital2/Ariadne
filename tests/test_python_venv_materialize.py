"""Contract for venv auto-materialization (Phase 2n.b).

When :class:`PythonIndexerAdapter` resolves to system Python AND the
project has a deps file (``pyproject.toml`` / ``requirements.txt``)
but no local venv, the adapter triggers this module to create one.
The unknowledgeable user gets high-quality SCIP indexing without
manually running ``uv sync`` / ``poetry install`` first.

Detection priority:

1. ``uv.lock`` OR ``pyproject.toml`` with ``[tool.uv]`` → ``'uv'``
2. ``pyproject.toml`` with ``[tool.poetry]`` → ``'poetry'``
3. ``requirements.txt`` → ``'pip'``
4. nothing → ``None``

Materialization commands:

- uv → ``uv sync``
- poetry → ``poetry install --no-root``
- pip → ``<system_python> -m venv .venv`` then
  ``./.venv/bin/pip install -r requirements.txt``

These tests are RED until ``docgen/python_venv_materialize.py`` exists.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


def _touch(path: Path, content: str = '') -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


class FakeRunner:
    """Stand-in for subprocess.run; records calls; canned response."""

    def __init__(
        self, *, returncode: int = 0, stderr: bytes = b'',
    ) -> None:
        self.returncode = returncode
        self.stderr = stderr
        self.calls: list[dict] = []

    def __call__(self, cmd, *, cwd=None, capture_output=True, **kwargs):
        self.calls.append({'cmd': list(cmd), 'cwd': cwd})
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=b'',
            stderr=self.stderr,
        )


# ---------------------------------------------------------------------------
# Detect deps tool
# ---------------------------------------------------------------------------


class TestDetectDepsTool:
    def test_uv_lock_detected_as_uv(self, tmp_path: Path) -> None:
        """``uv.lock`` is the strongest signal — overrides anything else."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(tmp_path / 'uv.lock')
        _touch(tmp_path / 'pyproject.toml', '[project]\nname = "x"\n')
        assert detect_deps_tool(tmp_path) == 'uv'

    def test_pyproject_with_uv_tool_section_detected_as_uv(
        self, tmp_path: Path,
    ) -> None:
        """``[tool.uv]`` in pyproject.toml signals uv even without a
        lockfile (e.g., fresh project, lockfile not yet generated)."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(
            tmp_path / 'pyproject.toml',
            '[project]\nname = "x"\n\n[tool.uv]\n',
        )
        assert detect_deps_tool(tmp_path) == 'uv'

    def test_pyproject_with_poetry_section_detected_as_poetry(
        self, tmp_path: Path,
    ) -> None:
        """``[tool.poetry]`` in pyproject.toml is poetry's marker."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(
            tmp_path / 'pyproject.toml',
            '[tool.poetry]\nname = "x"\nversion = "0.1.0"\n',
        )
        assert detect_deps_tool(tmp_path) == 'poetry'

    def test_requirements_txt_detected_as_pip(
        self, tmp_path: Path,
    ) -> None:
        """``requirements.txt`` (with no pyproject.toml) is the pip path."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(tmp_path / 'requirements.txt', 'requests==2.0\n')
        assert detect_deps_tool(tmp_path) == 'pip'

    def test_no_deps_file_returns_none(self, tmp_path: Path) -> None:
        """Empty cwd → no detection."""
        from docgen.python_venv_materialize import detect_deps_tool

        assert detect_deps_tool(tmp_path) is None

    def test_uv_lock_priority_over_poetry(self, tmp_path: Path) -> None:
        """If both ``uv.lock`` AND ``[tool.poetry]`` exist, uv wins —
        uv.lock is the active tool's lockfile, indicating uv is in use
        despite legacy poetry config."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(tmp_path / 'uv.lock')
        _touch(
            tmp_path / 'pyproject.toml',
            '[tool.poetry]\nname = "x"\nversion = "0.1.0"\n',
        )
        assert detect_deps_tool(tmp_path) == 'uv'

    def test_poetry_priority_over_pip(self, tmp_path: Path) -> None:
        """If pyproject.toml has ``[tool.poetry]`` AND requirements.txt
        exists, poetry wins (project's primary tool)."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(
            tmp_path / 'pyproject.toml',
            '[tool.poetry]\nname = "x"\nversion = "0.1.0"\n',
        )
        _touch(tmp_path / 'requirements.txt', 'requests\n')
        assert detect_deps_tool(tmp_path) == 'poetry'

    def test_pyproject_without_known_tool_section_returns_none(
        self, tmp_path: Path,
    ) -> None:
        """``pyproject.toml`` without ``[tool.uv]`` or ``[tool.poetry]``
        and no lockfile is ambiguous (could be hatchling, setuptools,
        etc.). Don't claim a tool we can't drive."""
        from docgen.python_venv_materialize import detect_deps_tool

        _touch(
            tmp_path / 'pyproject.toml',
            '[project]\nname = "x"\nversion = "0.1.0"\n',
        )
        assert detect_deps_tool(tmp_path) is None


# ---------------------------------------------------------------------------
# Materialize venv
# ---------------------------------------------------------------------------


class TestMaterializeVenv:
    def test_uv_invokes_uv_sync(self, tmp_path: Path) -> None:
        """Project with uv lockfile → ``uv sync`` runs in cwd."""
        from docgen.python_venv_materialize import materialize_venv

        _touch(tmp_path / 'uv.lock')
        _touch(tmp_path / 'pyproject.toml', '[project]\nname = "x"\n')

        runner = FakeRunner()
        result = materialize_venv(tmp_path, runner=runner)

        assert result.success
        assert result.tool == 'uv'
        assert len(runner.calls) == 1
        cmd = runner.calls[0]['cmd']
        assert cmd[0] == 'uv'
        assert 'sync' in cmd
        # Subprocess runs in cwd (so it finds the lockfile)
        assert runner.calls[0]['cwd'] == tmp_path

    def test_poetry_invokes_poetry_install_no_root(
        self, tmp_path: Path,
    ) -> None:
        """Project with [tool.poetry] → ``poetry install --no-root``."""
        from docgen.python_venv_materialize import materialize_venv

        _touch(
            tmp_path / 'pyproject.toml',
            '[tool.poetry]\nname = "x"\nversion = "0.1.0"\n',
        )

        runner = FakeRunner()
        result = materialize_venv(tmp_path, runner=runner)

        assert result.success
        assert result.tool == 'poetry'
        cmd = runner.calls[0]['cmd']
        assert cmd[0] == 'poetry'
        assert 'install' in cmd
        # --no-root: don't try to install the project itself, just deps
        assert '--no-root' in cmd

    def test_pip_creates_venv_then_pip_installs(
        self, tmp_path: Path,
    ) -> None:
        """Project with requirements.txt → two-step:
        1. ``<system_python> -m venv .venv``
        2. ``./.venv/bin/pip install -r requirements.txt``"""
        from docgen.python_venv_materialize import materialize_venv

        _touch(tmp_path / 'requirements.txt', 'requests\n')

        runner = FakeRunner()
        result = materialize_venv(
            tmp_path, runner=runner,
            system_python='/usr/bin/python3',
        )

        assert result.success
        assert result.tool == 'pip'
        # Two subprocess calls: venv create, pip install
        assert len(runner.calls) >= 2

        # First: venv create
        cmd_create = runner.calls[0]['cmd']
        assert '/usr/bin/python3' in cmd_create[0]
        assert '-m' in cmd_create
        assert 'venv' in cmd_create

        # Second: pip install -r requirements.txt
        cmd_install = runner.calls[1]['cmd']
        assert 'install' in cmd_install
        assert '-r' in cmd_install
        assert 'requirements.txt' in cmd_install

    def test_no_deps_file_returns_failure(
        self, tmp_path: Path,
    ) -> None:
        """Empty cwd → returns failure with tool='none'; no subprocess."""
        from docgen.python_venv_materialize import materialize_venv

        runner = FakeRunner()
        result = materialize_venv(tmp_path, runner=runner)

        assert result.success is False
        assert result.tool == 'none'
        assert runner.calls == []

    def test_subprocess_failure_reported(self, tmp_path: Path) -> None:
        """When the tool returns nonzero, surface stderr in
        error_message so the user can act on it."""
        from docgen.python_venv_materialize import materialize_venv

        _touch(tmp_path / 'uv.lock')
        _touch(tmp_path / 'pyproject.toml', '[project]\nname = "x"\n')

        runner = FakeRunner(
            returncode=1, stderr=b'uv sync failed: package not found',
        )
        result = materialize_venv(tmp_path, runner=runner)

        assert result.success is False
        assert result.tool == 'uv'
        assert 'uv sync failed' in result.error_message

    def test_binary_not_found_reported(self, tmp_path: Path) -> None:
        """When the tool isn't on PATH (FileNotFoundError), error_message
        names the missing tool so the user knows what to install."""
        from docgen.python_venv_materialize import materialize_venv

        _touch(tmp_path / 'uv.lock')
        _touch(tmp_path / 'pyproject.toml', '[project]\nname = "x"\n')

        def raising_runner(cmd, **kwargs):
            raise FileNotFoundError('uv not found on PATH')

        result = materialize_venv(tmp_path, runner=raising_runner)

        assert result.success is False
        assert result.tool == 'uv'
        assert 'uv' in result.error_message.lower()

    def test_pip_path_requires_system_python(
        self, tmp_path: Path,
    ) -> None:
        """pip path needs a system Python to run ``-m venv``. Without
        it, materialization can't proceed — report failure cleanly."""
        from docgen.python_venv_materialize import materialize_venv

        _touch(tmp_path / 'requirements.txt')

        runner = FakeRunner()
        result = materialize_venv(
            tmp_path, runner=runner, system_python=None,
        )

        assert result.success is False
        assert result.tool == 'pip'
        assert 'python' in result.error_message.lower()
        # Didn't even try to invoke
        assert runner.calls == []

    def test_runner_dependency_injectable(
        self, tmp_path: Path,
    ) -> None:
        """Production uses ``subprocess.run``; tests pass any callable.
        Pattern matches the rest of the SCIP indexer-adapter family."""
        from docgen.python_venv_materialize import materialize_venv

        _touch(tmp_path / 'uv.lock')
        _touch(tmp_path / 'pyproject.toml', '[project]\nname = "x"\n')

        called: list = []

        def custom_runner(cmd, **kwargs):
            called.append(list(cmd))
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        materialize_venv(tmp_path, runner=custom_runner)
        assert len(called) >= 1
