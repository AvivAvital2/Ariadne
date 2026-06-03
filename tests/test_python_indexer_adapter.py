"""Contract for ``PythonIndexerAdapter`` — Phase 2l second slice.

The adapter shells out to ``scip-python``. To keep tests deterministic
and OS-independent, the subprocess runner is injectable: tests pass a
``FakeRunner`` that records invocations and produces canned output;
production uses ``subprocess.run`` directly.

Tests articulate:
- Command shape: scip-python is invoked with the right flags
- cwd: subprocess runs in the indexer's declared working directory
- Output: produced .scip lands at the requested path
- Failure modes: nonzero returncode → IndexerResult.success=False
- Binary missing: FileNotFoundError → IndexerResult.success=False with
  a clear error message
- Version capture: the indexer version surfaces in IndexerResult

Phase 2n second slice — the adapter calls the interpreter resolver
(:mod:`docgen.python_resolver`) instead of requiring
``env_hints['python_path']`` from the user. The resolver was tested in
isolation in ``tests/test_python_resolver.py``. Tests here articulate
the *integration*:

- Adapter calls resolver with cwd + env_hints
- Resolved interpreter lands in the transient pyrightconfig.json
- ``IndexerResult`` exposes ``resolved_interpreter`` + ``resolution_source``
  so users can see what was picked
- Resolver returning ``'unresolved'`` aborts with a clean failure; the
  scip-python subprocess is NOT invoked
- ``env_hints['python_path']`` still wins (resolver's own priority logic)
- A user-authored ``pyrightconfig.json`` is respected even when the
  resolver succeeds
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest


def _make_venv(parent: Path, name: str = '.venv') -> Path:
    """Create a fake venv layout under ``parent``.

    Returns the path to ``bin/python`` so tests can pass it as
    env_hints or assert it landed in the resolution / config.
    """
    bin_dir = parent / name / 'bin'
    bin_dir.mkdir(parents=True)
    python = bin_dir / 'python'
    python.write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
    python.chmod(0o755)
    return python


def _no_system_python(name: str) -> str | None:
    """Stand-in for shutil.which that always returns None.

    Lets tests assert the resolver result is driven entirely by venv
    probes (or env_hints), with system fallback unambiguously off.
    """
    return None


class FakeRunner:
    """Drop-in for subprocess.run. Records invocations; canned response."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: bytes = b'',
        stderr: bytes = b'',
        scip_bytes: bytes = b'\x08\x01synthetic',
        raise_on_call: Exception | None = None,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.scip_bytes = scip_bytes
        self.raise_on_call = raise_on_call
        self.calls: list[dict] = []

    def __call__(self, cmd, *, cwd=None, capture_output=True, **kwargs):
        self.calls.append({
            'cmd': list(cmd), 'cwd': cwd,
        })
        if self.raise_on_call:
            raise self.raise_on_call
        # Simulate scip-python writing the .scip file when successful
        if self.returncode == 0 and '--output' in cmd:
            out_path = Path(cmd[cmd.index('--output') + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(self.scip_bytes)
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


# ---------------------------------------------------------------------------
# Command shape
# ---------------------------------------------------------------------------


class TestCommandShape:
    def test_invokes_scip_python_index(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)

        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success
        cmd = runner.calls[0]['cmd']
        # First arg is the binary
        assert cmd[0] == 'scip-python'
        # 'index' subcommand is present
        assert 'index' in cmd

    def test_output_flag_uses_target_path(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)
        out = tmp_path / 'subdir' / 'out.scip'
        adapter.run(cwd=tmp_path, output=out, env_hints={})

        cmd = runner.calls[0]['cmd']
        assert '--output' in cmd
        assert cmd[cmd.index('--output') + 1] == str(out)

    def test_runs_in_specified_cwd(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import PythonIndexerAdapter

        # cwd must exist on disk — Phase 2n's adapter writes a transient
        # pyrightconfig in cwd, which fails otherwise. The test's intent
        # is "subprocess called with the right cwd," not "cwd needn't
        # exist," so create the directory.
        pkg_root = tmp_path / 'pkg_root'
        pkg_root.mkdir()

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)
        adapter.run(
            cwd=pkg_root,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        # subprocess is called with cwd=<the indexer's cwd>
        assert runner.calls[0]['cwd'] == pkg_root


# ---------------------------------------------------------------------------
# Success
# ---------------------------------------------------------------------------


class TestSuccess:
    def test_returns_success_on_zero_returncode(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner(returncode=0)
        adapter = PythonIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success is True
        # .scip artifact was produced
        assert (tmp_path / 'out.scip').exists()

    def test_indexer_version_populated(self, tmp_path: Path) -> None:
        """Version surfaces in IndexerResult so manifest entries can
        record what produced the artifact."""
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        # At minimum, version starts with 'scip-python'
        assert result.indexer_version.startswith('scip-python')


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


class TestFailure:
    def test_nonzero_returncode_reports_failure(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner(
            returncode=1,
            stderr=b'pyright: cannot resolve types\n',
        )
        adapter = PythonIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success is False
        assert 'pyright' in result.error_message

    def test_binary_not_found_reports_failure(self, tmp_path: Path) -> None:
        """Missing scip-python on PATH surfaces as a clean error
        message, not an exception bubbling up."""
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner(
            raise_on_call=FileNotFoundError(
                "[Errno 2] No such file or directory: 'scip-python'",
            ),
        )
        adapter = PythonIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success is False
        assert 'scip-python' in result.error_message
        # Helpful hint that the user can act on
        assert 'install' in result.error_message.lower() or \
               'not found' in result.error_message.lower() or \
               'path' in result.error_message.lower()


# ---------------------------------------------------------------------------
# env_hints
# ---------------------------------------------------------------------------


class TestZeroConfigSetup:
    """Per design decision #1 (no pollution of consumed repos beyond
    .ariadne/ + .gitignore), the adapter writes a transient
    pyrightconfig.json to the project root when none exists, then
    cleans it up after the run.

    Pyright searches for pyrightconfig.json starting at the project
    root and walking up — there is no equivalent to scip-typescript's
    ``--infer-tsconfig``, so the transient-file approach is the only
    mechanism. The lifetime is bounded to the indexer's run.
    """

    def test_writes_transient_pyrightconfig_when_no_project_config(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import PythonIndexerAdapter

        # Use a real (test-created) interpreter path so the resolver's
        # validation passes and env_hints wins as the priority signal.
        explicit_venv_dir = tmp_path / 'myenv'
        explicit_venv_dir.mkdir()
        explicit_python = _make_venv(explicit_venv_dir, '.venv')

        # Probe the cwd at call-time to verify the transient file is
        # present DURING the subprocess invocation (the adapter writes
        # it before, deletes after).
        observed: dict = {'pyrightconfig_present': False, 'content': None}

        def probing_runner(
            cmd, cwd=None, capture_output=True, env=None, **kwargs,
        ):
            pc = Path(cwd) / 'pyrightconfig.json'
            observed['pyrightconfig_present'] = pc.exists()
            if pc.exists():
                observed['content'] = pc.read_text(encoding='utf-8')
            if '--output' in cmd:
                out_path = Path(cmd[cmd.index('--output') + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(runner=probing_runner)
        adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': str(explicit_python)},
        )

        # Transient config existed during the subprocess call
        assert observed['pyrightconfig_present'] is True
        # python_path made it into the config
        assert str(explicit_python) in observed['content']
        # ``useLibraryCodeForTypes: false`` MUST be set so pyright
        # doesn't walk into every imported library's source. With the
        # default ``true``, indexing on a Python with anaconda's
        # site-packages takes hours and floods stderr with warnings.
        # Bites a fix that omits the setting from the template.
        import json as _json
        parsed = _json.loads(observed['content'])
        assert parsed.get('useLibraryCodeForTypes') is False
        # Cleaned up after — no pollution
        assert not (tmp_path / 'pyrightconfig.json').exists()

    def test_excludes_argument_flows_into_pyrightconfig(
        self, tmp_path: Path,
    ) -> None:
        """User-supplied excludes (resolved from ariadne.yaml's
        exclude_policy / exclude_dirs / exempt_dirs / exclude
        patterns by ``cli_core.cmd_index``) MUST land in the transient
        pyrightconfig's ``exclude`` array alongside the maintenance
        baselines.

        Bites a regression where excludes are silently dropped — which
        sends pyright walking into ``.venv``, ``docs``, anaconda's
        site-packages, etc. and turns a 30s index into hours."""
        from docgen.scip_indexers import PythonIndexerAdapter

        venv_dir = tmp_path / 'env'
        venv_dir.mkdir()
        python = _make_venv(venv_dir, '.venv')

        observed: dict = {'content': None}

        def probing_runner(
            cmd, cwd=None, capture_output=True, env=None, **kwargs,
        ):
            pc = Path(cwd) / 'pyrightconfig.json'
            if pc.exists():
                observed['content'] = pc.read_text(encoding='utf-8')
            if '--output' in cmd:
                out_path = Path(cmd[cmd.index('--output') + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(runner=probing_runner)
        adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': str(python)},
            excludes=('**/.venv', '**/docs', '**/secrets/**'),
        )

        import json as _json
        parsed = _json.loads(observed['content'])
        excludes = parsed['exclude']
        # Maintenance defaults present (universal, hardcoded).
        assert '**/__pycache__' in excludes
        assert '**/.git' in excludes
        # User excludes flowed through.
        assert '**/.venv' in excludes
        assert '**/docs' in excludes
        assert '**/secrets/**' in excludes

    def test_does_not_overwrite_existing_pyrightconfig(
        self, tmp_path: Path,
    ) -> None:
        """User-provided pyrightconfig.json is preserved, not
        overwritten with a transient version."""
        from docgen.scip_indexers import PythonIndexerAdapter

        existing = tmp_path / 'pyrightconfig.json'
        existing.write_text(
            '{"venvPath": "/user/env", "venv": "myenv"}',
            encoding='utf-8',
        )

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': '/different/path/python'},
        )
        # File still has user content
        content = existing.read_text(encoding='utf-8')
        assert '/user/env' in content
        assert 'myenv' in content

    def test_pyrightconfig_cleaned_up_on_subprocess_failure(
        self, tmp_path: Path,
    ) -> None:
        """If scip-python returns nonzero, the transient pyrightconfig
        is still removed — no leftover pollution from failed runs."""
        from docgen.scip_indexers import PythonIndexerAdapter

        def failing_runner(cmd, cwd=None, **kwargs):
            return SimpleNamespace(
                returncode=1, stdout=b'', stderr=b'fake error',
            )

        adapter = PythonIndexerAdapter(runner=failing_runner)
        adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': '/x/python'},
        )

        # Even though the run failed, the transient file is gone
        assert not (tmp_path / 'pyrightconfig.json').exists()

    def test_pyrightconfig_cleaned_up_on_subprocess_exception(
        self, tmp_path: Path,
    ) -> None:
        """If the runner raises (e.g., FileNotFoundError for the binary),
        the transient pyrightconfig is still removed."""
        from docgen.scip_indexers import PythonIndexerAdapter

        def raising_runner(cmd, cwd=None, **kwargs):
            raise FileNotFoundError('scip-python not found')

        adapter = PythonIndexerAdapter(runner=raising_runner)
        adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': '/x/python'},
        )
        assert not (tmp_path / 'pyrightconfig.json').exists()

    def test_no_transient_written_when_resolver_fails(
        self, tmp_path: Path,
    ) -> None:
        """Phase 2n: when the resolver fails (no venv, no system
        python), the adapter returns failure and does NOT write a
        transient pyrightconfig — there's nothing meaningful to put in
        ``pythonPath``.

        Replaces the old ``test_no_transient_written_when_no_python_path_hint``
        which articulated the pre-Phase-2n contract (no env_hints →
        no transient). With the resolver in place, no env_hints just
        means probe; transient gets written when the resolver picks
        an interpreter (which is the common case)."""
        from docgen.scip_indexers import PythonIndexerAdapter

        observed: dict = {'pyrightconfig_present': False}

        def probing_runner(cmd, cwd=None, **kwargs):
            pc = Path(cwd) / 'pyrightconfig.json'
            observed['pyrightconfig_present'] = pc.exists()
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(
            runner=probing_runner,
            # Force the resolver to return 'unresolved' by pretending
            # no system python exists.
            resolver_which=lambda name: None,
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        assert result.success is False
        # Adapter aborted before invoking scip-python — probing_runner
        # never ran, so observed stays at its initial False.
        assert observed['pyrightconfig_present'] is False
        # And no transient was left behind on disk
        assert not (tmp_path / 'pyrightconfig.json').exists()


class TestEnvHints:
    def test_python_path_propagated_to_environment(
        self, tmp_path: Path,
    ) -> None:
        """When env_hints['python_path'] is set (and points at a real
        interpreter so the resolver accepts it), the adapter ensures
        scip-python's invocation can find that interpreter. The exact
        mechanism (env var, config file, flag) is implementation
        detail; the contract is that the value reaches the subprocess."""
        from docgen.scip_indexers import PythonIndexerAdapter

        # Phase 2n: env_hints flows through the resolver, which now
        # validates the path exists + is executable. Use a real path
        # so the resolver accepts it as the env_hints-source winner.
        explicit_venv_dir = tmp_path / 'myenv'
        explicit_venv_dir.mkdir()
        explicit_python = _make_venv(explicit_venv_dir, '.venv')

        captured_env = []
        captured_cmd = []

        def runner(cmd, *, cwd=None, capture_output=True, env=None,
                   **kwargs):
            captured_cmd.append(list(cmd))
            captured_env.append(dict(env) if env else {})
            # Simulate success + artifact
            if '--output' in cmd:
                out_path = Path(cmd[cmd.index('--output') + 1])
                out_path.parent.mkdir(parents=True, exist_ok=True)
                out_path.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(
                returncode=0, stdout=b'', stderr=b'',
            )

        adapter = PythonIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': str(explicit_python)},
        )

        # Either the env carries the value, or the cmd does.
        env_carries = any(
            str(explicit_python) in str(v)
            for env in captured_env for v in env.values()
        )
        cmd_carries = any(
            str(explicit_python) in str(arg)
            for cmd in captured_cmd for arg in cmd
        )
        assert env_carries or cmd_carries, (
            'python_path must reach scip-python via env or args'
        )


# ---------------------------------------------------------------------------
# Phase 2n second slice — resolver-driven adapter integration
# ---------------------------------------------------------------------------


class TestResolverIntegration:
    """The adapter calls :func:`docgen.python_resolver.resolve_python_interpreter`
    instead of requiring ``env_hints['python_path']`` from the user.

    Resolver behavior was tested in isolation in
    ``tests/test_python_resolver.py``. These tests pin the *integration*:

    - When env_hints lacks ``python_path``, adapter calls resolver and
      uses the resolved interpreter in the transient pyrightconfig.
    - ``IndexerResult`` exposes ``resolved_interpreter`` and
      ``resolution_source`` so users can see what was picked.
    - Resolver returning ``'unresolved'`` aborts cleanly — scip-python
      is NOT invoked, the result carries ``success=False`` with the
      resolver's error_message.
    - ``env_hints['python_path']`` (when set to a real path) wins via
      the resolver's own priority logic.
    - The transient pyrightconfig.json's ``include`` pattern dispatches
      on the new ``entry_kind`` parameter:
      * ``'package'`` → ``"include": ["."]`` (current behavior; default)
      * ``'scripts'`` → ``"include": ["./*.py"]`` so pyright treats
        each .py file as a standalone module (used for orphan
        directories from Phase 2j.b's script-entry emission).
    - User-authored pyrightconfig.json still wins; transient is
      only written when no project config exists.
    - ``resolver_which`` on the adapter constructor is dependency-
      injected so tests can stub system-python lookup.
    """

    def test_resolves_interpreter_from_cwd_venv_when_no_env_hints(
        self, tmp_path: Path,
    ) -> None:
        """No env_hints → adapter calls resolver → cwd/.venv used.
        The resolved interpreter path lands in the transient
        pyrightconfig content."""
        from docgen.scip_indexers import PythonIndexerAdapter

        venv_python = _make_venv(tmp_path)

        observed: dict = {'config_content': None}

        def probing_runner(cmd, cwd=None, **kwargs):
            pc = Path(cwd) / 'pyrightconfig.json'
            if pc.exists():
                observed['config_content'] = pc.read_text(encoding='utf-8')
            if '--output' in cmd:
                out = Path(cmd[cmd.index('--output') + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(runner=probing_runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        assert result.success
        assert observed['config_content'] is not None
        # The local .venv path is what the resolver picked
        assert str(venv_python) in observed['config_content']

    def test_indexer_result_exposes_resolved_interpreter_and_source(
        self, tmp_path: Path,
    ) -> None:
        """IndexerResult surfaces ``resolved_interpreter`` (Path) and
        ``resolution_source`` (provenance label) so the index command
        can log the choice; users can spot wrong picks without
        reading the transient config."""
        from docgen.scip_indexers import PythonIndexerAdapter

        _make_venv(tmp_path)

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        assert result.success
        assert result.resolution_source == 'cwd-venv'
        assert result.resolved_interpreter is not None
        assert '.venv' in str(result.resolved_interpreter)

    def test_unresolved_returns_failure_without_running_scip_python(
        self, tmp_path: Path,
    ) -> None:
        """No venv anywhere AND no system python → the adapter MUST
        return ``success=False`` with the resolver's error_message
        intact, and MUST NOT invoke scip-python (no point — there's
        nothing to point pyright at)."""
        from docgen.scip_indexers import PythonIndexerAdapter

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(
            runner=runner,
            resolver_which=lambda name: None,  # no system python
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        assert result.success is False
        assert 'python' in result.error_message.lower()
        # scip-python was NOT invoked
        assert runner.calls == []

    def test_unresolved_indexer_result_carries_unresolved_source(
        self, tmp_path: Path,
    ) -> None:
        """Failure result still carries the source label — set to
        ``'unresolved'`` per the resolver — and ``resolved_interpreter``
        is None."""
        from docgen.scip_indexers import PythonIndexerAdapter

        adapter = PythonIndexerAdapter(
            runner=FakeRunner(),
            resolver_which=lambda name: None,
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        assert result.resolution_source == 'unresolved'
        assert result.resolved_interpreter is None

    def test_env_hints_takes_priority_via_resolver(
        self, tmp_path: Path,
    ) -> None:
        """env_hints['python_path'] (when pointing at a real, executable
        file) wins over a local .venv via the resolver's own priority
        logic. The override path lands in the transient config, and
        resolution_source reports 'env_hints'."""
        from docgen.scip_indexers import PythonIndexerAdapter

        # Local .venv exists but should be overridden
        _make_venv(tmp_path)
        # Explicit override path
        explicit = tmp_path / 'explicit-bin' / 'python'
        explicit.parent.mkdir()
        explicit.write_text('#!/bin/sh\n')
        explicit.chmod(0o755)

        observed: dict = {'config_content': None}

        def probing_runner(cmd, cwd=None, **kwargs):
            pc = Path(cwd) / 'pyrightconfig.json'
            if pc.exists():
                observed['config_content'] = pc.read_text(encoding='utf-8')
            if '--output' in cmd:
                out = Path(cmd[cmd.index('--output') + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(runner=probing_runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={'python_path': str(explicit)},
        )

        assert result.success
        assert str(explicit) in observed['config_content']
        assert result.resolution_source == 'env_hints'

    def test_script_entry_kind_uses_per_file_include_pattern(
        self, tmp_path: Path,
    ) -> None:
        """``entry_kind='scripts'`` → transient pyrightconfig's include
        pattern is ``./*.py`` so pyright treats each .py file as a
        standalone module. This is what Phase 2j.b's script entries
        need (orphan directories without __init__.py)."""
        from docgen.scip_indexers import PythonIndexerAdapter

        _make_venv(tmp_path)
        (tmp_path / 'foo.py').write_text('def foo(): pass\n')
        (tmp_path / 'bar.py').write_text('def bar(): pass\n')

        observed: dict = {'config_content': None}

        def probing_runner(cmd, cwd=None, **kwargs):
            pc = Path(cwd) / 'pyrightconfig.json'
            if pc.exists():
                observed['config_content'] = pc.read_text(encoding='utf-8')
            if '--output' in cmd:
                out = Path(cmd[cmd.index('--output') + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(runner=probing_runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
            entry_kind='scripts',
        )

        assert result.success
        # Per-file include pattern, not the package-style "."
        assert './*.py' in observed['config_content']

    def test_package_entry_kind_uses_dot_include_pattern(
        self, tmp_path: Path,
    ) -> None:
        """``entry_kind='package'`` (default) → ``include: ["."]`` so
        pyright treats the cwd as a package root. Existing behavior
        preserved when the field is omitted."""
        from docgen.scip_indexers import PythonIndexerAdapter

        _make_venv(tmp_path)
        (tmp_path / '__init__.py').write_text('')

        observed: dict = {'config_content': None}

        def probing_runner(cmd, cwd=None, **kwargs):
            pc = Path(cwd) / 'pyrightconfig.json'
            if pc.exists():
                observed['config_content'] = pc.read_text(encoding='utf-8')
            if '--output' in cmd:
                out = Path(cmd[cmd.index('--output') + 1])
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_bytes(b'\x08\x01ok')
            return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')

        adapter = PythonIndexerAdapter(runner=probing_runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
            # entry_kind omitted; defaults to 'package'
        )

        assert result.success
        # Package-style "." include
        import json as _json
        config_data = _json.loads(observed['config_content'])
        assert '.' in config_data['include']

    def test_existing_pyrightconfig_respected_with_resolver(
        self, tmp_path: Path,
    ) -> None:
        """User-authored pyrightconfig.json wins even when the resolver
        would have written a transient config. No overwrite on either
        path (env_hints or resolver-found)."""
        from docgen.scip_indexers import PythonIndexerAdapter

        existing = tmp_path / 'pyrightconfig.json'
        existing.write_text(
            '{"venvPath": "/user/env", "venv": "myenv"}',
            encoding='utf-8',
        )
        _make_venv(tmp_path)  # resolver would otherwise pick this

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        assert result.success
        # User content preserved, no transient overwrote it
        content = existing.read_text(encoding='utf-8')
        assert '/user/env' in content
        assert 'myenv' in content


# ---------------------------------------------------------------------------
# Phase 2n.b — venv auto-materialization integration
# ---------------------------------------------------------------------------


def _write_uv_project(parent: Path) -> None:
    """Drop a uv-style project (pyproject.toml + uv.lock) at parent."""
    (parent / 'uv.lock').write_text('', encoding='utf-8')
    (parent / 'pyproject.toml').write_text(
        '[project]\nname = "x"\nversion = "0.1.0"\n',
        encoding='utf-8',
    )


class TestMaterializationIntegration:
    """The adapter triggers venv materialization when:

    1. The resolver returns ``source='system'`` (no local venv found)
    2. AND a recognized deps file exists in cwd
    3. AND ``skip_materialize`` is False (default)

    On success, the adapter re-resolves and the new local ``.venv``
    becomes the picked interpreter. On failure (uv missing, network
    error, etc.), the adapter falls through to the original system
    resolution rather than hard-failing — degraded SCIP quality is
    better than no SCIP at all.

    ``materializer`` is dependency-injected for testability, mirroring
    the ``runner`` / ``resolver_which`` pattern.
    """

    def test_system_resolution_with_deps_file_triggers_materialization(
        self, tmp_path: Path,
    ) -> None:
        """When resolver finds only system Python AND cwd has uv.lock,
        adapter calls materialize_venv, which (in this test) creates a
        ``.venv`` directory. Adapter re-resolves and finds the new venv,
        ending at ``resolution_source='cwd-venv'``."""
        from docgen.scip_indexers import PythonIndexerAdapter

        _write_uv_project(tmp_path)
        # No .venv yet — resolver would fall through to system

        materialize_calls: list = []

        def fake_materializer(cmd, *, cwd=None, **kwargs):
            materialize_calls.append({'cmd': list(cmd), 'cwd': cwd})
            # Simulate uv sync creating the venv on success
            venv_bin = Path(cwd) / '.venv' / 'bin'
            venv_bin.mkdir(parents=True)
            python = venv_bin / 'python'
            python.write_text('#!/bin/sh\nexec /usr/bin/python3 "$@"\n')
            python.chmod(0o755)
            return SimpleNamespace(
                returncode=0, stdout=b'', stderr=b'',
            )

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(
            runner=runner,
            materializer=fake_materializer,
            # Force resolver to find a system Python (deterministic
            # across CI / dev machines)
            resolver_which=lambda name: (
                '/usr/bin/python3' if name == 'python3' else None
            ),
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        # Materializer was called
        assert len(materialize_calls) == 1
        # First arg is 'uv' (since uv.lock detected)
        assert materialize_calls[0]['cmd'][0] == 'uv'
        # Re-resolution after materialization picked the new local .venv
        assert result.resolution_source == 'cwd-venv'
        assert result.resolved_interpreter is not None
        assert '.venv' in str(result.resolved_interpreter)

    def test_skip_materialize_flag_disables_materialization(
        self, tmp_path: Path,
    ) -> None:
        """``skip_materialize=True`` opts out — even when conditions are
        met for triggering. Useful for CI where you want to fail fast
        instead of running long-running install commands."""
        from docgen.scip_indexers import PythonIndexerAdapter

        _write_uv_project(tmp_path)

        materialize_calls: list = []

        def fake_materializer(cmd, **kwargs):
            materialize_calls.append(list(cmd))
            return SimpleNamespace(
                returncode=0, stdout=b'', stderr=b'',
            )

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(
            runner=runner,
            materializer=fake_materializer,
            skip_materialize=True,
            resolver_which=lambda name: (
                '/usr/bin/python3' if name == 'python3' else None
            ),
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        # Materializer NOT called
        assert materialize_calls == []
        # Resolution stays at 'system' (no .venv was created)
        assert result.resolution_source == 'system'

    def test_cwd_venv_resolution_skips_materialization(
        self, tmp_path: Path,
    ) -> None:
        """If a local .venv already exists, no materialization needed.
        Resolver returns 'cwd-venv'; adapter doesn't invoke materializer."""
        from docgen.scip_indexers import PythonIndexerAdapter

        # Pre-existing .venv (resolver picks this directly)
        _make_venv(tmp_path)
        # Plus a deps file (would normally trigger materialization)
        _write_uv_project(tmp_path)

        materialize_calls: list = []

        def fake_materializer(cmd, **kwargs):
            materialize_calls.append(list(cmd))
            return SimpleNamespace(
                returncode=0, stdout=b'', stderr=b'',
            )

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(
            runner=runner,
            materializer=fake_materializer,
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        # Materializer NOT called — resolver already had cwd-venv
        assert materialize_calls == []
        assert result.resolution_source == 'cwd-venv'

    def test_no_deps_file_skips_materialization(
        self, tmp_path: Path,
    ) -> None:
        """System resolution + NO deps file → no materialization
        attempted (nothing to install). Adapter proceeds with system
        Python."""
        from docgen.scip_indexers import PythonIndexerAdapter

        # No .venv, no deps file — just empty cwd
        materialize_calls: list = []

        def fake_materializer(cmd, **kwargs):
            materialize_calls.append(list(cmd))
            return SimpleNamespace(
                returncode=0, stdout=b'', stderr=b'',
            )

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(
            runner=runner,
            materializer=fake_materializer,
            resolver_which=lambda name: (
                '/usr/bin/python3' if name == 'python3' else None
            ),
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        # No deps file → no materialization
        assert materialize_calls == []
        assert result.resolution_source == 'system'

    def test_materialization_failure_falls_through_to_system(
        self, tmp_path: Path,
    ) -> None:
        """If materialization fails (binary missing, install error),
        adapter doesn't hard-fail. It continues with the original
        system resolution — degraded SCIP quality is better than none."""
        from docgen.scip_indexers import PythonIndexerAdapter

        _write_uv_project(tmp_path)

        def failing_materializer(cmd, **kwargs):
            raise FileNotFoundError('uv binary not found')

        runner = FakeRunner()
        adapter = PythonIndexerAdapter(
            runner=runner,
            materializer=failing_materializer,
            resolver_which=lambda name: (
                '/usr/bin/python3' if name == 'python3' else None
            ),
        )
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )

        # Adapter didn't hard-fail; scip-python ran with system Python
        assert result.success
        # Resolution remained 'system' (no successful materialization)
        assert result.resolution_source == 'system'
