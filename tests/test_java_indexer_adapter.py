"""Contract for ``JavaIndexerAdapter`` — Phase 2l fourth slice.

Mirrors Python/TypeScript: shell out to ``scip-java``, return
``IndexerResult``. Build-tool detection (sbt/Maven/Gradle) is
scip-java's concern; the adapter just runs the binary in cwd.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace


class FakeRunner:
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

    def __call__(self, cmd, *, cwd=None, capture_output=True, env=None,
                 **kwargs):
        self.calls.append({
            'cmd': list(cmd), 'cwd': cwd,
        })
        if self.raise_on_call:
            raise self.raise_on_call
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
    def test_invokes_scip_java_index(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success
        cmd = runner.calls[0]['cmd']
        assert cmd[0] == 'scip-java'
        assert 'index' in cmd

    def test_output_flag_uses_target_path(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        out = tmp_path / 'subdir' / 'out.scip'
        adapter.run(cwd=tmp_path, output=out, env_hints={})

        cmd = runner.calls[0]['cmd']
        assert '--output' in cmd
        assert cmd[cmd.index('--output') + 1] == str(out)

    def test_runs_in_specified_cwd(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path / 'project_root',
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert runner.calls[0]['cwd'] == tmp_path / 'project_root'


# ---------------------------------------------------------------------------
# Success / failure
# ---------------------------------------------------------------------------


class TestSuccessFailure:
    def test_returns_success_on_zero_returncode(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(returncode=0)
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is True
        assert (tmp_path / 'out.scip').exists()

    def test_indexer_version_starts_with_scip_java(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner()
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.indexer_version.startswith('scip-java')

    def test_nonzero_returncode_reports_failure(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(
            returncode=1,
            stderr=b'sbt: compilation failed in subproject foo\n',
        )
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert (
            'sbt' in result.error_message
            or 'compilation' in result.error_message
        )

    def test_binary_not_found_reports_failure(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import JavaIndexerAdapter

        runner = FakeRunner(
            raise_on_call=FileNotFoundError(
                "[Errno 2] No such file or directory: 'scip-java'",
            ),
        )
        adapter = JavaIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert 'scip-java' in result.error_message
        # Helpful hint pointing the user to the install path
        assert (
            'install' in result.error_message.lower()
            or 'not found' in result.error_message.lower()
            or 'coursier' in result.error_message.lower()
            or 'path' in result.error_message.lower()
        )
