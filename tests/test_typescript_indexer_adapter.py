"""Contract for ``TypescriptIndexerAdapter`` — Phase 2l third slice.

Mirrors the Python adapter's contract: shell out to ``scip-typescript``,
return ``IndexerResult``. tsconfig.json discovery and node_modules
resolution are scip-typescript's concerns; the adapter just runs the
binary in the right cwd.

Vue extractor pre-step is NOT in this slice — it's Phase 2h. A
follow-on commit will detect ``.vue`` files in cwd and run the
extractor script before scip-typescript.
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
            'cmd': list(cmd), 'cwd': cwd, 'env': dict(env) if env else {},
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
    def test_invokes_scip_typescript_index(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner()
        adapter = TypescriptIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert result.success
        cmd = runner.calls[0]['cmd']
        assert cmd[0] == 'scip-typescript'
        assert 'index' in cmd

    def test_output_flag_uses_target_path(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner()
        adapter = TypescriptIndexerAdapter(runner=runner)
        out = tmp_path / 'subdir' / 'out.scip'
        adapter.run(cwd=tmp_path, output=out, env_hints={})

        cmd = runner.calls[0]['cmd']
        assert '--output' in cmd
        assert cmd[cmd.index('--output') + 1] == str(out)

    def test_runs_in_specified_cwd(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner()
        adapter = TypescriptIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path / 'webapp',
            output=tmp_path / 'out.scip',
            env_hints={},
        )
        assert runner.calls[0]['cwd'] == tmp_path / 'webapp'


# ---------------------------------------------------------------------------
# Success / failure
# ---------------------------------------------------------------------------


class TestSuccessFailure:
    def test_returns_success_on_zero_returncode(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner(returncode=0)
        adapter = TypescriptIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is True
        assert (tmp_path / 'out.scip').exists()

    def test_indexer_version_starts_with_scip_typescript(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner()
        adapter = TypescriptIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.indexer_version.startswith('scip-typescript')

    def test_nonzero_returncode_reports_failure(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner(
            returncode=1,
            stderr=b"tsc: error TS2304: Cannot find name 'foo'\n",
        )
        adapter = TypescriptIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert 'tsc' in result.error_message or 'TS2304' in result.error_message

    def test_failure_surfaces_stdout_when_stderr_empty(
        self, tmp_path: Path,
    ) -> None:
        # scip-typescript writes "no files got indexed" to STDOUT with an
        # empty stderr. The adapter must surface that reason rather than
        # the opaque "returned nonzero exit" fallback (which hides why).
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner(
            returncode=1,
            stdout=b'error: no files got indexed. To fix this problem, '
                   b'make sure that the TypeScript projects contain '
                   b'input files or reference other projects.\n',
            stderr=b'',
        )
        adapter = TypescriptIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert 'no files got indexed' in result.error_message
        assert 'returned nonzero exit' not in result.error_message

    def test_binary_not_found_reports_failure(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        runner = FakeRunner(
            raise_on_call=FileNotFoundError(
                "[Errno 2] No such file or directory: 'scip-typescript'",
            ),
        )
        adapter = TypescriptIndexerAdapter(runner=runner)
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success is False
        assert 'scip-typescript' in result.error_message
        # Helpful hint that the user can act on
        assert (
            'install' in result.error_message.lower()
            or 'not found' in result.error_message.lower()
            or 'path' in result.error_message.lower()
        )


# ---------------------------------------------------------------------------
# Vue pre-step (Phase 2h.a) — extractor runs before scip-typescript when
# .vue files are present in cwd
# ---------------------------------------------------------------------------


class FakeVueExtractor:
    """Drop-in for the Vue SFC extractor. Records calls; canned result."""

    def __init__(
        self, *, success: bool = True, error_message: str = '',
    ) -> None:
        self.success = success
        self.error_message = error_message
        self.calls: list[Path] = []
        self.output_paths: list = []

    def extract(self, *, cwd: Path, output_path=None):
        self.calls.append(cwd)
        self.output_paths.append(output_path)
        from docgen.scip_indexers import VueExtractorResult
        return VueExtractorResult(
            success=self.success,
            error_message=self.error_message,
            output_path=str(output_path) if output_path else '',
        )


class TestVuePreStep:
    def test_invokes_vue_extractor_when_vue_files_present(
        self, tmp_path: Path,
    ) -> None:
        """A cwd containing ``.vue`` files triggers the Vue extractor
        pre-step before scip-typescript runs."""
        from docgen.scip_indexers import TypescriptIndexerAdapter

        # Drop a .vue file
        (tmp_path / 'Component.vue').write_text(
            '<script>export default {}</script>', encoding='utf-8',
        )

        runner = FakeRunner()
        extractor = FakeVueExtractor()
        adapter = TypescriptIndexerAdapter(
            runner=runner, vue_extractor=extractor,
        )
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )

        assert result.success
        assert len(extractor.calls) == 1
        assert extractor.calls[0] == tmp_path
        # And scip-typescript ran AFTER the extractor (extractor.calls
        # populated before runner.calls — record-order proves it)
        assert len(runner.calls) == 1

    def test_skips_extractor_when_no_vue_files(
        self, tmp_path: Path,
    ) -> None:
        """A cwd with only .ts/.js files skips the Vue pre-step."""
        from docgen.scip_indexers import TypescriptIndexerAdapter

        (tmp_path / 'app.ts').write_text(
            'export const x = 1', encoding='utf-8',
        )

        runner = FakeRunner()
        extractor = FakeVueExtractor()
        adapter = TypescriptIndexerAdapter(
            runner=runner, vue_extractor=extractor,
        )
        adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )

        # Extractor not called; scip-typescript still ran
        assert extractor.calls == []
        assert len(runner.calls) == 1

    def test_extractor_failure_halts_run(self, tmp_path: Path) -> None:
        """If Vue extraction fails, scip-typescript is not invoked
        and the adapter returns failure with the extractor's error."""
        from docgen.scip_indexers import TypescriptIndexerAdapter

        (tmp_path / 'Component.vue').write_text(
            '<script>export default {}</script>', encoding='utf-8',
        )

        runner = FakeRunner()
        extractor = FakeVueExtractor(
            success=False,
            error_message='vue extractor: parse error in Foo.vue',
        )
        adapter = TypescriptIndexerAdapter(
            runner=runner, vue_extractor=extractor,
        )
        result = adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )

        assert result.success is False
        assert 'parse error' in result.error_message
        # scip-typescript was NOT invoked
        assert runner.calls == []

    def test_finds_vue_files_in_subdirectories(
        self, tmp_path: Path,
    ) -> None:
        """``.vue`` files anywhere in the tree (not just at cwd root)
        trigger the extractor."""
        from docgen.scip_indexers import TypescriptIndexerAdapter

        deep = tmp_path / 'src' / 'components'
        deep.mkdir(parents=True)
        (deep / 'Foo.vue').write_text(
            '<script>export default {}</script>', encoding='utf-8',
        )

        runner = FakeRunner()
        extractor = FakeVueExtractor()
        adapter = TypescriptIndexerAdapter(
            runner=runner, vue_extractor=extractor,
        )
        adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert len(extractor.calls) == 1


# ---------------------------------------------------------------------------
# Zero-config setup (Phase 2l improvement)
# ---------------------------------------------------------------------------


class TestZeroConfigSetup:
    """Per design decision #1 (no pollution of consumed repos beyond
    .ariadne/ + .gitignore), the adapter should handle the no-tsconfig
    case without forcing users to author config files. scip-typescript
    has a built-in ``--infer-tsconfig`` flag for this."""

    def test_passes_infer_tsconfig_when_project_lacks_tsconfig(
        self, tmp_path: Path,
    ) -> None:
        """No tsconfig.json in cwd → --infer-tsconfig added to the
        scip-typescript invocation."""
        from docgen.scip_indexers import TypescriptIndexerAdapter

        # Empty project (no tsconfig.json)
        runner = FakeRunner()
        adapter = TypescriptIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        cmd = runner.calls[0]['cmd']
        assert '--infer-tsconfig' in cmd

    def test_omits_infer_tsconfig_when_project_has_tsconfig(
        self, tmp_path: Path,
    ) -> None:
        """User-provided tsconfig.json is respected — no --infer flag
        added."""
        from docgen.scip_indexers import TypescriptIndexerAdapter

        (tmp_path / 'tsconfig.json').write_text(
            '{"compilerOptions": {"allowJs": true}}',
            encoding='utf-8',
        )
        runner = FakeRunner()
        adapter = TypescriptIndexerAdapter(runner=runner)
        adapter.run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        cmd = runner.calls[0]['cmd']
        assert '--infer-tsconfig' not in cmd
