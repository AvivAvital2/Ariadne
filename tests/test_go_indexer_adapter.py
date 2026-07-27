"""Contract for ``GoIndexerAdapter`` — the scip-go integration.

Mirrors the Python/TypeScript/Java adapters: shell out to ``scip-go`` in the
module directory (rooted at ``go.mod``), return an ``IndexerResult``. scip-go
type-checks via ``go/packages`` and emits a ``.scip`` — no build-tool
orchestration (unlike scip-java), so this adapter is the simplest of the four.
Also asserts the language-registry wiring, since that's what makes ``go`` a
groundable Spool language with zero edits to discovery / the grounding gate.
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
        write_output: bool = True,
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.scip_bytes = scip_bytes
        self.raise_on_call = raise_on_call
        self.write_output = write_output
        self.calls: list[dict] = []

    def __call__(self, cmd, *, cwd=None, capture_output=True, env=None,
                 **kwargs):
        self.calls.append({'cmd': list(cmd), 'cwd': cwd, 'env': env})
        if self.raise_on_call:
            raise self.raise_on_call
        if self.returncode == 0 and self.write_output and '--output' in cmd:
            out_path = Path(cmd[cmd.index('--output') + 1])
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(self.scip_bytes)
        return SimpleNamespace(
            returncode=self.returncode, stdout=self.stdout, stderr=self.stderr,
        )


# ---------------------------------------------------------------------------
# Command shape
# ---------------------------------------------------------------------------


class TestCommandShape:
    def test_invokes_scip_go(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import GoIndexerAdapter

        runner = FakeRunner()
        result = GoIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'out.scip', env_hints={},
        )
        assert result.success
        assert runner.calls[0]['cmd'][0] == 'scip-go'

    def test_output_flag_uses_target_path(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import GoIndexerAdapter

        runner = FakeRunner()
        out = tmp_path / 'sub' / 'out.scip'
        GoIndexerAdapter(runner=runner).run(cwd=tmp_path, output=out,
                                            env_hints={})
        cmd = runner.calls[0]['cmd']
        assert '--output' in cmd
        assert cmd[cmd.index('--output') + 1] == str(out)

    def test_runs_in_specified_cwd(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import GoIndexerAdapter

        runner = FakeRunner()
        module = tmp_path / 'module_root'
        GoIndexerAdapter(runner=runner).run(
            cwd=module, output=tmp_path / 'o.scip', env_hints={},
        )
        assert runner.calls[0]['cwd'] == module


# ---------------------------------------------------------------------------
# Success / failure
# ---------------------------------------------------------------------------


class TestSuccessFailure:
    def test_success_on_zero_and_output_written(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import GoIndexerAdapter

        result = GoIndexerAdapter(runner=FakeRunner(returncode=0)).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={},
        )
        assert result.success is True
        assert result.indexer_version.startswith('scip-go')
        assert (tmp_path / 'o.scip').exists()

    def test_nonzero_returncode_is_failure(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import GoIndexerAdapter

        runner = FakeRunner(returncode=1, stderr=b'go: build failed in ./foo\n')
        result = GoIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={},
        )
        assert result.success is False
        assert 'build failed' in result.error_message or 'foo' in result.error_message

    def test_zero_exit_but_no_output_is_failure(self, tmp_path: Path) -> None:
        # scip-go can exit 0 yet emit no index (the spark-java trap): treat a
        # missing .scip as failure so the merge isn't fed a phantom path.
        from docgen.scip_indexers import GoIndexerAdapter

        runner = FakeRunner(returncode=0, write_output=False)
        result = GoIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={},
        )
        assert result.success is False

    def test_binary_not_found_gives_install_hint(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import GoIndexerAdapter

        runner = FakeRunner(raise_on_call=FileNotFoundError("'scip-go'"))
        result = GoIndexerAdapter(runner=runner).run(
            cwd=tmp_path, output=tmp_path / 'o.scip', env_hints={},
        )
        assert result.success is False
        assert 'scip-go' in result.error_message
        assert 'go install' in result.error_message.lower()


# ---------------------------------------------------------------------------
# Language-registry wiring — what makes `go` a groundable Spool language
# ---------------------------------------------------------------------------


class TestGoLanguageRegistration:
    def test_go_is_in_the_scip_registry(self) -> None:
        from docgen.scip_languages import LANGUAGES
        go = next((lang for lang in LANGUAGES if lang.name == 'go'), None)
        assert go is not None
        assert go.indexer_kind == 'go'
        # scip-go needs a module (go.mod) to type-check — no standalone mode.
        assert go.can_index_standalone is False

    def test_go_extension_and_marker_lookups(self) -> None:
        from docgen.scip_languages import _EXT_TO_LANG, _MARKER_TO_LANG
        assert _EXT_TO_LANG['.go'].indexer_kind == 'go'
        assert _MARKER_TO_LANG['go.mod'].name == 'go'

    def test_go_is_scip_eligible_for_grounding_gate(self) -> None:
        from spools import is_scip_eligible
        # the Spool grounding gate reads this — go must now pass without
        # --allow-ungrounded.
        assert is_scip_eligible('go') is True
        assert is_scip_eligible('.go') is True

    def test_default_registry_dispatches_go(self) -> None:
        from cli.index import _default_indexer_registry
        from docgen.scip_indexers import GoIndexerAdapter
        assert isinstance(_default_indexer_registry()['go'], GoIndexerAdapter)
