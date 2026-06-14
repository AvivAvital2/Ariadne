"""Tests for the ``ariadne index`` CLI command — Phase 2l.

Tests written first as the compass for what the command should do. The
real implementation will follow these contracts.

Scope of this test file: Python adapter only. TypeScript and Java
adapters are subsequent slices.

Architecture:
- ``IndexerAdapter`` is a per-language runner that invokes the
  appropriate scip-X tool. Production has ``PythonIndexerAdapter``,
  ``TypescriptIndexerAdapter``, ``JavaIndexerAdapter``.
- ``cmd_index`` reads ``<source>/.ariadne/manifest.json``, dispatches
  to the right adapter per entry, captures intermediate .scip files,
  then runs scip merge to produce ``<source>/.ariadne/index.scip``.
- Adapters and the merge step are dependency-injected so tests can
  substitute fakes without touching real subprocess calls.

Hard-fail semantics (decision #5): any adapter or merge failure halts
the run with a non-zero exit code.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fake adapter + merge for test injection
# ---------------------------------------------------------------------------


class FakeAdapter:
    """Drop-in replacement for a real IndexerAdapter. Records every
    invocation and produces a fixed synthetic .scip artifact."""

    def __init__(
        self,
        *,
        success: bool = True,
        version: str = 'fake-indexer/0.1',
        scip_bytes: bytes = b'\x08\x01synthetic',
        error_message: str = '',
    ) -> None:
        self.success = success
        self.version = version
        self.scip_bytes = scip_bytes
        self.error_message = error_message
        self.calls: list[dict] = []

    def run(
        self,
        *,
        cwd: Path,
        output: Path,
        env_hints: dict[str, str],
        entry_kind: str = 'package',
        progress_callback=None,
        excludes: tuple[str, ...] = (),
    ):
        # ``entry_kind`` was added to the production python adapter in
        # Phase 2n (per-pocket interpreter resolution). cli_core only
        # passes it when ``kind == 'python'`` (cli_core.py:1239-1241),
        # so the default keeps this stub working for typescript/java
        # tests that don't pass it.
        # ``progress_callback`` was added with the live-progress work;
        # cli_core wires it for the python adapter only.
        # ``excludes`` was added when cli_core started forwarding the
        # source's effective exclusion set (resolved from ariadne.yaml).
        # All three default to no-ops so non-python adapter tests work
        # without changes.
        self.calls.append({
            'cwd': cwd,
            'output': output,
            'env_hints': dict(env_hints),
            'entry_kind': entry_kind,
            'excludes': tuple(excludes),
        })
        if self.success:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(self.scip_bytes)
        from cli.index import IndexerResult
        return IndexerResult(
            success=self.success,
            indexer_version=self.version,
            error_message=self.error_message,
        )


class FakeMerger:
    """Fake for the scip-merge step. Concatenates the input bytes to
    simulate merging."""

    def __init__(self, *, success: bool = True) -> None:
        self.success = success
        self.calls: list[dict] = []

    def merge(self, inputs: list[Path], output: Path) -> bool:
        self.calls.append({'inputs': list(inputs), 'output': output})
        if self.success:
            output.parent.mkdir(parents=True, exist_ok=True)
            payload = b''.join(p.read_bytes() for p in inputs)
            output.write_bytes(payload)
        return self.success


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {'source': None, 'all': False, 'dry_run': False,
                'kind': None, 'db': None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def restore_global_config():
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


def _activate_yaml(yaml_path: Path) -> None:
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)


def _write_manifest(source_root: Path, indexers: list) -> None:
    manifest_dir = source_root / '.ariadne'
    manifest_dir.mkdir(exist_ok=True)
    (manifest_dir / 'manifest.json').write_text(
        json.dumps({
            'ariadne_version': '1',
            'source_name': 'mysrc',
            'indexers': indexers,
        }),
        encoding='utf-8',
    )


@pytest.fixture
def python_source(tmp_path: Path) -> Path:
    """A source layout with a Python package and a manifest declaring
    one python indexer entry."""
    (tmp_path / 'mypkg').mkdir()
    (tmp_path / 'mypkg' / '__init__.py').write_text(
        'def f(): ...', encoding='utf-8',
    )
    _write_manifest(tmp_path, [{
        'kind': 'python',
        'cwd': '.',
        'markers': ['mypkg/__init__.py'],
    }])
    return tmp_path


@pytest.fixture
def configured_for(python_source: Path, tmp_path: Path):
    """Write ariadne.yaml pointing at python_source and activate it."""
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n  mysrc:\n    path: {python_source}\n',
        encoding='utf-8',
    )
    _activate_yaml(yaml_path)
    return yaml_path


# ---------------------------------------------------------------------------
# Adapter dispatch + merge orchestration
# ---------------------------------------------------------------------------


class TestAdapterDispatch:
    def test_runs_python_adapter_for_python_entries(
        self, python_source: Path, configured_for: Path,
    ) -> None:
        from cli.index import cmd_index

        adapter = FakeAdapter()
        merger = FakeMerger()

        rc = cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={'python': adapter},
            merger=merger,
        )
        assert rc == 0
        assert len(adapter.calls) == 1
        # cwd resolved to absolute path
        assert adapter.calls[0]['cwd'].is_absolute()
        # Output path is under .ariadne/intermediate/
        out_path = adapter.calls[0]['output']
        assert out_path.parent.name == 'intermediate'
        assert out_path.parent.parent.name == '.ariadne'

    def test_writes_index_scip_single_intermediate_skips_merger(
        self, python_source: Path, configured_for: Path,
    ) -> None:
        """Single-language project: the lone intermediate is copied
        directly to ``.ariadne/index.scip`` — no merge needed and the
        external ``scip`` CLI isn't required. Pins the optimization
        that single-language projects don't need scip on PATH."""
        from cli.index import cmd_index

        adapter = FakeAdapter()
        merger = FakeMerger()
        cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={'python': adapter},
            merger=merger,
        )

        merged = python_source / '.ariadne' / 'index.scip'
        assert merged.exists()
        # Single intermediate → copy shortcut, merger not invoked.
        assert merger.calls == []

    def test_writes_merged_scip_for_multi_intermediate(
        self, tmp_path: Path,
    ) -> None:
        """Multi-language project (manifest has ≥2 indexer entries):
        the merger combines all intermediates into one
        ``.ariadne/index.scip``. Pairs with the single-intermediate
        test above so a fix that always-or-never invokes the merger
        fails at least one half."""
        from cli.index import cmd_index

        (tmp_path / 'mypkg').mkdir()
        (tmp_path / 'mypkg' / '__init__.py').write_text(
            'def f(): ...', encoding='utf-8',
        )
        # Two entries → two intermediates → merger required.
        _write_manifest(tmp_path, [
            {'kind': 'python', 'cwd': '.',
             'markers': ['mypkg/__init__.py']},
            {'kind': 'java', 'cwd': '.', 'markers': ['build.sbt']},
        ])
        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  mysrc:\n    path: {tmp_path}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        merger = FakeMerger()
        cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={
                'python': FakeAdapter(), 'java': FakeAdapter(),
            },
            merger=merger,
        )

        merged = tmp_path / '.ariadne' / 'index.scip'
        assert merged.exists()
        # Merger called once with the two intermediates.
        assert len(merger.calls) == 1
        assert len(merger.calls[0]['inputs']) == 2
        assert merger.calls[0]['output'] == merged

    def test_manifest_updated_with_scip_path_and_version(
        self, python_source: Path, configured_for: Path,
    ) -> None:
        """After ariadne index, each indexer entry in manifest.json
        gains scip_path + indexed_at + indexer_version. Subsequent
        ariadne sync reads these to load the artifact."""
        from cli.index import cmd_index

        adapter = FakeAdapter(version='scip-python/1.4.1')
        cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={'python': adapter},
            merger=FakeMerger(),
        )

        manifest = json.loads(
            (python_source / '.ariadne' / 'manifest.json').read_text(),
        )
        entry = next(
            e for e in manifest['indexers'] if e['kind'] == 'python'
        )
        assert 'scip_path' in entry
        assert entry['indexer_version'] == 'scip-python/1.4.1'
        assert 'indexed_at' in entry


class TestPerLanguageGroupingOrder:
    """The Indexing phase groups all scopes of a language together and
    runs languages smallest-volume-first — so the user sees one bar per
    language, and quick languages finish before the heavy JVM compile."""

    def test_runs_grouped_by_language_in_volume_order(
        self, tmp_path: Path,
    ) -> None:
        from cli.index import IndexerResult, cmd_index

        # Controlled volumes: python=1, java=2, typescript=4 files.
        (tmp_path / 'py').mkdir()
        (tmp_path / 'py' / 'a.py').write_text('x = 1\n', encoding='utf-8')
        (tmp_path / 'jv').mkdir()
        for n in ('A.java', 'B.java'):
            (tmp_path / 'jv' / n).write_text('class X {}\n', encoding='utf-8')
        (tmp_path / 'ts').mkdir()
        for n in ('a.ts', 'b.ts', 'c.ts'):
            (tmp_path / 'ts' / n).write_text('export const x = 1\n',
                                             encoding='utf-8')
        (tmp_path / 'ts2').mkdir()
        (tmp_path / 'ts2' / 'd.ts').write_text('export const y = 2\n',
                                               encoding='utf-8')

        # Interleaved + out of volume order; TS split across two scopes.
        _write_manifest(tmp_path, [
            {'kind': 'typescript', 'cwd': 'ts', 'markers': []},
            {'kind': 'python', 'cwd': 'py', 'markers': []},
            {'kind': 'typescript', 'cwd': 'ts2', 'markers': []},
            {'kind': 'java', 'cwd': 'jv', 'markers': []},
        ])
        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  mysrc:\n    path: {tmp_path}\n', encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        order: list[str] = []

        class Recorder:
            def __init__(self, kind: str) -> None:
                self.kind = kind

            def run(self, *, cwd, output, env_hints,
                    entry_kind='package', progress_callback=None,
                    excludes=()):
                order.append(self.kind)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_bytes(b'\x08\x01x')
                return IndexerResult(success=True, indexer_version='x/1')

        summary: list = []
        cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={
                'python': Recorder('python'),
                'java': Recorder('java'),
                'typescript': Recorder('typescript'),
            },
            merger=FakeMerger(),
            phase_summary=summary,
        )

        # python(1) → java(2) → typescript(4); both TS scopes contiguous.
        assert order == ['python', 'java', 'typescript', 'typescript']

        # The per-language summary (for the caller's nested ✓ Index block)
        # is one row per language, in the same volume order, with the
        # language's total file count and an elapsed time.
        assert [(s['language'], s['files']) for s in summary] == [
            ('Python', 1), ('Java', 2), ('TypeScript', 4),
        ]
        assert all(
            isinstance(s['seconds'], (int, float)) for s in summary
        )


class TestPolyglotPartialSlice:
    """Phase 2l first slice: only the python adapter exists. A polyglot
    manifest with both python and java entries should run the python
    adapter and skip java with a clear warning, not crash."""

    def test_unknown_kind_skipped_with_warning(
        self, tmp_path: Path,
    ) -> None:
        from cli.index import cmd_index

        (tmp_path / 'pkg').mkdir()
        (tmp_path / 'pkg' / '__init__.py').write_text('', encoding='utf-8')
        _write_manifest(tmp_path, [
            {'kind': 'python', 'cwd': '.',
             'markers': ['pkg/__init__.py']},
            {'kind': 'java', 'cwd': '.',
             'markers': ['build.sbt']},
        ])

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  mysrc:\n    path: {tmp_path}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        py_adapter = FakeAdapter()
        rc = cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={'python': py_adapter},
            merger=FakeMerger(),
        )
        # Python ran; java skipped (no adapter); overall succeeds
        assert rc == 0
        assert len(py_adapter.calls) == 1


# ---------------------------------------------------------------------------
# Hard-fail semantics
# ---------------------------------------------------------------------------


class TestHardFail:
    def test_adapter_failure_returns_nonzero(
        self, python_source: Path, configured_for: Path,
    ) -> None:
        """Per decision #5: any indexer failure halts the run with
        non-zero exit. No partial-success, no fallback."""
        from cli.index import cmd_index

        adapter = FakeAdapter(success=False, error_message='pyright crashed')
        merger = FakeMerger()

        rc = cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={'python': adapter},
            merger=merger,
        )
        assert rc != 0
        # Merge must NOT have been attempted
        assert merger.calls == []

    def test_merge_failure_returns_nonzero(
        self, tmp_path: Path,
    ) -> None:
        """Multi-intermediate path: when the merger reports failure,
        cmd_index exits nonzero. Single-intermediate projects bypass
        the merger entirely (see TestAdapterDispatch) so this contract
        only applies when there are 2+ intermediates."""
        from cli.index import cmd_index

        (tmp_path / 'mypkg').mkdir()
        (tmp_path / 'mypkg' / '__init__.py').write_text(
            'def f(): ...', encoding='utf-8',
        )
        _write_manifest(tmp_path, [
            {'kind': 'python', 'cwd': '.',
             'markers': ['mypkg/__init__.py']},
            {'kind': 'java', 'cwd': '.', 'markers': ['build.sbt']},
        ])
        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  mysrc:\n    path: {tmp_path}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        rc = cmd_index(
            _make_args(source='mysrc'),
            indexer_registry={
                'python': FakeAdapter(), 'java': FakeAdapter(),
            },
            merger=FakeMerger(success=False),
        )
        assert rc != 0


# ---------------------------------------------------------------------------
# --dry-run + --kind flags
# ---------------------------------------------------------------------------


class TestFlags:
    def test_dry_run_does_not_invoke_adapter_or_merge(
        self, python_source: Path, configured_for: Path,
    ) -> None:
        from cli.index import cmd_index

        adapter = FakeAdapter()
        merger = FakeMerger()
        rc = cmd_index(
            _make_args(source='mysrc', dry_run=True),
            indexer_registry={'python': adapter},
            merger=merger,
        )
        assert rc == 0
        assert adapter.calls == []
        assert merger.calls == []
        # No intermediate files, no merged .scip
        assert not (
            python_source / '.ariadne' / 'index.scip'
        ).exists()

    def test_kind_filter_runs_only_matching_adapters(
        self, tmp_path: Path,
    ) -> None:
        """--kind python runs only python entries even when other
        kinds are present in the manifest."""
        from cli.index import cmd_index

        (tmp_path / 'pkg').mkdir()
        (tmp_path / 'pkg' / '__init__.py').write_text('', encoding='utf-8')
        _write_manifest(tmp_path, [
            {'kind': 'python', 'cwd': '.',
             'markers': ['pkg/__init__.py']},
        ])

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  mysrc:\n    path: {tmp_path}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        py_adapter = FakeAdapter()
        rc = cmd_index(
            _make_args(source='mysrc', kind='python'),
            indexer_registry={'python': py_adapter},
            merger=FakeMerger(),
        )
        assert rc == 0
        assert len(py_adapter.calls) == 1
