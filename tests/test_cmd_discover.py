"""Tests for the ``ariadne discover`` CLI command — Phase 2k.

The command walks a source tree, runs the discovery engine, writes
``<source>/.ariadne/manifest.json`` with the detected indexer plan,
and ensures ``.ariadne/`` is in the source's ``.gitignore``.

These tests are the contract: each test articulates a specific
behavioral guarantee the command must satisfy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest


def _touch(path: Path, content: str = '') -> None:
    """Helper: create file (and parents) with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


def _make_args(**kwargs) -> argparse.Namespace:
    """Build argparse.Namespace with discover defaults + overrides."""
    defaults = {'source': None, 'all': False, 'dry_run': False, 'db': None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


@pytest.fixture(autouse=True)
def restore_global_config():
    """Save / restore the config singleton across tests so a test
    swapping in a custom Config doesn't poison the rest of the session.
    ``autouse=True`` applies to every test in the file."""
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


def _activate_yaml(yaml_path: Path) -> None:
    """Replace the global Config singleton with one loaded from the
    given yaml. Direct injection — no env-var indirection."""
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)


@pytest.fixture
def scalaproject_layout(tmp_path: Path) -> Path:
    """Polyglot source: sbt root + JS webapp + Python scripts dir."""
    _touch(tmp_path / 'build.sbt', 'name := "scalaproject"\n')
    _touch(tmp_path / 'webapp' / 'package.json', '{"name": "wc"}')
    _touch(tmp_path / 'scripts' / 'tools' / '__init__.py')
    return tmp_path


@pytest.fixture
def configured_yaml(tmp_path: Path, scalaproject_layout: Path):
    """Write an ariadne.yaml that points at the scalaproject fixture and
    return its path."""
    yaml_path = tmp_path / 'ariadne.yaml'
    yaml_path.write_text(
        f'sources:\n'
        f'  scalaproject:\n'
        f'    path: {scalaproject_layout}\n',
        encoding='utf-8',
    )
    return yaml_path


# ---------------------------------------------------------------------------
# Manifest output
# ---------------------------------------------------------------------------


class TestManifestOutput:
    def test_writes_manifest_for_polyglot_source(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """A discover run produces .ariadne/manifest.json with the
        indexer entries from the discovery engine."""
        from cli.core import cmd_discover

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args(source='scalaproject'))
        assert rc == 0

        manifest_path = scalaproject_layout / '.ariadne' / 'manifest.json'
        assert manifest_path.exists()
        data = json.loads(manifest_path.read_text(encoding='utf-8'))

        assert data['source_name'] == 'scalaproject'
        assert 'indexers' in data
        kinds = sorted(e['kind'] for e in data['indexers'])
        assert kinds == ['java', 'python', 'typescript']

    def test_manifest_paths_are_relative_to_source_root(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """Paths in manifest.json are relative to the source root —
        otherwise the file isn't portable across machines (e.g., when
        the same checkout sits at a different absolute path)."""
        from cli.core import cmd_discover

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))

        manifest_path = scalaproject_layout / '.ariadne' / 'manifest.json'
        data = json.loads(manifest_path.read_text(encoding='utf-8'))

        for entry in data['indexers']:
            cwd = entry['cwd']
            # Must be relative — no leading / or volume marker
            assert not cwd.startswith('/'), (
                f'manifest contains absolute path: {cwd}'
            )

    def test_empty_source_yields_empty_indexers_array(
        self, tmp_path: Path,
    ) -> None:
        """A source with no detectable indexer markers writes a manifest
        with an empty indexers list. Discovery succeeds; the user is
        informed that nothing to index was found."""
        from cli.core import cmd_discover

        # Empty source dir
        empty_src = tmp_path / 'empty_source'
        empty_src.mkdir()

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  empty:\n    path: {empty_src}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        rc = cmd_discover(_make_args(source='empty'))
        assert rc == 0

        manifest = json.loads(
            (empty_src / '.ariadne' / 'manifest.json').read_text(),
        )
        assert manifest['indexers'] == []


# ---------------------------------------------------------------------------
# .gitignore handling
# ---------------------------------------------------------------------------


class TestGitignore:
    def test_creates_gitignore_with_ariadne_line(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """If no .gitignore exists, create one containing ``.ariadne/``.
        """
        from cli.core import cmd_discover

        gitignore = scalaproject_layout / '.gitignore'
        assert not gitignore.exists()

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))

        assert gitignore.exists()
        content = gitignore.read_text(encoding='utf-8')
        assert '.ariadne/' in content.splitlines()

    def test_appends_to_existing_gitignore(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """An existing .gitignore (with other entries) gets the
        ``.ariadne/`` line appended; existing entries preserved."""
        from cli.core import cmd_discover

        gitignore = scalaproject_layout / '.gitignore'
        gitignore.write_text(
            'node_modules/\ndist/\n',
            encoding='utf-8',
        )

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))

        lines = gitignore.read_text(encoding='utf-8').splitlines()
        assert 'node_modules/' in lines
        assert 'dist/' in lines
        assert '.ariadne/' in lines

    def test_does_not_duplicate_existing_ariadne_line(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """If .ariadne/ is already in .gitignore, don't add it again
        on re-discovery."""
        from cli.core import cmd_discover

        gitignore = scalaproject_layout / '.gitignore'
        gitignore.write_text('.ariadne/\n', encoding='utf-8')

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))

        lines = gitignore.read_text(encoding='utf-8').splitlines()
        # Exactly one .ariadne/ line — no duplicate
        assert lines.count('.ariadne/') == 1


# ---------------------------------------------------------------------------
# Flags + error paths
# ---------------------------------------------------------------------------


class TestFlagsAndErrors:
    def test_dry_run_does_not_write_manifest(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """--dry-run shows what WOULD be written but doesn't touch
        the filesystem (no manifest, no .gitignore changes)."""
        from cli.core import cmd_discover

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args(source='scalaproject', dry_run=True))
        assert rc == 0

        manifest = scalaproject_layout / '.ariadne' / 'manifest.json'
        gitignore = scalaproject_layout / '.gitignore'
        assert not manifest.exists()
        assert not gitignore.exists()

    def test_unknown_source_returns_nonzero(
        self, configured_yaml: Path,
    ) -> None:
        """Asking discover to process a source that isn't in
        ariadne.yaml fails with a non-zero exit code."""
        from cli.core import cmd_discover

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args(source='ghost_source'))
        assert rc != 0

    def test_no_source_and_no_default_returns_nonzero(
        self, configured_yaml: Path,
    ) -> None:
        """If --source isn't given and there's no default_source
        in config, fail with a non-zero exit code (don't silently
        process all sources)."""
        from cli.core import cmd_discover

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args())  # no source, no all
        assert rc != 0

    def test_all_processes_every_configured_source(
        self, tmp_path: Path,
    ) -> None:
        """--all walks all configured sources, writing a manifest per
        source."""
        from cli.core import cmd_discover

        src_a = tmp_path / 'src_a'
        src_b = tmp_path / 'src_b'
        _touch(src_a / 'pkg' / '__init__.py')
        _touch(src_b / 'lib' / '__init__.py')

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n'
            f'  a:\n    path: {src_a}\n'
            f'  b:\n    path: {src_b}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        rc = cmd_discover(_make_args(all=True))
        assert rc == 0

        assert (src_a / '.ariadne' / 'manifest.json').exists()
        assert (src_b / '.ariadne' / 'manifest.json').exists()



# ---------------------------------------------------------------------------
# YAML auto-managed block (index_kinds + scip:) — Phase: discover writes config
# ---------------------------------------------------------------------------


class TestYamlAutoManagedBlock:
    def test_writes_index_kinds_for_javascript_when_typescript_detected(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """A scalaproject layout has a webapp/ subdir with package.json
        → typescript kind detected → index_kinds.javascript:scip lands
        in ariadne.yaml after discover."""
        from cli.core import cmd_discover
        from ruamel.yaml import YAML

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args(source='scalaproject'))
        assert rc == 0

        yaml = YAML(typ='safe')
        data = yaml.load(configured_yaml.read_text(encoding='utf-8'))
        index_kinds = data['sources']['scalaproject'].get('index_kinds')
        assert index_kinds == {
            'javascript': 'scip',
            'java': 'scip',
            'scala': 'scip',
        }

    def test_writes_scip_artifact_path_pointing_at_ariadne_index_scip(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """The auto-managed scip block points artifact_path at the
        canonical <source>/.ariadne/index.scip — what cmd_index produces."""
        from cli.core import cmd_discover
        from ruamel.yaml import YAML

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))

        yaml = YAML(typ='safe')
        data = yaml.load(configured_yaml.read_text(encoding='utf-8'))
        scip = data['sources']['scalaproject'].get('scip')
        assert scip is not None
        assert scip['artifact_path'].endswith(
            '.ariadne/index.scip',
        )
        assert scip['max_staleness_days'] == 7

    def test_idempotent_re_run_does_not_rewrite_yaml(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """Re-running discover with no source change leaves the YAML
        mtime untouched. Pairs with the write tests so a fix that
        always-rewrites fails this half."""
        from cli.core import cmd_discover

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))
        first_mtime = configured_yaml.stat().st_mtime_ns

        # Re-run discover
        cmd_discover(_make_args(source='scalaproject'))
        assert configured_yaml.stat().st_mtime_ns == first_mtime

    def test_python_only_source_writes_no_index_kinds_block(
        self, tmp_path: Path,
    ) -> None:
        """A pure-Python source (no .scala / .java / .ts / .js / .vue
        files) gets no index_kinds block — Python catalog stays on
        ast-grep because its qualified_names already align with
        scip-python output."""
        from cli.core import cmd_discover
        from ruamel.yaml import YAML

        src = tmp_path / 'pyproject'
        _touch(src / 'pkg' / '__init__.py')
        _touch(src / 'pkg' / 'core.py', 'def f(): ...')

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  pyproject:\n    path: {src}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        cmd_discover(_make_args(source='pyproject'))

        yaml = YAML(typ='safe')
        data = yaml.load(yaml_path.read_text(encoding='utf-8'))
        src_cfg = data['sources']['pyproject']
        assert 'index_kinds' not in src_cfg, (
            f'pure-Python source should not declare index_kinds; got '
            f"{src_cfg.get('index_kinds')}"
        )

