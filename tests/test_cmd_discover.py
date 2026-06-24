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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args(source='ghost_source'))
        assert rc != 0

    def test_no_source_and_no_default_returns_nonzero(
        self, configured_yaml: Path,
    ) -> None:
        """If --source isn't given and there's no default_source
        in config, fail with a non-zero exit code (don't silently
        process all sources)."""
        from cli.index import cmd_discover

        _activate_yaml(configured_yaml)
        rc = cmd_discover(_make_args())  # no source, no all
        assert rc != 0

    def test_all_processes_every_configured_source(
        self, tmp_path: Path,
    ) -> None:
        """--all walks all configured sources, writing a manifest per
        source."""
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover
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
        from cli.index import cmd_discover
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
        from cli.index import cmd_discover

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
        from cli.index import cmd_discover
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


# ---------------------------------------------------------------------------
# Re-discover must not orphan a built index (manifest desync regression)
# ---------------------------------------------------------------------------


def _simulate_index(source_root: Path) -> dict:
    """Stand in for ``ariadne index``: backfill ``scip_path`` (+ metadata) on
    every manifest entry and create the referenced ``.scip`` artifacts. Returns
    the written manifest dict."""
    manifest_path = source_root / '.ariadne' / 'manifest.json'
    data = json.loads(manifest_path.read_text(encoding='utf-8'))
    (source_root / '.ariadne' / 'intermediate').mkdir(parents=True, exist_ok=True)
    for i, entry in enumerate(data['indexers']):
        rel = f'intermediate/idx{i}.scip'
        (source_root / '.ariadne' / rel).write_bytes(b'\x08\x01')
        entry['scip_path'] = rel
        entry['indexed_at'] = '2026-06-01T00:00:00+00:00'
        entry['indexer_version'] = 'scip-x/1.0'
    manifest_path.write_text(json.dumps(data), encoding='utf-8')
    return data


class TestRediscoverPreservesBuiltArtifacts:
    """A bare ``discover`` rewrites ``manifest.json`` from topology alone. If it
    drops the ``scip_path`` that ``index`` backfilled, the built ``.scip`` is
    orphaned and the source goes invisible to every SCIP feature
    (callers/callees/impact/trace/data-model) until a full re-index. Discover
    must carry those refs forward when the artifact still exists."""

    def test_rediscover_preserves_scip_path_when_artifact_exists(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        from cli.index import cmd_discover

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))  # initial plan
        _simulate_index(scalaproject_layout)             # index backfills refs

        cmd_discover(_make_args(source='scalaproject'))  # RE-discover

        data = json.loads(
            (scalaproject_layout / '.ariadne' / 'manifest.json')
            .read_text(encoding='utf-8'))
        assert data['indexers'], data
        assert all(e.get('scip_path') for e in data['indexers']), (
            'discover orphaned a built index by dropping scip_path: '
            f'{data["indexers"]}'
        )
        # carried alongside scip_path, keyed by (kind, cwd, entry_kind)
        assert all(e.get('indexed_at') for e in data['indexers'])
        assert all(e.get('indexer_version') for e in data['indexers'])

    def test_rediscover_drops_scip_path_when_artifact_missing(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """If a referenced .scip no longer exists, the carry must NOT keep a
        dangling scip_path — that entry is honestly reported as unbuilt while
        the entries whose artifacts survive stay preserved."""
        from cli.index import cmd_discover

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))
        data = _simulate_index(scalaproject_layout)
        victim = data['indexers'][0]
        (scalaproject_layout / '.ariadne' / victim['scip_path']).unlink()

        cmd_discover(_make_args(source='scalaproject'))

        after = json.loads(
            (scalaproject_layout / '.ariadne' / 'manifest.json')
            .read_text(encoding='utf-8'))
        by_id = {(e['kind'], e['cwd']): e for e in after['indexers']}
        # artifact gone -> no dangling scip_path claimed
        assert 'scip_path' not in by_id[(victim['kind'], victim['cwd'])]
        # the others (artifacts intact) are still carried
        survivors = [
            e for e in after['indexers']
            if (e['kind'], e['cwd']) != (victim['kind'], victim['cwd'])
        ]
        assert survivors and all(e.get('scip_path') for e in survivors)

    def test_rediscover_carries_only_built_entries(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """Mixed prior — one built entry, one discovered-but-unbuilt entry, one
        absent from the prior entirely. Re-discover carries only the built one;
        no scip_path is fabricated for the unbuilt or the brand-new entry."""
        from cli.index import cmd_discover

        _activate_yaml(configured_yaml)
        cmd_discover(_make_args(source='scalaproject'))
        manifest_path = scalaproject_layout / '.ariadne' / 'manifest.json'
        fresh = json.loads(manifest_path.read_text(encoding='utf-8'))

        inter = scalaproject_layout / '.ariadne' / 'intermediate'
        inter.mkdir(parents=True, exist_ok=True)
        (inter / 'java.scip').write_bytes(b'\x08\x01')
        prior_entries = []
        for entry in fresh['indexers']:
            if entry['kind'] == 'java':            # built (artifact exists)
                prior_entries.append({**entry, 'scip_path': 'intermediate/java.scip'})
            elif entry['kind'] == 'typescript':    # discovered but unbuilt
                prior_entries.append({**entry})
            # python: omitted from prior entirely (new on re-discover)
        manifest_path.write_text(
            json.dumps({**fresh, 'indexers': prior_entries}), encoding='utf-8')

        cmd_discover(_make_args(source='scalaproject'))

        after = {e['kind']: e for e in json.loads(
            manifest_path.read_text(encoding='utf-8'))['indexers']}
        assert after['java'].get('scip_path') == 'intermediate/java.scip'
        assert 'scip_path' not in after['typescript']  # matched but unbuilt
        assert 'scip_path' not in after['python']       # no prior match

    def test_rediscover_survives_corrupt_prior_manifest(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """A corrupt/unreadable prior manifest must not crash discover — it
        falls back to a clean fresh write (nothing to preserve)."""
        from cli.index import cmd_discover

        _activate_yaml(configured_yaml)
        ariadne_dir = scalaproject_layout / '.ariadne'
        ariadne_dir.mkdir(parents=True, exist_ok=True)
        (ariadne_dir / 'manifest.json').write_text('{ not: json', encoding='utf-8')

        rc = cmd_discover(_make_args(source='scalaproject'))
        assert rc == 0
        data = json.loads((ariadne_dir / 'manifest.json').read_text(encoding='utf-8'))
        assert sorted(e['kind'] for e in data['indexers']) == [
            'java', 'python', 'typescript']

    def test_rediscover_empty_source_is_a_noop(self, tmp_path: Path) -> None:
        """Re-discovering an empty source (prior manifest exists, zero fresh
        indexers) carries nothing and doesn't crash — the loop is empty."""
        from cli.index import cmd_discover

        empty = tmp_path / 'empty_src'
        empty.mkdir()
        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  empty:\n    path: {empty}\n', encoding='utf-8')
        _activate_yaml(yaml_path)

        cmd_discover(_make_args(source='empty'))        # writes {indexers: []}
        rc = cmd_discover(_make_args(source='empty'))   # re-discover over prior
        assert rc == 0
        data = json.loads(
            (empty / '.ariadne' / 'manifest.json').read_text(encoding='utf-8'))
        assert data['indexers'] == []

    def test_run_discover_also_preserves_built_scip_path(
        self, scalaproject_layout: Path, configured_yaml: Path,
    ) -> None:
        """The MCP/onboard path (``run_discover``) shares the guarantee — it was
        the writer that orphaned a multi-package source's index, so it must carry scip_path too."""
        import config as config_module
        from cli.index import run_discover

        _activate_yaml(configured_yaml)
        cfg = config_module._global_config
        run_discover(cfg, 'scalaproject')
        _simulate_index(scalaproject_layout)
        run_discover(cfg, 'scalaproject')

        data = json.loads(
            (scalaproject_layout / '.ariadne' / 'manifest.json')
            .read_text(encoding='utf-8'))
        assert data['indexers'] and all(
            e.get('scip_path') for e in data['indexers']), data

