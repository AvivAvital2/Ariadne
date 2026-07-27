"""Slice (a) of the Spool plugin: ``spools:`` enablement + manifest gating.

Evolutionary test — grows demand by demand per IMPLEMENT.md. Synthetic
fixtures only: fake spool names, fake runtimes, tmp-path stores.
Design: designs/spool-environment-plugin.md §9 (manifest) · §18.2 (pin) ·
§18.6.4 (config-static enablement).
"""
import textwrap

import pytest

from config import Config, ConfigError
from library import Library
from spools import (
    SpoolError,
    SpoolManifest,
    SpoolSetting,
    enabled_spools,
    load_yaml_mapping,
    resolve_spools,
)


class _StubConfig:
    """Minimal config double: enabled_spools only calls ``to_dict()``. Lets us
    exercise enabled_spools' own input-validation contract directly, without
    routing through Config (which also guards the same shapes at load)."""

    def __init__(self, data):
        self._data = data

    def to_dict(self):
        return self._data


def _config(tmp_path, yaml_body):
    path = tmp_path / 'ariadne.yaml'
    path.write_text(textwrap.dedent(yaml_body))
    return Config(config_path=path)


class TestSpoolEnablement:
    def test_spool_enablement_lifecycle(self, tmp_path):
        # Demand 1 — parse: a `spools:` mapping exposes the enabled view;
        # `false` values and an absent key contribute nothing.
        cfg = _config(tmp_path, '''
            spools:
              fakebricks: true
              dormant: false
        ''')
        assert enabled_spools(cfg) == {'fakebricks': SpoolSetting()}

        cfg_no_spools = _config(tmp_path, '''
            docs_base: ./docs
        ''')
        assert enabled_spools(cfg_no_spools) == {}

        # Demand 2 — manifest schema: a synthetic manifest round-trips
        # through the loader; a manifest missing `target_runtime` is
        # rejected loudly (SpoolError naming the field), never soft-skipped.
        manifest_dir = tmp_path / 'spool-cache' / 'fakebricks'
        manifest_dir.mkdir(parents=True)
        (manifest_dir / 'manifest.yaml').write_text(textwrap.dedent('''
            environment: fakebricks
            version: '1.0.0'
            target_runtime: fake-17.3
            certified_docs: [docs/]
            checksum: abc123
        '''))
        manifest = SpoolManifest.load(manifest_dir / 'manifest.yaml')
        assert manifest == SpoolManifest(
            environment='fakebricks',
            version='1.0.0',
            target_runtime='fake-17.3',
            certified_docs=('docs/',),
            checksum='abc123',
        )

        with pytest.raises(SpoolError) as excinfo:
            SpoolManifest.from_dict({
                'environment': 'fakebricks',
                'version': '1.0.0',
                'checksum': 'abc123',
            })
        assert 'target_runtime' in str(excinfo.value)

        # CRIT-3c — a type-invalid field (valid YAML, but embedding_dim is
        # not a number, or certified_docs is a scalar) fails CLOSED as a
        # SpoolError — never a raw ValueError/TypeError that would escape the
        # resolve_spools guard and crash the query path.
        with pytest.raises(SpoolError):
            SpoolManifest.from_dict({
                'environment': 'fakebricks',
                'version': '1.0.0',
                'target_runtime': 'fake-17.3',
                'checksum': 'abc123',
                'embedding_dim': 'notanumber',
            })
        with pytest.raises(SpoolError):
            SpoolManifest.from_dict({
                'environment': 'fakebricks',
                'version': '1.0.0',
                'target_runtime': 'fake-17.3',
                'checksum': 'abc123',
                'certified_docs': 5,
            })

        # Demand 3 — fail-closed pin: the project's declared runtime vs the
        # cached pack's target_runtime mismatch refuses the load, names BOTH
        # versions, and registers nothing.
        cache_dir = tmp_path / 'spool-cache'
        cfg_pinned = _config(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-16.0
        ''')
        assert enabled_spools(cfg_pinned) == {
            'fakebricks': SpoolSetting(runtime='fake-16.0'),
        }
        resolution = resolve_spools(cfg_pinned, cache_dir=cache_dir)
        assert resolution.registered == {}
        (gap,) = resolution.gaps
        assert gap.spool == 'fakebricks'
        assert gap.reason == 'runtime-mismatch'
        assert 'fake-16.0' in gap.message
        assert 'fake-17.3' in gap.message

        # Demand 4 — honest gap: an enabled spool with no cached pack is a
        # structured "fetch it first" outcome, not a crash; nothing registers.
        cfg_absent = _config(tmp_path, '''
            spools:
              ghostspool: true
        ''')
        resolution = resolve_spools(cfg_absent, cache_dir=cache_dir)
        assert resolution.registered == {}
        (gap,) = resolution.gaps
        assert gap.spool == 'ghostspool'
        assert gap.reason == 'missing-pack'
        assert 'fetch' in gap.message.lower()

        # Demand 5 — happy path: a matching pin registers with kind='spool'
        # and appears in the scope-source set; no gaps.
        cfg_match = _config(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-17.3
        ''')
        resolution = resolve_spools(cfg_match, cache_dir=cache_dir)
        assert resolution.gaps == ()
        registration = resolution.registered['fakebricks']
        assert registration.kind == 'spool'
        assert registration.manifest.target_runtime == 'fake-17.3'
        assert resolution.scope_sources() == frozenset({'spool:fakebricks'})

        # Demand 6 (H1) — an UNPINNED enable is refused fail-closed. Without a
        # runtime pin the spool would accept ANY signed version (a substitution/
        # downgrade vector once packs are signed), so it does not register and
        # stays out of the query scope until the runtime is pinned; the gap
        # names the pack's runtime to guide the fix.
        cfg_unpinned = _config(tmp_path, '''
            spools:
              fakebricks: true
        ''')
        resolution = resolve_spools(cfg_unpinned, cache_dir=cache_dir)
        assert resolution.registered == {}
        (gap,) = resolution.gaps
        assert gap.spool == 'fakebricks'
        assert gap.reason == 'runtime-unpinned'
        assert 'fake-17.3' in gap.message
        assert resolution.scope_sources() == frozenset()

        # CRIT-3a — a BROKEN spool is REJECTED, never fatal: a corrupt
        # cached manifest becomes a structured gap; nothing registers.
        (cache_dir / 'fakebricks' / 'manifest.yaml').write_text('{{{{ not yaml')
        resolution = resolve_spools(cfg_unpinned, cache_dir=cache_dir)
        assert resolution.registered == {}
        (gap,) = resolution.gaps
        assert gap.reason == 'corrupt-pack'
        assert 'reinstall' in gap.message.lower()

        # CRIT-3c (integration) — a cached manifest that is valid YAML but
        # has a type-invalid field must ALSO degrade to a corrupt-pack gap,
        # never crash resolve_spools (which runs on every scoped query).
        (cache_dir / 'fakebricks' / 'manifest.yaml').write_text(textwrap.dedent('''
            environment: fakebricks
            version: '1.0.0'
            target_runtime: fake-17.3
            checksum: abc123
            embedding_dim: notanumber
        '''))
        resolution = resolve_spools(cfg_unpinned, cache_dir=cache_dir)
        assert resolution.registered == {}
        (gap,) = resolution.gaps
        assert gap.reason == 'corrupt-pack'

        # CRIT-3b — the natural config typo (list instead of mapping) is
        # rejected LOUDLY at config load, where the user can act — not at
        # query time.
        with pytest.raises(ConfigError) as excinfo:
            _config(tmp_path, '''
                spools: [fakebricks]
            ''')
        assert 'spools' in str(excinfo.value)
        assert 'mapping' in str(excinfo.value)


class TestSpoolCli:
    def test_reserved_spool_namespace_rejected(self, tmp_path):
        # CRIT-9 / HIGH-2: the 'spool:' source namespace is reserved for
        # installed packs. It must be un-nameable via source add / config —
        # else uninstall_pack would delete the user's own docs and the
        # origin fence would silently hide them.
        # Load path: a hand-edited / shared ariadne.yaml naming a spool:
        # source fails loudly at load, where the user can act.
        with pytest.raises(ConfigError) as excinfo:
            _config(tmp_path, '''
                sources:
                  "spool:databricks":
                    path: /x
            ''')
        assert 'spool:' in str(excinfo.value)
        assert 'reserved' in str(excinfo.value).lower()

        # Write path (source add / programmatic) is refused too.
        cfg = _config(tmp_path, '''
            sources:
              mylib:
                path: /x
        ''')
        with pytest.raises(ConfigError) as excinfo2:
            cfg.set_source_config('spool:databricks', path='/x')
        assert 'spool:' in str(excinfo2.value)
        # A real name still works — the guard is prefix-scoped, not blanket.
        assert cfg.set_source_config('databricks', path='/x') is True
        # A config with no `sources` mapping at all still validates — the
        # reserved-name scan is a no-op when there is nothing to scan.
        _config(tmp_path, 'spools:\n  databricks: true\n')

    def test_spools_status_command(self, tmp_path, monkeypatch, capsys):
        # Demand 6 — the `ariadne spools` status surface: registered spools
        # and gaps print with their messages; exit 1 iff gaps exist; the
        # command is registered in the real parser.
        from cli.main import create_parser
        import cli.spools_cmd as spools_cmd

        cache_dir = tmp_path / 'spool-cache'
        manifest_dir = cache_dir / 'fakebricks'
        manifest_dir.mkdir(parents=True)
        (manifest_dir / 'manifest.yaml').write_text(textwrap.dedent('''
            environment: fakebricks
            version: '1.0.0'
            target_runtime: fake-17.3
            checksum: abc123
        '''))
        cfg = _config(tmp_path, '''
            spools:
              fakebricks:
                runtime: fake-17.3
              ghostspool:
                runtime: fake-17.3
        ''')
        monkeypatch.setattr(spools_cmd, 'get_config', lambda: cfg)

        parser = create_parser()
        args = parser.parse_args(['spools', '--cache-dir', str(cache_dir)])
        assert args.command == 'spools'

        exit_code = spools_cmd.HANDLERS['spools'](args)
        out = capsys.readouterr().out
        assert 'fakebricks' in out
        assert 'fake-17.3' in out          # registered spool shows its runtime
        assert 'ghostspool' in out
        assert 'fetch' in out.lower()      # the honest-gap message surfaces
        assert exit_code == 1              # gaps present -> loud exit

        # gap resolved -> exit 0
        ghost_dir = cache_dir / 'ghostspool'
        ghost_dir.mkdir()
        (ghost_dir / 'manifest.yaml').write_text(textwrap.dedent('''
            environment: ghostspool
            version: '0.1'
            target_runtime: fake-17.3
            checksum: def456
        '''))
        exit_code = spools_cmd.HANDLERS['spools'](args)
        assert exit_code == 0

        # Demand (b2-5) — build/install subactions on the same command;
        # bare `spools` remains the status view (asserted above).
        repo = tmp_path / 'fakepack-repo'
        (repo / 'docs').mkdir(parents=True)
        cfg2 = _config(tmp_path, f'''
            sources:
              fakepack:
                path: {repo}
            spools:
              fakepack: true
        ''')
        monkeypatch.setattr(spools_cmd, 'get_config', lambda: cfg2)
        with Library(cfg2.db_path) as lib:
            lib.add_document(
                'explanation', 'pack me', 'body',
                source_files=[str(repo / 'docs' / 'a.md')],
                metadata={'provenance': 'human-doc'},
                source_name='fakepack',
            )
        out_zip = tmp_path / 'fakepack.zip'
        args = parser.parse_args([
            'spools', 'build', '--source', 'fakepack', '--version', '1.0',
            '--runtime', 'fake-17.3', '--certify', 'docs/',
            '--out', str(out_zip),
        ])
        assert spools_cmd.HANDLERS['spools'](args) == 0
        assert out_zip.exists()

        install_cache = tmp_path / 'cli-cache'
        args = parser.parse_args([
            'spools', '--cache-dir', str(install_cache),
            'install', str(out_zip),
        ])
        assert spools_cmd.HANDLERS['spools'](args) == 0
        assert (install_cache / 'fakepack' / 'manifest.yaml').exists()
        with Library(cfg2.db_path) as lib:
            (doc,) = [
                d for d in lib.list_documents()
                if d.source_name == 'fakepack'
            ]
            # the certified tag round-tripped build -> install -> store
            assert doc.metadata['provenance'] == 'official'


def test_enabled_spools_parses_projects(tmp_path):
    # A spool associates to a LIST of projects it cross-checks. enabled_spools
    # surfaces that list on the setting; absent (or `true`) = empty, meaning no
    # cross-check yet. Covers both projects branches + the disable path.
    cfg = _config(tmp_path, '''
        spools:
          databricks:
            runtime: dbr17.3-lts
            projects: [demo-spark-proj, other-spark-proj]
          delta:
            runtime: some-rt
          terraform: true
          dormant:
            enabled: false
    ''')
    settings = enabled_spools(cfg)
    assert settings['databricks'].runtime == 'dbr17.3-lts'
    assert settings['databricks'].projects == ('demo-spark-proj', 'other-spark-proj')
    assert settings['delta'].projects == ()      # dict without `projects` -> empty
    assert settings['terraform'].projects == ()  # `true` -> empty
    assert 'dormant' not in settings             # `enabled: false` -> excluded


def test_load_yaml_mapping_edges(tmp_path):
    # The shared spool loader: an empty file is an empty mapping; a non-mapping
    # (list/scalar) is refused loudly with the caller's error class.
    empty = tmp_path / 'empty.yaml'
    empty.write_text('')
    assert load_yaml_mapping(empty, SpoolError) == {}

    not_a_map = tmp_path / 'list.yaml'
    not_a_map.write_text('- a\n- b\n')
    with pytest.raises(SpoolError):
        load_yaml_mapping(not_a_map, SpoolError)


def test_enabled_spools_validates_its_input():
    # Defense-in-depth: enabled_spools rejects a malformed `spools:` mapping
    # loudly on its own (fail-loud, never silently skip), independent of the
    # Config-load guard that also catches these shapes earlier.
    with pytest.raises(SpoolError):
        enabled_spools(_StubConfig({'spools': ['fakebricks']}))   # not a mapping
    with pytest.raises(SpoolError):
        enabled_spools(_StubConfig({'spools': {'fakebricks': 5}}))  # bad value


def test_recipe_runtime_components_read(tmp_path):
    # Slice 3: the recipe's runtime→component map feeds the pack build; a
    # missing/malformed recipe yields {} (availability stays honest-None,
    # never a crash or a guessed version). YAML scalars coerce to str
    # (4.0 would otherwise parse as a float).
    from cli.spools_cmd import _recipe_runtime_components
    recipe = tmp_path / 'spools.yaml'
    recipe.write_text(
        'runtime: fake-17.3\n'
        'runtime_components:\n'
        '  spark: 4.0.0\n'
        '  sdk: 0.121.0\n'
    )
    assert _recipe_runtime_components(recipe) == {
        'spark': '4.0.0', 'sdk': '0.121.0'}
    assert _recipe_runtime_components(tmp_path / 'missing.yaml') == {}
    bad = tmp_path / 'bad.yaml'
    bad.write_text('{{{ not yaml')
    assert _recipe_runtime_components(bad) == {}
    not_map = tmp_path / 'notmap.yaml'
    not_map.write_text('runtime_components: [a, b]\n')
    assert _recipe_runtime_components(not_map) == {}


def test_recipe_corpus_shas_and_taxonomy_read(tmp_path):
    # Slice 3 rebuild fidelity: a manually rebuilt pack must carry the same
    # provenance a create-built pack does — the recipe's TOFU-pinned corpus
    # shas and its advisory taxonomy. Missing/malformed reads stay tolerant
    # (empty), never a crash.
    from cli.spools_cmd import _recipe_corpus_shas, _recipe_taxonomy
    recipe = tmp_path / 'spools.yaml'
    recipe.write_text(
        'runtime: fake-17.3\n'
        'taxonomy: [serialization, parallelism]\n'
        'corpus:\n'
        '  core:\n'
        '    url: https://example.invalid/core\n'
        '    tag: v1.0\n'
        '    sha: abc123\n'
        '  pinless:\n'
        '    url: https://example.invalid/pinless\n'
        '    tag: v2.0\n'
    )
    assert _recipe_corpus_shas(recipe) == {'core': 'abc123'}
    assert _recipe_taxonomy(recipe) == ('serialization', 'parallelism')
    assert _recipe_corpus_shas(tmp_path / 'missing.yaml') == {}
    assert _recipe_taxonomy(tmp_path / 'missing.yaml') == ()
    bad = tmp_path / 'bad.yaml'
    bad.write_text('{{{ not yaml')
    assert _recipe_corpus_shas(bad) == {}
    assert _recipe_taxonomy(bad) == ()
