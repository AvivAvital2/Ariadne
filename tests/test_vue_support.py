"""Design-derived contract tests for end-to-end Vue + TS support.

These tests are written from the approved design (plan: "End-to-end Vue + TS
indexing — SCIP graph + Leiden themes"), NOT from the implementation. Each
class pins one design contract; green means the implementation matches the
design, a red is a genuine design/implementation mismatch.

The design's two legs:

  Leg A — SCIP cross-source graph
    A1  A directory of only .vue files is discovered as a typescript scope
        (so `index_kinds: javascript: scip` gets auto-written).
    A2  scripts/scip/extract-vue-scripts.js turns each SFC's <script> blocks
        into a line-aligned `<name>.vue.script.{js,ts}` companion (line_offset
        0), merges <script>+<script setup> into ONE companion, skips
        template/style-only and src= blocks, writes vue-mapping.json, and
        only deletes companions it marked.
    A3  The TS adapter derives a per-scope vue-mapping path from its .scip
        output, hands it to the extractor, and reports it on IndexerResult.
    A4  `ariadne index` records that path on the manifest entry as
        `vue_mapping` (relative to .ariadne/).

  Leg B — catalog → embeddings → Leiden
    B2/B3  A .vue file is catalog-eligible and its elements are pulled from
           the (vue-mapped) SCIP index under the .vue path.
    B5     A .vue file with no SCIP index yields no elements and never runs
           ast-grep on raw SFC source.
    B4     resolve_index translates companion paths back to .vue using the
           manifest's merged vue_mapping; get_source_scip_config collects it.
    B6     The catalog walk includes .vue and excludes .vue.script.*
           companions (so symbols aren't double-counted).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
EXTRACTOR = REPO_ROOT / 'scripts' / 'scip' / 'extract-vue-scripts.js'


# ===========================================================================
# Leg A1 — discovery routes .vue through the typescript indexer
# ===========================================================================


class TestDiscoveryRoutesVue:
    def test_vue_only_directory_is_a_typescript_scope(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_discovery import discover

        comps = tmp_path / 'src' / 'components'
        comps.mkdir(parents=True)
        (comps / 'Button.vue').write_text(
            '<template><button/></template>\n'
            '<script setup lang="ts">const x = 1</script>\n',
            encoding='utf-8',
        )

        entries = discover(tmp_path)
        ts = [e for e in entries if e.kind == 'typescript']
        assert len(ts) == 1, 'a .vue-only dir must register as a TS scope'
        assert ts[0].cwd == comps


# ===========================================================================
# Leg A1 (incremental) — `ariadne sync` auto-configures a Vue source
#
# Design: because .vue is a SCIP-routable language, a changed .vue file in a
# source that hasn't declared `javascript: scip` makes sync run
# `discover --config-only`, which writes `index_kinds: javascript: scip` to
# ariadne.yaml (then hints to run `ariadne index`). It must NOT re-fire once
# declared.
# ===========================================================================


@pytest.fixture
def _restore_global_config():
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


def _activate_yaml(yaml_path: Path) -> None:
    import config as config_module
    config_module._global_config = config_module.Config(config_path=yaml_path)


class TestSyncAutoConfiguresVue:
    def test_changed_vue_file_triggers_config_only_discover(
        self, tmp_path: Path, _restore_global_config,
    ) -> None:
        from cli.generation import _maybe_auto_discover_for_new_language
        from config import get_config
        from ruamel.yaml import YAML

        src = tmp_path / 'webapp'
        vue = src / 'src' / 'components' / 'Foo.vue'
        vue.parent.mkdir(parents=True)
        vue.write_text(
            '<script setup lang="ts">const x = 1</script>\n', encoding='utf-8',
        )

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n  webapp:\n    path: {src}\n', encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        _maybe_auto_discover_for_new_language(
            get_config(), 'webapp', src, ['src/components/Foo.vue'],
        )

        data = YAML(typ='safe').load(yaml_path.read_text(encoding='utf-8'))
        assert data['sources']['webapp'].get('index_kinds') == {
            'javascript': 'scip',
        }

    def test_already_declared_vue_source_is_a_noop(
        self, tmp_path: Path, _restore_global_config,
    ) -> None:
        from cli.generation import _maybe_auto_discover_for_new_language
        from config import get_config

        src = tmp_path / 'webapp'
        vue = src / 'Foo.vue'
        vue.parent.mkdir(parents=True, exist_ok=True)
        vue.write_text('<script setup>const x = 1</script>\n', encoding='utf-8')

        yaml_path = tmp_path / 'ariadne.yaml'
        yaml_path.write_text(
            f'sources:\n'
            f'  webapp:\n'
            f'    path: {src}\n'
            f'    index_kinds:\n'
            f'      javascript: scip\n'
            f'    scip:\n'
            f'      artifact_path: {src / ".ariadne" / "index.scip"}\n',
            encoding='utf-8',
        )
        _activate_yaml(yaml_path)

        before = yaml_path.stat().st_mtime_ns
        _maybe_auto_discover_for_new_language(
            get_config(), 'webapp', src, ['Foo.vue'],
        )
        assert yaml_path.stat().st_mtime_ns == before, (
            'already-declared javascript:scip must not rewrite yaml for .vue'
        )


# ===========================================================================
# Leg A2 — the Node extractor (real @vue/compiler-sfc)
# ===========================================================================


def _node_modules_with_sfc() -> Path | None:
    """node_modules dir from which @vue/compiler-sfc resolves, or None.
    Honors ARIADNE_TEST_NODE_MODULES, else probes the repo cwd."""
    if shutil.which('node') is None:
        return None
    bases = []
    env_nm = os.environ.get('ARIADNE_TEST_NODE_MODULES')
    if env_nm:
        bases += [Path(env_nm).parent, Path(env_nm)]
    bases.append(REPO_ROOT)
    for base in bases:
        probe = subprocess.run(
            ['node', '-e',
             "try{process.stdout.write(require.resolve('@vue/compiler-sfc',"
             "{paths:[process.argv[1]]}))}catch(e){}", str(base)],
            capture_output=True, text=True,
        )
        if probe.stdout.strip():
            for parent in Path(probe.stdout.strip()).parents:
                if parent.name == 'node_modules':
                    return parent
    return None


@pytest.fixture
def vue_project(tmp_path: Path) -> Path:
    nm = _node_modules_with_sfc()
    if nm is None:
        pytest.skip('@vue/compiler-sfc not resolvable — needs a Vue toolchain')
    (tmp_path / 'node_modules').symlink_to(nm, target_is_directory=True)
    return tmp_path


def _run_extractor(cwd: Path, mapping_out: Path) -> subprocess.CompletedProcess:
    env = dict(os.environ)
    env['ARIADNE_VUE_MAPPING_OUTPUT'] = str(mapping_out)
    return subprocess.run(
        ['node', str(EXTRACTOR)],
        cwd=str(cwd), capture_output=True, text=True, env=env,
    )


def _line_of(text: str, needle: str) -> int:
    for i, line in enumerate(text.splitlines(), start=1):
        if needle in line:
            return i
    raise AssertionError(f'{needle!r} not found in:\n{text}')


class TestExtractorCompanions:
    def test_script_setup_companion_is_line_aligned(self, vue_project: Path) -> None:
        src = vue_project / 'src'
        src.mkdir()
        vue = src / 'Foo.vue'
        vue.write_text(
            '<template>\n'
            '  <button @click="login">Go</button>\n'
            '</template>\n'
            '\n'
            '<script setup lang="ts">\n'
            'export function login(): void {\n'
            '  return\n'
            '}\n'
            '</script>\n',
            encoding='utf-8',
        )
        mapping_out = vue_project / '.ariadne' / 'intermediate' / 'vue-mapping.json'

        result = _run_extractor(vue_project, mapping_out)
        assert result.returncode == 0, result.stderr

        companion = src / 'Foo.vue.script.ts'
        assert companion.exists()
        # Design: line_offset 0 — a symbol sits on the same line in both files.
        assert _line_of(vue.read_text(), 'function login') == \
            _line_of(companion.read_text(), 'function login')

        mapping = json.loads(mapping_out.read_text())
        entry = mapping['src/Foo.vue.script.ts']
        assert entry['original'] == 'src/Foo.vue'
        assert entry['line_offset'] == 0

    def test_dual_blocks_merge_into_single_companion(self, vue_project: Path) -> None:
        vue = vue_project / 'Dual.vue'
        vue.write_text(
            '<template><div/></template>\n'
            '<script lang="ts">\n'
            'export const NAME = "dual"\n'
            '</script>\n'
            '<script setup lang="ts">\n'
            'const count = 0\n'
            '</script>\n',
            encoding='utf-8',
        )
        mapping_out = vue_project / '.ariadne' / 'intermediate' / 'vue-mapping.json'

        result = _run_extractor(vue_project, mapping_out)
        assert result.returncode == 0, result.stderr

        companions = list(vue_project.glob('*.vue.script.*'))
        assert len(companions) == 1, 'both blocks must share ONE companion'
        text = companions[0].read_text()
        orig = vue.read_text()
        assert _line_of(orig, 'NAME') == _line_of(text, 'NAME')
        assert _line_of(orig, 'count') == _line_of(text, 'count')

    def test_template_only_and_external_src_produce_no_companion(
        self, vue_project: Path,
    ) -> None:
        (vue_project / 'NoScript.vue').write_text(
            '<template><p>hi</p></template>\n', encoding='utf-8',
        )
        (vue_project / 'Ext.vue').write_text(
            '<template><p/></template>\n<script src="./x.ts"></script>\n',
            encoding='utf-8',
        )
        mapping_out = vue_project / '.ariadne' / 'intermediate' / 'vue-mapping.json'

        result = _run_extractor(vue_project, mapping_out)
        assert result.returncode == 0, result.stderr
        assert list(vue_project.glob('*.vue.script.*')) == []

    def test_cleanup_only_deletes_marked_companions(self, vue_project: Path) -> None:
        user_file = vue_project / 'Legacy.vue.script.ts'
        user_file.write_text('export const userAuthored = true\n', encoding='utf-8')
        (vue_project / 'Real.vue').write_text(
            '<script setup lang="ts">const x = 1</script>\n', encoding='utf-8',
        )
        mapping_out = vue_project / '.ariadne' / 'intermediate' / 'vue-mapping.json'

        result = _run_extractor(vue_project, mapping_out)
        assert result.returncode == 0, result.stderr
        assert user_file.exists(), 'unmarked user file must not be deleted'
        assert user_file.read_text() == 'export const userAuthored = true\n'


# ===========================================================================
# Leg A3 — TS adapter derives + reports the mapping path
# ===========================================================================


class _RecordingExtractor:
    """Stands in for the Vue extractor; records the output_path it's told."""
    def __init__(self) -> None:
        self.output_paths: list = []

    def extract(self, *, cwd: Path, output_path=None):
        self.output_paths.append(output_path)
        from docgen.scip_indexers import VueExtractorResult
        return VueExtractorResult(success=True, output_path=str(output_path))


class _FakeRunner:
    """Stands in for subprocess.run for scip-typescript — writes the output."""
    def __call__(self, cmd, *, cwd=None, capture_output=True, env=None, **kw):
        if '--output' in cmd:
            out = Path(cmd[cmd.index('--output') + 1])
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_bytes(b'\x08\x01synthetic')
        from types import SimpleNamespace
        return SimpleNamespace(returncode=0, stdout=b'', stderr=b'')


class TestTsAdapterMappingPath:
    def test_derives_per_scope_mapping_path_and_reports_it(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        (tmp_path / 'Widget.vue').write_text(
            '<script setup>const x = 1</script>', encoding='utf-8',
        )
        output = tmp_path / '.ariadne' / 'intermediate' / 'index-webclient-typescript.scip'

        extractor = _RecordingExtractor()
        adapter = TypescriptIndexerAdapter(
            runner=_FakeRunner(), vue_extractor=extractor,
        )
        result = adapter.run(cwd=tmp_path, output=output, env_hints={})

        expected = output.with_name('vue-mapping-webclient-typescript.json')
        # Design: per-scope name derived from the .scip output (no clobbering
        # across multiple TS entries) and reported back for the manifest.
        assert extractor.output_paths == [expected]
        assert result.vue_mapping_path == str(expected)

    def test_no_vue_means_no_mapping_path(self, tmp_path: Path) -> None:
        from docgen.scip_indexers import TypescriptIndexerAdapter

        (tmp_path / 'app.ts').write_text('export const x = 1', encoding='utf-8')
        adapter = TypescriptIndexerAdapter(runner=_FakeRunner())
        result = adapter.run(
            cwd=tmp_path,
            output=tmp_path / 'index-typescript.scip',
            env_hints={},
        )
        assert result.vue_mapping_path == ''


# ===========================================================================
# Leg A4 — cmd_index records vue_mapping on the manifest entry
# ===========================================================================


class _MappingAdapter:
    """Fake indexer that reports a vue_mapping_path under .ariadne/."""
    def run(self, *, cwd: Path, output: Path, env_hints, **kw):
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b'\x08\x01synthetic')
        mapping = output.with_name('vue-mapping-typescript.json')
        from cli.core import IndexerResult
        return IndexerResult(
            success=True, indexer_version='fake/0.1',
            vue_mapping_path=str(mapping),
        )


class _CopyMerger:
    def merge(self, inputs, output: Path) -> bool:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(b''.join(p.read_bytes() for p in inputs))
        return True


class TestCmdIndexWritesVueMapping:
    def test_manifest_entry_gains_vue_mapping(self, tmp_path: Path) -> None:
        import config as config_module
        from cli.core import cmd_index

        # Source with a .vue and a typescript manifest entry.
        (tmp_path / 'Foo.vue').write_text('<script setup>const x=1</script>')
        ariadne = tmp_path / '.ariadne'
        ariadne.mkdir()
        (ariadne / 'manifest.json').write_text(json.dumps({
            'ariadne_version': '1', 'source_name': 'web',
            'indexers': [{'kind': 'typescript', 'cwd': '.', 'markers': []}],
        }), encoding='utf-8')

        yaml = tmp_path / 'ariadne.yaml'
        yaml.write_text(f'sources:\n  web:\n    path: {tmp_path}\n', encoding='utf-8')
        saved = config_module._global_config
        config_module._global_config = config_module.Config(config_path=yaml)
        try:
            rc = cmd_index(
                argparse.Namespace(
                    source='web', all=False, dry_run=False, kind=None, db=None,
                ),
                indexer_registry={'typescript': _MappingAdapter()},
                merger=_CopyMerger(),
            )
        finally:
            config_module._global_config = saved

        assert rc == 0
        entry = json.loads((ariadne / 'manifest.json').read_text())['indexers'][0]
        # Design: vue_mapping recorded relative to .ariadne/ so the loader
        # and resolve_index can find it.
        assert entry['vue_mapping'] == 'intermediate/vue-mapping-typescript.json'


# ===========================================================================
# Leg B2/B3/B5 — .vue catalog extraction routes through SCIP, never ast-grep
# ===========================================================================


def _vue_mapped_index(tmp_path: Path, rel_vue: str, fn: str):
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )
    sym = f'scip-typescript npm web 0.1 {rel_vue}.script#{fn}().'
    return ScipIndex(
        documents=(_ScipDoc(
            relative_path=rel_vue,  # already vue-mapped (the .vue path)
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(2, 0, 4, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(symbol=sym, kind='Function', display_name=fn),),
        ),),
        source_root=tmp_path,
    )


class TestVueCatalogExtraction:
    def test_vue_routes_to_scip_under_vue_path(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import catalog_extractor as ce
        from docgen.scip_config import SourceScipConfig

        f = tmp_path / 'src' / 'Foo.vue'
        f.parent.mkdir(parents=True)
        f.write_text(
            '<script setup lang="ts">\n\nexport function login(){}\n</script>\n',
            encoding='utf-8',
        )

        index = _vue_mapped_index(tmp_path, 'src/Foo.vue', 'login')
        monkeypatch.setattr(
            'docgen.scip_config.resolve_index', lambda cfg, lang: index,
        )
        cfg = SourceScipConfig(
            repo='web', artifact_path=tmp_path / 'idx.scip',
            index_kinds={'javascript': 'scip'},
        )
        elements = ce.extract_elements(f, source_root=tmp_path, source_config=cfg)

        assert len(elements) == 1
        assert 'login' in elements[0].qualified_name

    def test_vue_without_scip_returns_empty_and_never_ast_greps(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import catalog_extractor as ce

        f = tmp_path / 'Widget.vue'
        f.write_text(
            '<template><p>{{m}}</p></template>\n'
            '<script setup>const m = "hi"</script>\n',
            encoding='utf-8',
        )

        def boom(*a, **k):
            raise AssertionError('ast-grep MUST NOT parse raw .vue source')

        monkeypatch.setattr('docgen.catalog_extractor.SgRoot', boom)
        # No source_config → no SCIP index → design says return [], not ast-grep.
        assert ce.extract_elements(f, source_root=tmp_path) == []


# ===========================================================================
# Leg B4 — resolve_index applies the manifest's merged vue mapping
# ===========================================================================


class TestResolveIndexVueMapping:
    def test_get_source_scip_config_merges_manifest_vue_mappings(
        self, tmp_path: Path,
    ) -> None:
        from config import Config

        ariadne = tmp_path / '.ariadne'
        (ariadne / 'intermediate').mkdir(parents=True)
        (ariadne / 'manifest.json').write_text(json.dumps({'indexers': [
            {'kind': 'typescript', 'vue_mapping': 'intermediate/m1.json'},
            {'kind': 'typescript', 'vue_mapping': 'intermediate/m2.json'},
        ]}), encoding='utf-8')
        (ariadne / 'intermediate' / 'm1.json').write_text(json.dumps({
            'a/Foo.vue.script.ts': {'original': 'a/Foo.vue', 'line_offset': 0},
        }), encoding='utf-8')
        (ariadne / 'intermediate' / 'm2.json').write_text(json.dumps({
            'b/Bar.vue.script.ts': {'original': 'b/Bar.vue', 'line_offset': 0},
        }), encoding='utf-8')

        yaml = tmp_path / 'ariadne.yaml'
        yaml.write_text(
            f'sources:\n  web:\n    path: /tmp/web\n'
            f'    index_kinds:\n      javascript: scip\n'
            f'    scip:\n      artifact_path: {ariadne / "index.scip"}\n',
            encoding='utf-8',
        )
        scip_cfg = Config(yaml).get_source_scip_config('web')
        assert scip_cfg is not None
        assert set(scip_cfg.vue_mappings) == {
            'a/Foo.vue.script.ts', 'b/Bar.vue.script.ts',
        }

    def test_resolve_index_translates_companion_paths(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from docgen import scip_config as sc
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        artifact = tmp_path / 'index.scip'
        artifact.write_bytes(b'')

        sym = 'scip-typescript npm web 0.1 src/`Foo.vue.script`/login().'
        companion_index = ScipIndex(
            documents=(_ScipDoc(
                relative_path='src/Foo.vue.script.ts',
                occurrences=(_ScipOccurrence(
                    symbol=sym, range=(4, 0, 6, 0), is_definition=True,
                ),),
                symbols=(_ScipSymbol(symbol=sym, kind='Function'),),
            ),),
            source_root=tmp_path,
        )
        monkeypatch.setattr(
            ScipIndex, 'load',
            classmethod(lambda cls, p, *, repo, max_staleness_days: companion_index),
        )

        cfg = sc.SourceScipConfig(
            repo='web', artifact_path=artifact,
            index_kinds={'javascript': 'scip'},
            vue_mappings={'src/Foo.vue.script.ts': {
                'original': 'src/Foo.vue', 'line_offset': 0,
            }},
        )
        index = sc.resolve_index(cfg, 'javascript')
        assert index is not None
        assert index.documents[0].relative_path == 'src/Foo.vue'


# ===========================================================================
# Leg B6 — catalog walk includes .vue, excludes .vue.script.* companions
# ===========================================================================


def _touch(root: Path, rel: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('x', encoding='utf-8')


class TestCatalogWalkVue:
    def test_iter_catalog_files_includes_vue_skips_companions(
        self, tmp_path: Path,
    ) -> None:
        from docgen.catalog_writer import iter_catalog_files

        _touch(tmp_path, 'Foo.vue')
        _touch(tmp_path, 'Foo.vue.script.ts')
        _touch(tmp_path, 'Bar.vue.script.js')

        names = {f.name for f in iter_catalog_files(tmp_path)}
        assert names == {'Foo.vue'}

    def test_find_catalog_files_includes_vue_skips_companions(
        self, tmp_path: Path,
    ) -> None:
        from docgen.staleness import find_catalog_files

        _touch(tmp_path, 'Foo.vue')
        _touch(tmp_path, 'Foo.vue.script.ts')

        names = {f.name for f in find_catalog_files(tmp_path)}
        assert 'Foo.vue' in names
        assert 'Foo.vue.script.ts' not in names
