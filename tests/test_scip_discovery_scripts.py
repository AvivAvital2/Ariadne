"""Contract for script-directory discovery (Phase 2j.b).

Wave 2's ``discover()`` only emitted entries for directories with
package marker files (``__init__.py`` for Python, ``package.json`` for
TypeScript, ``build.sbt``/``pom.xml``/``build.gradle*`` for JVM). On
real polyglot repos like scalaproject, the dominant pattern is *orphan
script directories* — `.py` files dropped under
``src/main/resources/...`` and called from Scala, with no
``__init__.py`` in sight. The marker-only algorithm misses ~60% of
``.py`` files in scalaproject.

This phase extends ``discover()`` to:

1. Walk the tree exactly once. During the walk: classify marker files
   (existing) AND catalog every supported source-file extension per
   directory (new), all driven by the registry in
   ``docgen/scip_languages.py``.

2. Emit two flavors of :class:`DiscoveryEntry`:
   - **package** — current behavior (cwd is the parent of a marker dir,
     or the marker dir itself for TS/JVM)
   - **scripts** — new (cwd is the orphan directory itself), only for
     languages whose ``can_index_standalone=True``. JVM orphans are
     detected for visibility but NOT emitted (scip-java needs a build
     tool; we can't index standalone ``.scala`` files).

3. ``DiscoveryEntry`` gains an ``entry_kind: Literal['package', 'scripts']``
   field (default ``'package'``) so downstream PythonIndexerAdapter
   knows whether to write ``"include": ["."]`` or ``"include": ["./*.py"]``
   in the transient pyrightconfig.

These tests are RED until the implementation lands.
"""
from __future__ import annotations

from pathlib import Path


def _touch(path: Path, content: str = '') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# Script-directory detection — Python
# ---------------------------------------------------------------------------


class TestPythonScriptDirectories:
    def test_orphan_py_directory_emits_script_entry(
        self, tmp_path: Path,
    ) -> None:
        """A directory with ``.py`` files but no ``__init__.py`` emits
        a Python entry whose cwd is the directory itself (not its
        parent), and entry_kind is 'scripts'."""
        from docgen.scip_discovery import discover

        scripts_dir = tmp_path / 'orphan_scripts'
        _touch(scripts_dir / 'foo.py', 'def foo(): pass\n')
        _touch(scripts_dir / 'bar.py', 'def bar(): pass\n')

        entries = discover(tmp_path)
        py_entries = [e for e in entries if e.kind == 'python']
        assert len(py_entries) == 1
        e = py_entries[0]
        assert e.cwd == scripts_dir
        assert e.entry_kind == 'scripts'

    def test_package_directory_still_emits_package_entry(
        self, tmp_path: Path,
    ) -> None:
        """Existing __init__.py-rooted package behavior unchanged.
        cwd is the parent of the top-level package; entry_kind is
        'package'."""
        from docgen.scip_discovery import discover

        pkg = tmp_path / 'mypkg'
        _touch(pkg / '__init__.py')
        _touch(pkg / 'mod.py')

        entries = discover(tmp_path)
        py_entries = [e for e in entries if e.kind == 'python']
        assert len(py_entries) == 1
        e = py_entries[0]
        assert e.cwd == tmp_path
        assert e.entry_kind == 'package'

    def test_files_inside_package_NOT_emitted_as_scripts(
        self, tmp_path: Path,
    ) -> None:
        """A ``.py`` file inside an ``__init__.py``-rooted package is
        already covered by the package entry — must NOT also produce
        a script entry. Otherwise the same module gets indexed twice."""
        from docgen.scip_discovery import discover

        pkg = tmp_path / 'mypkg'
        _touch(pkg / '__init__.py')
        _touch(pkg / 'submodule.py')
        # Sub-directory with .py but NO __init__.py — but it's UNDER the
        # package root, so it's still part of the package's reach
        _touch(pkg / 'helpers' / 'util.py')

        entries = discover(tmp_path)
        py_entries = [e for e in entries if e.kind == 'python']
        # Exactly one entry — the package — covering the whole subtree
        assert len(py_entries) == 1
        assert py_entries[0].entry_kind == 'package'

    def test_orphan_py_at_source_root(self, tmp_path: Path) -> None:
        """A ``.py`` file directly at the source root with no
        ``__init__.py`` emits a script entry with cwd=tmp_path."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'serve_dp.py')

        entries = discover(tmp_path)
        py_entries = [e for e in entries if e.kind == 'python']
        assert len(py_entries) == 1
        assert py_entries[0].cwd == tmp_path
        assert py_entries[0].entry_kind == 'scripts'


# ---------------------------------------------------------------------------
# Script-directory detection — TypeScript/JavaScript
# ---------------------------------------------------------------------------


class TestTypescriptScriptDirectories:
    def test_orphan_ts_directory_emits_script_entry(
        self, tmp_path: Path,
    ) -> None:
        """Directory with ``.ts`` files and no ``package.json`` emits
        TypeScript script entry. scip-typescript handles standalone
        files via ``--infer-tsconfig``."""
        from docgen.scip_discovery import discover

        scripts_dir = tmp_path / 'utility_scripts'
        _touch(scripts_dir / 'release.ts', 'export const x = 1;\n')

        entries = discover(tmp_path)
        ts_entries = [e for e in entries if e.kind == 'typescript']
        assert len(ts_entries) == 1
        e = ts_entries[0]
        assert e.cwd == scripts_dir
        assert e.entry_kind == 'scripts'

    def test_orphan_js_directory_also_emits_typescript_entry(
        self, tmp_path: Path,
    ) -> None:
        """``.js`` files (no ``.ts``, no package.json) still trigger a
        TypeScript entry — scip-typescript handles modern JS too."""
        from docgen.scip_discovery import discover

        scripts_dir = tmp_path / 'js_scripts'
        _touch(scripts_dir / 'hello.js', 'module.exports = {};\n')

        entries = discover(tmp_path)
        ts_entries = [e for e in entries if e.kind == 'typescript']
        assert len(ts_entries) == 1
        assert ts_entries[0].entry_kind == 'scripts'

    def test_package_json_directory_still_emits_package_entry(
        self, tmp_path: Path,
    ) -> None:
        """Existing package.json-rooted behavior unchanged."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'package.json', '{"name": "x"}')
        _touch(tmp_path / 'src' / 'index.ts')

        entries = discover(tmp_path)
        ts_entries = [e for e in entries if e.kind == 'typescript']
        assert len(ts_entries) == 1
        assert ts_entries[0].cwd == tmp_path
        assert ts_entries[0].entry_kind == 'package'


# ---------------------------------------------------------------------------
# JVM orphan handling — detected but NOT emitted
# ---------------------------------------------------------------------------


class TestJvmOrphans:
    def test_orphan_scala_file_NOT_emitted_as_entry(
        self, tmp_path: Path,
    ) -> None:
        """A ``.scala`` file with no build.sbt/pom.xml/build.gradle
        does NOT produce a DiscoveryEntry. scip-java needs a build
        tool to compile; we can't index standalone .scala files.

        Layer C may later track such files as cross-language endpoint
        targets, but they're not first-class indexer entries.
        """
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'standalone' / 'Foo.scala', 'object Foo')

        entries = discover(tmp_path)
        # No JVM entry whatsoever — the orphan dir is silent
        jvm_entries = [e for e in entries if e.kind == 'java']
        assert jvm_entries == []

    def test_orphan_java_file_NOT_emitted(self, tmp_path: Path) -> None:
        """Same rule for ``.java``."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'lib' / 'Helper.java', 'class Helper {}')

        entries = discover(tmp_path)
        jvm_entries = [e for e in entries if e.kind == 'java']
        assert jvm_entries == []

    def test_jvm_with_marker_still_emitted(
        self, tmp_path: Path,
    ) -> None:
        """Sanity: when build.sbt IS present, JVM entry is emitted as
        before. Orphan handling shouldn't break the happy path."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'build.sbt')
        _touch(tmp_path / 'src' / 'Main.scala')

        entries = discover(tmp_path)
        jvm_entries = [e for e in entries if e.kind == 'java']
        assert len(jvm_entries) == 1
        assert jvm_entries[0].entry_kind == 'package'


# ---------------------------------------------------------------------------
# Multi-language directories
# ---------------------------------------------------------------------------


class TestMixedLanguages:
    def test_directory_with_py_and_ts_emits_both_script_entries(
        self, tmp_path: Path,
    ) -> None:
        """A single directory containing both .py and .ts files (no
        markers for either language) emits TWO script entries — one
        per language. Each indexer gets its own pass over the same
        cwd."""
        from docgen.scip_discovery import discover

        mixed = tmp_path / 'glue'
        _touch(mixed / 'tool.py')
        _touch(mixed / 'tool.ts')

        entries = discover(tmp_path)
        script_entries = [
            e for e in entries if e.entry_kind == 'scripts'
        ]
        assert len(script_entries) == 2
        kinds = sorted(e.kind for e in script_entries)
        assert kinds == ['python', 'typescript']
        assert all(e.cwd == mixed for e in script_entries)


# ---------------------------------------------------------------------------
# Adversarial — exclusions, hidden dirs, irrelevant extensions, cycles
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_excluded_directories_dont_surface_as_scripts(
        self, tmp_path: Path,
    ) -> None:
        """``exclude_dirs`` applies to script detection too. A directory
        named ``node_modules`` (or anything in the exclude set) must
        not produce script entries even if it contains .py files."""
        from docgen.scip_discovery import discover

        excluded = tmp_path / 'node_modules'
        _touch(excluded / 'leftover.py')

        entries = discover(
            tmp_path,
            exclude_dirs=frozenset({'node_modules'}),
        )
        assert all(
            'node_modules' not in str(e.cwd) for e in entries
        )

    def test_non_source_extensions_ignored(
        self, tmp_path: Path,
    ) -> None:
        """``.pyc`` and ``.pyi`` aren't source — they're cache/stub.
        A directory containing only those should NOT emit a Python
        script entry."""
        from docgen.scip_discovery import discover

        cache_dir = tmp_path / 'cache'
        _touch(cache_dir / 'foo.pyc')
        _touch(cache_dir / 'foo.pyi')

        entries = discover(tmp_path)
        py_entries = [e for e in entries if e.kind == 'python']
        assert py_entries == []

    def test_minified_js_does_not_trigger_typescript_scope(
        self, tmp_path: Path,
    ) -> None:
        """Vendored minified bundles (``*.min.js``) are built
        third-party artifacts, not source. A directory containing ONLY
        them must NOT become a TypeScript scope — scip-typescript rejects
        such a scope with "no files got indexed" and aborts the whole
        index (real-world case: a ``highcharts``/``jquery`` vendor dir).

        A real ``.ts``/``.js`` file alongside a bundle still emits a
        scope: the bundle is ignored, not a veto."""
        from docgen.scip_discovery import discover

        # Vendored-only dir → no scope at all.
        vendor = tmp_path / 'viz' / 'highcharts'
        _touch(vendor / 'highcharts-4.1.9.min.js', '/*min*/')
        _touch(vendor / 'jquery-1.9.0.min.js', '/*min*/')

        entries = discover(tmp_path)
        assert all('highcharts' not in str(e.cwd) for e in entries)

        # Real source + a bundle in the same dir → scope still emitted.
        app = tmp_path / 'app'
        _touch(app / 'index.ts', 'export const x = 1;\n')
        _touch(app / 'vendor.min.js', '/*min*/')

        entries = discover(tmp_path)
        app_ts = [
            e for e in entries
            if e.kind == 'typescript' and e.cwd == app
        ]
        assert len(app_ts) == 1

    def test_empty_directories_emit_nothing(
        self, tmp_path: Path,
    ) -> None:
        """Source root with no source files at all → no entries."""
        from docgen.scip_discovery import discover

        # Just create empty subdirectories
        (tmp_path / 'a' / 'b').mkdir(parents=True)
        (tmp_path / 'c').mkdir()

        entries = discover(tmp_path)
        assert entries == []

    def test_sibling_script_directories_each_emit(
        self, tmp_path: Path,
    ) -> None:
        """Two unrelated script directories at the same depth each
        emit their own entry — no coalescing across script siblings.
        Reflects that orphan dirs have no implicit package
        relationship."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'tools' / 'a.py')
        _touch(tmp_path / 'utils' / 'b.py')

        entries = discover(tmp_path)
        py_entries = sorted(
            (e for e in entries if e.kind == 'python'),
            key=lambda e: e.cwd,
        )
        assert len(py_entries) == 2
        assert py_entries[0].cwd == tmp_path / 'tools'
        assert py_entries[1].cwd == tmp_path / 'utils'

    def test_deeply_nested_orphan_directories_each_emit(
        self, tmp_path: Path,
    ) -> None:
        """Each ``.py``-bearing directory at ANY depth emits its own
        entry when no ``__init__.py`` is found in the chain — no
        magical coalescing under a common ancestor."""
        from docgen.scip_discovery import discover

        # Two cousins; each is its own orphan
        _touch(tmp_path / 'a' / 'b' / 'c' / 'x.py')
        _touch(tmp_path / 'a' / 'b' / 'd' / 'y.py')

        entries = discover(tmp_path)
        py_cwds = sorted(
            str(e.cwd) for e in entries if e.kind == 'python'
        )
        assert py_cwds == [
            str(tmp_path / 'a' / 'b' / 'c'),
            str(tmp_path / 'a' / 'b' / 'd'),
        ]

    def test_symlink_cycle_protection_holds(
        self, tmp_path: Path,
    ) -> None:
        """Existing symlink-cycle protection (Phase 2j.a) must keep
        working after the script-directory extension. A loop must NOT
        cause a script directory to be visited twice and emitted twice."""
        import os
        from docgen.scip_discovery import discover

        scripts_dir = tmp_path / 'scripts'
        _touch(scripts_dir / 'foo.py')
        # Create a symlink loop: scripts/back → tmp_path
        try:
            os.symlink(tmp_path, scripts_dir / 'back')
        except (OSError, NotImplementedError):
            import pytest as _pt
            _pt.skip('symlinks not supported on this platform')

        entries = discover(tmp_path)
        py_entries = [e for e in entries if e.kind == 'python']
        # Exactly one entry for scripts/, no duplicate from the cycle
        assert len(py_entries) == 1
        assert py_entries[0].cwd == scripts_dir


# ---------------------------------------------------------------------------
# Regression — scalaproject manifest shape
# ---------------------------------------------------------------------------


class TestScalaprojectShape:
    def test_simulates_scalaproject_polyglot_layout(
        self, tmp_path: Path,
    ) -> None:
        """End-to-end shape test on a synthetic scalaproject-like layout:
        package Python (pfe), script Python (azureml), package TS
        (webapp), JVM with build.sbt at root, plus orphan .scala
        (must NOT emit). Asserts the right entry counts and kinds."""
        from docgen.scip_discovery import discover

        # JVM root
        _touch(tmp_path / 'build.sbt')
        _touch(tmp_path / 'src' / 'main' / 'scala' / 'Main.scala')

        # Python package (web/pfe/pyfeatures)
        _touch(tmp_path / 'web' / 'pfe' / 'pyfeatures' / '__init__.py')
        _touch(tmp_path / 'web' / 'pfe' / 'pyfeatures' / 'mod.py')

        # Python orphan scripts (azureml/local_scripts)
        azureml = (
            tmp_path / 'models' / 'train_impl' / 'src' / 'main'
            / 'resources' / 'azureml' / 'local_scripts'
        )
        _touch(azureml / 'train.py')
        _touch(azureml / 'common.py')

        # TS package (webapp)
        _touch(tmp_path / 'webapp' / 'package.json', '{"name": "x"}')
        _touch(tmp_path / 'webapp' / 'src' / 'app.ts')

        # Top-level orphan script
        _touch(tmp_path / 'serve_dp.py')

        entries = discover(tmp_path)

        # Python: one package (web/pfe), one azureml script dir, one
        # top-level script (cwd=tmp_path). Use set comparison rather
        # than indexed access — paths sort lexicographically by string,
        # which gives a non-obvious ordering when one path is a prefix
        # of the other.
        py_entries = [e for e in entries if e.kind == 'python']
        assert len(py_entries) == 3

        package_py = [e for e in py_entries if e.entry_kind == 'package']
        assert len(package_py) == 1
        assert package_py[0].cwd == tmp_path / 'web' / 'pfe'

        scripts_py = [e for e in py_entries if e.entry_kind == 'scripts']
        assert len(scripts_py) == 2
        script_cwds = {e.cwd for e in scripts_py}
        assert script_cwds == {azureml, tmp_path}

        # TS: one package, no scripts (only the package.json-rooted dir)
        ts_entries = [e for e in entries if e.kind == 'typescript']
        assert len(ts_entries) == 1
        assert ts_entries[0].entry_kind == 'package'

        # JVM: one entry (build.sbt at root); orphan .scala files outside
        # the build root are NOT emitted (this layout has them inside,
        # so build.sbt covers them)
        jvm_entries = [e for e in entries if e.kind == 'java']
        assert len(jvm_entries) == 1
        assert jvm_entries[0].entry_kind == 'package'
