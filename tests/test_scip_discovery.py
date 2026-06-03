"""Tests for the SCIP indexer discovery engine — Phase 2j.

The discovery engine walks a source tree and identifies indexer-relevant
clusters via marker files:

- ``__init__.py`` → Python package
- ``package.json`` → JS/TS project
- ``build.sbt`` / ``*.sbt`` → sbt root
- ``pom.xml`` → Maven root
- ``build.gradle`` / ``build.gradle.kts`` → Gradle root

The output is a list of ``DiscoveryEntry`` records that
``ariadne discover`` writes to ``<source>/.ariadne/manifest.json``;
``ariadne index`` then runs the right indexer in each cwd.

Per design decision #6, ``pyproject.toml`` is **not** used as a Python
marker — it's unreliable (often present in non-Python repos as a
config carrier for tooling like ruff/black).
"""
from __future__ import annotations

from pathlib import Path


def _touch(path: Path, content: str = '') -> None:
    """Helper: create file (and parents) with content."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')


# ---------------------------------------------------------------------------
# Python — __init__.py-driven detection
# ---------------------------------------------------------------------------


class TestPythonDiscovery:
    def test_single_top_level_package_emits_one_entry(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover

        # tmp_path/mypkg/__init__.py — a top-level package
        _touch(tmp_path / 'mypkg' / '__init__.py')
        _touch(tmp_path / 'mypkg' / 'core.py', 'def f(): ...')

        entries = discover(tmp_path)
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        # cwd is the PARENT of the top-level package, so scip-python
        # sees "mypkg" as a package to index
        assert py[0].cwd == tmp_path

    def test_two_sibling_packages_coalesce_to_one_entry(self, tmp_path: Path) -> None:
        """Two top-level packages sharing the same parent directory
        produce ONE DiscoveryEntry rooted at that shared parent — one
        scip-python invocation walks both."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'pkg_a' / '__init__.py')
        _touch(tmp_path / 'pkg_b' / '__init__.py')

        entries = discover(tmp_path)
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        assert py[0].cwd == tmp_path

    def test_nested_subpackage_does_not_emit_separate_entry(
        self, tmp_path: Path,
    ) -> None:
        """A package nested under a package is a sub-package — its
        parent's scip-python invocation already covers it."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'mypkg' / '__init__.py')
        _touch(tmp_path / 'mypkg' / 'sub' / '__init__.py')

        entries = discover(tmp_path)
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        assert py[0].cwd == tmp_path

    def test_packages_at_different_parents_emit_separately(
        self, tmp_path: Path,
    ) -> None:
        """scalaproject-style: Python packages in genuinely separate
        subdirectories of the source root each get their own entry."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'scripts' / 'etl' / '__init__.py')
        _touch(tmp_path / 'services' / 'api' / '__init__.py')

        entries = discover(tmp_path)
        py = [e for e in entries if e.kind == 'python']
        cwds = sorted(str(e.cwd) for e in py)
        assert cwds == sorted([
            str(tmp_path / 'scripts'),
            str(tmp_path / 'services'),
        ])

    def test_pyproject_toml_alone_does_not_trigger_python_entry(
        self, tmp_path: Path,
    ) -> None:
        """Per decision #6, pyproject.toml is NOT a Python marker. A
        repo with only pyproject.toml (and no __init__.py) yields no
        Python entries."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'pyproject.toml', '[tool.ruff]\nline-length = 100\n')

        entries = discover(tmp_path)
        py = [e for e in entries if e.kind == 'python']
        assert py == []


# ---------------------------------------------------------------------------
# JavaScript/TypeScript — package.json-driven
# ---------------------------------------------------------------------------


class TestTypescriptDiscovery:
    def test_root_package_json_emits_typescript_entry(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'package.json', '{"name": "x"}')

        entries = discover(tmp_path)
        ts = [e for e in entries if e.kind == 'typescript']
        assert len(ts) == 1
        assert ts[0].cwd == tmp_path

    def test_subdirectory_package_json_emits_typescript_entry(
        self, tmp_path: Path,
    ) -> None:
        """scalaproject-like: webapp/ subdirectory has its own
        package.json. Discovery emits an entry rooted at that subdir."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'webapp' / 'package.json', '{"name": "x"}')

        entries = discover(tmp_path)
        ts = [e for e in entries if e.kind == 'typescript']
        assert len(ts) == 1
        assert ts[0].cwd == tmp_path / 'webapp'

    def test_nested_package_json_does_not_emit(self, tmp_path: Path) -> None:
        """Two nested package.json files (e.g., a workspace pattern):
        only the outer emits. Nested workspaces are handled by
        scip-typescript itself."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'package.json', '{"name": "outer"}')
        _touch(tmp_path / 'packages' / 'inner' / 'package.json', '{"name": "inner"}')

        entries = discover(tmp_path)
        ts = [e for e in entries if e.kind == 'typescript']
        assert len(ts) == 1
        assert ts[0].cwd == tmp_path


# ---------------------------------------------------------------------------
# Scala/Java — build.sbt / pom.xml / build.gradle
# ---------------------------------------------------------------------------


class TestJvmDiscovery:
    def test_build_sbt_emits_java_entry(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'build.sbt', 'name := "myproject"\n')

        entries = discover(tmp_path)
        java = [e for e in entries if e.kind == 'java']
        assert len(java) == 1
        assert java[0].cwd == tmp_path

    def test_pom_xml_emits_java_entry(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'pom.xml', '<project></project>')

        entries = discover(tmp_path)
        java = [e for e in entries if e.kind == 'java']
        assert len(java) == 1
        assert java[0].cwd == tmp_path

    def test_build_gradle_emits_java_entry(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'build.gradle', '')

        entries = discover(tmp_path)
        java = [e for e in entries if e.kind == 'java']
        assert len(java) == 1
        assert java[0].cwd == tmp_path

    def test_build_gradle_kts_emits_java_entry(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'build.gradle.kts', '')

        entries = discover(tmp_path)
        java = [e for e in entries if e.kind == 'java']
        assert len(java) == 1


# ---------------------------------------------------------------------------
# Polyglot — scalaproject-shaped repo
# ---------------------------------------------------------------------------


class TestPolyglot:
    def test_scala_root_with_python_subdir_and_typescript_webapp(
        self, tmp_path: Path,
    ) -> None:
        """Scalaproject-shaped: sbt at root, webapp with package.json,
        Python scripts in a separate subdir.
        """
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'build.sbt', 'name := "scalaproject"\n')
        _touch(tmp_path / 'webapp' / 'package.json', '{"name": "wc"}')
        _touch(tmp_path / 'scripts' / 'tools' / '__init__.py')

        entries = discover(tmp_path)
        kinds = sorted(e.kind for e in entries)
        assert kinds == ['java', 'python', 'typescript']

        # Spot-check cwds
        java_entry = next(e for e in entries if e.kind == 'java')
        ts_entry = next(e for e in entries if e.kind == 'typescript')
        py_entry = next(e for e in entries if e.kind == 'python')

        assert java_entry.cwd == tmp_path
        assert ts_entry.cwd == tmp_path / 'webapp'
        assert py_entry.cwd == tmp_path / 'scripts'


# ---------------------------------------------------------------------------
# Exclusion behavior
# ---------------------------------------------------------------------------


class TestExclusion:
    def test_excluded_dir_not_walked(self, tmp_path: Path) -> None:
        """A directory in ``exclude_dirs`` is not walked; nothing
        inside it is detected."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'mypkg' / '__init__.py')
        _touch(tmp_path / 'node_modules' / 'pkg' / '__init__.py')

        entries = discover(tmp_path, exclude_dirs=frozenset({'node_modules'}))
        py = [e for e in entries if e.kind == 'python']
        # mypkg detected; node_modules' __init__.py invisible
        assert len(py) == 1
        assert py[0].cwd == tmp_path

    def test_exempt_overrides_exclude(self, tmp_path: Path) -> None:
        """If the same directory is in both exclude_dirs AND
        exempt_dirs, exempt wins (it's force-walked)."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'special_node_modules' / 'pkg' / 'package.json',
               '{"name": "x"}')

        entries = discover(
            tmp_path,
            exclude_dirs=frozenset({'special_node_modules'}),
            exempt_dirs=frozenset({'special_node_modules'}),
        )
        ts = [e for e in entries if e.kind == 'typescript']
        assert len(ts) == 1


# ---------------------------------------------------------------------------
# Empty / degenerate
# ---------------------------------------------------------------------------


class TestDegenerate:
    def test_empty_source_returns_no_entries(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover
        assert discover(tmp_path) == []

    def test_source_with_only_irrelevant_files(self, tmp_path: Path) -> None:
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'README.md', '# Hello')
        _touch(tmp_path / 'config.yaml', 'key: value')

        assert discover(tmp_path) == []


# ---------------------------------------------------------------------------
# Adversarial — edge cases that real implementations actually break on
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_symlink_cycle_does_not_infinite_loop(self, tmp_path: Path) -> None:
        """A symlinked directory pointing back to its parent creates
        an infinite walk if the implementation doesn't guard against
        revisits. Without a check, a recursive walk re-enters the same
        tree forever (or hits Python's recursion limit, which is barely
        better than hanging).
        """
        a = tmp_path / 'a'
        a.mkdir()
        _touch(a / '__init__.py')
        # b symlinks back up to a → walking a then b is a cycle
        (a / 'b').symlink_to(a, target_is_directory=True)

        from docgen.scip_discovery import discover
        # Must return without hanging or raising RecursionError
        entries = discover(tmp_path)
        # The package at `a` should still be detected exactly once
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        assert py[0].cwd == tmp_path

    def test_source_root_does_not_exist(self, tmp_path: Path) -> None:
        """A nonexistent source_root returns an empty list, not a
        crash."""
        from docgen.scip_discovery import discover
        assert discover(tmp_path / 'does_not_exist') == []

    def test_source_root_is_a_file_not_a_directory(self, tmp_path: Path) -> None:
        """source_root pointing at a file (not a directory) returns
        empty."""
        from docgen.scip_discovery import discover
        f = tmp_path / 'a.txt'
        f.write_text('not a directory', encoding='utf-8')
        assert discover(f) == []

    def test_relative_source_root_resolves_to_absolute(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """A relative source_root (like ``Path('.')``) must work the
        same as an absolute one — the implementation needs to resolve
        before walking, otherwise the cwd assertions will mismatch
        depending on where the caller invoked from.
        """
        _touch(tmp_path / 'mypkg' / '__init__.py')

        from docgen.scip_discovery import discover
        monkeypatch.chdir(tmp_path)
        entries = discover(Path('.'))
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        # cwd should be absolute, not "."
        assert py[0].cwd.is_absolute()
        assert py[0].cwd == tmp_path.resolve()

    def test_deeply_nested_python_package(self, tmp_path: Path) -> None:
        """A package five levels deep should still be detected as
        a top-level package (its immediate parent has no __init__.py)."""
        from docgen.scip_discovery import discover

        deep = tmp_path / 'a' / 'b' / 'c' / 'd' / 'mypkg'
        _touch(deep / '__init__.py')

        entries = discover(tmp_path)
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        # cwd is the parent of the top-level package
        assert py[0].cwd == tmp_path / 'a' / 'b' / 'c' / 'd'

    def test_excluded_dir_with_marker_inside_does_not_emit(
        self, tmp_path: Path,
    ) -> None:
        """A package.json living inside an excluded directory should
        not emit, even if that directory contains a normally-walkable
        subdirectory pattern."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'node_modules' / 'pkg' / 'package.json',
               '{"name": "leaked"}')

        entries = discover(tmp_path, exclude_dirs=frozenset({'node_modules'}))
        ts = [e for e in entries if e.kind == 'typescript']
        assert ts == []

    def test_dir_with_both_python_package_and_jvm_build(
        self, tmp_path: Path,
    ) -> None:
        """A single directory that contains both __init__.py AND
        build.sbt should emit one of each (multi-language project at
        the same root). Both indexers will run from the same cwd."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'mypkg' / '__init__.py')
        _touch(tmp_path / 'build.sbt', 'name := "x"')

        entries = discover(tmp_path)
        kinds = sorted(e.kind for e in entries)
        assert kinds == ['java', 'python']

    def test_two_jvm_markers_in_same_dir_emits_one_entry(
        self, tmp_path: Path,
    ) -> None:
        """A directory with both pom.xml AND build.gradle (rare, but
        possible during a build-tool migration) should emit one Java
        entry, not two."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'pom.xml', '<project></project>')
        _touch(tmp_path / 'build.gradle', '')

        entries = discover(tmp_path)
        java = [e for e in entries if e.kind == 'java']
        assert len(java) == 1

    def test_walk_visits_each_directory_at_most_once(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        """The previous symlink-cycle test only passed because macOS's
        ELOOP kicked in at 32 symlink follows — the implementation itself
        had no cycle protection. This test enforces the actual property:
        every directory's resolved path is iterdir'd at most once.

        Without an explicit ``seen`` set keyed on resolved path, the same
        directory gets walked 32+ times before ELOOP saves us — that's
        a real performance bug on large source trees with legitimate
        cycles, and on Linux/Windows where ELOOP behavior differs.
        """
        a = tmp_path / 'a'
        a.mkdir()
        _touch(a / '__init__.py')
        (a / 'b').symlink_to(a, target_is_directory=True)

        # Instrument Path.iterdir to count calls per resolved path
        visit_counts: dict[str, int] = {}
        real_iterdir = Path.iterdir

        def counting_iterdir(self):
            try:
                resolved_key = str(self.resolve())
            except (OSError, RuntimeError):
                resolved_key = str(self)
            visit_counts[resolved_key] = visit_counts.get(resolved_key, 0) + 1
            return real_iterdir(self)

        monkeypatch.setattr(Path, 'iterdir', counting_iterdir)

        from docgen.scip_discovery import discover
        discover(tmp_path)

        duplicates = {p: c for p, c in visit_counts.items() if c > 1}
        assert duplicates == {}, (
            f'directories visited more than once (cycle protection missing):'
            f'\n  {duplicates}'
        )

    def test_returns_deterministic_order(self, tmp_path: Path) -> None:
        """Same input → same output order, run after run. Determinism
        matters because the manifest.json should not churn on every
        re-discovery."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'pkg_z' / '__init__.py')
        _touch(tmp_path / 'pkg_a' / '__init__.py')
        _touch(tmp_path / 'service' / 'package.json', '{"name": "x"}')
        _touch(tmp_path / 'lib' / 'build.sbt', 'name := "x"')

        a = discover(tmp_path)
        b = discover(tmp_path)
        assert a == b
        # Within-kind cwds should be sorted
        ts = [e for e in a if e.kind == 'typescript']
        ts_cwds = [e.cwd for e in ts]
        assert ts_cwds == sorted(ts_cwds)
