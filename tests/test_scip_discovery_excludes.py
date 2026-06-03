"""Contract for discover's exclusion behavior (Phase 2j.c).

Two gaps this phase closes:

1. ``DEFAULT_EXCLUDE_POLICY`` in ``config.py`` is missing ``_build``
   (the standard Sphinx output directory). Real-world impact: scalaproject's
   ``sdk/python/docs_api2/_build/html/_static`` surfaced as a TypeScript
   script entry containing Sphinx-generated JS assets.

2. ``SourceConfig.exclude`` (per-source file-glob patterns) is loaded
   into config but never passed to ``discover()``. Adding
   ``**/*.min.js`` to a source's ``exclude:`` in ``ariadne.yaml`` does
   nothing today. Discover gains an ``exclude_patterns`` parameter and
   respects it during the walk — both for source-file cataloging AND
   for marker-file detection.

These tests are RED until the implementation lands.
"""
from __future__ import annotations

from pathlib import Path


def _touch(path: Path, content: str = '') -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


# ---------------------------------------------------------------------------
# Default policy expansion — _build (Sphinx)
# ---------------------------------------------------------------------------


class TestDefaultPolicyExpansion:
    def test_default_policy_includes_underscore_build(self) -> None:
        """Sphinx writes generated docs to ``_build/`` by default. The
        directory contains autogen JS/CSS that must not be indexed."""
        from config import DEFAULT_EXCLUDE_POLICY
        assert '_build' in DEFAULT_EXCLUDE_POLICY


# ---------------------------------------------------------------------------
# discover() honors exclude_patterns
# ---------------------------------------------------------------------------


class TestExcludePatternsHonored:
    def test_pattern_excludes_matching_file(
        self, tmp_path: Path,
    ) -> None:
        """A file matching an exclude pattern is NOT cataloged. With
        only one .py file in a directory and that file excluded, no
        Python script entry is emitted.

        Uses a non-minified ``legacy.js`` as the TS-bearing file so this
        exercises ``exclude_patterns`` itself — minified ``*.min.js`` is
        filtered intrinsically (see test_minified_js_* in
        test_scip_discovery_scripts) and would mask the pattern's effect.
        """
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'orphans' / 'real.py')
        _touch(tmp_path / 'orphans' / 'legacy.js')

        # No exclusion → entries for both Python (real.py) and TS
        # (legacy.js)
        entries_no_exclude = discover(tmp_path)
        kinds = {e.kind for e in entries_no_exclude}
        assert 'python' in kinds
        assert 'typescript' in kinds

        # With **/legacy.js excluded → only Python entry, no TS entry
        entries_excluded = discover(
            tmp_path,
            exclude_patterns=frozenset({'**/legacy.js'}),
        )
        kinds_excluded = {e.kind for e in entries_excluded}
        assert 'python' in kinds_excluded
        assert 'typescript' not in kinds_excluded

    def test_pattern_excludes_marker_file(
        self, tmp_path: Path,
    ) -> None:
        """If an exclude pattern matches a marker file, the directory
        is NOT detected as a package. Useful for excluding vendored
        package.json files (e.g., docs/_static/package.json)."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'docs' / '_static' / 'package.json')

        # No exclusion → docs/_static is a TS package entry
        entries_no = discover(tmp_path)
        ts = [e for e in entries_no if e.kind == 'typescript']
        assert len(ts) == 1
        assert ts[0].cwd == tmp_path / 'docs' / '_static'

        # With docs/_static/package.json excluded → no TS entry
        entries_yes = discover(
            tmp_path,
            exclude_patterns=frozenset({'docs/_static/package.json'}),
        )
        ts2 = [e for e in entries_yes if e.kind == 'typescript']
        assert ts2 == []

    def test_glob_pattern_matches_relative_to_source_root(
        self, tmp_path: Path,
    ) -> None:
        """Patterns are matched against paths relative to source_root.
        ``**/*.min.js`` matches any depth; ``vendor/*.min.js`` only
        matches direct children of vendor/."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'vendor' / 'jquery.min.js')
        _touch(tmp_path / 'src' / 'app.js')

        # **/*.min.js excludes the vendor file but not src/app.js
        entries = discover(
            tmp_path,
            exclude_patterns=frozenset({'**/*.min.js'}),
        )
        ts = [e for e in entries if e.kind == 'typescript']
        # Only src/ surfaces (vendor/ is now empty after exclude)
        assert len(ts) == 1
        assert ts[0].cwd == tmp_path / 'src'

    def test_excluded_files_dont_block_other_languages(
        self, tmp_path: Path,
    ) -> None:
        """Excluding all .js files in a directory must not affect
        Python files in subdirectories of that directory."""
        from docgen.scip_discovery import discover

        # Vendor jquery at parent level
        _touch(tmp_path / 'resources' / 'jquery.min.js')
        # Real Python pocket nested below
        _touch(
            tmp_path / 'resources' / 'pyscripts' / 'tool.py',
            'def f(): pass\n',
        )

        entries = discover(
            tmp_path,
            exclude_patterns=frozenset({'**/*.min.js'}),
        )
        # The vendor TS entry is gone
        assert all(e.kind != 'typescript' for e in entries)
        # The Python script entry survives
        py = [e for e in entries if e.kind == 'python']
        assert len(py) == 1
        assert py[0].cwd == tmp_path / 'resources' / 'pyscripts'

    def test_no_patterns_matches_existing_behavior(
        self, tmp_path: Path,
    ) -> None:
        """Default ``exclude_patterns=frozenset()`` keeps existing
        behavior — no exclusion, all files cataloged."""
        from docgen.scip_discovery import discover

        _touch(tmp_path / 'pkg' / '__init__.py')
        _touch(tmp_path / 'pkg' / 'mod.py')

        entries_default = discover(tmp_path)
        entries_empty = discover(tmp_path, exclude_patterns=frozenset())

        # Same shape regardless of whether the param is omitted or empty
        assert len(entries_default) == len(entries_empty)
