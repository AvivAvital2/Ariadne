"""Contract for safe YAML mutation (Phase 2j.c).

When the user rejects a suspect during ``ariadne discover --review``,
the suggested exclude pattern needs to land in ``ariadne.yaml`` under
the right ``sources.<name>.exclude`` list. We use ruamel.yaml (new
dependency) to preserve formatting and comments — PyYAML would reflow
the file and lose user annotations.

These tests are RED until ``docgen/yaml_writer.py`` exists.
"""
from __future__ import annotations

from pathlib import Path


def _write_yaml(path: Path, content: str) -> None:
    path.write_text(content, encoding='utf-8')


# ---------------------------------------------------------------------------
# Append-to-existing-list
# ---------------------------------------------------------------------------


class TestAppendsToExistingExcludes:
    def test_appends_pattern_to_existing_list(
        self, tmp_path: Path,
    ) -> None:
        """A source already has ``exclude:`` with one entry; append
        adds a new pattern alongside without removing the existing."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        _write_yaml(yaml_path, '''
sources:
  scalaproject:
    path: /x/scalaproject
    exclude:
      - "**/secrets.json"
''')

        added = append_source_excludes(
            yaml_path, 'scalaproject', ['**/*.min.js'],
        )

        assert added == 1
        # Re-read with PyYAML for assertion (round-trip-safe is tested
        # separately)
        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text())
        excludes = data['sources']['scalaproject']['exclude']
        assert '**/secrets.json' in excludes
        assert '**/*.min.js' in excludes


class TestCreatesExcludeKeyWhenMissing:
    def test_creates_exclude_when_source_lacks_one(
        self, tmp_path: Path,
    ) -> None:
        """Source has no ``exclude:`` key — append creates it."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        _write_yaml(yaml_path, '''
sources:
  scalaproject:
    path: /x/scalaproject
''')

        added = append_source_excludes(
            yaml_path, 'scalaproject', ['vendor/*.min.js'],
        )
        assert added == 1

        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text())
        assert data['sources']['scalaproject']['exclude'] == ['vendor/*.min.js']


class TestSkipsDuplicates:
    def test_doesnt_re_add_existing_pattern(
        self, tmp_path: Path,
    ) -> None:
        """If the pattern is already in the source's exclude list,
        don't add it twice — return 0 indicating nothing was added."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        _write_yaml(yaml_path, '''
sources:
  scalaproject:
    path: /x/scalaproject
    exclude:
      - "**/*.min.js"
''')

        added = append_source_excludes(
            yaml_path, 'scalaproject', ['**/*.min.js'],
        )
        assert added == 0

        # No duplication
        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text())
        excludes = data['sources']['scalaproject']['exclude']
        assert excludes.count('**/*.min.js') == 1


# ---------------------------------------------------------------------------
# Mixed batch — some new, some existing
# ---------------------------------------------------------------------------


class TestMixedBatch:
    def test_returns_count_of_actually_added(
        self, tmp_path: Path,
    ) -> None:
        """If 3 patterns are passed and 1 is already present, return 2
        for actually-added; the existing one is left in place once."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        _write_yaml(yaml_path, '''
sources:
  scalaproject:
    path: /x/scalaproject
    exclude:
      - "**/*.min.js"
''')

        added = append_source_excludes(
            yaml_path, 'scalaproject',
            ['**/*.min.js', 'vendor/*.bundle.js', '**/*.min.css'],
        )
        assert added == 2  # only the two new ones

        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text())
        excludes = data['sources']['scalaproject']['exclude']
        assert '**/*.min.js' in excludes
        assert 'vendor/*.bundle.js' in excludes
        assert '**/*.min.css' in excludes


# ---------------------------------------------------------------------------
# Doesn't disturb other sources / fields
# ---------------------------------------------------------------------------


class TestPreservesOtherContent:
    def test_other_sources_untouched(self, tmp_path: Path) -> None:
        """Modifying source A's exclude list must not affect source B
        or top-level config."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        _write_yaml(yaml_path, '''
default_source: scalaproject
sources:
  scalaproject:
    path: /x/scalaproject
  other:
    path: /y/other
    depends_on:
      - scalaproject
docs_base: ./docs
''')

        append_source_excludes(
            yaml_path, 'scalaproject', ['vendor/*.min.js'],
        )

        import yaml as _yaml
        data = _yaml.safe_load(yaml_path.read_text())

        # scalaproject has the new exclude
        assert data['sources']['scalaproject']['exclude'] == ['vendor/*.min.js']
        # other source untouched
        assert data['sources']['other']['path'] == '/y/other'
        assert data['sources']['other']['depends_on'] == ['scalaproject']
        # top-level untouched
        assert data['default_source'] == 'scalaproject'
        assert data['docs_base'] == './docs'

    def test_preserves_comments_in_yaml(self, tmp_path: Path) -> None:
        """Using ruamel.yaml — comments survive round-trip. PyYAML
        would strip them. This is the whole reason for the new dep."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        original = '''# Top-level comment about ariadne config
default_source: scalaproject
sources:
  scalaproject:  # main monorepo
    path: /x/scalaproject
    # Excludes secrets and vendor noise
    exclude:
      - "**/secrets.json"
'''
        _write_yaml(yaml_path, original)

        append_source_excludes(
            yaml_path, 'scalaproject', ['**/*.min.js'],
        )

        new_content = yaml_path.read_text()
        # Comments preserved
        assert '# Top-level comment about ariadne config' in new_content
        assert '# main monorepo' in new_content
        assert '# Excludes secrets and vendor noise' in new_content


# ---------------------------------------------------------------------------
# Edge cases / errors
# ---------------------------------------------------------------------------


class TestErrors:
    def test_unknown_source_raises(self, tmp_path: Path) -> None:
        """Calling on a source that doesn't exist in the yaml is a
        caller bug — raise rather than silently no-op."""
        import pytest as _pt
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        _write_yaml(yaml_path, '''
sources:
  scalaproject:
    path: /x/scalaproject
''')

        with _pt.raises((KeyError, ValueError)):
            append_source_excludes(
                yaml_path, 'nonexistent', ['vendor/*.min.js'],
            )

    def test_missing_yaml_raises(self, tmp_path: Path) -> None:
        """No ariadne.yaml at the path → raise."""
        import pytest as _pt
        from docgen.yaml_writer import append_source_excludes

        with _pt.raises((FileNotFoundError, OSError)):
            append_source_excludes(
                tmp_path / 'nonexistent.yaml',
                'scalaproject',
                ['vendor/*.min.js'],
            )

    def test_empty_pattern_list_is_noop(self, tmp_path: Path) -> None:
        """Passing ``[]`` means nothing to do — return 0, don't touch
        the file."""
        from docgen.yaml_writer import append_source_excludes

        yaml_path = tmp_path / 'ariadne.yaml'
        original = '''
sources:
  scalaproject:
    path: /x/scalaproject
'''
        _write_yaml(yaml_path, original)

        added = append_source_excludes(yaml_path, 'scalaproject', [])
        assert added == 0
        # File contents unchanged
        assert yaml_path.read_text() == original
