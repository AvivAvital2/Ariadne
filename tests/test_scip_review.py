"""Contract for discover review tooling (Phase 2j.c).

The ``--review`` flow on ``ariadne discover`` walks suspect entries
and prompts the user y/N. This test file pins:

- :func:`classify_suspects` — heuristic identifies entries likely
  to be vendored bundles or mass-duplicated dirs.
- :func:`prompt_keep_entry` — interactive prompt with injectable
  input/output for testing.

These tests are RED until ``docgen/scip_review.py`` exists.
"""
from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Heuristic classifier — vendor minified
# ---------------------------------------------------------------------------


class TestVendorMinifiedDetection:
    def test_directory_with_only_min_js_flagged(self) -> None:
        """All markers match ``*.min.js`` → vendor_minified suspect."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/runtime/src/main/resources'),
            markers=(
                Path('/x/runtime/src/main/resources/jquery-2.1.4.min.js'),
                Path('/x/runtime/src/main/resources/jquery.tablesorter.min.js'),
            ),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        assert len(suspects) == 1
        assert suspects[0].reason == 'vendor_minified'
        assert suspects[0].entry == entry

    def test_directory_with_min_css_flagged(self) -> None:
        """``*.min.css`` also counts as vendor minified."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/static'),
            markers=(Path('/x/static/bootstrap.min.css'),),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        assert len(suspects) == 1
        assert suspects[0].reason == 'vendor_minified'

    def test_directory_with_bundle_pattern_flagged(self) -> None:
        """``*.bundle.js`` (webpack convention) counts as vendor-ish."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/dist'),
            markers=(
                Path('/x/dist/main.bundle.js'),
                Path('/x/dist/vendor.bundle.js'),
            ),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        assert len(suspects) == 1
        assert suspects[0].reason == 'vendor_minified'

    def test_directory_with_normal_js_NOT_flagged(self) -> None:
        """A directory containing only un-minified ``.js`` files is
        legitimate code, not vendor noise."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/scripts'),
            markers=(
                Path('/x/scripts/release.js'),
                Path('/x/scripts/setup.js'),
            ),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        # No vendor_minified flag — the heuristic only fires for
        # ALL-minified directories
        vendor = [s for s in suspects if s.reason == 'vendor_minified']
        assert vendor == []

    def test_mixed_min_and_normal_NOT_flagged(self) -> None:
        """A directory with BOTH minified and non-minified files isn't
        a vendor-only directory — could be intentional. Don't flag."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/static'),
            markers=(
                Path('/x/static/app.js'),
                Path('/x/static/jquery.min.js'),  # one outlier
            ),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        vendor = [s for s in suspects if s.reason == 'vendor_minified']
        assert vendor == []


# ---------------------------------------------------------------------------
# Heuristic classifier — mass duplicates
# ---------------------------------------------------------------------------


class TestMassDuplicateDetection:
    def test_ten_or_more_files_same_extension_flagged(self) -> None:
        """Directory with ≥10 files of same extension → mass_duplicate.
        Catches vendored-script collections and large auto-generated
        sets that aren't worth indexing."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        markers = tuple(
            Path(f'/x/testkit/src/main/resources/js/file{i}.js')
            for i in range(15)
        )
        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/testkit/src/main/resources/js'),
            markers=markers,
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        mass = [s for s in suspects if s.reason == 'mass_duplicate']
        assert len(mass) == 1

    def test_few_files_NOT_flagged(self) -> None:
        """A handful of files (<10) is normal — not a duplicate set."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='python',
            cwd=Path('/x/scripts'),
            markers=(
                Path('/x/scripts/a.py'),
                Path('/x/scripts/b.py'),
                Path('/x/scripts/c.py'),
            ),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        mass = [s for s in suspects if s.reason == 'mass_duplicate']
        assert mass == []


# ---------------------------------------------------------------------------
# Non-suspect entries — most discoveries should pass through
# ---------------------------------------------------------------------------


class TestNonSuspectEntries:
    def test_python_package_NOT_flagged(self) -> None:
        """Standard Python package entries pass through cleanly — they
        have an ``__init__.py`` marker and that's a positive signal."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='python',
            cwd=Path('/x/web/pfe'),
            markers=(Path('/x/web/pfe/pyfeatures/__init__.py'),),
            entry_kind='package',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        assert suspects == []

    def test_jvm_entry_NOT_flagged(self) -> None:
        """JVM package entries shouldn't be flagged as suspects either."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='java',
            cwd=Path('/x'),
            markers=(Path('/x/build.sbt'),),
            entry_kind='package',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        assert suspects == []


# ---------------------------------------------------------------------------
# Suggested patterns — what the user can paste into ariadne.yaml
# ---------------------------------------------------------------------------


class TestSuggestedPatterns:
    def test_vendor_minified_suggests_glob_in_cwd(self) -> None:
        """For a vendor-minified entry, the suggested pattern excludes
        all ``*.min.js`` files in that cwd (relative to source root)."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/runtime/src/main/resources'),
            markers=(
                Path('/x/runtime/src/main/resources/jquery-2.1.4.min.js'),
                Path('/x/runtime/src/main/resources/jquery.tablesorter.min.js'),
            ),
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        # Pattern is relative to source_root and globs the .min.js files
        assert suspects[0].suggested_pattern == (
            'runtime/src/main/resources/*.min.js'
        )

    def test_mass_duplicate_suggests_dir_glob(self) -> None:
        """For mass duplicates, suggest globbing the whole directory."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        markers = tuple(
            Path(f'/x/testkit/src/main/resources/js/file{i}.js')
            for i in range(15)
        )
        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/testkit/src/main/resources/js'),
            markers=markers,
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        # Glob covers all .js in that directory
        assert suspects[0].suggested_pattern == (
            'testkit/src/main/resources/js/*.js'
        )

    def test_description_includes_count_and_path(self) -> None:
        """The description shown at the prompt should let the user
        decide without leaving the terminal — count + path + kind."""
        from docgen.scip_discovery import DiscoveryEntry
        from docgen.scip_review import classify_suspects

        markers = tuple(
            Path(f'/x/runtime/src/main/resources/jq-{i}.min.js')
            for i in range(3)
        )
        entry = DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/runtime/src/main/resources'),
            markers=markers,
            entry_kind='scripts',
        )
        suspects = classify_suspects([entry], source_root=Path('/x'))
        desc = suspects[0].description
        assert '3' in desc  # count
        assert 'runtime/src/main/resources' in desc  # path


# ---------------------------------------------------------------------------
# Interactive prompt — y/N flow with injectable input/output
# ---------------------------------------------------------------------------


class _Inputter:
    """Stand-in for builtin input() that returns canned responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.prompts_seen: list[str] = []

    def __call__(self, prompt: str = '') -> str:
        self.prompts_seen.append(prompt)
        if not self._responses:
            return ''
        return self._responses.pop(0)


class _Outputter:
    """Stand-in for print() that captures output."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *args: object, **kwargs: object) -> None:
        self.lines.append(' '.join(str(a) for a in args))


def _make_suspect():
    from docgen.scip_discovery import DiscoveryEntry
    from docgen.scip_review import Suspect
    return Suspect(
        entry=DiscoveryEntry(
            kind='typescript',
            cwd=Path('/x/lib'),
            markers=(Path('/x/lib/jquery.min.js'),),
            entry_kind='scripts',
        ),
        reason='vendor_minified',
        suggested_pattern='lib/*.min.js',
        description='1 minified vendor bundle in lib/',
    )


class TestPromptKeepEntry:
    def test_y_returns_true(self) -> None:
        """``y`` (or ``yes``) means keep the entry."""
        from docgen.scip_review import prompt_keep_entry

        suspect = _make_suspect()
        out = _Outputter()
        result = prompt_keep_entry(
            suspect, input_fn=_Inputter(['y']), output_fn=out,
        )
        assert result is True

    def test_yes_full_word_returns_true(self) -> None:
        from docgen.scip_review import prompt_keep_entry

        suspect = _make_suspect()
        result = prompt_keep_entry(
            suspect, input_fn=_Inputter(['yes']), output_fn=_Outputter(),
        )
        assert result is True

    def test_n_returns_false(self) -> None:
        """``n`` (or ``no``) means exclude."""
        from docgen.scip_review import prompt_keep_entry

        suspect = _make_suspect()
        result = prompt_keep_entry(
            suspect, input_fn=_Inputter(['n']), output_fn=_Outputter(),
        )
        assert result is False

    def test_empty_input_defaults_to_exclude(self) -> None:
        """Hitting Enter without typing → default is exclude (capital
        N in the prompt). Conservative default for noise removal."""
        from docgen.scip_review import prompt_keep_entry

        suspect = _make_suspect()
        result = prompt_keep_entry(
            suspect, input_fn=_Inputter(['']), output_fn=_Outputter(),
        )
        assert result is False

    def test_uppercase_input_normalized(self) -> None:
        """``Y``, ``YES``, ``Yes`` all mean keep."""
        from docgen.scip_review import prompt_keep_entry

        for response in ('Y', 'YES', 'Yes'):
            result = prompt_keep_entry(
                _make_suspect(),
                input_fn=_Inputter([response]),
                output_fn=_Outputter(),
            )
            assert result is True, f'{response!r} should mean keep'

    def test_unrecognized_input_defaults_to_exclude(self) -> None:
        """Anything other than y/yes (case-insensitive) → exclude."""
        from docgen.scip_review import prompt_keep_entry

        for response in ('x', 'maybe', 'idk'):
            result = prompt_keep_entry(
                _make_suspect(),
                input_fn=_Inputter([response]),
                output_fn=_Outputter(),
            )
            assert result is False, (
                f'{response!r} should default to exclude'
            )

    def test_description_printed_to_user(self) -> None:
        """User sees the description before the prompt."""
        from docgen.scip_review import prompt_keep_entry

        suspect = _make_suspect()
        out = _Outputter()
        prompt_keep_entry(
            suspect, input_fn=_Inputter(['n']), output_fn=out,
        )
        all_output = '\n'.join(out.lines)
        assert suspect.description in all_output

    def test_suggested_pattern_printed(self) -> None:
        """User sees the suggested pattern they can paste into yaml."""
        from docgen.scip_review import prompt_keep_entry

        suspect = _make_suspect()
        out = _Outputter()
        prompt_keep_entry(
            suspect, input_fn=_Inputter(['n']), output_fn=out,
        )
        all_output = '\n'.join(out.lines)
        assert suspect.suggested_pattern in all_output
