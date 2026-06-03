"""Tests for multi-language staleness checks.

Pre-fix: ``get_stale_files`` and ``get_undocumented_files`` hardcoded a
``.py`` filter, so even when callers passed Scala/Java files the methods
silently dropped them. This caused ``ariadne generate --source scalaproject``
to process only Python files even with ``catalog_only_generator=True``.

Pre-existing callers already filter their input (e.g. find_catalog_files),
so the staleness layer shouldn't double-filter by language.
"""
from __future__ import annotations

from pathlib import Path


class TestGetStaleFilesIsLanguageAgnostic:
    def test_scala_file_returned_when_undocumented(
        self, tmp_path: Path,
    ) -> None:
        from docgen.staleness import StalenessTracker

        scala = tmp_path / 'X.scala'
        scala.write_text('class X\n', encoding='utf-8')
        py = tmp_path / 'm.py'
        py.write_text('x = 1\n', encoding='utf-8')

        with StalenessTracker(tmp_path / 'stale.db') as tracker:
            stale = tracker.get_stale_files([scala, py], base_path=tmp_path)

        # BOTH languages must come back as stale (undocumented).
        assert scala in stale, (
            f'scala file dropped by .py filter; got: {stale}'
        )
        assert py in stale

    def test_java_file_returned_when_undocumented(
        self, tmp_path: Path,
    ) -> None:
        from docgen.staleness import StalenessTracker

        java = tmp_path / 'X.java'
        java.write_text('class X {}\n', encoding='utf-8')

        with StalenessTracker(tmp_path / 'stale.db') as tracker:
            stale = tracker.get_stale_files([java], base_path=tmp_path)

        assert java in stale


class TestGetUndocumentedFilesIsLanguageAgnostic:
    def test_scala_file_returned_when_no_record(
        self, tmp_path: Path,
    ) -> None:
        from docgen.staleness import StalenessTracker

        scala = tmp_path / 'X.scala'
        scala.write_text('class X\n')

        with StalenessTracker(tmp_path / 'stale.db') as tracker:
            undocumented = tracker.get_undocumented_files(
                [scala], base_path=tmp_path,
            )

        assert scala in undocumented
