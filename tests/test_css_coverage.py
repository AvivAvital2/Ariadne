"""Phase 6 evolutionary-TDD walk for ``.css`` catalog coverage.

CSS is the small remaining coverage gap for multi-tier products that
include CSS alongside HTML / JS / TS (the C source's intended shape).
Like HOCON, CSS has no semantic symbols ast-grep can extract, so the
contract is:

- ``.css`` is in ``CATALOG_EXTS`` → discovery walks it
- ``_detect_language('foo.css')`` returns ``'css'``
- ``Language`` Literal includes ``'css'`` (static-type honesty)
- ``extract_elements(foo.css)`` returns ``[]`` (file_index only)
- Running catalog-sync on a .css file produces a ``file_index`` doc
  with no element docs, searchable by name

This file grows one demand at a time.
"""
from __future__ import annotations

from pathlib import Path


class TestCSSCoverage:
    # ---- T1 -----------------------------------------------------------
    # The smallest possible demand: ``_detect_language`` returns ``'css'``
    # for a ``.css`` path. Without this, .css files fall through to
    # ``None`` and never get a file_index doc.
    def test_t1_detect_language_returns_css(self) -> None:
        from docgen.catalog_extractor import _detect_language

        assert _detect_language(Path('/tmp/styles.css')) == 'css'

    # ---- T2 -----------------------------------------------------------
    # CSS has no semantic symbols to extract — selectors / rules aren't
    # "elements" in the Ariadne sense. ``extract_elements`` must return
    # an empty list rather than try to parse CSS with the JS or HTML
    # grammar (which would emit garbage).
    def test_t2_extract_elements_returns_empty_for_css(
        self, tmp_path: Path,
    ) -> None:
        from docgen.catalog_extractor import extract_elements

        css_file = tmp_path / 'styles.css'
        css_file.write_text(
            '.btn { color: red; }\n.btn:hover { color: blue; }\n',
            encoding='utf-8',
        )
        elements = extract_elements(css_file, source_root=tmp_path)
        assert elements == []

        # T1 still holds.
        from docgen.catalog_extractor import _detect_language
        assert _detect_language(css_file) == 'css'

    # ---- T3 -----------------------------------------------------------
    # ``.css`` must be in ``CATALOG_EXTS`` so the discovery walk
    # (``iter_catalog_files``) actually visits CSS files. Without this,
    # the file would be filtered out before extract_elements ever runs.
    def test_t3_css_is_in_catalog_exts(self) -> None:
        from docgen.catalog_writer import CATALOG_EXTS

        assert '.css' in CATALOG_EXTS, (
            '.css missing from CATALOG_EXTS — CSS files will not be '
            "discovered by iter_catalog_files and won't get file_index "
            'docs even though _detect_language knows about them.'
        )

    # ---- T4 -----------------------------------------------------------
    # The ``Language`` Literal type must include ``'css'`` so static
    # type checkers know it's a valid value. Without this, downstream
    # code annotated ``Language`` would reject the new value.
    def test_t4_css_is_in_language_literal(self) -> None:
        from typing import get_args
        from docgen.catalog_extractor import Language

        args = get_args(Language)
        assert 'css' in args, (
            f"Language Literal does not include 'css': got {args}"
        )
