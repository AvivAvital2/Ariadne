"""Tests for `.conf` (HOCON) file indexing.

HOCON (Typesafe Config) is the standard config format in Scala/JVM projects.
ast-grep doesn't natively support it, so we register `.conf` as a known
extension that gets a `file_index` doc (so the file is discoverable in
Ariadne's search and source_files lookups) but no per-element extraction.

Behavior contracts:
- `.conf` is in `CATALOG_EXTS` → discovery walks it
- `_detect_language` returns `'hocon'` for `.conf` → file_index metadata
  carries the right language tag instead of `'unknown'`
- `extract_elements` returns `[]` for `.conf` → no element-level extraction
  (HOCON-specific syntax would crash a JSON-only parser, so we deliberately
  skip per-element work)
- A real-world HOCON file (e.g. `application.conf` containing
  `activation { pub = [...] }`) gets a file_index doc whose content
  references the file path, making it findable in search.
"""
from __future__ import annotations

from pathlib import Path


def test_conf_is_in_catalog_exts():
    from docgen.catalog_writer import CATALOG_EXTS

    assert '.conf' in CATALOG_EXTS, (
        '.conf is missing from CATALOG_EXTS — Typesafe Config files '
        '(application.conf, reference.conf) will not be indexed and '
        "users won't be able to find them through Ariadne search."
    )


def test_detect_language_returns_hocon_for_conf():
    """A `.conf` file must report language='hocon' so the file_index
    metadata is honest about the format. Without this, file_index docs
    would carry language='unknown' which is a worse search signal."""
    from docgen.catalog_extractor import _detect_language

    p = Path('/tmp/application.conf')
    assert _detect_language(p) == 'hocon'


def test_hocon_is_in_language_literal():
    """The `Language` Literal type must include 'hocon' so static type
    checkers know it's a valid value. Otherwise downstream code that
    annotates `Language` will reject the new value."""
    from typing import get_args

    from docgen.catalog_extractor import Language

    args = get_args(Language)
    assert 'hocon' in args, (
        f"Language Literal does not include 'hocon': got {args}"
    )


def test_extract_elements_produces_hocon_keys(tmp_path: Path):
    """Per-element extraction now runs for HOCON via the Lark parser
    (task #60). Replaces the earlier empty-fallback contract that #59
    introduced. Each leaf key plus the enclosing block becomes a
    `hocon_key` element."""
    from docgen.catalog_extractor import extract_elements

    f = tmp_path / 'application.conf'
    f.write_text(
        'activation {\n'
        '  pub = []\n'
        '  maxCreationDateDaySpan = 30\n'
        '}\n',
        encoding='utf-8',
    )
    elements = extract_elements(f, tmp_path)
    assert len(elements) >= 3, (
        f"expected at least 3 elements (activation block + 2 leaves); "
        f"got {len(elements)}"
    )
    assert all(el.subtype == 'hocon_key' for el in elements)
    assert all(el.language == 'hocon' for el in elements)


def test_iter_catalog_files_includes_conf(tmp_path: Path):
    """`iter_catalog_files` (catalog-sync's discovery path) must
    surface `.conf` files in its walk, otherwise sync would never
    create a file_index doc for them."""
    from docgen.catalog_writer import iter_catalog_files

    (tmp_path / 'src').mkdir()
    (tmp_path / 'src' / 'application.conf').write_text(
        'activation { pub = [] }\n', encoding='utf-8',
    )
    # Drop in a Python file too to confirm the .conf file isn't
    # accidentally pruned by something other than CATALOG_EXTS.
    (tmp_path / 'src' / 'main.py').write_text('x = 1\n', encoding='utf-8')

    files = iter_catalog_files(tmp_path)
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)

    assert 'src/application.conf' in rels, (
        f'.conf file not surfaced by iter_catalog_files: {rels}'
    )


def test_find_catalog_files_includes_conf(tmp_path: Path):
    """`find_catalog_files` (the staleness/improve/coverage path) must
    likewise surface `.conf` files. Without this, `ariadne improve`
    would never see HOCON files as undocumented candidates."""
    from docgen.staleness import find_catalog_files

    (tmp_path / 'engine').mkdir(parents=True)
    (tmp_path / 'engine' / 'src' / 'main' / 'resources').mkdir(parents=True)
    conf = tmp_path / 'engine' / 'src' / 'main' / 'resources' / 'application.conf'
    conf.write_text('activation { pub = [] }\n', encoding='utf-8')

    files = find_catalog_files(tmp_path)
    rels = sorted(str(f.relative_to(tmp_path)) for f in files)
    assert 'engine/src/main/resources/application.conf' in rels
