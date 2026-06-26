"""Repair + prevention for the export→import round-trip that turned catalog
``location`` dicts into Python-repr strings.

- ``Library.repair_stringified_locations`` heals already-corrupted rows in
  place (lossless ``ast.literal_eval`` back to a dict).
- ``import_from_markdown`` must no longer corrupt a dict ``location`` on the
  way in (the symmetric parser fix).
- ``ariadne migrate --fix-locations`` wires the repair to the CLI.

All fixtures use synthetic data (src1 / pkg.mod.* / pkg/mod.py).
"""
from __future__ import annotations

import argparse

import pytest

from export import ExportConfig, _document_to_markdown, import_from_markdown
from library import Library
from schema import CATALOG_KIND_ELEMENT, CATALOG_KIND_FILE_INDEX


def _element(qn: str, location):
    """Catalog-element metadata with the given (possibly corrupted) location."""
    return {
        'kind': CATALOG_KIND_ELEMENT,
        'source_name': 'src1',
        'language': 'python',
        'subtype': 'function',
        'qualified_name': qn,
        'location': location,
    }


def test_repair_stringified_locations(tmp_path):
    """The migration revives str locations to dicts, leaves everything else
    untouched, reports unparseable values, and honors dry_run."""
    lib = Library(tmp_path / 'repair.db')

    good_dict = {'line_start': 10, 'line_end': 12, 'col_start': 0, 'col_end': 4}
    # (a) corrupted: location is the Python-repr string the bug produces.
    corrupted = lib.add_document(
        title='pkg.mod.broken', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.broken', repr(good_dict)),
    )
    # (b) already-clean dict location — must be left exactly as-is.
    clean = lib.add_document(
        title='pkg.mod.clean', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.clean', dict(good_dict)),
    )
    # (c) element with no location — skipped, not repaired.
    lib.add_document(
        title='pkg.mod.noloc', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.noloc', None),
    )
    # (d) a non-element catalog doc (file_index) — never inspected.
    lib.add_document(
        title='pkg/mod.py', content='x', content_type='catalog',
        source_files=['pkg/mod.py'],
        metadata={'kind': CATALOG_KIND_FILE_INDEX, 'source_name': 'src1', 'language': 'python'},
    )
    # (e) unparseable: a string that raises in literal_eval — left as-is, reported.
    bad_syntax = lib.add_document(
        title='pkg.mod.badsyntax', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.badsyntax', "{not valid"),
    )
    # (f) parses but not to a dict — left as-is, reported.
    not_a_dict = lib.add_document(
        title='pkg.mod.list', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.list', "[1, 2, 3]"),
    )

    # --- dry_run: reports the repair but writes nothing ---
    dry = lib.repair_stringified_locations(dry_run=True)
    assert dry.repaired == 1
    assert isinstance(lib.get_document(corrupted.id).metadata['location'], str)

    # --- real run ---
    res = lib.repair_stringified_locations()
    assert res.inspected == 5          # a,b,c,e,f are elements; d (file_index) excluded
    assert res.repaired == 1           # only (a)
    assert res.already_dict == 1       # (b)
    assert res.unparseable == 2        # (e), (f)
    assert {doc_id for doc_id, _ in res.unparseable_sample} == {bad_syntax.id, not_a_dict.id}

    # (a) is now a real dict with the original values restored.
    assert lib.get_document(corrupted.id).metadata['location'] == good_dict
    # (b) untouched, (e)/(f) left as strings (not silently dropped).
    assert lib.get_document(clean.id).metadata['location'] == good_dict
    assert lib.get_document(bad_syntax.id).metadata['location'] == "{not valid"
    assert lib.get_document(not_a_dict.id).metadata['location'] == "[1, 2, 3]"

    # --- idempotent: a second run finds nothing left to repair ---
    again = lib.repair_stringified_locations()
    assert again.repaired == 0
    lib.close()


def test_import_from_markdown_preserves_dict_location(tmp_path):
    """A dict-valued metadata field survives the export→import round trip
    (regression for the parser that only revived list-shaped values)."""
    src = Library(tmp_path / 'src.db')
    loc = {'line_start': 82, 'line_end': 90, 'col_start': 4, 'col_end': 20}
    meta = _element('pkg.mod.func', dict(loc))
    # A dict-shaped but *malformed* string must degrade gracefully: the parser
    # now routes ``{...}`` through literal_eval, so a bad one must be left as the
    # raw string rather than crash the import.
    meta['malformed'] = '{,}'
    doc = src.add_document(
        title='pkg.mod.func', content='body', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=meta,
    )

    export_dir = tmp_path / 'export'
    export_dir.mkdir()
    (export_dir / 'doc.md').write_text(_document_to_markdown(doc, ExportConfig()))

    dst = Library(tmp_path / 'dst.db')
    n = import_from_markdown(dst, export_dir)
    assert n == 1

    imported = dst.list_documents(content_type='catalog', limit=10)[0]
    assert imported.metadata['location'] == loc
    assert isinstance(imported.metadata['location'], dict)
    assert imported.metadata['malformed'] == '{,}'   # untouched, no crash
    src.close()
    dst.close()


def _migrate_args(db_path, **overrides):
    base = dict(
        db=str(db_path), check=False, source_files=False, fix_paths=False,
        fix_catalog_language=False, doc_ids=False, infer_source_name=False,
        fix_locations=False, dry_run=False, verbose=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def test_cmd_migrate_fix_locations(tmp_path):
    """`ariadne migrate --fix-locations` repairs corrupted rows end to end,
    honors --dry-run, and surfaces unparseable values."""
    from cli.maintenance import cmd_migrate

    db = tmp_path / 'cli.db'
    lib = Library(db)
    loc = {'line_start': 3, 'line_end': 3, 'col_start': 0, 'col_end': 9}
    broken = lib.add_document(
        title='pkg.mod.x', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.x', repr(loc)),
    )
    lib.close()

    # --dry-run: clean db (no unparseable), reports but writes nothing.
    assert cmd_migrate(_migrate_args(db, fix_locations=True, dry_run=True)) == 0
    lib_dry = Library(db)
    assert isinstance(lib_dry.get_document(broken.id).metadata['location'], str)
    # Add an unparseable row so the real run exercises the reporting branch.
    lib_dry.add_document(
        title='pkg.mod.bad', content='x', content_type='catalog',
        source_files=['pkg/mod.py'], metadata=_element('pkg.mod.bad', "{nope"),
    )
    lib_dry.close()

    # Real run: repairs the recoverable row, leaves the unparseable one.
    assert cmd_migrate(_migrate_args(db, fix_locations=True, verbose=True)) == 0
    lib2 = Library(db)
    assert lib2.get_document(broken.id).metadata['location'] == loc
    lib2.close()
