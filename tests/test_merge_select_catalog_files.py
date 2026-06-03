"""Contract tests for ``docgen.merge.select_catalog_files``.

The helper is the single source of truth for "which files does Ariadne
regenerate after a merge?" Both ``preview_merge`` (the dry-run reporter)
and ``execute_merge`` (the actual regenerator) must read it. If the two
paths disagree, the preview lies — the user sees "would regenerate N
files" but execute touches a different set.
"""

from __future__ import annotations


def test_select_catalog_files_keeps_supported_excludes_unsupported():
    """Every CATALOG_EXTS extension survives; non-catalog extensions drop.

    .md / .json / .yaml live in CATALOG_EXTS today (multi-language doc
    generation), so they MUST be included. .txt / .csv have no Ariadne
    extractor — drop them so we don't queue work that can't be done.
    """
    from docgen.merge import select_catalog_files

    files = [
        'main.py',
        'README.md',
        'config.json',
        'pipeline.yaml',
        'notes.txt',
        'data.csv',
    ]
    kept = sorted(select_catalog_files(files))

    assert 'main.py' in kept
    assert 'README.md' in kept
    assert 'config.json' in kept
    assert 'pipeline.yaml' in kept
    assert 'notes.txt' not in kept
    assert 'data.csv' not in kept


def test_select_catalog_files_matches_catalog_exts_constant():
    """The helper's accepted extensions must equal ``CATALOG_EXTS``.

    Asserts the contract directly against the canonical constant so a
    future addition to ``CATALOG_EXTS`` (e.g. ``.scala`` already shipped)
    doesn't silently leave merge regeneration behind.
    """
    from docgen.catalog_writer import CATALOG_EXTS
    from docgen.merge import select_catalog_files

    sample = [f'fixture{ext}' for ext in CATALOG_EXTS]
    sample.append('fixture.unknown')

    kept = set(select_catalog_files(sample))
    expected = {f'fixture{ext}' for ext in CATALOG_EXTS}

    assert kept == expected


def test_select_catalog_files_accepts_set_input():
    """``preview_merge`` passes a ``set[str]``; ensure that works too.

    The buggy original at ``preview_merge:330`` happened to accept any
    iterable; if a future refactor narrows the type to ``list[str]``
    the preview path silently breaks. Lock the contract.
    """
    from docgen.merge import select_catalog_files

    files = {'a.py', 'b.md'}
    kept = sorted(select_catalog_files(files))
    assert kept == ['a.py', 'b.md']
