"""Tests for type-aware staleness checks.

Today ``get_stale_files`` is binary: a file with any prior doc is
considered "documented", regardless of whether the requested doc types
are actually present. Adding architecture/qa/gotcha after explanation
required ``--force`` to bypass the staleness skip — annoying.

After this fix, ``get_stale_files(requested_types=...)`` treats a file
as stale when ANY requested type is missing, even if the file's hash
hasn't changed. ``--force`` is no longer needed for cross-type extension.
"""
from __future__ import annotations

import numpy as np


def _seed_library_with_docs(lib, file_path, doc_specs):
    """Helper: register doc rows in the library for the given file.

    ``doc_specs`` is a list of ``{"id": str, "content_type": str}`` dicts.
    Returns the list of generated doc IDs.
    """
    doc_ids = []
    for spec in doc_specs:
        doc = lib.add_document(
            content_type=spec['content_type'],
            title=spec.get('title', f"doc-{spec['content_type']}"),
            content=spec.get('content', 'body'),
            source_files=[str(file_path)],
            embedding=np.zeros(3072, dtype=np.float32),
            metadata={},
            doc_id=spec.get('id'),
            source_name=spec.get('source_name', 'test'),
        )
        doc_ids.append(doc.id)
    return doc_ids


def test_file_with_all_requested_types_is_not_stale(tmp_path):
    """When the file's prior docs cover every requested type, it's clean."""
    from docgen.staleness import StalenessTracker
    from library import Library

    src = tmp_path / 'x.py'
    src.write_text('x = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'test.db')
    try:
        doc_ids = _seed_library_with_docs(lib, src, [
            {'content_type': 'explanation'},
            {'content_type': 'architecture'},
        ])

        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(src, doc_ids, base_path=tmp_path)
            stale = tracker.get_stale_files(
                [src], base_path=tmp_path,
                requested_types=('explanation', 'architecture'),
                library=lib,
            )
            assert stale == [], (
                f'expected no stale files when all requested types present; '
                f'got {stale}'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_file_missing_requested_type_is_stale(tmp_path):
    """File has explanation but not architecture → stale when architecture
    is requested.
    """
    from docgen.staleness import StalenessTracker
    from library import Library

    src = tmp_path / 'y.py'
    src.write_text('y = 2\n', encoding='utf-8')

    lib = Library(tmp_path / 'test.db')
    try:
        doc_ids = _seed_library_with_docs(lib, src, [
            {'content_type': 'explanation'},
        ])

        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(src, doc_ids, base_path=tmp_path)
            stale = tracker.get_stale_files(
                [src], base_path=tmp_path,
                requested_types=('explanation', 'architecture'),
                library=lib,
            )
            assert src in stale, (
                'file with explanation but no architecture should be stale '
                'when both are requested'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_no_requested_types_falls_back_to_legacy_behavior(tmp_path):
    """Without ``requested_types``, current binary 'has any doc' rule wins."""
    from docgen.staleness import StalenessTracker
    from library import Library

    src = tmp_path / 'z.py'
    src.write_text('z = 3\n', encoding='utf-8')

    lib = Library(tmp_path / 'test.db')
    try:
        doc_ids = _seed_library_with_docs(lib, src, [
            {'content_type': 'explanation'},
        ])

        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(src, doc_ids, base_path=tmp_path)
            # No requested_types kwarg — legacy "has-any-doc" semantics.
            stale = tracker.get_stale_files([src], base_path=tmp_path)
            assert stale == [], 'legacy path: any doc should mark file clean'
        finally:
            tracker.close()
    finally:
        lib.close()


def test_changed_file_always_stale_regardless_of_types(tmp_path):
    """Hash mismatch trumps the type check — changed source always stale."""
    from docgen.staleness import StalenessTracker
    from library import Library

    src = tmp_path / 'w.py'
    src.write_text('w = 1\n', encoding='utf-8')

    lib = Library(tmp_path / 'test.db')
    try:
        doc_ids = _seed_library_with_docs(lib, src, [
            {'content_type': 'explanation'},
            {'content_type': 'architecture'},
        ])

        tracker = StalenessTracker(tmp_path / 'stale.db')
        try:
            tracker.record_documentation(src, doc_ids, base_path=tmp_path)
            # Modify the file → new hash.
            src.write_text('w = 999\n', encoding='utf-8')
            stale = tracker.get_stale_files(
                [src], base_path=tmp_path,
                requested_types=('explanation', 'architecture'),
                library=lib,
            )
            assert src in stale, (
                'hash mismatch must mark the file stale even when all '
                'requested types exist'
            )
        finally:
            tracker.close()
    finally:
        lib.close()


def test_file_with_no_record_is_stale(tmp_path):
    """Never-documented file is always stale (existing behavior preserved)."""
    from docgen.staleness import StalenessTracker

    src = tmp_path / 'fresh.py'
    src.write_text('v = 1\n', encoding='utf-8')

    tracker = StalenessTracker(tmp_path / 'stale.db')
    try:
        stale = tracker.get_stale_files(
            [src], base_path=tmp_path,
            requested_types=('explanation',),
            library=None,  # not consulted when no record exists
        )
        assert src in stale
    finally:
        tracker.close()
