"""Themes change-detection (which scopes incremental edge refresh) is
hash-based, not timestamp-based.

`refresh_themes` re-clusters every run, but only rebuilds semantic edges for
elements whose body hash (``metadata.sha_at_sync``) changed — leaving every
other element's edges (and thus the deterministic Leiden partition) stable. A
catalog-sync metadata-only refresh bumps ``updated_at`` without changing the
body hash, so it must NOT count as changed — otherwise its edges would be
rebuilt (via the non-deterministic HNSW index), churning the partition and
spuriously re-summarizing themes.
"""
from __future__ import annotations

import numpy as np
import pytest

from library import Library


def _add(library: Library, doc_id: str, sha: str) -> None:
    library.add_document(
        content_type='catalog',
        title=doc_id,
        content=f'function {doc_id}',
        source_files=[],
        embedding=np.zeros(8, dtype=np.float32),
        metadata={'kind': 'element', 'qualified_name': doc_id, 'sha_at_sync': sha},
        doc_id=doc_id,
    )


def _set_sha(library: Library, doc_id: str, sha: str, *, extra: dict | None = None) -> None:
    doc = library.get_document(doc_id)
    meta = dict(doc.metadata)
    meta['sha_at_sync'] = sha
    if extra:
        meta.update(extra)
    library.update_document(doc_id, metadata=meta)


@pytest.fixture
def library(tmp_path):
    lib = Library(tmp_path / 'hash_drift.db')
    yield lib
    lib.close()


def test_change_detection_is_hash_based_not_timestamp(library: Library) -> None:
    from docgen.themes import (
        _changed_catalog_elements,
        _record_theme_synced_hashes,
        _theme_synced_hashes_empty,
    )

    _add(library, 'A', 'h1')
    _add(library, 'B', 'h2')

    # Never recorded → both are changed (new → need edges).
    assert _theme_synced_hashes_empty(library) is True
    assert _changed_catalog_elements(library) == {'A', 'B'}

    _record_theme_synced_hashes(library)
    assert _theme_synced_hashes_empty(library) is False
    assert _changed_catalog_elements(library) == set()

    # COSMETIC refresh: bump updated_at + metadata but KEEP sha_at_sync → NOT
    # changed (so its edges aren't rebuilt and the partition stays stable).
    _set_sha(library, 'A', 'h1', extra={'line': 42})
    assert _changed_catalog_elements(library) == set()

    # REAL body change → detected → its edges get refreshed.
    _set_sha(library, 'B', 'h2-NEW')
    assert _changed_catalog_elements(library) == {'B'}

    _record_theme_synced_hashes(library, {'B'})
    assert _changed_catalog_elements(library) == set()
