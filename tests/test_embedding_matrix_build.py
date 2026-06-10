"""Tier 1 (BUILD) — document embedding matrix artifact.

See designs/embedding-matrix-tier1-build.md. Built via the evolving-TDD loop and
then split into focused tests (one behavior each) for failure localization.

Fixtures are synthetic only: source ``src1``, docs ``d1, d2, …``, tiny
``dim = 4`` hand-chosen vectors.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from library import Library
from library.embedding_matrix import (
    ARTIFACT_NAME,
    META_NAME,
    build_doc_embedding_matrix,
)


@pytest.fixture
def lib(tmp_path: Path) -> Library:
    library = Library(tmp_path / 'lib.db')
    yield library
    library.close()


def _add(library: Library, doc_id: str, vec: list[float] | None) -> None:
    """Add a synthetic doc, optionally with an embedding."""
    embedding = np.array(vec, dtype=np.float32) if vec is not None else None
    library.add_document(
        content_type='explanation',
        title=f'title-{doc_id}',
        content=f'content-{doc_id}',
        embedding=embedding,
        doc_id=doc_id,
        source_name='src1',
    )


def _read_meta(out_dir: Path) -> dict:
    return json.loads((out_dir / META_NAME).read_text())


def test_build_basic(lib: Library, tmp_path: Path) -> None:
    """Builds a float32 (N, dim) artifact + meta sidecar."""
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(lib, 'd2', [0.0, 1.0, 0.0, 0.0])

    npy_path = build_doc_embedding_matrix(lib, tmp_path)

    assert npy_path == tmp_path / ARTIFACT_NAME
    assert npy_path.exists()
    matrix = np.load(npy_path)
    assert matrix.shape == (2, 4)
    assert matrix.dtype == np.float32

    meta = _read_meta(tmp_path)
    assert meta['count'] == 2
    assert meta['dim'] == 4
    assert meta['ids'] == ['d1', 'd2']


def test_build_excludes_null_embeddings(lib: Library, tmp_path: Path) -> None:
    """Docs without an embedding are excluded from the matrix."""
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(lib, 'd2', [0.0, 1.0, 0.0, 0.0])
    _add(lib, 'd3', None)

    build_doc_embedding_matrix(lib, tmp_path)

    meta = _read_meta(tmp_path)
    assert meta['count'] == 2
    assert 'd3' not in meta['ids']


def test_build_row_order_roundtrip(lib: Library, tmp_path: Path) -> None:
    """Row k corresponds to meta['ids'][k] and round-trips to the stored vec."""
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(lib, 'd2', [0.0, 1.0, 0.0, 0.0])

    build_doc_embedding_matrix(lib, tmp_path)

    meta = _read_meta(tmp_path)
    matrix = np.load(tmp_path / ARTIFACT_NAME)
    assert np.allclose(matrix[meta['ids'].index('d1')], [1.0, 0.0, 0.0, 0.0])
    assert np.allclose(matrix[meta['ids'].index('d2')], [0.0, 1.0, 0.0, 0.0])


def test_build_stamp_deterministic_and_sensitive(lib: Library, tmp_path: Path) -> None:
    """build_stamp is stable for an unchanged DB and changes on mutation."""
    _add(lib, 'd1', [1.0, 0.0, 0.0, 0.0])
    _add(lib, 'd2', [0.0, 1.0, 0.0, 0.0])

    build_doc_embedding_matrix(lib, tmp_path)
    stamp = _read_meta(tmp_path)['build_stamp']

    build_doc_embedding_matrix(lib, tmp_path)
    assert _read_meta(tmp_path)['build_stamp'] == stamp  # unchanged DB → same stamp

    with lib._conn_provider.acquire() as conn:
        conn.execute(
            'UPDATE documents SET updated_at = ? WHERE id = ?',
            ('2099-01-01T00:00:00', 'd1'),
        )
    build_doc_embedding_matrix(lib, tmp_path)
    assert _read_meta(tmp_path)['build_stamp'] != stamp  # mutation → different stamp


def test_build_empty_db(lib: Library, tmp_path: Path) -> None:
    """An empty DB builds a zero-row matrix without raising."""
    build_doc_embedding_matrix(lib, tmp_path)

    meta = _read_meta(tmp_path)
    assert meta['count'] == 0
    assert meta['dim'] == 0
    assert np.load(tmp_path / ARTIFACT_NAME).shape[0] == 0
