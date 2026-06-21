"""Guardrail: editing a document's content invalidates its stale embedding.

The incremental embedding path (``rebuild --only-missing`` / post-import)
keys on ``embedding IS NULL`` to decide what to re-embed. For that to catch
*edited* docs — not just brand-new ones — ``update_document`` must null the
embedding (and drop the now-stale chunks) when the content actually changes.

Crucially this must happen *only* when content changed and the caller did
not supply a fresh embedding:

* content changed, no embedding given → invalidate (NULL + drop chunks).
* content identical → leave embedding/chunks intact (re-import stays free).
* title-only change → leave embedding intact.
* content changed *with* an explicit embedding → keep the caller's vector.
"""
from __future__ import annotations

import numpy as np
import pytest

from library import Library
from schema import Chunk


def _lib(tmp_path) -> Library:
    return Library(tmp_path / 'test.db')


def _embedded_doc(lib: Library, content: str = 'original body text'):
    """Add a document that already has both an embedding and a chunk."""
    emb = np.ones(8, dtype=np.float32)
    doc = lib.add_document(
        content_type='explanation', title='Doc', content=content, embedding=emb,
    )
    lib.add_chunk(
        Chunk(document_id=doc.id, chunk_index=0, content=content, embedding=emb),
    )
    return doc


def test_content_change_resets_embedding_to_null_and_drops_chunks(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)
    assert lib.get_document(doc.id).embedding is not None
    assert len(lib.get_chunks(doc.id)) == 1

    lib.update_document(doc.id, content='a completely different body of text')

    after = lib.get_document(doc.id)
    assert after.content == 'a completely different body of text'
    assert after.embedding is None, 'stale embedding must be invalidated on content change'
    assert lib.get_chunks(doc.id) == [], 'stale chunks must be dropped on content change'


def test_unchanged_content_preserves_embedding_and_chunks(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib, content='same body')

    lib.update_document(doc.id, content='same body', title='New Title')

    after = lib.get_document(doc.id)
    assert after.title == 'New Title'
    assert after.embedding is not None, 're-importing identical content must not re-embed'
    assert len(lib.get_chunks(doc.id)) == 1


def test_title_only_change_preserves_embedding(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)

    lib.update_document(doc.id, title='Renamed')

    after = lib.get_document(doc.id)
    assert after.embedding is not None
    assert len(lib.get_chunks(doc.id)) == 1


def test_new_content_with_explicit_embedding_keeps_that_embedding(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)
    fresh = np.full(8, 0.5, dtype=np.float32)

    lib.update_document(doc.id, content='brand new content', embedding=fresh)

    after = lib.get_document(doc.id)
    assert after.embedding is not None, 'a caller-supplied embedding must win over invalidation'
    np.testing.assert_allclose(after.embedding, fresh)


# --- update_document contract guardrails (same function Slice 2a modifies) ---


def test_update_nonexistent_document_returns_none(tmp_path):
    lib = _lib(tmp_path)
    assert lib.update_document('does-not-exist', content='x') is None


def test_empty_title_is_rejected(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)
    with pytest.raises(ValueError, match='title'):
        lib.update_document(doc.id, title='   ')


def test_empty_content_is_rejected(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)
    with pytest.raises(ValueError, match='content'):
        lib.update_document(doc.id, content='   ')


def test_source_files_and_metadata_update_preserve_embedding(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)

    lib.update_document(doc.id, source_files=['a.py'], metadata={'k': 'v'})

    after = lib.get_document(doc.id)
    assert after.source_files == ['a.py']
    assert after.metadata.get('k') == 'v'
    assert after.embedding is not None, 'no content change → embedding kept'


def test_noop_update_preserves_embedding_and_chunks(tmp_path):
    lib = _lib(tmp_path)
    doc = _embedded_doc(lib)

    result = lib.update_document(doc.id)

    assert result is not None
    assert result.embedding is not None
    assert len(lib.get_chunks(doc.id)) == 1
