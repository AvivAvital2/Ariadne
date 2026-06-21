"""Guardrail: the library can enumerate documents that still need embedding.

The incremental embed path (``rebuild --only-missing`` / post-import) asks
the library "what is missing?" via ``embedding IS NULL``. These pin the two
query methods that answer it:

* ``count_missing_embeddings()`` — how many docs need (re)embedding.
* ``list_documents_without_embedding(limit=None)`` — those docs, newest-first.
"""
from __future__ import annotations

import numpy as np

from library import Library


def _lib(tmp_path) -> Library:
    return Library(tmp_path / 't.db')


def _emb() -> np.ndarray:
    return np.ones(8, dtype=np.float32)


def _seed(lib: Library, embedded: list[str], naked: list[str]) -> None:
    for t in embedded:
        lib.add_document(content_type='explanation', title=t, content=f'body {t}', embedding=_emb())
    for t in naked:
        lib.add_document(content_type='explanation', title=t, content=f'body {t}')


def test_count_missing_embeddings_counts_only_null(tmp_path):
    lib = _lib(tmp_path)
    _seed(lib, embedded=['A', 'B'], naked=['C', 'D', 'E'])
    assert lib.count_missing_embeddings() == 3


def test_count_missing_is_zero_when_all_embedded(tmp_path):
    lib = _lib(tmp_path)
    _seed(lib, embedded=['A', 'B'], naked=[])
    assert lib.count_missing_embeddings() == 0


def test_list_documents_without_embedding_returns_only_null(tmp_path):
    lib = _lib(tmp_path)
    _seed(lib, embedded=['A', 'B'], naked=['C', 'D', 'E'])
    docs = lib.list_documents_without_embedding()
    assert {d.title for d in docs} == {'C', 'D', 'E'}
    assert all(d.embedding is None for d in docs)


def test_list_documents_without_embedding_respects_limit(tmp_path):
    lib = _lib(tmp_path)
    _seed(lib, embedded=[], naked=['C', 'D', 'E'])
    assert len(lib.list_documents_without_embedding(limit=2)) == 2


def test_list_documents_without_embedding_empty_when_all_embedded(tmp_path):
    lib = _lib(tmp_path)
    _seed(lib, embedded=['A', 'B'], naked=[])
    assert lib.list_documents_without_embedding() == []
