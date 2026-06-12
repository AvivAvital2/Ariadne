"""Contract tests for single-source DB slicing (``export_db``).

These tests are the spec for ``export_source_db`` / ``import_source_db``.
The headline contract is **retrieval parity**: querying a slice must return
the same documents, in the same order, with the same scores as the full
library does *when scoped to that source* — because that scoped view is what
the MCP server / Slack bridge already serves today.

Determinism notes:
- Embeddings are small unit vectors built from a seeded RNG, so dot-product
  ranking is fully deterministic and no network/API call is made.
- ``_count_tokens`` is patched out (autouse) so document insertion never
  reaches out to the Anthropic token-counting endpoint.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pytest

from export_db import (
    DEFAULT_EMBEDDING_MODEL,
    EmbeddingModelMismatch,
    ImportConflictError,
    SourceNotFoundError,
    export_source_db,
    import_source_db,
)
from library import Library
from schema import Chunk, Section
from search import batch_dot_similarity, top_k_indices

EMBED_DIM = 16


def _vec(seed: int) -> np.ndarray:
    """Deterministic unit vector for a given seed."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(EMBED_DIM).astype(np.float32)
    return (v / np.linalg.norm(v)).astype(np.float32)


def _rank(lib: Library, qvec: np.ndarray, candidate_ids: list[str], k: int) -> list[tuple[str, float]]:
    """Rank ``candidate_ids`` in ``lib`` by similarity to ``qvec``.

    Uses the library's own embedding store plus the production ranking
    primitives, so it reads the actually-stored bytes — a dropped or mangled
    embedding in the slice would change the result. Canonicalised by
    (-score, id) to remove any tie-order ambiguity.
    """
    embs = lib.get_embeddings_for_ids(candidate_ids)
    ids = [i for i in candidate_ids if i in embs]
    mat = np.stack([embs[i] for i in ids])
    sims = batch_dot_similarity(qvec, mat)
    top = top_k_indices(sims, k)
    scored = [(ids[j], float(sims[j])) for j in top]
    return sorted(scored, key=lambda x: (-x[1], x[0]))


def _canon(results) -> list[tuple[str, float]]:
    """Canonicalise SearchResult list the same way as ``_rank``."""
    scored = [(r.document.id, float(r.score)) for r in results]
    return sorted(scored, key=lambda x: (-x[1], x[0]))


@pytest.fixture(autouse=True)
def _no_token_count(monkeypatch):
    """Never call the Anthropic token-count API during tests."""
    monkeypatch.setattr('library.core.CoreMixin._count_tokens', lambda self, content: None)


@pytest.fixture
def full_db(tmp_path):
    """A two-source library: 'sdk' (the slice target) + 'noise' (must not leak)."""
    path = tmp_path / 'full.db'
    lib = Library(path)

    def add(sn, ct, title, files, seed):
        return lib.add_document(
            content_type=ct, title=title,
            content=f'{title}. Body text describing {title}.',
            source_files=files, embedding=_vec(seed), source_name=sn,
        ).id

    a = add('sdk', 'explanation', 'SDK Authentication Flow', ['sdk/auth.py'], 1)
    b = add('sdk', 'qa', 'How do I authenticate with the SDK?', ['sdk/auth.py'], 2)
    c = add('sdk', 'architecture', 'SDK Client Architecture', ['sdk/client.py'], 3)
    e = add('sdk', 'qa', 'How do I configure retries in the SDK?', ['sdk/client.py'], 4)
    sdk_ids = [a, b, c, e]
    qa = [(b, _vec(2)), (e, _vec(4))]  # each qa doc's own vector is a query

    lib.add_chunks_batch([Chunk(document_id=a, chunk_index=0, content='auth chunk', embedding=_vec(11))])
    lib.store_sections(a, [Section(document_id=a, index=0, heading='Auth',
                                   description='auth', content='## Auth\nbody', embedding=_vec(12))])

    n1 = add('noise', 'explanation', 'Billing Service Overview', ['billing/svc.py'], 50)
    n2 = add('noise', 'qa', 'How does billing reconciliation work?', ['billing/svc.py'], 51)
    noise_ids = [n1, n2]
    lib.add_chunks_batch([Chunk(document_id=n1, chunk_index=0, content='billing chunk', embedding=_vec(60))])

    lib.close()
    return {
        'path': str(path), 'sdk_ids': sdk_ids, 'noise_ids': noise_ids, 'qa': qa,
        'sdk_count': len(sdk_ids), 'sdk_chunk_doc': a, 'noise_chunk_doc': n1,
    }


# --------------------------------------------------------------------------
# Structural round-trip
# --------------------------------------------------------------------------

def test_export_contains_exactly_the_source_documents(full_db, tmp_path):
    out = tmp_path / 'slice.db'
    manifest = export_source_db(full_db['path'], 'sdk', out)
    sl = Library(out)
    try:
        ids = {d.id for d in sl.list_documents()}
    finally:
        sl.close()
    assert ids == set(full_db['sdk_ids'])
    assert manifest.doc_count == full_db['sdk_count']


def test_export_excludes_other_sources(full_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(full_db['path'], 'sdk', out)
    sl = Library(out)
    try:
        ids = {d.id for d in sl.list_documents()}
    finally:
        sl.close()
    # Pair the positive (source docs present) with the negative (no noise) so
    # an empty slice can't pass this vacuously.
    assert set(full_db['sdk_ids']) <= ids, 'source docs missing from slice'
    assert not (ids & set(full_db['noise_ids'])), 'noise-source docs leaked into the slice'


def test_export_carries_chunks_and_sections_for_source_only(full_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(full_db['path'], 'sdk', out)
    conn = sqlite3.connect(out)
    try:
        chunk_docs = {r[0] for r in conn.execute('SELECT DISTINCT document_id FROM chunks')}
        sec_docs = {r[0] for r in conn.execute('SELECT DISTINCT document_id FROM sections')}
    finally:
        conn.close()
    assert full_db['sdk_chunk_doc'] in chunk_docs
    assert full_db['sdk_chunk_doc'] in sec_docs
    assert full_db['noise_chunk_doc'] not in chunk_docs, 'noise chunk leaked'


def test_unknown_source_raises(full_db, tmp_path):
    with pytest.raises(SourceNotFoundError):
        export_source_db(full_db['path'], 'does-not-exist', tmp_path / 'x.db')


# --------------------------------------------------------------------------
# Embedding fidelity + retrieval parity (the load-bearing contract)
# --------------------------------------------------------------------------

def test_embeddings_carried_byte_for_byte(full_db, tmp_path):
    out = tmp_path / 'slice.db'
    export_source_db(full_db['path'], 'sdk', out)
    src = sqlite3.connect(f"file:{full_db['path']}?mode=ro", uri=True)
    bnd = sqlite3.connect(out)
    try:
        for did in full_db['sdk_ids']:
            ra = src.execute('SELECT embedding FROM documents WHERE id=?', (did,)).fetchone()
            rb = bnd.execute('SELECT embedding FROM documents WHERE id=?', (did,)).fetchone()
            assert rb is not None, f'doc {did} missing from slice'
            assert ra[0] is not None and ra[0] == rb[0], f'embedding for {did} not preserved'
    finally:
        src.close(); bnd.close()


def test_retrieval_parity_over_qa_battery(full_db, tmp_path):
    """Slice retrieval == full-library-scoped-to-source retrieval, for every
    qa-doc query: identical ids, identical order, scores within 1e-6."""
    out = tmp_path / 'slice.db'
    export_source_db(full_db['path'], 'sdk', out)

    full = Library(Path(full_db['path']))
    sl = Library(out)
    try:
        for _qid, qvec in full_db['qa']:
            baseline = _rank(full, qvec, full_db['sdk_ids'], k=3)
            got = _canon(sl.search(qvec, k=3))
            assert [g[0] for g in got] == [b[0] for b in baseline], 'ranking diverged'
            assert max(abs(g[1] - b[1]) for g, b in zip(got, baseline)) < 1e-6, 'scores diverged'
    finally:
        full.close(); sl.close()


def test_slice_answers_offline_with_no_api(full_db, tmp_path):
    """Top hit for a qa doc's own vector is that doc — using only shipped
    embeddings, no embedding service."""
    out = tmp_path / 'slice.db'
    export_source_db(full_db['path'], 'sdk', out)
    qid, qvec = full_db['qa'][0]
    sl = Library(out)
    try:
        results = sl.search(qvec, k=1)
    finally:
        sl.close()
    assert results, 'slice returned no results'
    assert results[0].document.id == qid


def test_export_without_embeddings_keeps_docs_but_drops_vectors(full_db, tmp_path):
    out = tmp_path / 'slice.db'
    manifest = export_source_db(full_db['path'], 'sdk', out, include_embeddings=False)
    sl = Library(out)
    try:
        docs = sl.list_documents()
    finally:
        sl.close()
    assert len(docs) == full_db['sdk_count']
    assert all(d.embedding is None for d in docs)
    assert manifest.includes_embeddings is False


def test_all_content_types_are_carried(tmp_path):
    """Every document type Ariadne generates for a source travels — not just qa."""
    path = tmp_path / 'full.db'
    lib = Library(path)
    types = ['explanation', 'architecture', 'qa', 'diagram', 'catalog', 'gotcha', 'finding']
    for i, ct in enumerate(types):
        lib.add_document(content_type=ct, title=f'{ct} doc', content=f'{ct} body',
                         source_files=['sdk/x.py'], embedding=_vec(i + 1), source_name='sdk')
    lib.close()
    out = tmp_path / 'slice.db'
    export_source_db(str(path), 'sdk', out)
    sl = Library(out)
    try:
        got = {d.content_type for d in sl.list_documents()}
    finally:
        sl.close()
    assert got == set(types), f'missing content types in slice: {set(types) - got}'


# --------------------------------------------------------------------------
# Merge into an already-populated database
# --------------------------------------------------------------------------

def _make_target_with_other_source(path):
    lib = Library(Path(path))
    other = lib.add_document(
        content_type='explanation', title='Unrelated Subsystem',
        content='Pre-existing doc that must survive the import.',
        source_files=['other/mod.py'], embedding=_vec(900), source_name='other',
    ).id
    lib.close()
    return other


def test_import_merges_source_and_leaves_other_untouched(full_db, tmp_path):
    bundle = tmp_path / 'bundle.db'
    export_source_db(full_db['path'], 'sdk', bundle)
    target = tmp_path / 'target.db'
    other_id = _make_target_with_other_source(target)

    import_source_db(target, bundle)

    tgt = Library(target)
    try:
        by_source = {}
        for d in tgt.list_documents():
            by_source.setdefault(d.source_name, set()).add(d.id)
    finally:
        tgt.close()
    assert by_source.get('sdk') == set(full_db['sdk_ids']), 'sdk docs not merged in'
    assert by_source.get('other') == {other_id}, 'unrelated source was disturbed'


def test_import_is_idempotent(full_db, tmp_path):
    bundle = tmp_path / 'bundle.db'
    export_source_db(full_db['path'], 'sdk', bundle)
    target = tmp_path / 'target.db'
    _make_target_with_other_source(target)

    import_source_db(target, bundle)
    tgt = Library(target)
    first = len([d for d in tgt.list_documents() if d.source_name == 'sdk'])
    tgt.close()

    import_source_db(target, bundle)
    tgt = Library(target)
    second = len([d for d in tgt.list_documents() if d.source_name == 'sdk'])
    tgt.close()

    assert first == full_db['sdk_count']
    assert first == second, 're-import changed the row count'


def test_import_rejects_embedding_model_mismatch(full_db, tmp_path):
    bundle = tmp_path / 'bundle.db'
    export_source_db(full_db['path'], 'sdk', bundle, embedding_model=DEFAULT_EMBEDDING_MODEL)
    target = tmp_path / 'target.db'
    _make_target_with_other_source(target)
    with pytest.raises(EmbeddingModelMismatch):
        import_source_db(target, bundle, expected_embedding_model='some-other-embedding-model')


def _seed_colliding_doc(path, doc_id):
    """Put a document with ``doc_id`` (a bundle id) but distinct content into the target."""
    lib = Library(Path(path))
    lib.add_document(
        content_type='qa', title='PRE-EXISTING TITLE',
        content='Original target content that conflicts with the bundle.',
        source_files=['sdk/auth.py'], embedding=_vec(1), doc_id=doc_id, source_name='sdk',
    )
    lib.close()


def test_import_skip_preserves_existing_on_conflict(full_db, tmp_path):
    bundle = tmp_path / 'bundle.db'
    export_source_db(full_db['path'], 'sdk', bundle)
    target = tmp_path / 'target.db'
    collide = full_db['sdk_ids'][0]
    _seed_colliding_doc(target, collide)

    report = import_source_db(target, bundle, on_conflict='skip')

    tgt = Library(target)
    try:
        kept = tgt.get_document(collide)
    finally:
        tgt.close()
    assert report.conflicts == 1
    assert kept.title == 'PRE-EXISTING TITLE', 'skip must not overwrite the existing doc'


def test_import_fail_raises_on_conflict(full_db, tmp_path):
    bundle = tmp_path / 'bundle.db'
    export_source_db(full_db['path'], 'sdk', bundle)
    target = tmp_path / 'target.db'
    _seed_colliding_doc(target, full_db['sdk_ids'][0])
    with pytest.raises(ImportConflictError):
        import_source_db(target, bundle, on_conflict='fail')


def test_import_rejects_invalid_on_conflict(full_db, tmp_path):
    bundle = tmp_path / 'bundle.db'
    export_source_db(full_db['path'], 'sdk', bundle)
    target = tmp_path / 'target.db'
    _make_target_with_other_source(target)
    with pytest.raises(ValueError):
        import_source_db(target, bundle, on_conflict='bogus')
