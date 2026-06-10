"""Shared memory-mapped document embedding matrix.

Tier 1 (BUILD): turn the embeddings already stored in ``ariadne.db`` into a
contiguous ``float32`` matrix artifact (``.npy``) plus a metadata sidecar.

Tier 2 (SERVE): ``EmbeddingMatrix`` ``mmap``s that artifact and ranks candidate
ids against a query embedding (full matmul, then index the result vector), with
a freshness check so a stale/absent matrix can fall back to the SQLite path.

See designs/embedding-matrix-tier1-build.md and -tier2-serve.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sqlite3 import Connection

    from numpy.typing import NDArray

    from library import Library

ARTIFACT_NAME = 'doc_embeddings.npy'
META_NAME = 'doc_embeddings.meta.json'


def embedding_stamp(conn: 'Connection') -> str:
    """A cheap fingerprint that changes iff the embedded-doc set or any embedding
    changes.

    Reads only ``(id, updated_at)`` — never the embedding bytes — so it stays
    cheap enough for a per-query freshness check (Tier 2). Sensitive to adds
    (new id), removals (missing id), and modifications (``updated_at`` advances
    on regeneration).
    """
    rows = conn.execute(
        'SELECT id, updated_at FROM documents WHERE embedding IS NOT NULL ORDER BY id'
    ).fetchall()
    digest = hashlib.sha256()
    for doc_id, updated_at in rows:
        digest.update(f'{doc_id}\x00{updated_at}\x00'.encode())
    return f'{len(rows)}:{digest.hexdigest()[:16]}'


def build_doc_embedding_matrix(library: 'Library', out_dir) -> Path:
    """Build the ``float32`` embedding matrix + meta sidecar from the documents
    that currently have embeddings.

    Rows are ordered by ``id`` (stable, reproducible); ``meta['ids'][k]`` is the
    doc id at row ``k``. Reads through the library so the matrix captures the
    *normalized* embeddings the ranker compares against. Returns the ``.npy``
    path.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            'SELECT id, embedding FROM documents WHERE embedding IS NOT NULL ORDER BY id'
        ).fetchall()
        stamp = embedding_stamp(conn)

    ids = [str(row[0]) for row in rows]
    if rows:
        matrix = np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows])
        matrix = np.ascontiguousarray(matrix, dtype=np.float32)
        dim = int(matrix.shape[1])
    else:
        dim = 0
        matrix = np.empty((0, 0), dtype=np.float32)

    npy_path = out_dir / ARTIFACT_NAME
    np.save(npy_path, matrix)
    meta = {'dim': dim, 'count': len(ids), 'ids': ids, 'build_stamp': stamp}
    (out_dir / META_NAME).write_text(json.dumps(meta))
    return npy_path


class EmbeddingMatrix:
    """A ``mmap``'d document embedding matrix + ``id → row`` index.

    Ranking scores the *whole* matrix (``batch_dot_similarity``, ~66 ms flat) and
    then indexes the result vector to the candidate rows — never slices the
    matrix (``M[rows]`` copies an ~866 MB submatrix). Per-row dot similarity is
    independent, so a candidate's score is identical to the SQLite path's.
    """

    def __init__(self, matrix, ids, dim, build_stamp) -> None:
        self.M = matrix
        self.ids = ids
        self.id_to_row = {doc_id: row for row, doc_id in enumerate(ids)}
        self.dim = dim
        self.build_stamp = build_stamp

    @classmethod
    def load(cls, out_dir) -> 'EmbeddingMatrix | None':
        """``mmap`` the artifact read-only. Returns ``None`` if it is absent."""
        out_dir = Path(out_dir)
        npy_path = out_dir / ARTIFACT_NAME
        meta_path = out_dir / META_NAME
        if not npy_path.exists() or not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        matrix = np.load(npy_path, mmap_mode='r')
        return cls(
            matrix=matrix,
            ids=list(meta['ids']),
            dim=int(meta['dim']),
            build_stamp=str(meta['build_stamp']),
        )

    def is_fresh(self, conn: 'Connection') -> bool:
        """True iff the live DB's embedding fingerprint matches the build stamp."""
        return embedding_stamp(conn) == self.build_stamp

    def rank(
        self, query_embedding: 'NDArray[np.float32]', candidate_ids, limit: int
    ) -> list[tuple[str, float]]:
        """Rank the in-matrix candidate ids by similarity. Ids absent from the
        matrix are skipped; closure scoping is the caller's (candidate_ids)."""
        from search import batch_dot_similarity, top_k_indices

        present = [cid for cid in candidate_ids if cid in self.id_to_row]
        if not present:
            return []
        rows = [self.id_to_row[cid] for cid in present]
        similarities = batch_dot_similarity(query_embedding, self.M)
        candidate_similarities = similarities[rows]
        top = top_k_indices(candidate_similarities, limit)
        return [(present[i], float(candidate_similarities[i])) for i in top]


def matrix_dir_for(library: 'Library') -> Path:
    """The directory holding the matrix artifact for a library — ``.ariadne/``
    next to its database file."""
    return Path(library._conn_provider.path).parent / '.ariadne'


def ensure_matrix(library: 'Library', out_dir=None) -> 'EmbeddingMatrix | None':
    """Build the matrix if it is absent or stale; reuse it if fresh.

    The offline build trigger and the build-on-startup primitive — never call it
    on the hot serve path (it can rebuild a ~1 GB artifact). Returns the loaded
    matrix.
    """
    out_dir = Path(out_dir) if out_dir is not None else matrix_dir_for(library)
    existing = EmbeddingMatrix.load(out_dir)
    if existing is not None:
        with library._conn_provider.acquire() as conn:
            if existing.is_fresh(conn):
                return existing
    build_doc_embedding_matrix(library, out_dir)
    return EmbeddingMatrix.load(out_dir)
