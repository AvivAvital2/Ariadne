"""Persisted HNSW index over catalog-element embeddings — the semantic-edge
kNN index, built once and reused (mirrors ``library.embedding_matrix``).

Building the HNSW graph over ~100k embeddings costs ~95–190s; persisting it
(``save_index``) lets later runs reload it in ~1.4s instead of rebuilding, so
the incremental edge refresh queries a warm index instead of reconstructing it
every time.

BUILD turns the embedded catalog into an index artifact (``.bin``) + a meta
sidecar (``dim``, ``count``, the ``label -> doc_id`` list, and a freshness
stamp). SERVE loads the artifact, freshness-checks it against the live DB, and
answers neighbor queries — mapping hnswlib's integer labels (row positions) back
to doc_ids.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from docgen.graph_builder import _build_index

if TYPE_CHECKING:
    from sqlite3 import Connection

    from library import Library

INDEX_NAME = 'semantic_index.bin'
META_NAME = 'semantic_index.meta.json'
_EF_SEARCH = 64

_CATALOG_EMBEDDED_SQL = (
    "SELECT id, embedding FROM documents "
    "WHERE content_type = 'catalog' AND embedding IS NOT NULL ORDER BY id"
)


def catalog_embedding_stamp(conn: 'Connection') -> str:
    """A cheap fingerprint of the embedded-catalog set.

    Changes iff a catalog element's embedding is added, removed, or regenerated
    (``updated_at`` advances). Reads only ``(id, updated_at)`` — never the
    embedding bytes — so it stays cheap enough for a per-load freshness check.
    """
    rows = conn.execute(
        "SELECT id, updated_at FROM documents "
        "WHERE content_type = 'catalog' AND embedding IS NOT NULL ORDER BY id"
    ).fetchall()
    digest = hashlib.sha256()
    for doc_id, updated_at in rows:
        digest.update(f'{doc_id}\x00{updated_at}\x00'.encode())
    return f'{len(rows)}:{digest.hexdigest()[:16]}'


def build_semantic_index(
    library: 'Library', out_dir, *,
    ef_construction: int = 200, M: int = 16, ef_search: int = _EF_SEARCH,
) -> Path:
    """Build the HNSW index over embedded catalog elements and persist it + a
    meta sidecar. Rows are ordered by id (stable, reproducible), so hnswlib
    label ``k`` corresponds to ``meta['ids'][k]``. Returns the ``.bin`` path.

    ``ef_construction=200`` keeps full build recall; it's a one-time cost since
    the artifact persists.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with library._conn_provider.acquire() as conn:
        rows = conn.execute(_CATALOG_EMBEDDED_SQL).fetchall()
        stamp = catalog_embedding_stamp(conn)

    ids = [str(row[0]) for row in rows]
    meta: dict = {
        'count': len(ids), 'ids': ids, 'ef_search': ef_search,
        'build_stamp': stamp,
    }
    index_path = out_dir / INDEX_NAME
    if rows:
        matrix = np.ascontiguousarray(
            np.stack([np.frombuffer(row[1], dtype=np.float32) for row in rows]),
            dtype=np.float32,
        )
        meta['dim'] = int(matrix.shape[1])
        index = _build_index(
            matrix, ef_construction=ef_construction, M=M, ef_search=ef_search,
        )
        index.save_index(str(index_path))
    else:
        meta['dim'] = 0
    (out_dir / META_NAME).write_text(json.dumps(meta))
    return index_path


class SemanticIndex:
    """A loaded, persisted HNSW index plus its ``label -> doc_id`` map.

    hnswlib speaks integer labels (here, row positions); queries return doc_ids.
    ``index`` is ``None`` for an empty corpus, in which case queries return ``[]``.
    """

    def __init__(self, index, ids, dim, ef_search, build_stamp) -> None:
        self.index = index
        self.ids = ids
        self.dim = dim
        self.ef_search = ef_search
        self.build_stamp = build_stamp

    @classmethod
    def load(cls, out_dir) -> 'SemanticIndex | None':
        """Load the persisted index + meta. Returns ``None`` if the meta sidecar
        is absent."""
        out_dir = Path(out_dir)
        meta_path = out_dir / META_NAME
        index_path = out_dir / INDEX_NAME
        if not meta_path.exists():
            return None
        meta = json.loads(meta_path.read_text())
        dim = int(meta['dim'])
        ef_search = int(meta.get('ef_search', _EF_SEARCH))
        ids = list(meta['ids'])
        build_stamp = str(meta['build_stamp'])
        if not ids or not index_path.exists():
            return cls(None, ids, dim, ef_search, build_stamp)
        import hnswlib
        index = hnswlib.Index(space='cosine', dim=dim)
        index.load_index(str(index_path), max_elements=int(meta['count']))
        index.set_ef(ef_search)
        return cls(index, ids, dim, ef_search, build_stamp)

    def is_fresh(self, conn: 'Connection') -> bool:
        """True iff the live DB's embedded-catalog fingerprint matches the build
        stamp."""
        return catalog_embedding_stamp(conn) == self.build_stamp

    def query_neighbors(self, vector, k: int) -> list[tuple[str, float]]:
        """Top-``k`` neighbors of one query vector as ``[(doc_id, cosine_sim)]``
        (self included if the vector is in the index — the caller filters)."""
        if self.index is None or not self.ids:
            return []
        query = np.ascontiguousarray(
            np.asarray(vector, dtype=np.float32).reshape(1, -1),
        )
        kk = min(k, len(self.ids))
        labels, distances = self.index.knn_query(query, k=kk)
        return [
            (self.ids[int(label)], 1.0 - float(dist))
            for label, dist in zip(labels[0], distances[0])
        ]


def index_dir_for(library: 'Library') -> Path:
    """The directory holding the index artifact — ``.ariadne/`` next to the
    database file (shared with the embedding matrix)."""
    from library.embedding_matrix import matrix_dir_for
    return matrix_dir_for(library)


def ensure_semantic_index(library: 'Library', out_dir=None) -> 'SemanticIndex':
    """Build the index if absent or stale; reuse it if fresh. The offline build
    trigger — not for a hot path (it can rebuild a ~1GB artifact). Returns the
    loaded :class:`SemanticIndex`."""
    out_dir = Path(out_dir) if out_dir is not None else index_dir_for(library)
    existing = SemanticIndex.load(out_dir)
    if existing is not None:
        with library._conn_provider.acquire() as conn:
            if existing.is_fresh(conn):
                return existing
    build_semantic_index(library, out_dir)
    return SemanticIndex.load(out_dir)
