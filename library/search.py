"""Semantic search and scope filtering operations."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from config import HUMAN_DOC_PROVENANCE, OFFICIAL_DOC_PROVENANCE

from schema import Chunk, ContentType, Document, SearchResult

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


# Human-authored docs (rst/markdown) rank below code-derived docs for the same
# query — a provenance down-weight multiplied into similarity alongside
# the usage-feedback weight. <1 sinks without hiding.
_HUMAN_DOC_RANK_WEIGHT = 0.8

_STALE_DOC_RANK_WEIGHT = 0.6
_OFFICIAL_DOC_RANK_WEIGHT = 0.9


def provenance_weight(metadata):
    """Similarity multiplier from a doc's provenance metadata, shared by every
    ranking path so the calibration is identical wherever search runs:
    human-authored docs rank below code-derived, and a doc whose rst autodoc
    target no longer resolves sinks further. 1.0 = no change."""
    weight = 1.0
    if metadata.get('provenance') == HUMAN_DOC_PROVENANCE:
        weight *= _HUMAN_DOC_RANK_WEIGHT
    if metadata.get('provenance') == OFFICIAL_DOC_PROVENANCE:
        weight *= _OFFICIAL_DOC_RANK_WEIGHT
    if metadata.get('stale_autodoc'):
        weight *= _STALE_DOC_RANK_WEIGHT
    return weight


class SearchMixin:
    """Semantic search and scope filtering.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    - self.list_documents() from CoreMixin
    - self.get_documents_batch() from CoreMixin
    """

    def search(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 5,
        content_type: ContentType | None = None,
    ) -> list[SearchResult]:
        """Search for documents by semantic similarity.

        Args:
            query_embedding: The query vector (1536 dimensions).
            k: Number of results to return.
            content_type: Filter by content type.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        from search import batch_dot_similarity, top_k_indices

        # Get all documents with embeddings
        docs = self.list_documents(content_type=content_type)
        docs_with_embeddings = [d for d in docs if d.embedding is not None]

        if not docs_with_embeddings:
            return []

        # Batch compute similarities (vectorized, much faster than per-doc loop)
        embeddings = np.stack([d.embedding for d in docs_with_embeddings])  # type: ignore[misc]
        similarities = batch_dot_similarity(query_embedding, embeddings)
        # Down-rank zero-hit documents (proven unhelpful in past searches)
        for i, doc in enumerate(docs_with_embeddings):
            stats = self.get_doc_hit_stats(doc.id)
            served, hits = stats['served'], stats['hits']
            if served == 0:
                weight = 1.0
            elif hits == 0 and served >= 5:
                weight = 0.3
            elif hits == 0 and served >= 3:
                weight = 0.5
            else:
                weight = 0.7 + 0.3 * (hits / served) if served > 0 else 1.0
            similarities[i] *= weight
            similarities[i] *= provenance_weight(doc.metadata)

        top_indices = top_k_indices(similarities, k)

        return [
            SearchResult(document=docs_with_embeddings[i], score=float(similarities[i]))
            for i in top_indices
        ]

    def search_chunks(
        self,
        query_embedding: NDArray[np.float32],
        k: int = 5,
    ) -> list[SearchResult]:
        """Search for documents by chunk-level semantic similarity.

        Args:
            query_embedding: The query vector (1536 dimensions).
            k: Number of results to return.

        Returns:
            List of SearchResult objects with matching chunks.
        """
        from search import batch_dot_similarity, top_k_indices

        # Get all chunks with embeddings
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                '''SELECT c.id, c.document_id, c.chunk_index, c.content, c.embedding
                   FROM chunks c WHERE c.embedding IS NOT NULL'''
            ).fetchall()

        if not rows:
            return []

        # Parse chunks and collect embeddings
        chunks: list[Chunk] = []
        raw_embeddings: list[NDArray[np.float32]] = []
        for row in rows:
            embedding_blob = row[4]
            if embedding_blob is None:
                continue
            emb = np.frombuffer(embedding_blob, dtype=np.float32).copy()
            chunks.append(Chunk(
                id=str(row[0]),
                document_id=str(row[1]),
                chunk_index=int(row[2]),
                content=str(row[3]),
                embedding=emb,
            ))
            raw_embeddings.append(emb)

        if not chunks:
            return []

        # Batch compute similarities (vectorized — handles 15K+ chunks efficiently)
        embeddings_matrix = np.stack(raw_embeddings)
        similarities = batch_dot_similarity(query_embedding, embeddings_matrix)
        top_indices = top_k_indices(similarities, k)
        top_chunks = [(chunks[i], float(similarities[i])) for i in top_indices]

        # Fetch parent documents in batch (one query instead of N)
        unique_doc_ids = list({chunk.document_id for chunk, _ in top_chunks})
        doc_cache = {d.id: d for d in self.get_documents_batch(unique_doc_ids)}

        results: list[SearchResult] = []
        for chunk, score in top_chunks:
            if chunk.document_id in doc_cache:
                results.append(SearchResult(
                    document=doc_cache[chunk.document_id],
                    score=score,
                    chunk=chunk,
                ))

        return results

    def find_documents_by_source_files(self, file_paths: list[str]) -> list[Document]:
        """Find documents that reference any of the given source files.

        Uses SQL LIKE queries to push filtering to the database instead of
        loading all documents into Python.

        Args:
            file_paths: List of file paths to search for.

        Returns:
            List of documents that reference any of the files.
        """
        if not file_paths:
            return []

        # Build SQL with LIKE patterns for each file path
        # source_files is stored as JSON array, so we match substrings
        like_clauses = []
        params: list[str] = []
        for fp in file_paths:
            # Use the basename for matching to handle relative/absolute path differences
            basename = fp.split('/')[-1] if '/' in fp else fp
            like_clauses.append('source_files LIKE ?')
            params.append(f'%{basename}%')

        where = ' OR '.join(like_clauses)
        query = (
            f'SELECT id, content_type, title, content, source_files, embedding, '
            f'created_at, updated_at, metadata, source_name '
            f'FROM documents WHERE {where}'
        )

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(query, params).fetchall()

        matching = [self._row_to_document(row) for row in rows]

        return matching

    def find_documents_page_by_source_files(
        self,
        file_paths: list[str],
        *,
        limit: int,
        content_types: list[str] | None = None,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        """Memory-bounded, paginated fetch by source file — the bounded counterpart to
        :meth:`find_documents_by_source_files` (which loads every match).

        Pushes the basename LIKE match, an optional ``content_type`` filter, a stable
        ``ORDER BY`` and ``LIMIT``/``OFFSET`` down to SQL so only the page's rows are
        loaded into Python. ``COUNT(*) OVER ()`` rides along on each row, so the full
        pre-LIMIT match total comes back in the same single scan — no extra COUNT query.

        Args:
            file_paths: File paths to match (by basename, as in find_documents_by_source_files).
            limit: Max rows to return for this page (required — this is the paginated fetch).
            content_types: If given, restrict to these types (pushed into the WHERE clause).
            offset: Number of leading rows to skip.

        Returns:
            ``(page_documents, total_matches)`` — page bounded by ``limit``; total is the
            full count before pagination.
        """
        if not file_paths:
            return [], 0

        like_clauses: list[str] = []
        params: list[object] = []
        for fp in file_paths:
            basename = fp.split('/')[-1] if '/' in fp else fp
            like_clauses.append('source_files LIKE ?')
            params.append(f'%{basename}%')
        where = '(' + ' OR '.join(like_clauses) + ')'

        if content_types:
            placeholders = ','.join('?' * len(content_types))
            where += f' AND content_type IN ({placeholders})'
            params.extend(content_types)

        query = (
            f'SELECT id, content_type, title, content, source_files, embedding, '
            f'created_at, updated_at, metadata, source_name, COUNT(*) OVER () AS total '
            f'FROM documents WHERE {where} ORDER BY content_type, id LIMIT ? OFFSET ?'
        )
        params.extend([limit, offset])

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(query, params).fetchall()

        if not rows:
            return [], 0

        page: list[Document] = []
        total = 0
        for row in rows:
            cols = tuple(row)
            total = cols[-1]  # COUNT(*) OVER () — identical on every row
            page.append(self._row_to_document(cols[:-1]))
        return page, total

    def filter_documents_by_scope(
        self,
        docs: list[Document],
        source_paths: list[Path],
    ) -> list[Document]:
        """Filter documents to only those whose source files are under given paths.

        Args:
            docs: List of documents to filter.
            source_paths: List of source directory paths to include.

        Returns:
            List of documents whose source_files are under any of the paths.
        """
        if not source_paths:
            return docs

        filtered: list[Document] = []
        source_path_strs = [str(p) for p in source_paths]

        for doc in docs:
            if not doc.source_files:
                # Include docs without source files (e.g., findings)
                filtered.append(doc)
                continue

            # Check if any source file is under any of the scope paths
            for sf in doc.source_files:
                for sp in source_path_strs:
                    # Check if source file path starts with or contains the source path
                    if sf.startswith(sp) or sp in sf or sf in sp:
                        filtered.append(doc)
                        break
                else:
                    continue
                break

        return filtered

    def search_with_scope(
        self,
        query_embedding: NDArray[np.float32],
        source_paths: list[Path] | None = None,
        k: int = 5,
        content_type: ContentType | None = None,
    ) -> list[SearchResult]:
        """Search for documents with optional scope filtering.

        Args:
            query_embedding: The query vector (1536 dimensions).
            source_paths: List of source paths to limit results to.
            k: Number of results to return.
            content_type: Filter by content type.

        Returns:
            List of SearchResult objects sorted by descending similarity.
        """
        from search import batch_dot_similarity, top_k_indices

        # Get all documents with embeddings
        docs = self.list_documents(content_type=content_type)
        docs_with_embeddings = [d for d in docs if d.embedding is not None]

        # Apply scope filtering if paths provided
        if source_paths:
            docs_with_embeddings = self.filter_documents_by_scope(
                docs_with_embeddings, source_paths
            )

        if not docs_with_embeddings:
            return []

        # Batch compute similarities (vectorized)
        embeddings = np.stack([d.embedding for d in docs_with_embeddings])  # type: ignore[misc]
        similarities = batch_dot_similarity(query_embedding, embeddings)
        for i, doc in enumerate(docs_with_embeddings):
            similarities[i] *= provenance_weight(doc.metadata)
        top_indices = top_k_indices(similarities, k)

        return [
            SearchResult(document=docs_with_embeddings[i], score=float(similarities[i]))
            for i in top_indices
        ]
