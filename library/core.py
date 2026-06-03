"""Core document CRUD, chunk, section, and sync state operations."""
from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

import numpy as np

from schema import Chunk, ContentType, Document, Section

if TYPE_CHECKING:
    from numpy.typing import NDArray

_logger = logging.getLogger(__name__)


class CoreMixin:
    """Document CRUD, chunk, section, and sync state operations.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    """

    def add_document(
        self,
        content_type: ContentType,
        title: str,
        content: str,
        source_files: list[str] | None = None,
        embedding: NDArray[np.float32] | None = None,
        metadata: dict[str, object] | None = None,
        doc_id: str | None = None,
        source_name: str | None = None,
    ) -> Document:
        """Add a new document to the library.

        Args:
            content_type: Type of content ('explanation', 'architecture', 'qa', 'diagram').
            title: Human-readable title.
            content: Full markdown content.
            source_files: List of related source file paths.
            embedding: Optional pre-computed embedding.
            metadata: Optional additional metadata.
            doc_id: Optional document ID. If provided and exists, updates instead of creating.
            source_name: Source attribution column. Used for per-source filtering.

        Returns:
            The created or updated Document instance.
        """
        # Validate inputs
        if not title or not title.strip():
            raise ValueError('Document title must not be empty')
        if not content or not content.strip():
            raise ValueError('Document content must not be empty')
        from schema import CONTENT_TYPES
        if content_type not in CONTENT_TYPES:
            raise ValueError(f'Invalid content_type: {content_type}. Must be one of {CONTENT_TYPES}')

        # If doc_id provided and document exists, update it instead
        if doc_id is not None:
            existing = self.get_document(doc_id)
            if existing is not None:
                return self.update_document(
                    doc_id,
                    title=title,
                    content=content,
                    source_files=source_files,
                    embedding=embedding,
                    metadata=metadata,
                ) or existing

        # Create new document (with provided ID or generated one)
        if doc_id is not None:
            doc = Document(
                id=doc_id,
                content_type=content_type,
                title=title,
                content=content,
                source_files=source_files or [],
                embedding=embedding,
                metadata=metadata or {},
                source_name=source_name,
            )
        else:
            doc = Document(
                content_type=content_type,
                title=title,
                content=content,
                source_files=source_files or [],
                embedding=embedding,
                metadata=metadata or {},
                source_name=source_name,
            )
        self._insert_document(doc)
        return doc

    def _count_tokens(self, content: str) -> int | None:                                                                                             
        """Count tokens in `content` via Anthropic's tokenizer.                                                                                                         
                                                                                                                                                                        
        Returns None on failure (missing key, network error, etc.) so ingestion                                                                                         
        never blocks on token counting. NULL rows can be filled later via                                                                                               
        `backfill_token_counts()`.                                                                                                                                      
        """     
        try:                                                                                                                                                            
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.count_tokens(                                                                                                                    
                model='claude-sonnet-4-6',
                messages=[{'role': 'user', 'content': content}],                                                                                                        
            )
            return response.input_tokens                                                                                                                                
        except Exception:
            return None

    def backfill_token_counts(
        self,                                                                                                                                                           
        limit: int | None = None,
        log_every: int = 50,                                                                                                                                            
    ) -> dict:  
        """Populate content_token_count where it is NULL.                                                                                                               
 
        Returns {processed, filled, errors, remaining_after}.                                                                                                           
        """     
        import logging
        log = logging.getLogger(__name__)                                                                                                                               
        with self._conn_provider.acquire() as conn:
            q = 'SELECT id, content FROM documents WHERE content_token_count IS NULL'                                                                                   
            if limit is not None:                                                                                                                                       
                q += f' LIMIT {int(limit)}'
            rows = conn.execute(q).fetchall()                                                                                                                           
                
        processed = 0                                                                                                                                                   
        filled = 0
        errors = 0
        for doc_id, content in rows:
            processed += 1                                                                                                                                              
            count = self._count_tokens(content)
            if count is None:                                                                                                                                           
                errors += 1
            else:
                with self._conn_provider.acquire() as conn:                                                                                                             
                    conn.execute(
                        'UPDATE documents SET content_token_count = ? WHERE id = ?',                                                                                    
                        (count, doc_id),                                                                                                                                
                    )
                filled += 1                                                                                                                                             
            if processed % log_every == 0:
                log.info(
                    'backfill_token_counts: %d processed, %d filled, %d errors',
                    processed, filled, errors,                                                                                                                          
                )
                                                                                                                                                                        
        with self._conn_provider.acquire() as conn:
            remaining_after = conn.execute(
                'SELECT COUNT(*) FROM documents WHERE content_token_count IS NULL'                                                                                      
            ).fetchone()[0]
                                                                                                                                                                        
        return {
            'processed': processed,
            'filled': filled,
            'errors': errors,                                                                                                                                           
            'remaining_after': remaining_after,
        }                                                                                                                                                               
                
    def _insert_document(self, doc: Document) -> None:
        """Insert a document into the database."""
        token_count = self._count_tokens(doc.content)
        with self._conn_provider.acquire() as conn:
            conn.execute(
                '''INSERT INTO documents
                   (id, content_type, title, content, source_files, embedding, created_at, updated_at, metadata, source_name, content_token_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    doc.id,
                    doc.content_type,
                    doc.title,
                    doc.content,
                    doc.source_files_json(),
                    doc.embedding_bytes(),
                    doc.created_at,
                    doc.updated_at,
                    doc.metadata_json(),
                    doc.source_name,
                    token_count,
                ),
            )

    def get_document(self, doc_id: str) -> Document | None:
        """Get a document by ID.

        Args:
            doc_id: The document ID.

        Returns:
            The Document if found, None otherwise.
        """
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                '''SELECT id, content_type, title, content, source_files, embedding,
                          created_at, updated_at, metadata, source_name
                   FROM documents WHERE id = ?''',
                (doc_id,),
            ).fetchone()

        if row is None:
            return None
        return self._row_to_document(row)

    def _row_to_document(self, row: tuple[object, ...]) -> Document:
        """Convert a database row to a Document."""
        return Document(
            id=str(row[0]),
            content_type=row[1],  # type: ignore[arg-type]
            title=str(row[2]),
            content=str(row[3]),
            source_files=row[4],  # type: ignore[arg-type]
            embedding=row[5],  # type: ignore[arg-type]
            created_at=str(row[6]),
            updated_at=str(row[7]),
            metadata=row[8],  # type: ignore[arg-type]
            source_name=str(row[9]) if len(row) > 9 and row[9] else None,
        )

    def list_documents(
        self,
        content_type: ContentType | None = None,
        limit: int | None = None,
    ) -> list[Document]:
        """List documents, optionally filtered by content type.

        Args:
            content_type: Filter by content type.
            limit: Maximum number of documents to return.

        Returns:
            List of matching documents.
        """
        query = 'SELECT id, content_type, title, content, source_files, embedding, created_at, updated_at, metadata, source_name FROM documents'
        params: list[object] = []

        if content_type is not None:
            query += ' WHERE content_type = ?'
            params.append(content_type)

        query += ' ORDER BY updated_at DESC'

        if limit is not None:
            query += ' LIMIT ?'
            params.append(limit)

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_document(row) for row in rows]

    def list_documents_lite(
        self,
        content_type: ContentType | None = None,
    ) -> list['DocumentMeta']:
        """List document metadata only — no content or embedding loaded.

        ~96% less memory than list_documents() (0.3 MB vs 8.1 MB for 372 docs).
        Use this when you only need titles, types, source files, or metadata.
        """
        from schema import DocumentMeta

        query = 'SELECT id, content_type, title, source_files, metadata, source_name FROM documents'
        params: list[object] = []

        if content_type is not None:
            query += ' WHERE content_type = ?'
            params.append(content_type)

        query += ' ORDER BY updated_at DESC'

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            DocumentMeta(
                id=str(row[0]),
                content_type=row[1],
                title=str(row[2]),
                source_files=json.loads(row[3]) if row[3] else [],
                metadata=json.loads(row[4]) if row[4] else {},
                source_name=str(row[5]) if row[5] else None,
            )
            for row in rows
        ]

    def get_embeddings_for_ids(self, doc_ids: list[str]) -> dict[str, 'NDArray[np.float32]']:
        """Load only embeddings for specific document IDs.

        Returns a dict mapping doc_id to embedding (skipping docs without embeddings).
        Much lighter than loading full documents when only embeddings are needed for ranking.
        """
        if not doc_ids:
            return {}

        placeholders = ','.join('?' * len(doc_ids))
        result: dict[str, NDArray[np.float32]] = {}
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'SELECT id, embedding FROM documents WHERE id IN ({placeholders}) AND embedding IS NOT NULL',
                doc_ids,
            ).fetchall()
        for row in rows:
            emb_blob = row[1]
            if emb_blob:
                result[str(row[0])] = np.frombuffer(emb_blob, dtype=np.float32).copy()
        return result

    def get_documents_batch(self, doc_ids: list[str]) -> list[Document]:
        """Fetch multiple documents in a single query.

        More efficient than calling get_document() per ID.
        """
        if not doc_ids:
            return []

        placeholders = ','.join('?' * len(doc_ids))
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'''SELECT id, content_type, title, content, source_files, embedding,
                           created_at, updated_at, metadata, source_name
                    FROM documents WHERE id IN ({placeholders})''',
                doc_ids,
            ).fetchall()

        # Preserve order matching input doc_ids
        doc_map = {str(row[0]): self._row_to_document(row) for row in rows}
        return [doc_map[did] for did in doc_ids if did in doc_map]

    def update_document(
        self,
        doc_id: str,
        *,
        title: str | None = None,
        content: str | None = None,
        source_files: list[str] | None = None,
        embedding: NDArray[np.float32] | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Document | None:
        """Update an existing document.

        Args:
            doc_id: The document ID to update.
            title: New title (if provided).
            content: New content (if provided).
            source_files: New source files (if provided).
            embedding: New embedding (if provided).
            metadata: New metadata (if provided).

        Returns:
            The updated Document, or None if not found.
        """
        # Validate non-None inputs
        if title is not None and not title.strip():
            raise ValueError('Document title must not be empty')
        if content is not None and not content.strip():
            raise ValueError('Document content must not be empty')

        existing = self.get_document(doc_id)
        if existing is None:
            return None

        updates: list[str] = []
        params: list[object] = []

        if title is not None:
            updates.append('title = ?')
            params.append(title)
        if content is not None:
            updates.append('content = ?')
            params.append(content)
        if source_files is not None:
            updates.append('source_files = ?')
            params.append(json.dumps(source_files))
        if embedding is not None:
            updates.append('embedding = ?')
            params.append(embedding.tobytes())
        if metadata is not None:
            updates.append('metadata = ?')
            params.append(json.dumps(metadata))

        if updates:
            from schema import _now_iso
            updates.append('updated_at = ?')
            params.append(_now_iso())
            params.append(doc_id)

            with self._conn_provider.acquire() as conn:
                conn.execute(
                    f'UPDATE documents SET {", ".join(updates)} WHERE id = ?',
                    params,
                )

        return self.get_document(doc_id)

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks.

        Args:
            doc_id: The document ID to delete.

        Returns:
            True if document was deleted, False if not found.
        """
        with self._conn_provider.acquire() as conn:
            # Chunks are deleted automatically via CASCADE
            cursor = conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))
            return cursor.rowcount > 0

    # Chunk operations

    def add_chunk(self, chunk: Chunk) -> None:
        """Add a chunk to the database.

        Args:
            chunk: The Chunk to add.
        """
        with self._conn_provider.acquire() as conn:
            conn.execute(
                '''INSERT INTO chunks (id, document_id, chunk_index, content, embedding)
                   VALUES (?, ?, ?, ?, ?)''',
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.embedding_bytes(),
                ),
            )

    def add_chunks_batch(self, chunks: list[Chunk]) -> None:
        """Add multiple chunks in a single transaction.

        Much faster than calling add_chunk() per chunk — one transaction
        instead of N transactions.
        """
        if not chunks:
            return
        with self._conn_provider.acquire() as conn:
            conn.executemany(
                '''INSERT INTO chunks (id, document_id, chunk_index, content, embedding)
                   VALUES (?, ?, ?, ?, ?)''',
                [
                    (c.id, c.document_id, c.chunk_index, c.content, c.embedding_bytes())
                    for c in chunks
                ],
            )

    def get_chunks(self, document_id: str) -> list[Chunk]:
        """Get all chunks for a document.

        Args:
            document_id: The parent document ID.

        Returns:
            List of Chunks ordered by chunk_index.
        """
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                '''SELECT id, document_id, chunk_index, content, embedding
                   FROM chunks WHERE document_id = ? ORDER BY chunk_index''',
                (document_id,),
            ).fetchall()

        return [
            Chunk(
                id=str(row[0]),
                document_id=str(row[1]),
                chunk_index=int(row[2]),
                content=str(row[3]),
                embedding=row[4],
            )
            for row in rows
        ]

    def delete_chunks(self, document_id: str) -> int:
        """Delete all chunks for a document.

        Args:
            document_id: The parent document ID.

        Returns:
            Number of chunks deleted.
        """
        with self._conn_provider.acquire() as conn:
            cursor = conn.execute('DELETE FROM chunks WHERE document_id = ?', (document_id,))
            return cursor.rowcount

    # Section operations

    def store_sections(self, document_id: str, sections: list[Section]) -> None:
        """Replace all sections for a document in a single transaction."""
        with self._conn_provider.acquire() as conn:
            conn.execute('DELETE FROM sections WHERE document_id = ?', (document_id,))
            if sections:
                conn.executemany(
                    '''INSERT INTO sections (document_id, idx, heading, description, content, embedding)
                       VALUES (?, ?, ?, ?, ?, ?)''',
                    [
                        (s.document_id, s.index, s.heading, s.description, s.content, s.embedding_bytes())
                        for s in sections
                    ],
                )

    def get_sections(self, document_id: str) -> list[Section]:
        """Get all sections for a document, ordered by index."""
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                '''SELECT document_id, idx, heading, description, content, embedding
                   FROM sections WHERE document_id = ? ORDER BY idx''',
                (document_id,),
            ).fetchall()
        return [
            Section(
                document_id=str(row[0]), index=int(row[1]),
                heading=str(row[2]), description=str(row[3]),
                content=str(row[4]), embedding=row[5],
            )
            for row in rows
        ]

    def get_sections_batch(self, doc_ids: list[str]) -> dict[str, list[Section]]:
        """Get sections for multiple documents in a single query."""
        if not doc_ids:
            return {}
        placeholders = ','.join('?' * len(doc_ids))
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'''SELECT document_id, idx, heading, description, content, embedding
                    FROM sections WHERE document_id IN ({placeholders}) ORDER BY document_id, idx''',
                doc_ids,
            ).fetchall()
        result: dict[str, list[Section]] = {did: [] for did in doc_ids}
        for row in rows:
            doc_id = str(row[0])
            if doc_id in result:
                result[doc_id].append(Section(
                    document_id=doc_id, index=int(row[1]),
                    heading=str(row[2]), description=str(row[3]),
                    content=str(row[4]), embedding=row[5],
                ))
        return result

    def get_section_embeddings_for_doc(self, document_id: str) -> list[tuple[int, NDArray[np.float32]]]:
        """Load only section index + embedding pairs for a document.

        Returns list of (section_index, embedding) tuples, skipping sections without embeddings.
        """
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                '''SELECT idx, embedding FROM sections
                   WHERE document_id = ? AND embedding IS NOT NULL ORDER BY idx''',
                (document_id,),
            ).fetchall()
        return [
            (int(row[0]), np.frombuffer(row[1], dtype=np.float32).copy())
            for row in rows if row[1]
        ]

    # Database maintenance

    def vacuum(self) -> None:
        """Optimize the database file size."""
        with self._conn_provider.acquire() as conn:
            conn.execute('VACUUM')

    # Sync state operations

    def get_sync_state(self, source_name: str) -> tuple[str, str] | None:
        """Get the last synced git hash for a source.

        Args:
            source_name: Name of the source (e.g., 'mylib').

        Returns:
            Tuple of (git_hash, synced_at) or None if never synced.
        """
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                'SELECT git_hash, synced_at FROM sync_state WHERE source_name = ?',
                (source_name,),
            ).fetchone()

        if row is None:
            return None
        return (str(row[0]), str(row[1]))

    def set_sync_state(self, source_name: str, git_hash: str) -> None:
        """Set the last synced git hash for a source.

        Args:
            source_name: Name of the source (e.g., 'mylib').
            git_hash: The git commit hash that was synced.
        """
        from schema import _now_iso

        with self._conn_provider.acquire() as conn:
            conn.execute(
                '''INSERT INTO sync_state (source_name, git_hash, synced_at)
                   VALUES (?, ?, ?)
                   ON CONFLICT(source_name) DO UPDATE SET
                   git_hash = excluded.git_hash,
                   synced_at = excluded.synced_at''',
                (source_name, git_hash, _now_iso()),
            )
