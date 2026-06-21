"""Schema definitions for the Claude Library.

This module defines the data structures used to store and retrieve
LLM-generated documentation about the mylib codebase.
"""
from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Literal, get_args

import numpy as np
from attrs import field, frozen
from numpy.typing import NDArray

__all__ = [
    'ARIADNE_NAMESPACE',
    'CATALOG_KINDS',
    'CATALOG_KIND_ELEMENT',
    'CATALOG_KIND_FILE_INDEX',
    'CONTENT_TYPES',
    'EMBEDDING_DIM',
    'CatalogKind',
    'Chunk',
    'ContentType',
    'Document',
    'DocumentMeta',
    'SearchResult',
    'Section',
    'doc_id_for',
    'generate_deterministic_id',
]

ContentType = Literal[
    'explanation', 'architecture', 'qa', 'diagram', 'catalog',
    'finding', 'gotcha', 'theme',
    # Role-aware optional layer: cached LLM-adapted response for a
    # non-default audience (e.g., product_manager). See
    # designs/role-aware-responses.md.
    'audience_response',
]
CONTENT_TYPES: tuple[str, ...] = get_args(ContentType)

# Discriminator stored in ``metadata['kind']`` on ``content_type='catalog'``
# docs: one ``element`` doc per extracted symbol (embedded, the searchable
# unit) and one ``file_index`` doc per source file (derived index data —
# stored unembedded, excluded from export/import, regenerated on import).
CatalogKind = Literal['element', 'file_index']
CATALOG_KIND_ELEMENT: CatalogKind = 'element'
CATALOG_KIND_FILE_INDEX: CatalogKind = 'file_index'
CATALOG_KINDS: tuple[str, ...] = get_args(CatalogKind)

EMBEDDING_DIM = 3072  # OpenAI text-embedding-3-small dimension


def _generate_id() -> str:
    """Generate a unique document ID."""
    return str(uuid.uuid4())


# Namespace UUID for Ariadne document IDs (fixed, used for UUID5 generation)
ARIADNE_NAMESPACE = uuid.UUID('a1b2c3d4-e5f6-7890-abcd-ef1234567890')


def generate_deterministic_id(content_type: str, title: str) -> str:
    """Generate a deterministic document ID based on content type and title.

    Older API kept for catalog-sync (which keys on qualified_name and
    file path through ``content_type``-prefixed strings). For LLM-generated
    docs use ``doc_id_for`` instead — it's keyed on ``(source, type, file)``
    so title collisions don't cause duplicate IDs.

    Args:
        content_type: The document type (e.g., 'explanation', 'architecture').
        title: The document title.

    Returns:
        A deterministic UUID5 string.
    """
    name = f'{content_type}:{title}'
    return str(uuid.uuid5(ARIADNE_NAMESPACE, name))


def doc_id_for(
    source_name: str,
    content_type: str,
    primary_key: str,
) -> str:
    """Deterministic document ID keyed on source + content type + file/group.

    Used by the orchestrator to give every (source, content_type, file)
    triple a stable UUID. Re-running ``ariadne generate`` for an existing
    file therefore updates the same row instead of creating a duplicate.

    Args:
        source_name: Source name (from ariadne.yaml).
        content_type: Doc type literal ('explanation', 'architecture', etc.).
        primary_key: Stable per-doc identifier — typically the source
            file's path relative to the source root for per-file docs,
            or ``"group:<package_name>"`` / ``"topic:<title>"`` for
            multi-file docs (where one file path can't represent the doc).

    Returns:
        A deterministic UUID5 string. Two calls with the same args always
        produce the same UUID.
    """
    name = f'{source_name}:{content_type}:{primary_key}'
    return str(uuid.uuid5(ARIADNE_NAMESPACE, name))


def _now_iso() -> str:
    """Return current UTC time in ISO format."""
    return datetime.now(UTC).isoformat()


def _source_files_converter(value: list[str] | str | None) -> list[str]:
    """Convert source_files to a list of strings."""
    if value is None:
        return []
    if isinstance(value, str):
        return json.loads(value)
    return list(value)


def _metadata_converter(value: dict[str, object] | str | None) -> dict[str, object]:
    """Convert metadata to a dict."""
    if value is None:
        return {}
    if isinstance(value, str):
        return dict(json.loads(value))
    return dict(value)


def _embedding_converter(value: NDArray[np.float32] | bytes | None) -> NDArray[np.float32] | None:
    """Convert embedding from bytes (SQLite storage) to numpy array."""
    if value is None:
        return None
    if isinstance(value, bytes):
        return np.frombuffer(value, dtype=np.float32)
    return value


@frozen
class DocumentMeta:
    """Lightweight document metadata — no content or embedding.

    Use this instead of Document when you only need to check titles,
    types, source files, or metadata. Avoids loading 15KB content +
    6KB embedding per doc.
    """
    id: str
    content_type: ContentType = 'explanation'
    title: str = ''
    source_files: list[str] = field(factory=list, converter=_source_files_converter)
    metadata: dict[str, object] = field(factory=dict, converter=_metadata_converter)
    source_name: str | None = None


@frozen
class Document:
    """A document stored in the Claude Library.

    Documents contain LLM-generated explanations, architecture descriptions,
    Q&A pairs, or diagrams about the mylib codebase.

    Attributes:
        id: Unique identifier for the document.
        content_type: Type of content ('explanation', 'architecture', 'qa', 'diagram').
        title: Human-readable title.
        content: Full markdown content.
        source_files: List of related source file paths.
        embedding: Vector embedding of the content (1536 dimensions for OpenAI).
        created_at: ISO timestamp of creation.
        updated_at: ISO timestamp of last update.
        metadata: Additional JSON-serializable metadata. Reserved keys:
            - status: Document status ("stable", "experimental", "deprecated")
            - branches: List of git branch glob patterns where doc should appear
            - feature: Semantic feature name for natural language matching
            - aliases: Alternative names users might use to reference this feature
            - supersedes: Document ID that this document supersedes (for conflict resolution)
            - source_name: Name of the source this document was generated from
    """
    id: str = field(factory=_generate_id)
    content_type: ContentType = 'explanation'
    title: str = ''
    content: str = ''
    source_files: list[str] = field(factory=list, converter=_source_files_converter)
    embedding: NDArray[np.float32] | None = field(default=None, converter=_embedding_converter)
    created_at: str = field(factory=_now_iso)
    updated_at: str = field(factory=_now_iso)
    metadata: dict[str, object] = field(factory=dict, converter=_metadata_converter)
    source_name: str | None = None

    def __repr__(self) -> str:
        """Concise repr without dumping full content."""
        sf = f', files={len(self.source_files)}' if self.source_files else ''
        emb = ', has_embedding' if self.embedding is not None else ''
        return f'Document(id={self.id!r}, type={self.content_type!r}, title={self.title!r}{sf}{emb})'

    def source_files_json(self) -> str:
        """Return source_files as JSON string for SQLite storage."""
        return json.dumps(self.source_files)

    def metadata_json(self) -> str:
        """Return metadata as JSON string for SQLite storage."""
        return json.dumps(self.metadata)

    def embedding_bytes(self) -> bytes | None:
        """Return embedding as bytes for SQLite storage."""
        if self.embedding is None:
            return None
        return self.embedding.tobytes()


@frozen
class Section:
    """A logical section of a document for targeted retrieval.

    Documents are split by ``## `` headings at generation time. Each section
    stores a one-line description and its own embedding so that search can
    return only the sections relevant to a query instead of the full document.

    Attributes:
        document_id: ID of the parent document.
        index: Position of this section within the document (0-based).
        heading: The section heading text (without the ``## `` prefix).
        description: One-line summary used for embedding-based matching.
        content: Full markdown content of the section (including heading).
        embedding: Vector embedding of the description.
    """
    document_id: str = ''
    index: int = 0
    heading: str = ''
    description: str = ''
    content: str = ''
    embedding: NDArray[np.float32] | None = field(default=None, converter=_embedding_converter)

    def embedding_bytes(self) -> bytes | None:
        """Return embedding as bytes for SQLite storage."""
        if self.embedding is None:
            return None
        return self.embedding.tobytes()


@frozen
class Chunk:
    """A chunk of a document for fine-grained semantic search.

    Large documents are split into chunks to enable more precise
    semantic matching of queries to relevant portions.

    Attributes:
        id: Unique identifier for the chunk.
        document_id: ID of the parent document.
        chunk_index: Position of this chunk within the document.
        content: Text content of this chunk.
        embedding: Vector embedding of the chunk content.
    """
    id: str = field(factory=_generate_id)
    document_id: str = ''
    chunk_index: int = 0
    content: str = ''
    embedding: NDArray[np.float32] | None = field(default=None, converter=_embedding_converter)

    def embedding_bytes(self) -> bytes | None:
        """Return embedding as bytes for SQLite storage."""
        if self.embedding is None:
            return None
        return self.embedding.tobytes()


@frozen
class SearchResult:
    """A search result with relevance score.

    Attributes:
        document: The matching document.
        score: Cosine similarity score (0 to 1, higher is more similar).
        chunk: The matching chunk, if search was at chunk level.
    """
    document: Document
    score: float
    chunk: Chunk | None = None
