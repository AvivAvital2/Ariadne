"""Ariadne - Semantic knowledge library for LLM-generated documentation.

This package provides a SQLite-backed library for storing and retrieving
LLM-generated documentation about source code. It supports:

- Storing documents with semantic embeddings
- Full-text and semantic search
- Automatic text chunking for fine-grained retrieval
- Export to markdown for version control
- Import from markdown for database rebuilding
- Config file support for managing multiple source paths

Example usage:

    from pathlib import Path
    import ariadne

    # Create or open a library
    library = ariadne.Library(Path('ariadne.db'))

    # Add documents with automatic embedding generation
    async with ariadne.LibraryWriter(library) as writer:
        doc = await writer.add_explanation(
            title="LLM Caching",
            content="The LLM caching system uses SQLite...",
            source_files=["src/core/cache/base.py"]
        )

    # Search for documents
    query_embedding = ariadne.embed_sync("How does caching work?")
    results = library.search(query_embedding, k=5)

    for result in results:
        print(f"{result.score:.3f}: {result.document.title}")

    # Export to markdown
    exporter = ariadne.LibraryExporter(library)
    exporter.export_all(Path('docs/'))

    library.close()
"""
# Docgen module exports — SourceAnalyzer/ModuleMetadata/ModuleGroup were
# removed in Catalog transition Phase 4.1 along with docgen.analyzer and
# docgen.metadata. New code should use docgen.catalog_extractor +
# docgen.catalog_enrich (multi-language) instead.
from docgen import (
    ContentValidator,
    CrossRefDetector,
    DocGenerator,
    DocGenOrchestrator,
    GeneratorConfig,
    OrchestratorConfig,
    SourceRecord,
    StalenessTracker,
    ValidationResult,
)
from embedding import (
    EMBEDDING_DIM,
    EmbeddingConfig,
    EmbeddingService,
    embed_batch_sync,
    embed_sync,
)
from export import ExportConfig, LibraryExporter, import_from_markdown
from library import Library
from schema import Chunk, ContentType, Document, SearchResult
from search import batch_cosine_similarity, batch_dot_similarity, cosine_similarity, top_k_indices
from writer import ChunkConfig, LibraryWriter, chunk_text

__all__ = [
    # Core classes
    'Library',
    'Document',
    'Chunk',
    'SearchResult',
    'ContentType',
    # Embedding
    'EmbeddingService',
    'EmbeddingConfig',
    'embed_sync',
    'embed_batch_sync',
    'EMBEDDING_DIM',
    # Search
    'cosine_similarity',
    'batch_cosine_similarity',
    'batch_dot_similarity',
    'top_k_indices',
    # Writing
    'LibraryWriter',
    'ChunkConfig',
    'chunk_text',
    # Export
    'LibraryExporter',
    'ExportConfig',
    'import_from_markdown',
    # Docgen
    'DocGenerator',
    'GeneratorConfig',
    'StalenessTracker',
    'SourceRecord',
    'CrossRefDetector',
    'ContentValidator',
    'ValidationResult',
    'DocGenOrchestrator',
    'OrchestratorConfig',
]
