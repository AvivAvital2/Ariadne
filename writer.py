"""Content ingestion and chunking for the Ariadne library.

This module provides functionality to add documents with automatic
text chunking and embedding generation.

**Operates below the closure chokepoint.** ``LibraryWriter`` is the
write-side companion to ``Library`` — embedding refresh, batch
rebuilds, and ingestion helpers operate over the whole library by
design (they're maintenance ops, not request-boundary reads). Raw
``self.library.X(...)`` access here is intentional — see
``designs/directional-closure-scoping.md`` § "Library-internal modules
— legitimately unscoped".
"""
from __future__ import annotations

__all__ = ['ChunkConfig', 'LibraryWriter', 'chunk_text', 'split_into_sections']

import re
from typing import TYPE_CHECKING

from attrs import frozen

from diagram_format import fence_dot
from embedding import EmbeddingConfig, EmbeddingService, doc_embedding_text
from library import Library
from schema import CATALOG_KIND_FILE_INDEX, Chunk, ContentType, Document, Section

if TYPE_CHECKING:
    pass

# Default chunking parameters
DEFAULT_CHUNK_SIZE = 500  # characters
DEFAULT_CHUNK_OVERLAP = 50  # characters
EMBED_BATCH_SIZE = 100
EMBED_MAX_CONCURRENT = 3


@frozen
class ChunkConfig:
    """Configuration for text chunking.

    Attributes:
        chunk_size: Target size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
        split_on_paragraphs: Whether to prefer splitting on paragraph boundaries.
    """
    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    split_on_paragraphs: bool = True


def chunk_text(text: str, config: ChunkConfig | None = None) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to chunk.
        config: Chunking configuration.

    Returns:
        List of text chunks.
    """
    if config is None:
        config = ChunkConfig()

    if not text or len(text) <= config.chunk_size:
        return [text] if text else []

    chunks: list[str] = []

    if config.split_on_paragraphs:
        # Split on paragraph boundaries first
        paragraphs = re.split(r'\n\n+', text)
        current_chunk = ''

        for para in paragraphs:
            if len(current_chunk) + len(para) + 2 <= config.chunk_size:
                if current_chunk:
                    current_chunk += '\n\n' + para
                else:
                    current_chunk = para
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                # Handle paragraphs longer than chunk_size
                if len(para) > config.chunk_size:
                    # Split long paragraph by sentences or fixed size
                    para_chunks = _split_long_text(para, config.chunk_size, config.chunk_overlap)
                    chunks.extend(para_chunks[:-1])
                    current_chunk = para_chunks[-1] if para_chunks else ''
                else:
                    current_chunk = para

        if current_chunk:
            chunks.append(current_chunk)
    else:
        # Simple fixed-size chunking with overlap
        chunks = _split_long_text(text, config.chunk_size, config.chunk_overlap)

    return chunks


def _split_long_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with overlap.

    Handles edge cases: text with no spaces/punctuation, overlap >= chunk_size,
    and ensures forward progress on every iteration to prevent infinite loops.
    """
    if chunk_size <= 0:
        return [text] if text else []

    # Ensure overlap is strictly less than chunk_size to guarantee forward progress
    overlap = min(overlap, chunk_size - 1)

    chunks: list[str] = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # Try to find a good break point (sentence end, word boundary)
        if end < len(text):
            # Look for sentence end
            for punct in ['. ', '! ', '? ', '\n']:
                last_punct = text.rfind(punct, start, end)
                if last_punct > start + chunk_size // 2:
                    end = last_punct + len(punct)
                    break
            else:
                # Fall back to word boundary
                last_space = text.rfind(' ', start, end)
                if last_space > start + chunk_size // 2:
                    end = last_space + 1

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start forward — guarantee at least 1 char progress
        new_start = end - overlap if end < len(text) else end
        if new_start <= start:
            new_start = start + 1  # Force forward progress
        start = new_start

    return chunks


def split_into_sections(content: str) -> list[tuple[str, str]]:
    """Split markdown content into sections by ``## `` headings.

    Returns a list of ``(heading, section_content)`` tuples. The first
    element may have an empty heading if the document starts with text
    before the first ``## `` heading.  ``section_content`` includes the
    heading line itself.
    """
    # Split on lines that start with ## (but not ### or deeper)
    parts = re.split(r'(?m)^(?=## (?!#))', content)
    sections: list[tuple[str, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Extract heading from first line
        first_line = part.split('\n', 1)[0]
        heading = re.sub(r'^##\s+', '', first_line).strip() if first_line.startswith('## ') else ''
        sections.append((heading, part))
    return sections


class LibraryWriter:
    """High-level interface for adding content to the Ariadne library.

    This class handles:
    - Document creation with automatic embedding generation
    - Text chunking for better search granularity
    - Batch processing for efficiency

    Example:
        >>> async with LibraryWriter(library) as writer:
        ...     await writer.add_explanation(
        ...         title="LLM Caching",
        ...         content="...",
        ...         source_files=["mylib/core.py"]
        ...     )
    """

    def __init__(
        self,
        library: Library,
        embedding_config: EmbeddingConfig | None = None,
        chunk_config: ChunkConfig | None = None,
    ) -> None:
        """Initialize the writer.

        Args:
            library: The Ariadne library to write to.
            embedding_config: Configuration for embedding generation.
            chunk_config: Configuration for text chunking.
        """
        self.library = library
        self.embedding_config = embedding_config
        self.chunk_config = chunk_config or ChunkConfig()
        self._embedding_service: EmbeddingService | None = None

    async def _get_embedding_service(self) -> EmbeddingService:
        """Get or create the embedding service."""
        if self._embedding_service is None:
            self._embedding_service = EmbeddingService(self.embedding_config)
        return self._embedding_service

    async def close(self) -> None:
        """Close resources."""
        if self._embedding_service is not None:
            await self._embedding_service.close()
            self._embedding_service = None

    async def __aenter__(self) -> LibraryWriter:
        return self

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def add_document(
        self,
        content_type: ContentType,
        title: str,
        content: str,
        source_files: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        create_chunks: bool = True,
        doc_id: str | None = None,
        source_name: str | None = None,
    ) -> Document:
        """Add a document with automatic embedding generation.

        Args:
            content_type: Type of content.
            title: Document title.
            content: Markdown content.
            source_files: Related source file paths.
            metadata: Additional metadata.
            create_chunks: Whether to create searchable chunks.
            source_name: Source attribution (forwarded to library).

        Returns:
            The created Document.
        """
        service = await self._get_embedding_service()

        # Generate embedding for the full document
        # Use title + first part of content for the document-level embedding
        doc_text = doc_embedding_text(title, content)
        doc_embedding = await service.embed(doc_text)

        # Create the document
        doc = self.library.add_document(
            content_type=content_type,
            title=title,
            content=content,
            source_files=source_files,
            embedding=doc_embedding,
            metadata=metadata,
            doc_id=doc_id,
            source_name=source_name,
        )

        # Create chunks if requested
        if create_chunks and len(content) > self.chunk_config.chunk_size:
            await self._create_chunks(doc.id, content, service)

        # Create sections for targeted retrieval
        await self._create_sections(doc.id, content, service)

        return doc

    async def _create_chunks(
        self,
        document_id: str,
        content: str,
        service: EmbeddingService,
    ) -> list[Chunk]:
        """Create and store chunks for a document."""
        # Split into chunks
        chunk_texts = chunk_text(content, self.chunk_config)

        if not chunk_texts:
            return []

        # Generate embeddings for all chunks
        chunk_embeddings = await service.embed_batch(chunk_texts)

        # Create and store chunks in batch (single transaction)
        chunks = [
            Chunk(document_id=document_id, chunk_index=i, content=text, embedding=embedding)
            for i, (text, embedding) in enumerate(zip(chunk_texts, chunk_embeddings))
        ]
        self.library.add_chunks_batch(chunks)

        return chunks

    async def _create_sections(
        self,
        document_id: str,
        content: str,
        service: EmbeddingService,
    ) -> list[Section]:
        """Split document into sections, generate description embeddings, and store."""
        raw_sections = split_into_sections(content)
        if len(raw_sections) <= 1:
            # Single-section documents don't benefit from section-level retrieval
            return []

        # Use headings as descriptions (concise, already one-line)
        descriptions = [heading or 'Introduction' for heading, _ in raw_sections]
        embeddings = await service.embed_batch(descriptions)

        sections = [
            Section(
                document_id=document_id,
                index=i,
                heading=heading or 'Introduction',
                description=descriptions[i],
                content=section_content,
                embedding=emb,
            )
            for i, ((heading, section_content), emb) in enumerate(zip(raw_sections, embeddings))
        ]
        self.library.store_sections(document_id, sections)
        return sections

    # Convenience methods for specific content types

    async def add_explanation(
        self,
        title: str,
        content: str,
        source_files: list[str] | None = None,
        metadata: dict[str, object] | None = None,
        source_name: str | None = None,
        doc_id: str | None = None,
    ) -> Document:
        """Add an explanation document.

        Explanations describe how specific parts of the codebase work.
        """
        return await self.add_document(
            content_type='explanation',
            title=title,
            content=content,
            source_files=source_files,
            metadata=metadata,
            source_name=source_name,
            doc_id=doc_id,
        )

    async def add_architecture(
        self,
        title: str,
        content: str,
        source_files: list[str] | None = None,
        source_name: str | None = None,
        doc_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Document:
        """Add an architecture document.

        Architecture documents describe system design and component relationships.
        """
        return await self.add_document(
            content_type='architecture',
            title=title,
            content=content,
            source_files=source_files,
            source_name=source_name,
            doc_id=doc_id,
            metadata=metadata,
        )

    async def add_qa(
        self,
        question: str,
        answer: str,
        source_files: list[str] | None = None,
        source_name: str | None = None,
        doc_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Document:
        """Add a Q&A document.

        Q&A documents capture questions and their answers about the codebase.
        """
        content = f'## Question\n\n{question}\n\n## Answer\n\n{answer}'

        return await self.add_document(
            content_type='qa',
            title=question,
            content=content,
            source_files=source_files,
            metadata={**(metadata or {}), 'question': question, 'answer': answer},
            source_name=source_name,
            doc_id=doc_id,
        )

    async def add_diagram(
        self,
        title: str,
        description: str,
        dot_code: str,
        source_files: list[str] | None = None,
        source_name: str | None = None,
        doc_id: str | None = None,
        metadata: dict[str, object] | None = None,
    ) -> Document:
        """Add a diagram document.

        Diagram documents contain a Graphviz DOT diagram with a description. DOT
        renders to PNG with a tiny native binary (`dot`), so the bridge can turn
        it into an image for Slack. The raw DOT is kept in metadata for reuse.
        """
        content = f'{description}\n\n{fence_dot(dot_code)}'

        return await self.add_document(
            content_type='diagram',
            title=title,
            content=content,
            source_files=source_files,
            metadata={**(metadata or {}), 'dot_code': dot_code},
            create_chunks=False,
            source_name=source_name,
            doc_id=doc_id,
        )

    async def update_document_embedding(self, doc_id: str) -> Document | None:
        """Regenerate embedding for an existing document.

        Useful when embeddings need to be refreshed after content changes.
        """
        doc = self.library.get_document(doc_id)
        if doc is None:
            return None

        service = await self._get_embedding_service()
        doc_text = doc_embedding_text(doc.title, doc.content)
        new_embedding = await service.embed(doc_text)

        return self.library.update_document(doc_id, embedding=new_embedding)

    async def rebuild_all_embeddings(
        self, only_missing: bool = False, on_progress=None,
    ) -> int:
        """Regenerate embeddings; return the count successfully (re)embedded.

        With only_missing, embeds only docs whose embedding is NULL (new
        or content-changed); otherwise the whole library. Batches run
        concurrently (capped at EMBED_MAX_CONCURRENT); a batch that fails
        after the client's retries is logged and skipped, so one failure
        cannot abort the run and skipped docs stay NULL for a re-run.

        ``on_progress(completed, total)`` is invoked after each batch with the
        running done-count and the total work size, so a caller (the CLI) can
        drive a progress bar / ETA. When omitted, progress is silent.
        """
        import asyncio

        docs = (
            self.library.list_documents_without_embedding()
            if only_missing
            else self.library.list_documents()
        )
        # file_index docs are pure derived index data, stored deliberately
        # unembedded — a full rebuild must not re-embed them (only_missing
        # already excludes them at the query level).
        docs = [d for d in docs if d.metadata.get('kind') != CATALOG_KIND_FILE_INDEX]
        service = await self._get_embedding_service()

        doc_texts = [doc_embedding_text(d.title, d.content) for d in docs]
        batches = []
        for i in range(0, len(docs), EMBED_BATCH_SIZE):
            batches.append((docs[i:i + EMBED_BATCH_SIZE], doc_texts[i:i + EMBED_BATCH_SIZE]))

        semaphore = asyncio.Semaphore(EMBED_MAX_CONCURRENT)

        async def fetch_batch(batch_docs, batch_texts):
            async with semaphore:
                embeddings = await service.embed_batch(batch_texts)
                return batch_docs, embeddings

        tasks = [fetch_batch(bd, bt) for bd, bt in batches]
        count = 0
        failed = 0
        total = len(docs)
        for coro in asyncio.as_completed(tasks):
            try:
                batch_docs, batch_embeddings = await coro
            except Exception as exc:
                failed += 1
                print(f'  rebuild: a batch failed after retries, skipping ({exc})')
                continue
            for doc, emb in zip(batch_docs, batch_embeddings):
                self.library.update_document(doc.id, embedding=emb)
                self.library.delete_chunks(doc.id)
                if len(doc.content) > self.chunk_config.chunk_size:
                    await self._create_chunks(doc.id, doc.content, service)
                count += 1
            if on_progress is not None:
                on_progress(count, total)
        if failed:
            print(f'  rebuild: {failed} batch(es) failed — re-run to embed the remainder')
        return count

    async def rebuild_all_embeddings_batch(
        self, strategy, only_missing: bool = False, on_progress=None,
    on_submit=None) -> int:
        """Batch-API counterpart of rebuild_all_embeddings — same target
        set, same persistence, ~50% of the price, minutes-to-hours latency.

        Submits ONE batch job carrying grouped doc-vector requests
        (EMBED_BATCH_SIZE texts per group) plus one chunk group per
        oversized doc, polls it to a terminal state, then writes vectors.
        A group that failed server-side is skipped with a notice — its
        docs stay NULL so a live ``rebuild --only-missing`` can top up.

        ``on_progress(completed_groups, total_groups)`` mirrors the live
        path's two-argument contract.
        """
        from docgen.llm.openai_batch import EmbeddingBatchRequest

        docs = (
            self.library.list_documents_without_embedding()
            if only_missing
            else self.library.list_documents()
        )
        docs = [d for d in docs if d.metadata.get('kind') != CATALOG_KIND_FILE_INDEX]
        if not docs:
            return 0

        requests: list[EmbeddingBatchRequest] = []
        doc_groups: list[list] = []
        for i in range(0, len(docs), EMBED_BATCH_SIZE):
            group = docs[i:i + EMBED_BATCH_SIZE]
            doc_groups.append(group)
            requests.append(EmbeddingBatchRequest(
                custom_id=f'doc:{len(doc_groups) - 1}',
                texts=[doc_embedding_text(d.title, d.content) for d in group],
            ))
        chunked: dict[str, list[str]] = {}
        for d in docs:
            if len(d.content) > self.chunk_config.chunk_size:
                texts = chunk_text(d.content, self.chunk_config)
                if texts:
                    chunked[d.id] = texts
                    requests.append(EmbeddingBatchRequest(
                        custom_id=f'chunks:{d.id}', texts=texts))

        submission = await strategy.submit_batch(requests)
        if on_submit is not None:
            on_submit(submission.batch_id)
        total_groups = len(requests)

        def _poll_progress(processing: int, succeeded: int, errored: int) -> None:
            if on_progress is not None:
                on_progress(succeeded + errored, total_groups)

        await strategy.poll_batch(submission.batch_id, on_progress=_poll_progress)
        results = await strategy.fetch_batch_results(submission.batch_id)

        count = 0
        for gi, group in enumerate(doc_groups):
            vectors = results.get(f'doc:{gi}')
            if not vectors or len(vectors) != len(group):
                print(f'  rebuild: embeddings batch group doc:{gi} failed, '
                      f'skipping {len(group)} doc(s)')
                continue
            for doc, vector in zip(group, vectors):
                self.library.update_document(doc.id, embedding=vector)
                count += 1
        for doc_id, texts in chunked.items():
            vectors = results.get(f'chunks:{doc_id}')
            if not vectors or len(vectors) != len(texts):
                print(f'  rebuild: embeddings batch chunks for {doc_id} failed, '
                      f'keeping existing chunks')
                continue
            self.library.delete_chunks(doc_id)
            chunks = [
                Chunk(document_id=doc_id, chunk_index=i, content=text, embedding=vector)
                for i, (text, vector) in enumerate(zip(texts, vectors))
            ]
            self.library.add_chunks_batch(chunks)
        return count
