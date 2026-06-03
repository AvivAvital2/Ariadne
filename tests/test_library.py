"""Tests for the Ariadne library storage module."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from library import Library
from schema import Chunk


@pytest.fixture
def temp_db(tmp_path: Path) -> Path:
    """Create a temporary database path."""
    return tmp_path / 'test_library.db'


@pytest.fixture
def library(temp_db: Path) -> Library:
    """Create a test library."""
    lib = Library(temp_db)
    yield lib
    lib.close()


class TestLibrary:
    """Tests for the Library class."""

    def test_create_library(self, temp_db: Path) -> None:
        """Test creating a new library."""
        with Library(temp_db) as library:
            assert library.count_documents() == 0
            assert temp_db.exists()

    def test_add_document(self, library: Library) -> None:
        """Test adding a document."""
        doc = library.add_document(
            content_type='explanation',
            title='Test Title',
            content='Test content',
            source_files=['file.py'],
        )

        assert doc.id is not None
        assert doc.title == 'Test Title'
        assert doc.content == 'Test content'
        assert doc.source_files == ['file.py']
        assert library.count_documents() == 1

    def test_add_document_with_embedding(self, library: Library) -> None:
        """Test adding a document with an embedding."""
        embedding = np.random.rand(1536).astype(np.float32)
        doc = library.add_document(
            content_type='explanation',
            title='Test',
            content='Content',
            embedding=embedding,
        )

        retrieved = library.get_document(doc.id)
        assert retrieved is not None
        assert retrieved.embedding is not None
        assert np.allclose(retrieved.embedding, embedding)

    def test_get_document(self, library: Library) -> None:
        """Test retrieving a document by ID."""
        doc = library.add_document(
            content_type='explanation',
            title='Get Test',
            content='Content',
        )

        retrieved = library.get_document(doc.id)
        assert retrieved is not None
        assert retrieved.title == 'Get Test'

    def test_get_document_not_found(self, library: Library) -> None:
        """Test retrieving a non-existent document."""
        retrieved = library.get_document('nonexistent')
        assert retrieved is None

    def test_list_documents(self, library: Library) -> None:
        """Test listing documents."""
        library.add_document(content_type='explanation', title='Doc 1', content='test content')
        library.add_document(content_type='architecture', title='Doc 2', content='test content')
        library.add_document(content_type='explanation', title='Doc 3', content='test content')

        all_docs = library.list_documents()
        assert len(all_docs) == 3

        explanations = library.list_documents(content_type='explanation')
        assert len(explanations) == 2

        limited = library.list_documents(limit=2)
        assert len(limited) == 2

    def test_update_document(self, library: Library) -> None:
        """Test updating a document."""
        doc = library.add_document(
            content_type='explanation',
            title='Original',
            content='Original content',
        )

        updated = library.update_document(
            doc.id,
            title='Updated',
            content='Updated content',
        )

        assert updated is not None
        assert updated.title == 'Updated'
        assert updated.content == 'Updated content'
        # updated_at should change
        assert updated.updated_at != doc.updated_at

    def test_update_document_not_found(self, library: Library) -> None:
        """Test updating a non-existent document."""
        result = library.update_document('nonexistent', title='New')
        assert result is None

    def test_delete_document(self, library: Library) -> None:
        """Test deleting a document."""
        doc = library.add_document(
            content_type='explanation',
            title='To Delete',
            content='test content',
        )

        assert library.count_documents() == 1
        assert library.delete_document(doc.id) is True
        assert library.count_documents() == 0
        assert library.get_document(doc.id) is None

    def test_delete_document_not_found(self, library: Library) -> None:
        """Test deleting a non-existent document."""
        assert library.delete_document('nonexistent') is False


class TestChunkOperations:
    """Tests for chunk operations."""

    def test_add_and_get_chunks(self, library: Library) -> None:
        """Test adding and retrieving chunks."""
        doc = library.add_document(
            content_type='explanation',
            title='Test',
            content='Full content',
        )

        chunk1 = Chunk(document_id=doc.id, chunk_index=0, content='Chunk 1')
        chunk2 = Chunk(document_id=doc.id, chunk_index=1, content='Chunk 2')

        library.add_chunk(chunk1)
        library.add_chunk(chunk2)

        chunks = library.get_chunks(doc.id)
        assert len(chunks) == 2
        assert chunks[0].content == 'Chunk 1'
        assert chunks[1].content == 'Chunk 2'

    def test_delete_chunks(self, library: Library) -> None:
        """Test deleting chunks for a document."""
        doc = library.add_document(
            content_type='explanation',
            title='Test',
            content='Content',
        )

        for i in range(3):
            chunk = Chunk(document_id=doc.id, chunk_index=i, content=f'Chunk {i}')
            library.add_chunk(chunk)

        assert library.count_chunks() == 3

        deleted = library.delete_chunks(doc.id)
        assert deleted == 3
        assert library.count_chunks() == 0

    def test_cascade_delete_chunks(self, library: Library) -> None:
        """Test that deleting a document cascades to its chunks."""
        doc = library.add_document(
            content_type='explanation',
            title='Test',
            content='Content',
        )

        for i in range(2):
            chunk = Chunk(document_id=doc.id, chunk_index=i, content=f'Chunk {i}')
            library.add_chunk(chunk)

        assert library.count_chunks() == 2

        library.delete_document(doc.id)
        assert library.count_chunks() == 0


class TestSearch:
    """Tests for search operations."""

    def test_search_with_embeddings(self, library: Library) -> None:
        """Test semantic search."""
        # Create documents with embeddings
        for i in range(3):
            embedding = np.zeros(1536, dtype=np.float32)
            embedding[i] = 1.0  # Different embedding per doc
            library.add_document(
                content_type='explanation',
                title=f'Doc {i}',
                content=f'Content {i}',
                embedding=embedding,
            )

        # Search with a query embedding
        query = np.zeros(1536, dtype=np.float32)
        query[0] = 1.0  # Should match Doc 0 best

        results = library.search(query, k=3)
        assert len(results) == 3
        assert results[0].document.title == 'Doc 0'
        assert results[0].score > 0.99  # Perfect match

    def test_search_with_content_type_filter(self, library: Library) -> None:
        """Test search with content type filter."""
        embedding = np.random.rand(1536).astype(np.float32)

        library.add_document(
            content_type='explanation',
            title='Explanation',
            content='test content',
            embedding=embedding,
        )
        library.add_document(
            content_type='architecture',
            title='Architecture',
            content='test content',
            embedding=embedding,
        )

        query = np.random.rand(1536).astype(np.float32)

        all_results = library.search(query, k=5)
        assert len(all_results) == 2

        filtered = library.search(query, k=5, content_type='explanation')
        assert len(filtered) == 1
        assert filtered[0].document.content_type == 'explanation'

    def test_search_empty_library(self, library: Library) -> None:
        """Test search on empty library."""
        query = np.random.rand(1536).astype(np.float32)
        results = library.search(query, k=5)
        assert len(results) == 0


class TestStats:
    """Tests for statistics and utilities."""

    def test_count_documents_by_type(self, library: Library) -> None:
        """Test counting documents by type."""
        library.add_document(content_type='explanation', title='1', content='test content')
        library.add_document(content_type='explanation', title='2', content='test content')
        library.add_document(content_type='qa', title='3', content='test content')

        assert library.count_documents() == 3
        assert library.count_documents(content_type='explanation') == 2
        assert library.count_documents(content_type='qa') == 1
        assert library.count_documents(content_type='diagram') == 0

    def test_vacuum(self, library: Library) -> None:
        """Test database vacuum."""
        # Add and delete documents to create fragmentation
        for i in range(10):
            doc = library.add_document(
                content_type='explanation',
                title=f'Doc {i}',
                content='x' * 1000,
            )
            library.delete_document(doc.id)

        # Should not raise
        library.vacuum()

    def test_add_document_validation(self, library: Library) -> None:
        """Test that add_document rejects invalid inputs."""
        with pytest.raises(ValueError, match='title must not be empty'):
            library.add_document('explanation', '', 'content')

        with pytest.raises(ValueError, match='title must not be empty'):
            library.add_document('explanation', '   ', 'content')

        with pytest.raises(ValueError, match='content must not be empty'):
            library.add_document('explanation', 'title', '')

        with pytest.raises(ValueError, match='content must not be empty'):
            library.add_document('explanation', 'title', '   ')

        with pytest.raises(ValueError, match='Invalid content_type'):
            library.add_document('blog_post', 'title', 'content')

    def test_add_document_valid_types(self, library: Library) -> None:
        """Test that all valid content types are accepted."""
        for ct in ('explanation', 'architecture', 'qa', 'diagram', 'finding'):
            doc = library.add_document(ct, f'Test {ct}', f'Content for {ct}')
            assert doc.content_type == ct

    def test_update_document_validation(self, library: Library) -> None:
        """Test that update_document rejects empty title/content."""
        doc = library.add_document('finding', 'original', 'original content')

        with pytest.raises(ValueError, match='title must not be empty'):
            library.update_document(doc.id, title='')

        with pytest.raises(ValueError, match='content must not be empty'):
            library.update_document(doc.id, content='   ')

        # None values should be accepted (means "don't change")
        updated = library.update_document(doc.id, title='new title')
        assert updated is not None
        assert updated.title == 'new title'
        assert updated.content == 'original content'

    def test_graph_operations(self, library: Library, tmp_path: Path) -> None:
        """Test graph build, stats, and export on a minimal setup."""
        # Add some docs with source files
        library.add_document('explanation', 'Module A', 'Content A', source_files=[str(tmp_path / 'a.py')])
        library.add_document('architecture', 'Module B', 'Content B', source_files=[str(tmp_path / 'b.py')])

        stats = library.get_graph_stats()
        assert stats['total_edges'] == 0  # No graph built yet

        # Build graph (empty source path — no imports to detect)
        counts = library.build_graph(tmp_path)
        assert isinstance(counts, dict)

        # Export graph JSON
        data = library.export_graph_json()
        assert 'nodes' in data
        assert 'edges' in data

    def test_usage_logging(self, library: Library) -> None:
        """Test usage event logging and retrieval."""
        event_id = library.log_usage('ariadne_search', 'test query', 3)
        assert event_id > 0

        # Mark as hit
        assert library.mark_hit(event_id, 'helpful')

        # Mark nonexistent event
        assert not library.mark_hit(99999)

        # Get stats
        stats = library.get_usage_stats(days=1)
        assert stats['total_calls'] >= 1
        assert stats['total_hits'] >= 1

    def test_usage_logging_with_document_ids(self, library: Library) -> None:
        """Test that document IDs are stored in usage events."""
        doc = library.add_document('finding', 'test', 'content')
        event_id = library.log_usage('ariadne_search', 'query', 1, document_ids=[doc.id])
        assert event_id > 0

    def test_search_returns_sorted_results(self, library: Library) -> None:
        """Test that search results are sorted by similarity score."""
        # Create docs with known embeddings
        dim = 1536
        e1 = np.random.RandomState(42).randn(dim).astype(np.float32)
        e2 = np.random.RandomState(43).randn(dim).astype(np.float32)
        library.add_document('explanation', 'Doc 1', 'Content 1', embedding=e1)
        library.add_document('explanation', 'Doc 2', 'Content 2', embedding=e2)

        # Search with e1 as query — Doc 1 should be most similar
        results = library.search(e1, k=2)
        assert len(results) == 2
        assert results[0].score >= results[1].score
        assert results[0].document.title == 'Doc 1'


class TestStatsBySourceDetailed:
    """Tests for ``stats_by_source_detailed`` — per-source breakdown
    that powers the ``ariadne status`` CLI command.

    The existing ``stats_by_source`` returns per-source totals only.
    This new method enriches it with:
      - ``by_content_type``: dict mapping content_type → count
      - ``chunk_content`` / ``chunk_embed``: bytes of chunks attributed
        to this source via JOIN on documents.source_name
      - ``section_content`` / ``section_embed``: same for sections

    These let ``ariadne status`` show:
      1. WHAT each source contains (type breakdown)
      2. HOW MUCH disk it accounts for (real attribution, not just docs)

    Producer-only — slim consumer picks it up via Library re-export.
    """

    def test_returns_per_source_content_type_breakdown(
        self, library: 'Library',
    ) -> None:
        """Each source carries a ``by_content_type`` dict mapping
        content_type to count, so the CLI can render a type matrix."""
        library.add_document(
            content_type='catalog', title='c1', content='x' * 100,
            source_name='src_a',
        )
        library.add_document(
            content_type='catalog', title='c2', content='x' * 100,
            source_name='src_a',
        )
        library.add_document(
            content_type='explanation', title='e1', content='y' * 200,
            source_name='src_a',
        )
        library.add_document(
            content_type='explanation', title='e2', content='y' * 200,
            source_name='src_b',
        )

        stats = library.stats_by_source_detailed()

        assert 'src_a' in stats
        assert stats['src_a']['doc_count'] == 3
        assert stats['src_a']['by_content_type'] == {
            'catalog': 2, 'explanation': 1,
        }

        assert 'src_b' in stats
        assert stats['src_b']['doc_count'] == 1
        assert stats['src_b']['by_content_type'] == {'explanation': 1}

    def test_attributes_doc_content_and_embedding_bytes(
        self, library: 'Library',
    ) -> None:
        """``doc_content`` and ``doc_embed`` carry the byte sums of
        ``documents.content`` and ``documents.embedding`` respectively,
        so the CLI can show what fraction of the DB each source owns."""
        e1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        library.add_document(
            content_type='explanation', title='e1', content='hello',
            source_name='src_a', embedding=e1,
        )
        library.add_document(
            content_type='explanation', title='e2', content='hi there',
            source_name='src_a', embedding=e1,
        )

        stats = library.stats_by_source_detailed()

        # 5 + 8 = 13 bytes of content
        assert stats['src_a']['doc_content'] == 13
        # 2 docs × 12 bytes (3 floats × 4 bytes each) = 24
        assert stats['src_a']['doc_embed'] == 24

    def test_attributes_chunks_via_join_to_source(
        self, library: 'Library',
    ) -> None:
        """Chunks attribute to a source via JOIN on documents.source_name —
        the chunks table itself doesn't carry source_name. This is the
        load-bearing query for "where does scalaproject's 7 GB go": almost
        all of it is in chunks, not in document rows."""
        e = np.array([0.1] * 5, dtype=np.float32)  # 20 bytes
        doc = library.add_document(
            content_type='explanation', title='e1', content='hi',
            source_name='src_a',
        )
        c1 = Chunk(
            document_id=doc.id, chunk_index=0,
            content='chunk-content-A',  # 15 bytes
            embedding=e,
        )
        c2 = Chunk(
            document_id=doc.id, chunk_index=1,
            content='chunk-content-BB',  # 16 bytes
            embedding=e,
        )
        library.add_chunk(c1)
        library.add_chunk(c2)

        stats = library.stats_by_source_detailed()

        assert stats['src_a']['chunk_count'] == 2
        assert stats['src_a']['chunk_content'] == 31  # 15 + 16
        assert stats['src_a']['chunk_embed'] == 40  # 2 × 20

    def test_includes_db_size_total(self, library: 'Library') -> None:
        """The footer key ``_total`` carries the on-disk DB size for
        the CLI to compute 'overhead' = db_size - sum(per-source)."""
        library.add_document(
            content_type='explanation', title='x', content='y',
            source_name='src_a',
        )

        stats = library.stats_by_source_detailed()

        assert '_total' in stats
        assert 'db_size_bytes' in stats['_total']
        assert stats['_total']['db_size_bytes'] > 0

    def test_unknown_source_attributed_to_unknown_bucket(
        self, library: 'Library',
    ) -> None:
        """Documents with no ``source_name`` (legacy / pre-migration)
        bucket into 'unknown' rather than being dropped. Producer's
        existing ``stats_by_source`` does this; we preserve it."""
        library.add_document(
            content_type='finding', title='f', content='legacy doc',
        )

        stats = library.stats_by_source_detailed()

        assert 'unknown' in stats
        assert stats['unknown']['doc_count'] >= 1


class TestPerSourceStatsForProgress:
    """Tests for the per-source incremental APIs that power the
    ``ariadne status`` CLI's progress bar.

    The batch ``stats_by_source_detailed()`` runs as one SQL pass and
    returns nothing until done — bad UX for large DBs (scalaproject's
    chunks JOIN reads 5+ GB). Splitting into:

      - ``list_source_names()``  — fast: returns the source list to
        size the progress bar's total
      - ``stats_for_source(src)`` — same per-source detail, scoped to
        one source via the ``documents(source_name)`` index

    lets the CLI iterate with a running total and per-source updates.
    The batch method stays for callers that just want everything.
    """

    def test_list_source_names_returns_distinct_sources(
        self, library: 'Library',
    ) -> None:
        """Each source_name appears exactly once, sorted alphabetically.
        Documents with no ``source_name`` bucket as 'unknown'."""
        library.add_document(
            'explanation', 't1', 'c', source_name='src_a',
        )
        library.add_document(
            'explanation', 't2', 'c', source_name='src_a',
        )
        library.add_document(
            'explanation', 't3', 'c', source_name='src_b',
        )
        library.add_document(
            'finding', 'legacy', 'c',  # no source_name
        )

        names = library.list_source_names()

        assert names == ['src_a', 'src_b', 'unknown']

    def test_list_source_names_returns_empty_for_empty_db(
        self, library: 'Library',
    ) -> None:
        """An empty DB returns ``[]`` — not None, not a SQL error.
        CLI can still build the progress bar (with total=0)."""
        assert library.list_source_names() == []

    def test_stats_for_source_returns_per_source_detail(
        self, library: 'Library',
    ) -> None:
        """``stats_for_source(name)`` returns the same shape as one
        entry of ``stats_by_source_detailed``, scoped to that source."""
        library.add_document(
            'catalog', 'c1', 'x' * 100, source_name='src_a',
        )
        library.add_document(
            'explanation', 'e1', 'y' * 200, source_name='src_a',
        )
        library.add_document(
            'catalog', 'c2', 'z' * 50, source_name='src_b',
        )

        stats = library.stats_for_source('src_a')

        assert stats['doc_count'] == 2
        assert stats['doc_content'] == 300  # 100 + 200
        assert stats['by_content_type'] == {
            'catalog': 1, 'explanation': 1,
        }
        # Other sources are excluded.
        assert 'src_b' not in str(stats)

    def test_stats_for_source_attributes_chunks_to_named_source(
        self, library: 'Library',
    ) -> None:
        """Chunk attribution scoped to one source via the same JOIN."""
        e = np.array([0.1] * 5, dtype=np.float32)  # 20 bytes
        doc_a = library.add_document(
            'explanation', 'da', 'a', source_name='src_a',
        )
        doc_b = library.add_document(
            'explanation', 'db', 'b', source_name='src_b',
        )
        library.add_chunk(Chunk(
            document_id=doc_a.id, chunk_index=0, content='aaa', embedding=e,
        ))
        library.add_chunk(Chunk(
            document_id=doc_b.id, chunk_index=0, content='bbb', embedding=e,
        ))

        stats_a = library.stats_for_source('src_a')

        assert stats_a['chunk_count'] == 1
        assert stats_a['chunk_content'] == 3  # only 'aaa'
        assert stats_a['chunk_embed'] == 20  # only doc_a's chunk

    def test_stats_for_source_unknown_returns_zero_shape(
        self, library: 'Library',
    ) -> None:
        """A source name no document uses returns the zero-shape dict
        rather than None — caller iterates uniformly without checks."""
        library.add_document(
            'explanation', 'x', 'c', source_name='src_a',
        )

        stats = library.stats_for_source('does_not_exist')

        assert stats['doc_count'] == 0
        assert stats['doc_content'] == 0
        assert stats['doc_embed'] == 0
        assert stats['by_content_type'] == {}
        assert stats['chunk_count'] == 0

    def test_stats_for_source_handles_unknown_bucket(
        self, library: 'Library',
    ) -> None:
        """``stats_for_source('unknown')`` returns docs with NULL
        source_name — same convention as the batch method."""
        library.add_document(
            'finding', 'legacy', 'no source here',
        )

        stats = library.stats_for_source('unknown')

        assert stats['doc_count'] == 1
        assert stats['doc_content'] == len('no source here')

    def test_source_signature_is_stable_for_unchanged_source(
        self, library: 'Library',
    ) -> None:
        """``source_signature`` returns the same value for an
        unmodified source. The CLI uses this as a cache key — same
        value → cache hit, skip the expensive chunks JOIN."""
        library.add_document(
            'explanation', 't1', 'c', source_name='src_a',
        )
        library.add_document(
            'explanation', 't2', 'c', source_name='src_a',
        )

        sig1 = library.source_signature('src_a')
        sig2 = library.source_signature('src_a')

        assert sig1 == sig2
        assert isinstance(sig1, str)
        assert sig1  # non-empty

    def test_source_signature_changes_when_doc_added(
        self, library: 'Library',
    ) -> None:
        """Adding a doc changes both COUNT and MAX(updated_at), so
        the signature changes — invalidating the CLI's cache entry."""
        library.add_document(
            'explanation', 't1', 'c', source_name='src_a',
        )
        sig_before = library.source_signature('src_a')

        library.add_document(
            'explanation', 't2', 'c', source_name='src_a',
        )
        sig_after = library.source_signature('src_a')

        assert sig_before != sig_after

    def test_source_signature_changes_when_doc_modified(
        self, library: 'Library',
    ) -> None:
        """Updating a doc bumps its ``updated_at``, so the signature
        changes even when COUNT is the same."""
        import time

        doc = library.add_document(
            'explanation', 't1', 'original', source_name='src_a',
        )
        sig_before = library.source_signature('src_a')

        # Sleep briefly so the timestamp changes (TEXT column at
        # second resolution per schema).
        time.sleep(1.1)
        library.update_document(doc.id, content='modified')

        sig_after = library.source_signature('src_a')
        assert sig_before != sig_after

    def test_source_signature_isolated_per_source(
        self, library: 'Library',
    ) -> None:
        """Changes to source A don't change the signature of source B."""
        library.add_document(
            'explanation', 'a1', 'c', source_name='src_a',
        )
        library.add_document(
            'explanation', 'b1', 'c', source_name='src_b',
        )
        sig_b_before = library.source_signature('src_b')

        library.add_document(
            'explanation', 'a2', 'c', source_name='src_a',
        )

        sig_b_after = library.source_signature('src_b')
        assert sig_b_before == sig_b_after

    def test_source_signature_returns_value_for_unknown_source(
        self, library: 'Library',
    ) -> None:
        """A source no document uses still returns a stable string —
        cache lookups don't need to special-case empty buckets."""
        sig = library.source_signature('does_not_exist')

        assert isinstance(sig, str)
        # Should be a stable, deterministic value (e.g. '0|').
        assert sig == library.source_signature('does_not_exist')
