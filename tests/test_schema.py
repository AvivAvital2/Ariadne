"""Tests for the Claude Library schema module."""
from __future__ import annotations

import json

import numpy as np

from schema import Chunk, Document, SearchResult


class TestDocument:
    """Tests for the Document class."""

    def test_create_document_with_defaults(self) -> None:
        """Test creating a document with default values."""
        doc = Document()

        assert doc.id is not None
        assert len(doc.id) == 36  # UUID format
        assert doc.content_type == 'explanation'
        assert doc.title == ''
        assert doc.content == ''
        assert doc.source_files == []
        assert doc.embedding is None
        assert doc.created_at is not None
        assert doc.updated_at is not None
        assert doc.metadata == {}

    def test_create_document_with_values(self) -> None:
        """Test creating a document with custom values."""
        embedding = np.random.rand(1536).astype(np.float32)
        doc = Document(
            id='test-id',
            content_type='architecture',
            title='Test Title',
            content='Test content',
            source_files=['file1.py', 'file2.py'],
            embedding=embedding,
            metadata={'key': 'value'},
        )

        assert doc.id == 'test-id'
        assert doc.content_type == 'architecture'
        assert doc.title == 'Test Title'
        assert doc.content == 'Test content'
        assert doc.source_files == ['file1.py', 'file2.py']
        assert np.array_equal(doc.embedding, embedding)
        assert doc.metadata == {'key': 'value'}

    def test_source_files_converter_from_string(self) -> None:
        """Test that source_files can be created from a JSON string."""
        doc = Document(source_files='["file1.py", "file2.py"]')
        assert doc.source_files == ['file1.py', 'file2.py']

    def test_source_files_converter_from_none(self) -> None:
        """Test that source_files defaults to empty list when None."""
        doc = Document(source_files=None)
        assert doc.source_files == []

    def test_metadata_converter_from_string(self) -> None:
        """Test that metadata can be created from a JSON string."""
        doc = Document(metadata='{"key": "value"}')
        assert doc.metadata == {'key': 'value'}

    def test_embedding_converter_from_bytes(self) -> None:
        """Test that embedding can be created from bytes."""
        original = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        doc = Document(embedding=original.tobytes())
        assert doc.embedding is not None
        assert np.allclose(doc.embedding, original)

    def test_source_files_json(self) -> None:
        """Test source_files_json serialization."""
        doc = Document(source_files=['file1.py', 'file2.py'])
        assert doc.source_files_json() == '["file1.py", "file2.py"]'

    def test_metadata_json(self) -> None:
        """Test metadata_json serialization."""
        doc = Document(metadata={'key': 'value', 'num': 42})
        result = json.loads(doc.metadata_json())
        assert result == {'key': 'value', 'num': 42}

    def test_embedding_bytes(self) -> None:
        """Test embedding_bytes serialization."""
        embedding = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        doc = Document(embedding=embedding)

        embedding_bytes = doc.embedding_bytes()
        assert embedding_bytes is not None

        restored = np.frombuffer(embedding_bytes, dtype=np.float32)
        assert np.allclose(restored, embedding)

    def test_embedding_bytes_none(self) -> None:
        """Test embedding_bytes returns None when embedding is None."""
        doc = Document()
        assert doc.embedding_bytes() is None


class TestChunk:
    """Tests for the Chunk class."""

    def test_create_chunk(self) -> None:
        """Test creating a chunk."""
        chunk = Chunk(
            document_id='doc-1',
            chunk_index=0,
            content='Test chunk content',
        )

        assert chunk.id is not None
        assert chunk.document_id == 'doc-1'
        assert chunk.chunk_index == 0
        assert chunk.content == 'Test chunk content'
        assert chunk.embedding is None

    def test_chunk_with_embedding(self) -> None:
        """Test creating a chunk with embedding."""
        embedding = np.random.rand(1536).astype(np.float32)
        chunk = Chunk(
            document_id='doc-1',
            chunk_index=1,
            content='Content',
            embedding=embedding,
        )

        assert np.array_equal(chunk.embedding, embedding)
        assert chunk.embedding_bytes() is not None


class TestSearchResult:
    """Tests for the SearchResult class."""

    def test_create_search_result(self) -> None:
        """Test creating a search result."""
        doc = Document(title='Test')
        result = SearchResult(document=doc, score=0.95)

        assert result.document == doc
        assert result.score == 0.95
        assert result.chunk is None

    def test_search_result_with_chunk(self) -> None:
        """Test creating a search result with a chunk."""
        doc = Document(title='Test')
        chunk = Chunk(document_id=doc.id, content='Chunk content')
        result = SearchResult(document=doc, score=0.87, chunk=chunk)

        assert result.chunk == chunk
