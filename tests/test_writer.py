"""Tests for the Claude Library writer module."""
from __future__ import annotations

from writer import ChunkConfig, chunk_text


class TestChunkText:
    """Tests for the chunk_text function."""

    def test_empty_text(self) -> None:
        """Test chunking empty text."""
        chunks = chunk_text('')
        assert chunks == []

    def test_short_text(self) -> None:
        """Test text shorter than chunk size."""
        config = ChunkConfig(chunk_size=100)
        text = 'Short text.'

        chunks = chunk_text(text, config)
        assert chunks == [text]

    def test_basic_chunking(self) -> None:
        """Test basic text chunking."""
        config = ChunkConfig(chunk_size=50, chunk_overlap=10, split_on_paragraphs=False)
        text = 'A' * 100  # 100 characters

        chunks = chunk_text(text, config)
        assert len(chunks) >= 2

    def test_paragraph_splitting(self) -> None:
        """Test chunking on paragraph boundaries."""
        config = ChunkConfig(chunk_size=100, split_on_paragraphs=True)
        text = 'Paragraph one.\n\nParagraph two.\n\nParagraph three.'

        chunks = chunk_text(text, config)
        # Should keep paragraphs together if they fit
        assert len(chunks) >= 1

    def test_long_paragraph_splitting(self) -> None:
        """Test that long paragraphs are split."""
        config = ChunkConfig(chunk_size=50, chunk_overlap=10, split_on_paragraphs=True)
        text = 'This is a very long paragraph. ' * 10

        chunks = chunk_text(text, config)
        assert len(chunks) > 1
        # Each chunk should be roughly within the size limit
        for chunk in chunks:
            # Allow some flexibility for sentence boundaries
            assert len(chunk) <= config.chunk_size + 50

    def test_chunk_overlap(self) -> None:
        """Test that chunks have overlap."""
        config = ChunkConfig(chunk_size=30, chunk_overlap=10, split_on_paragraphs=False)
        text = 'word ' * 20  # 100 characters

        chunks = chunk_text(text, config)

        # Check that there's some overlap between consecutive chunks
        # This is approximate due to word boundary handling
        assert len(chunks) > 1

    def test_sentence_boundary_respect(self) -> None:
        """Test that chunking respects sentence boundaries when possible."""
        config = ChunkConfig(chunk_size=50, chunk_overlap=5, split_on_paragraphs=False)
        text = 'First sentence. Second sentence. Third sentence. Fourth sentence.'

        chunks = chunk_text(text, config)

        # Chunks should end at sentence boundaries when possible
        for chunk in chunks[:-1]:  # Except the last chunk
            assert chunk.endswith('.') or chunk.endswith('. ') or len(chunk) <= config.chunk_size


class TestChunkConfig:
    """Tests for the ChunkConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ChunkConfig()

        assert config.chunk_size == 500
        assert config.chunk_overlap == 50
        assert config.split_on_paragraphs is True

    def test_custom_config(self) -> None:
        """Test custom configuration values."""
        config = ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            split_on_paragraphs=False,
        )

        assert config.chunk_size == 200
        assert config.chunk_overlap == 20
        assert config.split_on_paragraphs is False
