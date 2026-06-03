"""Tests for MCP service helper functions."""
from __future__ import annotations

from ariadne_mcp.service import _trim_related_documents


class TestTrimRelatedDocuments:
    """Tests for the Related Documents trimming function."""

    def test_no_related_section(self) -> None:
        """Content without Related Documents should be returned unchanged."""
        text = '# Title\n\nJust content, no related section.'
        assert _trim_related_documents(text) == text

    def test_empty_related_section(self) -> None:
        """Empty Related Documents section should be preserved as-is."""
        text = '# Title\n\n## Related Documents\n'
        result = _trim_related_documents(text)
        assert '## Related Documents' in result

    def test_import_links_prioritized(self) -> None:
        """Import-based links should be kept before mention-based links."""
        text = (
            "# Content\n\n## Related Documents\n\n"
            "- [A](a.md) - Mentions 'A'\n"
            "- [B](b.md) - References schema\n"
            "- [C](c.md) - Mentions 'C'\n"
            "- [D](d.md) - References create\n"
        )
        result = _trim_related_documents(text, max_links=2)
        lines = [l for l in result.split('\n') if l.startswith('- [')]
        # Import links (References) should come first
        assert 'References schema' in lines[0]
        assert 'References create' in lines[1]

    def test_max_links_respected(self) -> None:
        """Should not return more than max_links links."""
        links = '\n'.join(f'- [Doc {i}](x.md) - References x' for i in range(50))
        text = f'# Content\n\n## Related Documents\n\n{links}'
        result = _trim_related_documents(text, max_links=5)
        kept = [l for l in result.split('\n') if l.startswith('- [')]
        assert len(kept) == 5

    def test_ellipsis_shows_remaining_count(self) -> None:
        """When links are trimmed, show how many were omitted."""
        links = '\n'.join(f'- [Doc {i}](x.md) - References x' for i in range(20))
        text = f'# Content\n\n## Related Documents\n\n{links}'
        result = _trim_related_documents(text, max_links=5)
        assert '... and 15 more' in result

    def test_preserves_content_before_related(self) -> None:
        """Content before Related Documents section should be unchanged."""
        content = '# Title\n\nImportant content here.\n\n## Section\n\nMore stuff.'
        text = content + '\n\n## Related Documents\n\n- [X](x.md) - References x'
        result = _trim_related_documents(text, max_links=1)
        assert result.startswith(content)
