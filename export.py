"""Markdown export functionality for the Ariadne library.

This module provides functionality to export library contents to markdown files
and generate a manifest for easy browsing.
"""
from __future__ import annotations

__all__ = ['ExportConfig', 'export_library']

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from attrs import frozen

from library import Library
from schema import CATALOG_KIND_FILE_INDEX, ContentType, Document

# Path to locate project's CLAUDE.md relative to source
PROJECT_CLAUDE_MD_PATHS = ('CLAUDE.md', 'docs/CLAUDE.md')


@frozen
class ExportConfig:
    """Configuration for markdown export.

    Attributes:
        include_frontmatter: Whether to include YAML frontmatter in markdown files.
        include_source_files: Whether to include source file references.
        organize_by_type: Whether to organize files into subdirectories by content type.
    """
    include_frontmatter: bool = True
    include_source_files: bool = True
    organize_by_type: bool = True


def _slugify(text: str) -> str:
    """Convert text to a URL-friendly slug."""
    # Convert to lowercase and replace spaces with hyphens
    slug = text.lower().strip()
    slug = re.sub(r'[^\w\s-]', '', slug)
    slug = re.sub(r'[-\s]+', '-', slug)
    return slug[:50]  # Limit length


def _document_to_markdown(doc: Document, config: ExportConfig) -> str:
    """Convert a document to markdown format."""
    parts: list[str] = []

    if config.include_frontmatter:
        frontmatter_lines = [
            '---',
            f'id: {doc.id}',
            f'type: {doc.content_type}',
            f'title: "{doc.title}"',
            f'created_at: {doc.created_at}',
            f'updated_at: {doc.updated_at}',
        ]
        if doc.source_files:
            frontmatter_lines.append('source_files:')
            for sf in doc.source_files:
                frontmatter_lines.append(f'  - {sf}')
        if doc.metadata:
            frontmatter_lines.append('metadata:')
            for key, value in doc.metadata.items():
                if isinstance(value, str):
                    frontmatter_lines.append(f'  {key}: "{value}"')
                else:
                    frontmatter_lines.append(f'  {key}: {value}')
        frontmatter_lines.append('---')
        parts.append('\n'.join(frontmatter_lines))

    # Title as H1
    parts.append(f'# {doc.title}')

    # Source files section
    if config.include_source_files and doc.source_files:
        parts.append('\n**Related files:**')
        for sf in doc.source_files:
            parts.append(f'- `{sf}`')

    # Main content
    parts.append('')
    parts.append(doc.content)

    return '\n'.join(parts)


def _get_subdirectory(content_type: ContentType) -> str:
    """Get the subdirectory name for a content type."""
    mapping = {
        'explanation': 'explanations',
        'architecture': 'architecture',
        'qa': 'qa',
        'diagram': 'diagrams',
        'finding': 'findings',
    }
    return mapping.get(content_type, 'other')


class LibraryExporter:
    """Export library contents to markdown files.

    This class handles:
    - Converting documents to markdown format
    - Organizing files by content type
    - Generating a manifest for navigation

    Example:
        >>> exporter = LibraryExporter(library)
        >>> exporter.export_all(Path('docs/'))
    """

    def __init__(
        self,
        library: Library,
        config: ExportConfig | None = None,
    ) -> None:
        """Initialize the exporter.

        Args:
            library: The Ariadne library to export from.
            config: Export configuration.
        """
        self.library = library
        self.config = config or ExportConfig()

    def export_document(self, doc: Document, output_dir: Path) -> Path:
        """Export a single document to markdown.

        Args:
            doc: The document to export.
            output_dir: Base output directory.

        Returns:
            Path to the created file.
        """
        # Determine output path
        if self.config.organize_by_type:
            subdir = _get_subdirectory(doc.content_type)
            dir_path = output_dir / subdir
        else:
            dir_path = output_dir

        dir_path.mkdir(parents=True, exist_ok=True)

        # Create filename from title
        filename = f'{_slugify(doc.title)}.md'
        file_path = dir_path / filename

        # Convert to markdown
        markdown = _document_to_markdown(doc, self.config)

        # Write file
        file_path.write_text(markdown)

        return file_path

    def export_all(
        self,
        output_dir: Path,
        source_name: str | None = None,
        source_path: Path | None = None,
        dependencies: list[str] | None = None,
    ) -> list[Path]:
        """Export all documents to markdown files.

        Args:
            output_dir: Base output directory.
            source_name: Name of the source (for CLAUDE.md generation).
            source_path: Path to the project source (for CLAUDE.md merging).
            dependencies: List of dependency source names for cross-references.

        Returns:
            List of paths to created files.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        docs = self.library.list_documents()
        paths: list[Path] = []

        for doc in docs:
            # file_index docs are derived index data, not authored knowledge —
            # never export them. They're regenerated from the element docs on
            # import (see catalog_writer.regenerate_file_index_docs).
            if doc.metadata.get('kind') == CATALOG_KIND_FILE_INDEX:
                continue
            path = self.export_document(doc, output_dir)
            paths.append(path)

        # Generate manifest
        self.generate_manifest(output_dir)

        # Generate README
        self.generate_readme(output_dir)

        # Generate INDEX.md (meta document, always visible)
        self.generate_index(output_dir)

        # Generate CLAUDE.md (merging with project instructions)
        if source_name:
            claude_md_path = self.generate_claude_md(
                output_dir=output_dir,
                source_name=source_name,
                source_path=source_path,
                dependencies=dependencies,
            )
            paths.append(claude_md_path)

        return paths

    def generate_manifest(self, output_dir: Path) -> Path:
        """Generate a YAML manifest of all documents.

        Args:
            output_dir: Base output directory.

        Returns:
            Path to the manifest file.
        """
        docs = self.library.list_documents()

        # Group by content type
        by_type: dict[str, list[Document]] = {}
        for doc in docs:
            ct = doc.content_type
            if ct not in by_type:
                by_type[ct] = []
            by_type[ct].append(doc)

        # Build manifest content
        lines = [
            '# Ariadne Library Manifest',
            f'# Generated: {datetime.now().isoformat()}',
            f'# Total documents: {len(docs)}',
            '',
        ]

        for content_type in ['explanation', 'architecture', 'qa', 'diagram', 'finding']:
            type_docs = by_type.get(content_type, [])
            if not type_docs:
                continue

            subdir = _get_subdirectory(content_type)  # type: ignore[arg-type]
            lines.append(f'{content_type}:')

            for doc in sorted(type_docs, key=lambda d: d.title):
                filename = f'{_slugify(doc.title)}.md'
                if self.config.organize_by_type:
                    rel_path = f'{subdir}/{filename}'
                else:
                    rel_path = filename

                lines.append(f'  - id: {doc.id}')
                lines.append(f'    title: "{doc.title}"')
                lines.append(f'    path: {rel_path}')
                lines.append(f'    updated_at: {doc.updated_at}')
                if doc.source_files:
                    lines.append(f'    source_files: {doc.source_files}')
                if doc.metadata:
                    lines.append('    metadata:')
                    for key, value in doc.metadata.items():
                        if isinstance(value, str):
                            lines.append(f'      {key}: "{value}"')
                        elif isinstance(value, list):
                            lines.append(f'      {key}: {value}')
                        else:
                            lines.append(f'      {key}: {value}')

            lines.append('')

        manifest_path = output_dir / 'manifest.yaml'
        manifest_path.write_text('\n'.join(lines))

        return manifest_path

    def generate_readme(self, output_dir: Path) -> Path:
        """Generate a README file for the exported library.

        Args:
            output_dir: Base output directory.

        Returns:
            Path to the README file.
        """
        docs = self.library.list_documents()

        # Count by type
        counts: dict[str, int] = {}
        for doc in docs:
            ct = doc.content_type
            counts[ct] = counts.get(ct, 0) + 1

        lines = [
            '# Ariadne Library',
            '',
            'This directory contains exported documentation from the Ariadne library.',
            '',
            '## Contents',
            '',
            f'Total documents: {len(docs)}',
            '',
        ]

        for content_type, count in sorted(counts.items()):
            subdir = _get_subdirectory(content_type)  # type: ignore[arg-type]
            lines.append(f'- **{content_type.title()}** ({count}): [{subdir}/]({subdir}/)')

        lines.extend([
            '',
            '## Structure',
            '',
            '- `manifest.yaml` - Index of all documents with metadata',
            '- `INDEX.md` - Full document index with status info',
            '- `explanations/` - How things work in the codebase',
            '- `architecture/` - System design and component relationships',
            '- `qa/` - Questions and answers about the codebase',
            '- `diagrams/` - Visual diagrams (Graphviz DOT)',
            '- `findings/` - Session discoveries and conclusions',
            '',
            '## Usage',
            '',
            'These documents can be read directly or searched using Ariadne.',
            '',
            '```python',
            'import ariadne',
            '',
            'library = ariadne.Library(Path("ariadne.db"))',
            'results = library.search(query_embedding, k=5)',
            '```',
            '',
        ])

        readme_path = output_dir / 'README.md'
        readme_path.write_text('\n'.join(lines))

        return readme_path

    def generate_index(self, output_dir: Path) -> Path:
        """Generate INDEX.md with a complete document listing.

        This is a meta document that lists all available documents including
        their status, branches, and expiration info. It's always visible
        regardless of branch filtering.

        Args:
            output_dir: Base output directory.

        Returns:
            Path to the INDEX.md file.
        """
        docs = self.library.list_documents()

        # Count by type and status
        by_type: dict[str, list[Document]] = {}
        by_status: dict[str, list[Document]] = {}
        branch_specific: list[Document] = []

        for doc in docs:
            ct = doc.content_type
            if ct not in by_type:
                by_type[ct] = []
            by_type[ct].append(doc)

            status = doc.metadata.get('status', 'stable')
            if status not in by_status:
                by_status[status] = []
            by_status[status].append(doc)

            # Track branch-specific docs
            if doc.metadata.get('branches'):
                branch_specific.append(doc)

        lines = [
            '---',
            'type: meta',
            'status: meta',
            f'updated_at: {datetime.now().isoformat()}',
            '---',
            '',
            '# Available Documents',
            '',
            'This is an auto-generated index of all Ariadne documents.',
            '',
            '## Summary',
            '',
            f'**Total documents:** {len(docs)}',
            '',
            '### By Type',
            '',
        ]

        for content_type in ['explanation', 'architecture', 'qa', 'diagram', 'finding']:
            count = len(by_type.get(content_type, []))
            if count > 0:
                lines.append(f'- {content_type.title()}: {count}')

        lines.extend([
            '',
            '### By Status',
            '',
        ])

        for status in ['stable', 'experimental', 'deprecated']:
            count = len(by_status.get(status, []))
            if count > 0:
                lines.append(f'- {status.title()}: {count}')

        lines.extend([
            '',
            '## Document List',
            '',
            '| Title | Type | Status | Source Files |',
            '|-------|------|--------|--------------|',
        ])

        # Sort docs: stable first, then by title
        sorted_docs = sorted(docs, key=lambda d: (
            0 if d.metadata.get('status', 'stable') == 'stable' else 1,
            d.title.lower(),
        ))

        for doc in sorted_docs:
            status = doc.metadata.get('status', 'stable')
            source_files_str = ', '.join(doc.source_files[:2]) if doc.source_files else '-'
            if len(doc.source_files) > 2:
                source_files_str += f' (+{len(doc.source_files) - 2})'
            lines.append(f'| {doc.title} | {doc.content_type} | {status} | {source_files_str} |')

        # Branch-specific docs section
        if branch_specific:
            lines.extend([
                '',
                '## Branch-Specific Documents',
                '',
                'These documents are tagged for specific branches and may expire.',
                '',
                '| Title | Branch | Expires |',
                '|-------|--------|---------|',
            ])

            for doc in branch_specific:
                branches = doc.metadata.get('branches', [])
                branches_str = ', '.join(branches) if isinstance(branches, list) else str(branches)
                expires_at = doc.metadata.get('expires_at', '-')
                if expires_at != '-':
                    expires_at = expires_at[:10]  # Just the date part
                lines.append(f'| {doc.title} | {branches_str} | {expires_at} |')

        lines.append('')

        index_path = output_dir / 'INDEX.md'
        index_path.write_text('\n'.join(lines))

        return index_path

    def generate_claude_md(
        self,
        output_dir: Path,
        source_name: str,
        source_path: Path | None = None,
        ariadne_path: Path | None = None,
        dependencies: list[str] | None = None,
    ) -> Path:
        """Generate CLAUDE.md that merges project instructions with Ariadne sections.

        Args:
            output_dir: Base output directory (docs/<source>/).
            source_name: Name of the source (e.g., 'pythonproject').
            source_path: Path to the project source code (to find existing CLAUDE.md).
            ariadne_path: Path to Ariadne installation (for finding command).
            dependencies: List of dependency source names for cross-references.

        Returns:
            Path to the generated CLAUDE.md file.
        """
        if ariadne_path is None:
            ariadne_path = Path(__file__).parent.resolve()

        docs = self.library.list_documents()

        # Read project's existing CLAUDE.md if it exists
        project_content = ''
        if source_path:
            for rel_path in PROJECT_CLAUDE_MD_PATHS:
                claude_md_path = source_path / rel_path
                if claude_md_path.exists():
                    project_content = claude_md_path.read_text()
                    break

        # Build Ariadne sections
        ariadne_sections: list[str] = []

        # Knowledge Base section
        ariadne_sections.append('## Ariadne Knowledge Base')
        ariadne_sections.append('')

        # Add scope information if there are dependencies
        if dependencies:
            ariadne_sections.append(f'**Scope:** `{source_name}`')
            ariadne_sections.append(f'**Dependencies:** {", ".join(dependencies)}')
            ariadne_sections.append('')

        ariadne_sections.append('Before exploring code, check Ariadne docs for pre-researched documentation:')
        ariadne_sections.append(f'- `{output_dir}/manifest.yaml` - Index of all documents')
        ariadne_sections.append(f'- `{output_dir}/explanations/` - How systems work')
        ariadne_sections.append(f'- `{output_dir}/architecture/` - Design decisions')
        ariadne_sections.append(f'- `{output_dir}/findings/` - Session discoveries')

        # Add dependency doc paths
        if dependencies:
            from config import get_config
            cfg = get_config()
            ariadne_sections.append('')
            ariadne_sections.append('**Dependency documentation:**')
            for dep in dependencies:
                dep_docs_path = cfg.resolve_docs_path(dep)
                ariadne_sections.append(f'- `{dep_docs_path}/` - {dep} docs')

        ariadne_sections.append('')
        ariadne_sections.append('When asked about the codebase:')
        ariadne_sections.append('1. First check if Ariadne docs have relevant documentation')
        ariadne_sections.append('2. Read those docs before grepping/globbing source code')
        ariadne_sections.append("3. Only explore source if the docs don't cover the topic")
        ariadne_sections.append('')

        # Saving Findings section
        ariadne_sections.append('## Saving Findings')
        ariadne_sections.append('')
        ariadne_sections.append('To save a finding for future reference:')
        ariadne_sections.append('```bash')
        ariadne_sections.append(f'cd {ariadne_path} && uv run ariadne finding "Your finding here" --topic "Topic"')
        ariadne_sections.append('```')
        ariadne_sections.append('')

        # Key Systems section (auto-generated from doc titles)
        if docs:
            ariadne_sections.append('## Key Systems (from Ariadne)')
            ariadne_sections.append('')

            # Group by content type
            by_type: dict[str, list[Document]] = {}
            for doc in docs:
                ct = doc.content_type
                if ct not in by_type:
                    by_type[ct] = []
                by_type[ct].append(doc)

            # Show explanations and architecture docs
            for content_type in ['explanation', 'architecture']:
                type_docs = by_type.get(content_type, [])
                if type_docs:
                    for doc in sorted(type_docs, key=lambda d: d.title)[:10]:
                        ariadne_sections.append(f'- **{doc.title}**: {doc.content[:100].replace(chr(10), " ")}...')

            ariadne_sections.append('')

        # Merge project content with Ariadne sections
        lines: list[str] = []

        if project_content:
            # Remove any existing Ariadne sections from project content
            cleaned_content = self._remove_ariadne_sections(project_content)
            lines.append(cleaned_content.rstrip())
            lines.append('')
            lines.append('---')
            lines.append('')
            lines.append('# Ariadne-Generated Sections')
            lines.append('')
            lines.append('> The sections below are auto-generated by Ariadne. Edit via `ariadne edit-instructions`.')
            lines.append('')
        else:
            # No project CLAUDE.md, create header
            lines.append(f'# {source_name}')
            lines.append('')

        lines.extend(ariadne_sections)

        # Write the file
        claude_md_path = output_dir / 'CLAUDE.md'
        claude_md_path.write_text('\n'.join(lines))

        return claude_md_path

    def _remove_ariadne_sections(self, content: str) -> str:
        """Remove Ariadne-generated sections from existing CLAUDE.md content."""
        # Find the Ariadne separator and remove everything after it
        separator = '---\n\n# Ariadne-Generated Sections'
        if separator in content:
            content = content.split(separator, maxsplit=1)[0]

        # Also remove old-style Knowledge Base section if present
        kb_marker = '## Knowledge Base\nBefore exploring code, check Ariadne docs'
        if kb_marker in content:
            # Find and remove from this marker to the next section or end
            parts = content.split('## Knowledge Base')
            if len(parts) > 1:
                # Keep everything before Knowledge Base
                before = parts[0]
                # Find the next ## section in the remaining content
                remaining = parts[1]
                next_section = remaining.find('\n## ')
                if next_section > 0:
                    content = before + remaining[next_section + 1:]
                else:
                    content = before

        return content


def import_from_markdown(
    library: Library,
    markdown_dir: Path,
) -> int:
    """Import documents from markdown files.

    This allows rebuilding the database from exported markdown files.
    Note: Embeddings will need to be regenerated separately.

    Args:
        library: The library to import into.
        markdown_dir: Directory containing markdown files.

    Returns:
        Number of documents imported.
    """
    import re

    count = 0

    # Find all markdown files
    for md_file in markdown_dir.rglob('*.md'):
        if md_file.name in ('README.md', 'manifest.yaml'):
            continue

        content = md_file.read_text()

        # Parse frontmatter if present
        frontmatter: dict[str, Any] = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                frontmatter_text = parts[1]
                body = parts[2].strip()

                # Simple YAML parsing for frontmatter
                current_section: str | None = None
                section_data: dict[str, Any] = {}
                for line in frontmatter_text.strip().split('\n'):
                    if ':' in line and not line.startswith(' '):
                        # Save previous section if any
                        if current_section and section_data:
                            frontmatter[current_section] = section_data
                            section_data = {}
                        key, value = line.split(':', 1)
                        key = key.strip()
                        value = value.strip().strip('"')
                        if value:
                            frontmatter[key] = value
                            current_section = None
                        else:
                            # This is a section header (metadata:, source_files:)
                            current_section = key
                    elif line.startswith('  ') and current_section:
                        # Nested item under current section
                        line_stripped = line.strip()
                        if line_stripped.startswith('- '):
                            # List item
                            if current_section not in frontmatter:
                                frontmatter[current_section] = []
                            frontmatter[current_section].append(line_stripped[2:].strip())
                        elif ':' in line_stripped:
                            # Key-value in section (for metadata)
                            k, v = line_stripped.split(':', 1)
                            k = k.strip()
                            v = v.strip().strip('"')
                            # Try to parse as list if it looks like one
                            if v.startswith('[') and v.endswith(']'):
                                try:
                                    import ast
                                    v = ast.literal_eval(v)
                                except (ValueError, SyntaxError):
                                    pass
                            section_data[k] = v
                # Save final section
                if current_section and section_data:
                    frontmatter[current_section] = section_data

        # Extract title from H1 or frontmatter
        title = frontmatter.get('title', '')
        if not title:
            h1_match = re.match(r'^#\s+(.+)$', body, re.MULTILINE)
            if h1_match:
                title = h1_match.group(1)
                # Remove H1 from body
                body = body[h1_match.end():].strip()

        # Determine content type from frontmatter or directory
        content_type = frontmatter.get('type', 'explanation')
        from schema import CONTENT_TYPES
        if not content_type or content_type not in CONTENT_TYPES:
            # Infer from directory
            parent_name = md_file.parent.name
            type_mapping = {
                'explanations': 'explanation',
                'architecture': 'architecture',
                'qa': 'qa',
                'diagrams': 'diagram',
                'findings': 'finding',
            }
            content_type = type_mapping.get(parent_name, 'explanation')

        # Parse source files from frontmatter or body
        source_files: list[str] = []
        if 'source_files' in frontmatter:
            sf_value = frontmatter['source_files']
            if isinstance(sf_value, list):
                source_files = sf_value

        # Remove "Related files" section from body if present
        body = re.sub(r'\*\*Related files:\*\*\n(?:- `.+`\n)+', '', body).strip()

        # Add document (without embedding - needs separate step)
        # Use frontmatter ID if present, otherwise generate deterministic ID from title+type
        from schema import generate_deterministic_id
        doc_id = frontmatter.get('id')
        final_title = title or md_file.stem
        if doc_id is None:
            # Generate deterministic ID so re-importing same doc updates instead of duplicates
            doc_id = generate_deterministic_id(content_type, final_title)

        # Extract metadata if present
        metadata = frontmatter.get('metadata', {})
        if not isinstance(metadata, dict):
            metadata = {}

        # Skip any file_index docs left over in an older export — they're
        # derived index data, regenerated from the element docs after import.
        if metadata.get('kind') == CATALOG_KIND_FILE_INDEX:
            continue

        library.add_document(
            content_type=content_type,
            title=final_title,
            content=body,
            source_files=source_files,
            doc_id=doc_id,
            metadata=metadata,
        )

        count += 1

    return count
