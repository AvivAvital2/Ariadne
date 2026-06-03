"""Cross-reference detection and injection for documentation.

This module provides tools for detecting relationships between documents
and injecting "Related Documents" sections.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

from attrs import define, field, frozen

from schema import Document


@frozen
class CrossReference:
    """A cross-reference between documents.

    Attributes:
        source_id: ID of the document containing the reference.
        target_id: ID of the referenced document.
        ref_type: Type of reference (import, mention, related).
        context: Brief context about why the reference exists.
    """

    source_id: str
    target_id: str
    ref_type: str  # "import", "mention", "semantic"
    context: str = ''


@define
class CrossRefDetector:
    """Detects cross-references between documents.

    This class analyzes documents and their source files to find
    relationships that should be cross-referenced.

    Attributes:
        documents: List of documents to analyze.
        _source_to_doc: Mapping from source file to document IDs.
        _module_to_doc: Mapping from module name to document IDs.
    """

    documents: list[Document] = field(factory=list)
    _source_to_doc: dict[str, list[str]] = field(factory=lambda: defaultdict(list), init=False)
    _module_to_doc: dict[str, list[str]] = field(factory=lambda: defaultdict(list), init=False)
    _title_to_doc: dict[str, str] = field(factory=dict, init=False)

    def __attrs_post_init__(self) -> None:
        """Build lookup indexes from documents."""
        self._build_indexes()

    def _build_indexes(self) -> None:
        """Build indexes for efficient lookup."""
        self._source_to_doc.clear()
        self._module_to_doc.clear()
        self._title_to_doc.clear()

        for doc in self.documents:
            # Index by source files
            for source in doc.source_files:
                self._source_to_doc[source].append(doc.id)
                # Also index by module name derived from path
                module_name = self._path_to_module(source)
                if module_name:
                    self._module_to_doc[module_name].append(doc.id)

            # Index by title
            self._title_to_doc[doc.title.lower()] = doc.id

    def _path_to_module(self, path: str) -> str | None:
        """Convert a file path to a module name."""
        # Remove .py extension
        path = path.removesuffix('.py')
        # Convert path separators to dots
        parts = Path(path).parts
        if parts:
            return '.'.join(parts)
        return None

    def add_document(self, doc: Document) -> None:
        """Add a document to the detector.

        Args:
            doc: Document to add.
        """
        self.documents.append(doc)

        # Update indexes
        for source in doc.source_files:
            self._source_to_doc[source].append(doc.id)
            module_name = self._path_to_module(source)
            if module_name:
                self._module_to_doc[module_name].append(doc.id)

        self._title_to_doc[doc.title.lower()] = doc.id

    def detect_import_references(
        self,
        doc: Document,
        imports: list[str],
    ) -> list[CrossReference]:
        """Detect references based on import statements.

        Args:
            doc: The document to find references for.
            imports: List of imported module names.

        Returns:
            List of CrossReference objects.
        """
        refs = []
        for module_name in imports:
            # Look for documents about this module
            target_ids = self._module_to_doc.get(module_name, [])
            for target_id in target_ids:
                if target_id != doc.id:
                    refs.append(
                        CrossReference(
                            source_id=doc.id,
                            target_id=target_id,
                            ref_type='import',
                            context=f'Imports from {module_name}',
                        )
                    )
        return refs

    def detect_mention_references(self, doc: Document) -> list[CrossReference]:
        """Detect references based on content mentions.

        Args:
            doc: The document to find references for.

        Returns:
            List of CrossReference objects.
        """
        refs = []
        content_lower = doc.content.lower()

        for other_doc in self.documents:
            if other_doc.id == doc.id:
                continue

            # Check if the other document's title is mentioned
            title_lower = other_doc.title.lower()
            if len(title_lower) > 3 and title_lower in content_lower:
                refs.append(
                    CrossReference(
                        source_id=doc.id,
                        target_id=other_doc.id,
                        ref_type='mention',
                        context=f"Mentions '{other_doc.title}'",
                    )
                )
                continue

            # Check for class/function name mentions
            for source in other_doc.source_files:
                module_name = self._path_to_module(source)
                if module_name:
                    # Extract the last component (e.g., "feature" from "mylib")
                    short_name = module_name.split('.')[-1]
                    if len(short_name) > 3:
                        # Look for the name as a word boundary match
                        pattern = rf'\b{re.escape(short_name)}\b'
                        if re.search(pattern, doc.content, re.IGNORECASE):
                            refs.append(
                                CrossReference(
                                    source_id=doc.id,
                                    target_id=other_doc.id,
                                    ref_type='mention',
                                    context=f'References {short_name}',
                                )
                            )
                            break

        return refs

    def detect_semantic_references(
        self,
        doc: Document,
        similarity_threshold: float = 0.7,
    ) -> list[CrossReference]:
        """Detect references based on semantic similarity.

        This method requires that documents have embeddings.

        Args:
            doc: The document to find references for.
            similarity_threshold: Minimum cosine similarity for a reference.

        Returns:
            List of CrossReference objects.
        """
        if doc.embedding is None:
            return []

        refs = []
        for other_doc in self.documents:
            if other_doc.id == doc.id:
                continue
            if other_doc.embedding is None:
                continue

            # Compute cosine similarity
            similarity = self._cosine_similarity(doc.embedding, other_doc.embedding)
            if similarity >= similarity_threshold:
                refs.append(
                    CrossReference(
                        source_id=doc.id,
                        target_id=other_doc.id,
                        ref_type='semantic',
                        context=f'Similar content (score: {similarity:.2f})',
                    )
                )

        return refs

    def _cosine_similarity(self, a, b) -> float:
        """Dot product similarity for pre-normalized embeddings."""
        import numpy as np

        return float(np.dot(np.asarray(a), np.asarray(b)))

    def get_all_references(
        self,
        doc: Document,
        imports: list[str] | None = None,
        include_semantic: bool = False,
        similarity_threshold: float = 0.7,
    ) -> list[CrossReference]:
        """Get all cross-references for a document.

        Args:
            doc: The document to find references for.
            imports: List of imported module names (for import-based refs).
            include_semantic: Whether to include semantic similarity refs.
            similarity_threshold: Threshold for semantic refs.

        Returns:
            List of CrossReference objects, deduplicated.
        """
        refs = []

        # Import-based references
        if imports:
            refs.extend(self.detect_import_references(doc, imports))

        # Mention-based references
        refs.extend(self.detect_mention_references(doc))

        # Semantic references
        if include_semantic:
            refs.extend(
                self.detect_semantic_references(doc, similarity_threshold)
            )

        # Deduplicate by target_id, keeping the most specific ref type
        seen = {}
        ref_priority = {'import': 0, 'mention': 1, 'semantic': 2}
        for ref in refs:
            if ref.target_id not in seen:
                seen[ref.target_id] = ref
            else:
                existing = seen[ref.target_id]
                if ref_priority.get(ref.ref_type, 99) < ref_priority.get(existing.ref_type, 99):
                    seen[ref.target_id] = ref

        return list(seen.values())

    def build_reference_graph(
        self,
        include_semantic: bool = False,
    ) -> dict[str, list[CrossReference]]:
        """Build a graph of all cross-references.

        Args:
            include_semantic: Whether to include semantic similarity refs.

        Returns:
            Dict mapping document ID to list of outgoing references.
        """
        graph: dict[str, list[CrossReference]] = {}

        for doc in self.documents:
            refs = self.get_all_references(
                doc,
                include_semantic=include_semantic,
            )
            graph[doc.id] = refs

        return graph


def inject_related_section(
    content: str,
    references: list[CrossReference],
    doc_titles: dict[str, str],
) -> str:
    """Inject a "Related Documents" section into content.

    Args:
        content: The document content.
        references: List of cross-references to include.
        doc_titles: Mapping from document ID to title.

    Returns:
        Content with "Related Documents" section added.
    """
    if not references:
        return content

    # Build the related documents section
    lines = ['\n## Related Documents\n']

    for ref in references:
        title = doc_titles.get(ref.target_id, ref.target_id)
        # Create a relative link (assuming docs are in same directory)
        link = f'- [{title}]({ref.target_id}.md)'
        if ref.context:
            link += f' - {ref.context}'
        lines.append(link)

    related_section = '\n'.join(lines) + '\n'

    # Check if there's already a "Related" section
    if '## Related' in content:
        # Replace existing section. Wrap replacement in a lambda so re.sub
        # doesn't interpret backslash sequences (\d, \s, \1, etc.) in the
        # replacement string — doc titles can legitimately contain literal
        # backslashes (e.g., when documenting regex patterns or Windows paths).
        pattern = r'## Related.*?(?=\n## |\Z)'
        replacement = related_section.strip() + '\n'
        content = re.sub(
            pattern, lambda _m: replacement, content, flags=re.DOTALL,
        )
    else:
        # Append to end
        content = content.rstrip() + '\n' + related_section

    return content


def format_mermaid_graph(
    references: list[CrossReference],
    doc_titles: dict[str, str],
) -> str:
    """Generate a Mermaid diagram of document relationships.

    Args:
        references: List of cross-references.
        doc_titles: Mapping from document ID to title.

    Returns:
        Mermaid diagram code.
    """
    lines = ['```mermaid', 'graph LR']

    # Create short IDs for nodes
    node_ids = {}
    counter = 0
    for ref in references:
        if ref.source_id not in node_ids:
            node_ids[ref.source_id] = f'N{counter}'
            counter += 1
        if ref.target_id not in node_ids:
            node_ids[ref.target_id] = f'N{counter}'
            counter += 1

    # Add node definitions
    for doc_id, node_id in node_ids.items():
        title = doc_titles.get(doc_id, doc_id[:20])
        # Escape special characters for Mermaid
        title = title.replace('"', "'").replace('[', '(').replace(']', ')')
        lines.append(f'    {node_id}["{title}"]')

    # Add edges
    for ref in references:
        source_node = node_ids[ref.source_id]
        target_node = node_ids[ref.target_id]
        edge_label = ref.ref_type
        lines.append(f'    {source_node} -->|{edge_label}| {target_node}')

    lines.append('```')
    return '\n'.join(lines)
