"""Intelligence operations — explanation, clustering, review, and conflict resolution."""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

import numpy as np

from schema import Document

_logger = logging.getLogger(__name__)


class IntelligenceMixin:
    """Explanation, clustering, conflict resolution, and cross-document analysis.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    - self.list_documents_lite() from CoreMixin
    - self.get_documents_batch() from CoreMixin
    - self.get_embeddings_for_ids() from CoreMixin
    - self.find_documents_by_source_files() from SearchMixin
    """

    def explain(self, file_path: str) -> dict[str, Any]:
        """Get everything Ariadne knows about a specific file.

        Assembles all document types (explanation, architecture, topic, finding)
        related to the given file into a single composite response.

        Args:
            file_path: Path to the source file (absolute or relative).

        Returns:
            Dict with 'file', 'documents' (grouped by type), 'graph_neighbors',
            and 'summary' fields.
        """
        # Find all docs for this file
        docs = self.find_documents_by_source_files([file_path])

        # Also try with just the filename for partial matching
        if not docs:
            import os
            basename = os.path.basename(file_path)
            # Use lite query first to find matching IDs, then fetch full docs
            matching_ids = [m.id for m in self.list_documents_lite()
                          if any(basename in sf for sf in m.source_files)]
            docs = self.get_documents_batch(matching_ids) if matching_ids else []

        # Group by content type
        by_type: dict[str, list[dict[str, str]]] = {}
        for doc in docs:
            ct = doc.content_type
            if ct not in by_type:
                by_type[ct] = []
            by_type[ct].append({
                'id': doc.id,
                'title': doc.title,
                'content': doc.content,
            })

        # Find graph neighbors if graph exists
        neighbors: list[dict[str, str]] = []
        try:
            with self._conn_provider.acquire() as conn:
                # Files this file imports
                for row in conn.execute(
                    "SELECT target_id FROM doc_graph WHERE source_id LIKE ? AND edge_type = 'imports'",
                    (f'%{file_path}%',),
                ).fetchall():
                    neighbors.append({'file': row[0], 'relationship': 'imports'})
                # Files that import this file
                for row in conn.execute(
                    "SELECT source_id FROM doc_graph WHERE target_id LIKE ? AND edge_type = 'imports'",
                    (f'%{file_path}%',),
                ).fetchall():
                    neighbors.append({'file': row[0], 'relationship': 'imported_by'})
        except Exception:
            pass  # Graph may not be built

        # Build summary
        doc_count = sum(len(v) for v in by_type.values())
        types_found = list(by_type.keys())
        summary = (
            f'{doc_count} documents found for {file_path}: '
            f'{", ".join(f"{len(v)} {k}" for k, v in by_type.items())}'
            if doc_count > 0
            else f'No documentation found for {file_path}'
        )

        return {
            'file': file_path,
            'documents': by_type,
            'graph_neighbors': neighbors[:20],
            'total_documents': doc_count,
            'types_found': types_found,
            'summary': summary,
        }

    def auto_tag_clusters(self, n_clusters: int = 10) -> list[dict[str, Any]]:
        """Use embeddings to cluster docs and suggest topic labels.

        Groups documents by embedding similarity and derives cluster labels
        from the most common words in cluster members' titles.

        Returns:
            List of clusters with suggested_label, doc_count, and sample_titles.
        """
        from collections import Counter

        # Use lite docs + separate embedding load (avoids loading full content)
        lite_docs = self.list_documents_lite()
        doc_ids = [d.id for d in lite_docs]
        embeddings_map = self.get_embeddings_for_ids(doc_ids)
        docs_with_embeddings = [(d, embeddings_map[d.id]) for d in lite_docs if d.id in embeddings_map]
        if len(docs_with_embeddings) < n_clusters:
            return []

        embeddings = np.stack([emb for _, emb in docs_with_embeddings])
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = embeddings / norms

        # Initialize centroids randomly
        rng = np.random.RandomState(42)
        indices = rng.choice(len(normalized), n_clusters, replace=False)
        centroids = normalized[indices].copy()

        # Run k-means for a few iterations
        assignments = np.zeros(len(normalized), dtype=int)
        for _ in range(15):
            # Assign each doc to nearest centroid
            similarities = normalized @ centroids.T
            assignments = similarities.argmax(axis=1)
            # Update centroids
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    centroids[k] = normalized[mask].mean(axis=0)
                    norm = np.linalg.norm(centroids[k])
                    if norm > 0:
                        centroids[k] /= norm

        # Build cluster summaries
        clusters = []
        skip_words = {'architecture', 'module', 'test', 'base', 'the', 'and', 'for', 'of', 'in', 'a'}
        for k in range(n_clusters):
            mask = assignments == k
            cluster_items = [(doc, emb) for i, (doc, emb) in enumerate(docs_with_embeddings) if mask[i]]
            if not cluster_items:
                continue

            # Derive label from most common title words
            word_counts: Counter[str] = Counter()
            for doc, _ in cluster_items:
                for word in doc.title.lower().split():
                    if len(word) > 2 and word not in skip_words:
                        word_counts[word] += 1

            top_words = [w for w, _ in word_counts.most_common(3)]
            label = ' '.join(top_words).title() if top_words else f'Cluster {k}'

            clusters.append({
                'suggested_label': label,
                'doc_count': len(cluster_items),
                'sample_titles': [d.title for d, _ in cluster_items[:5]],
            })

        clusters.sort(key=lambda c: c['doc_count'], reverse=True)
        return clusters

    def suggest_rechunk(self, min_quality_score: float = 0.5) -> list[dict[str, Any]]:
        """Identify docs whose chunks may benefit from re-chunking.

        Checks for: chunks that split mid-sentence, very short chunks,
        chunks that start mid-paragraph.

        Returns:
            List of docs with chunk quality issues and re-chunk recommendations.
        """
        suggestions = []

        with self._conn_provider.acquire() as conn:
            # Get docs with their chunks
            doc_ids = conn.execute(
                'SELECT DISTINCT document_id FROM chunks'
            ).fetchall()

            for (doc_id,) in doc_ids:
                chunks = conn.execute(
                    'SELECT content FROM chunks WHERE document_id = ? ORDER BY chunk_index',
                    (doc_id,),
                ).fetchall()

                if len(chunks) < 2:
                    continue

                issues = []
                for i, (content,) in enumerate(chunks):
                    # Check for mid-sentence split (doesn't end with sentence-ending punct)
                    if content and not content.rstrip().endswith(('.', '!', '?', ':', '```')):
                        issues.append(f'chunk {i} ends mid-sentence')

                    # Check for very short chunks
                    if content and len(content.strip()) < 30:
                        issues.append(f'chunk {i} is very short ({len(content.strip())} chars)')

                if issues:
                    doc_row = conn.execute(
                        'SELECT title FROM documents WHERE id = ?', (doc_id,)
                    ).fetchone()
                    title = doc_row[0] if doc_row else '(unknown)'
                    quality = 1.0 - len(issues) / len(chunks)

                    if quality < min_quality_score:
                        suggestions.append({
                            'doc_id': doc_id,
                            'title': title,
                            'chunk_count': len(chunks),
                            'issues': issues[:5],
                            'quality_score': round(quality, 2),
                        })

        suggestions.sort(key=lambda s: s['quality_score'])
        return suggestions[:20]

    def relate_docs(self, doc_id: str) -> dict[str, Any]:
        """Explain how a doc relates to its graph neighbors.

        Uses graph traversal to find connected docs and summarizes
        the relationship types.

        Returns:
            Dict with doc info, neighbors grouped by relationship type.
        """
        doc = self.get_document(doc_id)
        if doc is None:
            return {'error': f'Document {doc_id} not found'}

        neighbors: dict[str, list[dict[str, str]]] = {
            'imports_from': [],
            'imported_by': [],
            'same_topic': [],
            'semantically_similar': [],
        }

        with self._conn_provider.acquire() as conn:
            # Files this doc's source files import
            for sf in doc.source_files:
                for row in conn.execute(
                    'SELECT target_id, edge_type FROM doc_graph WHERE source_id LIKE ?',
                    (f'%{sf}%',),
                ).fetchall():
                    target, etype = row
                    if etype == 'imports':
                        neighbors['imports_from'].append({'file': target.split('/')[-1], 'full_path': target})
                    elif etype == 'topic_member':
                        neighbors['same_topic'].append({'doc': target})

            # Files that import this doc's source files
            for sf in doc.source_files:
                for row in conn.execute(
                    "SELECT source_id FROM doc_graph WHERE target_id LIKE ? AND edge_type = 'imports'",
                    (f'%{sf}%',),
                ).fetchall():
                    neighbors['imported_by'].append({'file': row[0].split('/')[-1], 'full_path': row[0]})

        # Get graph-ranked related docs
        related = self.get_related(doc_id, max_hops=2, limit=5)
        for r in related:
            neighbors['semantically_similar'].append({
                'title': r['title'],
                'distance': r['distance'],
            })

        return {
            'doc_id': doc_id,
            'title': doc.title,
            'content_type': doc.content_type,
            'source_files': doc.source_files,
            'neighbors': neighbors,
        }

    def suggest_topics(self, source_path: Path, min_cluster_size: int = 3) -> list[dict[str, Any]]:
        """Analyze the graph to propose new cross-cutting topic documents.

        Finds clusters of highly-connected files that lack a unifying topic doc.
        Uses import graph to identify modules that work together but have no
        topic doc linking them.

        Args:
            source_path: Source root for path resolution.
            min_cluster_size: Minimum files in a cluster to suggest a topic.

        Returns:
            List of suggested topics with title, files, and rationale.
        """
        from collections import defaultdict

        with self._conn_provider.acquire() as conn:
            # Get all import edges
            edges = conn.execute(
                "SELECT source_id, target_id FROM doc_graph WHERE edge_type = 'imports'"
            ).fetchall()

            # Get files that already have topic docs
            topic_covered = set()
            for row in conn.execute(
                "SELECT source_id FROM doc_graph WHERE edge_type = 'topic_member'"
            ).fetchall():
                topic_covered.add(row[0])

        if not edges:
            return []

        # Build adjacency lists
        neighbors: dict[str, set[str]] = defaultdict(set)
        for src, tgt in edges:
            neighbors[src].add(tgt)
            neighbors[tgt].add(src)

        # Find clusters using connected component analysis on files
        # that share many imports (co-imported frequently)
        # Group by parent directory as a heuristic for topic boundaries
        dir_groups: dict[str, list[str]] = defaultdict(list)
        for f in neighbors:
            if not f.endswith('.py'):
                continue
            parts = f.split('/')
            if len(parts) >= 3:
                # Use grandparent/parent as group key
                group_key = '/'.join(parts[-3:-1])
            elif len(parts) >= 2:
                group_key = parts[-2]
            else:
                continue
            dir_groups[group_key].append(f)

        suggestions = []
        for group_key, files in sorted(dir_groups.items(), key=lambda x: -len(x[1])):
            if len(files) < min_cluster_size:
                continue

            # Check if this group already has a topic doc
            uncovered = [f for f in files if f not in topic_covered]
            if len(uncovered) < min_cluster_size:
                continue

            # Calculate internal connectivity
            internal_edges = sum(
                1 for f in files for n in neighbors.get(f, set()) if n in files
            )

            # Generate a human-readable title from the directory
            title_parts = group_key.replace('_', ' ').split('/')
            title = ' '.join(p.title() for p in title_parts)

            try:
                rel_files = [str(Path(f).relative_to(source_path)) for f in uncovered[:8]]
            except ValueError:
                rel_files = [f.split('/')[-1] for f in uncovered[:8]]

            suggestions.append({
                'title': title,
                'files': rel_files,
                'file_count': len(uncovered),
                'internal_edges': internal_edges,
                'rationale': f'{len(uncovered)} related files with {internal_edges} internal connections, no topic doc',
            })

        suggestions.sort(key=lambda s: s['internal_edges'], reverse=True)
        return suggestions[:10]

    def compare_files(self, file1: str, file2: str) -> dict[str, Any]:
        """Show how two files relate: shared imports, common docs, graph distance.

        Args:
            file1: Path to first file.
            file2: Path to second file.

        Returns:
            Dict with relationship details.
        """
        with self._conn_provider.acquire() as conn:
            # Shared imports (files both import from the same targets)
            imports_1 = {row[0] for row in conn.execute(
                "SELECT target_id FROM doc_graph WHERE source_id LIKE ? AND edge_type = 'imports'",
                (f'%{file1}%',),
            ).fetchall()}
            imports_2 = {row[0] for row in conn.execute(
                "SELECT target_id FROM doc_graph WHERE source_id LIKE ? AND edge_type = 'imports'",
                (f'%{file2}%',),
            ).fetchall()}
            shared_imports = imports_1 & imports_2

            # Direct relationship
            direct = conn.execute(
                'SELECT edge_type FROM doc_graph WHERE '
                '(source_id LIKE ? AND target_id LIKE ?) OR '
                '(source_id LIKE ? AND target_id LIKE ?)',
                (f'%{file1}%', f'%{file2}%', f'%{file2}%', f'%{file1}%'),
            ).fetchall()
            direct_edges = [row[0] for row in direct]

        # Docs that reference both files
        docs1 = self.find_documents_by_source_files([file1])
        docs2 = self.find_documents_by_source_files([file2])
        shared_docs = [d for d in docs1 if d.id in {d2.id for d2 in docs2}]

        relationship = 'unrelated'
        if direct_edges:
            relationship = 'directly connected'
        elif shared_imports:
            relationship = 'share common dependencies'
        elif shared_docs:
            relationship = 'share documentation'

        return {
            'file1': file1,
            'file2': file2,
            'relationship': relationship,
            'direct_edges': direct_edges,
            'shared_imports': [str(p).split('/')[-1] for p in list(shared_imports)[:10]],
            'shared_docs': [d.title for d in shared_docs],
            'file1_imports': len(imports_1),
            'file2_imports': len(imports_2),
        }

    def decision_log(self) -> list[dict[str, str]]:
        """Extract all design decisions from architecture docs into a single log."""
        decisions = []
        for doc in self.list_documents(content_type='architecture'):
            # Find "Design Decisions" or "Design Decision" sections
            sections = re.split(r'^##\s+', doc.content, flags=re.MULTILINE)
            for section in sections:
                header_match = re.match(r'(.+)\n', section)
                if not header_match:
                    continue
                header = header_match.group(1).strip()
                if 'design' in header.lower() and 'decision' in header.lower():
                    # Extract individual decisions (### subsections or lettered items)
                    subsections = re.split(r'^###\s+', section, flags=re.MULTILINE)
                    if len(subsections) > 1:
                        for sub in subsections[1:]:
                            sub_header = sub.split('\n')[0].strip()
                            body = '\n'.join(sub.split('\n')[1:]).strip()[:300]
                            decisions.append({
                                'source': doc.title,
                                'decision': sub_header,
                                'rationale': body,
                            })
                    else:
                        body = section[len(header):].strip()[:300]
                        decisions.append({
                            'source': doc.title,
                            'decision': header,
                            'rationale': body,
                        })
        return decisions

    def review_checklist(self, changed_files: list[str]) -> list[dict[str, str]]:
        """Generate a PR review checklist from Ariadne's knowledge of changed files."""
        checklist: list[dict[str, str]] = []
        seen_checks: set[str] = set()

        for f in changed_files:
            docs = self.find_documents_by_source_files([f])
            gotchas = self.extract_gotchas(f)
            basename = Path(f).stem

            for gotcha in gotchas[:3]:
                check = gotcha['text'][:100]
                if check not in seen_checks:
                    checklist.append({'file': basename, 'check': check, 'type': 'gotcha'})
                    seen_checks.add(check)

            # Extract specific patterns to check from docs
            for doc in docs:
                if 'thread' in doc.content.lower() or 'concurrent' in doc.content.lower():
                    check = 'Verify thread safety'
                    if check not in seen_checks:
                        checklist.append({'file': basename, 'check': check, 'type': 'thread_safety'})
                        seen_checks.add(check)
                if 'temporal' in doc.content.lower() and 'leak' in doc.content.lower():
                    check = 'Check temporal leak prevention'
                    if check not in seen_checks:
                        checklist.append({'file': basename, 'check': check, 'type': 'temporal'})
                        seen_checks.add(check)
                if 'validate' in doc.content.lower() or 'validation' in doc.content.lower():
                    check = 'Verify input validation'
                    if check not in seen_checks:
                        checklist.append({'file': basename, 'check': check, 'type': 'validation'})
                        seen_checks.add(check)

            # Check if tests exist
            tests = self.find_tests_for(f)
            if not tests:
                checklist.append({'file': basename, 'check': 'No tests found — add test coverage', 'type': 'missing_tests'})

        return checklist

    def impact_radius(self, file_path: str) -> dict[str, Any]:
        """Calculate how many files/tests/docs would be affected by changing a file."""
        # Direct dependents (files that import this one)
        dependents: set[str] = set()
        with self._conn_provider.acquire() as conn:
            for row in conn.execute(
                "SELECT source_id FROM doc_graph WHERE target_id LIKE ? AND edge_type = 'imports'",
                (f'%{file_path}%',),
            ).fetchall():
                dependents.add(row[0])

        # Transitive dependents (2 hops)
        transitive: set[str] = set()
        with self._conn_provider.acquire() as conn2:
            for dep in dependents:
                for row2 in conn2.execute(
                    "SELECT source_id FROM doc_graph WHERE target_id LIKE ? AND edge_type = 'imports'",
                    (f'%{dep.split("/")[-1]}%',),
                ).fetchall():
                    transitive.add(row2[0])

        all_affected = dependents | transitive
        affected_docs = self.find_documents_by_source_files([file_path])
        tests = self.find_tests_for(file_path)

        return {
            'file': file_path,
            'direct_dependents': len(dependents),
            'transitive_dependents': len(transitive),
            'total_affected_files': len(all_affected),
            'affected_docs': len(affected_docs),
            'affected_tests': len(tests),
            'radius_score': len(dependents) * 2 + len(transitive) + len(tests),
            'top_dependents': sorted(f.split('/')[-1] for f in list(dependents)[:10]),
        }

    def coupling_report(self, source_path: Path) -> list[dict[str, Any]]:
        """Identify highly-coupled file pairs (many mutual imports/refs)."""
        from collections import Counter

        pair_counts: Counter[tuple[str, str]] = Counter()

        with self._conn_provider.acquire() as conn:
            edges = conn.execute(
                "SELECT source_id, target_id FROM doc_graph WHERE edge_type = 'imports'"
            ).fetchall()

        for src, tgt in edges:
            pair = tuple(sorted([src.split('/')[-1], tgt.split('/')[-1]]))
            pair_counts[pair] += 1

        # Find pairs with bidirectional imports (A imports B AND B imports A)
        edge_set = {(src.split('/')[-1], tgt.split('/')[-1]) for src, tgt in edges}
        bidirectional = []
        seen = set()
        for a, b in edge_set:
            if (b, a) in edge_set and (a, b) not in seen and (b, a) not in seen:
                bidirectional.append({
                    'file1': a, 'file2': b,
                    'mutual_imports': True,
                    'total_edges': pair_counts.get((min(a, b), max(a, b)), 0),
                })
                seen.add((a, b))

        # Also find pairs with high unidirectional coupling
        high_coupling = []
        for pair, count in pair_counts.most_common(20):
            if count >= 2 and pair not in seen:
                high_coupling.append({
                    'file1': pair[0], 'file2': pair[1],
                    'mutual_imports': False,
                    'total_edges': count,
                })

        return sorted(bidirectional + high_coupling, key=lambda x: x['total_edges'], reverse=True)[:20]

    def summarize_file(self, file_path: str) -> str:
        """Generate a one-paragraph summary of a file from its docs."""
        explain_data = self.explain(file_path)

        if explain_data['total_documents'] == 0:
            return f'No documentation available for {file_path}.'

        # Use the first explanation doc's first paragraph
        for ct in ('explanation', 'architecture', 'finding'):
            docs = explain_data['documents'].get(ct, [])
            for doc in docs:
                paragraphs = [p.strip() for p in doc['content'].split('\n\n') if len(p.strip()) > 50]
                if paragraphs:
                    # Skip headers
                    for para in paragraphs:
                        if not para.startswith('#') and not para.startswith('```'):
                            return para[:500]

        return f'{explain_data["summary"]}'

    def detect_conflicts(self, docs: list[Document]) -> list[tuple[Document, Document]]:
        """Detect conflicting documents based on title and source_files.

        Two documents conflict if they:
        - Have the same title (case-insensitive), OR
        - Have overlapping source_files

        Args:
            docs: List of documents to check for conflicts.

        Returns:
            List of (doc1, doc2) tuples representing conflicting pairs.
        """
        conflicts: list[tuple[Document, Document]] = []
        seen_titles: dict[str, Document] = {}
        seen_files: dict[str, Document] = {}

        for doc in docs:
            title_lower = doc.title.lower()

            # Check title conflict
            if title_lower in seen_titles:
                conflicts.append((seen_titles[title_lower], doc))
            else:
                seen_titles[title_lower] = doc

            # Check source file overlap
            for sf in doc.source_files:
                if sf in seen_files:
                    existing = seen_files[sf]
                    # Only add if not already a conflict pair
                    pair = (existing, doc)
                    if pair not in conflicts and (doc, existing) not in conflicts:
                        conflicts.append(pair)
                else:
                    seen_files[sf] = doc

        return conflicts

    def resolve_conflicts(
        self,
        docs: list[Document],
        branch: str | None = None,
        source_precedence: list[str] | None = None,
    ) -> list[Document]:
        """Resolve conflicts between documents using precedence rules.

        Resolution precedence (highest to lowest):
        1. Documents with explicit 'supersedes' metadata
        2. Branch-specific documents over base documents
        3. Subdirectory source over parent source (via source_precedence)
        4. Most recently updated document

        Args:
            docs: List of documents to resolve conflicts for.
            branch: Current git branch for branch-based precedence.
            source_precedence: List of source names in precedence order
                              (first = highest, e.g., child sources before parents).

        Returns:
            List of documents with conflicts resolved (superseded docs removed).
        """
        # Build supersedes mapping
        superseded_ids: set[str] = set()
        for doc in docs:
            supersedes = doc.metadata.get('supersedes')
            if supersedes:
                if isinstance(supersedes, str):
                    superseded_ids.add(supersedes)
                elif isinstance(supersedes, list):
                    superseded_ids.update(supersedes)

        # Remove explicitly superseded documents
        remaining = [d for d in docs if d.id not in superseded_ids]

        # Detect conflicts among remaining docs
        conflicts = self.detect_conflicts(remaining)

        # Resolve each conflict
        to_remove: set[str] = set()
        for doc1, doc2 in conflicts:
            winner = self._resolve_conflict_pair(
                doc1, doc2, branch, source_precedence
            )
            loser = doc2 if winner.id == doc1.id else doc1
            to_remove.add(loser.id)

        # Filter out losing documents
        return [d for d in remaining if d.id not in to_remove]

    def _resolve_conflict_pair(
        self,
        doc1: Document,
        doc2: Document,
        branch: str | None,
        source_precedence: list[str] | None,
    ) -> Document:
        """Resolve a conflict between two documents.

        Args:
            doc1: First document.
            doc2: Second document.
            branch: Current git branch.
            source_precedence: Source names in precedence order.

        Returns:
            The winning document.
        """
        import fnmatch

        # Rule 1: Branch-specific wins over base
        if branch:
            doc1_branches = doc1.metadata.get('branches', [])
            doc2_branches = doc2.metadata.get('branches', [])

            doc1_matches = any(
                fnmatch.fnmatch(branch, p)
                for p in (doc1_branches if isinstance(doc1_branches, list) else [])
            )
            doc2_matches = any(
                fnmatch.fnmatch(branch, p)
                for p in (doc2_branches if isinstance(doc2_branches, list) else [])
            )

            # Branch-specific wins over base (no branches = base)
            if doc1_matches and not doc2_matches:
                return doc1
            if doc2_matches and not doc1_matches:
                return doc2

        # Rule 2: Subdirectory source wins over parent
        if source_precedence:
            doc1_source = doc1.metadata.get('source_name', '')
            doc2_source = doc2.metadata.get('source_name', '')

            if doc1_source in source_precedence and doc2_source in source_precedence:
                doc1_idx = source_precedence.index(doc1_source)
                doc2_idx = source_precedence.index(doc2_source)
                # Lower index = higher precedence
                if doc1_idx < doc2_idx:
                    return doc1
                elif doc2_idx < doc1_idx:
                    return doc2

        # Rule 3: Most recently updated wins
        if doc1.updated_at > doc2.updated_at:
            return doc1
        return doc2
