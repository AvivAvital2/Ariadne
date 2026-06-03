"""Quality assessment, health checks, linting, and debt analysis."""
from __future__ import annotations

import logging
import re
from datetime import UTC
from pathlib import Path
from typing import Any

import numpy as np

from schema import ContentType

_logger = logging.getLogger(__name__)


class QualityMixin:
    """Quality, linting, health checks, and debt scoring.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    - self.list_documents() from CoreMixin
    - self.list_documents_lite() from CoreMixin
    """

    def health_check(self) -> dict[str, Any]:
        """Run a comprehensive health check on the library.

        Checks database integrity, embedding coverage, orphaned chunks,
        and configuration status.

        Returns:
            Dict with 'status' ('healthy', 'warnings', 'errors'),
            'checks' list, and 'summary'.
        """
        import os

        checks: list[dict[str, str]] = []
        errors = 0
        warnings = 0

        # 1. Database exists and is readable
        if self.path.exists():
            db_size = os.path.getsize(self.path)
            checks.append({'name': 'database', 'status': 'ok', 'detail': f'{db_size / 1024 / 1024:.1f} MB'})
        else:
            checks.append({'name': 'database', 'status': 'error', 'detail': f'Not found: {self.path}'})
            errors += 1

        with self._conn_provider.acquire() as conn:
            # 2. Document count
            doc_count = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            checks.append({'name': 'documents', 'status': 'ok', 'detail': f'{doc_count} documents'})

            # 3. Embedding coverage
            with_embeddings = conn.execute('SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL').fetchone()[0]
            if doc_count > 0:
                pct = with_embeddings * 100 // doc_count
                status = 'ok' if pct >= 90 else 'warning' if pct >= 50 else 'error'
                if status == 'warning':
                    warnings += 1
                elif status == 'error':
                    errors += 1
                checks.append({'name': 'embeddings', 'status': status, 'detail': f'{with_embeddings}/{doc_count} ({pct}%)'})

            # 4. Orphaned chunks (parent document deleted)
            orphans = conn.execute(
                'SELECT COUNT(*) FROM chunks c LEFT JOIN documents d ON c.document_id = d.id WHERE d.id IS NULL'
            ).fetchone()[0]
            if orphans > 0:
                checks.append({'name': 'orphaned_chunks', 'status': 'warning', 'detail': f'{orphans} orphaned chunks'})
                warnings += 1
            else:
                checks.append({'name': 'orphaned_chunks', 'status': 'ok', 'detail': 'None'})

            # 5. Chunk count
            chunk_count = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()[0]
            checks.append({'name': 'chunks', 'status': 'ok', 'detail': f'{chunk_count} chunks'})

            # 6. Graph status
            graph_edges = conn.execute('SELECT COUNT(*) FROM doc_graph').fetchone()[0]
            if graph_edges > 0:
                checks.append({'name': 'graph', 'status': 'ok', 'detail': f'{graph_edges} edges'})
            else:
                checks.append({'name': 'graph', 'status': 'warning', 'detail': 'Not built (run: ariadne graph --build)'})
                warnings += 1

            # 7. Duplicate titles
            dupes = conn.execute(
                'SELECT title, COUNT(*) as cnt FROM documents GROUP BY title HAVING cnt > 1'
            ).fetchall()
            if dupes:
                checks.append({'name': 'duplicate_titles', 'status': 'warning', 'detail': f'{len(dupes)} duplicate titles'})
                warnings += 1
            else:
                checks.append({'name': 'duplicate_titles', 'status': 'ok', 'detail': 'None'})

        # 8. Stale source file references
        stale_refs = self.find_stale_source_refs()
        if stale_refs:
            checks.append({'name': 'stale_source_refs', 'status': 'warning',
                           'detail': f'{len(stale_refs)} docs reference missing files'})
            warnings += 1
        else:
            checks.append({'name': 'stale_source_refs', 'status': 'ok', 'detail': 'All source files exist'})

        overall = 'healthy' if errors == 0 and warnings == 0 else 'warnings' if errors == 0 else 'errors'
        return {
            'status': overall,
            'checks': checks,
            'errors': errors,
            'warnings': warnings,
            'summary': f'{overall}: {len(checks)} checks, {errors} errors, {warnings} warnings',
        }

    def _documented_paths(self) -> set[str]:
        """Get all source file paths that have documentation."""
        paths: set[str] = set()
        for doc in self.list_documents_lite():
            paths.update(doc.source_files)
        return paths

    # Documentation debt score

    def doc_debt_score(self, source_path: Path) -> dict[str, Any]:
        """Calculate a documentation debt score (0-100, lower is better).

        Combines: coverage %, staleness, hit rate, graph priority gaps,
        and duplicate titles into a single actionable metric.

        Args:
            source_path: Source root for coverage calculation.

        Returns:
            Dict with 'score' (0-100), 'grade' (A-F), component scores,
            and improvement suggestions.
        """
        from docgen.staleness import find_python_files

        suggestions: list[str] = []

        # 1. Coverage (0-40 points of debt)
        all_files = [f for f in find_python_files(source_path) if f.name != '__init__.py']
        documented_paths = self._documented_paths()
        undocumented = [f for f in all_files if str(f) not in documented_paths]
        coverage_pct = (len(all_files) - len(undocumented)) / max(1, len(all_files)) * 100
        coverage_debt = max(0, 40 - coverage_pct * 0.4)
        if coverage_debt > 20:
            suggestions.append(f'Generate docs for {len(undocumented)} undocumented files (run: ariadne improve)')

        # 2. Embedding coverage (0-15 points)
        with self._conn_provider.acquire() as conn:
            total_docs = conn.execute('SELECT COUNT(*) FROM documents').fetchone()[0]
            with_embeddings = conn.execute('SELECT COUNT(*) FROM documents WHERE embedding IS NOT NULL').fetchone()[0]
        embed_pct = with_embeddings / max(1, total_docs) * 100
        embed_debt = max(0, 15 - embed_pct * 0.15)
        if embed_debt > 5:
            suggestions.append(f'Rebuild embeddings ({total_docs - with_embeddings} docs missing, run: ariadne rebuild)')

        # 3. Duplicate titles (0-15 points)
        dupes = self.find_duplicate_titles()
        dupe_count = sum(len(ids) - 1 for ids in dupes.values())
        dupe_debt = min(15, dupe_count * 0.5)
        if dupe_debt > 3:
            suggestions.append(f'Fix {dupe_count} duplicate doc titles (degrades search accuracy)')

        # 4. Graph coverage (0-15 points)
        graph_stats = self.get_graph_stats()
        if graph_stats['total_edges'] == 0:
            graph_debt = 15.0
            suggestions.append('Build the dependency graph (run: ariadne graph --build)')
        else:
            priorities = self.get_priorities(source_path)
            high_priority_undoc = sum(1 for p in priorities if p['priority_score'] > 5 and p['doc_count'] == 0)
            graph_debt = min(15, high_priority_undoc * 1.5)
            if high_priority_undoc > 0:
                suggestions.append(f'{high_priority_undoc} high-connectivity files lack docs')

        # 5. Hit rate (0-15 points)
        usage = self.get_usage_stats(days=30)
        hit_rate = usage.get('hit_rate', 0)
        search_calls = usage.get('total_calls', 0)
        if search_calls >= 5:
            hit_debt = max(0, 15 - hit_rate * 15)
        else:
            hit_debt = 7.5  # Not enough data — neutral
        if hit_rate < 0.3 and search_calls >= 5:
            suggestions.append(f'Hit rate is {hit_rate:.0%} — review miss feedback (run: ariadne gaps)')

        total_debt = coverage_debt + embed_debt + dupe_debt + graph_debt + hit_debt
        score = round(total_debt, 1)

        # Grade
        if score < 10:
            grade = 'A'
        elif score < 25:
            grade = 'B'
        elif score < 45:
            grade = 'C'
        elif score < 65:
            grade = 'D'
        else:
            grade = 'F'

        return {
            'score': score,
            'grade': grade,
            'components': {
                'coverage_debt': round(coverage_debt, 1),
                'embedding_debt': round(embed_debt, 1),
                'duplicate_debt': round(dupe_debt, 1),
                'graph_debt': round(graph_debt, 1),
                'hit_rate_debt': round(hit_debt, 1),
            },
            'coverage_percent': round(coverage_pct, 1),
            'suggestions': suggestions,
        }

    def lint_docs(self) -> list[dict[str, Any]]:
        """Check document quality: missing sections, broken refs, inconsistencies.

        Returns:
            List of issues found, each with doc_id, title, issue_type, detail.
        """
        # Preload all doc IDs to avoid N+1 queries for reference checking
        all_doc_ids = {doc.id for doc in self.list_documents_lite()}
        issues: list[dict[str, Any]] = []

        for doc in self.list_documents():
            # Check for very short content (likely incomplete)
            if len(doc.content) < 100:
                issues.append({
                    'doc_id': doc.id, 'title': doc.title,
                    'issue_type': 'too_short',
                    'detail': f'Content is only {len(doc.content)} chars',
                })

            # Check for missing Overview/Summary section
            if doc.content_type in ('explanation', 'architecture'):
                content_lower = doc.content.lower()
                if 'overview' not in content_lower and 'summary' not in content_lower:
                    issues.append({
                        'doc_id': doc.id, 'title': doc.title,
                        'issue_type': 'missing_overview',
                        'detail': 'No Overview or Summary section found',
                    })

            # Check for broken internal doc references
            refs = re.findall(r'\[([^\]]+)\]\(([a-f0-9-]+\.md)\)', doc.content)
            for ref_title, ref_file in refs:
                ref_id = ref_file.replace('.md', '')
                if ref_id not in all_doc_ids:
                    issues.append({
                        'doc_id': doc.id, 'title': doc.title,
                        'issue_type': 'broken_reference',
                        'detail': f'References non-existent doc: {ref_title} ({ref_id})',
                    })

            # Check for empty source_files
            if not doc.source_files and doc.content_type in ('explanation', 'architecture'):
                issues.append({
                    'doc_id': doc.id, 'title': doc.title,
                    'issue_type': 'no_source_files',
                    'detail': 'No source files linked',
                })

        return issues

    def shrink_suggestions(self, max_chars: int = 12000) -> list[dict[str, Any]]:
        """Propose specific splits for oversized documents.

        Analyzes large docs to find natural split points (## headers)
        and suggests how to break them into focused pieces.

        Returns:
            List of suggestions with doc info and proposed split titles.
        """
        suggestions = []

        # Only load content for oversized docs (SQL pre-filter)
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT id, title, content FROM documents WHERE LENGTH(content) > ?',
                (max_chars,),
            ).fetchall()

        for row in rows:
            doc_id, title, content = row
            headers = re.findall(r'^## (.+)$', content, re.MULTILINE)
            if len(headers) >= 3:
                suggestions.append({
                    'doc_id': doc_id,
                    'title': title,
                    'content_size': len(content),
                    'sections': len(headers),
                    'proposed_splits': [f'{title} — {h}' for h in headers[:6]],
                })

        suggestions.sort(key=lambda s: s['content_size'], reverse=True)
        return suggestions

    # Semantic deduplication

    def find_semantic_duplicates(self, similarity_threshold: float = 0.95) -> list[dict[str, Any]]:
        """Find document pairs with very similar content using embeddings.

        Goes beyond title matching to find docs with nearly identical content
        that could be merged.

        Args:
            similarity_threshold: Minimum cosine similarity to flag as duplicate (default: 0.95).

        Returns:
            List of dicts with doc1_id, doc1_title, doc2_id, doc2_title, similarity.
        """
        # Use lite docs + separate embedding load (avoids loading full content)
        lite_docs = self.list_documents_lite()
        doc_ids = [d.id for d in lite_docs]
        embeddings_map = self.get_embeddings_for_ids(doc_ids)
        docs = [(d, embeddings_map[d.id]) for d in lite_docs if d.id in embeddings_map]
        if len(docs) < 2:
            return []

        embeddings = np.stack([emb for _, emb in docs])
        sim_matrix = embeddings @ embeddings.T

        duplicates = []
        for i in range(len(docs)):
            for j in range(i + 1, len(docs)):
                sim = float(sim_matrix[i, j])
                if sim >= similarity_threshold:
                    doc_i, _ = docs[i]
                    doc_j, _ = docs[j]
                    duplicates.append({
                        'doc1_id': doc_i.id,
                        'doc1_title': doc_i.title,
                        'doc2_id': doc_j.id,
                        'doc2_title': doc_j.title,
                        'similarity': round(sim, 4),
                    })

        duplicates.sort(key=lambda d: d['similarity'], reverse=True)
        return duplicates[:50]

    def find_low_value_documents(self, min_serves: int = 3, days: int | None = 30) -> list[dict[str, Any]]:
        """Find documents that are served frequently but never in a hit event.

        These docs look relevant to the search engine (returned by embedding
        similarity) but aren't actually useful to the caller (never marked as
        hit). Candidates for rewriting, splitting, or removal.

        Args:
            min_serves: Minimum times served to be considered (default: 3).
            days: Limit to last N days (None = all time).

        Returns:
            List of dicts with document_id, title, serve_count, hit_count, sorted by serve_count desc.
        """
        import json as json_mod
        from collections import Counter
        from datetime import datetime, timedelta

        where = ''
        params: list[str] = []
        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            where = 'AND timestamp >= ?'
            params.append(cutoff)

        served: Counter[str] = Counter()
        hit_in: Counter[str] = Counter()

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'SELECT outcome, returned_document_ids FROM usage_events '
                f'WHERE returned_document_ids IS NOT NULL {where}',
                params,
            ).fetchall()

            for outcome, doc_ids_json in rows:
                for did in json_mod.loads(doc_ids_json) if doc_ids_json else []:
                    served[did] += 1
                    if outcome == 'hit':
                        hit_in[did] += 1

            results = []
            for did, count in served.most_common():
                if count < min_serves:
                    break
                hits = hit_in.get(did, 0)
                if hits == 0:
                    title_row = conn.execute('SELECT title, content_type, LENGTH(content) FROM documents WHERE id = ?', (did,)).fetchone()
                    if title_row:
                        results.append({
                            'document_id': did,
                            'title': title_row[0],
                            'content_type': title_row[1],
                            'content_size': title_row[2],
                            'serve_count': count,
                            'hit_count': 0,
                        })

        return results

    def find_stale_source_refs(self) -> list[dict[str, str]]:
        """Find documents whose source_files paths no longer exist on disk.

        Returns:
            List of dicts with document_id, title, missing_file.
        """
        results = []
        for doc in self.list_documents_lite():
            for sf in doc.source_files:
                if not Path(sf).exists():
                    results.append({
                        'document_id': doc.id,
                        'title': doc.title,
                        'missing_file': sf,
                    })
                    break  # One missing file per doc is enough
        return results

    def find_oversized_documents(self, max_chars: int = 15000) -> list[dict[str, Any]]:
        """Find documents larger than a threshold that may benefit from splitting.

        Large documents consume more tokens when served via MCP and may contain
        mixed topics that reduce search precision.

        Args:
            max_chars: Character threshold (default: 15000).

        Returns:
            List of dicts with id, title, content_type, content_size, sorted by size desc.
        """
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT id, title, content_type, LENGTH(content) as size '
                'FROM documents WHERE LENGTH(content) > ? ORDER BY size DESC',
                (max_chars,),
            ).fetchall()

        return [
            {'id': row[0], 'title': row[1], 'content_type': row[2], 'content_size': row[3]}
            for row in rows
        ]

    # Deduplication

    def find_duplicate_titles(self) -> dict[str, list[str]]:
        """Find documents with duplicate titles.

        Returns:
            Dict mapping title to list of document IDs that share it.
            Only includes titles with 2+ documents.
        """
        from collections import defaultdict

        by_title: dict[str, list[str]] = defaultdict(list)
        for doc in self.list_documents_lite():
            by_title[doc.title].append(doc.id)
        return {t: ids for t, ids in by_title.items() if len(ids) > 1}

    # Stats and utilities

    def count_documents(self, content_type: ContentType | None = None) -> int:
        """Count documents in the library.

        Args:
            content_type: Filter by content type.

        Returns:
            Number of documents.
        """
        query = 'SELECT COUNT(*) FROM documents'
        params: list[object] = []

        if content_type is not None:
            query += ' WHERE content_type = ?'
            params.append(content_type)

        with self._conn_provider.acquire() as conn:
            result = conn.execute(query, params).fetchone()
            return int(result[0]) if result else 0

    def count_chunks(self) -> int:
        """Count total chunks in the library.

        Returns:
            Number of chunks.
        """
        with self._conn_provider.acquire() as conn:
            result = conn.execute('SELECT COUNT(*) FROM chunks').fetchone()
            return int(result[0]) if result else 0
