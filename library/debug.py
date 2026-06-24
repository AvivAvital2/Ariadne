"""Debugging, tracing, code patterns, and gotcha management."""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC
from pathlib import Path
from typing import Any
from ast_utils import safe_ast_parse

_logger = logging.getLogger(__name__)


class DebugMixin:
    """Debugging, stack traces, code patterns, complexity analysis, and gotchas.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    - self.list_documents_lite() from CoreMixin
    - self.find_documents_by_source_files() from SearchMixin
    - self.get_document() from CoreMixin
    - self.list_documents() from CoreMixin
    - self.update_document() from CoreMixin
    - self.explain() from SearchMixin
    - self._documented_paths() from CoreMixin
    """

    # Stack trace analysis

    def stack_explain(self, traceback_text: str) -> list[dict[str, Any]]:
        """Annotate each stack frame with Ariadne doc context.

        Parses a Python traceback, identifies each frame's file and function,
        finds relevant Ariadne docs, and returns annotated frames.

        Returns:
            List of frame dicts with file, function, line, doc_context.
        """
        frames = []
        # Parse standard Python traceback format
        frame_pattern = re.compile(
            r'File "([^"]+)", line (\d+), in (\w+)'
        )

        # Pre-load doc metadata and content ONCE (not per frame)
        lite_docs = self.list_documents_lite()
        # Build a mapping of basename → doc titles for fast lookup
        basename_to_titles: dict[str, list[str]] = {}
        for doc in lite_docs:
            for sf in doc.source_files:
                key = Path(sf).name
                basename_to_titles.setdefault(key, []).append(doc.title)

        for match in frame_pattern.finditer(traceback_text):
            filepath, lineno, funcname = match.group(1), match.group(2), match.group(3)
            basename = Path(filepath).name

            file_docs = basename_to_titles.get(basename, [])

            # SQL-targeted search for function name in content, filtered by file
            context = ''
            for _, _, content, _ in self._find_docs_matching_content(funcname, source_filter=basename):
                for para in content.split('\n\n'):
                    if funcname in para and len(para.strip()) > 30:
                        context = para.strip()[:300]
                        break
                if context:
                    break

            frames.append({
                'file': filepath,
                'line': int(lineno),
                'function': funcname,
                'docs': file_docs[:3],
                'context': context,
            })

        # Extract the final exception line
        exc_match = re.search(r'^(\w*Error|\w*Exception):\s*(.+)$', traceback_text, re.MULTILINE)
        if exc_match:
            frames.append({
                'file': '(exception)',
                'line': 0,
                'function': exc_match.group(1),
                'docs': [],
                'context': exc_match.group(2).strip()[:200],
            })

        return frames

    # Freshen docs

    def freshen_file(self, file_path: str) -> dict[str, Any]:
        """Identify which docs for a file need regeneration.

        Compares doc update timestamps with source file modification time.

        Returns:
            Dict with file, docs (with staleness info), and recommendation.
        """
        import os

        source_mtime = None
        p = Path(file_path)
        if p.exists():
            source_mtime = os.path.getmtime(p)

        docs = self.find_documents_by_source_files([file_path])
        doc_info = []
        stale_count = 0

        for doc in docs:
            from datetime import datetime
            try:
                doc_time = datetime.fromisoformat(doc.updated_at).timestamp()
            except (ValueError, TypeError):
                doc_time = 0

            is_stale = source_mtime is not None and doc_time < source_mtime
            if is_stale:
                stale_count += 1

            doc_info.append({
                'id': doc.id,
                'title': doc.title,
                'content_type': doc.content_type,
                'updated_at': doc.updated_at,
                'is_stale': is_stale,
            })

        return {
            'file': file_path,
            'total_docs': len(docs),
            'stale_docs': stale_count,
            'docs': doc_info,
            'recommendation': f'Regenerate {stale_count} stale docs' if stale_count > 0
                             else 'All docs are up-to-date' if docs
                             else 'No docs exist — generate with: ariadne generate',
        }

    # Code analysis

    def detect_patterns(self, source_path: Path) -> list[dict[str, Any]]:
        """Detect recurring code patterns across the codebase.

        Scans source files for common patterns: singleton, factory,
        async/sync dual, attrs frozen/define, abstract base classes,
        context managers, and protocol implementations.

        Returns:
            List of detected patterns with name, count, and example files.
        """
        import ast
        from collections import defaultdict

        from docgen.staleness import find_python_files

        pattern_files: dict[str, list[str]] = defaultdict(list)
        all_files = find_python_files(source_path)

        for f in all_files:
            if f.name == '__init__.py':
                continue
            try:
                source = f.read_text(encoding='utf-8')
                tree = safe_ast_parse(source, filename=str(f))
            except (SyntaxError, UnicodeDecodeError):
                continue

            rel = str(f.relative_to(source_path)) if f.is_relative_to(source_path) else f.name

            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    decorators = [ast.unparse(d) if isinstance(d, (ast.Attribute, ast.Call)) else getattr(d, 'id', '') for d in node.decorator_list]
                    bases = [ast.unparse(b) for b in node.bases]

                    if any('frozen' in d for d in decorators):
                        pattern_files['attrs @frozen (immutable)'].append(rel)
                    if any('define' in d for d in decorators):
                        pattern_files['attrs @define (mutable)'].append(rel)
                    if 'ABC' in bases or any('ABC' in b for b in bases):
                        pattern_files['Abstract Base Class'].append(rel)
                    if any('Protocol' in b for b in bases):
                        pattern_files['Protocol'].append(rel)

                    # Check for singleton pattern (class-level _instance)
                    for item in node.body:
                        if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                            if item.target.id == '_instance':
                                pattern_files['Singleton'].append(rel)

                    # Check for context manager
                    methods = {n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
                    if '__enter__' in methods and '__exit__' in methods:
                        pattern_files['Context Manager (sync)'].append(rel)
                    if '__aenter__' in methods and '__aexit__' in methods:
                        pattern_files['Context Manager (async)'].append(rel)

                    # Async/sync dual pattern
                    has_async = any(isinstance(n, ast.AsyncFunctionDef) for n in node.body)
                    has_sync = any(isinstance(n, ast.FunctionDef) and not n.name.startswith('_') for n in node.body)
                    if has_async and has_sync:
                        pattern_files['Async/Sync Dual'].append(rel)

                elif isinstance(node, ast.FunctionDef):
                    # Factory functions
                    if node.name.startswith('create_') or node.name.startswith('from_'):
                        pattern_files['Factory Function'].append(rel)

        results = []
        for pattern, files in sorted(pattern_files.items(), key=lambda x: -len(x[1])):
            unique_files = sorted(set(files))
            results.append({
                'pattern': pattern,
                'count': len(unique_files),
                'files': unique_files[:10],
            })

        return results

    def file_complexity(self, source_path: Path) -> list[dict[str, Any]]:
        """Score files by complexity and cross-reference with doc coverage.

        Measures: line count, class count, function count, import count.
        Flags complex + undocumented files as highest risk.

        Returns:
            List sorted by risk score (complexity x missing docs), descending.
        """
        import ast

        from docgen.staleness import find_python_files

        documented_paths = self._documented_paths()

        results = []
        for f in find_python_files(source_path):
            if f.name == '__init__.py':
                continue
            try:
                source = f.read_text(encoding='utf-8')
                tree = safe_ast_parse(source, filename=str(f))
            except (SyntaxError, UnicodeDecodeError):
                continue

            lines = len(source.split('\n'))
            classes = sum(1 for n in ast.walk(tree) if isinstance(n, ast.ClassDef))
            functions = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)))
            imports = sum(1 for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)))

            complexity = lines * 0.01 + classes * 5 + functions * 2 + imports * 0.5
            has_docs = str(f) in documented_paths
            risk = complexity * (0 if has_docs else 1)

            rel = str(f.relative_to(source_path)) if f.is_relative_to(source_path) else f.name
            results.append({
                'file': rel,
                'lines': lines,
                'classes': classes,
                'functions': functions,
                'imports': imports,
                'complexity': round(complexity, 1),
                'documented': has_docs,
                'risk_score': round(risk, 1),
            })

        results.sort(key=lambda r: r['risk_score'], reverse=True)
        return results

    def who_knows(self, query: str, source_path: Path) -> list[dict[str, Any]]:
        """Identify developers with the most context on a topic.

        Uses git blame on files referenced by matching docs to find
        who authored the most relevant code.

        Returns:
            List of authors sorted by relevance score.
        """
        import subprocess
        from collections import Counter

        # Phase 1: lite scan for title matches (fast, no content loaded)
        matching_files = set()
        query_lower = query.lower()
        title_match_ids = []
        for doc in self.list_documents_lite():
            if query_lower in doc.title.lower():
                matching_files.update(doc.source_files)
            else:
                title_match_ids.append(doc.id)

        # Phase 2: check content only for non-title-matches (SQL pre-filter)
        if title_match_ids:
            for _, _, _, source_files_str in self._find_docs_matching_content(query):
                source_files = json.loads(source_files_str) if source_files_str else []
                matching_files.update(source_files)

        if not matching_files:
            return []

        author_lines: Counter[str] = Counter()
        for sf in matching_files:
            sf_path = Path(sf)
            if not sf_path.exists():
                continue
            try:
                result = subprocess.run(
                    ['git', 'blame', '--porcelain', str(sf_path)],
                    capture_output=True, text=True, timeout=10,
                    cwd=source_path,
                )
                if result.returncode != 0:
                    continue
                for line in result.stdout.split('\n'):
                    if line.startswith('author '):
                        author = line[7:].strip()
                        if author and author != 'Not Committed Yet':
                            author_lines[author] += 1
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue

        total = sum(author_lines.values())
        return [
            {'author': author, 'lines': count, 'percentage': round(count / total * 100, 1)}
            for author, count in author_lines.most_common(10)
        ]

    # Debugging support

    def diagnose(self, error_message: str, limit: int = 5) -> dict[str, Any]:
        """Search docs for code paths related to an error message or stack trace.

        Extracts file paths, function names, and error types from the message,
        then finds relevant docs and explains likely causes.

        Args:
            error_message: Stack trace, error text, or symptoms description.

        Returns:
            Dict with matched_docs, extracted_files, extracted_errors, and context.
        """
        # Extract file paths from stack trace
        file_refs = re.findall(r'(?:File\s+["\']|/)(\S+\.py)', error_message)
        # Extract function names
        func_refs = re.findall(r'(?:in |def |\.)\b(\w+)\b\(', error_message)
        # Extract exception types
        error_types = re.findall(r'\b(\w*Error|\w*Exception)\b', error_message)

        # Find docs for referenced files
        matched_docs = []
        if file_refs:
            docs = self.find_documents_by_source_files(file_refs)
            matched_docs.extend({'title': d.title, 'id': d.id, 'match': 'file_reference'} for d in docs[:limit])

        # Search by error type + function names as query
        search_terms = ' '.join(set(error_types + func_refs[:3]))
        if search_terms.strip():
            query_results = self._text_search(search_terms, limit)
            matched_docs.extend(
                {'title': d.title, 'id': d.id, 'match': 'content_search'}
                for d in query_results if d.id not in {m['id'] for m in matched_docs}
            )

        return {
            'extracted_files': list(set(file_refs))[:10],
            'extracted_functions': list(set(func_refs))[:10],
            'extracted_errors': list(set(error_types)),
            'matched_docs': matched_docs[:limit],
            'doc_count': len(matched_docs),
        }

    def _find_docs_matching_content(self, keyword: str, source_filter: str | None = None) -> list:
        """Find documents whose content contains a keyword, using SQL pre-filter.

        Args:
            keyword: Text to search for in document content.
            source_filter: If set, also require this string in source_files.

        Returns:
            List of (id, title, content, source_files) tuples for matching docs.
        """
        if source_filter:
            query_sql = (
                'SELECT id, title, content, source_files FROM documents '
                'WHERE content LIKE ? AND source_files LIKE ?'
            )
            params: tuple = (f'%{keyword}%', f'%{source_filter}%')
        else:
            query_sql = 'SELECT id, title, content, source_files FROM documents WHERE content LIKE ?'
            params = (f'%{keyword}%',)

        with self._conn_provider.acquire() as conn:
            return conn.execute(query_sql, params).fetchall()

    def _text_search(self, query: str, limit: int = 5) -> list:
        """Simple text search across doc titles and content.

        Uses SQL LIKE pre-filter to avoid loading all documents into memory.
        """
        query_lower = query.lower()
        words = query_lower.split()
        if not words:
            return []

        # SQL pre-filter: only load docs where title or content matches at least one word
        conditions = ' OR '.join(['LOWER(title) LIKE ? OR SUBSTR(LOWER(content), 1, 3000) LIKE ?'] * len(words))
        params = []
        for w in words:
            params.extend([f'%{w}%', f'%{w}%'])

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'SELECT id, title, SUBSTR(content, 1, 3000) AS content_head, content_type, source_files '
                f'FROM documents WHERE {conditions}',
                params,
            ).fetchall()

        scored = []
        for row in rows:
            doc_id, title, content_head, content_type, source_files_str = row
            title_lower = title.lower()
            content_lower = content_head.lower() if content_head else ''
            score = 0.0
            for w in words:
                if w in title_lower:
                    score += 2
                if w in content_lower:
                    score += 0.5
            if score > 0:
                # Load the full document only for the scored results
                doc = self.get_document(doc_id)
                if doc:
                    scored.append((doc, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [d for d, _ in scored[:limit]]

    def trace_function(self, function_name: str) -> dict[str, Any]:
        """Show the call chain for a function from graph + docs.

        Returns:
            Dict with callers, callees, docs, and error paths.
        """
        # Find docs mentioning this function (SQL pre-filter)
        matching_docs = []
        callers = []
        callees = []

        for doc_id, title, content, source_files_str in self._find_docs_matching_content(function_name):
            matching_docs.append({'title': title, 'id': doc_id})
            source_files = json.loads(source_files_str) if source_files_str else []

            calls = re.findall(rf'{function_name}\s*\(', content)
            if calls:
                callers.extend(source_files)

            error_patterns = re.findall(
                rf'(?:raise\s+(\w+).*?(?:in|near|from)\s+.*?{function_name})',
                content, re.IGNORECASE,
            )
            if not error_patterns:
                error_patterns = re.findall(r'raise\s+(\w+Error|\w+Exception)', content)

        # Find graph neighbors for files that contain this function
        with self._conn_provider.acquire() as conn:
            for doc in matching_docs[:3]:
                doc_data = self.get_document(doc['id'])
                if doc_data:
                    for sf in doc_data.source_files:
                        for row in conn.execute(
                            "SELECT target_id FROM doc_graph WHERE source_id LIKE ? AND edge_type = 'imports'",
                            (f'%{sf}%',),
                        ).fetchall():
                            callees.append(row[0].split('/')[-1])

        return {
            'function': function_name,
            'docs': matching_docs[:10],
            'source_files': list(set(callers))[:10],
            'imports_to': sorted(set(callees))[:10],
            'doc_count': len(matching_docs),
        }

    def side_effects(self, function_name: str) -> dict[str, Any]:
        """Identify side effects of a function from docs + code patterns.

        Returns:
            Dict with categorized side effects: db_writes, file_io, state_mutations, api_calls.
        """
        effects: dict[str, list[str]] = {
            'db_writes': [], 'file_io': [], 'state_mutations': [],
            'api_calls': [], 'exceptions': [],
        }

        for _, title, content, _ in self._find_docs_matching_content(function_name):
            if any(kw in content for kw in ('INSERT', 'UPDATE', 'DELETE', 'execute(', 'conn.')):
                effects['db_writes'].append(title)
            if any(kw in content for kw in ('write_text', 'write_bytes', 'open(', 'Path(', '.write(')):
                effects['file_io'].append(title)
            if any(kw in content for kw in ('self._', 'self.', 'state', 'cache', 'register')):
                effects['state_mutations'].append(title)
            if any(kw in content for kw in ('httpx', 'requests', 'api', 'client.post', 'client.get')):
                effects['api_calls'].append(title)
            for exc in re.findall(r'raise\s+(\w+)', content):
                effects['exceptions'].append(exc)

        # Deduplicate
        for k in effects:
            effects[k] = sorted(set(effects[k]))

        return {'function': function_name, 'effects': effects}

    def debug_context(self, file_path: str, source_path: Path | None = None) -> dict[str, Any]:
        """One-shot context dump for debugging a file.

        Assembles: file docs, dependencies, recent git changes, known issues,
        and related test files.
        """
        import subprocess

        explain_data = self.explain(file_path)

        # Find test files (reuse find_tests_for)
        test_files = [t['path'].split('/')[-1] for t in self.find_tests_for(file_path)]

        # Recent git changes
        recent_changes = []
        if source_path:
            try:
                result = subprocess.run(
                    ['git', 'log', '--oneline', '-5', '--', file_path],
                    capture_output=True, text=True, timeout=5, cwd=source_path,
                )
                if result.returncode == 0:
                    recent_changes = [l.strip() for l in result.stdout.strip().split('\n') if l.strip()]
            except Exception:
                pass

        # Known issues (findings only — much smaller scan than all docs)
        known_issues = []
        for doc in self.list_documents(content_type='finding'):
            if file_path.rsplit('/', maxsplit=1)[-1] in doc.content:
                known_issues.append(doc.title)

        return {
            'file': file_path,
            'docs': explain_data,
            'test_files': sorted(set(test_files)),
            'recent_changes': recent_changes,
            'known_issues': known_issues,
            'graph_neighbors': explain_data.get('graph_neighbors', []),
        }

    def find_tests_for(self, file_path: str) -> list[dict[str, str]]:
        """Find test files that exercise a given source file.

        Checks: naming convention (test_<name>.py), import analysis,
        and doc cross-references.

        Returns:
            List of test file dicts with path and match_type.
        """
        basename = Path(file_path).stem
        results = []
        seen_files: set[str] = set()

        # 1. By naming convention (lite query — only needs source_files)
        for doc in self.list_documents_lite():
            for sf in doc.source_files:
                sf_name = sf.split('/')[-1]
                if sf_name.startswith('test_') and basename in sf_name:
                    if sf not in seen_files:
                        results.append({'path': sf, 'match_type': 'naming_convention'})
                        seen_files.add(sf)

        # 2. By graph imports (test files that import this file)
        with self._conn_provider.acquire() as conn:
            for row in conn.execute(
                "SELECT source_id FROM doc_graph WHERE target_id LIKE ? AND edge_type = 'imports'",
                (f'%{file_path}%',),
            ).fetchall():
                sf = row[0]
                if 'test' in sf.lower() and sf not in seen_files:
                    results.append({'path': sf, 'match_type': 'imports'})
                    seen_files.add(sf)

        return results

    def find_tests_for_topic(self, topic: str) -> list[dict[str, str]]:
        """Find all tests related to a topic by searching docs and graph.

        Args:
            topic: Topic name like "ingest", "serialization", "temporal".

        Returns:
            List of test file dicts with path and relevance.
        """
        topic_lower = topic.lower()
        test_files: dict[str, str] = {}  # path -> reason

        # 1. Find docs matching the topic
        matching_docs = self._text_search(topic, limit=10)
        for doc in matching_docs:
            for sf in doc.source_files:
                # Find tests for each related source file
                for test in self.find_tests_for(sf):
                    if test['path'] not in test_files:
                        test_files[test['path']] = f'tests {sf.split("/")[-1]} (matches topic "{topic}")'

        # 2. Find test files whose names match the topic (lite — only needs paths)
        for doc in self.list_documents_lite():
            for sf in doc.source_files:
                if 'test' in sf.lower() and topic_lower in sf.lower():
                    if sf not in test_files:
                        test_files[sf] = f'filename matches topic "{topic}"'

        return [{'path': p, 'relevance': r} for p, r in sorted(test_files.items())]

    def extract_gotchas(self, file_path: str) -> list[dict[str, str]]:
        """Extract documented pitfalls and gotchas for a file.

        Searches docs for warning patterns: "gotcha", "pitfall", "careful",
        "note:", "warning:", "important:", "caveat".

        Returns:
            List of gotchas with source doc and text.
        """
        gotchas = []
        warning_patterns = re.compile(
            r'(?:gotcha|pitfall|careful|caveat|warning|important|note|beware|'
            r'do not|don\'t|never|avoid|watch out|silently|subtle)',
            re.IGNORECASE,
        )

        basename = Path(file_path).stem
        for _, title, content, source_files_str in self._find_docs_matching_content(basename):
            source_files = json.loads(source_files_str) if source_files_str else []
            if not any(basename in sf for sf in source_files) and basename not in title.lower():
                continue

            for line in content.split('\n'):
                if warning_patterns.search(line) and len(line.strip()) > 20:
                    gotchas.append({
                        'source': title,
                        'text': line.strip()[:200],
                    })

        # Deduplicate by text
        seen = set()
        unique = []
        for g in gotchas:
            if g['text'] not in seen:
                seen.add(g['text'])
                unique.append(g)

        return unique[:20]

    def get_gotchas(self, file_paths: list[str], limit: int = 20) -> list:
        """Return gotcha documents whose source_files overlap with the given paths.

        Filters out documents with ``metadata.status == "deprecated"``.

        Args:
            file_paths: File paths to match against document source_files.
            limit: Maximum number of documents to return.

        Returns:
            List of gotcha documents sorted by encounter_count descending.
        """
        docs = self.list_documents(content_type='gotcha')
        path_set = set(file_paths)
        matched = [
            doc for doc in docs
            if path_set & set(doc.source_files)
            and (doc.metadata or {}).get('status') != 'deprecated'
        ]
        matched.sort(
            key=lambda d: d.metadata.get('encounter_count', 0),  # type: ignore[union-attr]
            reverse=True,
        )
        return matched[:limit]

    def deprecate_stale_gotchas(self, days: int = 30) -> int:
        """Mark gotcha documents older than *days* as deprecated.

        Sets ``metadata["status"] = "deprecated"`` on each stale gotcha
        whose ``updated_at`` is older than the cutoff.

        Args:
            days: Number of days after which a gotcha is considered stale.

        Returns:
            Count of gotchas that were deprecated.
        """
        from datetime import datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
        docs = self.list_documents(content_type='gotcha')
        count = 0
        for doc in docs:
            if (doc.metadata or {}).get('status') == 'deprecated':
                continue
            if doc.updated_at <= cutoff:
                meta = dict(doc.metadata or {})
                meta['status'] = 'deprecated'
                self.update_document(doc.id, metadata=meta)
                count += 1
        return count
