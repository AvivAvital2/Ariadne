"""Dependency graph management — build, query, and export."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

_logger = logging.getLogger(__name__)


class GraphMixin:
    """Dependency graph building, querying, and visualization.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    """

    def build_graph(
        self,
        source_path: Path,
        extra_source_paths: list[Path] | None = None,
        source_name: str | None = None,
    ) -> dict[str, int]:
        """Build the dependency graph from source files and documents.

        Scans source files for import relationships and links files to their
        documents. Supports cross-source dependency detection via extra_source_paths.

        When ``source_name`` is provided AND the cross-source SCIP graph
        is loaded for that source, atomically includes SCIP-derived
        ``scip_calls`` edges (Phase 5). Backwards compat: omitting
        ``source_name`` preserves pre-Phase-5 behavior (ast-grep imports
        only).

        Args:
            source_path: Primary source root.
            extra_source_paths: Additional source roots for cross-source imports.
            source_name: Optional source name (matches ariadne.yaml key)
                that enables SCIP-precise call-edge enrichment.

        Returns:
            Dict with edge counts by type.
        """
        import ast

        from docgen.staleness import find_python_files

        # Collect files from all source paths for cross-source import resolution
        all_source_roots = [source_path] + (extra_source_paths or [])
        all_files: list[Path] = []
        module_to_path: dict[str, str] = {}

        for sp in all_source_roots:
            if not sp.exists():
                continue
            for f in find_python_files(sp):
                all_files.append(f)
                for base in (sp, sp.parent):
                    try:
                        rel = f.relative_to(base)
                        module = str(rel).replace('/', '.').replace('.py', '').replace('.__init__', '')
                        module_to_path[module] = str(f)
                    except ValueError:
                        continue

        edges: list[tuple[str, str, str, float]] = []

        # 1. Import edges (File→File)
        for f in all_files:
            try:
                tree = ast.parse(f.read_text(encoding='utf-8'), filename=str(f))
            except (SyntaxError, UnicodeDecodeError):
                continue
            f_str = str(f)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = module_to_path.get(alias.name)
                        if target and target != f_str:
                            edges.append((f_str, target, 'imports', 1.0))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for candidate in (node.module, node.module.rsplit('.', 1)[0] if '.' in node.module else None):
                        if candidate:
                            target = module_to_path.get(candidate)
                            if target and target != f_str:
                                edges.append((f_str, target, 'imports', 1.0))
                                break

        # 2. File→Doc edges (from source_files — lite, only need metadata)
        docs = self.list_documents_lite()
        for doc in docs:
            is_topic = doc.metadata.get('topic', False)
            edge_type = 'topic_member' if is_topic else 'documents'
            weight = 0.8 if is_topic else 1.0
            for sf in doc.source_files:
                edges.append((sf, doc.id, edge_type, weight))

        # 3. Write to database (transaction ensures atomic replace)
        with self._conn_provider.acquire() as conn:
            conn.execute('BEGIN')
            try:
                conn.execute('DELETE FROM doc_graph')
                conn.executemany(
                    'INSERT OR REPLACE INTO doc_graph (source_id, target_id, edge_type, weight) '
                    'VALUES (?, ?, ?, ?)',
                    edges,
                )
                conn.execute('COMMIT')
            except Exception:
                conn.execute('ROLLBACK')
                raise

        counts: dict[str, int] = {}
        for _, _, etype, _ in edges:
            counts[etype] = counts.get(etype, 0) + 1

        # Phase 5: enrich with SCIP-derived call edges when source_name
        # was passed and the cross-source SCIP graph is loaded for it.
        if source_name is not None:
            scip_count = self.enrich_doc_graph_with_scip(
                source_name, source_path,
            )
            if scip_count:
                counts['scip_calls'] = scip_count

        return counts

    def enrich_doc_graph_with_scip(
        self, source_name: str, source_root: Path,
    ) -> int:
        """Add SCIP-derived call edges to ``doc_graph`` for
        ``source_name``. Edges are at file granularity
        (caller_file → callee_file) with edge_type ``'scip_calls'`` to
        coexist alongside the existing ``'imports'`` and other types.

        Returns the count of edges inserted. Self-edges (caller and
        callee in the same file) are skipped — the doc graph operates
        at file granularity, so a file referencing itself adds noise
        without information.

        Returns 0 when ``source_name`` has no SCIP data loaded
        (consistent with decision #4: SCIP-only, no fallback).
        """
        from docgen.scip_cross_source import CrossSourceGraph

        graph = CrossSourceGraph()
        with self._conn_provider.acquire() as conn:
            graph.load_from(conn)

        if not graph.has_scip(source_name):
            return 0

        edges_to_add: list[tuple[str, str, str, float]] = []
        for edge in graph.edges_in_source(source_name):
            caller_file = str(source_root / edge.caller.file)
            callee_file = str(source_root / edge.callee.file)
            if caller_file == callee_file:
                continue
            edges_to_add.append(
                (caller_file, callee_file, 'scip_calls', 1.0),
            )

        if not edges_to_add:
            return 0

        with self._conn_provider.acquire() as conn:
            conn.executemany(
                'INSERT OR IGNORE INTO doc_graph '
                '(source_id, target_id, edge_type, weight) '
                'VALUES (?, ?, ?, ?)',
                edges_to_add,
            )
        return len(edges_to_add)

    def update_graph_for_files(self, changed_files: list[Path], source_path: Path) -> dict[str, int]:
        """Incrementally update the graph for specific changed files.

        Only re-parses the changed files and updates their edges, instead of
        rebuilding the entire graph. Much faster for single-file changes.

        Args:
            changed_files: List of files that changed.
            source_path: Source root for module name resolution.

        Returns:
            Dict with edge counts that were updated.
        """
        import ast

        # Build module name index for import resolution
        module_to_path: dict[str, str] = {}
        with self._conn_provider.acquire() as conn:
            # Get all existing file nodes from the graph
            for row in conn.execute(
                "SELECT DISTINCT source_id FROM doc_graph WHERE edge_type = 'imports'"
            ).fetchall():
                path = row[0]
                for base in (source_path, source_path.parent):
                    try:
                        rel = Path(path).relative_to(base)
                        module = str(rel).replace('/', '.').replace('.py', '').replace('.__init__', '')
                        module_to_path[module] = path
                    except ValueError:
                        continue

        updated_edges: list[tuple[str, str, str, float]] = []

        for f in changed_files:
            if not f.exists() or f.suffix != '.py':
                continue
            f_str = str(f)

            # Remove old edges for this file
            with self._conn_provider.acquire() as conn:
                conn.execute("DELETE FROM doc_graph WHERE source_id = ? AND edge_type = 'imports'", (f_str,))

            # Parse new imports
            try:
                tree = ast.parse(f.read_text(encoding='utf-8'), filename=f_str)
            except (SyntaxError, UnicodeDecodeError):
                continue

            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        target = module_to_path.get(alias.name)
                        if target and target != f_str:
                            updated_edges.append((f_str, target, 'imports', 1.0))
                elif isinstance(node, ast.ImportFrom) and node.module:
                    for candidate in (node.module, node.module.rsplit('.', 1)[0] if '.' in node.module else None):
                        if candidate:
                            target = module_to_path.get(candidate)
                            if target and target != f_str:
                                updated_edges.append((f_str, target, 'imports', 1.0))
                                break

        # Insert updated edges
        if updated_edges:
            with self._conn_provider.acquire() as conn:
                conn.executemany(
                    'INSERT OR REPLACE INTO doc_graph (source_id, target_id, edge_type, weight) '
                    'VALUES (?, ?, ?, ?)',
                    updated_edges,
                )

        return {'imports_updated': len(updated_edges), 'files_processed': len(changed_files)}

    def get_graph_stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        with self._conn_provider.acquire() as conn:
            total = conn.execute('SELECT COUNT(*) FROM doc_graph').fetchone()[0]
            by_type = dict(conn.execute(
                'SELECT edge_type, COUNT(*) FROM doc_graph GROUP BY edge_type'
            ).fetchall())
            sources = conn.execute('SELECT COUNT(DISTINCT source_id) FROM doc_graph').fetchone()[0]
            targets = conn.execute('SELECT COUNT(DISTINCT target_id) FROM doc_graph').fetchone()[0]
        return {'total_edges': total, 'by_type': by_type, 'source_nodes': sources, 'target_nodes': targets}

    def get_priorities(self, source_path: Path) -> list[dict[str, Any]]:
        """Get files ranked by priority score (edges x missing doc coverage)."""
        from docgen.staleness import find_python_files

        all_files = {str(f) for f in find_python_files(source_path) if f.name != '__init__.py'}

        with self._conn_provider.acquire() as conn:
            inbound: dict[str, int] = dict(conn.execute(
                "SELECT target_id, COUNT(*) FROM doc_graph WHERE edge_type='imports' GROUP BY target_id"
            ).fetchall())
            outbound: dict[str, int] = dict(conn.execute(
                "SELECT source_id, COUNT(*) FROM doc_graph WHERE edge_type='imports' GROUP BY source_id"
            ).fetchall())
            doc_count: dict[str, int] = dict(conn.execute(
                "SELECT source_id, COUNT(*) FROM doc_graph "
                "WHERE edge_type IN ('documents','topic_member') GROUP BY source_id"
            ).fetchall())

        results = []
        for f in all_files:
            ib, ob = inbound.get(f, 0), outbound.get(f, 0)
            total = ib + ob
            docs = doc_count.get(f, 0)
            coverage = min(docs / 2.0, 1.0)
            priority = total * (1.0 - coverage)
            try:
                rel = str(Path(f).relative_to(source_path))
            except ValueError:
                rel = f
            results.append({
                'file': rel, 'inbound_edges': ib, 'outbound_edges': ob,
                'total_edges': total, 'doc_count': docs,
                'coverage_percent': coverage * 100, 'priority_score': priority,
            })
        results.sort(key=lambda r: r['priority_score'], reverse=True)
        return results

    def get_related(self, doc_id: str, max_hops: int = 2, limit: int = 10) -> list[dict[str, Any]]:
        """Get related documents ranked by graph distance."""
        from collections import deque

        with self._conn_provider.acquire() as conn:
            visited: dict[str, float] = {doc_id: 0}
            queue: deque[tuple[str, float, int]] = deque([(doc_id, 0, 0)])

            while queue:
                node, dist, hops = queue.popleft()
                if hops >= max_hops:
                    continue
                for row in conn.execute(
                    'SELECT target_id, weight FROM doc_graph WHERE source_id = ? '
                    'UNION '
                    'SELECT source_id, weight FROM doc_graph WHERE target_id = ?',
                    (node, node),
                ).fetchall():
                    neighbor, weight = row[0], row[1]
                    new_dist = dist + (1.0 / weight if weight > 0 else 10.0)
                    if neighbor not in visited or new_dist < visited[neighbor]:
                        visited[neighbor] = new_dist
                        queue.append((neighbor, new_dist, hops + 1))

            visited.pop(doc_id, None)
            results = []
            for node_id, distance in sorted(visited.items(), key=lambda x: x[1]):
                row = conn.execute(
                    'SELECT id, title, content_type FROM documents WHERE id = ?', (node_id,)
                ).fetchone()
                if row:
                    results.append({'id': row[0], 'title': row[1], 'content_type': row[2], 'distance': distance})
                    if len(results) >= limit:
                        break
        return results

    def get_related_batch(
        self, doc_ids, max_hops: int = 2, limit: int = 10,
    ) -> dict[str, list[dict[str, Any]]]:
        """Batch equivalent of :meth:`get_related` for many seeds.

        Loads the whole ``doc_graph`` and the ``id -> (title, content_type)``
        map ONCE, then walks each seed in memory — replacing the per-seed,
        per-node query storm (``O(seeds x visited)`` round-trips) with two
        bulk reads. Output is identical to calling ``get_related`` for each
        id: same neighbours, distances, ordering, and ``limit`` cut. The
        equivalence is pinned by ``tests/test_get_related_batch.py``.
        """
        from collections import defaultdict

        adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
        seen: dict[str, set[tuple[str, float]]] = defaultdict(set)
        with self._conn_provider.acquire() as conn:
            for source_id, target_id, weight in conn.execute(
                'SELECT source_id, target_id, weight FROM doc_graph'
            ).fetchall():
                # Undirected like get_related's source/target UNION; dedupe on
                # (neighbour, weight) so parallel edges of differing weight survive.
                for node, neighbor in (
                    (source_id, target_id), (target_id, source_id),
                ):
                    pair = (neighbor, weight)
                    if pair not in seen[node]:
                        seen[node].add(pair)
                        adjacency[node].append(pair)
            doc_meta = {
                row[0]: (row[1], row[2])
                for row in conn.execute(
                    'SELECT id, title, content_type FROM documents'
                ).fetchall()
            }
        # Match SQLite's UNION row order (by neighbour id, then weight) so the
        # distance-tie ordering at the ``limit`` cut is identical to get_related.
        for neighbors in adjacency.values():
            neighbors.sort()

        return {
            doc_id: self._related_from_adjacency(
                doc_id, adjacency, doc_meta, max_hops, limit,
            )
            for doc_id in doc_ids
        }

    @staticmethod
    def _related_from_adjacency(
        seed: str,
        adjacency: dict[str, list[tuple[str, float]]],
        doc_meta: dict[str, tuple[str, str]],
        max_hops: int,
        limit: int,
    ) -> list[dict[str, Any]]:
        """In-memory twin of get_related's BFS over a prebuilt adjacency.

        Same relax-and-requeue walk, hop bound, and hydrate-only-documents
        + ``limit`` semantics — just no per-node database round-trips.
        """
        from collections import deque

        visited: dict[str, float] = {seed: 0}
        queue: deque[tuple[str, float, int]] = deque([(seed, 0, 0)])
        while queue:
            node, dist, hops = queue.popleft()
            if hops >= max_hops:
                continue
            for neighbor, weight in adjacency.get(node, ()):
                new_dist = dist + (1.0 / weight if weight > 0 else 10.0)
                if neighbor not in visited or new_dist < visited[neighbor]:
                    visited[neighbor] = new_dist
                    queue.append((neighbor, new_dist, hops + 1))

        visited.pop(seed, None)
        results: list[dict[str, Any]] = []
        for node_id, distance in sorted(visited.items(), key=lambda x: x[1]):
            meta = doc_meta.get(node_id)
            if meta is not None:
                results.append({
                    'id': node_id, 'title': meta[0],
                    'content_type': meta[1], 'distance': distance,
                })
                if len(results) >= limit:
                    break
        return results

    def export_graph_json(self) -> dict[str, Any]:
        """Export the graph as JSON for D3.js visualization."""
        import json as json_mod

        nodes: dict[str, dict[str, Any]] = {}
        edges: list[dict[str, Any]] = []

        with self._conn_provider.acquire() as conn:
            for row in conn.execute('SELECT source_id, target_id, edge_type, weight FROM doc_graph').fetchall():
                src, tgt, etype, weight = row
                edges.append({'source': src, 'target': tgt, 'type': etype, 'weight': weight})
                for nid in (src, tgt):
                    if nid not in nodes:
                        nodes[nid] = {'id': nid, 'type': 'file', 'title': nid.split('/')[-1], 'doc_count': 0}

            for row in conn.execute('SELECT id, title, content_type, source_files, metadata FROM documents').fetchall():
                doc_id, title, ctype, sf_json, meta_json = row
                meta = json_mod.loads(meta_json) if meta_json else {}
                if doc_id in nodes:
                    nodes[doc_id].update({'type': 'topic' if meta.get('topic') else ctype, 'title': title})
                source_files = json_mod.loads(sf_json) if sf_json else []
                for sf in source_files:
                    if sf in nodes:
                        nodes[sf]['doc_count'] = nodes[sf].get('doc_count', 0) + 1

        return {'nodes': list(nodes.values()), 'edges': edges}
