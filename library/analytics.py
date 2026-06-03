"""Usage analytics, quality scoring, gap analysis, and metrics."""
from __future__ import annotations

import json
import logging
import re
from datetime import UTC
from typing import Any

_logger = logging.getLogger(__name__)


def _build_ascii_histogram(
    score_dist: dict[int, int], median: int, avg: float, total: int,
) -> str:
    """Build an ASCII histogram of quality scores 1-10."""
    if total == 0:
        return 'No quality scores recorded yet.'
    max_count = max(score_dist.values()) if score_dist else 1
    lines = [f'Quality Score Distribution (median: {median}, avg: {avg:.1f}, n={total})']
    for score in range(1, 11):
        count = score_dist.get(score, 0)
        bar_len = round(count / max_count * 10) if max_count > 0 else 0
        bar = '\u2588' * bar_len
        marker = '  \u2190 median' if score == median else ''
        lines.append(f' {score:2d} \u2595{bar:<10s} ({count}){marker}')
    return '\n'.join(lines)


class AnalyticsMixin:
    """Usage tracking, quality scores, gap analysis, trends, and ROI.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    - self.get_documents_batch() from CoreMixin
    - self.count_documents() from QualityMixin
    """

    # ------------------------------------------------------------------
    # Usage tracking
    # ------------------------------------------------------------------

    def log_usage(
        self,
        tool_name: str,
        query: str | None,
        result_count: int,
        document_ids: list[str] | None = None,
    ) -> int:
        """Log a tool usage event.

        Args:
            tool_name: Name of the MCP tool that was called.
            query: The search query or parameters.
            result_count: Number of results returned.
            document_ids: IDs of documents returned (for per-doc analytics).

        Returns:
            The auto-generated event ID.
        """
        import json as json_mod

        from schema import _now_iso

        ts = _now_iso()
        doc_ids_json = json_mod.dumps(document_ids) if document_ids else None
        # Outcome semantics:
        #   'hit'  — auto-marked useful when results came back (overridden
        #            later by ariadne_log_miss if the agent reports the
        #            results weren't useful)
        #   'call' — neutral "awaiting-feedback" sentinel for zero-result
        #            calls. Matches the schema DEFAULT in library.py:152
        #            (``outcome TEXT NOT NULL DEFAULT 'call'``). Not 'miss'
        #            — 'miss' is a user-driven label from ariadne_log_miss;
        #            conflating zero-result with explicit miss blurs the
        #            gap-analysis signal.
        outcome = 'hit' if result_count > 0 else 'call'
        with self._conn_provider.acquire() as conn:
            cursor = conn.execute(
                'INSERT INTO usage_events '
                '(timestamp, tool_name, query, result_count, returned_document_ids, outcome) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (ts, tool_name, query, result_count, doc_ids_json, outcome),
            )
            return cursor.lastrowid  # type: ignore[return-value]

    def _mark_outcome(
        self, event_id: int, outcome: str, feedback: str | None = None,
    ) -> bool:
        """Update a usage event's outcome, optional feedback, and quality score.

        Parses 'score:N' from feedback string (e.g., "score:7 — found the pattern").
        Returns True if the event was found and updated.
        """
        import re as _re

        quality_score = None
        if feedback:
            match = _re.search(r'score:\s*(\d+)', feedback)
            if match:
                quality_score = max(1, min(10, int(match.group(1))))

        with self._conn_provider.acquire() as conn:
            cursor = conn.execute(
                'UPDATE usage_events SET outcome = ?, feedback = ?, quality_score = ? WHERE id = ?',
                (outcome, feedback, quality_score, event_id),
            )
            return cursor.rowcount > 0


    def get_doc_hit_stats(self, doc_id: str) -> dict:
        """Get served/hit counts for a document across all usage events."""
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT outcome, quality_score FROM usage_events '
                'WHERE returned_document_ids LIKE ?',
                (f'%"{doc_id}"%',),
            ).fetchall()
        served = len(rows)
        hits = sum(1 for r in rows if r['outcome'] == 'hit' and r['quality_score'] is not None and r['quality_score'] >= 5)
        return {'served': served, 'hits': hits}

    def mark_hit(self, event_id: int, feedback: str | None = None) -> bool:
        """Mark a usage event as a hit (useful result)."""
        return self._mark_outcome(event_id, 'hit', feedback)

    def mark_miss(self, event_id: int, feedback: str) -> bool:
        """Mark a usage event as a miss (not useful)."""
        return self._mark_outcome(event_id, 'miss', feedback)

    def get_usage_stats(
        self,
        days: int | None = None,
        tool_name: str | None = None,
    ) -> dict[str, Any]:
        """Get usage statistics.

        Args:
            days: Limit to the last N days (None = all time).
            tool_name: Filter to a specific tool.

        Returns:
            Dict with total_calls, total_hits, total_misses, hit_rate,
            by_tool, by_day, avg_calls_per_day, recent_feedback.
        """
        from datetime import datetime, timedelta

        where_clauses: list[str] = []
        params: list[str | int] = []

        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            where_clauses.append('timestamp >= ?')
            params.append(cutoff)
        if tool_name is not None:
            where_clauses.append('tool_name = ?')
            params.append(tool_name)

        where_sql = ('WHERE ' + ' AND '.join(where_clauses)) if where_clauses else ''

        with self._conn_provider.acquire() as conn:
            # Totals
            row = conn.execute(
                f'SELECT COUNT(*), '
                f"SUM(CASE WHEN outcome = 'hit' THEN 1 ELSE 0 END), "
                f"SUM(CASE WHEN outcome = 'miss' THEN 1 ELSE 0 END) "
                f'FROM usage_events {where_sql}',
                params,
            ).fetchone()
            total_calls = row[0] or 0
            total_hits = row[1] or 0
            total_misses = row[2] or 0

            # By tool
            by_tool: dict[str, dict[str, int | float]] = {}
            tool_params = [p for p in params if p != tool_name] if tool_name else list(params)
            tool_where_clauses = [c for c in where_clauses if 'tool_name' not in c]
            tool_where_sql = ('WHERE ' + ' AND '.join(tool_where_clauses)) if tool_where_clauses else ''
            for trow in conn.execute(
                f'SELECT tool_name, COUNT(*), '
                f"SUM(CASE WHEN outcome = 'hit' THEN 1 ELSE 0 END), "
                f"SUM(CASE WHEN outcome = 'miss' THEN 1 ELSE 0 END) "
                f'FROM usage_events {tool_where_sql} GROUP BY tool_name',
                tool_params,
            ).fetchall():
                t_calls = trow[1] or 0
                t_hits = trow[2] or 0
                t_misses = trow[3] or 0
                by_tool[trow[0]] = {
                    'calls': t_calls,
                    'hits': t_hits,
                    'misses': t_misses,
                    'hit_rate': t_hits / t_calls if t_calls > 0 else 0.0,
                }

            # By day (last 90 days max)
            by_day: list[dict[str, int | str]] = []
            for drow in conn.execute(
                f'SELECT DATE(timestamp) as day, COUNT(*), '
                f"SUM(CASE WHEN outcome = 'hit' THEN 1 ELSE 0 END), "
                f"SUM(CASE WHEN outcome = 'miss' THEN 1 ELSE 0 END) "
                f'FROM usage_events {where_sql} '
                f'GROUP BY DATE(timestamp) ORDER BY day DESC LIMIT 90',
                params,
            ).fetchall():
                by_day.append({
                    'date': drow[0],
                    'calls': drow[1] or 0,
                    'hits': drow[2] or 0,
                    'misses': drow[3] or 0,
                })
            by_day.reverse()

            # Avg calls per day
            num_days = len(by_day) or 1
            avg_calls_per_day = total_calls / num_days

            # Recent feedback
            fb_where_parts = list(where_clauses) + ['feedback IS NOT NULL']
            fb_where_sql = 'WHERE ' + ' AND '.join(fb_where_parts)
            recent_feedback: list[dict[str, str]] = []
            for frow in conn.execute(
                f'SELECT timestamp, tool_name, query, outcome, feedback '
                f'FROM usage_events {fb_where_sql} '
                f'ORDER BY timestamp DESC LIMIT 20',
                params,
            ).fetchall():
                recent_feedback.append({
                    'timestamp': frow[0],
                    'tool_name': frow[1],
                    'query': frow[2] or '',
                    'outcome': frow[3],
                    'feedback': frow[4],
                })

        # Quality score stats
        with self._conn_provider.acquire() as conn:
            quality_where = where_sql + ' AND quality_score IS NOT NULL' if where_sql else 'WHERE quality_score IS NOT NULL'
            scores_rows = conn.execute(
                f'SELECT quality_score FROM usage_events {quality_where}',
                params,
            ).fetchall()
        scores = sorted([r[0] for r in scores_rows])

        avg_quality = sum(scores) / len(scores) if scores else 0.0
        median_quality = scores[len(scores) // 2] if scores else 0
        score_dist = {i: scores.count(i) for i in range(1, 11)}
        histogram = _build_ascii_histogram(score_dist, median_quality, avg_quality, len(scores))

        return {
            'total_calls': total_calls,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'hit_rate': total_hits / total_calls if total_calls > 0 else 0.0,
            'avg_quality_score': round(avg_quality, 1),
            'median_quality_score': median_quality,
            'scored_count': len(scores),
            'score_distribution': score_dist,
            'quality_histogram': histogram,
            'by_tool': by_tool,
            'by_day': by_day,
            'avg_calls_per_day': avg_calls_per_day,
            'recent_feedback': recent_feedback,
        }

    def get_usage_event(self, event_id: int) -> dict[str, Any] | None:
        """Fetch a single usage event by ID."""
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                'SELECT id, timestamp, tool_name, query, result_count, outcome, '
                'feedback, returned_document_ids, quality_score '
                'FROM usage_events WHERE id = ?',
                (event_id,),
            ).fetchone()
        if not row:
            return None
        return {
            'id': row[0], 'timestamp': row[1], 'tool_name': row[2],
            'query': row[3], 'result_count': row[4], 'outcome': row[5],
            'feedback': row[6],
            'returned_document_ids': json.loads(row[7]) if row[7] else [],
            'quality_score': row[8],
        }

    def get_low_score_events(
        self, max_score: int = 5, limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent low-score events for improvement processing.

        Args:
            max_score: Maximum score to include (default: 5).
            limit: Maximum events to return.

        Returns:
            List of usage event dicts with actionable low scores.
        """
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT id, timestamp, tool_name, query, result_count, outcome, '
                'feedback, returned_document_ids, quality_score '
                'FROM usage_events '
                'WHERE quality_score IS NOT NULL AND quality_score <= ? '
                'ORDER BY timestamp DESC LIMIT ?',
                (max_score, limit),
            ).fetchall()
        events = []
        for row in rows:
            feedback = row[6] or ''
            reason = self._extract_reason(feedback)
            if reason and self._is_specific_reason(reason):
                events.append({
                    'id': row[0], 'timestamp': row[1], 'tool_name': row[2],
                    'query': row[3], 'result_count': row[4], 'outcome': row[5],
                    'feedback': feedback,
                    'returned_document_ids': json.loads(row[7]) if row[7] else [],
                    'quality_score': row[8],
                    'reason': reason,
                })
        return events

    def diagnose_low_score(self, event_id: int) -> dict[str, Any]:
        """Analyze a low-score usage event and identify documentation gaps.

        Parses the reason from feedback, checks if it's specific enough to act on,
        and identifies which source files and documents are involved.

        Returns:
            Dict with event, reason, is_actionable, source_files, recommendation.
        """
        event = self.get_usage_event(event_id)
        if not event:
            return {'error': f'Event {event_id} not found', 'is_actionable': False}

        score = event.get('quality_score')
        if score is None or score > 5:
            return {
                'event': event, 'is_actionable': False,
                'recommendation': 'Score is not low enough to warrant improvement.',
            }

        feedback = event.get('feedback', '') or ''
        reason = self._extract_reason(feedback)

        if not reason or not self._is_specific_reason(reason):
            return {
                'event': event, 'reason': reason or '(none)', 'is_actionable': False,
                'recommendation': 'Reason is too vague to act on.',
            }

        # Find documents that were returned for this query
        doc_ids = event.get('returned_document_ids', [])
        docs = self.get_documents_batch(doc_ids) if doc_ids else []

        # Collect source files covered by returned docs
        source_files: set[str] = set()
        for doc in docs:
            source_files.update(doc.source_files)

        return {
            'event': event,
            'reason': reason,
            'is_actionable': True,
            'score': score,
            'query': event.get('query', ''),
            'returned_docs': [{'id': d.id, 'title': d.title} for d in docs],
            'source_files': sorted(source_files),
            'recommendation': f'Regenerate docs targeting gap: {reason}',
        }

    @staticmethod
    def _extract_reason(feedback: str) -> str:
        """Extract the reason from feedback like 'score:4 — reason here'."""
        match = re.search(r'score:\s*\d+\s*[—–\-]\s*(.*)', feedback)
        return match.group(1).strip() if match else ''

    @staticmethod
    def _is_specific_reason(reason: str) -> bool:
        """Check if a reason is specific enough to act on.

        Rejects short or vague reasons like "not helpful" or "bad results".
        """
        if len(reason) < 15:
            return False
        vague = ('not helpful', 'bad results', 'not useful', "didn't help",
                 'wrong', 'useless', 'no good', 'bad', 'poor')
        reason_lower = reason.lower().strip('.!?')
        return reason_lower not in vague

    def get_gap_report(self, days: int | None = None) -> dict[str, Any]:
        """Get a report of documentation gaps based on miss feedback.

        Args:
            days: Limit to the last N days (None = all time).

        Returns:
            Dict with total_misses, miss_rate, top_gaps (grouped by feedback),
            and recent_misses.
        """
        from datetime import datetime, timedelta

        where_parts: list[str] = ["outcome = 'miss'"]
        params: list[str] = []

        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            where_parts.append('timestamp >= ?')
            params.append(cutoff)

        where_sql = 'WHERE ' + ' AND '.join(where_parts)

        with self._conn_provider.acquire() as conn:
            # Total events for miss rate calculation
            total_row = conn.execute(
                'SELECT COUNT(*) FROM usage_events'
                + (' WHERE timestamp >= ?' if days else ''),
                params[:1] if days else [],
            ).fetchone()
            total_events = total_row[0] or 0

            # Miss count
            miss_row = conn.execute(
                f'SELECT COUNT(*) FROM usage_events {where_sql}',
                params,
            ).fetchone()
            total_misses = miss_row[0] or 0

            # Top gaps grouped by feedback
            gaps_where = where_sql + ' AND feedback IS NOT NULL'
            top_gaps: list[dict[str, Any]] = []
            for grow in conn.execute(
                f'SELECT feedback, COUNT(*) as cnt, '
                f'GROUP_CONCAT(query, " | ") as queries, '
                f'MAX(timestamp) as last_seen '
                f'FROM usage_events {gaps_where} '
                f'GROUP BY feedback ORDER BY cnt DESC LIMIT 20',
                params,
            ).fetchall():
                queries = grow[2] or ''
                top_gaps.append({
                    'feedback': grow[0],
                    'count': grow[1],
                    'example_queries': [q.strip() for q in queries.split(' | ') if q.strip()][:5],
                    'last_seen': grow[3],
                })

            # Recent misses
            recent_misses: list[dict[str, str]] = []
            for mrow in conn.execute(
                f'SELECT timestamp, tool_name, query, feedback '
                f'FROM usage_events {where_sql} '
                f'ORDER BY timestamp DESC LIMIT 20',
                params,
            ).fetchall():
                recent_misses.append({
                    'timestamp': mrow[0],
                    'tool_name': mrow[1],
                    'query': mrow[2] or '',
                    'feedback': mrow[3] or '',
                })

        return {
            'total_misses': total_misses,
            'miss_rate': total_misses / total_events if total_events > 0 else 0.0,
            'top_gaps': top_gaps,
            'recent_misses': recent_misses,
        }

    def source_signature(self, source_name: str) -> str:
        """Cheap stale-detection signature for a source.

        Returns a string formatted as ``"{count}|{max_updated_at}"``.
        Changes whenever a document is added, removed, or modified for
        the given source. Used by the ``ariadne status`` CLI as a cache
        key — if the signature matches the cached value, the
        per-source stats can be served from cache, skipping the
        expensive chunks/sections JOIN.

        Edge case: the signature only watches ``documents``. If chunks
        are added or removed for an existing doc WITHOUT touching the
        doc's ``updated_at``, the signature wouldn't notice. The
        producer's writer always touches ``updated_at`` when modifying
        chunks, so this is a non-issue in practice. Escape hatch when
        debugging: delete the cache file alongside ``ariadne.db``.
        """
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(MAX(updated_at), '') "
                'FROM documents '
                "WHERE COALESCE(source_name, 'unknown') = ?",
                (source_name,),
            ).fetchone()
        return f'{row[0]}|{row[1]}'

    def list_source_names(self) -> list[str]:
        """Return distinct source names in the library, sorted.

        Documents with NULL ``source_name`` bucket as 'unknown' to
        match the COALESCE convention used by ``stats_by_source*``.
        Used by the ``ariadne status`` CLI to size its progress bar
        before iterating per-source attribution.
        """
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                "SELECT DISTINCT COALESCE(source_name, 'unknown') AS src "
                'FROM documents ORDER BY src',
            ).fetchall()
        return [row[0] for row in rows]

    def stats_for_source(self, source_name: str) -> dict[str, Any]:
        """Per-source detailed stats — scoped variant of
        ``stats_by_source_detailed``.

        Returns the same shape as one entry of the batch method, but
        runs a focused per-source query — much faster on large DBs
        where the batch method's full chunks-table JOIN is the
        bottleneck. Useful when iterating sources with a progress bar
        (the ``ariadne status`` CLI does this).

        An unknown ``source_name`` returns the zero-shape dict rather
        than ``None`` so callers iterate uniformly without special
        casing.
        """
        result: dict[str, Any] = {
            'doc_count': 0,
            'doc_content': 0,
            'doc_embed': 0,
            'by_content_type': {},
            'chunk_count': 0,
            'chunk_content': 0,
            'chunk_embed': 0,
            'section_count': 0,
            'section_content': 0,
            'section_embed': 0,
        }

        with self._conn_provider.acquire() as conn:
            # Doc totals for this source.
            row = conn.execute(
                'SELECT COUNT(*), '
                '  COALESCE(SUM(LENGTH(content)), 0), '
                '  COALESCE(SUM(LENGTH(embedding)), 0) '
                'FROM documents '
                "WHERE COALESCE(source_name, 'unknown') = ?",
                (source_name,),
            ).fetchone()
            result['doc_count'] = row[0]
            result['doc_content'] = row[1]
            result['doc_embed'] = row[2]

            # Content-type breakdown for this source.
            for row in conn.execute(
                'SELECT content_type, COUNT(*) '
                'FROM documents '
                "WHERE COALESCE(source_name, 'unknown') = ? "
                'GROUP BY content_type',
                (source_name,),
            ).fetchall():
                result['by_content_type'][row[0]] = row[1]

            # Chunk attribution scoped to this source via JOIN.
            row = conn.execute(
                'SELECT COUNT(*), '
                '  COALESCE(SUM(LENGTH(c.content)), 0), '
                '  COALESCE(SUM(LENGTH(c.embedding)), 0) '
                'FROM chunks c JOIN documents d ON c.document_id = d.id '
                "WHERE COALESCE(d.source_name, 'unknown') = ?",
                (source_name,),
            ).fetchone()
            result['chunk_count'] = row[0]
            result['chunk_content'] = row[1]
            result['chunk_embed'] = row[2]

            # Section attribution scoped to this source via JOIN.
            row = conn.execute(
                'SELECT COUNT(*), '
                '  COALESCE(SUM(LENGTH(s.content)), 0), '
                '  COALESCE(SUM(LENGTH(s.embedding)), 0) '
                'FROM sections s JOIN documents d ON s.document_id = d.id '
                "WHERE COALESCE(d.source_name, 'unknown') = ?",
                (source_name,),
            ).fetchone()
            result['section_count'] = row[0]
            result['section_content'] = row[1]
            result['section_embed'] = row[2]

        return result

    def stats_by_source_detailed(self) -> dict[str, Any]:
        """Per-source breakdown enriched with content-type counts and
        bytes attributed across documents, chunks, and sections.

        Powers the ``ariadne status`` CLI command and any tooling that
        wants "where does the disk go" attribution. The existing
        ``stats_by_source`` returns per-source doc totals only; this
        method adds:

          - ``by_content_type``: ``{content_type: count}`` per source
          - ``chunk_content`` / ``chunk_embed``: byte sums attributed to
            the source via ``JOIN documents ON chunks.document_id``
          - ``section_content`` / ``section_embed``: same for sections

        The footer ``_total`` carries the DB file size on disk so
        callers can compute overhead = db_size - sum(per-source attribution).

        Returns:
            Dict keyed by source name, plus ``_total``. Each per-source
            entry has the keys: ``doc_count``, ``doc_content``,
            ``doc_embed``, ``by_content_type``, ``chunk_count``,
            ``chunk_content``, ``chunk_embed``, ``section_count``,
            ``section_content``, ``section_embed``.
        """
        import os

        db_size = os.path.getsize(self.path) if self.path.exists() else 0
        result: dict[str, Any] = {}

        with self._conn_provider.acquire() as conn:
            # Per-source doc totals (count, content bytes, embed bytes).
            for row in conn.execute(
                "SELECT COALESCE(source_name, 'unknown') AS src, "
                '  COUNT(*) AS doc_count, '
                '  COALESCE(SUM(LENGTH(content)), 0) AS doc_content, '
                '  COALESCE(SUM(LENGTH(embedding)), 0) AS doc_embed '
                'FROM documents GROUP BY src',
            ).fetchall():
                result[row[0]] = {
                    'doc_count': row[1],
                    'doc_content': row[2],
                    'doc_embed': row[3],
                    'by_content_type': {},
                    'chunk_count': 0,
                    'chunk_content': 0,
                    'chunk_embed': 0,
                    'section_count': 0,
                    'section_content': 0,
                    'section_embed': 0,
                }

            # Per-source content_type breakdown.
            for row in conn.execute(
                "SELECT COALESCE(source_name, 'unknown') AS src, "
                '  content_type, COUNT(*) '
                'FROM documents GROUP BY src, content_type',
            ).fetchall():
                src, ct, count = row
                if src in result:
                    result[src]['by_content_type'][ct] = count

            # Per-source chunk attribution via JOIN. The chunks table
            # itself doesn't carry source_name — it travels through
            # documents.id → chunks.document_id.
            for row in conn.execute(
                "SELECT COALESCE(d.source_name, 'unknown') AS src, "
                '  COUNT(*), '
                '  COALESCE(SUM(LENGTH(c.content)), 0), '
                '  COALESCE(SUM(LENGTH(c.embedding)), 0) '
                'FROM chunks c JOIN documents d ON c.document_id = d.id '
                'GROUP BY src',
            ).fetchall():
                src = row[0]
                if src in result:
                    result[src]['chunk_count'] = row[1]
                    result[src]['chunk_content'] = row[2]
                    result[src]['chunk_embed'] = row[3]

            # Per-source section attribution via the same JOIN pattern.
            for row in conn.execute(
                "SELECT COALESCE(d.source_name, 'unknown') AS src, "
                '  COUNT(*), '
                '  COALESCE(SUM(LENGTH(s.content)), 0), '
                '  COALESCE(SUM(LENGTH(s.embedding)), 0) '
                'FROM sections s JOIN documents d ON s.document_id = d.id '
                'GROUP BY src',
            ).fetchall():
                src = row[0]
                if src in result:
                    result[src]['section_count'] = row[1]
                    result[src]['section_content'] = row[2]
                    result[src]['section_embed'] = row[3]

        result['_total'] = {'db_size_bytes': db_size}
        return result

    def stats_by_source(self) -> dict[str, dict[str, int]]:
        """Get document count and content size grouped by source.

        Uses source_name column (set during generation). Falls back to 'unknown'.

        Returns:
            Dict keyed by source name with doc_count, content_size, chunk_count.
        """
        import os

        result: dict[str, dict[str, int]] = {}
        db_size = os.path.getsize(self.path) if self.path.exists() else 0

        with self._conn_provider.acquire() as conn:
            # Aggregate docs by source_name in a single query
            for row in conn.execute(
                "SELECT COALESCE(source_name, 'unknown') AS src, "
                '  COUNT(*) AS doc_count, '
                '  SUM(LENGTH(content)) AS content_size, '
                '  SUM(LENGTH(embedding)) AS embedding_size '
                'FROM documents GROUP BY src'
            ).fetchall():
                result[row[0]] = {
                    'doc_count': row[1],
                    'content_size': row[2] or 0,
                    'embedding_size': row[3] or 0,
                }

            # Count chunks per source in a single query
            for row in conn.execute(
                "SELECT COALESCE(d.source_name, 'unknown') AS src, COUNT(*) "
                'FROM chunks c JOIN documents d ON c.document_id = d.id '
                'GROUP BY src'
            ).fetchall():
                if row[0] in result:
                    result[row[0]]['chunk_count'] = row[1]

        # Ensure chunk_count exists for all sources
        for stats in result.values():
            stats.setdefault('chunk_count', 0)

        result['_total'] = {'db_size_bytes': db_size}  # type: ignore[assignment]
        return result

    def usage_by_document(self, days: int | None = None, limit: int = 20) -> list[dict[str, Any]]:
        """Get per-document usage stats (how often each doc was served).

        Args:
            days: Limit to last N days (None = all time).
            limit: Max results to return.

        Returns:
            List of dicts with document_id, title, serve_count, sorted by serve_count desc.
        """
        import json as json_mod
        from collections import Counter
        from datetime import datetime, timedelta

        where_clause = ''
        params: list[str] = []
        if days is not None:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()
            where_clause = 'WHERE timestamp >= ?'
            params.append(cutoff)

        doc_counter: Counter[str] = Counter()

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'SELECT returned_document_ids FROM usage_events '
                f'{where_clause} AND returned_document_ids IS NOT NULL',
                params,
            ).fetchall()

            for (doc_ids_json,) in rows:
                if doc_ids_json:
                    for doc_id in json_mod.loads(doc_ids_json):
                        doc_counter[doc_id] += 1

            # Resolve titles in a single batch query
            top_items = doc_counter.most_common(limit)
            top_ids = [doc_id for doc_id, _ in top_items]
            title_map: dict[str, tuple[str, str]] = {}
            if top_ids:
                placeholders = ','.join('?' * len(top_ids))
                for row in conn.execute(
                    f'SELECT id, title, content_type FROM documents WHERE id IN ({placeholders})',
                    top_ids,
                ).fetchall():
                    title_map[row[0]] = (row[1], row[2])

            results: list[dict[str, Any]] = []
            for doc_id, count in top_items:
                title, ctype = title_map.get(doc_id, ('(deleted)', 'unknown'))
                results.append({
                    'document_id': doc_id,
                    'title': title,
                    'content_type': ctype,
                    'serve_count': count,
                })

        return results

    # ------------------------------------------------------------------
    # Near misses, trends, and ROI
    # ------------------------------------------------------------------

    def near_misses(self, days: int = 7) -> list[dict[str, Any]]:
        """Predict what docs developers will need next based on miss patterns.

        Analyzes recent miss feedback to identify emerging topics.
        Proactively suggests generating docs before more misses occur.

        Returns:
            List of predicted needs with topic, miss_count, and suggestion.
        """
        from collections import Counter

        gap_data = self.get_gap_report(days=days)
        miss_topics: Counter[str] = Counter()

        for gap in gap_data.get('top_gaps', []):
            # Extract key words from feedback
            feedback = gap.get('feedback', '')
            words = [w.lower() for w in feedback.split() if len(w) > 3]
            for w in words:
                if w not in ('searched', 'found', 'docs', 'documentation', 'about', 'looking', 'need', 'help'):
                    miss_topics[w] += gap.get('count', 1)

        predictions = []
        for topic, count in miss_topics.most_common(10):
            # Check if we already have docs for this topic
            existing = self._text_search(topic, limit=1)
            if not existing:
                predictions.append({
                    'topic': topic,
                    'miss_count': count,
                    'suggestion': f'Generate docs about "{topic}" — searched {count}x with no results',
                    'has_docs': False,
                })
            elif count >= 3:
                predictions.append({
                    'topic': topic,
                    'miss_count': count,
                    'suggestion': f'Improve docs about "{topic}" — existing docs not satisfying {count} searches',
                    'has_docs': True,
                })

        return predictions

    # Trends and ROI

    def get_trends(self, days: int = 30) -> dict[str, Any]:
        """Show how key metrics have changed over time.

        Returns:
            Dict with daily metrics: doc_count, hit_rate, calls per day.
        """
        from datetime import datetime, timedelta

        with self._conn_provider.acquire() as conn:
            cutoff = (datetime.now(UTC) - timedelta(days=days)).isoformat()

            daily_usage = []
            for row in conn.execute(
                "SELECT DATE(timestamp) as day, COUNT(*) as calls, "
                "SUM(CASE WHEN outcome='hit' THEN 1 ELSE 0 END) as hits, "
                "SUM(CASE WHEN outcome='miss' THEN 1 ELSE 0 END) as misses "
                "FROM usage_events WHERE timestamp >= ? "
                "GROUP BY DATE(timestamp) ORDER BY day",
                (cutoff,),
            ).fetchall():
                calls = row[1] or 0
                hits = row[2] or 0
                daily_usage.append({
                    'date': row[0],
                    'calls': calls,
                    'hits': hits,
                    'misses': row[3] or 0,
                    'hit_rate': hits / calls if calls > 0 else 0,
                })

        doc_count = self.count_documents()
        return {
            'current_doc_count': doc_count,
            'daily_usage': daily_usage,
            'total_days_tracked': len(daily_usage),
        }

    def estimate_roi(self, days: int = 30) -> dict[str, Any]:
        """Estimate return on investment of the Ariadne library.

        Estimates time saved per hit (avoids manual code exploration)
        vs cost of generating docs (API calls + compute time).

        Returns:
            Dict with estimated time saved, cost, and ROI ratio.
        """
        usage = self.get_usage_stats(days=days)
        total_hits = usage.get('total_hits', 0)
        total_calls = usage.get('total_calls', 0)

        # Estimates (conservative)
        avg_minutes_saved_per_hit = 5  # A useful doc saves ~5 min of grep/read
        avg_minutes_per_miss = 1  # Miss costs ~1 min (search + realize no result)
        doc_count = self.count_documents()
        # Per-doc cost tracks the CONFIGURED model's rates, not a fixed
        # constant — a hardcoded GPT-4o-mini $0.02 understated Opus by
        # ~5-10x and made the ROI ratio meaningless.
        from docgen.pricing import per_doc_generation_cost
        try:
            from config import get_config
            _model = get_config().model
        except Exception:
            _model = ''
        avg_generation_cost_per_doc_usd = per_doc_generation_cost(_model)

        time_saved_minutes = total_hits * avg_minutes_saved_per_hit
        time_wasted_minutes = usage.get('total_misses', 0) * avg_minutes_per_miss
        net_time_saved = time_saved_minutes - time_wasted_minutes
        generation_cost_usd = doc_count * avg_generation_cost_per_doc_usd

        return {
            'total_hits': total_hits,
            'total_calls': total_calls,
            'estimated_time_saved_minutes': net_time_saved,
            'estimated_time_saved_hours': round(net_time_saved / 60, 1),
            'estimated_generation_cost_usd': round(generation_cost_usd, 2),
            'doc_count': doc_count,
            'hit_rate': usage.get('hit_rate', 0),
            'roi_ratio': round(net_time_saved / max(1, generation_cost_usd * 60), 1),  # minutes saved per dollar
        }

    # ------------------------------------------------------------------
    # Persistent query cache
    # ------------------------------------------------------------------

    _CACHE_TTL_DAYS = 30

    def cache_get(self, cache_key: str) -> str | None:
        """Get a cached query result. Returns JSON string or None on miss/expired."""
        from datetime import datetime, timedelta

        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                'SELECT result_json, created_at FROM query_cache WHERE cache_key = ?',
                (cache_key,),
            ).fetchone()
        if not row:
            return None
        created = datetime.fromisoformat(row[1])
        if datetime.now(UTC) - created > timedelta(days=self._CACHE_TTL_DAYS):
            # Expired — prune lazily
            with self._conn_provider.acquire() as conn:
                conn.execute('DELETE FROM query_cache WHERE cache_key = ?', (cache_key,))
            return None
        return row[0]

    def cache_put(
        self, cache_key: str, branch: str, query_text: str, result_json: str,
    ) -> None:
        """Store a query result in the persistent cache."""
        from schema import _now_iso

        with self._conn_provider.acquire() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO query_cache '
                '(cache_key, branch, query_text, result_json, created_at) '
                'VALUES (?, ?, ?, ?, ?)',
                (cache_key, branch, query_text, result_json, _now_iso()),
            )

    def cache_clear(self, branch: str | None = None) -> int:
        """Clear the persistent cache. If branch given, only clear that branch."""
        with self._conn_provider.acquire() as conn:
            if branch:
                cur = conn.execute('DELETE FROM query_cache WHERE branch = ?', (branch,))
            else:
                cur = conn.execute('DELETE FROM query_cache')
            return cur.rowcount

    def cache_prune(self) -> int:
        """Remove expired cache entries (older than TTL)."""
        from datetime import datetime, timedelta

        cutoff = (datetime.now(UTC) - timedelta(days=self._CACHE_TTL_DAYS)).isoformat()
        with self._conn_provider.acquire() as conn:
            cur = conn.execute('DELETE FROM query_cache WHERE created_at < ?', (cutoff,))
            return cur.rowcount
