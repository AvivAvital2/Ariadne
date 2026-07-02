"""Facade methods delegating to Library and self-improvement logic."""
from __future__ import annotations

import logging
from pathlib import Path

from ariadne_mcp.models import (
    DayStats,
    DocumentUsageResponse,
    FeedbackEntry,
    GapEntry,
    GapReportResponse,
    MissEntry,
    ProjectStatsResponse,
    ToolStats,
    UsageStatsResponse,
)

_logger = logging.getLogger(__name__)


class FacadesMixin:
    """Thin facades over Library methods, plus self-improvement.

    Expects the composed class to provide:
    - self.library: Library
    - self.config: Config
    - self.generate() from AdminMixin
    """

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def usage_stats(
        self,
        days: int = 30,
        tool_name: str | None = None,
    ) -> UsageStatsResponse:
        """Get usage statistics."""
        raw = self.library.get_usage_stats(days=days, tool_name=tool_name)

        return UsageStatsResponse(
            total_calls=raw['total_calls'],
            total_hits=raw['total_hits'],
            total_misses=raw['total_misses'],
            hit_rate=raw['hit_rate'],
            avg_calls_per_day=raw['avg_calls_per_day'],
            by_tool={
                name: ToolStats(**data)
                for name, data in raw['by_tool'].items()
            },
            by_day=[DayStats(**d) for d in raw['by_day']],
            recent_feedback=[FeedbackEntry(**f) for f in raw['recent_feedback']],
        )

    def gap_report(self, days: int = 30) -> GapReportResponse:
        """Get documentation gap report."""
        raw = self.library.get_gap_report(days=days)

        return GapReportResponse(
            total_misses=raw['total_misses'],
            miss_rate=raw['miss_rate'],
            top_gaps=[GapEntry(**g) for g in raw['top_gaps']],
            recent_misses=[MissEntry(**m) for m in raw['recent_misses']],
        )

    async def gap_analysis(self, days: int = 30) -> str | None:
        """Run LLM-powered gap analysis. Returns analysis text or None."""
        raw = self.library.get_gap_report(days=days)
        if raw['total_misses'] == 0:
            return None
        try:
            from gap_analysis import analyze_gaps
            report = await analyze_gaps(raw['recent_misses'])
            lines = [report.summary, '']
            for rec in report.recommendations:
                lines.append(f'## {rec.theme} ({rec.miss_count} misses)')
                lines.append(f'   {rec.description}')
                lines.append(f'   Recommendation: {rec.recommendation}')
            return '\n'.join(lines)
        except ImportError:
            return 'LLM analysis unavailable (gap_analysis module not found).'
        except Exception as e:
            _logger.warning('Gap analysis failed: %s', e)
            return f'LLM analysis failed: {e}'

    def project_stats(self) -> ProjectStatsResponse:
        """Get per-project document and size statistics."""
        from ariadne_mcp.models import ProjectStatsResponse, SourceStats

        raw = self.library.stats_by_source()
        total_meta = raw.pop('_total', {})

        sources = {
            name: SourceStats(
                doc_count=data['doc_count'],
                content_size=data['content_size'],
                embedding_size=data['embedding_size'],
                chunk_count=data.get('chunk_count', 0),
            )
            for name, data in raw.items()
        }

        return ProjectStatsResponse(
            by_source=sources,
            total_documents=sum(s.doc_count for s in sources.values()),
            total_content_size=sum(s.content_size for s in sources.values()),
            db_size_bytes=total_meta.get('db_size_bytes', 0),
        )

    def document_usage(self, days: int = 30, limit: int = 20) -> DocumentUsageResponse:
        """Get per-document serving statistics."""
        from ariadne_mcp.models import DocumentServeStats, DocumentUsageResponse

        raw = self.library.usage_by_document(days=days, limit=limit)

        return DocumentUsageResponse(
            documents=[
                DocumentServeStats(
                    document_id=r['document_id'],
                    title=r['title'],
                    content_type=r['content_type'],
                    serve_count=r['serve_count'],
                )
                for r in raw
            ],
            days=days,
        )

    # ------------------------------------------------------------------
    # Knowledge & explanation (facade over Library)
    # ------------------------------------------------------------------

    def explain(self, file_path: str, kinds: list[str] | None = None, sections_only: bool = False, offset: int = 0, limit: int | None = None) -> dict:
        """Explain a file using Ariadne docs."""
        return self.library.explain(file_path, kinds=kinds, sections_only=sections_only, offset=offset, limit=limit)

    def review_checklist(self, changed_files: list[str]) -> list[dict]:
        """Generate PR review checklist from knowledge of changed files."""
        return self.library.review_checklist(changed_files)

    def impact_radius(self, file_path: str) -> dict:
        """Show what would break if a file changes."""
        return self.library.impact_radius(file_path)

    def summarize_file(self, file_path: str) -> str:
        """Summarize a file using Ariadne docs."""
        return self.library.summarize_file(file_path)

    # ------------------------------------------------------------------
    # Debugging (facade over Library)
    # ------------------------------------------------------------------

    def diagnose(self, error_message: str) -> dict:
        """Diagnose an error using Ariadne docs."""
        return self.library.diagnose(error_message)

    def find_tests_for(self, target: str) -> list[dict[str, str]]:
        """Find test files related to a target."""
        return self.library.find_tests_for(target)

    def find_tests_for_topic(self, target: str) -> list:
        """Find test files related to a topic."""
        return self.library.find_tests_for_topic(target)

    def find_documents_by_source_files(self, files: list[str]) -> list:
        """Find documents covering given source files."""
        return self.library.find_documents_by_source_files(files)

    def full_debug_context(self, file_path: str, source_path: Path | None = None) -> dict:
        """Assemble complete debug context for a file.

        Composes debug_context + extract_gotchas + find_tests_for into one call.
        """
        result = self.library.debug_context(file_path, source_path)
        result['gotchas'] = self.library.extract_gotchas(file_path)
        result['test_files'] = self.library.find_tests_for(file_path)
        return result

    # ------------------------------------------------------------------
    # Graph (facade over Library)
    # ------------------------------------------------------------------

    def build_graph(self, source_path: Path, extra_source_paths: list[Path] | None = None) -> dict:
        """Build the dependency graph from source files."""
        return self.library.build_graph(source_path, extra_source_paths)

    def get_graph_stats(self) -> dict:
        """Get graph statistics."""
        return self.library.get_graph_stats()

    def get_priorities(self, source_path: Path) -> list:
        """Get files ranked by priority (undocumented + highly connected first)."""
        return self.library.get_priorities(source_path)

    def export_graph_json(self) -> dict:
        """Export graph as JSON for visualization."""
        return self.library.export_graph_json()

    # ------------------------------------------------------------------
    # Self-improvement
    # ------------------------------------------------------------------

    def self_improve(self, event_id: int) -> dict:
        """Diagnose a low-score event and trigger doc regeneration if actionable.

        Only acts on events with score ≤5 and a specific, actionable reason.
        Skips vague feedback and areas not covered by source files.

        Returns:
            Dict with diagnosis, action_taken, and details.
        """
        diagnosis = self.library.diagnose_low_score(event_id)
        if not diagnosis.get('is_actionable'):
            return {
                'diagnosis': diagnosis,
                'action_taken': False,
                'detail': diagnosis.get('recommendation', 'No action needed.'),
            }

        source_files = diagnosis.get('source_files', [])
        reason = diagnosis.get('reason', '')

        if not source_files:
            return {
                'diagnosis': diagnosis,
                'action_taken': False,
                'detail': 'No source files found — cannot regenerate docs from nothing.',
            }

        # Trigger regeneration for each source file
        regenerated = []
        for sf in source_files:
            try:
                self.generate(path=sf)
                regenerated.append(sf)
            except Exception:
                _logger.debug('Regeneration failed for %s', sf, exc_info=True)

        return {
            'diagnosis': diagnosis,
            'action_taken': len(regenerated) > 0,
            'regenerated_files': regenerated,
            'detail': (
                f'Regenerated docs for {len(regenerated)} files targeting gap: {reason}'
                if regenerated else 'Regeneration failed for all source files.'
            ),
        }
