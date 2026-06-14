"""Portable analytics report — a durable distillation of ``usage_events``.

In the "build the DB elsewhere, ship it to the serving box" model,
*replacing* ``ariadne.db`` wipes ``usage_events``. This module snapshots the
hit/miss/score signal into a serializable report *before* the swap, so
``improve``/``gaps`` can keep consuming analytics after the content DB is
rebuilt — and so insights survive across DB generations rather than dying
with each one.

The report carries exactly the signals those commands read live
(``get_usage_stats`` / ``get_gap_report`` / ``usage_by_document`` /
``find_low_value_documents``) plus the optional LLM ``GapReport``, and can
hand them back in the ``get_gap_report`` shape via :meth:`as_gap_report`.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any

from attrs import asdict, frozen

from gap_analysis import GapRecommendation, GapReport

if TYPE_CHECKING:
    from library import Library


@frozen
class AnalyticsReport:
    """A serializable snapshot of usage analytics, decoupled from the db."""

    window_days: int
    usage_summary: dict[str, Any]
    missed_queries: tuple[dict[str, Any], ...]
    recent_misses: tuple[dict[str, Any], ...]
    doc_signals: dict[str, Any]
    gaps: GapReport | None = None

    def to_dict(self) -> dict[str, Any]:
        """Plain JSON-native dict (nested attrs -> dicts, tuples -> lists)."""
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalyticsReport:
        gaps_data = data.get('gaps')
        gaps: GapReport | None = None
        if gaps_data is not None:
            gaps = GapReport(
                total_misses=gaps_data['total_misses'],
                analysis_period_days=gaps_data['analysis_period_days'],
                recommendations=tuple(
                    GapRecommendation(
                        theme=r['theme'],
                        miss_count=r['miss_count'],
                        description=r['description'],
                        recommendation=r['recommendation'],
                        example_queries=tuple(r.get('example_queries', ())),
                    )
                    for r in gaps_data.get('recommendations', ())
                ),
                summary=gaps_data['summary'],
            )
        return cls(
            window_days=data['window_days'],
            usage_summary=data['usage_summary'],
            missed_queries=tuple(data.get('missed_queries', ())),
            recent_misses=tuple(data.get('recent_misses', ())),
            doc_signals=data.get('doc_signals', {}),
            gaps=gaps,
        )

    @classmethod
    def from_json(cls, text: str) -> AnalyticsReport:
        return cls.from_dict(json.loads(text))

    def as_gap_report(self) -> dict[str, Any]:
        """Re-expose the signal in the ``get_gap_report`` shape.

        ``improve`` (Step 1) and the ``gaps`` command read
        ``{total_misses, top_gaps, recent_misses}`` off ``get_gap_report``;
        returning that shape here lets them consume this report in place of a
        live — possibly just-wiped — ``usage_events`` table.
        """
        return {
            'total_misses': self.usage_summary.get('total_misses', 0),
            'top_gaps': list(self.missed_queries),
            'recent_misses': list(self.recent_misses),
        }


def build_analytics_report(
    library: Library, *, days: int = 30, gaps: GapReport | None = None,
) -> AnalyticsReport:
    """Snapshot ``library``'s usage analytics into a portable report.

    Reads only the existing analytics methods so it stays in lockstep with
    what ``improve``/``gaps`` consume. ``gaps`` carries the optional LLM
    ``GapReport`` (built separately, since that needs network); the raw
    hit/miss/score signal is always captured.
    """
    gap = library.get_gap_report(days=days)
    return AnalyticsReport(
        window_days=days,
        usage_summary=library.get_usage_stats(days=days),
        missed_queries=tuple(gap['top_gaps']),
        recent_misses=tuple(gap['recent_misses']),
        doc_signals={
            'top_served': library.usage_by_document(days=days),
            'low_value': library.find_low_value_documents(days=days),
        },
        gaps=gaps,
    )
