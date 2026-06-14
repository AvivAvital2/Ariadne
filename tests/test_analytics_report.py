"""Evolving TDD for the portable analytics report.

Background: in the "build the DB elsewhere, ship it to the serving box"
model, *replacing* ``ariadne.db`` wipes ``usage_events``. This report is
the durable distillation of that signal — built before the swap so that
``improve``/``gaps`` can still consume hit/miss/score analytics afterwards,
and so insights survive across DB generations.

One test grows here with the feature: capture -> round-trip -> consume.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from analytics_report import AnalyticsReport, build_analytics_report
from cli.status import cmd_usage
from library import Library


@pytest.fixture
def library(tmp_path: Path):
    lib = Library(tmp_path / 'analytics_report.db')
    yield lib
    lib.close()


def _seed(library) -> None:
    """Two hits (one carrying a ``score:N``) and a miss-with-feedback —
    enough to populate every signal the report must carry.
    """
    h1 = library.log_usage(
        tool_name='ariadne_search', query='how caching works',
        result_count=2, document_ids=['doc-a', 'doc-b'],
    )
    library.mark_hit(h1, feedback='score:8 - exactly the pattern I needed')
    library.log_usage(
        tool_name='ariadne_search', query='how retries work',
        result_count=1, document_ids=['doc-a'],
    )
    m1 = library.log_usage(
        tool_name='ariadne_search', query='token refresh flow',
        result_count=0,
    )
    library.mark_miss(m1, feedback='no doc on token refresh')


def test_analytics_report_captures_round_trips_and_consumes(library) -> None:
    """The report captures the hit/miss/score signal ``improve``/``gaps``
    rely on, round-trips through JSON unchanged, and hands that signal
    back in the ``get_gap_report`` shape those commands already consume —
    so analytics keep flowing after the content DB is rebuilt and swapped.

    This single test grows with the feature rather than fragmenting into a
    parade of isolated cases.
    """
    _seed(library)

    # --- Demand 1: build captures the usage summary + missed queries ---
    report = build_analytics_report(library, days=30)
    assert report.window_days == 30
    assert report.usage_summary['total_calls'] == 3
    assert report.usage_summary['total_misses'] == 1
    assert 'no doc on token refresh' in [g['feedback'] for g in report.missed_queries]

    # --- Demand 2: response scores (quality_score) are captured ---
    # The seeded hit carried "score:8". get_usage_stats already aggregates
    # quality_score, so the report carries it via usage_summary — reused
    # rather than bolting on a redundant scores field.
    assert report.usage_summary['scored_count'] == 1
    assert report.usage_summary['avg_quality_score'] == 8.0
    assert report.usage_summary['score_distribution'][8] == 1

    # --- Demand 3: per-document signals (the inputs improve Steps 3/3c read) ---
    # Faithfully carries usage_by_document + find_low_value_documents, so the
    # report is a drop-in for those live reads once the db is swapped.
    assert report.doc_signals['top_served'] == library.usage_by_document(days=30)
    assert report.doc_signals['low_value'] == library.find_low_value_documents(days=30)

    # --- Demand 4: the signal survives a JSON round-trip (it gets shipped) ---
    restored = AnalyticsReport.from_json(report.to_json())
    assert restored.window_days == 30
    assert restored.usage_summary['total_misses'] == 1
    assert 'no doc on token refresh' in [g['feedback'] for g in restored.missed_queries]

    # --- Demand 5: hands the signal back in the get_gap_report shape ---
    # This is the consumption contract: improve/gaps read this instead of a
    # live (just-wiped) usage_events via --from-report.
    gap_shaped = restored.as_gap_report()
    assert gap_shaped['total_misses'] == 1
    assert 'no doc on token refresh' in [g['feedback'] for g in gap_shaped['top_gaps']]
    assert isinstance(gap_shaped['recent_misses'], list)


def test_usage_command_exports_a_consumable_report(tmp_path: Path) -> None:
    """``ariadne usage --export-report PATH`` writes the portable report —
    the producer half of the replace-DB workflow.

    End-to-end (a distinct layer from the report logic above, hence its own
    test): seed a db, run the handler, read the file back, confirm it carries
    the signal in the consumable shape.
    """
    db = tmp_path / 'usage.db'
    lib = Library(db)
    _seed(lib)
    lib.close()

    out = tmp_path / 'analytics-report.json'
    args = argparse.Namespace(
        db=db, days=30, tool=None, by_document=False,
        top_served=None, export_report=str(out),
    )
    assert cmd_usage(args) == 0
    assert out.exists()

    report = AnalyticsReport.from_json(out.read_text())
    assert report.usage_summary['total_misses'] == 1
    assert report.as_gap_report()['total_misses'] == 1
