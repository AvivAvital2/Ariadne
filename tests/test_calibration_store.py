"""CalibrationStore: persist real per-call LLM usage and aggregate it,
so the cost estimator can self-tune to this codebase + model instead of
relying on fixed heuristic token counts.
"""
from __future__ import annotations

import pytest


def test_records_and_means_per_bucket(tmp_path):
    from docgen.calibration import CalibrationStore

    store = CalibrationStore(tmp_path / 'cal.db')
    store.record(
        phase='describe', doc_type='element', language='python',
        model='claude-opus-4-8', input_tokens=210, output_tokens=55,
    )
    store.record(
        phase='describe', doc_type='element', language='python',
        model='claude-opus-4-8', input_tokens=230, output_tokens=65,
    )

    cal = store.mean_tokens(
        phase='describe', model='claude-opus-4-8',
        doc_type='element', language='python',
    )
    assert cal is not None
    assert cal.n == 2
    assert cal.mean_input == 220.0   # (210 + 230) / 2
    assert cal.mean_output == 60.0   # (55 + 65) / 2


def test_no_data_returns_none(tmp_path):
    from docgen.calibration import CalibrationStore

    store = CalibrationStore(tmp_path / 'cal.db')
    assert store.mean_tokens(phase='generate', model='claude-opus-4-8') is None


def test_aggregates_across_language_when_unspecified(tmp_path):
    """Omitting language/doc_type aggregates over the matching rows —
    useful for a phase-wide fallback when a specific bucket has no data."""
    from docgen.calibration import CalibrationStore

    store = CalibrationStore(tmp_path / 'cal.db')
    store.record(phase='generate', doc_type='explanation', language='python',
                 model='m', input_tokens=1000, output_tokens=1200)
    store.record(phase='generate', doc_type='architecture', language='java',
                 model='m', input_tokens=2000, output_tokens=1800)

    # Filter to one bucket.
    py = store.mean_tokens(phase='generate', model='m',
                           doc_type='explanation', language='python')
    assert py.n == 1 and py.mean_output == 1200.0

    # Phase-wide (no doc_type/language) → averages both rows.
    allrows = store.mean_tokens(phase='generate', model='m')
    assert allrows.n == 2
    assert allrows.mean_input == 1500.0   # (1000 + 2000) / 2
    assert allrows.mean_output == 1500.0  # (1200 + 1800) / 2

    # A different model is isolated.
    assert store.mean_tokens(phase='generate', model='other') is None


def test_usage_capture_routes_to_observer_within_context():
    """emit_usage forwards to the active observer, tagged with the
    current context. Outside a context, or with no observer, it's a
    no-op (so library code can call it unconditionally)."""
    from docgen.calibration import (
        emit_usage, set_usage_observer, usage_context,
    )

    recorded: list[dict] = []
    with set_usage_observer(lambda **kw: recorded.append(kw)):
        with usage_context(phase='describe', doc_type='element',
                           language='python'):
            emit_usage(model='m', input_tokens=10, output_tokens=5)
    assert recorded == [{
        'phase': 'describe', 'doc_type': 'element', 'language': 'python',
        'model': 'm', 'input_tokens': 10, 'output_tokens': 5,
    }]

    # No active context → no-op (don't record uncontextualized usage).
    recorded.clear()
    with set_usage_observer(lambda **kw: recorded.append(kw)):
        emit_usage(model='m', input_tokens=1, output_tokens=1)
    assert recorded == []

    # No observer → no-op (never raises).
    with usage_context(phase='describe', doc_type='element', language=None):
        emit_usage(model='m', input_tokens=1, output_tokens=1)


def test_record_observer_persists_to_store(tmp_path):
    """A CalibrationStore can serve as the observer, so captured usage
    lands in the store end-to-end."""
    from docgen.calibration import (
        CalibrationStore, emit_usage, set_usage_observer, usage_context,
    )

    store = CalibrationStore(tmp_path / 'cal.db')
    with set_usage_observer(store.record):
        with usage_context(phase='describe', doc_type='element',
                           language='python'):
            emit_usage(model='m', input_tokens=300, output_tokens=70)
    cal = store.mean_tokens(phase='describe', model='m')
    assert cal is not None and cal.n == 1
    assert cal.mean_input == 300.0 and cal.mean_output == 70.0


def test_persists_across_instances(tmp_path):
    from docgen.calibration import CalibrationStore

    db = tmp_path / 'cal.db'
    CalibrationStore(db).record(
        phase='describe', doc_type='element', language='ts',
        model='m', input_tokens=100, output_tokens=50,
    )
    # Re-open: data survives.
    cal = CalibrationStore(db).mean_tokens(phase='describe', model='m')
    assert cal is not None and cal.n == 1
