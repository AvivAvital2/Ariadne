"""Tests for the dry-run cost estimator.

The estimator takes the file list, requested doc types, and model name,
and produces a token + dollar estimate without making any LLM calls.
Used by ``ariadne generate --dry-run`` to give users a preview.
"""
from __future__ import annotations

from pathlib import Path


def test_describe_tokens_per_call_prefers_calibration(tmp_path):
    """The describe estimate uses real calibrated tokens/call when the
    store has data for the model, else the 200/60 heuristic."""
    from cli.dry_run import (
        _DESCRIBE_INPUT_TOKENS_PER_CALL,
        _DESCRIBE_OUTPUT_TOKENS_PER_CALL,
        _describe_tokens_per_call,
    )
    from docgen.calibration import CalibrationStore

    store = CalibrationStore(tmp_path / 'cal.db')
    # No data yet → heuristic.
    assert _describe_tokens_per_call(store, 'claude-opus-4-8') == (
        _DESCRIBE_INPUT_TOKENS_PER_CALL, _DESCRIBE_OUTPUT_TOKENS_PER_CALL,
    )
    # Seed real usage → calibrated.
    store.record(phase='describe', doc_type='element', language='python',
                 model='claude-opus-4-8', input_tokens=500, output_tokens=120)
    assert _describe_tokens_per_call(store, 'claude-opus-4-8') == (500.0, 120.0)
    # A different model still falls back.
    assert _describe_tokens_per_call(store, 'gpt-5.4') == (
        _DESCRIBE_INPUT_TOKENS_PER_CALL, _DESCRIBE_OUTPUT_TOKENS_PER_CALL,
    )


def test_estimate_cost_uses_calibrated_output_when_provided():
    """When real per-bucket output tokens are known (from past runs),
    estimate_cost uses them instead of the flat AVG_OUTPUT_TOKENS_PER_CALL
    heuristic — the single biggest accuracy lever."""
    from docgen.pricing import estimate_cost

    files = [(Path('a.py'), 4000)]

    def calibrated_output(doc_type, language):
        return 400.0  # actual runs show ~400, not the 1500 default

    base = estimate_cost(
        files=files, doc_types=('explanation',), model='claude-opus-4-7',
    )
    calibrated = estimate_cost(
        files=files, doc_types=('explanation',), model='claude-opus-4-7',
        output_tokens_for=calibrated_output,
    )

    assert base.output_tokens == 1500, 'heuristic baseline'
    assert calibrated.output_tokens == 400, 'one call × calibrated 400'
    assert calibrated.total_cost_usd < base.total_cost_usd
    # A bucket with no calibration (returns None) falls back to heuristic.
    mixed = estimate_cost(
        files=files, doc_types=('explanation',), model='claude-opus-4-7',
        output_tokens_for=lambda dt, lang: None,
    )
    assert mixed.output_tokens == 1500


def test_per_doc_generation_cost_tracks_model_rates():
    """ROI's per-doc generation cost must follow the configured model's
    rates, not a fixed GPT-4o-mini constant ($0.02)."""
    from docgen.pricing import per_doc_generation_cost

    opus = per_doc_generation_cost('claude-opus-4-8')
    sonnet = per_doc_generation_cost('claude-sonnet-4-6')
    assert opus > sonnet > 0, 'pricier model → higher per-doc cost'
    # Opus output alone (~1500 tokens × $25/M ≈ $0.0375) exceeds the old
    # flat $0.02 — the hardcoded value was way under for Opus.
    assert opus > 0.02
    # Unknown model → a sane non-zero fallback, never a crash or 0.
    assert per_doc_generation_cost('mystery-model-9') > 0


def test_per_doc_type_breakdown_sums_to_aggregate():
    """The per-doc-type breakdown (for the dry-run picker) must account
    for the same total as the aggregate estimate — one entry per type,
    summing to the whole."""
    from docgen.pricing import estimate_cost, estimate_generate_by_doc_type

    files = [(Path('a.py'), 4000), (Path('b.py'), 8000)]
    types = ('explanation', 'architecture', 'qa', 'gotcha', 'diagram')

    aggregate = estimate_cost(
        files=files, doc_types=types, model='claude-opus-4-7',
    )
    per_type = estimate_generate_by_doc_type(
        files, types, 'claude-opus-4-7',
    )

    assert [t for t, _ in per_type] == list(types), (
        'one entry per requested doc type, in order'
    )
    summed = sum(est.total_cost_usd for _, est in per_type)
    assert abs(summed - aggregate.total_cost_usd) < 1e-6, (
        f'per-type costs ({summed}) must sum to the aggregate '
        f'({aggregate.total_cost_usd})'
    )


def test_estimate_returns_zero_for_empty_files():
    from docgen.pricing import estimate_cost

    est = estimate_cost(
        files=(),
        doc_types=('explanation',),
        model='gpt-5.4',
    )
    assert est.file_count == 0
    assert est.total_calls == 0
    assert est.input_tokens == 0
    assert est.output_tokens == 0
    assert est.total_cost_usd == 0.0


def test_estimate_known_model_produces_dollar_amount(tmp_path):
    """Single file with known size + model produces a non-zero cost."""
    from docgen.pricing import estimate_cost

    f = tmp_path / 'x.py'
    f.write_text('x = 1\n' * 100, encoding='utf-8')  # ~600 bytes

    est = estimate_cost(
        files=((f, f.stat().st_size),),
        doc_types=('explanation', 'architecture'),
        model='gpt-5.4',
    )
    assert est.file_count == 1
    # Python supports both doc types → 2 calls
    assert est.total_calls == 2
    assert est.input_tokens > 0
    assert est.output_tokens > 0
    assert est.total_cost_usd > 0.0
    # Sanity: at gpt-5.4 rates ($2.50/M in, $15/M out), one tiny file
    # shouldn't cost more than a fraction of a cent
    assert est.total_cost_usd < 0.10


def test_estimate_unknown_model_flags_no_rate():
    """Unknown model produces an estimate with rates=None and total=0;
    the caller can then warn the user to set a known model.
    """
    from docgen.pricing import estimate_cost

    est = estimate_cost(
        files=(),
        doc_types=('explanation',),
        model='custom-finetune-v9',
    )
    assert est.rates is None
    assert est.total_cost_usd == 0.0


def test_estimate_filters_doc_types_per_language(tmp_path):
    """JSON/YAML/MD only support 'explanation' — multi-type request
    must shrink to one call per file for those languages.
    """
    from docgen.pricing import estimate_cost

    json_file = tmp_path / 'config.json'
    json_file.write_text('{}\n', encoding='utf-8')
    py_file = tmp_path / 'x.py'
    py_file.write_text('x = 1\n', encoding='utf-8')

    est = estimate_cost(
        files=(
            (json_file, json_file.stat().st_size),
            (py_file, py_file.stat().st_size),
        ),
        doc_types=('explanation', 'architecture', 'qa'),
        model='gpt-5.4',
    )
    # JSON => 1 call (explanation only); Python => 3 calls.
    assert est.total_calls == 4


def test_estimate_scales_linearly_with_file_count(tmp_path):
    """10 files of similar size cost ~10× a single file."""
    from docgen.pricing import estimate_cost

    f = tmp_path / 'x.py'
    f.write_text('x = 1\n' * 100, encoding='utf-8')
    size = f.stat().st_size

    one = estimate_cost(
        files=((f, size),),
        doc_types=('explanation',),
        model='gpt-5.4',
    )
    ten = estimate_cost(
        files=tuple((f, size) for _ in range(10)),
        doc_types=('explanation',),
        model='gpt-5.4',
    )
    # Allow a small overhead but expect ~10×.
    assert 8.5 * one.total_cost_usd <= ten.total_cost_usd <= 11 * one.total_cost_usd


def test_estimate_gives_uncertainty_band():
    """Estimator returns lower/upper bounds reflecting tokenizer variance.
    The mid-point should fall inside the band.
    """
    from docgen.pricing import estimate_cost

    est = estimate_cost(
        files=((Path('/tmp/fake.py'), 10000),),
        doc_types=('explanation',),
        model='gpt-5.4',
    )
    assert est.cost_lower_bound <= est.total_cost_usd <= est.cost_upper_bound
    # Bounds should be meaningfully wide for a tokenizer-ish heuristic
    assert est.cost_upper_bound > est.cost_lower_bound


def test_pricing_table_has_entries_for_supported_models():
    """Sanity: the table includes the models we recommend in the one-pager."""
    from docgen.pricing import LLM_PRICING

    for m in (
        'gpt-5.2', 'gpt-5.4', 'gpt-5.5',
        'claude-opus-4-6', 'claude-opus-4-7', 'claude-opus-4-8',
        'claude-sonnet-4-6',
    ):
        assert m in LLM_PRICING, f'{m} missing from LLM_PRICING'
        in_rate, out_rate = LLM_PRICING[m]
        assert in_rate > 0 and out_rate > 0
