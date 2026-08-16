"""Historical cost calibration must reproduce the saved-trace facts.

The live22 traces are the calibration corpus: approximately 174 calls
and 3,988,737 input tokens, concentrated in formulation, exact-route
selection, and symbol selection. A cost module that cannot reproduce
those recorded facts must not be trusted to project a budget.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp_cost", ROOT / "evaluation" / "chain-benchmark" / "exp_cost.py")
exp_cost = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(exp_cost)

LIVE22_TRACES = (ROOT / "evaluation" / "chain-benchmark"
                 / "live22-diagnostic-answers-traces")
PRICES = ROOT / "evaluation" / "chain-benchmark" / "price-config-v1.json"


@pytest.mark.skipif(
    not LIVE22_TRACES.is_dir(),
    reason="saved live22 traces are a local calibration corpus")
class TestLive22Calibration:
    def test_reproduces_saved_call_and_token_totals(self):
        profile = exp_cost.trace_cost_profile(LIVE22_TRACES)

        assert profile["total"]["calls"] == 174
        assert profile["total"]["input_tokens"] == 3988737
        assert profile["total"]["questions"] == 22
        assert 180000 <= profile["total"]["input_tokens_per_question"] <= (
            182000)

    def test_reproduces_phase_concentration(self):
        profile = exp_cost.trace_cost_profile(LIVE22_TRACES)
        phases = profile["per_phase"]

        formulation = phases["completion"]["input_tokens"]
        exact_route = phases["scip-exact-route-select"]["input_tokens"]
        symbols = phases["scip-symbol-select"]["input_tokens"]
        assert 2.4e6 <= formulation <= 2.6e6
        assert 7.0e5 <= exact_route <= 7.9e5
        assert 4.2e5 <= symbols <= 4.9e5
        share = (formulation + exact_route + symbols) / (
            profile["total"]["input_tokens"])
        assert share >= 0.90

    def test_projects_the_recorded_live_cost_magnitude(self):
        profile = exp_cost.trace_cost_profile(LIVE22_TRACES)
        prices = exp_cost.load_price_config(PRICES)

        projection = exp_cost.project_cost(
            profile, prices, "claude-opus-4-8")

        # The recorded run cost $21.256; the projection from token
        # telemetry alone must land in the same magnitude (caching and
        # per-run variance explain the remainder).
        assert 15.0 <= projection["total_usd"] <= 30.0
        assert projection["price_config_version"] == "price-config-v1"


class TestPriceConfig:
    def test_unversioned_price_files_are_refused(self, tmp_path):
        bad = tmp_path / "prices.json"
        bad.write_text('{"models": {}}')
        with pytest.raises(ValueError, match="version"):
            exp_cost.load_price_config(bad)

    def test_unknown_models_are_refused(self):
        prices = exp_cost.load_price_config(PRICES)
        with pytest.raises(ValueError, match="absent"):
            exp_cost.project_cost(
                {"total": {"input_tokens": 1, "output_tokens": 1,
                           "questions": 1}},
                prices, "unknown-model")
