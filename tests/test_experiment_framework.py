"""Sealed artifacts, Pareto comparison, and calibration rejections.

The framework exists to eliminate known false confidence. Sealing uses
two digests: the artifact digest covers everything — timing and cost are
acceptance evidence, so mutating them invalidates the seal — while the
evidence digest normalizes only an explicit, schema-versioned list of
volatile paths so deterministic semantic comparison across runs stays
possible. The comparison rules must automatically reject the recorded
failed trajectory (raw up, downstream down, bodies +84.6%) and must
accept a pure cost reduction with identical frontiers.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location(
        name, ROOT / "evaluation" / "chain-benchmark" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp_seal = load("exp_seal")
exp_compare = load("exp_compare")


def make_payload():
    return {
        "schema": "ariadne-blind-shadow-v2",
        "provenance": {
            "git_head": "abc", "module_sha256": {"library/a.py": "x"},
            "feature_flags": {"forward_depth": 0}},
        "questions": [{
            "id": 1,
            "ledger": "m.py:8 `helper()`",
            "prompt_sha256": "p1",
            "usage": {"input_tokens": 100, "output_tokens": 10},
            "cost_usd": 0.01,
            "timings": {"walk": 1.23},
        }],
    }


class TestSealV3:
    def test_seal_and_verify_round_trip(self):
        sealed = exp_seal.seal(make_payload())
        exp_seal.verify_seal(sealed)
        assert sealed["seal"]["schema"] == "ariadne-experiment-seal-v3"

    def test_changing_timing_invalidates_the_artifact_digest(self):
        sealed = exp_seal.seal(make_payload())
        sealed["questions"][0]["timings"] = {"walk": 9.99}
        with pytest.raises(ValueError, match="modified after sealing"):
            exp_seal.verify_seal(sealed)

    def test_timing_differences_keep_the_evidence_digest_equal(self):
        first = make_payload()
        second = make_payload()
        second["questions"][0]["timings"] = {"walk": 9.99}

        assert (exp_seal.evidence_digest(first)
                == exp_seal.evidence_digest(second))
        assert (exp_seal.artifact_digest(first)
                != exp_seal.artifact_digest(second))

    @pytest.mark.parametrize("mutate", [
        lambda payload: payload["questions"][0].__setitem__(
            "cost_usd", 9.99),
        lambda payload: payload["questions"][0]["usage"].__setitem__(
            "input_tokens", 1),
        lambda payload: payload["questions"][0].__setitem__(
            "prompt_sha256", "forged"),
        lambda payload: payload["questions"][0].__setitem__(
            "ledger", "forged"),
        lambda payload: payload["provenance"]["feature_flags"].__setitem__(
            "forward_depth", 2),
        lambda payload: payload["provenance"].__setitem__(
            "git_head", "other"),
    ], ids=["cost", "usage", "prompt", "evidence", "flags", "provenance"])
    def test_nonvolatile_changes_invalidate_both_digests(self, mutate):
        baseline = make_payload()
        mutated = make_payload()
        mutate(mutated)

        assert (exp_seal.artifact_digest(baseline)
                != exp_seal.artifact_digest(mutated))
        assert (exp_seal.evidence_digest(baseline)
                != exp_seal.evidence_digest(mutated))
        sealed = exp_seal.seal(baseline)
        mutate(sealed)
        with pytest.raises(ValueError, match="modified after sealing"):
            exp_seal.verify_seal(sealed)

    def test_unknown_seal_algorithms_are_refused(self):
        sealed = exp_seal.seal(make_payload())
        sealed["seal"]["algorithm"] = "md5-trust-me"
        with pytest.raises(ValueError, match="algorithm"):
            exp_seal.verify_seal(sealed)

    def test_unsupported_seal_schema_versions_are_refused(self):
        sealed = exp_seal.seal(make_payload())
        sealed["seal"]["schema"] = "ariadne-experiment-seal-v99"
        with pytest.raises(ValueError, match="schema"):
            exp_seal.verify_seal(sealed)

    def test_unsealed_artifacts_are_refused(self):
        with pytest.raises(ValueError, match="unsealed"):
            exp_seal.verify_seal(make_payload())

    def test_provenance_is_mandatory(self):
        payload = make_payload()
        payload.pop("provenance")
        with pytest.raises(ValueError, match="provenance"):
            exp_seal.seal(payload)


def grade_fixture(frontiers):
    return {"claims": [
        {"question": question, "claim": claim, "frontier": frontier}
        for (question, claim), frontier in frontiers.items()]}


def artifacts_fixture(*, raw, bodies, excerpts, ledger_chars, final_chars,
                      prompt_tokens=1000, model_calls=8):
    return {"questions": [{
        "raw_pool": [["s"]] * raw,
        "retained": {"body_symbols": ["b"] * bodies},
        "materialized_excerpts": [["f"]] * excerpts,
        "ledger": "x" * ledger_chars,
        "final_artifact": "y" * final_chars,
        "prompt_tokens_total": prompt_tokens,
        "model_calls": model_calls,
    }]}


class TestParetoComparison:
    def test_the_recorded_failed_trajectory_is_rejected_automatically(self):
        baseline_grade = grade_fixture({
            (8, "delta-values"): "final", (66, "indexed"): "store"})
        candidate_grade = grade_fixture({
            (8, "delta-values"): "retained", (66, "indexed"): "raw"})
        result = exp_compare.compare(
            baseline_grade=baseline_grade,
            candidate_grade=candidate_grade,
            baseline_artifacts=artifacts_fixture(
                raw=24483 // 100, bodies=866, excerpts=1127,
                ledger_chars=125896, final_chars=125896),
            candidate_artifacts=artifacts_fixture(
                raw=35379 // 100, bodies=1599, excerpts=1660,
                ledger_chars=204768, final_chars=204768))

        assert result["accepted"] is False
        assert any("regressed" in reason for reason in result["reasons"])
        assert any("bodies grew" in reason for reason in result["reasons"])

    def test_raw_only_conversion_never_qualifies(self):
        baseline_grade = grade_fixture({(1, "a"): "store"})
        candidate_grade = grade_fixture({(1, "a"): "raw"})
        flat = artifacts_fixture(
            raw=100, bodies=10, excerpts=50, ledger_chars=1000,
            final_chars=1000)

        result = exp_compare.compare(
            baseline_grade=baseline_grade, candidate_grade=candidate_grade,
            baseline_artifacts=flat, candidate_artifacts=flat)

        assert result["accepted"] is False
        assert any("raw growth alone" in reason
                   for reason in result["reasons"])

    def test_a_downstream_conversion_without_growth_is_accepted(self):
        baseline_grade = grade_fixture({(1, "a"): "raw"})
        candidate_grade = grade_fixture({(1, "a"): "final"})
        flat = artifacts_fixture(
            raw=100, bodies=10, excerpts=50, ledger_chars=1000,
            final_chars=1000)

        result = exp_compare.compare(
            baseline_grade=baseline_grade, candidate_grade=candidate_grade,
            baseline_artifacts=flat, candidate_artifacts=flat)

        assert result["accepted"] is True
        assert result["converted"] == [
            {"claim": [1, "a"], "from": "raw", "to": "final"}]

    def test_identical_frontiers_with_material_cost_drop_are_accepted(self):
        same_grade = grade_fixture({(1, "a"): "final", (2, "b"): "internal_menu"})

        result = exp_compare.compare(
            baseline_grade=same_grade, candidate_grade=same_grade,
            baseline_artifacts=artifacts_fixture(
                raw=100, bodies=10, excerpts=50, ledger_chars=1000,
                final_chars=1000, prompt_tokens=10000, model_calls=8),
            candidate_artifacts=artifacts_fixture(
                raw=100, bodies=10, excerpts=50, ledger_chars=1000,
                final_chars=1000, prompt_tokens=7000, model_calls=8))

        assert result["accepted"] is True
        assert any("cost" in reason or "prompt" in reason
                   for reason in result["reasons"])

    def test_mismatched_claim_key_sets_are_refused(self):
        baseline_grade = grade_fixture({(1, "a"): "final"})
        candidate_grade = grade_fixture({(1, "renamed"): "final"})
        flat = artifacts_fixture(
            raw=10, bodies=1, excerpts=5, ledger_chars=100,
            final_chars=100)

        with pytest.raises(ValueError, match="claim key"):
            exp_compare.compare(
                baseline_grade=baseline_grade,
                candidate_grade=candidate_grade,
                baseline_artifacts=flat, candidate_artifacts=flat)


class TestEvidenceVolatilePaths:
    def test_output_path_in_argv_does_not_break_evidence_comparison(self):
        first = make_payload()
        second = make_payload()
        first["provenance"]["command"] = {
            "argv": ["experiment.py", "--out", "/tmp/run-a.json"],
            "questions": "questions.json"}
        second["provenance"]["command"] = {
            "argv": ["experiment.py", "--out", "/tmp/run-b.json"],
            "questions": "questions.json"}

        assert (exp_seal.evidence_digest(first)
                == exp_seal.evidence_digest(second))
        assert (exp_seal.artifact_digest(first)
                != exp_seal.artifact_digest(second))

    def test_structured_command_identity_stays_evidence_relevant(self):
        first = make_payload()
        second = make_payload()
        first["provenance"]["command"] = {"questions": "questions.json"}
        second["provenance"]["command"] = {"questions": "OTHER.json"}

        assert (exp_seal.evidence_digest(first)
                != exp_seal.evidence_digest(second))
