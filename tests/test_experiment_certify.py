"""Certificates without outcome leakage; a certifier that can also say yes.

The known-failure rejections matter only if the certifier is not
always-rejecting, so a synthetic complete, bounded, strongly-attributed
preflight must qualify for a canary while oracle modes, over-budget
projections, ambiguity, weak fingerprints, and dirty runtime manifests
are each refused for their own named reason.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evaluation" / "chain-benchmark"
sys.path.insert(0, str(BENCH))


def load(name):
    spec = importlib.util.spec_from_file_location(name, BENCH / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


exp_seal = load("exp_seal")
exp_certificate = load("exp_certificate")
exp_registry = load("exp_registry")
exp_backtest = load("exp_backtest")


def sealed_preflight(*, mode="blind-facet", ambiguous=0, strong=True,
                     untracked=()):
    payload = {
        "schema": "ariadne-blind-shadow-v3",
        "mode": mode,
        "provenance": {
            "runtime_manifest": {
                "manifest_sha256": "m" * 8,
                "untracked_runtime_files": list(untracked)},
            "database_fingerprint": {
                "level": "strong" if strong else "fast",
                "strong_sha256": "d" * 8},
            "questions_sha256": "q" * 8,
            "embedding_cache_sha256": "e" * 8,
            "command": {"questions": "questions.json"},
        },
        "questions": [{
            "id": 1,
            "resolution_census": {
                "exact": 10, "ambiguous": ambiguous, "missing": 0},
            "ledger": "x"}],
    }
    return exp_seal.seal(payload)


def full_grade(ceiling=9):
    return {"stage_vector": {
        stage: ceiling for stage in (
            "store", "raw", "internal_menu", "retained", "materialized",
            "ledger", "final")}}


BOUNDED_COST = {"usd_per_question": 0.25, "calls_per_question": 2}


class TestCanaryPrediction:
    def test_complete_bounded_preflight_qualifies(self):
        verdict = exp_certificate.predict_canary_eligibility(
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(),
            cost_projection=BOUNDED_COST,
            store_ceiling=9)

        assert verdict["passed"] is True

    def test_over_budget_complete_preflight_is_refused(self):
        verdict = exp_certificate.predict_canary_eligibility(
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(),
            cost_projection={"usd_per_question": 0.97,
                             "calls_per_question": 7.9},
            store_ceiling=9)

        assert verdict["passed"] is False
        assert any("exceeds" in reason for reason in verdict["reasons"])

    def test_oracle_mode_can_never_certify(self):
        verdict = exp_certificate.predict_canary_eligibility(
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(mode="oracle"),
            cost_projection=BOUNDED_COST,
            store_ceiling=9)

        assert verdict["passed"] is False
        assert any("prohibited mode" in reason
                   for reason in verdict["reasons"])

    def test_incomplete_stages_are_refused_with_named_stages(self):
        grade = full_grade()
        grade["stage_vector"]["internal_menu"] = 2
        verdict = exp_certificate.predict_canary_eligibility(
            grade_report=grade,
            artifacts_payload=sealed_preflight(),
            cost_projection=BOUNDED_COST,
            store_ceiling=9)

        assert verdict["passed"] is False
        assert any("stage internal_menu" in reason
                   for reason in verdict["reasons"])

    def test_ambiguous_identities_are_refused(self):
        verdict = exp_certificate.predict_canary_eligibility(
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(ambiguous=3),
            cost_projection=BOUNDED_COST,
            store_ceiling=9)

        assert verdict["passed"] is False
        assert any("ambiguous" in reason for reason in verdict["reasons"])

    def test_outcome_data_is_structurally_excluded(self):
        with pytest.raises(TypeError):
            exp_certificate.predict_canary_eligibility(
                grade_report=full_grade(),
                artifacts_payload=sealed_preflight(),
                cost_projection=BOUNDED_COST,
                store_ceiling=9,
                live_report={"questions_passed": 22})


class TestIssuedCertificates:
    def test_paid_types_require_strong_fingerprint_and_clean_runtime(self):
        weak = exp_certificate.issue(
            certificate_type="paid-canary-eligibility",
            verdict={"passed": True, "reasons": []},
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(strong=False),
            cost_projection=BOUNDED_COST,
            issued_at="2026-08-12T00:00:00Z")
        assert weak["passed"] is False
        assert any("strong database" in reason
                   for reason in weak["reasons"])

        dirty = exp_certificate.issue(
            certificate_type="paid-canary-eligibility",
            verdict={"passed": True, "reasons": []},
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(
                untracked=["library/document_scope.py"]),
            cost_projection=BOUNDED_COST,
            issued_at="2026-08-12T00:00:00Z")
        assert dirty["passed"] is False
        assert any("uncommitted runtime" in reason
                   for reason in dirty["reasons"])

    def test_certificates_are_sealed_and_tamper_evident(self, tmp_path):
        import json

        certificate = exp_certificate.issue(
            certificate_type="post-selection-preservation",
            verdict={"passed": True, "reasons": []},
            grade_report=full_grade(),
            artifacts_payload=sealed_preflight(),
            cost_projection=BOUNDED_COST,
            issued_at="2026-08-12T00:00:00Z")
        path = tmp_path / "certificate.json"
        path.write_text(json.dumps(certificate))
        loaded = exp_certificate.load_certificate(path)
        assert loaded["type"] == "post-selection-preservation"

        certificate["passed"] = True
        certificate["reasons"] = []
        certificate["cost_projection"] = {"usd_per_question": 0.01}
        path.write_text(json.dumps(certificate))
        with pytest.raises(ValueError, match="modified after sealing"):
            exp_certificate.load_certificate(path)


class TestBacktest:
    def test_known_failed_runs_are_true_negatives_with_stated_coverage(
            self):
        table = exp_backtest.calibration_table(rescore=False)

        assert table["counts"]["false_positive"] == 0
        present = [row for row in exp_registry.registry()
                   if row["answers_present"]]
        if present:
            assert table["counts"]["true_negative"] >= len(present)
            assert table["coverage"]["backtestable"] >= len(present)
        else:
            # A clean archive carries no saved paid artifacts: every entry
            # must be honestly not_backtestable, never a fabricated verdict.
            assert table["counts"]["not_backtestable"] == len(
                table["rows"])
        assert "backtestable" in table["coverage"]["statement"]

    def test_missing_artifacts_are_not_backtestable_never_fabricated(self):
        rows = {row["run_id"]: row for row in exp_registry.registry()}
        for row in rows.values():
            if not row["answers_present"]:
                assert row["backtest_level"] == "not_backtestable"
