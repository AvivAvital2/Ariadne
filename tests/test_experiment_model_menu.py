"""Model-facing grading: only cards actually sent to a model count.

An internal route that never reached a prompt is internal-only; a card
cut by the 500 cap is removed-by-cap; a card the model saw but did not
select is visible-not-selected unless the reply was truncated; and the
final outcome can never influence the prediction — it joins strictly at
calibration time.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "evaluation" / "chain-benchmark"
sys.path.insert(0, str(BENCH))

SPEC = importlib.util.spec_from_file_location(
    "exp_model_menu", BENCH / "exp_model_menu.py")
exp_model_menu = importlib.util.module_from_spec(SPEC)
sys.modules["exp_model_menu"] = exp_model_menu
SPEC.loader.exec_module(exp_model_menu)


def artifact(*, symbol_cards=(), selected_ids=(), at_cap=False,
             truncated=False, internal=(), retained=(), hydrated=(),
             answer=""):
    return {
        "id": 1,
        "phases": [{
            "phase": "scip-symbol-select",
            "cards": [list(card) for card in symbol_cards],
            "parsed_selected_ids": list(selected_ids),
            "at_cap": at_cap,
            "truncated": truncated,
        }],
        "route_candidates": {"R1": list(internal)},
        "selected_symbols": list(retained),
        "hydrated_symbols": list(hydrated),
        "answer": answer,
    }


class TestItemLifecycle:
    def test_internal_route_absent_from_prompt_is_internal_only(self):
        row = exp_model_menu.item_lifecycle(
            artifact(internal=["pkg.Hidden.run"]), "pkg.Hidden.run")

        assert row["internal_candidate"] is True
        assert row["first_visible_phase"] is None
        assert row["earliest_model_facing_loss"] == "internal-only"

    def test_candidate_missing_at_cap_is_removed_by_cap(self):
        row = exp_model_menu.item_lifecycle(
            artifact(internal=["pkg.Hidden.run"], at_cap=True,
                     symbol_cards=[["S1", "pkg.Other.name"]]),
            "pkg.Hidden.run")

        assert row["earliest_model_facing_loss"] == "removed-by-cap"

    def test_visible_card_not_selected(self):
        row = exp_model_menu.item_lifecycle(
            artifact(symbol_cards=[["S1", "pkg.Seen.run"]],
                     selected_ids=["S9"]),
            "pkg.Seen.run")

        assert row["first_visible_phase"] == "scip-symbol-select"
        assert row["earliest_model_facing_loss"] == "visible-not-selected"

    def test_truncated_reply_reclassifies_nonselection(self):
        row = exp_model_menu.item_lifecycle(
            artifact(symbol_cards=[["S1", "pkg.Seen.run"]],
                     selected_ids=[], truncated=True),
            "pkg.Seen.run")

        assert row["earliest_model_facing_loss"] == (
            "selection-reply-truncated")

    def test_wrong_module_same_display_never_satisfies_identity(self):
        row = exp_model_menu.item_lifecycle(
            artifact(symbol_cards=[["S1", "shadow.pkg.Seen.run"]]),
            "pkg.Seen.run")

        assert row["first_visible_phase"] is None
        assert row["earliest_model_facing_loss"] == "never-generated"

    def test_selected_then_pruned_and_survival(self):
        pruned = exp_model_menu.item_lifecycle(
            artifact(symbol_cards=[["S1", "pkg.Seen.run"]],
                     selected_ids=["S1"]),
            "pkg.Seen.run")
        assert pruned["earliest_model_facing_loss"] == (
            "selected-then-pruned")

        survived = exp_model_menu.item_lifecycle(
            artifact(symbol_cards=[["S1", "pkg.Seen.run"]],
                     selected_ids=["S1"], retained=["pkg.Seen.run"],
                     hydrated=["pkg.Seen.run"], answer="run at x"),
            "pkg.Seen.run")
        assert survived["earliest_model_facing_loss"] is None


class TestPredictionAndCalibration:
    def make_gold(self):
        return {"questions": [{
            "id": 1,
            "claims": [{
                "id": "claim-1",
                "anchors": [{"target": {"symbol": "pkg.Seen.run"}}],
                "review": {},
                "candidate_paths": [],
                "witnesses": [{
                    "file": "x.py", "line_start": 1, "line_end": 1,
                    "contains": ["run at x"]}],
            }],
        }]}

    def surviving_artifact(self):
        return artifact(
            symbol_cards=[["S1", "pkg.Seen.run"]], selected_ids=["S1"],
            retained=["pkg.Seen.run"], hydrated=["pkg.Seen.run"],
            answer="the call happens: run at x")

    def test_outcome_cannot_influence_prediction(self):
        import inspect

        parameters = inspect.signature(
            exp_model_menu.predict_claim_preservable).parameters
        assert "observed" not in parameters
        assert "claim_passed" not in parameters

        payload = {"questions": [self.surviving_artifact()]}
        gold = self.make_gold()
        optimistic = exp_model_menu.calibrate(
            payload, gold, {(1, "claim-1"): True})
        pessimistic = exp_model_menu.calibrate(
            payload, gold, {(1, "claim-1"): False})
        assert (optimistic["claims"][0]["predicted_preservable"]
                == pessimistic["claims"][0]["predicted_preservable"])

    def test_claim_level_confusion_matrix(self):
        payload = {"questions": [self.surviving_artifact()]}
        gold = self.make_gold()

        as_pass = exp_model_menu.calibrate(
            payload, gold, {(1, "claim-1"): True})
        assert as_pass["counts"]["true_positive"] == 1
        assert as_pass["recall"] == 1.0

        as_fail = exp_model_menu.calibrate(
            payload, gold, {(1, "claim-1"): False})
        assert as_fail["counts"]["false_positive"] == 1

    def test_unreplayed_claims_are_not_backtestable(self):
        gold = self.make_gold()
        table = exp_model_menu.calibrate(
            {"questions": []}, gold, {})

        assert table["counts"]["not_backtestable"] == 1

    def test_all_negative_classifier_is_flagged_by_recall(self):
        losing = artifact(symbol_cards=[], internal=["pkg.Seen.run"])
        payload = {"questions": [losing]}
        gold = self.make_gold()

        table = exp_model_menu.calibrate(
            payload, gold, {(1, "claim-1"): True})

        assert table["counts"]["false_negative"] == 1
        assert table["recall"] == 0.0
