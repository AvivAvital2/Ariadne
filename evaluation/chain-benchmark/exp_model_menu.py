"""Model-facing menu grading and claim-level preservation prediction.

``internal_menu_complete`` (graph surfaces never shown to a model) and
``model_menu_complete`` (the exact card surface a phase prompt carried)
are different measurements and must never be conflated. Each reviewed
item gets its earliest MODEL-FACING loss, and each claim a prediction —
preservable or not — computed strictly from pre-outcome replay state:
phase cards, scripted replies, selection telemetry, and the final
deterministic artifact. Observed outcomes join only afterward, in the
calibration matrix.
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent.parent))

from shadow_eval import required_items

#: Proof-role → responsible model-facing phase (§3.3 responsibility map).
PHASE_RESPONSIBILITY = {
    "symbol": "scip-symbol-select",
    "route": "scip-exact-route-select",
    "body": "scip-body-select",
}

MODEL_FACING_LOSSES = (
    "never-generated", "internal-only", "removed-before-prompt",
    "removed-by-cap", "visible-not-selected", "selection-reply-truncated",
    "selected-then-pruned", "body-card-absent",
    "body-visible-not-selected", "materialization", "ledger",
    "finalization")


def phase_by_name(artifact: dict) -> dict:
    phases: dict = {}
    for record in artifact.get("phases", ()):
        phases.setdefault(record["phase"], []).append(record)
    return phases


def _symbol_cards(phases) -> dict:
    """S-label -> exact card text (the qualified name, by menu design)."""
    cards = {}
    for record in phases.get("scip-symbol-select", ()):
        for label, text in record.get("cards", ()):
            if label.startswith("S"):
                cards[label] = text
    return cards


def item_lifecycle(artifact: dict, symbol: str) -> dict:
    """Earliest model-facing loss for one reviewed symbol identity."""
    phases = phase_by_name(artifact)
    internal = {
        name for route in artifact.get(
            "route_candidates", {}).values() for name in route}
    symbol_records = phases.get("scip-symbol-select", ())
    cards = _symbol_cards(phases)
    matching = [label for label, text in cards.items() if text == symbol]
    at_cap = any(record.get("at_cap") for record in symbol_records)
    selected_ids = {
        token for record in symbol_records
        for token in record.get("parsed_selected_ids", ())}
    truncated = any(
        record.get("truncated") for record in symbol_records)
    retained = symbol in set(artifact.get("selected_symbols", ()))
    hydrated = symbol in set(artifact.get("hydrated_symbols", ()))
    in_answer = symbol.rsplit(".", 1)[-1] in (
        artifact.get("answer") or "")

    row = {
        "symbol": symbol,
        "internal_candidate": symbol in internal,
        "first_visible_phase": (
            "scip-symbol-select" if matching else None),
        "card_ids": matching,
        "visible_before_cap": bool(matching),
        "model_selected": any(
            label in selected_ids for label in matching),
        "retained": retained,
        "hydrated": hydrated,
        "in_final_artifact": in_answer,
    }
    if not matching:
        if symbol in internal:
            row["earliest_model_facing_loss"] = (
                "removed-by-cap" if at_cap else "internal-only")
        else:
            row["earliest_model_facing_loss"] = "never-generated"
    elif not row["model_selected"] and not retained:
        row["earliest_model_facing_loss"] = (
            "selection-reply-truncated" if truncated
            else "visible-not-selected")
    elif not retained:
        row["earliest_model_facing_loss"] = "selected-then-pruned"
    elif not hydrated:
        row["earliest_model_facing_loss"] = "materialization"
    elif not in_answer:
        row["earliest_model_facing_loss"] = "finalization"
    else:
        row["earliest_model_facing_loss"] = None
    return row


def model_menu_complete(artifact: dict, items: dict) -> dict:
    """Every required symbol on an exact card actually sent to the model."""
    rows = [item_lifecycle(artifact, symbol)
            for symbol in sorted(items["symbols"])]
    visible = [row for row in rows if row["first_visible_phase"]]
    return {
        "model_menu_complete": bool(rows) and len(visible) == len(rows),
        "items": rows,
    }


def predict_claim_preservable(artifact: dict, items: dict) -> dict:
    """Pre-outcome prediction: all required identities survive to the
    final deterministic artifact, and every witness fragment is present
    in it. Consumes replay state only — never observed scores."""
    lifecycle = [item_lifecycle(artifact, symbol)
                 for symbol in sorted(items["symbols"])]
    losses = [row for row in lifecycle
              if row["earliest_model_facing_loss"] is not None]
    answer = artifact.get("answer") or ""
    fragments = [
        fragment for witness in items["witnesses"]
        for fragment in witness["contains"]]
    missing_fragments = [
        fragment for fragment in fragments if fragment not in answer]
    preservable = not losses and not missing_fragments
    return {
        "predicted_preservable": preservable,
        "item_losses": [
            {"symbol": row["symbol"],
             "loss": row["earliest_model_facing_loss"]}
            for row in losses],
        "missing_fragments_total": len(missing_fragments),
        "fragments_total": len(fragments),
    }


def calibrate(replay_payload: dict, gold: dict, observed: dict) -> dict:
    """Claim-level confusion matrix: sealed predictions vs rescored truth.

    ``observed`` maps (question_id, claim_id) -> bool from the CURRENT
    scorer over saved answers; it enters only here, after prediction.
    """
    artifacts = {int(row["id"]): row
                 for row in replay_payload["questions"]}
    counts = {"true_positive": 0, "false_positive": 0,
              "true_negative": 0, "false_negative": 0,
              "not_backtestable": 0}
    rows = []
    for question in gold["questions"]:
        question_id = int(question["id"])
        artifact = artifacts.get(question_id)
        for claim in question.get("claims", ()):
            key = (question_id, str(claim.get("id")))
            record = {"question": question_id, "claim": key[1]}
            if artifact is None or key not in observed:
                record["classification"] = "not_backtestable"
                counts["not_backtestable"] += 1
                rows.append(record)
                continue
            prediction = predict_claim_preservable(
                artifact, required_items(claim))
            record["predicted_preservable"] = (
                prediction["predicted_preservable"])
            record["item_losses"] = prediction["item_losses"]
            record["observed_passed"] = observed[key]
            predicted = prediction["predicted_preservable"]
            actual = observed[key]
            if predicted and actual:
                classification = "true_positive"
            elif predicted and not actual:
                classification = "false_positive"
            elif not predicted and not actual:
                classification = "true_negative"
            else:
                classification = "false_negative"
            record["classification"] = classification
            counts[classification] += 1
            rows.append(record)
    graded = sum(value for name, value in counts.items()
                 if name != "not_backtestable")
    positives = counts["true_positive"] + counts["false_negative"]
    negatives = counts["true_negative"] + counts["false_positive"]
    return {
        "counts": counts,
        "coverage": {
            "claims": len(rows), "graded": graded,
            "not_backtestable": counts["not_backtestable"]},
        "precision": (
            counts["true_positive"]
            / max(counts["true_positive"] + counts["false_positive"], 1)),
        "recall": counts["true_positive"] / max(positives, 1),
        "specificity": counts["true_negative"] / max(negatives, 1),
        "claims": rows,
    }
