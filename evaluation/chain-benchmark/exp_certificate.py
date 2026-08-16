"""Certificates: sealed judgments about what a build has earned.

Three distinct types with distinct authority. ``paid-canary-eligibility``
says only that free preflight evidence bounds cost and covers proof well
enough that buying a representative canary yields information — it never
predicts 22/22. ``paid-full22-eligibility`` additionally requires a
fresh, passing representative live result. ``post-selection-preservation``
is diagnostic and can never authorize spending.

The prediction function accepts NO outcome data — no answers, no live
report, no observed scores. Backtesting joins predictions with observed
outcomes strictly after both exist, which is what keeps a false
confidence from certifying itself.
"""
from __future__ import annotations

import json
from pathlib import Path

from exp_seal import seal, verify_seal

CANARY_QUESTIONS = (4, 6, 67, 147)

LIMITS = {
    "max_selector_cards": 64,
    "max_selector_tokens": 8000,
    "max_input_tokens_per_question": 30000,
    "max_model_calls_per_question": 3,
    "max_usd_per_question": 0.30,
}

PROHIBITED_MODES = ("oracle", "gold-steered", "deterministic-fallback")


def predict_canary_eligibility(*, grade_report: dict,
                               artifacts_payload: dict,
                               cost_projection: dict,
                               store_ceiling: int) -> dict:
    """Free-preflight prediction. Outcome data is structurally excluded:
    the signature accepts none, and callers cannot smuggle it in through
    these three inputs (a sealed artifact, its grade, and a projection).
    """
    verify_seal(artifacts_payload)
    if artifacts_payload.get("mode") in PROHIBITED_MODES:
        return _verdict(False, [
            f"prohibited mode {artifacts_payload.get('mode')!r} cannot "
            "certify a paid run"])
    reasons = []
    vector = grade_report["stage_vector"]
    for stage in ("internal_menu", "retained", "materialized",
                  "ledger", "final"):
        if vector.get(stage, 0) < store_ceiling:
            reasons.append(
                f"stage {stage} at {vector.get(stage, 0)}/"
                f"{store_ceiling} store-recoverable claims")
    per_question = cost_projection.get("usd_per_question")
    if per_question is None:
        reasons.append("no cost projection")
    elif per_question > LIMITS["max_usd_per_question"]:
        reasons.append(
            f"projected ${per_question}/question exceeds "
            f"${LIMITS['max_usd_per_question']}")
    calls = cost_projection.get("calls_per_question")
    if calls is not None and calls > LIMITS[
            "max_model_calls_per_question"]:
        reasons.append(
            f"{calls} projected calls/question exceeds "
            f"{LIMITS['max_model_calls_per_question']}")
    ambiguous = sum(
        (question.get("resolution_census") or {}).get("ambiguous", 0)
        for question in artifacts_payload.get("questions", ()))
    if ambiguous:
        reasons.append(
            f"{ambiguous} ambiguous occurrence identities; certification "
            "requires exact resolution")
    return _verdict(not reasons, reasons or ["all canary gates satisfied"])


def _verdict(passed: bool, reasons: list) -> dict:
    return {"passed": bool(passed), "reasons": list(reasons)}


def issue(*, certificate_type: str, verdict: dict, grade_report: dict,
          artifacts_payload: dict, cost_projection: dict,
          issued_at: str) -> dict:
    """A sealed certificate; expiry is any change to build/store identity."""
    provenance = artifacts_payload.get("provenance") or {}
    fingerprint = provenance.get("database_fingerprint") or {}
    if certificate_type in ("paid-canary-eligibility",
                            "paid-full22-eligibility"):
        if fingerprint.get("level") != "strong":
            verdict = _verdict(False, [
                *verdict["reasons"],
                "paid certification requires the strong database "
                "fingerprint"])
        manifest = provenance.get("runtime_manifest") or {}
        if manifest.get("untracked_runtime_files"):
            verdict = _verdict(False, [
                *verdict["reasons"],
                "paid certification refuses uncommitted runtime files: "
                + ", ".join(manifest["untracked_runtime_files"][:5])])
    certificate = {
        "schema": "ariadne-certificate-v1",
        "type": certificate_type,
        "issued_at": issued_at,
        "expiry_policy": "invalid on any build, store, question, or "
                         "price-config change",
        "passed": verdict["passed"],
        "reasons": verdict["reasons"],
        "prohibited_modes": list(PROHIBITED_MODES),
        "limits": dict(LIMITS),
        "stage_vector": grade_report.get("stage_vector"),
        "runtime_manifest_sha256": (
            (provenance.get("runtime_manifest") or {}).get(
                "manifest_sha256")),
        "database_fingerprint": fingerprint,
        "questions_sha256": (
            (provenance.get("command") or {}).get("questions")
            and provenance.get("questions_sha256")),
        "embedding_cache_sha256": provenance.get("embedding_cache_sha256"),
        "artifact_digest": (artifacts_payload.get("seal") or {}).get(
            "artifact_digest"),
        "grade_sha256": _digest(grade_report),
        "cost_projection": cost_projection,
        "provenance": {"issued_from": "exp_certificate.issue"},
    }
    return seal(certificate)


def _digest(value) -> str:
    import hashlib

    return hashlib.sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        default=str).encode()).hexdigest()


def load_certificate(path) -> dict:
    certificate = json.loads(Path(path).read_text())
    verify_seal(certificate)
    if certificate.get("schema") != "ariadne-certificate-v1":
        raise ValueError("unsupported certificate schema")
    return certificate
