"""Artifact sealing: a grade is only as trustworthy as its input.

Two digests with distinct jobs. The ARTIFACT digest covers the complete
payload — timings, costs, usage, prompts, configuration, and evidence —
so any post-seal modification invalidates it: timing and cost are
acceptance evidence, not decoration. The EVIDENCE digest exists solely
for deterministic semantic comparison across runs: it normalizes an
explicit, schema-versioned list of volatile JSON paths and nothing else —
no field is ever excluded merely because its key is named like a timer.
"""
from __future__ import annotations

import copy
import hashlib
import json

SEAL_SCHEMA = "ariadne-experiment-seal-v3"
ARTIFACT_ALGORITHM = "sha256-canonical-json-v3"
EVIDENCE_ALGORITHM = "sha256-evidence-normalized-v1"
EVIDENCE_VOLATILE_PATHS_V1 = (
    ("questions", "*", "timings"),
    # The raw invocation line varies per run (an --out path is not
    # evidence); the command's semantic identity stays under the
    # structured provenance.command fields, and the artifact digest
    # still covers argv byte-for-byte.
    ("provenance", "command", "argv"),
)


def _canonical_json(value) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":")).encode()


def _without_seal(payload: dict) -> dict:
    return {key: value for key, value in payload.items() if key != "seal"}


def artifact_digest(payload: dict) -> str:
    """Complete-content digest; any modification invalidates it."""
    return hashlib.sha256(
        _canonical_json(_without_seal(payload))).hexdigest()


def evidence_digest(payload: dict) -> str:
    """Semantic digest: identical evidence hashes equal across runs."""
    normalized = copy.deepcopy(_without_seal(payload))
    for path in EVIDENCE_VOLATILE_PATHS_V1:
        _remove_wildcard(normalized, path)
    return hashlib.sha256(_canonical_json(normalized)).hexdigest()


def _remove_wildcard(value, path) -> None:
    head, rest = path[0], path[1:]
    if head == "*":
        if isinstance(value, list):
            children = value
        elif isinstance(value, dict):
            children = list(value.values())
        else:
            return
        for child in children:
            if rest:
                _remove_wildcard(child, rest)
        return
    if not isinstance(value, dict) or head not in value:
        return
    if rest:
        _remove_wildcard(value[head], rest)
    else:
        value.pop(head)


def seal(payload: dict) -> dict:
    """Return the payload with a v3 seal. Provenance is mandatory."""
    if not payload.get("provenance"):
        raise ValueError(
            "refusing to seal an artifact without build provenance")
    sealed = dict(payload)
    sealed["seal"] = {
        "schema": SEAL_SCHEMA,
        "algorithm": ARTIFACT_ALGORITHM,
        "artifact_digest": artifact_digest(payload),
        "evidence_algorithm": EVIDENCE_ALGORITHM,
        "evidence_digest": evidence_digest(payload),
    }
    return sealed


def verify_seal(payload: dict) -> None:
    """Raise ValueError unless the payload carries a valid v3 seal."""
    stamp = payload.get("seal")
    if not stamp:
        raise ValueError("artifact is unsealed; grading refused")
    if stamp.get("schema") != SEAL_SCHEMA:
        raise ValueError(
            f"unsupported seal schema {stamp.get('schema')!r}; refused")
    if stamp.get("algorithm") != ARTIFACT_ALGORITHM:
        raise ValueError(
            f"unknown seal algorithm {stamp.get('algorithm')!r}; refused")
    if stamp.get("evidence_algorithm") != EVIDENCE_ALGORITHM:
        raise ValueError(
            "unknown evidence algorithm "
            f"{stamp.get('evidence_algorithm')!r}; refused")
    if not payload.get("provenance"):
        raise ValueError("artifact carries no build provenance; refused")
    if stamp.get("artifact_digest") != artifact_digest(payload):
        raise ValueError(
            "artifact was modified after sealing; grading refused")
    if stamp.get("evidence_digest") != evidence_digest(payload):
        raise ValueError(
            "artifact evidence was modified after sealing; grading refused")
