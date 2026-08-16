# Compiler-edge relation wording shared by prompts, ledgers, and proof appendices.
from __future__ import annotations

_RELATION_VERBS = {
    "calls": "calls",
    "references": "references",
    "shared_reference": "is referenced by",
    "referenced_by": "is referenced by",
    "called_by": "is called by",
    "contains": "contains",
    "implements": "implements",
    "implemented_by": "is implemented by",
}

_RELATION_SITES = {
    "calls": "called at",
    "references": "referenced at",
    "shared_reference": "referenced by at",
    "referenced_by": "referenced by at",
    "called_by": "called by at",
    "contains": "contained at",
    "implements": "implemented at",
    "implemented_by": "implemented by at",
    "localized": "localized at",
}


def transition_verb(relation: str) -> str:
    normalized = str(relation or "").strip().lower()
    return _RELATION_VERBS.get(
        normalized, normalized.replace("_", " ") or "relates to")


def relation_site_phrase(relation: str) -> str:
    normalized = str(relation or "").strip().lower()
    fallback = f"{normalized.replace("_", " ")} at" if normalized else "related at"
    return _RELATION_SITES.get(normalized, fallback)
def edge_relation(edge_type: str) -> str:
    normalized = str(edge_type or "").strip().lower()
    mapping = {
        "call": "calls",
        "type_ref": "references",
        "implements": "implements",
        "incoming_call": "called_by",
        "incoming_type_ref": "shared_reference",
        "incoming_implements": "implemented_by",
        "contains": "contains",
    }
    return mapping.get(normalized, normalized or "references")
