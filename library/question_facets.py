"""Deterministic question facets: exact spans, never paraphrases.

A facet is a span of the question itself — an identifier, a comparison
side, an ordered clause — with generically ranked roles. Facets are the
planner-independent representation retrieval seeds from and completeness
grades against: a planner that silently narrows the question cannot
erase a facet, because facets never pass through a model. No domain
vocabulary participates; role ranking uses generic English verbs only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_BACKTICKED = re.compile(r"`([^`]+)`")
_DOTTED = re.compile(r"\b[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+\b")
_CAMEL = re.compile(
    r"\b(?:[A-Z][a-z0-9]+(?:[A-Z][a-z0-9]*)+"
    r"|[a-z]+(?:[A-Z][a-z0-9]*)+)\b")
_SNAKE = re.compile(r"\b[a-z]+(?:_[a-z0-9]+)+\b")
_BETWEEN = re.compile(
    r"\bbetween\s+(.+?)\s+and\s+(.+?)(?=[,.?;]|\s+when\b|\s+while\b|$)",
    re.IGNORECASE)
_VERSUS = re.compile(
    r"(\S+(?:\s+\S+)?)\s+(?:versus|vs\.?|compared\s+(?:to|with))\s+"
    r"(\S+(?:\s+\S+)?)", re.IGNORECASE)
_ORDERING = re.compile(
    r"\b(before|after|then|first|finally|once)\b", re.IGNORECASE)

_ROLE_VERBS = {
    "entry": ("register", "start", "invoke", "trigger", "enter", "install"),
    "decision": ("decide", "detect", "determine", "check", "validate",
                 "whether", "choose"),
    "transformation": ("rewrite", "transform", "convert", "build",
                       "generate", "derive"),
    "handoff": ("delegate", "pass", "route", "dispatch", "forward"),
    "terminal": ("write", "commit", "emit", "persist", "return", "final"),
}


@dataclass(frozen=True)
class QuestionFacet:
    id: str
    exact_text: str
    start: int
    end: int
    kind: str
    identifiers: tuple[str, ...]
    roles: tuple[str, ...]


def extract_question_facets(question: str) -> tuple[QuestionFacet, ...]:
    """Extract exact-span facets; deterministic; no paraphrasing."""
    text = str(question or "")
    spans: list[tuple[int, int, str, tuple[str, ...]]] = []

    for match in _BACKTICKED.finditer(text):
        spans.append((match.start(1), match.end(1), "identifier",
                      (match.group(1),)))
    for pattern in (_DOTTED, _CAMEL, _SNAKE):
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), "identifier",
                          (match.group(0),)))

    for match in _BETWEEN.finditer(text):
        for group in (1, 2):
            spans.append((
                match.start(group), match.end(group), "comparison-side",
                _identifiers_in(match.group(group))))
    for match in _VERSUS.finditer(text):
        for group in (1, 2):
            spans.append((
                match.start(group), match.end(group), "comparison-side",
                _identifiers_in(match.group(group))))

    for match in _ORDERING.finditer(text):
        spans.append((match.start(), match.end(), "ordering", ()))

    # Deduplicate identifier spans contained inside an identical wider
    # identifier span (backticked vs bare); keep every distinct kind.
    ordered = sorted(set(spans))
    kept: list[tuple[int, int, str, tuple[str, ...]]] = []
    for span in ordered:
        start, end, kind, identifiers = span
        if kind == "identifier" and any(
                other_start <= start and end <= other_end
                and (other_start, other_end) != (start, end)
                for other_start, other_end, other_kind, _ in ordered
                if other_kind == "identifier"):
            continue
        kept.append(span)

    lowered = text.lower()
    sequence_present = bool(_ORDERING.search(text))
    facets = []
    for index, (start, end, kind, identifiers) in enumerate(kept, start=1):
        roles = []
        for role, verbs in _ROLE_VERBS.items():
            if any(verb in lowered for verb in verbs):
                roles.append(role)
        if kind == "comparison-side":
            roles.append("comparison-side")
        if kind == "ordering" or (kind == "identifier" and sequence_present):
            roles.append("sequence")
        facets.append(QuestionFacet(
            id=f"F{index}",
            exact_text=text[start:end],
            start=start,
            end=end,
            kind=kind,
            identifiers=tuple(identifiers),
            roles=tuple(sorted(set(roles)))))
    return tuple(facets)


def _identifiers_in(text: str) -> tuple[str, ...]:
    found = []
    for pattern in (_DOTTED, _CAMEL, _SNAKE):
        for match in pattern.finditer(text):
            if match.group(0) not in found:
                found.append(match.group(0))
    return tuple(found)
