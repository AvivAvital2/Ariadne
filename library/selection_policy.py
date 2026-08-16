"""Typed monotonic selection: promotion, never destructive intersection.

A model reply promotes candidates into the preferred set; obligation and
connector bindings are authoritative and never gated by token overlap;
nothing leaves the retained set without a recorded reason; a truncated or
partially parseable reply is incomplete and fails open instead of hard
deleting unnamed evidence. Growth is bounded by counts, never character
ceilings, and soft limits report overflow instead of silently trimming.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

_TRUNCATION_FINISH_REASONS = frozenset(
    {"length", "max_tokens", "max_output_tokens"})


@dataclass(frozen=True)
class OccurrenceRef:
    """Canonical occurrence identity: module-safe, never a bare name."""

    source_name: str
    canonical_id: str
    file: str
    line_start: int
    line_end: int
    site: str = ""


@dataclass(frozen=True)
class CompletionSignal:
    """Provider metadata for one selection reply, consumed as recorded."""

    finish_reason: str | None = None
    output_tokens: int | None = None
    max_tokens: int | None = None
    schema_valid: bool = True
    unknown_ids: tuple[str, ...] = ()
    partial_trailing_token: str | None = None

    @property
    def truncated(self) -> bool:
        # A reply cut at the output cap can still report finish_reason
        # "stop"; the token counts are the authoritative signal.
        if self.finish_reason in _TRUNCATION_FINISH_REASONS:
            return True
        return (
            self.output_tokens is not None
            and self.max_tokens is not None
            and self.output_tokens >= self.max_tokens)

    @property
    def malformed(self) -> bool:
        return (
            not self.schema_valid
            or self.partial_trailing_token is not None)


@dataclass(frozen=True)
class CapEvent:
    """A soft limit was exceeded; evidence was kept and the event reported."""

    cap: str
    limit: int
    available: int
    retained: int
    discarded: int


@dataclass(frozen=True, eq=True)
class SelectionDecision:
    candidates: tuple[OccurrenceRef, ...]
    model_preferred: tuple[OccurrenceRef, ...]
    obligation_required: Mapping[str, tuple[OccurrenceRef, ...]]
    connector_required: tuple[OccurrenceRef, ...]
    reserve: tuple[OccurrenceRef, ...]
    retained: tuple[OccurrenceRef, ...]
    discarded: Mapping[OccurrenceRef, str]
    completion_status: str
    cap_events: tuple[CapEvent, ...] = ()


def classify_reply_tokens(
    tokens: Sequence[str],
    valid_ids: Collection[str],
) -> tuple[tuple[str, ...], tuple[str, ...], str | None]:
    """Split reply tokens into (known, unknown, partial_trailing_token).

    A trailing token that is a strict prefix of a valid id is the
    signature of a reply cut mid-token and must never be silently
    dropped as merely unknown.
    """
    ordered_valid = list(valid_ids)
    valid_set = set(ordered_valid)
    known: list[str] = []
    unknown: list[str] = []
    partial: str | None = None
    for position, token in enumerate(tokens):
        if token in valid_set:
            known.append(token)
        elif position == len(tokens) - 1 and any(
                candidate.startswith(token) and candidate != token
                for candidate in ordered_valid):
            partial = token
        else:
            unknown.append(token)
    return tuple(known), tuple(unknown), partial


def resolve_selection(
    *,
    candidates: Sequence[OccurrenceRef],
    model_preferred: Collection[OccurrenceRef],
    obligation_required: Mapping[str, Sequence[OccurrenceRef]],
    connector_required: Collection[OccurrenceRef] = (),
    completion: CompletionSignal,
    reserve_limit: int = 8,
    retained_soft_cap: int | None = None,
    select_all_on_incomplete: bool = False,
) -> SelectionDecision:
    """Resolve one selection phase into a typed monotonic decision.

    retained = model_preferred ∪ obligation_required ∪ connector_required;
    the unretained remainder stays selectable in a bounded reserve; every
    drop beyond the reserve carries a reason. An incomplete reply
    (truncated, malformed, or empty) never converts into hard deletion:
    with select_all_on_incomplete the caller has judged the menu
    proof-scoped and bounded, so every card is retained.
    """
    ordered_candidates = _dedup(candidates)
    preferred = _dedup(model_preferred)
    obligation_map = {
        obligation_id: _dedup(targets)
        for obligation_id, targets in obligation_required.items()}
    connectors = _dedup(connector_required)

    status = _completion_status(completion, preferred)

    if status != "ok" and select_all_on_incomplete:
        retained = ordered_candidates
        reserve: tuple[OccurrenceRef, ...] = ()
        discarded: dict[OccurrenceRef, str] = {}
    else:
        required = set(preferred) | set(connectors)
        for targets in obligation_map.values():
            required.update(targets)
        retained_in_pool = [
            occurrence for occurrence in ordered_candidates
            if occurrence in required]
        pool_identities = set(ordered_candidates)
        out_of_pool = [
            occurrence
            for occurrence in _dedup(
                list(preferred) + [
                    target
                    for targets in obligation_map.values()
                    for target in targets
                ] + list(connectors))
            if occurrence not in pool_identities]
        retained = tuple(retained_in_pool + out_of_pool)
        remainder = [
            occurrence for occurrence in ordered_candidates
            if occurrence not in required]
        reserve = tuple(remainder[:reserve_limit])
        discarded = {
            occurrence: "over-cap:reserve"
            for occurrence in remainder[reserve_limit:]}

    cap_events: list[CapEvent] = []
    if retained_soft_cap is not None and len(retained) > retained_soft_cap:
        cap_events.append(CapEvent(
            cap="retained_soft_cap",
            limit=retained_soft_cap,
            available=len(ordered_candidates),
            retained=len(retained),
            discarded=len(discarded)))

    return SelectionDecision(
        candidates=ordered_candidates,
        model_preferred=preferred,
        obligation_required=obligation_map,
        connector_required=connectors,
        reserve=reserve,
        retained=retained,
        discarded=discarded,
        completion_status=status,
        cap_events=tuple(cap_events))


def _completion_status(
    signal: CompletionSignal,
    preferred: tuple[OccurrenceRef, ...],
) -> str:
    if signal.truncated:
        return "truncated"
    if signal.malformed:
        return "malformed"
    if not preferred:
        return "empty"
    return "ok"


def _dedup(
    occurrences: Collection[OccurrenceRef],
) -> tuple[OccurrenceRef, ...]:
    seen: set[OccurrenceRef] = set()
    ordered: list[OccurrenceRef] = []
    for occurrence in occurrences:
        if occurrence not in seen:
            seen.add(occurrence)
            ordered.append(occurrence)
    return tuple(ordered)


def signal_from_usage(
    usage: Mapping | None,
    *,
    max_tokens: int | None = None,
) -> CompletionSignal:
    """Build a CompletionSignal from a recorded per-completion usage row.

    The row is the provider usage dict recorded for one completion (see
    llm.chat_complete): token counts plus, when the provider reports one,
    a ``stop_reason``. The request cap falls back to the row's own
    ``max_tokens`` when the caller does not supply it. A missing row
    yields a signal that can never claim truncation — unknowable is not
    evidence of completeness, and callers must not treat it as such.
    """
    row = dict(usage or {})
    cap = max_tokens if max_tokens is not None else row.get("max_tokens")
    return CompletionSignal(
        finish_reason=row.get("stop_reason"),
        output_tokens=row.get("output_tokens"),
        max_tokens=cap)


def trailing_menu_token(reply: str, valid_ids: Collection[str]) -> str | None:
    """Return the reply's trailing token when it was cut mid-label.

    A trailing token that is a strict prefix of a menu id (``B7, B8, B``
    against B-labels) is the signature of a reply truncated mid-token; a
    token that is itself a valid id stays ambiguous and is left to the
    usage metadata to judge.
    """
    match = re.search(r"([A-Za-z]+\d*)\s*$", str(reply or ""))
    if match is None:
        return None
    token = match.group(1)
    identifiers = [str(identifier) for identifier in valid_ids]
    if token in identifiers:
        return None
    if any(identifier.startswith(token) and identifier != token
           for identifier in identifiers):
        return token
    return None
