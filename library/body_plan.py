"""Monotonic definition-body planning: required proof outranks all noise.

Optional retrieval growth must never change which proof-bearing bodies
materialize. The plan derives *required* bodies deterministically — from
obligation-bound occurrences, the narrowest caller extent containing each
retained compiler transition site, the callee definitions of retained
causal transitions, and implementations reached through ``implements`` —
at occurrence-extent identity, so overloads sharing one qualified name
stay distinct. Required bodies are admitted before any optional cap;
a required body that cannot be admitted becomes an explicit gap, never a
silent deletion. Discovery provenance such as ``obligation_continuation``
is optional by itself, but a continuation hop participating in a retained
transition is mandatory like any other caller.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from library.selection_policy import CapEvent

_CAUSAL_RELATIONS = frozenset({"calls", "implements"})


@dataclass(frozen=True, order=True)
class BodyRef:
    source: str
    canonical_id: str
    qualified_name: str
    file: str
    line_start: int
    line_end: int


@dataclass(frozen=True, order=True)
class BodyRequirement:
    body_ref: BodyRef
    reason: str
    facet_ids: tuple = ()
    obligation_ids: tuple = ()
    transition_occurrence: "tuple | None" = None


@dataclass(frozen=True)
class DefinitionBodyPlan:
    required: tuple = ()
    optional: tuple = ()
    selected: tuple = ()
    discarded: tuple = ()
    cap_events: tuple = ()
    gaps: tuple = ()


def derive_definition_body_plan(
        *, hops, retained_symbols, bindings, facet_requirements=(),
        optional_cap: "int | None" = None,
        required_cap: "int | None" = None) -> DefinitionBodyPlan:
    """Derive the deterministic body plan for one selected evidence set."""
    retained = {str(name) for name in retained_symbols}
    candidates: list[BodyRef] = []
    seen_refs: set = set()
    for hop in hops:
        citation = getattr(hop, "citation", hop)
        if citation.line_end <= citation.line_start:
            continue
        ref = BodyRef(
            source=str(citation.source_name),
            canonical_id="",
            qualified_name=str(citation.qualified_name),
            file=str(citation.file),
            line_start=int(citation.line_start),
            line_end=int(citation.line_end))
        if ref not in seen_refs:
            seen_refs.add(ref)
            candidates.append(ref)
    candidates.sort()

    requirements: set = set()

    def narrowest_containing(file: str, line: int) -> "BodyRef | None":
        containing = [
            ref for ref in candidates
            if ref.file == file and ref.line_start <= line <= ref.line_end]
        if not containing:
            return None
        return min(
            containing,
            key=lambda ref: (ref.line_end - ref.line_start,
                             ref.line_start, ref.qualified_name))

    for hop in hops:
        citation = getattr(hop, "citation", hop)
        name = str(citation.qualified_name)
        parent = str(citation.parent_qualified_name or "")
        site_file = str(citation.call_site_file or "")
        site_line = int(citation.call_site_line or 0)
        if not parent or not site_file or not site_line:
            continue
        if name not in retained:
            continue
        caller = narrowest_containing(site_file, site_line)
        if caller is not None:
            requirements.add(BodyRequirement(
                body_ref=caller,
                reason=f"transition-caller:{site_file}:{site_line}",
                transition_occurrence=(site_file, site_line)))
        if str(citation.relation) in _CAUSAL_RELATIONS:
            if citation.line_end > citation.line_start:
                requirements.add(BodyRequirement(
                    body_ref=BodyRef(
                        source=str(citation.source_name), canonical_id="",
                        qualified_name=name, file=str(citation.file),
                        line_start=int(citation.line_start),
                        line_end=int(citation.line_end)),
                    reason="causal-callee",
                    transition_occurrence=(site_file, site_line)))

    bound_by_symbol: dict = {}
    for obligation, symbol in bindings:
        bound_by_symbol.setdefault(str(symbol), []).append(int(obligation))
    for ref in candidates:
        obligations = bound_by_symbol.get(ref.qualified_name)
        if obligations and ref.qualified_name in retained:
            requirements.add(BodyRequirement(
                body_ref=ref,
                reason="obligation-bound:" + ",".join(
                    f"C{number}" for number in sorted(set(obligations))),
                obligation_ids=tuple(sorted(set(obligations)))))
    for requirement in facet_requirements:
        requirements.add(requirement)

    required = tuple(sorted(requirements))
    required_refs: list[BodyRef] = []
    for requirement in required:
        if requirement.body_ref not in required_refs:
            required_refs.append(requirement.body_ref)
    required_refs.sort()

    optional = tuple(
        ref for ref in candidates if ref not in set(required_refs))

    cap_events: list = []
    gaps: list = []
    admitted_required = list(required_refs)
    if required_cap is not None and len(required_refs) > required_cap:
        admitted_required = required_refs[:max(int(required_cap), 0)]
        overflow = required_refs[len(admitted_required):]
        cap_events.append(CapEvent(
            cap="required_bodies", limit=int(required_cap),
            available=len(required_refs),
            retained=len(admitted_required), discarded=0))
        gaps.extend(
            f"required-body-overflow:{ref.qualified_name}"
            f"@{ref.file}:{ref.line_start}-{ref.line_end}"
            for ref in overflow)

    discarded: list = []
    selected = list(admitted_required)
    optional_budget = (len(optional) if optional_cap is None
                       else max(int(optional_cap), 0))
    for ref in optional:
        if optional_budget > 0:
            selected.append(ref)
            optional_budget -= 1
        else:
            discarded.append((ref, "over-cap:optional"))

    return DefinitionBodyPlan(
        required=required,
        optional=optional,
        selected=tuple(selected),
        discarded=tuple(discarded),
        cap_events=tuple(cap_events),
        gaps=tuple(gaps))
