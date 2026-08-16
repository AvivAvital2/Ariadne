"""Chain-derived confidence; retrieval similarity is not proof completeness."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConfidenceAssessment:
    level: str
    reasons: tuple[str, ...] = ()


def _count_reason(count: int, singular: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {singular}{suffix}"


def assess_chain_confidence(evidence, *, claims_total: int,
                            supported_claims: int) -> ConfidenceAssessment:
    """Grade only properties the SCIP bundle and claim ledger can demonstrate."""
    if evidence is None or not getattr(evidence, "hops", ()):
        return ConfidenceAssessment("low", ("no compiler chain",))
    unsupported = max(claims_total - supported_claims, 0)
    if unsupported:
        return ConfidenceAssessment(
            "low", (_count_reason(unsupported, 'rejected formulation claim'),))
    if claims_total == 0:
        return ConfidenceAssessment("low", ("no supported claims",))

    hops = tuple(evidence.hops)
    materialized = sum(bool(getattr(hop, "source_excerpts", ())) for hop in hops)
    reasons: list[str] = []
    if materialized != len(hops):
        reasons.append(f"source materialized for {materialized}/{len(hops)} hops")
    for attribute, noun in (("source_gaps", "source gap"),
                            ("caller_frontiers", "caller frontier"),
                            ("unresolved_paths", "unresolved path")):
        count = len(getattr(evidence, attribute, ()) or ())
        if count:
            reasons.append(_count_reason(count, noun))
    disconnected, executable_frontiers = executable_topology_gaps(evidence)
    if disconnected:
        reasons.append(_count_reason(len(disconnected), "disconnected mandatory endpoint"))
    if executable_frontiers:
        reasons.append(_count_reason(len(executable_frontiers), "executable frontier"))
    truncation = getattr(evidence, "truncation_reason", "")
    if truncation:
        reasons.append(truncation)
    return ConfidenceAssessment("low" if reasons else "high", tuple(reasons))
@dataclass(frozen=True)
class CompletenessAssessment:
    complete: bool
    reasons: tuple[str, ...] = ()
def assess_chain_completeness(evidence, *, claims_total: int = 0,
                              supported_claims: int = 0) -> CompletenessAssessment:
    """Fail closed unless the observed SCIP traversal is topologically closed."""
    if evidence is None or not getattr(evidence, "hops", ()):
        return CompletenessAssessment(False, ("no compiler chain",))
    reasons = []
    transitions = account_transitions(evidence)
    if transitions.unaccounted:
        reasons.append(_count_reason(transitions.unaccounted, "unaccounted transition"))
    hops = tuple(evidence.hops)
    materialized = sum(bool(getattr(hop, "source_excerpts", ())) for hop in hops)
    if materialized != len(hops):
        reasons.append(f"source materialized for {materialized}/{len(hops)} hops")
    for attribute, noun in (("source_gaps", "source gap"),
                            ("caller_frontiers", "caller frontier")):
        count = len(getattr(evidence, attribute, ()) or ())
        if count:
            reasons.append(_count_reason(count, noun))
    mandatory_forks = getattr(evidence, "mandatory_fan_outs", None)
    if mandatory_forks is None:
        mandatory_forks = getattr(evidence, "fan_outs", ())
    if mandatory_forks:
        reasons.append(_count_reason(len(mandatory_forks), "unresolved fork"))
    disconnected, executable_frontiers = executable_topology_gaps(evidence)
    if disconnected:
        reasons.append(_count_reason(len(disconnected), "disconnected mandatory endpoint"))
    if executable_frontiers:
        reasons.append(_count_reason(len(executable_frontiers), "executable frontier"))
    truncation = getattr(evidence, "truncation_reason", "")
    if truncation:
        reasons.append(truncation)
    return CompletenessAssessment(not reasons, tuple(reasons))
@dataclass(frozen=True)
class TransitionAccounting:
    total: int
    accounted: int

    @property
    def unaccounted(self) -> int:
        return self.total - self.accounted


def account_transitions(evidence) -> TransitionAccounting:
    total = accounted = 0
    for hop in tuple(getattr(evidence, "hops", ()) or ()):
        citation = getattr(hop, "citation", None)
        if citation is None or getattr(citation, "relation", "") == "localized":
            continue
        total += 1
        if (getattr(citation, "qualified_name", "")
                and getattr(citation, "file", "")
                and int(getattr(citation, "line_start", 0) or 0) > 0
                and getattr(citation, "parent_qualified_name", "")
                and getattr(citation, "call_site_file", "")
                and int(getattr(citation, "call_site_line", 0) or 0) > 0):
            accounted += 1
    return TransitionAccounting(total, accounted)


def assess_formulation_coverage(*, claims_total: int,
                                supported_claims: int) -> CompletenessAssessment:
    rejected = max(claims_total - supported_claims, 0)
    if rejected:
        return CompletenessAssessment(
            False, (_count_reason(rejected, "rejected formulation claim"),))
    if claims_total == 0:
        return CompletenessAssessment(False, ("no supported claims",))
    return CompletenessAssessment(True, ())
def assess_scope_completeness(evidence) -> CompletenessAssessment:
    """Report whether retrieval/localization left any unpositioned source path."""
    paths = tuple(getattr(evidence, "unresolved_paths", ()) or ()) if evidence is not None else ()
    reasons = tuple(f"unresolved positioning path: {path}" for path in paths)
    return CompletenessAssessment(not reasons, reasons)
def executable_topology_gaps(evidence) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return disconnected mandatory symbols and open depth frontiers on their routes."""
    citations = [
        getattr(hop, "citation", None)
        for hop in tuple(getattr(evidence, "hops", ()) or ())
    ]
    citations = [citation for citation in citations if citation is not None]
    mandatory = {
        citation.qualified_name for citation in citations
        if citation.relation in ("localized", "shared_reference")
    }
    adjacency: dict[str, set[str]] = {}
    for citation in citations:
        parent = getattr(citation, "parent_qualified_name", "")
        child = getattr(citation, "qualified_name", "")
        if citation.relation == "localized" or not parent or not child:
            continue
        adjacency.setdefault(parent, set()).add(child)
        adjacency.setdefault(child, set()).add(parent)
    disconnected = tuple(sorted(symbol for symbol in mandatory if symbol not in adjacency))
    reachable: set[str] = set()
    pending = [symbol for symbol in mandatory if symbol in adjacency]
    while pending:
        symbol = pending.pop()
        if symbol in reachable:
            continue
        reachable.add(symbol)
        pending.extend(adjacency.get(symbol, ()) - reachable)
    frontiers = tuple(sorted({
        citation.qualified_name for citation in citations
        if getattr(citation, "stop_reason", "") == "depth"
        and (citation.qualified_name in reachable
             or citation.parent_qualified_name in reachable)
    }))
    return disconnected, frontiers
def derive_display_confidence(*, chain_complete: bool,
                              formulation_complete: bool,
                              scope_complete: bool) -> str:
    """Aggregate independent proof dimensions without calling scope doubt an error."""
    if not chain_complete or not formulation_complete:
        return "low"
    return "high" if scope_complete else "medium"
def assess_selection_coverage(candidate_routes, selected_routes, *,
                              required_symbols=()) -> CompletenessAssessment:
    """Require a route and every mandatory root; alternatives need not all be read."""
    route_map = (dict(candidate_routes) if hasattr(candidate_routes, "items")
                 else {str(route): () for route in (candidate_routes or ())})
    selected = tuple(dict.fromkeys(selected_routes or ()))
    reasons = []
    if route_map and not selected:
        reasons.append("no candidate route selected")
    selected_symbols = {
        str(symbol) for route_id in selected
        for symbol in route_map.get(route_id, ())}
    missing = set(required_symbols or ()) - selected_symbols
    if missing:
        reasons.append(_count_reason(
            len(missing), "required route symbol missing"))
    return CompletenessAssessment(not reasons, tuple(reasons))


def assess_obligation_coverage(
        bindings, *, represented_symbols) -> CompletenessAssessment:
    """Every planned obligation needs at least one materialized bound target.

    Completeness is graded against the obligation plan, never against the
    claims the answer happens to contain: an obligation whose bound target
    symbols all failed to materialize is missing proof, whatever the
    narration says about itself.
    """
    grouped: dict = {}
    for obligation, symbol in bindings:
        grouped.setdefault(int(obligation), []).append(str(symbol))
    represented = set(represented_symbols)
    uncovered = sorted(
        number for number, symbols in grouped.items()
        if not represented.intersection(symbols))
    if uncovered:
        return CompletenessAssessment(False, tuple(
            f"obligation C{number} has no materialized proof"
            for number in uncovered))
    return CompletenessAssessment(True, ())
