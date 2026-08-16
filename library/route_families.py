"""Route-family cards: obligation-scoped causal alternatives for one selector.

Candidates are gathered independently per obligation under a fixed budget
— never one global raw pool — and clustered into source-aware families
keyed by module, endpoints, and relation sequence. Node identity is the
occurrence (name, file, extent), so overloads, companions, and module
shadows never merge by qualified name. Every obligation keeps a bounded
reserve that stays selectable when a reply is empty, truncated, or
malformed; the pool is never materialized wholesale. The selector prompt
carries cards only — no source bodies.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from library.question_facets import extract_question_facets
from library.clews import (
    _lexical_tokens,
    deterministic_clew_matches,
    rank_clew_matches,
)
from library.structural_assembly import (
    facet_symbol_seeds,
    obligation_seeded_expansion,
)

RESERVE_PER_OBLIGATION = 3


@dataclass(frozen=True)
class RouteFamilyCard:
    card_id: str
    obligation_id: str
    source: str
    module: str
    entry: str
    terminal: str
    route_ids: tuple = ()
    node_identities: tuple = ()
    node_qualified_names: tuple = ()
    relation_sequence: tuple = ()
    summary: str = ""
    retrieval_origins: tuple = ()
    required_body_refs: tuple = ()
    estimated_expansion_cost: int = 0


@dataclass(frozen=True)
class FamilyCards:
    cards: tuple = ()
    reserve_by_obligation: dict = field(default_factory=dict)
    overflow_by_obligation: dict = field(default_factory=dict)
    unresolved_obligations: tuple = ()


@dataclass(frozen=True)
class FamilySelection:
    selected_by_obligation: dict = field(default_factory=dict)
    retained_by_obligation: dict = field(default_factory=dict)
    unresolved_obligations: tuple = ()
    unknown_ids: tuple = ()


def _module_of(file: str) -> str:
    return file.split("/", 1)[0] if file else ""
def _routes_from_citations(citations) -> list:
    """Parent-linked chains from expansion citations, deterministic.

    Chains fork per child: every alternative continuation forms its own
    route (one per leaf path), because sibling call sites are exactly
    the alternatives a route-family menu exists to expose. A repeated
    name ends its path; ordering is deterministic by file, line, then
    qualified name.
    """
    children = {}
    roots = []
    for citation in citations:
        parent = citation.parent_qualified_name
        if parent:
            children.setdefault(parent, []).append(citation)
        else:
            roots.append(citation)
    routes = []
    seen = set()

    def walk(chain):
        cursor = chain[-1]
        nexts = sorted(
            children.get(cursor.qualified_name, ()),
            key=lambda c: (c.call_site_line, c.qualified_name))
        extended = False
        for candidate in nexts:
            if candidate.qualified_name in {
                    node.qualified_name for node in chain}:
                continue
            walk([*chain, candidate])
            extended = True
        if not extended:
            names = tuple(node.qualified_name for node in chain)
            key = (names, chain[0].file, chain[0].line_start)
            if key not in seen:
                seen.add(key)
                routes.append(list(chain))

    for root in sorted(roots, key=lambda c: (
            c.file, c.line_start, c.qualified_name)):
        walk([root])
    return routes


def _identity_rows(conn, source, names) -> list:
    rows = []
    for name in names:
        for row in conn.execute(
                "SELECT canonical_id, qualified_name, file, line_start, "
                "line_end FROM scip_symbols WHERE source_name = ? AND "
                "qualified_name = ? AND canonical_id NOT GLOB ? "
                "ORDER BY canonical_id",
                (source, str(name), "local *")):
            rows.append(tuple(row))
    return rows
def build_route_family_cards(
        conn, *, source: str, question: str, obligations,
        semantic_seed_ids=(), per_obligation_budget: int = 10,
        total_budget: int = 64, clew_matches = (), trace = ()) -> FamilyCards:
    """Obligation-scoped family generation under fixed budgets.

    The shared recall pool re-ranks per obligation with the committed
    deterministic lexical scorer, so different obligations surface
    different families. Families order by how strongly their OWN node
    identities overlap the question and the obligation requirement —
    a family whose names carry the asked-about concepts outranks an
    incidental neighbor regardless of which seed surfaced it — and no
    single retrieval origin may take more than two menu slots, so one
    seed's fanout cannot crowd every alternative off the menu.
    """
    facet_identifiers = tuple(dict.fromkeys(
        identifier for facet in extract_question_facets(question)
        for identifier in facet.identifiers))
    facet_ids = facet_symbol_seeds(conn, facet_identifiers, source=source)
    anchored_seed_names = [
        str(identifier) for identifier in facet_identifiers]
    _semantic_ids = [str(seed) for seed in semantic_seed_ids if seed]
    if _semantic_ids:
        anchored_seed_names.extend(
            str(row[0]) for row in conn.execute(
                "SELECT qualified_name FROM scip_symbols "
                "WHERE canonical_id IN ("
                + ",".join("?" * len(_semantic_ids)) + ")",
                _semantic_ids))

    class _Match:
        def __init__(self, symbols):
            self.clew = type("Clew", (), {"route": []})()
            self.target_symbols = tuple(
                (1, symbol) for symbol in symbols)

    all_cards = []
    reserve_by_obligation = {}
    overflow_by_obligation = {}
    unresolved = []
    label_counter = 0
    for obligation in obligations:
        obligation_id = str(obligation["id"])
        bound = [str(symbol) for symbol in obligation.get("symbols", ())]
        obligation_matches = list(clew_matches)
        if obligation_matches:
            obligation_matches = deterministic_clew_matches(
                f"{question} {obligation.get('text', '')}",
                obligation_matches, limit=12)
        obligation_tokens = set(_lexical_tokens(
            f"{question} {obligation.get('text', '')}"))
        if isinstance(trace, dict):
            trace[obligation_id] = {
                "shortlist": [
                    match.clew.id for match in obligation_matches],
                "lexical_order": [
                    match.clew.id for match in rank_clew_matches(
                        f"{question} {obligation.get('text', '')}",
                        list(clew_matches))],
            }
        seed_relevance: dict = {}
        for name in anchored_seed_names:
            seed_relevance.setdefault(name, -1)
        for match_index, match in enumerate(obligation_matches):
            for name in (
                    *(str(name)
                      for name in getattr(match.clew, "route", ())),
                    *(str(symbol) for _rank, symbol
                      in match.target_symbols if symbol)):
                seed_relevance.setdefault(name, match_index)
        default_relevance = len(obligation_matches)
        expansion = obligation_seeded_expansion(
            conn, [_Match(bound), *obligation_matches], source=source,
            question_seed_ids=tuple(facet_ids),
            catalog_seed_ids=tuple(semantic_seed_ids),
            depth=2, forward_depth=2, per_seed_limit=6, reserve_limit=8,
            preference_tokens=tuple(obligation_tokens))
        if isinstance(trace, dict):
            trace[obligation_id]["expansion_reasons"] = dict(expansion.reasons)

        families = []
        # Identity families: every exact occurrence of a bound or
        # semantic seed gets a card of its own — occurrence identity,
        # never a merged qualified name.
        seed_rows = _identity_rows(conn, source, bound)
        for canonical, qname, file, line_start, line_end in seed_rows:
            families.append({
                "module": _module_of(file),
                "entry": qname, "terminal": qname,
                "relations": (),
                "nodes": ((qname, canonical, file, line_start,
                           line_end),),
                "origins": ("identifier",),
                "rank": (0, 0, -1, 0, file, line_start),
            })
        for route in _routes_from_citations(expansion.citations):
            nodes = tuple(
                (node.qualified_name, "", node.file, node.line_start,
                 node.line_end) for node in route)
            relations = tuple(
                node.relation for node in route)
            reverse_like = any(
                relation in ("called_by", "shared_reference",
                             "implemented_by", "referenced_by")
                for relation in relations)
            relevance = min(
                (seed_relevance.get(node.qualified_name,
                                    default_relevance)
                 for node in route), default=default_relevance)
            overlap = len(obligation_tokens.intersection(
                _lexical_tokens(" ".join(
                    node.qualified_name for node in route))))
            families.append({
                "module": _module_of(route[0].file),
                "entry": route[0].qualified_name,
                "terminal": route[-1].qualified_name,
                "relations": relations,
                "nodes": nodes,
                "origins": (
                    ("reverse",) if reverse_like else ("forward",)),
                "rank": (1, -overlap, relevance,
                         1 if reverse_like else 2,
                         route[0].file, route[0].line_start),
            })

        # Cluster duplicates by causal identity, keep source diversity:
        # at most three families per (module, entry owner) so one
        # namespace cannot consume the obligation's budget.
        deduped = {}
        for family in families:
            key = (family["module"], family["entry"],
                   family["terminal"], family["relations"],
                   family["nodes"])
            if key not in deduped:
                deduped[key] = family
        ordered = sorted(deduped.values(), key=lambda f: f["rank"])
        diversity_counts = {}
        diverse = []
        spill = []
        for family in ordered:
            owner = family["entry"].rsplit(".", 1)[0]
            group = (family["module"], owner)
            diversity_counts[group] = diversity_counts.get(group, 0) + 1
            if diversity_counts[group] <= 3:
                diverse.append(family)
            else:
                spill.append(family)
                _record_family_drop(trace, obligation_id, family,
                                    "diversity-cap:3-per-module-owner")

        budget = max(int(per_obligation_budget), 1)
        chosen = []
        origin_counts: dict = {}
        overflowed = []
        for family in diverse:
            origin = family["rank"][2]
            origin_counts[origin] = origin_counts.get(origin, 0) + 1
            if origin_counts[origin] > 2:
                overflowed.append(family)
                _record_family_drop(trace, obligation_id, family,
                                    "origin-cap:2")
            elif len(chosen) >= budget:
                overflowed.append(family)
                _record_family_drop(trace, obligation_id, family,
                                    f"budget:{budget}")
            else:
                chosen.append(family)
        remainder = overflowed + spill
        if not chosen:
            unresolved.append(obligation_id)

        obligation_cards = []
        for family in chosen:
            label_counter += 1
            body_refs = tuple(sorted({
                (node[0], node[2], node[3], node[4])
                for node in (family["nodes"][0], family["nodes"][-1])}))
            obligation_cards.append(RouteFamilyCard(
                card_id=f"F{label_counter}",
                obligation_id=obligation_id,
                source=source,
                module=family["module"],
                entry=family["entry"],
                terminal=family["terminal"],
                route_ids=(f"{obligation_id}:r{label_counter}",),
                node_identities=family["nodes"],
                node_qualified_names=tuple(
                    node[0] for node in family["nodes"]),
                relation_sequence=family["relations"],
                summary=" -> ".join(
                    node[0] for node in family["nodes"]),
                retrieval_origins=family["origins"],
                required_body_refs=body_refs,
                estimated_expansion_cost=sum(
                    max(ref[3] - ref[2], 1) for ref in body_refs)))
        all_cards.extend(obligation_cards)
        reserve_ids = [card.card_id for card in obligation_cards[
            :RESERVE_PER_OBLIGATION]]
        reserve_by_obligation[obligation_id] = tuple(reserve_ids)
        overflow_by_obligation[obligation_id] = len(remainder)

    if len(all_cards) > total_budget:
        # Round-robin per obligation so trimming stays proportional.
        by_obligation = {}
        for card in all_cards:
            by_obligation.setdefault(card.obligation_id, []).append(card)
        kept = []
        cursors = {key: 0 for key in by_obligation}
        while len(kept) < total_budget:
            advanced = False
            for key in sorted(by_obligation):
                cursor = cursors[key]
                if cursor < len(by_obligation[key]) and (
                        len(kept) < total_budget):
                    kept.append(by_obligation[key][cursor])
                    cursors[key] = cursor + 1
                    advanced = True
            if not advanced:
                break
        dropped = {card.card_id for card in all_cards} - {
            card.card_id for card in kept}
        if isinstance(trace, dict):
            for card in all_cards:
                if card.card_id in dropped:
                    _record_family_drop(
                        trace, card.obligation_id,
                        {"entry": card.entry, "terminal": card.terminal},
                        f"total-budget:{total_budget}")
        for card in all_cards:
            if card.card_id in dropped:
                overflow_by_obligation[card.obligation_id] = (
                    overflow_by_obligation.get(card.obligation_id, 0) + 1)
        all_cards = kept

    return FamilyCards(
        cards=tuple(all_cards),
        reserve_by_obligation=reserve_by_obligation,
        overflow_by_obligation=overflow_by_obligation,
        unresolved_obligations=tuple(unresolved))


def render_family_selector_prompt(cards: FamilyCards, *,
                                  question: str) -> str:
    """One compact selector surface. Cards only — never source bodies."""
    lines = [
        "Select the route families that prove each obligation. Reply "
        "with one line per obligation: O<i>: F<id> F<id> ...",
        f"Question: {question}",
        "",
    ]
    current = None
    for card in cards.cards:
        if card.obligation_id != current:
            current = card.obligation_id
            lines.append(f"{current}:")
        relations = ",".join(card.relation_sequence) or "identity"
        origins = ",".join(card.retrieval_origins)
        lines.append(
            f"  {card.card_id}. [{card.module}] {card.summary} "
            f"({relations}) — via {origins}; bodies "
            f"{len(card.required_body_refs)}; "
            f"cost~{card.estimated_expansion_cost}")
    for obligation_id, count in sorted(
            cards.overflow_by_obligation.items()):
        if count:
            lines.append(
                f"  note: {obligation_id} has {count} additional "
                "families beyond this menu")
    return "\n".join(lines)
def resolve_family_selection(reply: str, cards: FamilyCards, *,
                             truncated: bool = False) -> FamilySelection:
    """One structured reply resolves every obligation, fail-open bounded.

    Only complete obligation lines parse. Unknown family ids, duplicate
    obligation records, and families claimed for the wrong obligation are
    rejected — the affected obligation falls back to its bounded reserve
    and is reported unresolved. Incidental numbers outside obligation
    lines never select anything, and a truncated reply keeps only the
    bounded reserve; nothing ever expands the whole pool.
    """
    import re

    valid = {card.card_id: card.obligation_id for card in cards.cards}
    obligations = sorted({card.obligation_id for card in cards.cards})
    selected: dict = {}
    unknown: list = []
    malformed: list = []
    seen_lines: set = set()
    for match in re.finditer(
            r"(?m)^\s*(O\d+)\s*:\s*(.*)$", str(reply or "")):
        obligation_id = match.group(1)
        if obligation_id in seen_lines:
            malformed.append(f"duplicate-line:{obligation_id}")
            selected.pop(obligation_id, None)
            continue
        seen_lines.add(obligation_id)
        if obligation_id not in obligations:
            malformed.append(f"unknown-obligation:{obligation_id}")
            continue
        chosen = []
        for token in re.findall(r"\bF\d+\b", match.group(2)):
            if token not in valid:
                unknown.append(token)
            elif valid[token] != obligation_id:
                malformed.append(
                    f"wrong-obligation:{token}->{obligation_id}")
            elif token not in chosen:
                chosen.append(token)
        if chosen:
            selected[obligation_id] = tuple(chosen)

    duplicated = {entry.split(":", 1)[1] for entry in malformed
                  if entry.startswith("duplicate-line:")}
    unresolved = []
    retained = {}
    for obligation_id in obligations:
        chosen = selected.get(obligation_id, ())
        if chosen and not truncated and obligation_id not in duplicated:
            retained[obligation_id] = tuple(chosen)
            continue
        reserve = cards.reserve_by_obligation.get(obligation_id, ())
        if chosen and truncated:
            retained[obligation_id] = tuple(dict.fromkeys(
                (*chosen, *reserve)))[:RESERVE_PER_OBLIGATION]
        else:
            retained[obligation_id] = tuple(reserve)
        unresolved.append(obligation_id)
    return FamilySelection(
        selected_by_obligation={
            key: tuple(value) for key, value in selected.items()
            if key not in duplicated},
        retained_by_obligation=retained,
        unresolved_obligations=tuple(unresolved),
        unknown_ids=tuple(dict.fromkeys((*unknown, *malformed))))
@dataclass(frozen=True)
class FamilyExpansion:
    """Deterministic output of selected families: exact identities only."""

    citations: tuple = ()
    required_body_extents: tuple = ()
    edge_sites: tuple = ()
    expanded_card_ids: tuple = ()
    gaps: tuple = ()
def expand_selected_families(conn, cards: FamilyCards,
                             selection: FamilySelection, *,
                             source: str,
                             route_nodes_per_family: int = 8,
                             definitions_per_family: int = 4,
                             total_body_extents: int = 24) -> (
        "FamilyExpansion"):
    """Family ids — never qualified names — determine what materializes.

    Every retained card contributes its exact route (compiler-verified
    consecutive edges via the occurrence-aware route materializer), its
    endpoint definition extents, and the narrowest extent containing each
    selected edge site — under separate bounds for route nodes,
    definitions per family, and total bodies. Overflow becomes an
    explicit unresolved proof gap; it never triggers select-all.
    """
    from library.structural_assembly import citations_from_qualified_routes

    retained_ids = tuple(dict.fromkeys(
        card_id
        for ids in selection.retained_by_obligation.values()
        for card_id in ids))
    by_id = {card.card_id: card for card in cards.cards}
    routes = []
    extents = []
    gaps: list = []
    for card_id in retained_ids:
        card = by_id.get(card_id)
        if card is None:
            continue
        names = list(card.node_qualified_names)
        if len(names) > route_nodes_per_family:
            gaps.append(
                f"route-nodes-overflow:{card_id}:"
                f"{len(names)}>{route_nodes_per_family}")
            names = names[:route_nodes_per_family]
        if len(names) > 1:
            routes.append(names)
        family_extents = list(card.required_body_refs)
        if len(family_extents) > definitions_per_family:
            gaps.append(
                f"definitions-overflow:{card_id}:"
                f"{len(family_extents)}>{definitions_per_family}")
            family_extents = family_extents[:definitions_per_family]
        extents.extend(family_extents)

    citations = tuple(citations_from_qualified_routes(
        conn, routes, source=source)) if routes else ()

    identity_citations = []
    for card_id in retained_ids:
        card = by_id.get(card_id)
        if card is None or len(card.node_qualified_names) > 1:
            continue
        for name, _canonical, file, line_start, line_end in (
                card.node_identities):
            identity_citations.append(_identity_citation(
                name, file, line_start, line_end, source))

    edge_sites = tuple(sorted({
        (citation.call_site_file, citation.call_site_line)
        for citation in citations
        if citation.call_site_file and citation.call_site_line}))

    site_extents = []
    known_extents = sorted({
        (node[0], node[2], node[3], node[4])
        for card_id in retained_ids
        for node in (by_id[card_id].node_identities
                     if card_id in by_id else ())})
    for site_file, site_line in edge_sites:
        containing = [
            extent for extent in known_extents
            if extent[1] == site_file
            and extent[2] <= site_line <= extent[3]]
        if containing:
            narrowest = min(
                containing, key=lambda extent: (
                    extent[3] - extent[2], extent[2], extent[0]))
            site_extents.append(narrowest)

    required = sorted({*extents, *site_extents})
    if len(required) > total_body_extents:
        for extent in required[total_body_extents:]:
            gaps.append(
                f"total-bodies-overflow:{extent[0]}"
                f"@{extent[1]}:{extent[2]}-{extent[3]}")
        required = required[:total_body_extents]
    return FamilyExpansion(
        citations=tuple((*citations, *identity_citations)),
        required_body_extents=tuple(required),
        edge_sites=edge_sites,
        expanded_card_ids=retained_ids,
        gaps=tuple(gaps))


def _identity_citation(name, file, line_start, line_end, source):
    from library.structural_assembly import StructuralCitation

    return StructuralCitation(
        qualified_name=str(name), file=str(file),
        line_start=int(line_start), source_name=source,
        relation="localized", hop=1, call_site_file="",
        call_site_line=0, stop_reason="selected_family",
        line_end=int(line_end))


def _record_family_drop(trace, obligation_id, family, reason) -> None:
    """Bounded drop ledger: every rejected family names its cap."""
    if not isinstance(trace, dict):
        return
    drops = trace.setdefault(obligation_id, {}).setdefault("dropped", [])
    if len(drops) >= 200:
        return
    drops.append({"entry": family["entry"],
                  "terminal": family["terminal"], "reason": reason})
