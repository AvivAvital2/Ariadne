"""The bridge into synthesis: what the LLM receives, and what the response carries.

``index -> fetch document -> curate bundle -> formulate with LLM -> return response``.
Stages one to three produce a chain and its documents; this renders them for stage four and
carries the coordinates into stage five.

The whole point is the inversion: **the evidence is the spine and prose is commentary.**
Before this, ``ask`` retrieved eight documents, concatenated them and synthesized, so no
answer could name a line and the store's 2.5M compiler-precise edges never reached the
model. Here the chain is the structure of the prompt, in execution order, and prose hangs
off individual hops.

Order is not cosmetic. ``MergeIntoCommand.runMerge``'s hops read as the MERGE algorithm —
metadata checks, then ``isInsertOnly``, then the three executors it chooses between — only
because they are ordered by call-site line. Ranking them by anything destroys that, which is
why nothing here reorders.

``unsupported_locations`` is the one guard: an answer may not name a ``file:line`` the bundle
does not contain. Deterministic, no judge, because the recurring failure is output that reads
correctly and cites something that was never there.
"""
from __future__ import annotations
import os

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from library.relation_semantics import relation_site_phrase

if TYPE_CHECKING:# pragma: no cover - typing only
    from library import Library
    from library.chain_bundle import ChainBundle

#: ``path/to/file.ext:123`` as it appears in prose.
_LOCATION = re.compile(r'\b([\w./-]+\.[A-Za-z]{1,6}):(\d+)\b')


@dataclass(frozen=True)
class AnswerEvidence:
    """Everything stage four needs and stage five returns."""

    spine: str = ''
    bundle_citations: list = field(default_factory=list)
    themes: list = field(default_factory=list)
    locations: frozenset = frozenset()
    unresolved_paths: tuple = ()
    truncation_reason: str = ''
    #: The curated hops themselves, so a consumer can offer them as a menu before
    #: spending them (:mod:`library.chain_menu`). ``bundle_citations`` stays because the
    #: response payload is built from coordinates alone.
    hops: tuple = ()
    #: Dispatches the walk reported instead of expanding — see
    #: :func:`library.chain_disclosure.describe_fan_out`. Structured, not prose: the
    #: wording belongs to the presentation layer.
    fan_outs: tuple = ()
    mandatory_fan_outs: tuple = ()
    source_gaps: tuple = ()
    seed_provenance: tuple = ()
    def citations(self) -> list[dict]:
        """The response payload — coordinates, not prose, and each one once.

        Deduplicated by definition. Measured at production width the chain reached 2,645
        hops across 973 distinct symbols, and returning all 2,645 shipped 939,389 characters
        (~235,000 tokens) to the caller for one question. A definition reached from a second
        call site is new evidence *in the spine*, where the call site is shown; in a list of
        coordinates it is the same coordinate twice.
        """
        payload: list[dict] = []
        seen: set[tuple[str, str, int]] = set()
        for hop in self.bundle_citations:
            key = (hop.qualified_name, hop.file, hop.line_start)
            if key in seen:
                continue
            seen.add(key)
            payload.append({
                'qualified_name': hop.qualified_name,
                'file': hop.file,
                'line': hop.line_start,
                'relation': hop.relation,
                'hop': hop.hop,
                'call_site': f'{hop.call_site_file}:{hop.call_site_line}',
                'stop_reason': hop.stop_reason,
            })
        return payload
    def cited_by(self, answer: str) -> list[dict]:
        pattern = re.compile(r"\b([\w./-]+\.[A-Za-z]{1,6}):(\d+)(?:-(\d+))?\b")
        claims = []
        for match in pattern.finditer(answer or ""):
            start = int(match.group(2))
            finish = min(max(int(match.group(3) or start), start), start + 500)
            resolved = resolve_location(f"{match.group(1)}:{start}", self.locations)
            if resolved is None:
                continue
            path, resolved_start = resolved.rsplit(":", 1)
            claims.append((path, int(resolved_start), finish))
        payload = []
        seen = set()
        for hop in self.bundle_citations:
            definition_end = hop.line_end or hop.line_start
            for path, start, finish in claims:
                if path == hop.file:
                    line = max(start, hop.line_start)
                    line_end = min(finish, definition_end)
                    if line > line_end:
                        continue
                elif path == hop.call_site_file and start <= hop.call_site_line <= finish:
                    line = line_end = hop.call_site_line
                else:
                    continue
                key = (hop.qualified_name, path, line, line_end)
                if key in seen:
                    continue
                seen.add(key)
                payload.append({
                    "qualified_name": hop.qualified_name,
                    "file": path,
                    "line": line,
                    "line_end": line_end,
                    "relation": hop.relation,
                    "hop": hop.hop,
                    "call_site": f"{hop.call_site_file}:{hop.call_site_line}",
                    "stop_reason": hop.stop_reason,
                })
        return payload

    def summary(self) -> dict:
        """The chain's shape, for a caller who should not be handed every coordinate."""
        from library.chain_confidence import account_transitions
        return {
            'hops': len(self.bundle_citations),
            'symbols': len({hop.qualified_name for hop in self.bundle_citations}),
            'files': len({hop.file for hop in self.bundle_citations}),
            'locations': len(self.locations),
            'forks': [
                {
                    'qualified_name': fan_out.qualified_name,
                    'file': fan_out.file,
                    'line': fan_out.line_start,
                    'implementations_in_index': fan_out.implementations,"by_package": [list(item) for item in fan_out.by_package], "test_implementations": fan_out.tests
                }
                for fan_out in self.fan_outs
            ],
            'truncation': self.truncation_reason,"transitions": {"total": account_transitions(self).total, "accounted": account_transitions(self).accounted, "unaccounted": account_transitions(self).unaccounted}, "source_gaps": list(self.source_gaps), "unresolved_paths": list(self.unresolved_paths), "caller_frontiers": list(self.caller_frontiers)
        }
    caller_frontiers: tuple = ()

ANSWER_MAX_TOKENS = 4096
PROMPT_CHARS_PER_TOKEN: float = 2.25


def spine_budget_chars(model: 'str | None' = None, *,
                       reserved_output_tokens: int = ANSWER_MAX_TOKENS) -> 'int | None':
    """Characters of chain the prompt may carry: whatever the context window leaves.

    Derived, not declared. The number this replaced -- 20,000 characters -- was set when
    the prompt still carried the retrieved documents beside the chain; measured on a
    production question, those eight documents were 15,754 tokens of prose. Nothing but
    the chain travels now, so that premise is void, and the only claimants on the window
    are the answer reserved inside it and the instructions that frame it.

    ``None`` means the window is unknown, and then nothing is cut. A model absent from
    :data:`docgen.pricing.MODEL_CONTEXT_WINDOW_TOKENS` is reported rather than assigned a
    believable limit, and an over-long prompt fails at the API -- the boundary that owns
    the rule -- instead of being quietly trimmed here.

    This bounds the prompt; it does not make the chain the right size. Measured at
    production width (8 retrieved documents, source ``databricks``, depth 3) the uncapped
    spine is 2,645 hops and ~1.34M tokens -- down from 25,313 hops and ~13.9M once prose
    stopped seeding the walk and type references stopped being traversed -- so a 1M-token
    window admits it whole. What each hop contributes is its ``catalog`` description —
    measured, 883 distinct documents for 2,645 hops, ~88,600 tokens — and its coordinates.
    """
    from docgen.pricing import (
        PROMPT_OVERHEAD_TOKENS,
        context_window_tokens,
    )
    if model is None:
        try:
            from config import get_config
            model = get_config().model
        except Exception:# noqa: BLE001 -- a scoped call need not have a config
            return None
    window = context_window_tokens(model or '')
    if window is None:
        return None
    return int(max(window - reserved_output_tokens - PROMPT_OVERHEAD_TOKENS, 0)
               * PROMPT_CHARS_PER_TOKEN)
#: A line reference with no file: ``(and again at :166)``. The colon must not follow a path
#: or word character, which is what separates it from ``File.scala:166``, ``12:30`` and
#: ``ratio 3:4``.
_BARE_LINE = re.compile(r'(?<![\w./-]):(\d+)\b')


def expand_bare_lines(answer: str) -> str:
    """Give every bare ``:166`` the file named before it.

    An answer writes "invoked at MergeIntoCommand.scala:130 (and again at :166)". The second
    reference is a real claim about a real line, and it inherits its file from the first —
    but :data:`_LOCATION` requires ``file.ext:line``, so as written it is invisible to
    :func:`unsupported_locations` and to :meth:`AnswerEvidence.cited_by`. It is neither
    checked nor returned to the caller.

    Done here rather than in the prompt: instructing the model to repeat the file name spends
    tokens on every answer and depends on it complying. Rewriting afterwards costs nothing
    and always holds.

    A bare reference is expanded **whether or not the result resolves**. Expanding only the
    ones that check out would hide a wrong claim from the guard, which is backwards — the
    point is that every coordinate the answer states gets verified.
    """
    if not answer:
        return answer
    result: list[str] = []
    position = 0
    antecedent = ''
    for match in re.finditer(r'([\w./-]+\.[A-Za-z]{1,6}):(\d+)|(?<![\w./-]):(\d+)\b',
                             answer):
        if match.group(1):# a full coordinate: it sets the antecedent
            antecedent = match.group(1)
            continue
        if not antecedent:# nothing to inherit from
            continue
        result.append(answer[position:match.start()])
        result.append(f'{antecedent}:{match.group(3)}')
        position = match.end()
    result.append(answer[position:])
    return ''.join(result)


def locations_for(hops) -> frozenset:
    """Every coordinate the spine puts in front of the model — definitions and call sites.

    Both are admissible because both are shown. The prompt renders the site as ``called at file:line`` for a call
    edge and ``referenced at file:line`` for a type reference, and tells the model that
    site is what the index recorded, so citing either is citing what the chain showed. A live run cited five call sites and the guard reported all five as
    invented, because this set held definitions only.
    """
    coordinates: set[str] = set()
    for entry in hops:
        citation = getattr(entry, 'citation', entry)
        for line in range(citation.line_start, (citation.line_end or citation.line_start) + 1):
            coordinates.add(f"{citation.file}:{line}")
        if citation.call_site_file:
            coordinates.add(f'{citation.call_site_file}:{citation.call_site_line}')
    return frozenset(coordinates)


def resolve_location(named: str, locations) -> 'str | None':
    """The coordinate ``named`` refers to, or ``None`` when that cannot be decided.

    An answer writes ``InsertOnlyMergeExecutor.scala:53``; the index stores a 78-character
    path. A file name and line matching exactly one known coordinate **is** that coordinate,
    and the index is what says so. Two matches cannot be pinned, so they are reported rather
    than assumed — the point of this check is that a location is verified, not believed.

    Measured on a live answer: 11 of 11 shortened citations resolved to exactly one
    coordinate, and every one had been reported as a fabrication before this existed.
    """
    if "/.../" in named:
        basename = named.rsplit("/", 1)[-1]
        matches = [known for known in locations if known.endswith(f"/{basename}")]
        return matches[0] if len(matches) == 1 else None
    if named in locations:
        return named
    suffix = f'/{named}'
    matches = [known for known in locations if known.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None
def render_spine(bundle: "ChainBundle", max_chars: "int | None" = None) -> str:
    """Render ordered hops, reserving context for compiler-mandatory routes."""
    if not bundle.hops:
        return ""
    described: set[str] = set()
    rendered_entries: list[tuple[bool, list[str], int]] = []
    for entry in bundle.hops:
        hop = entry.citation
        mandatory = hop.relation in ("localized", "shared_reference")
        indent = "  " * max(hop.hop - 1, 0)
        if hop.stop_reason == "revisit":
            site = relation_site_phrase(hop.relation)
            rendered = [f"{indent}{hop.qualified_name.rsplit('.', 1)[-1]} (already shown) {site} {hop.call_site_file}:{hop.call_site_line}"]
        else:
            site = relation_site_phrase(hop.relation)
            rendered = [f"{indent}{hop.qualified_name}  [{hop.file}:{hop.line_start}]  {site} {hop.call_site_file}:{hop.call_site_line}"]
            if entry.evidence and hop.qualified_name not in described:
                described.add(hop.qualified_name)
                rendered.append(f"{indent}    {entry.evidence.strip()}")
            for excerpt in entry.source_excerpts:
                rendered.append(f"{indent}    Source {excerpt.kind} [{excerpt.file}:{excerpt.line_start}-{excerpt.line_end}] sha256={excerpt.sha256}")
                rendered.extend(f"{indent}        {line}" for line in excerpt.content.splitlines())
        rendered_entries.append((mandatory, rendered, sum(len(line) + 1 for line in rendered)))
    future_mandatory = sum(cost for mandatory, _lines, cost in rendered_entries if mandatory)
    reserve_routes = max_chars is None or future_mandatory <= max_chars
    lines: list[str] = []
    used = 0
    omitted = 0
    for index, (mandatory, rendered, cost) in enumerate(rendered_entries):
        if not reserve_routes:
            if max_chars is not None and used + cost > max_chars and lines:
                omitted += len(rendered_entries) - index
                break
            lines.extend(rendered)
            used += cost
            continue
        if mandatory:
            future_mandatory -= cost
        if max_chars is not None and not mandatory and used + cost + future_mandatory > max_chars:
            omitted += 1
            continue
        lines.extend(rendered)
        used += cost
    if omitted:
        lines.append(f"... {omitted} ancillary hop(s) omitted to fit the context; the mandatory route remains complete.")
    if bundle.themes:
        lines.append("")
        lines.append("This chain runs through:")
        for theme in bundle.themes:
            breadth = "" if theme.coherent else " (broad)"
            lines.append(f'  {theme.hops} hop(s) in "{theme.title}"{breadth}')
    if bundle.source_gaps:
        lines.append("")
        lines.append("Source gaps (do not infer claims for these ranges):")
        lines.extend(f"- {gap}" for gap in bundle.source_gaps)
    return "\n".join(lines)
def evidence_for(
    library: 'Library',
    documents,
    *,
    question: str = '', source: str,
    depth: int = 3,
    max_spine_chars: 'int | None' = None,
    caller_depth = 1, clew_symbols=(),
clew_matches=(), positioning_documents=(), defer_source: bool = False) -> AnswerEvidence:
    """Run stages one to three over what retrieval returned.

    Seeds come from the retrieved documents — the localization step measured at production
    width, which takes required-slot reach from 35% to 59% — and the walk expands them,
    because a seed naming a type has no outgoing call edge of its own.

    ``clew_symbols`` adds the qualified names of a route retrieval matched (see
    :mod:`library.clews`). A document tells the walk which *file* to start in; a clew tells it
    which *path*, which is the one thing document seeding cannot express — the walk itself has
    never seen the question. Measured on the databricks pack, pooled clew strategies contain
    92.8% of the symbols answer keys require against 66.0% for one document-seeded walk.

    An accepted clew is authoritative positioning: its intact route replaces broad document
    seeds for traversal. Documents remain the fallback when no route passes selection, while
    explicit question-symbol seeds are added separately as mandatory evidence. This prevents a
    short route from inheriting every adjacent symbol in every retrieved file.
    """
    from docgen.scip_paths import indexer_cwds
    from library.chain_bundle import curate_bundle
    from library.structural_assembly import chain_from_seeds, seeds_from_documents, caller_roots, question_symbol_seeds, reference_bridges, localized_citations, question_ranked_seeds, citations_from_qualified_routes, obligation_reference_closure, connect_obligation_targets, selected_route_call_fanout

    root = None
    try:
        from config import get_config
        root = str(get_config().get_all_source_paths().get(source) or '') or None
    except Exception:# noqa: BLE001 — config is optional for a scoped call
        root = None
    cwds = indexer_cwds(root) if root else ()

    with library._conn_provider.acquire() as conn:
        seed_set = seeds_from_documents(
            conn, documents, source=source, indexer_cwds=cwds, source_root=root)
        seeds = list(seed_set.seeds)
        document_seeds = tuple(seeds)
        catalog_qnames = []
        for document in positioning_documents:
            metadata = (document.get("metadata") if isinstance(document, dict) else getattr(document, "metadata", None)) or {}
            qualified_name = metadata.get("qualified_name") if isinstance(metadata, dict) else None
            if qualified_name and qualified_name not in catalog_qnames:
                catalog_qnames.append(str(qualified_name))
        targeted_seeds = []
        for start in range(0, len(catalog_qnames), 300):
            chunk = catalog_qnames[start:start + 300]
            marks = ",".join("?" * len(chunk))
            targeted_seeds.extend(row[0] for row in conn.execute(
                f"SELECT canonical_id FROM scip_symbols WHERE source_name = ? "
                f"AND qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
                [source, *chunk, "local *"]))
        if not targeted_seeds:
            targeted_seed_set = seeds_from_documents(
                conn, positioning_documents, source=source,
                indexer_cwds=cwds, source_root=root)
            targeted_seeds = list(question_ranked_seeds(
                conn, targeted_seed_set.seeds, question, source=source, limit=4))
        explicit_question_seeds = tuple(question_symbol_seeds(conn, question, source=source))
        question_seeds = list(dict.fromkeys((
            *explicit_question_seeds,
            *targeted_seeds)))
        positioned_seeds = []
        if clew_matches or clew_symbols:
            listed = list(dict.fromkeys(
                [str(name) for match in clew_matches for name in match.clew.route]
                + [str(symbol) for match in clew_matches
                   for _obligation, symbol in match.target_symbols]
                + [str(name) for name in clew_symbols]))
            for start in range(0, len(listed), 300):
                chunk = listed[start:start + 300]
                placeholders = ','.join('?' * len(chunk))
                positioned_seeds += [row[0] for row in conn.execute(
                    f'SELECT canonical_id, source_name FROM scip_symbols '
                    f'WHERE qualified_name IN ({placeholders}) '
                    f"AND canonical_id NOT LIKE 'local %'", chunk)
                    if row[1] == source]
        if clew_matches and positioned_seeds:
            ranked_fallback = (question_ranked_seeds(
                conn, seeds, question, source=source, limit=12)
                if question.strip() else ())
            seeds = list(dict.fromkeys((*positioned_seeds, *ranked_fallback)))
        else:
            seeds += positioned_seeds
            seeds = list(question_ranked_seeds(
                conn, seeds, question, source=source))
        seeds = list(dict.fromkeys((*seeds, *targeted_seeds)))
        selected_route_mode = bool(clew_matches)
        if selected_route_mode:
            caller_expansion = caller_roots(conn, [], source=source, depth=0)
            citations = citations_from_qualified_routes(
                conn, [tuple(match.clew.route) for match in clew_matches],
                source=source)
            citations.extend(selected_route_call_fanout(
                            conn, clew_matches, source=source))
            selected_targets = list(dict.fromkeys(
                symbol for match in clew_matches
                for _obligation, symbol in match.target_symbols))
            citations.extend(localized_citations(
                conn, selected_targets, source=source))
            citations.extend(connect_obligation_targets(
                conn, clew_matches, source=source))
            citations.extend(obligation_reference_closure(
                conn, clew_matches, source=source, question=question))
            _empty, truncation = chain_from_seeds(conn, [], source=source, depth=0)
        else:
            caller_expansion = caller_roots(
                conn, seeds, source=source, depth=caller_depth)
            if caller_expansion.roots:
                combined_seeds = list(dict.fromkeys((
                    *caller_expansion.roots, *caller_expansion.uncovered_seeds)))
                walked, truncation = chain_from_seeds(
                    conn, combined_seeds, source=source, depth=depth)
                citations = [*caller_expansion.citations, *walked]
            else:
                citations, truncation = chain_from_seeds(
                    conn, seeds, source=source, depth=depth)
    with library._conn_provider.acquire() as conn:
        question_caller_expansion = caller_roots(
            conn, question_seeds, source=source, depth=caller_depth)
        question_localized = localized_citations(conn, question_seeds, source=source)
        question_rooted, question_truncation = chain_from_seeds(
            conn, question_caller_expansion.roots, source=source, depth=1)
        question_seeded, _ = chain_from_seeds(
            conn, question_seeds, source=source, depth=1)
        origins_by_id = {}
        for canonical_id in seeds:
            if canonical_id in positioned_seeds:
                origin = "clew"
            elif canonical_id in targeted_seeds:
                origin = "catalog"
            else:
                origin = "document_fallback"
            origins_by_id.setdefault(canonical_id, set()).add(origin)
        for canonical_id in targeted_seeds:
            origins_by_id.setdefault(canonical_id, set()).add("catalog")
        for canonical_id in explicit_question_seeds:
            origins_by_id.setdefault(canonical_id, set()).add("question_symbol")
        for canonical_id in caller_expansion.roots:
            origins_by_id.setdefault(canonical_id, set()).add("caller_expansion")
        for canonical_id in question_caller_expansion.roots:
            origins_by_id.setdefault(canonical_id, set()).add("caller_expansion")
        provenance_by_symbol = {}
        provenance_ids = list(origins_by_id)
        for start in range(0, len(provenance_ids), 300):
            chunk = provenance_ids[start:start + 300]
            marks = ",".join("?" * len(chunk))
            for canonical_id, qualified_name, owner in conn.execute(
                    f"SELECT canonical_id, qualified_name, source_name FROM scip_symbols "
                    f"WHERE canonical_id IN ({marks})", chunk):
                if owner != source:
                    continue
                provenance_by_symbol.setdefault(str(qualified_name), set()).update(
                    origins_by_id.get(canonical_id, ()))
    question_citations = [*question_localized, *question_caller_expansion.citations, *question_rooted, *question_seeded]
    seen_coordinates = {
        (citation.qualified_name, citation.file, citation.line_start,
         citation.call_site_file, citation.call_site_line)
        for citation in question_citations
    }
    question_citations.extend(
        citation for citation in citations
        if (citation.qualified_name, citation.file, citation.line_start,
            citation.call_site_file, citation.call_site_line) not in seen_coordinates)
    citations = question_citations
    with library._conn_provider.acquire() as conn:
        bridge_expansion = reference_bridges(
            conn, citations, source=source, question=question)
    bridge_targets = {
        citation.parent_qualified_name for citation in bridge_expansion.citations
        if citation.parent_qualified_name}
    target_citations = [
        citation for citation in citations
        if citation.qualified_name in bridge_targets]
    citations = [*target_citations, *bridge_expansion.citations, *citations]
    for citation in (*target_citations, *bridge_expansion.citations):
        provenance_by_symbol.setdefault(citation.qualified_name, set()).add(
            "reference_bridge")
    seed_provenance = tuple(
        {"symbol": symbol, "origins": sorted(origins)}
        for symbol, origins in sorted(provenance_by_symbol.items()))

    if not citations:
        return AnswerEvidence(unresolved_paths=seed_set.unresolved_paths,
                              fan_outs=truncation.fan_outs, caller_frontiers = tuple(dict.fromkeys((*caller_expansion.gated_targets, *question_caller_expansion.gated_targets, *bridge_expansion.gated_targets))), mandatory_fan_outs = mandatory_fan_outs(question, question_truncation.fan_outs, truncation.fan_outs), seed_provenance=seed_provenance)

    bundle = curate_bundle(
        library, citations, source=source, source_root=root,
        materialize_source=not defer_source,
        fetch_documents=not defer_source)
    budget = (max_spine_chars if max_spine_chars is not None
              else spine_budget_chars())
    spine_shown = render_spine(bundle, budget)
    # Locations stay the FULL bundle: a hop cut from the prompt is still real, so an
    # answer naming it must not be called a fabrication.
    locations = locations_for(bundle.hops)
    return AnswerEvidence(
        spine=spine_shown,
        bundle_citations=[hop.citation for hop in bundle.hops],
        themes=list(bundle.themes),
        locations=locations,
        unresolved_paths=seed_set.unresolved_paths,
        truncation_reason=_why_truncated(bundle, spine_shown),
        # a fork the walk declined to expand is evidence about the code, so it travels
        fan_outs=truncation.fan_outs,
        hops=tuple(bundle.hops),
    caller_frontiers = tuple(dict.fromkeys((*caller_expansion.gated_targets, *question_caller_expansion.gated_targets, *bridge_expansion.gated_targets))), source_gaps = bundle.source_gaps, mandatory_fan_outs = mandatory_fan_outs(question, question_truncation.fan_outs, truncation.fan_outs), seed_provenance=seed_provenance)


def unsupported_locations(answer: str, evidence: AnswerEvidence) -> tuple[str, ...]:
    """Every ``file:line`` the answer names that the bundle does not contain.

    The line is part of the claim, so the right file at the wrong line is unsupported —
    the same standard the eval's admissibility gate applies. An answer naming no location
    is not reported: saying nothing locational is honest, inventing one is not.
    """
    return tuple(
        found for found in
        (f'{m.group(1)}:{m.group(2)}' for m in _LOCATION.finditer(answer or ''))
        if resolve_location(found, evidence.locations) is None
    )


def _why_truncated(bundle: 'ChainBundle', spine: str) -> str:
    """What the caller should know was left out, if anything."""
    # One budget, one report. Curation attaches every explained hop; the spine is the only
    # stage that drops anything, and it says so in the text it returns.
    return 'chain truncated' if 'omitted to fit the context' in spine else ''
def bounded_prompt(prompt: str, *, max_chars: int | None = None,
                   model: str | None = None) -> str:
    """Apply a final provider-safe ceiling after every prompt transformation."""
    budget = spine_budget_chars(model) if max_chars is None else max_chars
    external = os.environ.get("ARIADNE_MAX_PROMPT_CHARS")
    if external:
        external_limit = max(int(external), 0)
        budget = external_limit if budget is None else min(budget, external_limit)
    if budget is None or len(prompt) <= budget:
        return prompt
    marker = "\n[prompt truncated]"
    if budget <= len(marker):
        return marker[-budget:]
    return prompt[:budget - len(marker)] + marker
def mandatory_fan_outs(question: str, routed, discovered) -> tuple:
    """Forks on the question-localized receiver route, not adjacent graph noise."""
    known = set(discovered or ())
    return tuple(dict.fromkeys(fork for fork in (routed or ()) if not known or fork in known))
def catalog_positioning_documents(library, question: str, *, sources,
                                  limit: int = 2, query_embedding=None,
                                  matrix_provider=lambda: None):
    """A tiny, source-scoped catalog supplement for compiler positioning.

    Catalog prose is not answer evidence. It only identifies at most ``limit`` code files
    whose descriptions share several question concepts; SCIP still derives every route.
    """
    stop = {"about", "after", "also", "and", "are", "does", "for", "from",
            "have", "how", "into", "relative", "that", "the", "then", "this",
            "what", "when", "where", "which", "with", "would"}
    aliases = {"emitted": "emit", "emitting": "emit", "rows": "row",
               "resulting": "result", "writes": "write", "writing": "write"}
    tokens = []
    raw_words = re.findall(r"[A-Za-z][A-Za-z0-9_]{2,}", question or "")
    for raw in raw_words:
        token = aliases.get(raw.lower(), raw.lower())
        if token not in stop and token not in tokens:
            tokens.append(token)
    tokens = tokens[:12]
    uppercase = [raw for raw in raw_words if raw.isupper()]
    anchor_words = uppercase or [raw for raw in raw_words if raw[:1].isupper()]
    anchors = []
    for raw in anchor_words:
        token = aliases.get(raw.lower(), raw.lower())
        if token not in stop and token not in anchors:
            anchors.append(token)
    if not anchors:
        anchors = sorted(tokens, key=lambda token: (-len(token), token))[:3]
    sources = tuple(dict.fromkeys(str(source) for source in sources if source))
    if limit <= 0 or len(tokens) < 3 or not anchors or not sources:
        return []
    if query_embedding is not None:
        import json
        import math
        from library.embedding_ranking import select_ranker
        with library._conn_provider.acquire() as conn:
            candidate_ids = [str(row[0]) for row in conn.execute(
                "SELECT id FROM documents WHERE content_type IN ('catalog', 'explanation') "
                f"AND source_name IN ({','.join('?' * len(sources))}) "
                "AND embedding IS NOT NULL AND lower(title) NOT LIKE '%test%'",
                list(sources))]
        ranker = select_ranker(
            len(candidate_ids), matrix_provider, library)
        ranked = ranker.rank(query_embedding, candidate_ids,
                             max(limit * 12, 48))
        ranked_ids = [doc_id for doc_id, _score in ranked]
        _source_marks = ",".join("?" * len(sources))
        _lexical_match_sql = " OR ".join(["lower(title) LIKE ?"] * len(anchors))
        with library._conn_provider.acquire() as conn:
            _lexical_ids = [str(row[0]) for row in conn.execute(
                "SELECT id FROM documents WHERE content_type IN ('catalog', 'explanation') "
                f"AND source_name IN ({_source_marks}) "
                f"AND ({_lexical_match_sql}) AND lower(title) NOT LIKE '%test%' LIMIT 96",
                [*sources, *[f"%{anchor}%" for anchor in anchors]])]
        ranked_ids = list(dict.fromkeys([*ranked_ids, *_lexical_ids]))
        metadata = {}
        with library._conn_provider.acquire() as conn:
            for start in range(0, len(ranked_ids), 400):
                chunk = ranked_ids[start:start + 400]
                if not chunk:
                    continue
                marks = ','.join('?' * len(chunk))
                for doc_id, title, source_files in conn.execute(
                        f"SELECT id, title, source_files FROM documents WHERE id IN ({marks})",
                        chunk):
                    metadata[str(doc_id)] = (str(title), json.loads(source_files or "[]"))

        def normalize_inflection(token):
            token = token.lower()
            if len(token) > 5 and token.endswith("ies"):
                return token[:-3] + "y"
            for suffix in ("ing", "ed", "ion", "es", "s"):
                if len(token) > len(suffix) + 3 and token.endswith(suffix):
                    return token[:-len(suffix)]
            return token
        def role_tokens(text):
            parts = re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+",
                text or "")
            return [normalize_inflection(part) for part in parts
                    if len(part) > 1 and part.lower() not in stop]
        question_sequence = role_tokens(question)
        question_set = set(question_sequence)
        question_pairs = set(zip(question_sequence, question_sequence[1:]))
        heading_sequences = {
            doc_id: role_tokens(metadata.get(doc_id, ("", []))[0])
            for doc_id in ranked_ids}
        frequencies = {token: sum(token in set(sequence)
                                  for sequence in heading_sequences.values())
                       for token in question_set}
        population = max(len(heading_sequences), 1)
        similarity = {doc_id: score for doc_id, score in ranked}
        def role_score(doc_id):
            sequence = heading_sequences.get(doc_id, [])
            overlap = question_set.intersection(sequence)
            pairs = set(zip(sequence, sequence[1:]))
            weighted = sum(1.0 + math.log((population + 1) /
                                          (frequencies[token] + 1))
                           for token in overlap)
            return (len(question_pairs.intersection(pairs)), weighted,
                    len(overlap), similarity.get(doc_id, 0.0))
        ranked_ids.sort(key=lambda doc_id: tuple(-value for value in role_score(doc_id))
                        + (doc_id,))
        chosen = []
        seen_files = set()
        seen_partitions = set()
        for doc_id in ranked_ids:
            _title, files = metadata.get(doc_id, ("", []))
            first_file = str(files[0]) if files else ""
            normalized = f"/{first_file.lower().strip('/')}"
            if not first_file or any(segment in normalized for segment in (
                    "/test/", "/tests/", "/benchmark/", "/benchmarks/",
                    "/target/", "/generated/")):
                continue
            partition = first_file.split("/", 1)[0]
            if first_file in seen_files:
                continue
            if partition in seen_partitions and len(chosen) < min(limit, 2):
                continue
            chosen.append(doc_id)
            seen_files.add(first_file)
            seen_partitions.add(partition)
            if len(chosen) == limit:
                break
        if chosen:
            documents = {document.id: document
                         for document in library.get_documents_batch(chosen)}
            return [documents[doc_id] for doc_id in chosen if doc_id in documents]
    match_sql = " OR ".join(["lower(title) LIKE ?"] * len(anchors))
    source_marks = ",".join("?" * len(sources))
    patterns = [f"%{token}%" for token in anchors]
    import json
    import math

    with library._conn_provider.acquire() as conn:
        rows = conn.execute(
            "SELECT id, title, content, source_files, content_type, "
            "json_extract(metadata, '$.qualified_name') FROM documents "
            f"WHERE content_type IN ('catalog', 'explanation') "
            f"AND source_name IN ({source_marks}) "
            f"AND ({match_sql}) AND lower(title) NOT LIKE '%test%' LIMIT 4096",
            [*sources, *patterns]).fetchall()
        qnames = [str(row[5]) for row in rows if row[5]]
        edge_counts = {}
        scip_sources = tuple(dict.fromkeys(
            source.removeprefix("spool:") for source in sources))
        for start in range(0, len(qnames), 400):
            chunk = qnames[start:start + 400]
            marks = ",".join("?" * len(chunk))
            source_marks_for_scip = ",".join("?" * len(scip_sources))
            for qualified_name, count in conn.execute(
                    "SELECT s.qualified_name, count(*) FROM scip_symbols s "
                    "JOIN scip_edges e ON e.caller_canonical_id = s.canonical_id "
                    f"WHERE s.source_name IN ({source_marks_for_scip}) "
                    f"AND s.qualified_name IN ({marks}) GROUP BY s.qualified_name",
                    [*scip_sources, *chunk]):
                edge_counts[str(qualified_name)] = int(count)

    def words(text):
        parts = re.findall(
            r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+", text or "")
        return {aliases.get(part.lower(), part.lower()) for part in parts
                if len(part) > 1 and part.lower() not in stop}

    question_words = set(tokens)
    prepared = []
    frequencies = {token: 0 for token in question_words}
    for doc_id, title, content, source_files, content_type, qualified_name in rows:
        title_words = words(title)
        document_words = words(f"{title} {content}")
        for token in question_words.intersection(document_words):
            frequencies[token] += 1
        files = json.loads(source_files or "[]")
        first_file = str(files[0]) if files else ""
        partition = first_file.split("/", 1)[0] if "/" in first_file else ""
        normalized_file = f"/{first_file.lower().strip('/')}"
        if any(segment in normalized_file for segment in (
                "/test/", "/tests/", "/benchmark/", "/benchmarks/",
                "/target/", "/generated/")):
            continue
        prepared.append((doc_id, title, content, title_words, document_words,
                         partition, first_file, content_type,
                         str(qualified_name or "")))
    ranked = []
    population = max(len(prepared), 1)
    for (doc_id, title, content, title_words, document_words,
         partition, first_file, content_type, qualified_name) in prepared:
        matched = question_words.intersection(document_words)
        overlap = len(matched)
        weighted = sum(
            1.0 + math.log((population + 1) / (frequencies[token] + 1))
            for token in matched)
        anchor_bonus = 2 * len(set(anchors).intersection(title_words))
        title_overlap = len(question_words.intersection(title_words))
        if overlap >= 3:
            type_rank = 0 if content_type == "explanation" else 1
            graph_degree = min(edge_counts.get(qualified_name, 0), 50)
            ranked.append((-title_overlap, -graph_degree, type_rank,
                           -(weighted + anchor_bonus), -overlap,
                           len(content or ""), str(title), str(doc_id),
                           partition, first_file))
    ranked.sort()
    chosen = []
    seen_partitions = set()
    seen_files = set()
    for item in ranked:
        doc_id, partition, first_file = item[7], item[8], item[9]
        if partition in seen_partitions or first_file in seen_files:
            continue
        chosen.append(doc_id)
        seen_partitions.add(partition)
        seen_files.add(first_file)
        if len(chosen) == limit:
            break
    for item in ranked:
        doc_id, first_file = item[7], item[9]
        if len(chosen) == limit:
            break
        if doc_id not in chosen and first_file not in seen_files:
            chosen.append(doc_id)
            seen_files.add(first_file)
    ids = chosen
    return library.get_documents_batch(ids)
