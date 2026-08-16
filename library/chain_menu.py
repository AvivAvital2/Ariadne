"""Offer the bundle before spending it: one line per thing the model could read.

Stage three used to hand synthesis the whole bundle. Measured at production width that is
2,645 hop lines and 883 descriptions — 240,945 tokens, $1.20 a question — of which 68% is
coordinates for hops the answer never mentions. A broad question doubles it.

So the chain is offered as a menu first. The model names what it wants, and only those
bodies are fetched: 973 definitions and ~1,060 sections costs about $0.13, and the second
call carries the handful that were chosen. The saving is not "titles are cheaper than
documents" — it is that a menu is **per symbol** while a chain is per occurrence, and a
``file:line`` is only needed for the hops an answer actually cites.

What this does *not* do is decide anything about the code. SCIP still decides what the
chain contains; the menu only lets the question influence which part of it is read, which
nothing in the walk can do — the walk has never seen the question. Selection is therefore
additive: a caller is free to send the structural spine as well, so a bad pick costs tokens
rather than evidence.

Two halves, labelled, because they are not equally reliable:

* **definitions** — the ``catalog`` entry for a symbol the walk reached, anchored to an
  exact ``file:line`` from the index. This is what an answer cites.
* **sections** — headings of the ``explanation`` documents covering the same files.
  Generated prose about a module, with no line-level anchor. Background, not citation.

Selection is by number. A number either labels something or it does not, so a model cannot
conjure a symbol by misspelling one and nothing here needs fuzzy matching; unknown numbers
are reported for the caller to see rather than interpreted.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from library.relation_semantics import relation_site_phrase, transition_verb

if TYPE_CHECKING:  # pragma: no cover - typing only
    from library import Library
    from library.chain_bundle import BundleHop

#: How much of a description a menu line shows. A display preference, not a derivation:
#: enough to recognise what a definition is for, and no more, because the body is what the
#: second call fetches. Measured, the first line of a catalog description averages ~360
#: characters and the whole menu is 1,920 lines.
SUMMARY_CHARS = 90

#: ``1``/``12`` for a definition, ``S1``/``S12`` for a section, however the model writes it.
_CHOICE = re.compile(r'\b(S?)(\d{1,4})\b', re.IGNORECASE)


@dataclass(frozen=True)
class ChainMenu:
    """What the chain offers, and how to read a reply about it."""

    text: str = ''
    #: menu number -> qualified name
    symbols: dict = field(default_factory=dict)
    #: menu label (``S3``) -> ``(document_id, section idx)``
    sections: dict = field(default_factory=dict)
    section_files: dict = field(default_factory=dict)
    section_titles: dict = field(default_factory=dict)


@dataclass(frozen=True)
class Selection:
    """What the model asked for, resolved against the menu it was given."""

    symbols: list = field(default_factory=list)
    sections: list = field(default_factory=list)
    #: Labels that matched nothing. Reported, never guessed at.
    unknown: tuple = ()
    route_ids: tuple = ()
    section_ids: tuple = ()
    occurrence_keys: tuple = ()


def _owner_qualified(qualified_name: str) -> str:
    """``Classic.writeAllChanges`` — enough to choose between, without the full package.

    A bare last segment cannot be chosen between: a corpus has many ``apply`` and many
    ``write``. The owner is what disambiguates, and the full package name is what the
    second call fetches anyway.
    """
    parts = qualified_name.split('.')
    return '.'.join(parts[-2:]) if len(parts) > 1 else qualified_name


def _summary(content: str) -> str:
    """The sentence a choice is made on, out of a catalog body.

    A catalog document opens with kind, qualified name, package, path and signature, then
    carries ``Description: ...``. The menu line already prints the name, so the header adds
    only length: measured on the live store, taking the first line produced entries like
    ``catalog.CatalogManager — scala_object org.apache.spark.sql.connector.catalog…``, which
    says nothing the name had not. The description is the part that distinguishes.
    """
    lines = [line.strip() for line in (content or '').splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith('description:'):
            return line.split(':', 1)[1].strip()
    # No description: take the signature the header ends with (``… :: trait Foo``) rather
    # than the header itself, which carries the file path — the one thing a menu must not
    # spend, and what the end-to-end harness caught on its first run.
    for line in lines:
        if ' :: ' in line:
            return line.split(' :: ', 1)[1].strip()
    return lines[0] if lines else ''
def menu_for(library: 'Library', hops: list['BundleHop'], *, source: str,
             include_summaries: bool = True) -> ChainMenu:
    """One line per definition the chain reached, then one per section covering its files."""
    definitions: dict[str, 'BundleHop'] = {}
    for hop in hops:
        definitions.setdefault(hop.citation.qualified_name, hop)
    if not definitions:
        return ChainMenu()

    files = sorted({hop.citation.file for hop in definitions.values()})
    sections: dict[str, tuple[str, int]] = {}
    section_titles: dict[str, str] = {}
    section_lines: list[str] = []
    # A hop's ``evidence`` is what rationing allowed to travel as proof; ``plumbing`` and
    # ``revisit`` carry none by design. Choosing is a different job from evidencing, so the
    # menu reads the document itself — otherwise a quarter of the lines are bare names and
    # the model picks blind. The documents are already resolved; this only reads them.
    described: dict[str, str] = {}
    wanted_docs = sorted({hop.document_id for hop in definitions.values()
                          if hop.document_id})
    with library._conn_provider.acquire() as conn:
        if include_summaries:
          for start in range(0, len(wanted_docs), 400):
            chunk = wanted_docs[start:start + 400]
            placeholders = ','.join('?' * len(chunk))
            for doc_id, content in conn.execute(
                    f'SELECT id, content FROM documents WHERE id IN ({placeholders})',
                    chunk):
                described[doc_id] = content or ''
        rows: list[tuple[str, str, int, str, str]] = []
        for start in range(0, len(files), 400):
            chunk = files[start:start + 400]
            like = ' OR '.join(['d.source_files LIKE ?'] * len(chunk))
            rows += conn.execute(
                f'SELECT d.id, d.title, s.idx, s.heading, d.source_files FROM sections s '
                f'JOIN documents d ON d.id = s.document_id '
                f"WHERE d.content_type = 'explanation' AND d.source_name IN (?, ?) "
                f"AND ({like}) ORDER BY d.title, s.idx",
                [source, (source[6:] if source.startswith("spool:") else f"spool:{source}"), *[f"%{path}%" for path in chunk]]).fetchall()
    section_files: dict[str, tuple[str, ...]] = {}
    for doc_id, title, idx, heading, source_files in rows:
        label = f'S{len(sections) + 1}'
        section_titles[label] = f'{title} {heading}'
        sections[label] = (doc_id, idx)
        section_files[label] = tuple(path for path in files if path in (source_files or ''))
        section_lines.append(f'  {label}. {title} -> {heading}')

    symbols: dict[str, str] = {}
    definition_lines: list[str] = []
    for number, (qualified_name, hop) in enumerate(definitions.items(), start=1):
        symbols[str(number)] = qualified_name
        # No ``file:line`` here. A coordinate is what an answer *cites*, and the second call
        # carries it for the handful chosen; in the menu it is 87,000 characters of JVM path
        # across 973 lines, spent on a choice that is made by name and purpose.
        summary = _summary(described.get(hop.document_id or '', '') or hop.evidence or '')
        shown = summary[:SUMMARY_CHARS].rstrip()
        definition_lines.append(
            f'  {number}. {_owner_qualified(qualified_name)}'
            + (f' — {shown}{"…" if len(summary) > SUMMARY_CHARS else ""}' if shown else ''))

    text = '\n'.join([
        'DEFINITIONS the chain reached. Each has an exact file:line in the index, supplied '
        'with the body when you ask for it — these are what an answer cites.',
        *definition_lines,
    ] + ([
        '',
        'SECTIONS of the documents covering the same files. Generated prose about a module, '
        'with no line-level anchor: background only, never cited.',
        *section_lines,
    ] if section_lines else []))
    return ChainMenu(text=text, symbols=symbols, sections=sections,
                     section_files=section_files, section_titles = section_titles)


def resolve_selection(menu: ChainMenu, reply: str) -> Selection:
    """The numbers in ``reply``, resolved against ``menu``. Order is the reply's order."""
    chosen_symbols: list[str] = []
    chosen_sections: list[tuple[str, int]] = []
    unknown: list[str] = []
    for prefix, digits in _CHOICE.findall(reply or ''):
        if prefix:
            label = f'S{digits}'
            target = menu.sections.get(label)
            if target is None:
                unknown.append(label)
            elif target not in chosen_sections:
                chosen_sections.append(target)
            continue
        qualified_name = menu.symbols.get(digits)
        if qualified_name is None:
            unknown.append(digits)
        elif qualified_name not in chosen_symbols:
            chosen_symbols.append(qualified_name)
    return Selection(symbols=chosen_symbols, sections=chosen_sections,
                     unknown=tuple(unknown))


@dataclass(frozen=True)
class Fetched:
    """The bodies the model asked for, and nothing else."""

    #: qualified name -> the ``catalog`` description
    definitions: dict = field(default_factory=dict)
    #: ``(document title, heading, content)`` for each chosen section
    sections: list = field(default_factory=list)


def fetch_selected(library: 'Library', selection: Selection,
                   hops: list['BundleHop']) -> Fetched:
    """Read only what was chosen.

    Document ids come from the hops rather than being recomputed, so this cannot disagree
    with what :func:`library.chain_bundle.curate_bundle` already resolved.
    """
    ids = {hop.citation.qualified_name: hop.document_id for hop in hops
           if hop.document_id}
    wanted = {name: ids[name] for name in selection.symbols if name in ids}
    definitions: dict[str, str] = {}
    sections: list[tuple[str, str, str]] = []
    if not wanted and not selection.sections:
        return Fetched()
    with library._conn_provider.acquire() as conn:
        if wanted:
            by_id = {doc_id: name for name, doc_id in wanted.items()}
            order = list(wanted)
            found: dict[str, str] = {}
            for start in range(0, len(by_id), 400):
                chunk = list(by_id)[start:start + 400]
                placeholders = ','.join('?' * len(chunk))
                for doc_id, content in conn.execute(
                        f'SELECT id, content FROM documents '
                        f'WHERE id IN ({placeholders})', chunk):
                    found[by_id[doc_id]] = content or ''
            definitions = {name: found[name] for name in order if name in found}
        for document_id, idx in selection.sections:
            row = conn.execute(
                'SELECT d.title, s.heading, s.content FROM sections s '
                'JOIN documents d ON d.id = s.document_id '
                'WHERE s.document_id = ? AND s.idx = ?', (document_id, idx)).fetchone()
            if row is not None:
                sections.append((row[0], row[1], row[2] or ''))
    return Fetched(definitions=definitions, sections=sections)
def render_selected(hops: list["BundleHop"], selection: Selection,
                    fetched: Fetched, max_chars: int | None = None) -> str:
    """Render the complete spine within the same hard budget as the full spine."""
    chosen = set(selection.symbols)
    lines: list[str] = []
    shown: set[str] = set()
    used = 0
    omitted = 0
    for index, hop in enumerate(hops):
        name = hop.citation.qualified_name
        indent = "  " * max(hop.citation.hop - 1, 0)
        site = relation_site_phrase(hop.citation.relation)
        rendered = [
            f"{indent}{name}  [{hop.citation.file}:{hop.citation.line_start}]"
            f"  {site} {hop.citation.call_site_file}:{hop.citation.call_site_line}"]
        if hop.citation.relation == "localized":
            rendered[0] += "  QUESTION-LOCALIZED — mandatory"
        elif hop.citation.relation == "shared_reference":
            rendered[0] += "  IDENTITY BRIDGE — mandatory"
        body = fetched.definitions.get(name) if name in chosen else None
        if body and name not in shown:
            shown.add(name)
            rendered.append(f"{indent}    {_summary(body)}")
        for excerpt in hop.source_excerpts:
            rendered.append(
                f"{indent}    Source {excerpt.kind} "
                f"[{excerpt.file}:{excerpt.line_start}-{excerpt.line_end}] "
                f"sha256={excerpt.sha256}")
            rendered.extend(
                f"{indent}        {line}" for line in excerpt.content.splitlines())
        cost = sum(len(line) + 1 for line in rendered)
        if max_chars is not None and used + cost > max_chars:
            omitted = len(hops) - index
            break
        lines.extend(rendered)
        used += cost
    if omitted:
        lines.append(f"... {omitted} further hop(s) omitted to fit the context.")
    else:
        remaining = len({hop.citation.qualified_name for hop in hops}) - len(chosen)
        if remaining > 0:
            lines.append(f"... {remaining} further definition(s) remain as structural "
                         f"evidence; their optional descriptions were not requested.")
    if fetched.sections:
        lines.append("")
        lines.append("Background (module prose, no line-level anchor — do not cite):")
        for title, heading, content in fetched.sections:
            lines.append(f"  {title} -> {heading}")
            lines.append(f"    {content.strip()}")
    text = "\n".join(lines)
    if max_chars is not None and len(text) > max_chars:
        marker = "\n... evidence omitted to fit the context."
        return marker[-max_chars:] if max_chars <= len(marker) else text[:max_chars - len(marker)] + marker
    return text
@dataclass(frozen=True)
class EvidenceGraphNode:
    id: str
    symbol: str
    occurrence: tuple
    file: str
    line: int


@dataclass(frozen=True)
class EvidenceGraphEdge:
    source: str
    target: str
    relation: str
    file: str
    line: int


@dataclass(frozen=True)
class EvidenceGraph:
    nodes: tuple = ()
    edges: tuple = ()
    roots: tuple = ()
    terminals: tuple = ()


def evidence_graph_for(hops) -> EvidenceGraph:
    """Build the typed occurrence graph recorded by structural traversal."""
    hops = list(hops)
    indexes_by_name = {}
    parents = {}
    for index, hop in enumerate(hops):
        citation = hop.citation
        candidates = tuple(indexes_by_name.get(citation.parent_qualified_name, ()))
        containing = [
            candidate for candidate in candidates
            if hops[candidate].citation.file == citation.call_site_file
            and hops[candidate].citation.line_start <= citation.call_site_line
            <= (hops[candidate].citation.line_end or
                hops[candidate].citation.line_start)]
        if containing:
            parent = min(containing, key=lambda candidate: (
                (hops[candidate].citation.line_end or
                 hops[candidate].citation.line_start)
                - hops[candidate].citation.line_start, -candidate))
        elif candidates:
            parent = candidates[-1]
        else:
            parent = None
        if parent is not None:
            parents[index] = parent
        indexes_by_name.setdefault(citation.qualified_name, []).append(index)
    nodes = tuple(EvidenceGraphNode(
        id=f"G{index + 1}", symbol=hop.citation.qualified_name,
        occurrence=_occurrence_key(hop), file=hop.citation.file,
        line=hop.citation.line_start) for index, hop in enumerate(hops))
    edges = tuple(EvidenceGraphEdge(
        source=nodes[parent].id, target=nodes[index].id,
        relation=hops[index].citation.relation,
        file=hops[index].citation.call_site_file,
        line=hops[index].citation.call_site_line)
        for index, parent in parents.items())
    targets = {edge.target for edge in edges}
    sources = {edge.source for edge in edges}
    return EvidenceGraph(
        nodes=nodes, edges=edges,
        roots=tuple(node.id for node in nodes if node.id not in targets),
        terminals=tuple(node.id for node in nodes if node.id not in sources))
def selection_for_graph_symbols(
        graph: EvidenceGraph, symbols, *, occurrence_keys=()) -> Selection:
    """Select exact route occurrences plus bounded recorded ancestor connectors."""
    wanted = set(symbols)
    wanted_occurrences = set(occurrence_keys)
    by_id = {node.id: node for node in graph.nodes}
    parent = {edge.target: edge.source for edge in graph.edges}
    if wanted_occurrences:
        starts = [node for node in graph.nodes
                  if node.occurrence in wanted_occurrences]
        connected_symbols = {node.symbol for node in starts if node.id in parent}
        for seed in tuple(starts):
            if seed.id in parent or seed.symbol in connected_symbols:
                continue
            candidates = tuple(node for node in graph.nodes if node.symbol == seed.symbol and node.id != seed.id)
            if not candidates:
                continue
            candidate = min(candidates, key=lambda node: (
                node.occurrence[8] if len(node.occurrence) > 8 else 0,
                node.occurrence[6] if len(node.occurrence) > 6 else 0,
                node.id))
            starts.append(candidate)
            connected_symbols.add(seed.symbol)
    else:
        starts = [node for node in graph.nodes if node.symbol in wanted]
    selected = set()
    for node in starts:
        current = node.id
        while current and current not in selected:
            selected.add(current)
            current = parent.get(current)
    chosen = [node for node in graph.nodes if node.id in selected]
    return Selection(
        symbols=list(dict.fromkeys(node.symbol for node in chosen)),
        occurrence_keys=tuple(node.occurrence for node in chosen))


def merge_selections(first: Selection, second: Selection) -> Selection:
    """Union two evidence selections without dropping route or section identity."""
    return Selection(
        symbols=list(dict.fromkeys((*first.symbols, *second.symbols))),
        sections=list(dict.fromkeys((*first.sections, *second.sections))),
        unknown=tuple(dict.fromkeys((*first.unknown, *second.unknown))),
        route_ids=tuple(dict.fromkeys((*first.route_ids, *second.route_ids))),
        section_ids=tuple(dict.fromkeys((*first.section_ids, *second.section_ids))),
        occurrence_keys=tuple(dict.fromkeys(
            (*first.occurrence_keys, *second.occurrence_keys))))
@dataclass(frozen=True)
class RouteMenu:
    text: str = ""
    routes: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)
    mandatory_symbols: tuple = ()
    route_occurrences: dict = field(default_factory=dict)
    route_sections: dict = field(default_factory=dict)
    section_titles: dict = field(default_factory=dict)
    route_summaries: dict = field(default_factory=dict)
def _occurrence_key(hop):
    citation = hop.citation
    return (citation.qualified_name, citation.file, citation.line_start,
            citation.line_end, citation.parent_qualified_name,
            citation.call_site_file, citation.call_site_line, citation.relation,
            citation.hop, citation.stop_reason)
def _render_route_cards(
        routes, *, route_sections=None, section_titles=None,
        route_summaries=None, sections=None,
        header="SCIP ROUTES — choose route IDs required by the question.") -> str:
    """Render exact routes with semantic symbols adjacent to each route ID."""
    route_sections = dict(route_sections or {})
    section_titles = dict(section_titles or {})
    route_summaries = dict(route_summaries or {})
    sections = dict(sections or {})
    lines = [header]
    for label, route in routes.items():
        shown = " -> ".join(route)
        local_sections = route_sections.get(label, ())
        section_suffix = (
            f" [sections {','.join(local_sections)}]" if local_sections else "")
        summary = route_summaries.get(label, "")
        summary_suffix = f" — {summary[:SUMMARY_CHARS]}" if summary else ""
        lines.append(
            f"  {label}. {shown}{section_suffix}{summary_suffix}")
    if sections:
        lines.extend(("", "ROUTE-LOCAL SECTION TITLES:"))
        lines.extend(
            f"  {label}. {section_titles.get(label, '')}"
            for label in sections)
    return "\n".join(lines)
def route_menu_for(library, hops, *, source: str) -> RouteMenu:
    """Offer complete, occurrence-preserving SCIP paths and route-local sections."""
    hops = list(hops)
    indexes_by_name = {}
    parents = {}
    children = set()
    mandatory = []
    for index, hop in enumerate(hops):
        citation = hop.citation
        candidates = tuple(indexes_by_name.get(citation.parent_qualified_name, ()))
        containing = [
            candidate for candidate in candidates
            if hops[candidate].citation.file == citation.call_site_file
            and hops[candidate].citation.line_start <= citation.call_site_line
            <= (hops[candidate].citation.line_end or hops[candidate].citation.line_start)]
        if containing:
            parent = min(
                containing,
                key=lambda candidate: (
                    (hops[candidate].citation.line_end or hops[candidate].citation.line_start)
                    - hops[candidate].citation.line_start,
                    -candidate))
        elif candidates:
            parent = candidates[-1]
        else:
            parent = None
        if parent is not None:
            parents[index] = parent
            children.add(parent)
        indexes_by_name.setdefault(citation.qualified_name, []).append(index)
        if citation.relation in ("localized", "shared_reference"):
            mandatory.append(citation.qualified_name)
    terminals = [
        index for index, hop in enumerate(hops)
        if hop.citation.stop_reason in ("leaf", "depth", "reference", "plumbing")
        or index not in children]
    legacy = menu_for(library, hops, source=source, include_summaries=False)
    routes = {}
    route_occurrences = {}
    route_sections = {}
    route_summaries = {}
    semantic_routes = {}
    for terminal in terminals:
        path = []
        current = terminal
        while current not in path:
            path.append(current)
            if current not in parents:
                break
            current = parents[current]
        path.reverse()
        occurrences = tuple(_occurrence_key(hops[index]) for index in path)
        if not occurrences:
            continue
        names = tuple(hops[index].citation.qualified_name for index in path)
        files = {hops[index].citation.file for index in path}
        section_labels = tuple(
            section_label for section_label, section_files in legacy.section_files.items()
            if files.intersection(section_files))
        existing = semantic_routes.get(names)
        if existing is not None:
            route_occurrences[existing] = tuple(dict.fromkeys(
                (*route_occurrences[existing], *occurrences)))
            route_sections[existing] = tuple(dict.fromkeys(
                (*route_sections[existing], *section_labels)))
            continue
        label = f"R{len(routes) + 1}"
        semantic_routes[names] = label
        routes[label] = names
        route_occurrences[label] = occurrences
        route_sections[label] = section_labels
        route_summaries[label] = _summary(hops[terminal].evidence or "")
    text = _render_route_cards(
        routes, route_sections=route_sections,
        section_titles=legacy.section_titles,
        route_summaries=route_summaries,
        sections=legacy.sections,
        header="SCIP ROUTES   choose route IDs whose endpoints and transitions answer the question.")
    return RouteMenu(
        text=text, routes=routes, sections=legacy.sections,
        mandatory_symbols=tuple(dict.fromkeys(mandatory)),
        route_occurrences=route_occurrences,
        route_sections=route_sections,
        section_titles=legacy.section_titles,
        route_summaries=route_summaries)

def resolve_route_selection(menu: RouteMenu, reply: str) -> Selection:
    # A selected route already contains its localized/shared-reference nodes.
    # Injecting mandatory nodes from unrelated routes recreates a disconnected full graph.
    symbols = []
    sections = []
    unknown = []
    route_ids = []
    section_ids = []
    occurrence_keys = []
    for prefix, digits in re.findall(r"\b([RS])(\d{1,4})\b", reply or "", re.I):
        label = f"{prefix.upper()}{digits}"
        if prefix.upper() == "R":
            route = menu.routes.get(label)
            if route is None:
                unknown.append(label)
            else:
                if label not in route_ids:
                    route_ids.append(label)
                    occurrence_keys.extend(menu.route_occurrences.get(label, ()))
                symbols.extend(name for name in route if name not in symbols)
        else:
            section = menu.sections.get(label)
            if section is None:
                unknown.append(label)
            elif section not in sections:
                section_ids.append(label)
                sections.append(section)
    return Selection(symbols=symbols, sections=sections, unknown=tuple(unknown),
                     route_ids=tuple(route_ids), section_ids=tuple(section_ids),
                     occurrence_keys=tuple(occurrence_keys))

def render_selected_routes(hops, selection: Selection, fetched: Fetched,
                           max_chars: int | None = None) -> str:
    chosen = set(selection.symbols)
    occurrences = set(selection.occurrence_keys)
    selected_hops = [hop for hop in hops
                     if (_occurrence_key(hop) in occurrences if occurrences
                         else hop.citation.qualified_name in chosen)]
    text = render_selected(selected_hops, selection, fetched, max_chars=max_chars)
    omitted = len(hops) - len(selected_hops)
    if omitted and (max_chars is None or len(text) + 80 <= max_chars):
        text += f"\n... {omitted} hop occurrence(s) belong to unselected SCIP routes."
    return text
def project_selected_evidence(evidence, selection: Selection, *,
                              hydrated_hops=None, source_gaps=()):
    """Return the exact evidence projection supplied to formulation."""
    from dataclasses import replace
    from library.chain_answer import locations_for

    chosen = set(selection.symbols)
    occurrences = set(selection.occurrence_keys)
    available_hops = evidence.hops if hydrated_hops is None else hydrated_hops
    hops = tuple(hop for hop in available_hops
                 if (_occurrence_key(hop) in occurrences if occurrences
                     else hop.citation.qualified_name in chosen))
    citations = [hop.citation for hop in hops]
    selected_names = {citation.qualified_name for citation in citations}
    selected_files = {citation.file for citation in citations}
    fan_outs = tuple(fan_out for fan_out in evidence.fan_outs
                     if fan_out.qualified_name in selected_names)
    mandatory_fan_outs = tuple(
        fan_out for fan_out in evidence.mandatory_fan_outs
        if fan_out.qualified_name in selected_names)
    caller_frontiers = tuple(
        frontier for frontier in evidence.caller_frontiers
        if frontier in selected_names)
    source_gaps = tuple(source_gaps) if hydrated_hops is not None else tuple(
        gap for gap in evidence.source_gaps
        if any(path in gap for path in selected_files))
    return replace(
        evidence, hops=hops, bundle_citations=citations,
        locations=locations_for(hops), themes=[], unresolved_paths=(),
        truncation_reason="", fan_outs=fan_outs,
        mandatory_fan_outs=mandatory_fan_outs,
        caller_frontiers=caller_frontiers, source_gaps=source_gaps)
def reference_paths_to_bridges(references, bridges):
    """Keep only compiler reference chains that terminate at selected bridges."""
    needed = {
        citation.parent_qualified_name for citation in bridges
        if (citation.stop_reason == "reference_bridge"
            and citation.relation == "shared_reference")}
    selected = set()
    changed = True
    while changed:
        changed = False
        for index, citation in enumerate(references):
            if index in selected or citation.qualified_name not in needed:
                continue
            selected.add(index)
            needed.add(citation.parent_qualified_name)
            changed = True
    retained = []
    pairs = set()
    for index, citation in enumerate(references):
        pair = (citation.parent_qualified_name, citation.qualified_name)
        if index in selected and pair not in pairs:
            pairs.add(pair)
            retained.append(citation)
    return tuple(retained)
def hydrate_selected_hops(library, hops, selection: Selection, *,
                          source: str, source_root: str | None,
                          definition_body_symbols=None,
                          definition_body_query=None,
                          reference_query: str = ""):
    # Fetch selected bodies plus bounded compiler dispatch and reference closure.
    from library.chain_bundle import curate_bundle
    from library.structural_assembly import (
        qualified_call_fanout, qualified_owner_closure,
        qualified_reference_fanout, qualified_reverse_reference_fanout,
        qualified_same_owner_reference_fanout,
        selected_route_branch_fanout)

    occurrences = set(selection.occurrence_keys)
    chosen = set(selection.symbols)
    selected = [hop for hop in hops
                if (_occurrence_key(hop) in occurrences if occurrences
                    else hop.citation.qualified_name in chosen)]
    citations = [hop.citation for hop in selected]
    selected_reference_targets = tuple(dict.fromkeys(
        hop.citation.qualified_name for hop in selected
        if hop.citation.relation in ("references", "shared_reference")))
    body_symbols = (None if definition_body_symbols is None else
                    tuple(dict.fromkeys(definition_body_symbols)))
    materialized_body_symbols = body_symbols
    connection_provider = getattr(library, "_conn_provider", None)
    if body_symbols and connection_provider is not None:
        query = str(reference_query or "").strip()
        with library._conn_provider.acquire() as conn:
            dependencies = qualified_call_fanout(
                conn, body_symbols, source=source, depth=1,
                recursive_per_root=0, max_recursive_total=0)
            body_marks = ",".join("?" * len(body_symbols))
            abstract_body_symbols = {
                str(row[0]) for row in conn.execute(
                    f"SELECT qualified_name FROM scip_symbols "
                    f"WHERE source_name=? AND qualified_name IN ({body_marks}) "
                    "AND kind='AbstractMethod' AND canonical_id NOT GLOB ?",
                    [source, *body_symbols, "local *"])}
            implementation_symbols = tuple(dict.fromkeys(
                citation.qualified_name for citation in dependencies
                if citation.relation == "implements"
                and citation.parent_qualified_name in abstract_body_symbols))
            implementation_dependencies = (
                qualified_call_fanout(
                    conn, implementation_symbols, source=source, depth=1,
                    per_root=8, recursive_per_root=0,
                    max_recursive_total=0)
                if implementation_symbols else ())
            branch_citations = selected_route_branch_fanout(
                conn, (), source=source, roots=body_symbols,
                per_root=1, child_per_sibling=0)
            branch_siblings = {
                citation.qualified_name for citation in branch_citations
                if citation.stop_reason == "selected_branch_sibling"}
            branch_body_symbols = ()
            reference_roots = tuple(dict.fromkeys((
                *body_symbols, *implementation_symbols)))
            references = qualified_reference_fanout(
                conn, reference_roots, source=source, question=query,
                depth=1, recursive_per_root=0, max_recursive_total=0)
            selected_reference_set = set(selected_reference_targets)
            body_reference_targets = tuple(dict.fromkeys(
                citation.qualified_name for citation in references
                if citation.qualified_name not in selected_reference_set))
            selected_reverse_references = (
                qualified_reverse_reference_fanout(
                    conn, selected_reference_targets,
                    source=source, question=query,
                    per_root=2, owner_per_root=1,
                    max_total=min(len(selected_reference_targets) * 2, 12))
                if selected_reference_targets else ())
            body_reverse_references = (
                qualified_reverse_reference_fanout(
                    conn, body_reference_targets,
                    source=source, question=query,
                    per_root=1, owner_per_root=1, max_total=8, lift_members = True, reserve_registrars = True)
                if body_reference_targets else ())
            reverse_references = (
                *selected_reverse_references, *body_reverse_references)
            reverse_consumer_symbols = tuple(dict.fromkeys(
                citation.qualified_name
                for citation in selected_reverse_references))
            reverse_consumer_references = (
                qualified_same_owner_reference_fanout(
                    conn, reverse_consumer_symbols,
                    source=source, question=query, per_root=4, excluded_targets = selected_reference_targets)
                if reverse_consumer_symbols else ())
            materialized_body_symbols = tuple(dict.fromkeys((
                *body_symbols,
                *implementation_symbols,
                *(citation.qualified_name
                  for citation in implementation_dependencies),
                *branch_body_symbols,
                *(citation.qualified_name
                  for citation in reverse_references),
                *(citation.qualified_name
                  for citation in reverse_consumer_references))))
            discovered = (
                *dependencies, *implementation_dependencies,
                *branch_citations, *references, *reverse_references,
                *reverse_consumer_references)
            citations.extend(discovered)
            citations.extend(qualified_owner_closure(
                conn,
                (*body_symbols,
                 *(citation.qualified_name for citation in discovered)),
                source=source))
    bundle = curate_bundle(
        library, citations, source=source,
        source_root=source_root, materialize_source=True,
        materialize_definition_bodies=True,
        definition_body_symbols=materialized_body_symbols,
        definition_body_query=definition_body_query)
    return tuple(bundle.hops), tuple(bundle.source_gaps)


def complete_selection_with_body_dependencies(
        selection: Selection, hydrated_hops, body_symbols, *,
        max_per_body: int = 4) -> Selection:
    # Retain selected-body dependencies, one dispatch layer, and proven owners.
    selected_bodies = tuple(dict.fromkeys(body_symbols or ()))
    if not selected_bodies:
        return selection

    by_calls = {}
    by_references = {}
    by_reverse_reference_target = {}
    owner_hops = {}
    member_owner_hops = {}
    branch_occurrences = {}
    for hop in hydrated_hops:
        citation = hop.citation
        if (citation.stop_reason == "selected_route_fanout"
                and citation.relation in ("calls", "implements")):
            by_calls.setdefault(citation.parent_qualified_name, []).append(hop)
        elif (citation.stop_reason == "selected_reference"
              and citation.relation == "references"):
            by_references.setdefault(
                citation.parent_qualified_name, []).append(hop)
        elif (citation.stop_reason == "selected_reference_caller"
              and citation.relation == "referenced_by"):
            by_reverse_reference_target.setdefault(
                citation.parent_qualified_name, []).append(hop)
        elif citation.stop_reason in (
                "selected_branch_caller", "selected_branch_sibling"):
            branch_occurrences[_occurrence_key(hop)] = True
        elif (citation.stop_reason == "selected_owner"
              and citation.relation == "localized"):
            owner_hops[citation.qualified_name] = hop
        elif (citation.stop_reason == "selected_owner_member"
              and citation.relation == "contains"):
            member_owner_hops[citation.qualified_name] = hop

    symbols = []
    direct_occurrences = {}
    reference_targets = [
        target for target in by_reverse_reference_target
        if target in selection.symbols]
    implementation_symbols = []
    selected_occurrences = set(selection.occurrence_keys)
    for hop in hydrated_hops:
        citation = hop.citation
        if (_occurrence_key(hop) in selected_occurrences
                and citation.relation in ("references", "shared_reference")
                and citation.qualified_name not in reference_targets):
            reference_targets.append(citation.qualified_name)

    def retain_calls(parent):
        count = 0
        for hop in by_calls.get(parent, ()):
            citation = hop.citation
            key = _occurrence_key(hop)
            if key in direct_occurrences:
                continue
            if count >= max(max_per_body, 0):
                break
            count += 1
            direct_occurrences[key] = True
            target = citation.qualified_name
            if target not in symbols:
                symbols.append(target)
            if (citation.relation == "implements"
                    and target not in implementation_symbols):
                implementation_symbols.append(target)

    for parent in selected_bodies:
        retain_calls(parent)
        for hop in by_references.get(parent, ()):
            key = _occurrence_key(hop)
            direct_occurrences[key] = True
            target = hop.citation.qualified_name
            if target not in symbols:
                symbols.append(target)
            if target not in reference_targets:
                reference_targets.append(target)
            member = member_owner_hops.get(target)
            if member is not None:
                owner_name = member.citation.parent_qualified_name
                if owner_name not in reference_targets:
                    reference_targets.append(owner_name)
    for implementation in tuple(implementation_symbols):
        retain_calls(implementation)

    reverse_occurrences = {}
    reverse_frontier = list(reference_targets)
    reverse_expanded = set()
    reverse_consumers = []
    while reverse_frontier:
        parent = reverse_frontier.pop(0)
        if parent in reverse_expanded:
            continue
        reverse_expanded.add(parent)
        for caller in by_reverse_reference_target.get(parent, ()):
            reverse_occurrences[_occurrence_key(caller)] = True
            caller_name = caller.citation.qualified_name
            if caller_name not in symbols:
                symbols.append(caller_name)
            if caller_name not in reverse_consumers:
                reverse_consumers.append(caller_name)
            member = member_owner_hops.get(caller_name)
            if member is None:
                continue
            owner_name = member.citation.parent_qualified_name
            owner = owner_hops.get(owner_name)
            if owner is None:
                continue
            reverse_occurrences[_occurrence_key(owner)] = True
            reverse_occurrences[_occurrence_key(member)] = True
            if owner_name not in symbols:
                symbols.append(owner_name)
            if (owner_name not in reverse_expanded
                    and owner_name not in reverse_frontier):
                reverse_frontier.append(owner_name)
    reference_frontier = list(reverse_consumers)
    reference_expanded = set()
    while reference_frontier:
        parent = reference_frontier.pop(0)
        if parent in reference_expanded:
            continue
        reference_expanded.add(parent)
        for hop in by_references.get(parent, ()):
            key = _occurrence_key(hop)
            direct_occurrences[key] = True
            target = hop.citation.qualified_name
            if target not in symbols:
                symbols.append(target)
            if (target not in reference_expanded
                    and target not in reference_frontier):
                reference_frontier.append(target)

    retained_names = set((*selection.symbols, *selected_bodies, *symbols))
    owner_symbols = []
    owner_occurrences = {}
    for member_name in sorted(retained_names):
        member = member_owner_hops.get(member_name)
        if member is None:
            continue
        owner = owner_hops.get(member.citation.parent_qualified_name)
        if owner is None:
            continue
        owner_name = owner.citation.qualified_name
        if owner_name not in owner_symbols:
            owner_symbols.append(owner_name)
        owner_occurrences[_occurrence_key(owner)] = True
        owner_occurrences[_occurrence_key(member)] = True

    return merge_selections(
        selection,
        Selection(
            symbols=[*owner_symbols, *symbols],
            occurrence_keys=tuple((
                *owner_occurrences,
                *branch_occurrences,
                *direct_occurrences,
                *reverse_occurrences))))
def all_route_selection(menu: RouteMenu) -> Selection:
    """The explicit, occurrence-preserving selection of every offered route."""
    symbols = []
    occurrences = []
    for label, route in menu.routes.items():
        symbols.extend(name for name in route if name not in symbols)
        occurrences.extend(menu.route_occurrences.get(label, ()))
    return Selection(
        symbols=symbols, route_ids=tuple(menu.routes),
        occurrence_keys=tuple(occurrences))
def _route_root(qualified_name: str) -> str:
    """Stable semantic owner for grouping alternative occurrence routes."""
    parts = qualified_name.split(".")
    member_names = {
        "apply", "run", "execute", "process", "write", "read", "commit",
        "build", "create", "resolve", "rewrite", "plan", "prepare", "<init>"}
    if (len(parts) > 1
            and (parts[-1].lower() in member_names
                 or (parts[-1] and parts[-1][0].islower()))):
        return parts[-2]
    return parts[-1]


def _semantic_tokens(text: str) -> set[str]:
    parts = re.findall(
        r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+", text or "")
    stop = {"a", "an", "and", "does", "how", "into", "normal", "of", "or",
            "the", "to", "use", "why", "with"}
    aliases = {"analysis": "analyze", "analysed": "analyze", "rewriting": "rewrite"}
    return {aliases.get(part.lower(), part.lower()) for part in parts
            if part.lower() not in stop and len(part) > 1}
def complete_route_selection(
        menu: RouteMenu, selection: Selection, question: str, *,
        max_branches: int = 4, max_sibling_branches: int = 2) -> Selection:
    """Complete semantic roots and bounded compiler forks for comparisons.

    The model remains authoritative about detail. This guard prevents either a
    question-named root or one side of a small compiler dispatch fork from
    disappearing. Fork completion is enabled only by explicit comparison
    language and refuses ambiguous owners with more alternatives than the
    configured bound.
    """
    if not menu.routes:
        return selection
    question_tokens = _semantic_tokens(question)
    selected_ids = list(selection.route_ids)
    selected_signatures = {
        frozenset(question_tokens.intersection(_semantic_tokens(
            _route_root(menu.routes[label][0]))))
        for label in selected_ids if label in menu.routes and menu.routes[label]}
    representatives = {}
    if max_branches > 0:
        for label, route in menu.routes.items():
            if not route:
                continue
            root = _route_root(route[0])
            signature = frozenset(
                question_tokens.intersection(_semantic_tokens(root)))
            occurrence_files = {
                str(key[1]) for key in menu.route_occurrences.get(label, ())
                if len(key) > 1}
            if (len(signature) < 2 or signature in selected_signatures
                    or root.lower().endswith(("suite", "test", "tests"))
                    or any("/test/" in file or "/tests/" in file
                           for file in occurrence_files)):
                continue
            candidate = (len(signature), -len(route), label)
            if (signature not in representatives
                    or candidate > representatives[signature][0]):
                representatives[signature] = (candidate, label)
    ranked = sorted(
        representatives.values(),
        key=lambda item: (-item[0][0], -item[0][1],
                          int(item[1][1:])))[:max(max_branches, 0)]
    symbols = list(selection.symbols)
    occurrences = list(selection.occurrence_keys)

    def retain(label):
        if label in selected_ids:
            return
        selected_ids.append(label)
        for symbol in menu.routes[label]:
            if symbol not in symbols:
                symbols.append(symbol)
        occurrences.extend(menu.route_occurrences.get(label, ()))

    for _, label in ranked:
        retain(label)

    comparison = re.search(
        r"\b(?:versus|vs\.?|compare(?:d|s|ing)?|comparison|"
        r"differences?|respectively)\b", question, re.I)
    sibling_additions = 0
    if comparison and max_sibling_branches > 0:
        for selected_label in tuple(selected_ids):
            selected_route = menu.routes.get(selected_label, ())
            if len(selected_route) < 2:
                continue
            root = selected_route[0]
            selected_branches = {
                menu.routes[label][1]
                for label in selected_ids
                if label in menu.routes
                and len(menu.routes[label]) >= 2
                and menu.routes[label][0] == root}
            alternatives = {}
            for label, route in menu.routes.items():
                if (label in selected_ids or len(route) < 2
                        or route[0] != root or route[1] in selected_branches):
                    continue
                occurrence_files = {
                    str(key[1]) for key in menu.route_occurrences.get(label, ())
                    if len(key) > 1}
                if any("/test/" in file or "/tests/" in file
                       for file in occurrence_files):
                    continue
                candidate = (
                    abs(len(route) - len(selected_route)),
                    len(route), int(label[1:]), label)
                current = alternatives.get(route[1])
                if current is None or candidate < current:
                    alternatives[route[1]] = candidate
            if (not alternatives
                    or len(alternatives) > max_sibling_branches):
                continue
            for *_, label in sorted(alternatives.values()):
                retain(label)
                sibling_additions += 1
                if sibling_additions >= max_sibling_branches:
                    break
            if sibling_additions >= max_sibling_branches:
                break
    return Selection(
        symbols=symbols, sections=list(selection.sections),
        unknown=selection.unknown, route_ids=tuple(selected_ids),
        section_ids=selection.section_ids,
        occurrence_keys=tuple(dict.fromkeys(occurrences)))
@dataclass(frozen=True)
class ComponentMenu:
    text: str = ""
    components: dict = field(default_factory=dict)
    routes: dict = field(default_factory=dict)
def evidence_graph_report(graph: EvidenceGraph, seed_provenance=()) -> dict:
    """Serialize every typed node and edge, grouped by weak component."""
    provenance_by_symbol = {}
    for item in seed_provenance:
        symbol = str(item.get("symbol", ""))
        for origin in item.get("origins", ()):
            provenance_by_symbol.setdefault(symbol, set()).add(str(origin))
    neighbors = {node.id: set() for node in graph.nodes}
    for edge in graph.edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
    by_id = {node.id: node for node in graph.nodes}
    remaining = set(by_id)
    components = []
    for start in (node.id for node in graph.nodes):
        if start not in remaining:
            continue
        stack, found = [start], set()
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            found.add(current)
            stack.extend(neighbors[current])
        nodes = [node for node in graph.nodes if node.id in found]
        edges = [edge for edge in graph.edges
                 if edge.source in found and edge.target in found]
        component_origins = {}
        for node in nodes:
            for origin in provenance_by_symbol.get(node.symbol, ()):
                component_origins.setdefault(origin, set()).add(node.symbol)
        components.append({
            "id": f"C{len(components) + 1}",
            "seed_origins": {origin: sorted(symbols)
                             for origin, symbols in sorted(component_origins.items())},
            "roots": [by_id[node_id].symbol for node_id in graph.roots
                      if node_id in found],
            "terminals": [by_id[node_id].symbol for node_id in graph.terminals
                          if node_id in found],
            "nodes": [{"id": node.id, "symbol": node.symbol,
                       "file": node.file, "line": node.line}
                      for node in nodes],
            "edges": [{"source": by_id[edge.source].symbol,
                       "target": by_id[edge.target].symbol,
                       "relation": edge.relation, "file": edge.file,
                       "line": edge.line} for edge in edges],
        })
    return {"node_count": len(graph.nodes), "edge_count": len(graph.edges),
            "component_count": len(components), "components": components}
def component_menu_for(graph: EvidenceGraph, menu: RouteMenu) -> ComponentMenu:
    """Summarize weakly connected occurrences with semantic boundary names."""
    neighbors = {node.id: set() for node in graph.nodes}
    for edge in graph.edges:
        neighbors[edge.source].add(edge.target)
        neighbors[edge.target].add(edge.source)
    by_id = {node.id: node for node in graph.nodes}
    remaining = set(by_id)
    components = {}
    routes = {}
    cards = []
    for start in (node.id for node in graph.nodes):
        if start not in remaining:
            continue
        stack = [start]
        found = []
        while stack:
            current = stack.pop()
            if current not in remaining:
                continue
            remaining.remove(current)
            found.append(current)
            stack.extend(sorted(neighbors[current], reverse=True))
        found_set = set(found)
        ordered = tuple(node.id for node in graph.nodes if node.id in found_set)
        label = f"G{len(components) + 1}"
        components[label] = ordered
        occurrences = {by_id[node_id].occurrence for node_id in ordered}
        symbols = {by_id[node_id].symbol for node_id in ordered}
        route_ids = []
        for route_id, route in menu.routes.items():
            route_occurrences = set(menu.route_occurrences.get(route_id, ()))
            if route_occurrences:
                belongs = bool(occurrences.intersection(route_occurrences))
            else:
                belongs = bool(symbols.intersection(route))
            if belongs:
                route_ids.append(route_id)
        routes[label] = tuple(route_ids)
        roots = [by_id[node_id].symbol for node_id in ordered
                 if node_id in graph.roots]
        terminals = [by_id[node_id].symbol for node_id in ordered
                     if node_id in graph.terminals]
        boundary = set(roots + terminals)
        middle = [by_id[node_id].symbol for node_id in ordered
                  if by_id[node_id].symbol not in boundary]
        parts = []
        if roots:
            parts.append("entry " + ", ".join(
                _owner_qualified(name) for name in roots[:2]))
        if middle:
            parts.append("via " + ", ".join(
                _owner_qualified(name) for name in middle[:4]))
        if terminals:
            parts.append("terminal " + ", ".join(
                _owner_qualified(name) for name in terminals[:4]))
        cards.append(f"  {label}. " + "; ".join(parts))
    lines = [
        "SCIP CONNECTED COMPONENTS; choose every component needed by the question.",
        *cards]
    return ComponentMenu(
        text="\n".join(lines), components=components, routes=routes)
def resolve_component_selection(components: ComponentMenu, reply: str) -> tuple[str, ...]:
    """Expand selected graph IDs while keeping obligation IDs unambiguous."""
    digits = re.findall(r"\bG(\d{1,5})\b", reply or "", re.I)
    legacy = not digits and ":" not in (reply or "")
    if legacy:
        digits = re.findall(r"\bC(\d{1,5})\b", reply or "", re.I)
    route_ids = []
    for value in digits:
        key = f"C{value}" if legacy and f"C{value}" in components.routes else f"G{value}"
        for route_id in components.routes.get(key, ()):
            if route_id not in route_ids:
                route_ids.append(route_id)
    return tuple(route_ids)
@dataclass(frozen=True)
class ModuleMenu:
    text: str = ""
    modules: dict = field(default_factory=dict)
    owners: dict = field(default_factory=dict)
    symbols: dict = field(default_factory=dict)
def module_menu_for(menu: RouteMenu, *, graph: EvidenceGraph | None = None) -> ModuleMenu:
    """Expose exact SCIP members with typed graph-neighbor context."""
    by_symbol = {}
    symbol_sections = {}
    for route_id, route in menu.routes.items():
        for symbol in dict.fromkeys(route):
            by_symbol.setdefault(symbol, []).append(route_id)
            labels = symbol_sections.setdefault(symbol, [])
            for section in menu.route_sections.get(route_id, ()):
                title = menu.section_titles.get(section, "")
                if title and title not in labels:
                    labels.append(title)
    incoming = {}
    outgoing = {}
    if graph is not None:
        by_id = {node.id: node for node in graph.nodes}
        for edge in graph.edges:
            source = by_id[edge.source].symbol
            target = by_id[edge.target].symbol
            incoming.setdefault(target, []).append((source, edge.relation))
            outgoing.setdefault(source, []).append((target, edge.relation))
    modules = {}
    owners = {}
    symbols = {}
    lines = ["SCIP METHOD GRAPH; choose every exact member needed by the question."]
    for symbol, route_ids in by_symbol.items():
        label = f"M{len(modules) + 1}"
        modules[label] = tuple(route_ids)
        owners[label] = _route_root(symbol)
        symbols[label] = symbol
        contexts = []
        for neighbor, relation in incoming.get(symbol, ())[:1]:
            verb = transition_verb(relation)
            contexts.append(f"{_owner_qualified(neighbor)} -{verb}-> {_owner_qualified(symbol)}")
        for neighbor, relation in outgoing.get(symbol, ())[:1]:
            verb = transition_verb(relation)
            contexts.append(f"{_owner_qualified(symbol)} -{verb}-> {_owner_qualified(neighbor)}")
        shown = " | ".join(contexts) if contexts else _owner_qualified(symbol)
        titles = symbol_sections.get(symbol, ())[:2]
        suffix = f" — {'; '.join(titles)}" if titles else ""
        lines.append(f"  {label}. {shown}{suffix}")
    return ModuleMenu(
        text="\n".join(lines), modules=modules, owners=owners, symbols=symbols)
def resolve_module_symbols(modules: ModuleMenu, reply: str) -> tuple[str, ...]:
    """Return the exact SCIP members explicitly selected from method cards."""
    symbols = []
    for digits in re.findall(r"\bM(\d{1,5})\b", reply or "", re.I):
        symbol = modules.symbols.get(f"M{digits}")
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return tuple(symbols)
def resolve_module_owners(modules: ModuleMenu, reply: str) -> tuple[str, ...]:
    """Return the semantic owners explicitly named by a module reply."""
    owners = []
    for digits in re.findall(r"\bM(\d{1,5})\b", reply or "", re.I):
        owner = modules.owners.get(f"M{digits}")
        if owner and owner not in owners:
            owners.append(owner)
    return tuple(owners)


def resolve_module_selection(modules: ModuleMenu, reply: str,
                             route_menu: RouteMenu) -> tuple[str, ...]:
    """Resolve module IDs, accepting direct route IDs for compatibility."""
    route_ids = []
    for prefix, digits in re.findall(r"\b([MR])(\d{1,5})\b", reply or "", re.I):
        label = f"{prefix.upper()}{digits}"
        candidates = (modules.modules.get(label, ()) if prefix.upper() == "M" else ((label,) if label in route_menu.routes else ()))
        for route_id in candidates:
            if route_id not in route_ids:
                route_ids.append(route_id)
    return tuple(route_ids)
def routes_for_modules(menu: RouteMenu, route_ids) -> RouteMenu:
    """Expand selected components into routes with summaries and section titles."""
    chosen = [label for label in menu.routes if label in set(route_ids)]
    sections = tuple(dict.fromkeys(
        section for label in chosen
        for section in menu.route_sections.get(label, ())))
    routes = {label: menu.routes[label] for label in chosen}
    route_sections = {
        label: menu.route_sections.get(label, ()) for label in chosen}
    route_summaries = {
        label: menu.route_summaries.get(label, "") for label in chosen}
    section_map = {
        label: menu.sections[label] for label in sections
        if label in menu.sections}
    titles = {
        label: menu.section_titles.get(label, "") for label in section_map}
    lines = ["SCIP ROUTES — expanded from selected graph components."]
    for label, route in routes.items():
        shown = " -> ".join(_owner_qualified(name) for name in route)
        local_sections = route_sections.get(label, ())
        section_suffix = (
            f" [sections {','.join(local_sections)}]" if local_sections else "")
        summary = route_summaries.get(label, "")
        summary_suffix = f" — {summary[:SUMMARY_CHARS]}" if summary else ""
        lines.append(f"  {label}. {shown}{section_suffix}{summary_suffix}")
    if section_map:
        lines.extend(("", "ROUTE-LOCAL SECTION TITLES:"))
        lines.extend(f"  {label}. {titles[label]}" for label in section_map)
    available = {name for route in routes.values() for name in route}
    return RouteMenu(
        text=_render_route_cards(routes, route_sections=route_sections, section_titles=titles, route_summaries=route_summaries, sections=section_map, header="SCIP ROUTES   expanded from selected graph components."), routes=routes, sections=section_map,
        mandatory_symbols=tuple(
            name for name in menu.mandatory_symbols if name in available),
        route_occurrences={
            label: menu.route_occurrences.get(label, ()) for label in chosen},
        route_sections=route_sections, section_titles=titles,
        route_summaries=route_summaries)
def scope_route_menu(menu: RouteMenu, question: str, *,
                     max_families: int = 32, route_scores = None, max_per_root = 4, required_owners = (), required_symbols = ()) -> RouteMenu:
    """Keep exact question symbols and one representative per semantic clause."""
    route_scores = dict(route_scores or {})
    question_tokens = _semantic_tokens(question)
    exact_terms = tuple(dict.fromkeys(
        match.group(0).rstrip(".?")
        for match in re.finditer(
            r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+|[A-Za-z_]*[A-Z][A-Za-z0-9_]*",
            question)
        if ("." in match.group(0) or "_" in match.group(0)
            or any(char.isupper() for char in match.group(0)[1:]))))
    def route_score(label, route, tokens):
        route_tokens = set().union(*(_semantic_tokens(name) for name in route)) if route else set()
        summary_tokens = _semantic_tokens(menu.route_summaries.get(label, ""))
        return (3 * len(tokens.intersection(route_tokens))
                + 2 * len(tokens.intersection(summary_tokens)))

    chosen = []
    for symbol in required_symbols:
        matches = [(len(route), int(label[1:]), label)
                   for label, route in menu.routes.items() if symbol in route]
        if matches:
            label = min(matches)[2]
            if label not in chosen:
                chosen.append(label)
    for owner in required_owners:
        owner_routes = []
        for label, route in menu.routes.items():
            owned_names = [name for name in route if _route_root(name) == owner]
            if not owned_names:
                continue
            entry = any(name.rsplit(".", 1)[-1].lower() in {
                "apply", "run", "execute", "process", "rewrite", "plan"}
                        for name in owned_names)
            owner_routes.append((not entry, len(route), int(label[1:]), label))
        if owner_routes:
            label = min(owner_routes)[3]
            if label not in chosen:
                chosen.append(label)
    for term in exact_terms:
        lowered = term.lower()
        matches = []
        for label, route in menu.routes.items():
            names = " ".join(route).lower()
            if lowered not in names:
                continue
            exact_segment = any(
                lowered in {part.lower() for part in name.split(".")}
                for name in route)
            matches.append((not exact_segment, len(route), int(label[1:]), label))
        if matches:
            label = min(matches)[3]
            if label not in chosen:
                chosen.append(label)

    clauses = [part for part in re.split(
        r"[?;:]|\b(?:and|versus|while|relative to)\b", question, flags=re.I)
        if _semantic_tokens(part)]
    for clause in clauses:
        tokens = _semantic_tokens(clause)
        ranked_clause = sorted(
            ((-route_score(label, route, tokens), len(route), int(label[1:]), label)
             for label, route in menu.routes.items()
             if route_score(label, route, tokens) > 0))
        if ranked_clause:
            label = ranked_clause[0][3]
            if label not in chosen:
                chosen.append(label)

    representatives = {}
    for label, route in menu.routes.items():
        route_tokens = set().union(*(_semantic_tokens(name) for name in route)) if route else set()
        section_labels = menu.route_sections.get(label, ())
        section_tokens = set().union(*(
            _semantic_tokens(menu.section_titles.get(section, ""))
            for section in section_labels)) if section_labels else set()
        symbol_overlap = question_tokens.intersection(route_tokens)
        section_overlap = question_tokens.intersection(section_tokens)
        summary_overlap = question_tokens.intersection(
                    _semantic_tokens(menu.route_summaries.get(label, "")))
        score = (3 * len(symbol_overlap) + 2 * len(section_overlap)
                         + 2 * len(summary_overlap)
                         + 5 * max(route_scores.get(label, 0.0), 0.0))
        if score <= 0:
            continue
        family = (
                    _route_root(route[0]) if route else "",
                    _route_root(route[-1]) if route else "",
                    tuple(sorted(symbol_overlap.union(section_overlap, summary_overlap))))
        candidate = (-score, len(route), int(label[1:]), label)
        if family not in representatives or candidate < representatives[family]:
            representatives[family] = candidate
    ranked_representatives = sorted(representatives.values())
    owner_best = {}
    for item in ranked_representatives:
        label = item[3]
        route = menu.routes.get(label, ())
        owners = tuple(dict.fromkeys((
            _route_root(route[0]) if route else "",
            _route_root(route[-1]) if route else "")))
        for owner in owners:
            if owner not in owner_best or item < owner_best[owner]:
                owner_best[owner] = item
    limit = max(max_families, len(exact_terms))
    for item in sorted(set(owner_best.values())):
        if len(chosen) >= limit:
            break
        if item[3] not in chosen:
            chosen.append(item[3])
    owner_counts = {}
    for label in chosen:
        route = menu.routes.get(label, ())
        for owner in set((
                _route_root(route[0]) if route else "",
                _route_root(route[-1]) if route else "")):
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
    for item in ranked_representatives:
        if len(chosen) >= limit:
            break
        label = item[3]
        if label in chosen:
            continue
        route = menu.routes.get(label, ())
        owners = set((
            _route_root(route[0]) if route else "",
            _route_root(route[-1]) if route else ""))
        if owners and all(owner_counts.get(owner, 0) >= max_per_root
                          for owner in owners):
            continue
        chosen.append(label)
        for owner in owners:
            owner_counts[owner] = owner_counts.get(owner, 0) + 1
    if not chosen:
        chosen = sorted(menu.routes, key=lambda label: (
            len(menu.routes[label]), int(label[1:])))[:max_families]
    chosen_sections = tuple(dict.fromkeys(
        section for label in chosen
        for section in menu.route_sections.get(label, ())))
    routes = {label: menu.routes[label] for label in chosen}
    sections = {label: menu.sections[label] for label in chosen_sections
                if label in menu.sections}
    titles = {label: menu.section_titles.get(label, "") for label in sections}
    route_sections = {
        label: tuple(section for section in menu.route_sections.get(label, ())
                     if section in sections)
        for label in chosen}
    route_summaries = {
            label: menu.route_summaries.get(label, "") for label in chosen}
    lines = ["SCIP ROUTES — scoped semantic families; choose every family required by the question."]
    for label, route in routes.items():
        shown = " -> ".join(_owner_qualified(name) for name in route)
        local_sections = route_sections.get(label, ())
        suffix = f" [sections {','.join(local_sections)}]" if local_sections else ""
        summary = route_summaries.get(label, "")
        summary_suffix = f" — {summary[:SUMMARY_CHARS]}" if summary else ""
        lines.append(f"  {label}. {shown}{suffix}{summary_suffix}")
    if sections:
        lines.extend(("", "ROUTE-LOCAL SECTION TITLES:"))
        lines.extend(f"  {label}. {titles[label]}" for label in sections)
    available_symbols = {name for route in routes.values() for name in route}
    return RouteMenu(
        text=_render_route_cards(routes, route_sections=route_sections, section_titles=titles, route_summaries=route_summaries, sections=sections, header="SCIP ROUTES   scoped semantic families; choose every family required by the question."), routes=routes, sections=sections,
        mandatory_symbols=tuple(name for name in menu.mandatory_symbols
                                if name in available_symbols),
        route_occurrences={label: menu.route_occurrences.get(label, ())
                           for label in chosen},
        route_sections=route_sections, section_titles=titles, route_summaries = route_summaries)
def route_section_embedding_scores(library, menu: RouteMenu, selection: Selection,
                                   question_embedding) -> dict[str, float]:
    """Cosine-rank route-local sections with batched reads from the existing index."""
    import numpy as np

    query = np.asarray(question_embedding, dtype=np.float32)
    query_norm = float(np.linalg.norm(query))
    if not query_norm:
        return {}
    labels = tuple(dict.fromkeys(
        label for route_id in selection.route_ids
        for label in menu.route_sections.get(route_id, ())))
    wanted = {menu.sections[label]: label for label in labels if label in menu.sections}
    document_ids = tuple(dict.fromkeys(document_id for document_id, _ in wanted))
    scores = {}
    with library._conn_provider.acquire() as conn:
        for start in range(0, len(document_ids), 400):
            chunk = document_ids[start:start + 400]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT document_id, idx, embedding FROM sections "
                f"WHERE document_id IN ({placeholders}) AND embedding IS NOT NULL",
                chunk).fetchall()
            for document_id, index, blob in rows:
                label = wanted.get((document_id, index))
                if label is None or not blob:
                    continue
                vector = np.frombuffer(blob, dtype=np.float32)
                if vector.shape != query.shape:
                    continue
                denominator = query_norm * float(np.linalg.norm(vector))
                if denominator:
                    scores[label] = float(np.dot(query, vector) / denominator)
    return scores
def select_route_sections(menu: RouteMenu, selection: Selection, question: str, *,
                          max_sections: int = 4, section_titles=None,
                          section_scores=None) -> Selection:
    """Attach embedding-ranked, route-local sections after SCIP route selection."""
    titles = dict(section_titles or menu.section_titles)
    scores = dict(section_scores or {})
    candidates = tuple(dict.fromkeys(section for route_id in selection.route_ids for section in menu.route_sections.get(route_id, ())))
    question_tokens = _semantic_tokens(question)

    def stem(token):
        if token.endswith("ing") and len(token) > 5:
            return token[:-3].rstrip("n")
        if token.endswith("ed") and len(token) > 4:
            return token[:-2]
        return token

    wanted = {stem(token) for token in question_tokens}
    ranked = []
    for label in candidates:
        title_tokens = {stem(token) for token in _semantic_tokens(titles.get(label, ""))}
        overlap = wanted.intersection(title_tokens)
        semantic = scores.get(label)
        if semantic is None and not overlap:
            continue
        if semantic is not None and semantic <= 0 and not overlap:
            continue
        ranked.append((-(semantic if semantic is not None else -1.0),
                       -len(overlap), int(label[1:]), label))
    labels = tuple(item[3] for item in sorted(ranked)[:max_sections])
    sections = []
    for label in labels:
        target = menu.sections.get(label)
        if target is not None and target not in sections:
            sections.append(target)
    return Selection(
        symbols=list(selection.symbols), sections=sections, unknown=selection.unknown,
        route_ids=selection.route_ids, section_ids=labels,
        occurrence_keys=selection.occurrence_keys)
def retain_symbol_routes(menu: RouteMenu, selection: Selection, symbols) -> Selection:
    """Make exact method-card choices additive to probabilistic route selection."""
    route_ids = list(selection.route_ids)
    selected_symbols = list(selection.symbols)
    occurrences = list(selection.occurrence_keys)
    for symbol in symbols:
        candidates = [(len(route), int(label[1:]), label) for label, route in menu.routes.items() if symbol in route]
        if not candidates:
            continue
        label = min(candidates)[2]
        if label not in route_ids:
            route_ids.append(label)
        for member in menu.routes[label]:
            if member not in selected_symbols:
                selected_symbols.append(member)
        occurrences.extend(menu.route_occurrences.get(label, ()))
    return Selection(
        symbols=selected_symbols, sections=list(selection.sections), unknown=selection.unknown,
        route_ids=tuple(route_ids), section_ids=selection.section_ids,
        occurrence_keys=tuple(dict.fromkeys(occurrences)))
def selection_owners(selection: Selection, owners=()) -> tuple[str, ...]:
    """Close explicit module owners over owners reached by selected routes."""
    result = list(owners)
    for symbol in selection.symbols:
        owner = _route_root(symbol)
        if owner and owner not in result:
            result.append(owner)
    return tuple(result)
def retain_owner_routes(menu: RouteMenu, selection: Selection, owners, question: str, *,
                        route_scores=None, max_per_owner: int = 2) -> Selection:
    """Keep an entry route and the strongest semantic route for selected owners."""
    if max_per_owner <= 0:
        return selection
    scores = dict(route_scores or {})
    route_ids = list(selection.route_ids)
    symbols = list(selection.symbols)
    occurrences = list(selection.occurrence_keys)
    question_tokens = _semantic_tokens(question)
    entry_names = {"apply", "run", "execute", "process", "rewrite", "plan", "commit"}
    for owner in owners:
        candidates = []
        for label, route in menu.routes.items():
            owned = [name for name in route if _route_root(name) == owner]
            if not owned:
                continue
            entry = any(name.rsplit(".", 1)[-1].lower() in entry_names for name in owned)
            overlap = len(question_tokens.intersection(
                set().union(*(_semantic_tokens(name) for name in route))))
            candidates.append((label, entry, overlap, scores.get(label, 0.0), len(route)))
        if not candidates:
            continue
        chosen = []
        entries = [item for item in candidates if item[1]]
        if entries:
            chosen.append(min(entries, key=lambda item: (item[4], int(item[0][1:]))))
        for item in sorted(candidates, key=lambda item: (
                -item[3], -item[2], item[4], int(item[0][1:]))):
            if item[0] not in {existing[0] for existing in chosen}:
                chosen.append(item)
            if len(chosen) >= max_per_owner:
                break
        for label, *_ in chosen[:max_per_owner]:
            if label in route_ids:
                continue
            route_ids.append(label)
            for symbol in menu.routes[label]:
                if symbol not in symbols:
                    symbols.append(symbol)
            occurrences.extend(menu.route_occurrences.get(label, ()))
    return Selection(
        symbols=symbols, sections=list(selection.sections), unknown=selection.unknown,
        route_ids=tuple(route_ids), section_ids=selection.section_ids,
        occurrence_keys=tuple(dict.fromkeys(occurrences)))
def retain_mandatory_routes(menu: RouteMenu, selection: Selection, *,
                            max_routes: int = 8) -> Selection:
    """Cover compiler-mandatory symbols with a bounded number of short routes.

    A mandatory symbol can occur in hundreds of alternative routes.  Retaining every
    occurrence turns a narrow probabilistic selection back into the complete graph and
    makes both hydration and formulation uneconomic.  Greedy set cover keeps the safety
    property while choosing at most ``max_routes`` representatives.
    """
    mandatory = set(menu.mandatory_symbols) - set(selection.symbols)
    if not mandatory or max_routes <= 0:
        return selection
    route_ids = list(selection.route_ids)
    symbols = list(selection.symbols)
    occurrences = list(selection.occurrence_keys)
    candidates = list(menu.routes)
    for _ in range(max_routes):
        ranked = []
        for label in candidates:
            route = menu.routes[label]
            covered = mandatory.intersection(route)
            if covered:
                ranked.append((-len(covered), len(route), int(label[1:]), label,
                               covered))
        if not ranked:
            break
        _, _, _, label, covered = min(ranked)
        route_ids.append(label)
        for symbol in menu.routes[label]:
            if symbol not in symbols:
                symbols.append(symbol)
        occurrences.extend(menu.route_occurrences.get(label, ()))
        mandatory.difference_update(covered)
        candidates.remove(label)
        if not mandatory:
            break
    return Selection(
        symbols=symbols, sections=list(selection.sections), unknown=selection.unknown,
        route_ids=tuple(route_ids), section_ids=selection.section_ids,
        occurrence_keys=tuple(dict.fromkeys(occurrences)))
@dataclass(frozen=True)
class DefinitionBodyMenu:
    """Compact cards offered before any selected source body is fetched."""
    text: str = ""
    symbols: dict = field(default_factory=dict)
    required_symbols: tuple = ()


@dataclass(frozen=True)
class DefinitionBodySelection:
    symbols: tuple = ()
    unknown: tuple = ()
def definition_body_menu(hops, selection: Selection) -> DefinitionBodyMenu:
    """Describe selected compiler definitions without source text or file paths."""
    occurrences = set(selection.occurrence_keys)
    chosen = set(selection.symbols)
    selected = [
        hop for hop in hops
        if (_occurrence_key(hop) in occurrences if occurrences
            else hop.citation.qualified_name in chosen)]
    definitions = {}
    definition_extents = {}
    incoming = set()
    outgoing = set()
    selected_names = {
        hop.citation.qualified_name for hop in selected}
    for hop in selected:
        citation = hop.citation
        if citation.line_end > citation.line_start:
            definitions.setdefault(citation.qualified_name, citation)
            definition_extents.setdefault(
                citation.qualified_name, set()).add((
                    citation.file, citation.line_start, citation.line_end))
        parent = citation.parent_qualified_name
        if parent and parent in selected_names:
            incoming.add(citation.qualified_name)
            outgoing.add(parent)
    required_symbols = tuple(
        name for name in definitions
        if (name in incoming.intersection(outgoing)
            or len(definition_extents.get(name, ())) > 1))
    symbols = {}
    lines = [
        "DEFINITION BODY CARDS   choose only bodies whose implementation must be read "
        "to answer every part of the question."]
    for index, (qualified_name, citation) in enumerate(definitions.items(), start=1):
        label = f"B{index}"
        symbols[label] = qualified_name
        owner = _owner_qualified(qualified_name)
        parent = _owner_qualified(citation.parent_qualified_name)
        role = ("route root" if citation.relation == "localized" or not parent
                else f"{citation.relation} from {parent}")
        extent = citation.line_end - citation.line_start + 1
        required = " [route transition]" if qualified_name in required_symbols else ""
        lines.append(
            f"  {label}. {owner}   {role}; {extent}-line definition{required}")
    return DefinitionBodyMenu(
        text="\n".join(lines), symbols=symbols,
        required_symbols=required_symbols)


def resolve_definition_body_selection(
        menu: DefinitionBodyMenu, reply: str) -> DefinitionBodySelection:
    symbols = []
    unknown = []
    for digits in re.findall(r"\bB(\d{1,4})\b", reply or "", re.I):
        label = f"B{digits}"
        symbol = menu.symbols.get(label)
        if symbol is None:
            if label not in unknown:
                unknown.append(label)
        elif symbol not in symbols:
            symbols.append(symbol)
    return DefinitionBodySelection(symbols=tuple(symbols), unknown=tuple(unknown))


def all_definition_body_selection(menu: DefinitionBodyMenu) -> DefinitionBodySelection:
    return DefinitionBodySelection(symbols=tuple(menu.symbols.values()))
def complete_definition_body_selection(
        menu: DefinitionBodyMenu,
        selection: DefinitionBodySelection) -> DefinitionBodySelection:
    """Keep compiler route-transition bodies in addition to model choices."""
    symbols = list(selection.symbols)
    for symbol in menu.required_symbols:
        if symbol not in symbols:
            symbols.append(symbol)
    return DefinitionBodySelection(
        symbols=tuple(symbols), unknown=selection.unknown)
def definition_body_selection_requires_llm(
        menu: DefinitionBodyMenu, selector_mode: str, *, auto_select_max: int = 8) -> bool:
    """Use one compact selector only when the final exact route set is broad."""
    return selector_mode != "deterministic" and len(menu.symbols) > auto_select_max
def resolve_obligation_route_selection(
        menu: RouteMenu, reply: str, *, max_per_obligation: int = 3) -> Selection:
    """Resolve a minimal complete route set from obligation-labelled output.

    Each fixed obligation may retain a few route fragments when the compiler graph is
    disconnected, but a model cannot turn uncertainty into dozens of unrelated routes.
    Plain ``R`` replies remain compatible with deterministic and fallback selectors.
    """
    labelled = re.findall(r"(?im)^\s*C\d{1,2}\s*:\s*([^\n]+)", reply or "")
    if not labelled:
        return resolve_route_selection(menu, reply)
    route_labels = []
    section_labels = []
    for fragment in labelled:
        routes = []
        for prefix, digits in re.findall(r"\b([RS])(\d{1,4})\b", fragment, re.I):
            label = f"{prefix.upper()}{digits}"
            if prefix.upper() == "R":
                if label in menu.routes and label not in routes:
                    routes.append(label)
            elif label in menu.sections and label not in section_labels:
                section_labels.append(label)
        for label in routes[:max(max_per_obligation, 0)]:
            if label not in route_labels:
                route_labels.append(label)
    labels = " ".join((*route_labels, *section_labels))
    return resolve_route_selection(menu, labels)
