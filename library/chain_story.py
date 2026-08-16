"""Compact, compiler-grounded evidence IR for constrained narration."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from library.source_chunks import (derive_source_chunks, render_source_ledger, source_chunk_values)
from library.relation_semantics import transition_verb
from library.relation_semantics import relation_site_phrase
_COMPACT_CLOSURE_REASONS = frozenset((
    "selected_route_fanout",
    "selected_caller",
    "selected_reference",
    "reference_bridge",
    "selected_reference_caller",
    "selected_owner",
    "selected_owner_member",
))
@dataclass(frozen=True)
class StoryNode:
    id: str
    symbol: str
    file: str
    line: int
    call_file: str
    call_line: int
    relation: str
    description: str = ""
    excerpts: tuple = field(default_factory=tuple)
    stop_reason: str = ""


@dataclass(frozen=True)
class StoryEdge:
    id: str
    source: str
    target: str
    relation: str
    file: str
    line: int


@dataclass(frozen=True)
class StoryIR:
    nodes: tuple[StoryNode, ...] = ()
    edges: tuple[StoryEdge, ...] = ()
    sections: tuple = ()
    chunks: tuple = ()


def _occurrence(citation):
    return (citation.qualified_name, citation.file, citation.line_start,
            citation.line_end, citation.parent_qualified_name,
            citation.call_site_file, citation.call_site_line, citation.relation,
            citation.hop, citation.stop_reason)
def _compact_description(content: str) -> str:
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("description:"):
            return line.split(":", 1)[1].strip()
    return lines[0] if lines else ""
def build_story_ir(hops, selection, fetched) -> StoryIR:
    """Project selected SCIP occurrences into nodes and only recorded edges."""
    occurrences = set(selection.occurrence_keys)
    symbols = set(selection.symbols)
    chosen = [hop for hop in hops
              if (_occurrence(hop.citation) in occurrences if occurrences
                  else hop.citation.qualified_name in symbols)]
    selected_source_ranges = tuple(
        (excerpt.source_name, excerpt.file, excerpt.line_start, excerpt.line_end)
        for hop in chosen for excerpt in hop.source_excerpts
        if excerpt.kind in ("definition_body", "definition_slice"))

    def already_in_selected_source(excerpt) -> bool:
        if excerpt.kind not in ("definition", "call_site", "body_edge"):
            return False
        return any(
            excerpt.source_name == source_name and excerpt.file == file
            and line_start <= excerpt.line_start
            and excerpt.line_end <= line_end
            for source_name, file, line_start, line_end in selected_source_ranges)

    nodes_list = []
    rendered_bodies = set()
    for index, hop in enumerate(chosen, start=1):
        excerpts = []
        for excerpt in hop.source_excerpts:
            if already_in_selected_source(excerpt):
                continue
            if excerpt.kind in ("definition_body", "definition_slice"):
                key = (excerpt.source_name, excerpt.file, excerpt.line_start,
                       excerpt.line_end, excerpt.sha256)
                if key in rendered_bodies:
                    continue
                rendered_bodies.add(key)
            excerpts.append(excerpt)
        nodes_list.append(StoryNode(
            id=f"N{index}", symbol=hop.citation.qualified_name,
            file=hop.citation.file, line=hop.citation.line_start,
            call_file=hop.citation.call_site_file,
            call_line=hop.citation.call_site_line,
            relation=hop.citation.relation,
            description=_compact_description(
                fetched.definitions.get(hop.citation.qualified_name, "")),
            excerpts=tuple(excerpts),
            stop_reason=hop.citation.stop_reason))
    nodes = tuple(nodes_list)
    by_symbol = {}
    for node in nodes:
        by_symbol.setdefault(node.symbol, node)
    edges = []
    for hop, target in zip(chosen, nodes):
        source = by_symbol.get(hop.citation.parent_qualified_name)
        if source is None:
            continue
        edges.append(StoryEdge(
            id=f"E{len(edges) + 1}", source=source.id, target=target.id,
            relation=hop.citation.relation, file=hop.citation.call_site_file,
            line=hop.citation.call_site_line))
    bodies = tuple(
        excerpt for hop in chosen for excerpt in hop.source_excerpts
        if excerpt.kind in ("definition_body", "definition_slice"))
    seeds = [
        (hop.citation.call_site_file, hop.citation.call_site_line)
        for hop in chosen
        if hop.citation.call_site_file and hop.citation.call_site_line]
    for hop in chosen:
        for excerpt in hop.source_excerpts:
            if excerpt.kind == "body_edge":
                seeds.extend(
                    (excerpt.file, line)
                    for line in range(excerpt.line_start, excerpt.line_end + 1))
    sites = tuple(
        excerpt for node in nodes for excerpt in node.excerpts
        if excerpt.kind in ("call_site", "body_edge", "doc_header"))
    return StoryIR(nodes=nodes, edges=tuple(edges),
                   sections=tuple(fetched.sections),
                   chunks=derive_source_chunks(bodies, seeds, sites=sites))
def render_story_evidence(story: StoryIR, *, compact_source: bool = False) -> str:
    """Render each value once; narration refers to stable placeholders.

    ``compact_source`` replaces raw definition-body dumps with the exact-source
    chunk ledger: the model reads the causal statements and references
    ``{{X#}}``; deterministic expansion re-attaches the code afterwards.
    """
    compact = bool(compact_source and story.chunks)
    lines = [
        "EVIDENCE IR — write a concise story using only these placeholders.",
        ("Use {{N#}} for nodes, {{E#}} for transitions, and {{X#}} on its own "
         "line wherever the exact source belongs. Do not invent transitions; "
         "never retype or paraphrase code."
         if compact else
         "Use {{N#}} for nodes and {{E#}} for transitions. Do not invent transitions."),
        "Nodes:",
    ]
    for node in story.nodes:
        site = relation_site_phrase(node.relation).removesuffix(" at")
        lines.append(
            f"  {{{{{node.id}}}}}: {node.symbol} [{node.file}:{node.line}]; "
            f"{site} at {node.call_file}:{node.call_line}")
        # A retained body dependency is proven by its ledger chunk and its
        # edge; one coordinate line is all the narration needs for it.
        if compact and node.stop_reason in _COMPACT_CLOSURE_REASONS:
            continue
        if node.description:
            lines.append(f"    description: {node.description}")
        for excerpt in node.excerpts:
            if compact and excerpt.kind in (
                    "definition_body", "definition_slice",
                    "call_site", "body_edge", "doc_header"):
                continue
            lines.append(
                f"    source {excerpt.kind} [{excerpt.file}:{excerpt.line_start}-"
                f"{excerpt.line_end}]: {excerpt.content}")
    lines.append("Transitions:")
    for edge in story.edges:
        lines.append(
            f"  {{{{{edge.id}}}}}: {{{{{edge.source}}}}} {edge.relation} "
            f"{{{{{edge.target}}}}} at {edge.file}:{edge.line}")
    if compact:
        lines.append(render_source_ledger(story.chunks))
    if story.sections:
        lines.append("Relevant background sections (explanation only, never proof):")
        for title, heading, content in story.sections:
            lines.append(f"  {title} -> {heading}: {content.strip()}")
    return "\n".join(lines)
def render_formulation_spine(story: StoryIR) -> str:
    """The formulation prompt's evidence spine.

    Once exact-source chunks exist the ledger replaces raw body dumps — the
    model references ``{{X#}}`` and deterministic expansion re-attaches the
    code. A story with no chunks keeps the classic rendering unchanged.
    """
    return render_story_evidence(story, compact_source=bool(story.chunks))


_PLACEHOLDER = re.compile(r"\{\{([NEX]\d+)\}\}|(?<![\w{])([NEX]\d+)(?![\w}])")
def expand_story_placeholders(answer: str, story: StoryIR, *,
                              strict: bool = True) -> str:
    """Expand known IDs; optionally mark unknown IDs for claim filtering."""
    values = {
        node.id: f"{node.symbol} ({node.file}:{node.line})"
        for node in story.nodes}
    for edge in story.edges:
        verb = transition_verb(edge.relation)
        values[edge.id] = f"{verb} at {edge.file}:{edge.line}"
    values.update(source_chunk_values(story.chunks))
    matches = list(_PLACEHOLDER.finditer(answer or ""))
    ids = {match.group(1) or match.group(2) for match in matches}
    unknown = sorted(ids - set(values))
    if unknown and strict:
        raise ValueError("unknown evidence placeholder(s): " + ", ".join(unknown))
    return _PLACEHOLDER.sub(
        lambda match: values.get(
            match.group(1) or match.group(2),
            f"[unsupported evidence {match.group(1) or match.group(2)}]"),
        answer or "")
def render_unreferenced_story_evidence(draft: str, story: StoryIR) -> str:
    """Append compiler transitions and exact source omitted by narration.

    Referenced placeholders remain the model's prose.  Any selected transition
    or source chunk it omits is attached deterministically from StoryIR, once,
    so repair cannot erase compiler proof and no code is regenerated by the
    model.
    """
    referenced = {
        match.group(1) or match.group(2)
        for match in _PLACEHOLDER.finditer(draft or "")
    }
    nodes = {node.id: node for node in story.nodes}
    lines = []
    for edge in story.edges:
        if edge.id in referenced:
            continue
        source = nodes.get(edge.source)
        target = nodes.get(edge.target)
        if source is None or target is None:
            continue
        verb = transition_verb(edge.relation)
        statement = (f"{source.symbol} {verb} {target.symbol} "
                     f"at {edge.file}:{edge.line}")
        if statement not in (draft or ""):
            lines.append(statement)
    chunk_values = source_chunk_values(story.chunks)
    lines.extend(
        chunk_values[chunk.id] for chunk in story.chunks
        if chunk.id not in referenced
    )
    if not lines:
        return ""
    return "\n\nEvidence:\n" + "\n".join(lines)
