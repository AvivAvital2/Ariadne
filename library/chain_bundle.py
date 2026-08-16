"""Curate the bundle a chain hands to synthesis — the index leads, documents follow.

Step three of ``index -> fetch document -> curate bundle -> formulate -> respond``.
``library/structural_assembly.chain_from`` produces the chain; this attaches the prose.

Two properties make this the inversion the north star asks for:

* **Documents are fetched by deterministic id, never searched.** ``_element_doc_id`` is a
  pure function of ``(source, qualified_name)``, so the chain decides which documents are
  read — embeddings do not get a second vote, and the id is inherently source-scoped.
* **Coordinates are unconditional; prose is rationed.** A ``file:line`` is what makes an
  answer checkable, so every hop keeps it. Only the prose is budgeted, and only against
  the LLM's context — a real constraint, unlike the count cap that used to truncate the
  graph walk itself (see ``structural_assembly``'s module docstring).

Prose is spent by reading each citation's ``stop_reason`` — the traversal's own record of
why it stopped there. ``descended``, ``leaf`` and ``depth`` are chain material; ``plumbing``
(fan-in at the descent boundary) and ``revisit`` (a body already walked) are cited for their
coordinates and nothing more. An earlier version inferred this from the trace's shape, which
starved exactly the wrong hops: a destination like ``writeAllChanges`` is a leaf, and the
leaf is usually where the work happens.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
import re


if TYPE_CHECKING:  # pragma: no cover - typing only
    from library import Library
    from library.structural_assembly import StructuralCitation

@dataclass(frozen=True)
class BundleHop:
    """One hop: always coordinates, sometimes prose."""

    citation: 'StructuralCitation'
    document_id: str | None = None
    title: str | None = None
    evidence: str | None = None
    source_excerpts: tuple = ()


@dataclass(frozen=True)
class ChainTheme:
    """A concept the chain passes through, and how much of the chain sits in it.

    Reported at **chain level, not per hop**: measured on the live store the themes a
    chain touches have a median membership of 1,292, so as a per-hop label a theme says
    almost nothing. Aggregated it is a map — the ``runMerge`` chain puts 35 hops in
    *Transaction Log Engine*, 10 in *Delta Lake Error Taxonomy*, 5 in *Metrics,
    Accumulators & Time Abstractions* — for about sixteen lines of context.
    """

    title: str
    cluster_id: str
    hops: int
    member_count: int
    coherent: bool
@dataclass(frozen=True)
class ChainBundle:
    """What synthesis receives, plus an account of what was left out."""

    hops: list[BundleHop] = field(default_factory=list)
    themes: list[ChainTheme] = field(default_factory=list)
    documents_found: int = 0
    source_gaps: tuple[str, ...] = ()


#: Stop reasons whose hop carries its document. ``plumbing`` and ``revisit`` are cited for
#: their coordinates alone — a revisited body was already explained above, and plumbing is
#: named without being opened.
#:
#: ``reference`` earns its document for the same reason it earns a citation: the type a body
#: touches is part of what the body does, and a catalog entry for a type is a signature and a
#: sentence, not a body.
EXPLAINED = frozenset({'descended', 'leaf', 'depth', 'reference'})
def curate_bundle(
    library: 'Library',
    citations: list['StructuralCitation'],
    *,
    source: str,
source_root=None, materialize_source: bool = True,
fetch_documents: bool = True, materialize_definition_bodies: bool = False,
definition_body_symbols=None, definition_body_query = None) -> ChainBundle:
    """Attach each hop's document to it. Coordinates always; a description when the hop
    is chain material and the catalog has one.

    The division of labour this rests on: **a generated description can be wrong, a SCIP
    coordinate cannot.** So the description is what synthesis reads and ``file:line`` is what
    makes the resulting claim checkable. Source text does not travel — that is what the
    coordinates are for, and a reader who wants the body has an exact place to open.

    The document is the per-symbol ``catalog`` entry, fetched by deterministic id from the
    symbol the walk reached. It is not search-retrieved prose and not a docstring: measured
    at production width, 2,362 of 2,645 hops have one, 883 distinct documents totalling
    ~88,600 tokens against ~227,700 for the same chain quoted from source.

    No size budget here: ``render_spine`` is the only thing that bounds the prompt,
    and it cuts from the tail preserving execution order while saying what it
    dropped. A second budget in curation changed nothing about the prompt — the
    spine measured ~20,000 chars at every cap from 6k to 200k — it only chose which
    hops were explained, blindly, before the renderer knew what it would keep.
    """
    from docgen.catalog_writer import _element_doc_id

    if not citations:
        return ChainBundle()
    _sibling_extents = {}
    _body_edge_owners = {}
    from library.source_materialization import materialize_citations
    if not materialize_source:
        source_materialization = None
        source_gaps = ()
    elif source_root is None:
        source_materialization = None
        source_gaps = (f'{source}: source root unavailable',)
    else:
        _sibling_extents = indexed_definition_extents(
            library,
            definition_body_symbols if materialize_definition_bodies else (),
            source=source)
        _body_edge_owners = indexed_definition_edge_sites(library, definition_body_symbols if materialize_definition_bodies else (), source=source)
        source_materialization = materialize_citations(
            citations, {source: source_root}, extra_ranges = (*frontier_edge_ranges(library, citations, source=source), *((source, file, line, "body_edge") for file, line in sorted(_body_edge_owners)), *definition_body_ranges(citations, enabled=materialize_definition_bodies, symbols=definition_body_symbols), *((source, file, line_start, line_end, "definition_body") for file, line_start, line_end in sorted(_sibling_extents))))
        source_gaps = source_materialization.gaps
    source_by_coordinate = {}
    if source_materialization is not None:
        for excerpt in source_materialization.excerpts:
            source_by_coordinate.setdefault(
                (excerpt.source_name, excerpt.file, excerpt.line_start, excerpt.kind), excerpt)

    wanted = {
        citation.qualified_name: _element_doc_id(source, citation.qualified_name)
        for citation in citations
    }
    found: dict[str, tuple[str, str]] = {}
    ids = sorted(set(wanted.values()))
    names = sorted(wanted)
    clusters: dict[str, set[str]] = {}
    theme_rows: dict[str, tuple[str, int, bool]] = {}
    if fetch_documents:
      with library._conn_provider.acquire() as conn:
        for start in range(0, len(ids), 400):
            chunk = ids[start:start + 400]
            placeholders = ','.join('?' * len(chunk))
            for doc_id, title, content in conn.execute(
                    f'SELECT id, title, content FROM documents '
                    f'WHERE id IN ({placeholders})', chunk):
                found[doc_id] = (title, content or '')
            # Which themes these elements belong to. The join to `documents` is inner
            # and needs no fallback: `themes.doc_id` carries a foreign key to
            # `documents`, so a theme cannot outlive its summary — attempting to insert
            # one raises IntegrityError.
            for cluster_id, element_id, theme_title, members, coherent in conn.execute(
                    f'SELECT tm.cluster_id, tm.element_id, d.title, t.member_count, '
                    f'       t.coherent '
                    f'FROM theme_members tm '
                    f'JOIN themes t ON t.cluster_id = tm.cluster_id '
                    f'JOIN documents d ON d.id = t.doc_id '
                    f'WHERE tm.element_id IN ({placeholders})', chunk):
                clusters.setdefault(cluster_id, set()).add(element_id)
                theme_rows[cluster_id] = (theme_title, members, bool(coherent))

    hops: list[BundleHop] = []
    for citation in citations:
        doc_id = wanted[citation.qualified_name]
        record = found.get(doc_id)
        title, content = record if record is not None else (None, '')
        # The document is the evidence the model reads. A hop with none still travels on its
        # coordinates: 10.7% of hops at production width have no catalog entry, and that is
        # a gap to close in doc generation, not one to paper over here with something the
        # model might misread.
        evidence = (content or None
                    if citation.stop_reason in EXPLAINED else None)
        hops.append(BundleHop(
            citation=citation,
            document_id=doc_id if (record is not None or not fetch_documents) else None,
            title=title,
            evidence=evidence,
        source_excerpts = (tuple(
            excerpt for excerpt in source_materialization.excerpts
            if _excerpt_matches_citation(excerpt, citation)
            or _is_sibling_body(excerpt, citation, _sibling_extents) or _is_selected_body_edge(excerpt, citation, _body_edge_owners))
        if source_materialization is not None else ())))

    # Most of the chain first; a narrower cluster breaks ties, being the more specific
    # claim. Breadth and coherence travel so the consumer can weight them — a broad
    # cluster still names where the chain lives, so it is flagged, never filtered out.
    themes = sorted(
        (
            ChainTheme(title=theme_rows[cluster_id][0], cluster_id=cluster_id,
                       hops=sum(1 for citation in citations
                                if wanted[citation.qualified_name] in elements),
                       member_count=theme_rows[cluster_id][1],
                       coherent=theme_rows[cluster_id][2])
            for cluster_id, elements in clusters.items()
        ),
        key=lambda theme: (-theme.hops, theme.member_count, theme.title),
    )

    return ChainBundle(hops=hops, themes=themes, documents_found=len(found), source_gaps = source_gaps)
def _excerpt_matches_citation(excerpt, citation) -> bool:
    """Attach materialized proof only to the compiler occurrence that owns it."""
    finish = citation.line_end or citation.line_start
    if excerpt.kind == "definition":
        return (excerpt.file == citation.file
                and excerpt.line_start == citation.line_start)
    if excerpt.kind == "call_site":
        return (excerpt.file == citation.call_site_file
                and excerpt.line_start == citation.call_site_line)
    if excerpt.kind == "body_edge":
        return (excerpt.file == citation.file
                and citation.line_start <= excerpt.line_start <= finish)
    if excerpt.kind == "definition_body":
        return (excerpt.file == citation.file
                and excerpt.line_start == citation.line_start
                and excerpt.line_end == finish)
    if excerpt.kind == "definition_slice":
        return (excerpt.file == citation.file
                and citation.line_start <= excerpt.line_start
                and excerpt.line_end <= finish)
    if excerpt.kind == "doc_header":
        return (excerpt.file == citation.file
                and excerpt.line_end + 1 == citation.line_start)
    return False
def frontier_edge_ranges(library, citations, *, source: str) -> tuple:
    frontier = [citation for citation in citations if citation.stop_reason == "depth"]
    if not frontier:
        return ()
    by_name = {citation.qualified_name: citation for citation in frontier}
    names = sorted(by_name)
    canonical = {}
    with library._conn_provider.acquire() as conn:
        for start in range(0, len(names), 300):
            chunk = names[start:start + 300]
            placeholders = ",".join("?" * len(chunk))
            for cid, qn, owner in conn.execute(
                    f"SELECT canonical_id, qualified_name, source_name FROM scip_symbols "
                    f"WHERE qualified_name IN ({placeholders})", chunk):
                if owner == source and qn in by_name:
                    canonical[cid] = by_name[qn]
        ranges = set()
        ids = sorted(canonical)
        for start in range(0, len(ids), 300):
            chunk = ids[start:start + 300]
            placeholders = ",".join("?" * len(chunk))
            for caller, callee, file, line in conn.execute(
                    f"SELECT caller_canonical_id, callee_canonical_id, file, line "
                    f"FROM scip_edges WHERE caller_canonical_id IN ({placeholders}) "
                    f"AND edge_type IN ('call', 'type_ref', 'implements')", chunk):
                citation = canonical[caller]
                if (not str(callee).startswith("local ") and file == citation.file
                        and citation.line_start <= line <= (citation.line_end or citation.line_start)):
                    ranges.add((source, file, int(line), "body_edge"))
    return tuple(sorted(ranges))
def indexed_symbols_covered_by_source(library, hops, *, source: str) -> tuple[str, ...]:
    source_ranges = {}
    declaration_ranges = {}
    for hop in hops:
        for excerpt in hop.source_excerpts:
            if excerpt.source_name != source or excerpt.kind == "doc_header":
                continue
            bounds = (int(excerpt.line_start), int(excerpt.line_end))
            source_ranges.setdefault(str(excerpt.file), []).append(bounds)
            if excerpt.kind in ("definition_body", "definition_slice"):
                declaration_ranges.setdefault(str(excerpt.file), []).append(bounds)
    if not source_ranges:
        return ()
    names = set()
    endpoint_ids = set()
    files = sorted(source_ranges)
    with library._conn_provider.acquire() as conn:
        declaration_files = sorted(declaration_ranges)
        for start in range(0, len(declaration_files), 300):
            chunk = declaration_files[start:start + 300]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT file, line_start, qualified_name FROM scip_symbols "
                f"WHERE source_name = ? AND file IN ({placeholders}) "
                "AND canonical_id NOT GLOB \"local *\" AND line_start > 0 "
                "ORDER BY file, line_start, qualified_name",
                (source, *chunk))
            for file, line_start, qualified_name in rows:
                if any(first <= int(line_start) <= last
                       for first, last in declaration_ranges[str(file)]):
                    names.add(str(qualified_name))
        for start in range(0, len(files), 300):
            chunk = files[start:start + 300]
            placeholders = ",".join("?" * len(chunk))
            rows = conn.execute(
                f"SELECT file, line, caller_canonical_id, callee_canonical_id, edge_type "
                f"FROM scip_edges WHERE file IN ({placeholders}) "
                "ORDER BY file, line, caller_canonical_id, callee_canonical_id",
                chunk)
            for file, line, caller, callee, edge_type in rows:
                if edge_type == "contains":
                    continue
                if not any(first <= int(line) <= last
                           for first, last in source_ranges[str(file)]):
                    continue
                if not str(caller).startswith("local "):
                    endpoint_ids.add(str(caller))
                if not str(callee).startswith("local "):
                    endpoint_ids.add(str(callee))
        endpoint_ids = sorted(endpoint_ids)
        for start in range(0, len(endpoint_ids), 300):
            chunk = endpoint_ids[start:start + 300]
            placeholders = ",".join("?" * len(chunk))
            for qualified_name, in conn.execute(
                    f"SELECT DISTINCT qualified_name FROM scip_symbols "
                    f"WHERE source_name = ? AND canonical_id IN ({placeholders}) "
                    "AND canonical_id NOT GLOB \"local *\" ORDER BY qualified_name",
                    (source, *chunk)):
                names.add(str(qualified_name))
    return tuple(sorted(names))
def indexed_definition_edge_sites(library, symbols, *, source: str) -> dict:
    """Compiler edge coordinates owned by each selected qualified body."""
    names = sorted({str(symbol) for symbol in symbols or () if symbol})
    if not names:
        return {}
    callers = {}
    with library._conn_provider.acquire() as conn:
        for start in range(0, len(names), 400):
            chunk = names[start:start + 400]
            placeholders = ",".join("?" * len(chunk))
            for canonical_id, qualified_name in conn.execute(
                    f"SELECT canonical_id, qualified_name FROM scip_symbols "
                    f"WHERE source_name = ? AND qualified_name IN ({placeholders}) "
                    f"AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                    (source, *chunk, "local *")):
                callers[str(canonical_id)] = str(qualified_name)
        sites = {}
        canonical_ids = sorted(callers)
        edge_placeholders = ",".join("?" * 3)
        for start in range(0, len(canonical_ids), 400):
            chunk = canonical_ids[start:start + 400]
            placeholders = ",".join("?" * len(chunk))
            for caller, callee, file, line in conn.execute(
                    f"SELECT caller_canonical_id, callee_canonical_id, file, line "
                    f"FROM scip_edges WHERE caller_canonical_id IN ({placeholders}) "
                    f"AND edge_type IN ({edge_placeholders}) "
                    f"ORDER BY file, line, callee_canonical_id",
                    [*chunk, "call", "type_ref", "implements"]):
                if str(callee).startswith("local "):
                    continue
                key = (str(file), int(line))
                owners = sites.setdefault(key, [])
                owner = callers[str(caller)]
                if owner not in owners:
                    owners.append(owner)
    return {key: tuple(owners) for key, owners in sites.items()}
def definition_body_ranges(
        citations, *, enabled: bool, symbols=None) -> tuple:
    """Full compiler extents for the already-selected definitions, deduplicated."""
    if not enabled:
        return ()
    chosen = None if symbols is None else set(symbols)
    return tuple(dict.fromkeys(
        (citation.source_name, citation.file, citation.line_start,
         citation.line_end, "definition_body")
        for citation in citations
        if citation.line_start > 0
        and citation.line_end > citation.line_start
        and (chosen is None or citation.qualified_name in chosen)))
def indexed_definition_extents(library, symbols, *, source: str) -> dict:
    """Every extent the index records for the selected body symbols.

    Overloads share a qualified name; the walk's citation carries only the
    extent it reached, so the delegating stub would silently stand in for the
    implementation overload unless every recorded extent is materialized.
    Returns ``(file, line_start, line_end) -> (qualified names)`` so the
    excerpts can attach to the hops that own them.
    """
    names = sorted({str(symbol) for symbol in symbols or () if symbol})
    if not names:
        return {}
    extents: dict[tuple, list] = {}
    with library._conn_provider.acquire() as conn:
        for start in range(0, len(names), 400):
            chunk = names[start:start + 400]
            placeholders = ",".join("?" * len(chunk))
            for file, line_start, line_end, name in conn.execute(
                    f"SELECT DISTINCT file, line_start, line_end, qualified_name "
                    f"FROM scip_symbols "
                    f"WHERE source_name = ? AND qualified_name IN ({placeholders}) "
                    f"AND canonical_id NOT GLOB 'local *' "
                    f"AND line_end > line_start "
                    f"ORDER BY file, line_start, line_end, qualified_name",
                    (source, *chunk)):
                key = (str(file), int(line_start), int(line_end))
                owners = extents.setdefault(key, [])
                if name not in owners:
                    owners.append(name)
    return {key: tuple(owners) for key, owners in extents.items()}


def _is_sibling_body(excerpt, citation, sibling_extents) -> bool:
    """A same-name overload's body belongs to that name's occurrence."""
    if excerpt.kind != "definition_body":
        return False
    owners = sibling_extents.get(
        (excerpt.file, excerpt.line_start, excerpt.line_end), ())
    return citation.qualified_name in owners
def _is_selected_body_edge(excerpt, citation, edge_owners) -> bool:
    if excerpt.kind != "body_edge":
        return False
    return citation.qualified_name in edge_owners.get(
        (excerpt.file, excerpt.line_start), ())
def slice_definition_body_excerpts(
        excerpts, query: str, *, evidence_lines=(), causal_lines=(),
        context_lines: int = 1, short_body_lines: int = 48) -> tuple:
    """Keep complete short bodies and causal spans from long definitions.

    This is semantic source isolation rather than a character budget. Short
    compiler extents remain complete. Long definitions keep their signature,
    the continuous span between independently relevant or route-edge anchors,
    and any additional exact compiler edge sites. A body with no usable
    signal remains whole so source proof is never silently discarded.
    """
    from library.source_materialization import SourceExcerpt

    def stem(value: str) -> str:
        token = value.lower()
        if token.endswith("ies") and len(token) > 4:
            return token[:-3] + "y"
        if token.endswith("ing") and len(token) > 5:
            token = token[:-3]
        elif token.endswith("ed") and len(token) > 4:
            token = token[:-2]
        elif token.endswith("s") and len(token) > 3:
            token = token[:-1]
        if len(token) > 3 and token[-1:] == token[-2:-1]:
            token = token[:-1]
        return token

    stop = {
        "a", "an", "and", "are", "do", "does", "how", "in", "is",
        "it", "of", "or", "the", "to", "what", "when", "where", "which",
    }

    def parts(value: str) -> list[str]:
        return [
            part.lower() for part in re.findall(
                r"[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+",
                value or "")
            if part.lower() not in stop and len(part) > 1]

    wanted = {stem(part) for part in parts(query)}
    compiler_lines = {
        (str(file), int(line)) for file, line in evidence_lines if file and line}
    route_lines = {
        (str(file), int(line)) for file, line in causal_lines if file and line}
    output = []
    for excerpt in excerpts:
        if excerpt.kind != "definition_body":
            output.append(excerpt)
            continue
        lines = excerpt.content.splitlines()
        if len(lines) <= max(int(short_body_lines), 0):
            output.append(excerpt)
            continue
        line_tokens = []
        for line in lines:
            values = set(parts(line))
            values.update(
                value[:-1] for value in tuple(values)
                if value.endswith("s") and len(value) > 3)
            line_tokens.append(values)
        frequencies = {}
        for values in line_tokens:
            for token in wanted.intersection(values):
                frequencies[token] = frequencies.get(token, 0) + 1
        rarity_ceiling = max(1, len(lines) // 12)
        discriminating = {
            token for token, frequency in frequencies.items()
            if frequency <= rarity_ceiling}
        semantic = {
            index for index, values in enumerate(line_tokens)
            if discriminating.intersection(values)}
        compiler = {
            line - excerpt.line_start
            for file, line in compiler_lines
            if file == excerpt.file
            and excerpt.line_start <= line <= excerpt.line_end}
        causal = {
            line - excerpt.line_start
            for file, line in route_lines
            if file == excerpt.file
            and excerpt.line_start <= line <= excerpt.line_end}
        if not semantic and not compiler and not causal:
            output.append(excerpt)
            continue
        chosen = {0}
        radius = max(int(context_lines), 0)
        anchors = semantic.union(causal)
        if len(anchors) >= 2:
            start = max(min(anchors) - radius, 0)
            finish = min(max(anchors) + radius, len(lines) - 1)
            chosen.update(range(start, finish + 1))
        else:
            for index in anchors:
                chosen.update(range(
                    max(index - radius, 0),
                    min(index + radius + 1, len(lines))))
        chosen.update(compiler)
        ordered = sorted(chosen)
        ranges = []
        start = finish = ordered[0]
        for index in ordered[1:]:
            if index == finish + 1:
                finish = index
            else:
                ranges.append((start, finish))
                start = finish = index
        ranges.append((start, finish))
        for start, finish in ranges:
            line_start = excerpt.line_start + start
            line_end = excerpt.line_start + finish
            output.append(SourceExcerpt(
                source_name=excerpt.source_name,
                file=excerpt.file,
                line_start=line_start,
                line_end=line_end,
                kind="definition_slice",
                content="\n".join(lines[start:finish + 1]),
                sha256=excerpt.sha256))
    return tuple(output)
