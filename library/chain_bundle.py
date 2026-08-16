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
) -> ChainBundle:
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

    wanted = {
        citation.qualified_name: _element_doc_id(source, citation.qualified_name)
        for citation in citations
    }
    found: dict[str, tuple[str, str]] = {}
    ids = sorted(set(wanted.values()))
    names = sorted(wanted)
    clusters: dict[str, set[str]] = {}
    theme_rows: dict[str, tuple[str, int, bool]] = {}
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
            document_id=doc_id if record is not None else None,
            title=title,
            evidence=evidence,
        ))

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

    return ChainBundle(hops=hops, themes=themes, documents_found=len(found))
