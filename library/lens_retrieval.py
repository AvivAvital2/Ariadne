"""Scoped retrieval for the lens (designs/spool-lens-router.md §5).

P13 throughout: admission is CATEGORICAL — a doc enters the spool path
because a crisp entity matched it, or because it cleared an explicit
precision gate over DOC-GRADE candidates. Similarity ranks within what's
admitted; similarity-to-the-repo never gates (it rewards redundancy and
suppresses exactly the novelty the spool exists to add — measured).
"""
import numpy as np
from attrs import frozen

from library.lens_router import symbol_matches

# Measured (north star §7): over DOC-GRADE candidates the seam and
# pure-code cosine distributions stop overlapping (seam min 0.530 vs
# control max 0.498) — 0.52 sits in the gap. Provisional; calibrate with
# real usage like the other gate constants.

# Doc-grade = prose + doc-derived md_sections. Generated element stubs
# match on surface form (dunders, name fragments) and are excluded from the
# SEMANTIC pool — they stay reachable through entity admission and symbol
# lookup.
_DOC_GRADE_SQL = (
    "(content_type != 'catalog' "
    "OR json_extract(metadata, '$.subtype') = 'md_section')"
)


@frozen
class SpoolContribution:
    """One admitted spool doc + HOW it was connected — the labeled assembly
    cites the connection ('entity(<term>)' / 'semantic(<cosine>)') so the
    synthesis can weigh decisiveness honestly."""
    doc_id: str
    connection: str   # 'entity' | 'semantic'
    detail: str       # the matched term, or the cosine as text


def doc_grade_spool_candidates(library, sources) -> list:
    """Doc ids of the spool side's DOC-GRADE semantic candidates.

    Includes the spool's OWN theme summaries: theme docs are null-source,
    but their ``themes.association`` equals the spool source id, so they are
    spool-side content (bidirectional lens) — base ('') themes never match.
    """
    placeholders = ','.join('?' * len(sources))
    with library._conn_provider.acquire() as conn:
        return [
            row[0] for row in conn.execute(
                f'SELECT id FROM documents '
                f'WHERE source_name IN ({placeholders}) '
                f'AND {_DOC_GRADE_SQL} '
                f'UNION '
                f'SELECT doc_id FROM themes '
                f'WHERE association IN ({placeholders})',
                (*sources, *sources),
            )
        ]


def docs_for_entity_hits(library, sources, hits, *, per_hit=3) -> dict:
    """``{doc_id: matched term}`` for every crisp hit — the categorical
    admissions. A symbol hit admits its ELEMENT doc even though stubs are
    not doc-grade: the doc-grade predicate filters only the semantic pool,
    never entity admissions (P13)."""
    placeholders = ','.join('?' * len(sources))
    out: dict = {}
    with library._conn_provider.acquire() as conn:
        symbol_hits = [h for h in hits if h.layer == 'symbol']
        if symbol_hits:
            qn_to_doc = dict(conn.execute(
                f"SELECT json_extract(metadata, '$.qualified_name'), id "
                f'FROM documents WHERE source_name IN ({placeholders}) '
                f"AND content_type = 'catalog' "
                f"AND json_extract(metadata, '$.kind') = 'element' "
                f"AND json_extract(metadata, '$.qualified_name') IS NOT NULL",
                tuple(sources),
            ).fetchall())
            names = list(qn_to_doc)
            for hit in symbol_hits:
                for qn in symbol_matches(hit.term, names, limit=per_hit):
                    out.setdefault(qn_to_doc[qn], hit.term)
        for hit in hits:
            like = f'%{hit.term}%'
            if hit.layer in ('title', 'alias'):
                rows = conn.execute(
                    f'SELECT id FROM documents '
                    f'WHERE source_name IN ({placeholders}) '
                    f"AND content_type != 'catalog' "
                    f'AND title LIKE ? COLLATE NOCASE LIMIT ?',
                    (*sources, like, per_hit),
                ).fetchall()
            elif hit.layer == 'heading':
                rows = conn.execute(
                    f'SELECT DISTINCT d.id FROM sections s '
                    f'JOIN documents d ON d.id = s.document_id '
                    f'WHERE d.source_name IN ({placeholders}) '
                    f'AND s.heading LIKE ? COLLATE NOCASE LIMIT ?',
                    (*sources, like, per_hit),
                ).fetchall()
                # md_section elements carry normalized headings as
                # qualified names — the same vocabulary, element-doc form.
                rows += conn.execute(
                    f'SELECT id FROM documents '
                    f'WHERE source_name IN ({placeholders}) '
                    f"AND content_type = 'catalog' "
                    f"AND json_extract(metadata, '$.subtype') = 'md_section' "
                    f"AND json_extract(metadata, '$.qualified_name') "
                    f'LIKE ? COLLATE NOCASE LIMIT ?',
                    (*sources, f"%{hit.term.replace(' ', '_')}%", per_hit),
                ).fetchall()
            else:
                continue
            for (doc_id,) in rows:
                out.setdefault(doc_id, hit.term)
    return out


JUNK_FLOOR = 0.35


def breadth_speaks(spool_scores_desc, repo_top, *, window: int = 50,
                   floor: float = JUNK_FLOOR) -> bool:
    """The ratified participation criterion for NON-ENTITY spool evidence:
    the spool speaks iff at least a QUARTER of its top window outscores the
    repo's single best doc AND its own best clears the junk floor.

    Measured (two archetypes, cached-vector regression set): expertise shows
    up as corroboration — 18-50 of 50 docs beat the repo's best when the
    environment owns a question; 1-6 when it is vocabulary noise. Every
    score-magnitude criterion (absolute gates, margins, ratios, z-scores)
    failed the same batteries; the COUNT is the quantity that separates.
    Small candidate pools scale the requirement (a quarter of what exists,
    minimum one)."""
    scores = list(spool_scores_desc[:window])
    if not scores or scores[0] < floor:
        return False
    need = max(1, min(window, len(scores)) // 4)
    return sum(1 for s in scores if s > repo_top) >= need


def lens_share(limit: int) -> int:
    """The LENS side's window share on a routed question — derived, not
    chosen: roughly a third of the window (floor 1), so the PRIMARY side
    strictly outweighs the lens at every limit >= 3 (a 2-window can only
    tie) and the share scales with the window instead of freezing at a
    constant. Used by both lens directions and the no-crisp fallback."""
    return max(1, (limit + 1) // 3)


def select_spool_docs(library, matrix, sources, hits, *, query_embedding=None,
                      limit=8, gate=JUNK_FLOOR, fill_allowed=True) -> list:
    """The fuse/expert-only spool path: categorical entity admissions plus
    the query-cosine fill over DOC-GRADE candidates above the gate.

    Entity admission is categorical but CAPPED at half the window and
    cosine-ORDERED when an embedding is available — measured online, shallow
    name-matches (readme/code-of-conduct docs matching a product name by
    title) otherwise saturate the window and starve the fill that carries
    the substantive docs. Fill shortfall backfills from the remaining entity
    admissions, so the window never runs half-empty when evidence exists.
    With no query embedding (offline callers, embedder down) → entity
    admissions only — degraded, never blocked."""
    entity_docs = docs_for_entity_hits(library, sources, hits)
    if query_embedding is None or matrix is None:
        return [
            SpoolContribution(doc_id, 'entity', term)
            for doc_id, term in list(entity_docs.items())[:limit]
        ]

    in_matrix = [d for d in entity_docs if d in matrix.id_to_row]
    ranked_entity = [
        doc_id for doc_id, _ in matrix.rank(
            query_embedding, in_matrix, len(in_matrix))
    ] if in_matrix else []
    ranked_entity += [d for d in entity_docs if d not in matrix.id_to_row]

    entity_cap = max(1, limit // 2)
    picked = [
        SpoolContribution(doc_id, 'entity', entity_docs[doc_id])
        for doc_id in ranked_entity[:entity_cap]
    ]
    candidates = [] if not fill_allowed else [
        doc_id for doc_id in doc_grade_spool_candidates(library, sources)
        if doc_id not in entity_docs and doc_id in matrix.id_to_row
    ]
    if candidates:
        for doc_id, score in matrix.rank(
                query_embedding, candidates, limit - len(picked)):
            if score >= gate:
                picked.append(
                    SpoolContribution(doc_id, 'semantic', f'{score:.2f}'))
    if len(picked) < limit:
        have = {c.doc_id for c in picked}
        for doc_id in ranked_entity:
            if len(picked) >= limit:
                break
            if doc_id not in have:
                picked.append(
                    SpoolContribution(doc_id, 'entity', entity_docs[doc_id]))
    return picked
