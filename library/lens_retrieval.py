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
SPOOL_FALLBACK_GATE = 0.52

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
    """Doc ids of the spool side's DOC-GRADE semantic candidates."""
    placeholders = ','.join('?' * len(sources))
    with library._conn_provider.acquire() as conn:
        return [
            row[0] for row in conn.execute(
                f'SELECT id FROM documents '
                f'WHERE source_name IN ({placeholders}) '
                f'AND {_DOC_GRADE_SQL}',
                tuple(sources),
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


def select_spool_docs(library, matrix, sources, hits, *, query_embedding=None,
                      limit=8, gate=SPOOL_FALLBACK_GATE) -> list:
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
    candidates = [
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


def fallback_spool_docs(library, matrix, sources, seed_doc_ids, *,
                        gate=SPOOL_FALLBACK_GATE, limit=3,
                        restrict_to=None) -> list:
    """The semantic fallback tier (repo-only + fallback / honest-gap
    probing): per-doc two-hop from repo seed docs into DOC-GRADE candidates,
    admitted iff the best seed cosine clears the gate. Contributions are
    'considerations'-grade — the assembly labels them semantic, never
    decisive."""
    if matrix is None:
        return []
    seeds = [s for s in seed_doc_ids if s in matrix.id_to_row]
    if not seeds:
        return []
    candidates = [
        doc_id for doc_id in doc_grade_spool_candidates(library, sources)
        if doc_id in matrix.id_to_row
        # The surface tier's consumption: when the question resolved to
        # surfaces, the fallback pool is RESTRICTED to surface-tagged docs
        # — categorical scoping (P13); similarity still ranks inside it.
        and (restrict_to is None or doc_id in restrict_to)
    ]
    if not candidates:
        return []
    best: dict = {}
    for seed in seeds:
        seed_vec = np.asarray(
            matrix.M[matrix.id_to_row[seed]], dtype=np.float32)
        for doc_id, score in matrix.rank(seed_vec, candidates, limit * 3):
            if score >= gate and score > best.get(doc_id, 0.0):
                best[doc_id] = score
    ranked = sorted(best.items(), key=lambda item: -item[1])[:limit]
    return [
        SpoolContribution(doc_id, 'semantic', f'{score:.2f}')
        for doc_id, score in ranked
    ]
