"""Entity index for the lens router (designs/spool-lens-router.md §4).

Three layers per side, all from existing data (no LLM, no embeddings):

- **symbols** — catalog element qualified names, kind-filtered (never
  ``variable`` — the catalog's local-var over-capture must not fake entity
  relevance), matched with the router's structural rules;
- **titles** — non-catalog doc titles (catalog element titles ARE qualified
  names and are owned by the symbol layer);
- **headings** — section headings of the side's documents.

Recipe-declared product aliases join as their own layer. The spool side is
built over the union of the reserved ``spool:<name>`` source and its corpus
source id.
"""
from attrs import frozen

from library.lens_router import (
    EntityHit,
    classify_entity,
    is_distinctive,
    match_symbol,
)

# variables: local-var over-capture must not fake relevance.
# md_section: doc-derived section elements carry normalized HEADINGS as
# qualified names — documentation vocabulary, not API symbols; they join the
# heading layer instead (see build_entity_index).
_ROUTER_EXCLUDED_SUBTYPES = ('variable', 'md_section')


@frozen
class EntityIndex:
    """In-memory resolve surface for one side (repo or spool)."""
    symbol_names: tuple
    titles: tuple      # lowercased
    headings: tuple    # lowercased
    aliases: tuple     # lowercased

    def resolve(self, term: str) -> tuple:
        """Crisp structural resolution of one candidate term — at most one
        hit per layer; non-distinctive terms never resolve (rule 1)."""
        term = term.strip()
        if not term or not is_distinctive(term):
            return ()
        hits = []
        term_l = term.lower()
        if term_l in self.aliases:
            hits.append(EntityHit(term, 'alias', classify_entity(term, 'alias')))
        if self.symbol_names and match_symbol(term, self.symbol_names):
            hits.append(EntityHit(term, 'symbol', classify_entity(term, 'symbol')))
        if any(term_l in title for title in self.titles):
            hits.append(EntityHit(term, 'title', classify_entity(term, 'title')))
        if any(term_l in heading for heading in self.headings):
            hits.append(EntityHit(term, 'heading', classify_entity(term, 'heading')))
        return tuple(hits)


def build_entity_index(library, sources, *, aliases=()) -> EntityIndex:
    """Build one side's index over ``sources`` (a repo's closure sources, or
    the spool's corpus + reserved id union)."""
    symbol_names = []
    for source in sources:
        symbol_names.extend(
            qn for qn, _ in library.list_catalog_element_names(
                source, exclude_subtypes=_ROUTER_EXCLUDED_SUBTYPES)
        )
    placeholders = ','.join('?' * len(sources))
    with library._conn_provider.acquire() as conn:
        section_headings = [
            row[0].rsplit('.', 1)[-1].replace('_', ' ').lower()
            for row in conn.execute(
                f"SELECT json_extract(metadata, '$.qualified_name') "
                f'FROM documents '
                f"WHERE content_type = 'catalog' "
                f"AND json_extract(metadata, '$.kind') = 'element' "
                f"AND json_extract(metadata, '$.subtype') = 'md_section' "
                f'AND json_extract(metadata, '
                f"'$.source_name') IN ({placeholders})",
                tuple(sources),
            )
            if row[0]
        ]
        titles = [
            row[0].lower()
            for row in conn.execute(
                f'SELECT title FROM documents '
                f'WHERE source_name IN ({placeholders}) '
                f"AND content_type != 'catalog'",
                tuple(sources),
            )
        ]
        headings = [
            row[0].lower()
            for row in conn.execute(
                f'SELECT s.heading FROM sections s '
                f'JOIN documents d ON d.id = s.document_id '
                f'WHERE d.source_name IN ({placeholders}) '
                f'AND s.heading IS NOT NULL',
                tuple(sources),
            )
        ]
    return EntityIndex(
        symbol_names=tuple(symbol_names),
        titles=tuple(titles),
        headings=tuple(headings) + tuple(section_headings),
        aliases=tuple(a.lower() for a in aliases),
    )
