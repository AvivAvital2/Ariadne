"""Surface tags — the surface tier's data (designs/spool-lens-router.md §4).

The recipe declares the target kind's interaction surfaces (serialization,
parallelism, IO, …) as keyword VOCABULARIES; a deterministic tag pass writes
``(doc, surface)`` rows for the spool's DOC-GRADE docs (the semantic
retrieval candidates — generated element stubs stay untagged), the pack
ships the table, and the question side matches the same vocabularies. This
is the surface tier's bridge when no entity is crisp: categorical (P13),
stems only (word-start bounded), no embeddings, no LLM. The
embedding-vs-descriptor tagger remains the documented upgrade path.
"""
import re

from attrs import frozen

_DOC_GRADE_SQL = (
    "(content_type != 'catalog' "
    "OR json_extract(metadata, '$.subtype') = 'md_section')"
)


def _stem_pattern(surfaces) -> dict:
    """Per-surface compiled pattern: a word STARTING with any of the
    surface's stems matches ('serializ' → serializer/serialization; never
    mid-word)."""
    return {
        name: re.compile(
            r'(?<![a-z0-9])(?:'
            + '|'.join(re.escape(stem.lower()) for stem in stems)
            + r')',
        )
        for name, stems in surfaces.items() if stems
    }


def match_surfaces(text_parts, surfaces) -> set:
    """The surfaces whose vocabulary matches any of ``text_parts``."""
    patterns = _stem_pattern(surfaces)
    joined = ' '.join(part.lower() for part in text_parts if part)
    return {name for name, pattern in patterns.items()
            if pattern.search(joined)}


def surfaces_for_question(question: str, surfaces) -> list:
    """The question's surfaces, recipe order — the router's surface signal."""
    matched = match_surfaces([question], surfaces)
    return [name for name in surfaces if name in matched]


_SCHEMA = '''
CREATE TABLE IF NOT EXISTS surface_tags (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    surface TEXT NOT NULL,
    UNIQUE(source_name, doc_id, surface)
);
CREATE INDEX IF NOT EXISTS idx_surface_tags_surface
    ON surface_tags(source_name, surface);
'''


@frozen
class SurfaceTagRow:
    source_name: str
    doc_id: str
    surface: str


def init_surface_tags_schema(conn) -> None:
    conn.executescript(_SCHEMA)


def upsert_surface_tags(conn, rows) -> None:
    """Idempotent — the UNIQUE key dedupes re-tagging."""
    conn.executemany(
        'INSERT OR IGNORE INTO surface_tags (source_name, doc_id, surface) '
        'VALUES (?, ?, ?)',
        [(r.source_name, r.doc_id, r.surface) for r in rows],
    )


def docs_for_surfaces(conn, source_names, surfaces) -> set:
    """Doc ids tagged with ANY of ``surfaces`` — the surface-scoped
    candidate set for retrieval."""
    if not surfaces:
        return set()
    src_ph = ','.join('?' * len(source_names))
    surf_ph = ','.join('?' * len(surfaces))
    return {
        row[0] for row in conn.execute(
            f'SELECT doc_id FROM surface_tags '
            f'WHERE source_name IN ({src_ph}) AND surface IN ({surf_ph})',
            (*source_names, *surfaces),
        )
    }


def tag_surfaces_for_source(library, *, outer_sources, corpus_source,
                            surfaces) -> int:
    """Tag the spool's DOC-GRADE docs by matching each surface's vocabulary
    against title + qualified_name + the doc's section headings. Keyed by
    the CORPUS source name (like version_facts). Returns rows written.
    Idempotent."""
    placeholders = ','.join('?' * len(outer_sources))
    rows_out = []
    with library._conn_provider.acquire() as conn:
        init_surface_tags_schema(conn)
        docs = conn.execute(
            f'SELECT d.id, d.title, '
            f"json_extract(d.metadata, '$.qualified_name'), "
            f"(SELECT group_concat(s.heading, ' ') FROM sections s "
            f'WHERE s.document_id = d.id) '
            f'FROM documents d '
            f'WHERE d.source_name IN ({placeholders}) '
            f'AND {_DOC_GRADE_SQL}',
            tuple(outer_sources),
        ).fetchall()
        for doc_id, title, qualified_name, headings in docs:
            parts = [title or '',
                     (qualified_name or '').replace('_', ' ').replace('.', ' '),
                     headings or '']
            for surface in match_surfaces(parts, surfaces):
                rows_out.append(SurfaceTagRow(
                    source_name=corpus_source, doc_id=doc_id,
                    surface=surface))
        upsert_surface_tags(conn, rows_out)
    return len(rows_out)


def surfaces_from_resolution(resolution) -> dict:
    """The merged surface vocabularies across the registered spools'
    manifests — the consumer's question-side half of the surface tier.
    Tolerates registrations, bare manifests, and dicts."""
    merged: dict = {}
    for name in getattr(resolution, 'registered', {}) or {}:
        holder = resolution.registered[name]
        manifest = getattr(holder, 'manifest', holder)
        surfaces = getattr(manifest, 'surfaces', None)
        if surfaces is None and isinstance(manifest, dict):
            surfaces = manifest.get('surfaces')
        merged.update(surfaces or {})
    return merged
