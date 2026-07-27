"""Version facts — the declarative version layer (north star §12, Slice 2).

The spool's version-pinned value claim is unfalsifiable while the pin lives
only in the manifest: the corpus text carries thousands of structured
version markers, inert. This module extracts them into facts
(``qualified_name → since/deprecated + version``), stores them per source,
and joins them against the runtime's component-version map so "is X
available on MY runtime" resolves deterministically.

Extraction is DETERMINISTIC MARKERS ONLY — annotations and doc directives,
never prose ("this is deprecated in spirit" must not extract): these facts
arm the corrector, so a wrong fact asserted authoritatively is the harm
model. Precision beats recall; recall is measured, not assumed.
"""
import re
from pathlib import Path

from attrs import frozen

# Structured markers. Each pattern's first group is the version (absent for
# the bare Java annotation).
_MARKERS = (
    ('since', re.compile(r'@Since\(\s*"([^"]+)"\s*\)')),
    ('since', re.compile(r'\.\.\s*versionadded::\s*([\w.\-]+)')),
    ('deprecated', re.compile(
        r'@deprecated\(\s*"[^"]*"\s*,\s*"([^"]+)"\s*\)')),
    ('deprecated', re.compile(r'\.\.\s*deprecated::\s*([\w.\-]+)')),
    ('deprecated', re.compile(r'@Deprecated\b()')),
)

_EVIDENCE_CHARS = 80


@frozen
class VersionFact:
    """One extracted marker: what it asserts, for which version, and the
    matched text (the corrector cites evidence, never bare claims)."""
    fact: str            # 'since' | 'deprecated' | 'removed'
    version: str | None  # component version string; None for bare markers
    evidence: str


@frozen
class VersionFactRow(VersionFact):
    """A stored fact — attached to a symbol in a source, with provenance."""
    source_name: str = ''
    qualified_name: str = ''
    doc_id: str = ''
    # WHICH corpus repo the fact came from (first path segment under
    # the corpus root): one spool source aggregates several
    # differently-versioned repos, so availability joins per
    # component, never one-version-per-source.
    component: str = ''


def extract_version_facts(content: str) -> list:
    """All structured version markers in one doc's content."""
    facts = []
    seen = set()
    for fact, pattern in _MARKERS:
        for match in pattern.finditer(content or ''):
            version = match.group(1) or None
            key = (fact, version)
            if key in seen:
                continue
            seen.add(key)
            facts.append(VersionFact(
                fact=fact, version=version,
                evidence=match.group(0)[:_EVIDENCE_CHARS] or fact,
            ))
    return facts


_SCHEMA = '''
CREATE TABLE IF NOT EXISTS version_facts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_name TEXT NOT NULL,
    qualified_name TEXT NOT NULL,
    fact TEXT NOT NULL,
    version TEXT,
    evidence TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    component TEXT NOT NULL DEFAULT '',
    UNIQUE(source_name, qualified_name, fact, version)
);
CREATE INDEX IF NOT EXISTS idx_version_facts_qn
    ON version_facts(source_name, qualified_name);
'''


def init_version_facts_schema(conn) -> None:
    conn.executescript(_SCHEMA)
    # Defensive migration: the component column postdates the first shipped
    # schema; the table is derived data, older rows just carry ''.
    cols = {row[1] for row in conn.execute('PRAGMA table_info(version_facts)')}
    if 'component' not in cols:
        conn.execute(
            "ALTER TABLE version_facts ADD COLUMN component TEXT NOT NULL "
            "DEFAULT ''")
    # Bare @Deprecated facts carry version=NULL, and the table's UNIQUE
    # constraint treats NULLs as DISTINCT — every re-extraction/install pass
    # multiplied them. Self-heal legacy duplicates (keep the oldest row),
    # then enforce NULL-safe idempotency with a COALESCE unique index
    # (INSERT OR IGNORE respects unique indexes too).
    conn.execute(
        'DELETE FROM version_facts WHERE id NOT IN ('
        ' SELECT MIN(id) FROM version_facts'
        " GROUP BY source_name, qualified_name, fact, COALESCE(version, ''))")
    conn.execute(
        'CREATE UNIQUE INDEX IF NOT EXISTS idx_version_facts_dedupe '
        'ON version_facts('
        "source_name, qualified_name, fact, COALESCE(version, ''))")


def upsert_version_facts(conn, rows) -> None:
    """Idempotent insert — the UNIQUE key dedupes re-extraction."""
    conn.executemany(
        'INSERT OR IGNORE INTO version_facts '
        '(source_name, qualified_name, fact, version, evidence, doc_id, '
        'component) VALUES (?, ?, ?, ?, ?, ?, ?)',
        [(r.source_name, r.qualified_name, r.fact, r.version, r.evidence,
          r.doc_id, r.component) for r in rows],
    )


def query_version_facts(conn, source_names, qualified_name) -> list:
    placeholders = ','.join('?' * len(source_names))
    rows = conn.execute(
        f'SELECT source_name, qualified_name, fact, version, evidence, '
        f'doc_id, component FROM version_facts '
        f'WHERE source_name IN ({placeholders}) AND qualified_name = ?',
        (*source_names, qualified_name),
    ).fetchall()
    return [
        VersionFactRow(
            fact=fact, version=version, evidence=evidence,
            source_name=source_name, qualified_name=qualified_name_,
            doc_id=doc_id, component=component,
        )
        for source_name, qualified_name_, fact, version, evidence, doc_id,
        component in rows
    ]


# SQL prefilter: only docs whose content can possibly carry a marker get the
# regex pass (the corpus holds ~200k docs; markers live in a few thousand).
_PREFILTER_SQL = (
    "(content LIKE '%@Since%' OR content LIKE '%versionadded%' "
    "OR content LIKE '%@deprecated%' OR content LIKE '%@Deprecated%' "
    "OR content LIKE '%.. deprecated::%')"
)


# Lines read around an element's location when its marker payload must come
# from the source file (the stored signature is truncated at the annotation).
_SOURCE_WINDOW_BEFORE = 2
_SOURCE_WINDOW_AFTER = 2


def _source_window(path_text: str, line_start, line_end, cache: dict):
    """The located lines ± the window from a corpus source file, or None
    when the file is unreadable (consumer machines don't have the corpus —
    facts are extracted where it exists and TRAVEL as facts)."""
    lines = cache.get(path_text)
    if lines is None:
        try:
            lines = Path(path_text).read_text(
                encoding='utf-8', errors='replace').splitlines()
        except OSError:
            lines = False
        cache[path_text] = lines
    if lines is False or not line_start:
        return None
    start = max(0, int(line_start) - 1 - _SOURCE_WINDOW_BEFORE)
    end = min(len(lines), int(line_end or line_start) + _SOURCE_WINDOW_AFTER)
    return '\n'.join(lines[start:end])


def extract_facts_for_source(library, *, outer_sources, corpus_source,
                             source_root=None) -> int:
    """Extract facts for every catalog ELEMENT doc of one spool corpus and
    persist them under the CORPUS source name (the entity indexes key
    symbols the same way). ``outer_sources`` are the document rows' outer
    ids (``spool:<name>`` for installed docs, the corpus id at build time).

    The store's bare markers are only the PREFILTER: element signatures are
    truncated at the annotation (measured — zero stored '@Since(' payloads),
    so the version is read from the located source-file window. Runs at
    build/backfill time where the corpus checkout exists; a missing file is
    an honest skip, and the resulting facts ship in the pack so consumers
    never need the files. Returns the number of facts written. Idempotent.
    """
    import json as _json

    root = Path(source_root).resolve() if source_root else None

    def _component(path_text: str) -> str:
        if root is None:
            return ''
        try:
            return Path(path_text).resolve().relative_to(root).parts[0]
        except (ValueError, IndexError):
            return ''

    placeholders = ','.join('?' * len(outer_sources))
    rows_out = []
    file_cache: dict = {}
    with library._conn_provider.acquire() as conn:
        init_version_facts_schema(conn)
        rows = conn.execute(
            f"SELECT id, content, source_files, "
            f"json_extract(metadata, '$.qualified_name'), "
            f"json_extract(metadata, '$.location.line_start'), "
            f"json_extract(metadata, '$.location.line_end') "
            f'FROM documents '
            f'WHERE source_name IN ({placeholders}) '
            f"AND content_type = 'catalog' "
            f"AND json_extract(metadata, '$.kind') = 'element' "
            f"AND json_extract(metadata, '$.source_name') = ? "
            f"AND json_extract(metadata, '$.qualified_name') IS NOT NULL "
            f'AND {_PREFILTER_SQL}',
            (*outer_sources, corpus_source),
        ).fetchall()
        for doc_id, content, source_files, qualified_name, line_start, \
                line_end in rows:
            files = _json.loads(source_files) if source_files else []
            text = _source_window(
                files[0], line_start, line_end, file_cache) if files else None
            facts = extract_version_facts(text) if text else []
            if not facts:
                # No readable source window — the stored content can still
                # carry a complete marker (rare); bare markers extract no
                # version and versionless 'since' is meaningless, keep only
                # versioned facts + bare deprecations.
                facts = [
                    f for f in extract_version_facts(content)
                    if f.version or f.fact == 'deprecated'
                ]
            component = _component(files[0]) if files else ''
            for fact in facts:
                rows_out.append(VersionFactRow(
                    fact=fact.fact, version=fact.version,
                    evidence=fact.evidence, source_name=corpus_source,
                    qualified_name=qualified_name, doc_id=doc_id,
                    component=component,
                ))
        upsert_version_facts(conn, rows_out)
    return len(rows_out)


def version_at_most(a: str, b: str) -> bool:
    """``a <= b`` on numeric dotted segments; non-numeric suffixes
    (``-preview``, ``-rc1``) are ignored for the comparison — the numeric
    prefix decides. Missing segments count as zero (``3.5 == 3.5.0``)."""
    def numeric(version):
        parts = []
        for segment in version.split('.'):
            digits = re.match(r'\d+', segment)
            if not digits:
                break
            parts.append(int(digits.group(0)))
        return parts

    left, right = numeric(a), numeric(b)
    width = max(len(left), len(right))
    left += [0] * (width - len(left))
    right += [0] * (width - len(right))
    return left <= right


def runtime_availability(conn, source_names, qualified_name,
                         runtime_components) -> dict:
    """Join a symbol's facts against the runtime's component map:
    ``{'available': bool | None, 'since': ..., 'deprecated': ...}``.

    ``available`` is True iff a ``since`` fact exists at or below the
    runtime's component version, False iff every ``since`` fact is above
    it, and None when there are no facts or no component version to judge
    against — the honest gap, never a guess."""
    facts = query_version_facts(conn, source_names, qualified_name)
    result: dict = {'available': None, 'since': None, 'deprecated': None}

    def _runtime_version(fact):
        # Per-fact join: the fact's COMPONENT names its repo's versioning;
        # a source-level key is only the single-repo fallback.
        if fact.component and runtime_components.get(fact.component):
            return runtime_components[fact.component]
        for source in source_names:
            if runtime_components.get(source):
                return runtime_components[source]
        return None

    for fact in facts:
        if fact.fact == 'since':
            result['since'] = fact.version
            runtime_version = _runtime_version(fact)
            if runtime_version and fact.version:
                result['available'] = version_at_most(
                    fact.version, runtime_version)
        elif fact.fact == 'deprecated':
            result['deprecated'] = fact.version or 'unversioned'
    return result


def runtime_components_from_resolution(resolution) -> dict:
    """The merged runtime→component-version map across the registered
    spools' manifests — the availability join's right-hand side, sourced
    from the PIN, never caller-supplied dicts. Tolerates every shape the
    resolution holds: a registration with ``.manifest``, a bare manifest
    object, or a plain dict."""
    merged: dict = {}
    for name in getattr(resolution, 'registered', {}) or {}:
        holder = resolution.registered[name]
        manifest = getattr(holder, 'manifest', holder)
        components = getattr(manifest, 'runtime_components', None)
        if components is None and isinstance(manifest, dict):
            components = manifest.get('runtime_components')
        merged.update(components or {})
    return merged


def facts_for_terms(conn, source_names, terms, *, limit: int = 12) -> list:
    """Facts whose qualified-name LAST SEGMENT matches one of ``terms`` —
    the ask path's deterministic lookup (the A/B eval showed the synthesis
    saying "cannot determine" on facts this table held). Corpus-scoped,
    capped."""
    terms = [t for t in terms if t]
    if not terms or not source_names:
        return []
    src_ph = ','.join('?' * len(source_names))
    term_ph = ','.join('?' * len(terms))
    rows = conn.execute(
        f"SELECT source_name, qualified_name, fact, version, evidence, "
        f"doc_id, component FROM version_facts "
        f"WHERE source_name IN ({src_ph}) "
        f"AND (qualified_name IN ({term_ph}) OR "
        + ' OR '.join(["qualified_name LIKE '%.' || ?"] * len(terms))
        + f") LIMIT {int(limit)}",
        (*source_names, *terms, *terms),
    ).fetchall()
    out = []
    term_set = set(terms)
    for row in rows:
        last = row[1].rsplit('.', 1)[-1]
        if last in term_set or row[1] in term_set:
            out.append(VersionFactRow(
                source_name=row[0], qualified_name=row[1], fact=row[2],
                version=row[3], evidence=row[4], doc_id=row[5],
                component=row[6]))
    return out
