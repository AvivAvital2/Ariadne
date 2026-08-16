"""Catalog element lookup + fuzzy suggestions.                                                                                                                   
                                                                                                                                                                                     
Fast path: direct doc-id lookup via _element_doc_id.
Miss path: fuzzy match against all catalog element qualified_names.                                                                                                                  
"""                                                                                                                                                                                  
from __future__ import annotations

from schema import CATALOG_KIND_ELEMENT

import difflib

from library.word_tokens import segment_word_tokens as _segment_word_tokens
from typing import TYPE_CHECKING, Any

from docgen.catalog_writer import _element_doc_id

if TYPE_CHECKING:
    from library import Library
                


# Catalog subtypes that denote a config KEY definition (key=value), across every
# config format the catalog knows -- HOCON, YAML, JSON, and Dockerfile ENV/ARG
# (dockerfile_stage / dockerfile_expose are not config keys).
_CONFIG_KEY_SUBTYPES = frozenset({
    'hocon_key', 'yaml_key', 'json_key', 'dockerfile_env', 'dockerfile_arg',
})


def lookup_symbol(
    library: "Library",
    source_name: str,                                                                                                                                                                
    file: str | None,
    qualified_name: str,                                                                                                                                                             
) -> dict[str, Any]:
    """Look up a single catalog element by (source, qualified_name).                                                                                                                 
                                                                                                                                                                                     
    Returns a flat dict of element fields when found:
        {found: True, language, subtype, signature, location, description,                                                                                                           
         parent_qualified_name, file, qualified_name}                                                                                                                                
                                                                                                                                                                                     
    When not found, returns:                                                                                                                                                         
        {found: False, error: "not_in_catalog",                                                                                                                                      
         suggestions_in_file: [...], suggestions_in_source: [...]}                                                                                                                   
                                                                                                                                                                                     
    The `file` parameter is used only to separate same-file suggestions from                                                                                                         
    across-source suggestions when the lookup misses. It is NOT part of the                                                                                                          
    lookup key (qualified_name is globally unique within a source).                                                                                                                  
    """                                                                                                                                                                              
    doc_id = _element_doc_id(source_name, qualified_name)                                                                                                                            
    doc = library.get_document(doc_id)                                                                                                                                               
    if doc is not None and doc.metadata.get('kind') == CATALOG_KIND_ELEMENT:
        loc = doc.metadata.get('location') or {}                                                                                                                                     
        return {
            'found': True,                                                                                                                                                           
            'language': doc.metadata.get('language'),
            'subtype': doc.metadata.get('subtype'),                                                                                                                                  
            'qualified_name': doc.metadata.get('qualified_name'),
            'signature': doc.metadata.get('signature'),                                                                                                                              
            'location': {                                                                                                                                                            
                'line_start': loc.get('line_start'),
                'line_end': loc.get('line_end'),                                                                                                                                     
                'col_start': loc.get('col_start'),
                'col_end': loc.get('col_end'),                                                                                                                                       
            },
            'parent_qualified_name': doc.metadata.get('parent_qualified_name'),                                                                                                      
            'description': doc.metadata.get('description'),                                                                                                                          
            'file': (doc.source_files[0] if doc.source_files else None),
        }                                                                                                                                                                            
                
    # Miss — compute fuzzy suggestions                                                                                                                                               
    in_file, in_source = fuzzy_suggestions(library, source_name, file, qualified_name)                                                                                               
    return {
        'found': False,                                                                                                                                                              
        'error': 'not_in_catalog',                                                                                                                                                   
        'qualified_name': qualified_name,
        'suggestions_in_file': in_file,                                                                                                                                              
        'suggestions_in_source': in_source,
    }                                                                                                                                                                                
 
                                                                                                                                                                                     
def rank_symbol_candidates(query: str, qualified_names, limit: int = 5) -> list[str]:
    """Rank catalog qualified_names against a (possibly bare) symbol query.

    Structural matches outrank lexical similarity, and a quality floor drops
    far matches entirely — an unmatchable query yields [] rather than the
    least-bad garbage. (Plain difflib over full dotted names suggested
    unrelated symbols at ratio ~0.4 and missed case-styled constants.)

    Tiers: last segment == query > last segment ==, case-folded > query is an
    inner segment (member access under the queried symbol) > query's words are
    a subset of the last segment's words (any naming style) > of an inner
    segment's words > difflib on the case-folded LAST segments, floored at 0.7
    and always ranked below every structural tier. Ties break deterministically
    by (shorter qualified_name, lexicographic).
    """
    q = query.rsplit('.', 1)[-1]
    ql = q.lower()
    q_tokens = _segment_word_tokens(q)
    # Necessary-condition prefilter for the token tiers: if a segment contains
    # every query word as a word token, the longest one appears as a substring.
    q_probe = max(q_tokens, key=len) if q_tokens else None

    scored: list[tuple[float, int, str]] = []
    difflib_pool: dict[str, list[str]] = {}
    for qn in qualified_names:
        segments = qn.split('.')
        last = segments[-1]
        if last == q:
            score = 1.0
        elif last.lower() == ql:
            score = 0.95
        elif q in segments[:-1]:
            score = 0.9
        elif (q_probe is not None and q_probe in last.lower()
                and q_tokens <= _segment_word_tokens(last)):
            score = 0.8
        elif q_probe is not None and any(
            q_probe in s.lower() and q_tokens <= _segment_word_tokens(s)
            for s in segments[:-1]
        ):
            score = 0.7
        else:
            difflib_pool.setdefault(last.lower(), []).append(qn)
            continue
        scored.append((score, len(qn), qn))

    # Lexical tier: near-typos of the LAST segment only — full dotted names
    # inflate difflib ratios with shared prefixes and dots.
    if difflib_pool:
        for last_l in difflib.get_close_matches(ql, difflib_pool, n=limit * 3, cutoff=0.7):
            ratio = difflib.SequenceMatcher(None, ql, last_l).ratio()
            for qn in difflib_pool[last_l]:
                scored.append((0.69 * ratio, len(qn), qn))

    scored.sort(key=lambda t: (-t[0], t[1], t[2]))
    return [qn for _, _, qn in scored[:limit]]
def fuzzy_suggestions(
    library: "Library",
    source_name: str,
    file: str | None,
    qualified_name: str,
    limit: int = 5,
) -> tuple[list[str], list[str]]:
    """Return (suggestions_in_file, suggestions_in_source) — up to `limit` each.

    Candidates come from the per-source, unbounded element-name pool
    (:meth:`Library.list_catalog_element_names`) — never from a globally
    truncated document listing, which let one bulk-imported source evict
    every other source's suggestions. Same-file matches are prioritized;
    matches from the source's other files fill the second list, excluding any
    already in the first. Ranking and the quality floor live in
    :func:`rank_symbol_candidates`.
    """
    pairs = library.list_catalog_element_names(source_name)
    same_file_qns: list[str] = []
    other_qns: list[str] = []
    for qn, d_file in pairs:
        if file is not None and d_file == file:
            same_file_qns.append(qn)
        else:
            other_qns.append(qn)

    in_file = rank_symbol_candidates(qualified_name, same_file_qns, limit=limit)
    seen = set(in_file)
    in_source = [
        q for q in rank_symbol_candidates(qualified_name, other_qns, limit=limit * 2)
        if q not in seen
    ][:limit]
    return in_file, in_source
def list_elements_in_file(
    library: "Library",
    source_name: str,
    file: str,
) -> list[str]:
    """List qualified_names of all catalog elements whose source file matches `file`.

    Path matching is suffix-based: the doc's source_files entry must either
    end with `file` or vice versa, mirroring `lookup_symbol`'s file scoping.
    Reads the per-source element-name pool, not a globally truncated document
    listing.
    """
    pairs = library.list_catalog_element_names(source_name)
    return [
        qn for qn, d_file in pairs
        if d_file and (d_file.endswith(file) or file.endswith(d_file))
    ]


__all__ = ['config_usage', 'fuzzy_suggestions', 'get_element_body', 'list_elements_in_file', 'lookup_symbol', 'rank_symbol_candidates']


def _resolve_source_file(file_path: str, source_name: str) -> "Path":
    """Turn a catalog's stored path into something readable.

    Sources catalogued in place store ABSOLUTE paths; a spool stores paths
    relative to its corpus root (``delta/spark/.../X.scala``). Reading the
    stored value literally therefore worked for the former and failed for every
    spool symbol with ``read_failed: No such file or directory`` -- a knowledge
    tool unable to return the code it had indexed.

    Absolute-and-present wins untouched; otherwise the path is joined to the
    source's configured root. A still-missing file falls through to the caller's
    ``read_failed``, so a genuine absence stays loud.
    """
    from pathlib import Path as _Path

    p = _Path(file_path)
    if p.is_absolute() and p.exists():
        return p
    try:
        from config import get_config
        root = get_config().get_source_path(source_name)
    except Exception:      # noqa: BLE001 - config problems must not mask the read error
        return p
    if root is None:
        return p
    candidate = _Path(root) / file_path
    return candidate if candidate.exists() else p
def _scip_body_end(library, source_name: str, qualified_name: str):
    """The body end SCIP recorded for this symbol, or ``None``.

    ``MAX`` because overloads share a qualified name. Only an ``int`` is trusted: the
    body path is also called with test doubles, and a line number that is not a number
    is not a line number.
    """
    import sqlite3 as _sqlite3

    try:
        with library._conn_provider.acquire() as conn:
            row = conn.execute(
                'SELECT MAX(line_end) FROM scip_symbols '
                'WHERE source_name = ? AND qualified_name = ?',
                (source_name, qualified_name),
            ).fetchone()
    except (AttributeError, TypeError, _sqlite3.OperationalError):
        # No SCIP tier in this store (or not a real Library). Not an error: the
        # catalog location still answers, exactly as it did before.
        return None
    end = row[0] if row else None
    return end if isinstance(end, int) else None


def get_element_body(
    library: "Library",
    source_name: str,
    file: str | None,
    qualified_name: str,
) -> dict[str, Any]:
    """Return catalog element metadata plus the current body text."""
    from pathlib import Path as _P
    info = lookup_symbol(library, source_name, file, qualified_name)
    if not info.get('found'):
        return info
    file_path = info.get('file')
    loc = info.get('location') or {}
    ls = loc.get('line_start')
    le = loc.get('line_end')
    if not file_path or ls is None or le is None:
        info['body_error'] = 'missing_file_or_location'
        return info
    # A catalog location is the IDENTIFIER span -- measured on the live store,
    # ClassicMergeExecutor.writeAllChanges records line 285..285 with cols 16..31,
    # the 15 characters of its own name -- so a body read returned a signature
    # fragment. SCIP persists the definition's body extent, so prefer the wider of
    # the two. Widening only: a catalog extent that was already right is never cut.
    scip_end = _scip_body_end(library, source_name, qualified_name)
    if scip_end is not None and scip_end > le:
        le = scip_end
        info['body_extent'] = 'scip'
    resolved = _resolve_source_file(file_path, source_name)
    try:
        text = resolved.read_text(encoding='utf-8')
    except OSError as e:
        info['body_error'] = f'read_failed: {e}'
        return info
    src_lines = text.splitlines()
    body_lines = src_lines[ls - 1:le]
    info['body'] = '\n'.join(body_lines)
    info['body_line_count'] = len(body_lines)
    return info
def _dedup_read_sites(sites):
    """De-dup read-site dicts by ``(file, line)``, keeping first-seen
    order. A key read several times on the same line (e.g. a split-path
    chain whose segments are separate literals) collapses to one site."""
    seen: set[tuple[Any, Any]] = set()
    out: list[dict[str, Any]] = []
    for s in sites:
        loc = (s['file'], s['line'])
        if loc in seen:
            continue
        seen.add(loc)
        out.append(s)
    return out


def config_usage(
    library: "Library",
    source_name: str,
    key: str,
) -> dict[str, Any]:
    """Bridge a Typesafe Config key to its literal default (from the catalog)
    and the code sites that read it.

    Prefers the Tier 2 ``config_reads`` index — call-site-verified getter
    reads (chain-aware), each carrying its own resolved ``value`` and
    per-site ``confidence`` (``'config-resolved'`` or, for
    unsupported-language files, ``'string-match'``). When a key has no
    ``config_reads`` rows (e.g. the index predates the config-read pass),
    falls back to the Tier 1 string-literal value-equality join
    (``confidence='string-match'``). Read sites are de-duped by
    ``(file, line)``. See designs/config-code-bridge/tier2-resolution.md
    (Feature 6).
    """
    from docgen.scip_config_index import query_config_reads_by_key
    from docgen.scip_string_literal_index import query_string_literals_by_value

    definitions: list[dict[str, Any]] = []
    for d in library.list_documents(content_type='catalog'):
        md = d.metadata or {}
        if md.get('source_name') != source_name or md.get('kind') != CATALOG_KIND_ELEMENT:
            continue
        if md.get('subtype') not in _CONFIG_KEY_SUBTYPES:
            continue
        qn = md.get('qualified_name') or ''
        if qn == key or qn.endswith('.' + key):
            loc = md.get('location') or {}
            definitions.append({
                'qualified_name': qn,
                'file': (d.source_files[0] if d.source_files else None),
                'line': loc.get('line_start'),
                'default_value': md.get('signature'),
            })

    with library._conn_provider.acquire() as conn:
        reads = query_config_reads_by_key(
            source_name=source_name, key=key, conn=conn,
        )
        if reads:
            read_sites = _dedup_read_sites(
                {
                    'file': str(r.file), 'line': r.line,
                    'value': r.value, 'confidence': r.confidence,
                }
                for r in reads
            )
            confidence = 'config-resolved'
        else:
            literals = query_string_literals_by_value(
                source_name=source_name, value=key, conn=conn,
            )
            read_sites = _dedup_read_sites(
                {
                    'file': str(lit.file), 'line': lit.line_start,
                    'owning_symbol_id': lit.owning_symbol_id,
                    'confidence': 'string-match',
                }
                for lit in literals
            )
            confidence = 'string-match'

    notes: list[str] = []
    if not read_sites:
        notes.append(
            'no read sites found - the key may be read via a split/relative path '
            '(getConfig("a").getString("b")), built dynamically, or in an unindexed file.'
        )
    return {
        'found': bool(definitions),
        'key': key,
        'definitions': definitions,
        'read_sites': read_sites,
        'read_count': len(read_sites),
        'confidence': confidence,
        'notes': notes,
    }
