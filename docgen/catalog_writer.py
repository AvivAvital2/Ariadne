"""Catalog writer: bridges extracted elements into Ariadne Library."""
from __future__ import annotations

import asyncio as _asyncio
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

from docgen.catalog_extractor import ElementInfo, extract_elements
from schema import generate_deterministic_id

if TYPE_CHECKING:
    from library import Library
    from writer import LibraryWriter
 
                                                                                                                                                                                     
@frozen
class SyncSummary:
    file: str = ''
    added: int = 0
    modified: int = 0
    removed: int = 0
    unchanged: int = 0
    skipped: bool = False
    # SCIP-backed extraction (Scala/Java) failure carrier. None on the
    # happy path; populated when extract_elements raised a ScipError and
    # the caller didn't opt in to degraded mode.
    scip_error: object | None = None



CATALOG_EXTS = {
    '.py',
    '.html', '.htm',
    '.js', '.jsx', '.ts', '.tsx', '.mjs',
    '.json',
    '.yaml', '.yml',
    '.md', '.markdown',
    # SCIP-backed languages (Catalog transition + SCIP plan Phase C).
    '.scala', '.sbt', '.java',
    # Vue SFCs route through the javascript SCIP path (see
    # catalog_extractor._detect_language). The .vue.script.* companions
    # the extractor leaves behind are skipped via is_vue_companion so
    # their symbols aren't double-counted against the real .vue file.
    '.vue',
    # HOCON (Typesafe Config). File-index only — ast-grep doesn't parse
    # HOCON, so extract_elements returns []. Surfacing the file in the
    # index is enough for "where is activation.pub configured?" lookups.
    '.conf',
    # CSS. File-index only — no semantic symbols to extract; the
    # extractor returns [] (see catalog_extractor.extract_elements).
    # Closes the multi-language coverage gap for products that ship
    # CSS alongside HTML / JS / TS.
    '.css',
    # reStructuredText (Sphinx docs). File-index only (no
    # ast-grep grammar); see designs/rst-support.md.
    '.rst',
}

import re as _re

_VUE_COMPANION_RE = _re.compile(r'\.vue\.script\.(js|ts|jsx|tsx)$')


def is_vue_companion(path: "Path | str") -> bool:
    """True for ``*.vue.script.{js,ts,jsx,tsx}`` files — the transient
    companions the Vue extractor writes for scip-typescript. They must be
    skipped by catalog walks: their symbols belong to the real ``.vue``
    file (resolved via the vue-mapped SCIP index), not to the companion."""
    name = path if isinstance(path, str) else path.name
    return _VUE_COMPANION_RE.search(name) is not None                                                                                                       
                                                                                                                                                                                     
_MINIFIED_SUFFIXES = (
    '.min.js', '.min.mjs', '.min.cjs', '.min.jsx', '.min.css',
)
# Directory names whose JSON is test/fixture data, not documentation-worthy source.
_TEST_DIR_NAMES = frozenset({'test', 'tests', '__tests__', 'cypress', 'fixtures'})


def is_catalog_noise(path: "Path | str") -> bool:
    """True for files that pass the extension filter but are never worth the
    embed + LLM-describe cost: minified/vendored bundles, lockfiles, a
    framework's generated CSS output (e.g. Tailwind's compiled
    ``tailwind-output.css``), and JSON that lives under a test or fixtures
    directory. Generated *source* (e.g. ``*_pb2.py``) is deliberately NOT
    treated as noise — excluding it is an opt-in per-source choice.
    """
    p = Path(path)
    name = p.name.lower()
    if name.endswith(_MINIFIED_SUFFIXES):
        return True
    if name == 'package-lock.json' or name.endswith('-lock.json'):
        return True
    if name.endswith('.css') and 'tailwind' in name and 'output' in name:
        return True
    if p.suffix.lower() == '.json' and any(
        part.lower() in _TEST_DIR_NAMES for part in p.parts
    ):
        return True
    return False
 
def _now_iso() -> str:                                                                                                                                                               
    return datetime.now(UTC).isoformat()


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()                                                                                                                                                             
    try:
        h.update(path.read_bytes())                                                                                                                                                  
    except OSError:
        return ''                                                                                                                                                                    
    return h.hexdigest()
                                                                                                                                                                                     
                
def _element_doc_id(source_name: str, qualified_name: str) -> str:
    return generate_deterministic_id('catalog', f'element:{source_name}:{qualified_name}')
                                                                                                                                                                                     
 
def _file_index_doc_id(source_name: str, rel_path: str) -> str:                                                                                                                      
    return generate_deterministic_id('catalog', f'file_index:{source_name}:{rel_path}')                                                                                              
 
                                                                                                                                                                                     
def _element_content(el: ElementInfo) -> str:
    parent = f' in {el.parent_qualified_name}' if el.parent_qualified_name else ''                                                                                                   
    return f'{el.subtype} {el.qualified_name}{parent} [{el.language}] {el.file}:{el.line_start}-{el.line_end} :: {el.signature}'                                                     
                                                                                                                                                                                     
                                                                                                                                                                                     
def _element_metadata(el: ElementInfo, source_name: str) -> dict:
    meta: dict = {
        'kind': 'element',
        'source_name': source_name,
        'language': el.language,
        'subtype': el.subtype,
        'qualified_name': el.qualified_name,
        'signature': el.signature,
        'location': {
            'line_start': el.line_start,
            'line_end': el.line_end,
            'col_start': el.col_start,
            'col_end': el.col_end,
        },
        'parent_qualified_name': el.parent_qualified_name,
        'sha_at_sync': el.body_sha,
    }
    if el.documentation is not None:
        meta['documentation'] = el.documentation
    return meta
                                                                                                                                                                                     
                
def _file_index_metadata(source_name: str, language: str, file_sha: str, element_ids: list[str]) -> dict:
    return {                                                                                                                                                                         
        'kind': 'file_index',
        'source_name': source_name,                                                                                                                                                  
        'language': language,
        'file_sha': file_sha,                                                                                                                                                        
        'element_ids': list(element_ids),
        'last_synced': _now_iso(),                                                                                                                                                   
    }                                                                                                                                                                                
 
                                                                                                                                                                                     
def _detect_language(path: Path) -> str:
    """Return language string for a catalog file_index entry.

    Delegates to the canonical detector in catalog_extractor (which knows
    every supported extension). Falls back to 'unknown' for unsupported
    extensions; in practice this should not happen because iter_catalog_files
    pre-filters by CATALOG_EXTS.
    """
    from docgen.catalog_extractor import _detect_language as _detect_lang_ext
    return _detect_lang_ext(path) or 'unknown'


def iter_catalog_files(
    source_root: Path,
    exclude_patterns: tuple[str, ...] = (),
    exclude_dir_names: tuple[str, ...] | None = None,
) -> list[Path]:
    """Walk ``source_root`` and return catalog-eligible files.

    Shares the same exclusion semantics as
    :func:`docgen.staleness.find_catalog_files`: pruning is by
    directory NAME at every depth, drawn from a single canonical
    policy. No more two-list drift.

    Args:
        source_root: Directory to walk.
        exclude_patterns: Glob patterns matched per-file via Path.match.
            Use for files that share an extension with normal source
            but contain secrets (``**/secrets.json``, ``**/.env``).
        exclude_dir_names: Directory NAMES pruned from the walk.
            ``None`` (default) falls back to
            ``config.DEFAULT_EXCLUDE_POLICY``; pass an explicit tuple
            (typically ``Config.resolve_excluded_dirs(source_name)``)
            to override. ``()`` is honored verbatim (full walk).
    """
    from config import DEFAULT_EXCLUDE_POLICY

    if exclude_dir_names is None:
        exclude_dir_names = DEFAULT_EXCLUDE_POLICY

    excluded_set = frozenset(exclude_dir_names)
    results: list[Path] = []
    for path in source_root.rglob('*'):
        rel_parts = path.relative_to(source_root).parts
        if any(part in excluded_set for part in rel_parts):
            continue
        if not path.is_file():
            continue
        if path.suffix.lower() not in CATALOG_EXTS:
            continue
        if is_vue_companion(path):
            continue
        if is_catalog_noise(path):
            continue
        if any(path.match(pat) for pat in exclude_patterns):
            continue
        results.append(path)
    return results
async def sync_file_catalog(
    library: "Library",
    writer: "LibraryWriter",
    source_name: str,
    source_root: Path,
    file: Path,
    *,
    source_config=None,
    force: bool = False,
) -> SyncSummary:
    """Sync the catalog for a single file.

    ``force=True`` bypasses the file_sha short-circuit so the file is
    re-extracted even when the existing ``file_index`` doc has a
    matching sha. Useful after extractor / config changes that the sha
    check wouldn't otherwise notice.
    """
    from config import get_config
    from scope_resolution import make_scoped_library

    rel = str(file.relative_to(source_root))
    current_sha = _file_sha(file)
    if not current_sha:
        return SyncSummary(file=rel, skipped=True)

    # Read-side closure-scoped view; writes still go via the raw
    # ``library`` because ScopedLibrary intentionally doesn't expose
    # mutation methods.
    scoped = make_scoped_library(get_config(), library, source_name)

    index_id = _file_index_doc_id(source_name, rel)
    existing_index = scoped.get_document(index_id)

    if (
        not force
        and existing_index
        and existing_index.metadata.get('file_sha') == current_sha
    ):
        return SyncSummary(
            file=rel,
            unchanged=len(existing_index.metadata.get('element_ids', [])),
        )                       
                                                                                                                                                                        
    # SCIP-backed extraction (Scala/Java) raises ``ScipError`` subclasses
    # when the index is missing/stale/corrupt. Trap them so a single
    # broken language doesn't abort the whole batch sync; the caller
    # surfaces ``scip_error`` per file via the CLI.
    from docgen.scip_config import ScipError
    try:
        new_elements = extract_elements(
            file, source_root, source_config=source_config,
        )
    except ScipError as e:
        return SyncSummary(file=rel, skipped=True, scip_error=e)
    new_by_qn = {el.qualified_name: el for el in new_elements}
                                                                                                                                                                        
    old_ids: list[str] = list(existing_index.metadata.get('element_ids', [])) if existing_index else []
    old_by_qn: dict[str, tuple[str, str]] = {}
    for did in old_ids:
        d = scoped.get_document(did)
        if d:                                                                                                                                                           
            qn = d.metadata.get('qualified_name', '')                                                                                                                   
            if qn:                                                                                                                                                      
                old_by_qn[qn] = (d.id, str(d.metadata.get('sha_at_sync', '')))                                                                                        
                                                                                                                                                                        
    final_element_ids: list[str] = []
    added = modified = removed = unchanged = 0

    # Collect new elements in a single pass so we can batch-embed them.
    # Per-element ``writer.add_document`` makes one OpenAI call each — for
    # files with N elements that's N calls. With batching it's 1 call per
    # file regardless of N, which keeps catalog-sync below 502-territory
    # on large codebases.
    new_to_add: list[tuple[str, object, str]] = []  # (qn, el, doc_id)

    for qn, el in new_by_qn.items():
        doc_id = _element_doc_id(source_name, qn)
        existing = old_by_qn.get(qn)
        if existing is None:
            new_to_add.append((qn, el, doc_id))
            added += 1
        elif existing[1] != el.body_sha:
            library.update_document(
                doc_id,
                content=_element_content(el),
                metadata=_element_metadata(el, source_name),
            )
            modified += 1
        else:
            # Body unchanged but line numbers may have shifted due to edits
            # above this element. Refresh metadata so lookups return fresh
            # positions; skip expensive content regeneration.
            library.update_document(
                doc_id,
                metadata=_element_metadata(el, source_name),
            )
            unchanged += 1
        final_element_ids.append(doc_id)

    # Batch-embed all new elements, then bulk-insert. OpenAI caps per
    # request at 2048 inputs / ~300k tokens; chunk into safe sub-batches
    # so a single huge file (e.g. a Scala module with thousands of
    # symbols) doesn't 400.
    if new_to_add:
        embed_service = await writer._get_embedding_service()
        texts = [
            f'{qn}\n\n{_element_content(el)[:2000]}'
            for (qn, el, _) in new_to_add
        ]
        SUB_BATCH = 500
        embeddings: list = []
        for i in range(0, len(texts), SUB_BATCH):
            chunk = texts[i:i + SUB_BATCH]
            try:
                embeddings.extend(await embed_service.embed_batch(chunk))
            except Exception as e:
                # Re-raise with file context so the user knows which file
                # triggered an embedding API failure — without this, all
                # they see is "400 Bad Request" and the progress bar's
                # current-file line, which under concurrency > 1 isn't
                # necessarily the offender.
                est_tokens = sum(max(1, len(t) // 3) for t in chunk)
                raise RuntimeError(
                    f"Embedding batch failed for {rel} "
                    f"({len(chunk)} items, ~{est_tokens} est. tokens): {e}"
                ) from e
        for (qn, el, doc_id), emb in zip(new_to_add, embeddings):
            library.add_document(
                content_type='catalog',
                title=qn,
                content=_element_content(el),
                source_files=[el.file],
                embedding=emb,
                metadata=_element_metadata(el, source_name),
                doc_id=doc_id,
                source_name=source_name,
            )                                                                                                                                
                
    for qn, (doc_id, _sha) in old_by_qn.items():
        if qn not in new_by_qn:
            library.delete_document(doc_id)                                                                                                                           
            removed += 1                                                                                                                                                
                                                                                                                                                                        
    lang = _detect_language(file)                                                                                                                                       
    index_meta = _file_index_metadata(source_name, lang, current_sha, final_element_ids)                                                                                
    index_content = f'Catalog index for {rel} -- {len(final_element_ids)} elements.'
    if existing_index:                                                                                                                                                
        library.update_document(index_id, content=index_content, metadata=index_meta)                                                                                   
    else:                                                                                                                                                               
        await writer.add_document(
            content_type='catalog',
            title=f'file_index:{source_name}:{rel}',
            content=index_content,
            source_files=[str(file)],
            metadata=index_meta,
            doc_id=index_id,
            source_name=source_name,
        )                                                                                                                                                               
                                
    return SyncSummary(                                                                                                                                                 
        file=rel, added=added, modified=modified, removed=removed, unchanged=unchanged,
    )
async def sync_source_catalog(
    library: "Library",
    writer: "LibraryWriter",
    source_name: str,
    source_root: Path,
    *,
    source_config=None,
    on_progress=None,
    concurrency: int = 8,
    exclude_patterns: tuple[str, ...] = (),
    exclude_dir_names: tuple[str, ...] = (),
    force: bool = False,
) -> list[SyncSummary]:
    """Sync every catalog file under ``source_root``.

    ``concurrency`` bounds the number of files processed in parallel —
    SQLite's WAL mode serializes writes, but the embedding API roundtrips
    are network-bound and benefit massively from parallel calls.

    ``on_progress(current, total, current_file)`` — optional callback fired
    after each file completes with the running counter and the file's
    string path; called once more at the end with ``current == total``
    and ``current_file is None``.

    ``exclude_patterns`` / ``exclude_dir_names`` — forwarded to
    ``iter_catalog_files``. Use to keep secrets / credentials out of the
    catalog (and therefore out of embeddings + docs DB).
    """
    files = iter_catalog_files(
        source_root,
        exclude_patterns=exclude_patterns,
        exclude_dir_names=exclude_dir_names,
    )
    total = len(files)
    if total == 0:
        if on_progress is not None:
            on_progress(0, 0, None)
        return []

    sem = _asyncio.Semaphore(max(1, concurrency))
    completed = 0
    progress_lock = _asyncio.Lock()

    async def process_one(f: Path) -> SyncSummary:
        nonlocal completed
        async with sem:
            summary = await sync_file_catalog(
                library, writer, source_name, source_root, f,
                source_config=source_config,
                force=force,
            )
        async with progress_lock:
            completed += 1
            if on_progress is not None:
                on_progress(completed, total, str(f))
        return summary

    summaries = await _asyncio.gather(*[process_one(f) for f in files])
    if on_progress is not None:
        on_progress(total, total, None)
    return list(summaries)
                                                                                                                                                                                     
                
__all__ = [
    'SyncSummary',
    'iter_catalog_files',
    'sync_file_catalog',                                                                                                                                                             
    'sync_source_catalog',
]                                                                                                                                                                                    

                                                                                                                                                            
                
# ---------------------------------------------------------------------------                                                                                                        
# Incremental catalog update (task 8eda5ad62414)
# ---------------------------------------------------------------------------                                                                                                        
 
                                                                                                                                                                                     
_notify_file_locks: "dict[str, _asyncio.Lock]" = {}
                                                                                                                                                                                     
                
def _get_file_lock(rel_path: str) -> "_asyncio.Lock":                                                                                                                                
    lock = _notify_file_locks.get(rel_path)
    if lock is None:                                                                                                                                                                 
        lock = _asyncio.Lock()
        _notify_file_locks[rel_path] = lock                                                                                                                                          
    return lock 


async def notify_changed(
    library: "Library",
    writer: "LibraryWriter",
    source_name: str,
    changed_files: "list[str]",
    source_root: "Path | str | None" = None,
    *,
    source_config=None,
) -> "dict[str, dict]":
    """Incremental catalog update for a batch of changed files.
                                                                                                                                                                                     
    Handles add / modify / remove within files, plus cross-file moves
    (same qualified_name present in a different file after the batch).                                                                                                               
                                                                                                                                                                                     
    Concurrency: per-file asyncio.Lock prevents corruption when multiple                                                                                                             
    callers touch the same file simultaneously.                                                                                                                                      
                
    Returns a summary keyed by relative path:                                                                                                                                        
        {rel_path: {added, modified, removed, moved, unchanged, deleted}}                                                                                                            
    """
    from config import get_config
    from scope_resolution import make_scoped_library

    cfg = get_config()
    if source_root is None:
        resolved = cfg.resolve_source(source_name)
        if resolved is None:
            raise ValueError(f'Source not found: {source_name}')
        source_root = resolved
    source_root = Path(source_root)

    # Read-side closure-scoped view. Writes still go via the raw
    # ``library`` because ScopedLibrary doesn't expose mutation
    # methods (per the chokepoint discipline in
    # designs/directional-closure-scoping.md).
    scoped = make_scoped_library(cfg, library, source_name)
                                                                                                                                                                                     
    # Acquire per-file locks in sorted order (deterministic, avoids deadlock)
    rels = sorted(set(changed_files))                                                                                                                                                
    locks = [_get_file_lock(r) for r in rels]
                                                                                                                                                                                     
    summary: "dict[str, dict]" = {
        r: {'added': 0, 'modified': 0, 'removed': 0,                                                                                                                                 
            'moved': 0, 'unchanged': 0, 'deleted': False}                                                                                                                            
        for r in rels
    }                                                                                                                                                                                
                
    async def _aenter_all(locks):
        for lk in locks:                                                                                                                                                             
            await lk.acquire()                                                                                                                                                       
                                                                                                                                                                                     
    def _release_all(locks):                                                                                                                                                         
        for lk in reversed(locks):
            try:                                                                                                                                                                     
                lk.release()
            except RuntimeError:                                                                                                                                                     
                pass

    await _aenter_all(locks)
    try:                                                                                                                                                                             
        # --- collect new state + existing state ---
        file_states: "dict[str, dict]" = {}                                                                                                                                          
        from docgen.scip_config import ScipError
        for rel in rels:
            path = Path(source_root) / rel
            if path.exists() and path.is_file() and path.suffix.lower() in CATALOG_EXTS:
                try:
                    new_elements = extract_elements(
                        path, source_root, source_config=source_config,
                    )
                except ScipError as e:
                    # Don't crash the whole batch on a SCIP miss for one
                    # file — record the failure and skip element processing.
                    summary[rel]['scip_error'] = str(e)
                    file_states[rel] = {
                        'path': path,
                        'exists': False,
                        'sha': '',
                        'new_elements': [],
                    }
                    continue
                file_states[rel] = {
                    'path': path,
                    'exists': True,
                    'sha': _file_sha(path),
                    'new_elements': new_elements,
                }
            else:
                file_states[rel] = {
                    'path': path,
                    'exists': False,
                    'sha': '',
                    'new_elements': [],
                }                                                                                                                                                                    
 
        # Map of qualified_name in the NEW state across batch (for move detection)                                                                                                   
        new_qn_to_rel: "dict[str, str]" = {}                                                                                                                                         
        for rel, st in file_states.items():
            for el in st['new_elements']:                                                                                                                                            
                new_qn_to_rel[el.qualified_name] = rel                                                                                                                               
 
        # Load existing state: file_index docs + element docs for files in batch                                                                                                     
        existing_indexes: "dict[str, object]" = {}
        # qualified_name -> (doc_id, source_rel, sha_at_sync)                                                                                                                        
        existing_by_qn: "dict[str, tuple[str, str, str]]" = {}
        for rel in rels:                                                                                                                                                             
            idx_id = _file_index_doc_id(source_name, rel)
            idx = scoped.get_document(idx_id)                                                                                                                                       
            if idx is None:
                continue                                                                                                                                                             
            existing_indexes[rel] = idx
            for eid in idx.metadata.get('element_ids', []):                                                                                                                          
                d = scoped.get_document(eid)
                if d is None:                                                                                                                                                        
                    continue
                qn = d.metadata.get('qualified_name', '')
                if qn:                                                                                                                                                               
                    existing_by_qn[qn] = (d.id, rel, str(d.metadata.get('sha_at_sync', '')))
                                                                                                                                                                                     
        new_element_ids_per_file: "dict[str, list[str]]" = {r: [] for r in rels}
        processed_qns: set = set()                                                                                                                                                   
                                                                                                                                                                                     
        # --- apply add/modify/move ---
        for rel in rels:                                                                                                                                                             
            st = file_states[rel]
            if not st['exists']:
                summary[rel]['deleted'] = True                                                                                                                                       
                continue
                                                                                                                                                                                     
            existing_idx = existing_indexes.get(rel)
            if existing_idx is not None and existing_idx.metadata.get('file_sha') == st['sha']:
                # Short-circuit: file unchanged                                                                                                                                      
                ids = list(existing_idx.metadata.get('element_ids', []))
                summary[rel]['unchanged'] = len(ids)                                                                                                                                 
                new_element_ids_per_file[rel] = ids
                for eid in ids:                                                                                                                                                      
                    d = scoped.get_document(eid)
                    if d is not None:                                                                                                                                                
                        processed_qns.add(d.metadata.get('qualified_name', ''))
                continue                                                                                                                                                             
 
            for el in st['new_elements']:
                doc_id = _element_doc_id(source_name, el.qualified_name)
                existing = existing_by_qn.get(el.qualified_name)
                if existing is None:
                    await writer.add_document(
                        content_type='catalog',
                        title=el.qualified_name,
                        content=_element_content(el),
                        source_files=[el.file],
                        metadata=_element_metadata(el, source_name),
                        doc_id=doc_id,
                        source_name=source_name,
                    )
                    summary[rel]['added'] += 1                                                                                                                                       
                    new_element_ids_per_file[rel].append(doc_id)
                else:                                                                                                                                                                
                    old_doc_id, old_rel, old_sha = existing
                    if old_rel != rel:                                                                                                                                               
                        # Cross-file move — update in place
                        library.update_document(                                                                                                                                     
                            old_doc_id,
                            content=_element_content(el),
                            metadata=_element_metadata(el, source_name),                                                                                                             
                        )
                        summary[rel]['moved'] += 1                                                                                                                                   
                    elif old_sha != el.body_sha:
                        library.update_document(                                                                                                                                     
                            old_doc_id,
                            content=_element_content(el),                                                                                                                            
                            metadata=_element_metadata(el, source_name),
                        )                                                                                                                                                            
                        summary[rel]['modified'] += 1
                    else:                                                                                                                                                            
                        summary[rel]['unchanged'] += 1
                    new_element_ids_per_file[rel].append(old_doc_id)
                processed_qns.add(el.qualified_name)                                                                                                                                 
 
        # --- apply removals (elements absent from new state AND not moved) ---                                                                                                      
        for qn, (doc_id, old_rel, _sha) in existing_by_qn.items():                                                                                                                   
            if qn in processed_qns:
                continue                                                                                                                                                             
            if qn in new_qn_to_rel and new_qn_to_rel[qn] != old_rel:
                # Moved elsewhere — already handled in that file's processing                                                                                                        
                continue                                                                                                                                                             
            # True removal                                                                                                                                                           
            library.delete_document(doc_id)                                                                                                                                          
            summary[old_rel]['removed'] += 1                                                                                                                                         
                
        # --- update/delete file_index docs ---
        for rel in rels:                                                                                                                                                             
            st = file_states[rel]
            idx_id = _file_index_doc_id(source_name, rel)                                                                                                                            
            existing_idx = existing_indexes.get(rel)
                                                                                                                                                                                     
            if not st['exists']:
                if existing_idx is not None:                                                                                                                                         
                    library.delete_document(idx_id)
                continue

            if existing_idx is not None and existing_idx.metadata.get('file_sha') == st['sha']:
                continue  # unchanged; nothing to write

            lang = _detect_language(st['path'])
            meta = _file_index_metadata(
                source_name, lang, st['sha'], new_element_ids_per_file[rel],
            )                                                                                                                                                                        
            content = (
                f'Catalog index for {rel} -- '                                                                                                                                       
                f'{len(new_element_ids_per_file[rel])} elements.'                                                                                                                    
            )
            if existing_idx is not None:                                                                                                                                             
                library.update_document(idx_id, content=content, metadata=meta)
            else:
                await writer.add_document(
                    content_type='catalog',
                    title=f'file_index:{source_name}:{rel}',
                    content=content,
                    source_files=[str(st['path'])],
                    metadata=meta,
                    doc_id=idx_id,
                    source_name=source_name,
                )
        return summary
    finally:
        _release_all(locks)


__all__ = [
    'SyncSummary',
    'iter_catalog_files',
    'notify_changed',
    'sync_file_catalog',
    'sync_source_catalog',
]
