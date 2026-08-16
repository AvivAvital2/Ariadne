"""Is the index behind the working tree?

Rebuilt to own **only** that question. The module this replaces also owned path
normalization, and that mixing is why the normalization bug — trusting the longest matching
indexer ``cwd``, so ``spark/sql/core/...`` over-stripped to ``core/...`` — sat unnoticed
next to staleness reporting that nothing about paths depended on. Normalization now lives in
``docgen/scip_paths.py``, and the path functions here delegate to it so there is one rule.

Freshness is deliberately **conservative**: a file that resolves under no indexer ``cwd``
is left unflagged, because a re-index is the remedy for adds and removals and a false alarm
trains people to ignore the warning. A source with no manifest stays silent — *cannot
determine* is not the same as *fresh*, and it must not be reported as either.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

from docgen.scip_paths import indexer_cwds, scip_candidates


def _parse(iso: str | None) -> 'datetime | None':
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso)
    except ValueError:
        return None


def changed_indexed_files(source_root, indexers, indexed_files):
    """``(earliest_indexed_at, sorted[changed relative file])``.

    A file is changed when it resolves under some indexer's ``cwd`` to a path whose mtime
    is newer than that indexer's ``indexed_at``. Resolving under no ``cwd`` leaves it
    unflagged — see the module docstring on why silence beats a false alarm.
    """
    root = Path(source_root)
    scopes = [(root / ix.get('cwd', '.'), _parse(ix.get('indexed_at')))
              for ix in indexers]
    changed = []
    for relative in indexed_files:
        for scope_root, cutoff in scopes:
            if cutoff is None:
                continue
            try:
                mtime = datetime.fromtimestamp(
                    (scope_root / relative).stat().st_mtime, tz=timezone.utc)
            except OSError:
                continue          # not under this cwd — try the next indexer
            if mtime > cutoff:
                changed.append(relative)
            break                 # resolved here; stop scanning
    stamps = [ix.get('indexed_at') for ix in indexers if ix.get('indexed_at')]
    return (min(stamps) if stamps else None), sorted(set(changed))


def stale_report_for_files(cfg, files_by_source):
    """``{source: (indexed_at, [changed file])}`` for each source behind its tree.

    Connection-free: callers supply the indexed file lists. ``ignore_staleness: true``
    exempts a source entirely; a glob list exempts matching files.
    """
    out: dict = {}
    for name, files in files_by_source.items():
        ignore = cfg.source_ignore_staleness(name)
        if ignore is True:
            continue
        source_config = cfg.get_source_config(name)
        if source_config is None:
            continue
        root = Path(source_config.path).expanduser().resolve()
        try:
            manifest = json.loads(
                (root / '.ariadne' / 'manifest.json').read_text(encoding='utf-8'))
        except (OSError, ValueError):
            continue              # never indexed via manifest -> unknown, stay silent
        indexers = manifest.get('indexers') or []
        if not indexers:
            continue
        candidates = files
        if isinstance(ignore, (list, tuple)) and ignore:
            candidates = [f for f in files
                          if not any(fnmatch(f, pattern) for pattern in ignore)]
        indexed_at, changed = changed_indexed_files(root, indexers, candidates)
        if changed:
            out[name] = (indexed_at, changed)
    return out


def stale_report(conn, cfg, source_names):
    """``stale_report_for_files`` with the indexed files read from the store."""
    files_by_source = {}
    for name in source_names:
        files_by_source[name] = [
            row[0] for row in conn.execute(
                'SELECT DISTINCT file FROM scip_symbols WHERE source_name = ?',
                (name,))
        ]
    return stale_report_for_files(cfg, files_by_source)


def freshness_warning(name, indexed_at, changed):
    """One line, shared by the CLI and MCP read paths so it reads the same everywhere."""
    count = len(changed)
    shown = ', '.join(changed[:3]) + (f', +{count - 3} more' if count > 3 else '')
    when = f' (indexed {indexed_at})' if indexed_at else ''
    return (f'index may be stale: {count} indexed file(s) in "{name}" changed since '
            f'the last index{when} — e.g. {shown}. Results may be incomplete; '
            f'run `ariadne index --source {name}` (or `ariadne sync`) to refresh.')


def source_for_file(cfg, file_path):
    """The configured source owning ``file_path`` — longest matching root wins.

    Longest, so a source nested inside another repository resolves to the nested one.
    """
    target = Path(file_path).expanduser()
    best: tuple[int, str] | None = None
    for name, root in cfg.get_all_source_paths().items():
        resolved = Path(str(root)).expanduser()
        try:
            target.resolve().relative_to(resolved.resolve())
        except (OSError, ValueError):
            continue
        depth = len(resolved.resolve().as_posix())
        if best is None or depth > best[0]:
            best = (depth, name)
    return best[1] if best is not None else None


def resolve_scip_file(cfg, file_path, conn=None):
    """``(source_name, scip-relative file)`` for ``file_path``, or ``(None, None)``.

    The returned path is the file as ``scip_symbols`` stores it — relative to the
    **indexer's** ``cwd``, not the source root, so a multi-package source resolves.

    Pass ``conn`` to have candidates **verified against the index**, which is the accurate
    form: measured over 91 retrieved files, verification joins 59 against 36 for the rule
    this module previously used (trusting the longest matching ``cwd``, which over-strips).
    Without a connection the least-strip candidate is returned unverified — best effort,
    because it preserves the most path.
    """
    source = source_for_file(cfg, file_path)
    if source is None:
        return None, None
    root = Path(str(cfg.get_all_source_paths()[source])).expanduser().resolve()
    try:
        relative = Path(file_path).expanduser().resolve().relative_to(root).as_posix()
    except (OSError, ValueError):
        return source, None

    candidates = scip_candidates(relative, indexer_cwds(root), None)
    if conn is None:
        # The least strip that actually strips: `candidates[0]` is the path unchanged,
        # which is the 0-of-91 failure, and the last is the over-strip.
        stripped = [c for c in candidates if c != relative]
        return source, (stripped[0] if stripped else relative)
    for candidate in candidates:
        if conn.execute(
                'SELECT 1 FROM scip_symbols WHERE source_name = ? AND file = ? LIMIT 1',
                (source, candidate)).fetchone() is not None:
            return source, candidate
    return source, None


def absolute_from_scip(cfg, source, scip_rel):
    """Absolute path of a ``scip_symbols``-relative file — the inverse cwd-strip.

    Tries each indexer ``cwd`` longest first and returns the first that exists, falling
    back to ``source_root/scip_rel`` (correct for a single-root source).
    """
    root = Path(str(cfg.get_all_source_paths()[source])).expanduser().resolve()
    for cwd in sorted(indexer_cwds(root), key=len, reverse=True):
        candidate = root / scip_rel if cwd in ('.', '') else root / cwd / scip_rel
        if candidate.exists():
            return str(candidate)
    return str(root / scip_rel)
