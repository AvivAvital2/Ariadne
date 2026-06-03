"""SCIP indexer-location discovery (SCIP-everywhere, Phase 2j + 2j.b).

Walks a source tree and identifies indexer-relevant clusters via the
language registry in :mod:`docgen.scip_languages`. The output is fed
to ``ariadne discover`` which writes ``<source>/.ariadne/manifest.json``;
``ariadne index`` then consumes the manifest and runs each indexer in
its declared cwd.

The walk is **single-pass**: every file is examined exactly once. During
that walk we both classify marker files (signals a "package" entry) and
catalog source-file extensions (signals a "scripts" entry candidate).
No second pass.

Two flavors of :class:`DiscoveryEntry`:

- ``entry_kind='package'`` — directory has a marker file (``__init__.py``,
  ``package.json``, ``build.sbt``/``pom.xml``/``build.gradle*``). The
  indexer treats the cwd as a package root.
- ``entry_kind='scripts'`` — directory has source files of an
  ``can_index_standalone`` language but NO marker. The indexer treats
  each file as a standalone module (pyright via ``include: ["./*.py"]``;
  scip-typescript via ``--infer-tsconfig``). JVM ``can_index_standalone``
  is False, so JVM orphans are detected for visibility but NOT emitted
  — scip-java needs a build tool.

Coalescing rules (post-walk):

- **Python packages:** emit one entry per *top-level* package — i.e. a
  directory with ``__init__.py`` whose parent does NOT have
  ``__init__.py``. The ``cwd`` is the parent directory (so scip-python
  sees the package as a package). Sibling top-level packages sharing
  the same parent coalesce into one entry.
- **TypeScript/JVM packages:** emit only the topmost marker in any tree
  branch — nested ``package.json`` files (workspaces, vendored deps)
  are scip-typescript's concern, not ours.
- **Script entries:** one entry per ``(directory, language)`` pair. No
  cross-directory coalescing — orphan dirs have no implicit package
  relationship.

Coverage check for script entries: a directory is "covered" by a
package entry of the same language if it falls under that entry's cwd.
This avoids double-indexing files that pyright/scip-typescript would
already pick up via the package's include pattern.

Exclusion handling: a directory whose name is in ``exclude_dirs`` is
skipped entirely (its subtree is not walked). If the same directory is
also in ``exempt_dirs``, exempt wins and the directory IS walked.
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from attrs import frozen

from docgen.scip_languages import (
    _EXT_TO_LANG,
    _MARKER_TO_LANG,
    LANGUAGES,
    LanguageDef,
)


@lru_cache(maxsize=1024)
def _glob_to_regex(pattern: str) -> re.Pattern[str]:
    """Translate a gitignore-style glob to a compiled regex.

    Supports:
    - ``**`` — any number of path components (including zero); when
      followed by ``/``, the slash is optional so ``**/foo`` matches
      both ``foo`` and ``a/b/foo``
    - ``*`` — anything except ``/``
    - ``?`` — a single character except ``/``

    Patterns are matched against POSIX-normalized paths relative to
    the source root (no leading slash). Cached because a single
    discover run typically uses the same pattern set hundreds of
    times during the walk.
    """
    parts: list[str] = []
    i = 0
    n = len(pattern)
    while i < n:
        c = pattern[i]
        if c == '*' and i + 1 < n and pattern[i + 1] == '*':
            parts.append('.*')
            i += 2
            if i < n and pattern[i] == '/':
                # Make the separator after ** optional so '**/foo' also
                # matches 'foo' at the root.
                parts.append('/?')
                i += 1
        elif c == '*':
            parts.append('[^/]*')
            i += 1
        elif c == '?':
            parts.append('[^/]')
            i += 1
        else:
            parts.append(re.escape(c))
            i += 1
    return re.compile('^' + ''.join(parts) + '$')


# Kept as a Literal for type-hint stability across modules that import
# it. The set is the union of indexer_kind values from LANGUAGES.
IndexerKind = Literal['python', 'typescript', 'java']


@frozen
class DiscoveryEntry:
    """One indexer location identified by discovery.

    ``entry_kind`` distinguishes:

    - ``'package'`` — the indexer's cwd has a marker file; it should
      treat the cwd as a package root.
    - ``'scripts'`` — the indexer's cwd contains standalone source files
      with no marker; it should treat each file as an isolated module
      (e.g., pyright with ``include: ["./*.py"]``).

    Default is ``'package'`` so existing callers / tests that don't
    specify the field continue to work.
    """
    kind: IndexerKind
    cwd: Path
    markers: tuple[Path, ...]
    entry_kind: Literal['package', 'scripts'] = 'package'


_MINIFIED_JS_SUFFIXES = ('.min.js', '.min.mjs', '.min.cjs', '.min.jsx')


def _is_minified_js(filename: str) -> bool:
    """True for minified JS bundles like ``jquery-1.9.0.min.js``.

    Minified files are built third-party artifacts, not source — there
    are no meaningful symbols to index. Excluding them from the source
    catalog means a directory holding ONLY vendored bundles (jQuery,
    Highcharts, …) never becomes a standalone TypeScript scope, which
    scip-typescript would otherwise reject with "no files got indexed",
    aborting the entire index. A directory that also has real ``.ts`` /
    ``.js`` source still gets its scope — the bundle is simply ignored.
    """
    lower = filename.lower()
    return lower.endswith(_MINIFIED_JS_SUFFIXES)


def _by_name(name: str) -> LanguageDef:
    """Look up a LanguageDef by its name. Used internally for places
    that need to bridge the python/jvm coalesce-by-parent rule."""
    for lang in LANGUAGES:
        if lang.name == name:
            return lang
    raise KeyError(f'no language named {name!r} in registry')


def discover(
    source_root: Path,
    *,
    exclude_dirs: frozenset[str] = frozenset(),
    exempt_dirs: frozenset[str] = frozenset(),
    exclude_patterns: frozenset[str] = frozenset(),
) -> list[DiscoveryEntry]:
    """Walk ``source_root`` and identify per-language indexer scopes.

    ``exclude_patterns`` are gitignore-style file globs matched against
    paths relative to ``source_root``. Files matching any pattern are
    skipped during the walk — they don't trigger marker detection AND
    they don't get cataloged for script-entry emission. This is what
    ``SourceConfig.exclude`` from ``ariadne.yaml`` flows into.

    Returns a list of DiscoveryEntry, sorted by ``(kind, cwd)`` for
    deterministic output. Pure Python — uses pathlib, no shell.
    """
    source_root = source_root.resolve()
    if not source_root.is_dir():
        return []

    # Pre-compile patterns (cached at module level via lru_cache).
    pattern_regexes = tuple(_glob_to_regex(p) for p in exclude_patterns)

    def is_excluded_by_pattern(file_path: Path) -> bool:
        if not pattern_regexes:
            return False
        try:
            rel = file_path.relative_to(source_root)
        except ValueError:
            return False
        rel_posix = rel.as_posix()
        return any(r.match(rel_posix) for r in pattern_regexes)

    # Single walk populates BOTH:
    # - Marker dirs per language (for package detection)
    # - File catalog: dir → {language → list of source files}
    package_markers: dict[LanguageDef, dict[Path, str]] = {
        lang: {} for lang in LANGUAGES
    }
    file_catalog: dict[Path, dict[LanguageDef, list[Path]]] = {}

    # Cycle guard, keyed on resolved path. Without this, a symlink loop
    # walks the same subtree N times before the kernel's ELOOP returns
    # an error (32 follows on macOS, varies elsewhere). With it, every
    # directory is walked exactly once.
    seen: set[Path] = set()

    def walk(directory: Path) -> None:
        name = directory.name
        if name in exclude_dirs and name not in exempt_dirs:
            return

        try:
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            return
        if resolved in seen:
            return
        seen.add(resolved)

        try:
            entries_in_dir = list(directory.iterdir())
        except (OSError, PermissionError):
            return

        files = [e for e in entries_in_dir if e.is_file()]
        subdirs = [e for e in entries_in_dir if e.is_dir()]

        for f in files:
            # Apply exclude_patterns FIRST — a file excluded here
            # contributes neither markers nor to the file catalog.
            if is_excluded_by_pattern(f):
                continue
            # 1. Marker-file detection (package entries)
            marker_lang = _MARKER_TO_LANG.get(f.name)
            if marker_lang is not None:
                package_markers[marker_lang].setdefault(directory, f.name)
            # 2. Source-file cataloging (script entries; also includes
            #    files in package dirs — we filter via coverage in
            #    Phase B).
            ext_lang = _EXT_TO_LANG.get(f.suffix.lower())
            if ext_lang is not None and not _is_minified_js(f.name):
                file_catalog.setdefault(directory, {}).setdefault(
                    ext_lang, [],
                ).append(f)

        for d in subdirs:
            walk(d)

    walk(source_root)

    entries: list[DiscoveryEntry] = []

    # ----- Phase A: package entries -----

    # Python: coalesce top-level packages by parent dir.
    python_lang = _by_name('python')
    py_marker_dirs = set(package_markers[python_lang].keys())
    top_level_python = sorted(
        d for d in py_marker_dirs if d.parent not in py_marker_dirs
    )
    python_scopes: dict[Path, list[Path]] = {}
    for pkg_dir in top_level_python:
        parent = pkg_dir.parent
        # Bound parent to source_root. When source_root itself is a
        # Python package (has ``__init__.py`` at its top — e.g. Ariadne),
        # ``pkg_dir.parent`` would walk OUTSIDE the source. scip-python
        # would then operate on whatever sibling repos live alongside,
        # silently indexing tens of thousands of unrelated files.
        try:
            parent.relative_to(source_root)
        except ValueError:
            parent = source_root
        python_scopes.setdefault(parent, []).append(
            pkg_dir / '__init__.py',
        )
    for parent in sorted(python_scopes):
        markers = tuple(sorted(python_scopes[parent]))
        entries.append(DiscoveryEntry(
            kind='python',
            cwd=parent,
            markers=markers,
            entry_kind='package',
        ))

    # TypeScript / JVM: emit topmost marker in any tree branch
    # (no parent-coalescing — these tools handle nested workspaces /
    # multi-module builds themselves).
    for lang in LANGUAGES:
        if lang.name == 'python':
            continue
        marker_dirs = package_markers[lang]
        top = sorted(
            d for d in marker_dirs
            if not any(p in marker_dirs for p in d.parents)
        )
        for d in top:
            marker_filename = marker_dirs[d]
            entries.append(DiscoveryEntry(
                kind=lang.indexer_kind,  # type: ignore[arg-type]
                cwd=d,
                markers=(d / marker_filename,),
                entry_kind='package',
            ))

    # ----- Phase B: script entries -----

    # Build coverage map: each language → set of cwds of its package
    # entries. A directory is "covered" if it's under any of those cwds.
    package_cwds_by_lang: dict[LanguageDef, set[Path]] = {
        lang: set() for lang in LANGUAGES
    }
    for entry in entries:
        if entry.entry_kind != 'package':
            continue
        # Map indexer_kind back to LanguageDef
        for lang in LANGUAGES:
            if lang.indexer_kind == entry.kind:
                package_cwds_by_lang[lang].add(entry.cwd)
                break

    def _covered(directory: Path, lang: LanguageDef) -> bool:
        for pkg_cwd in package_cwds_by_lang[lang]:
            try:
                directory.relative_to(pkg_cwd)
                return True
            except ValueError:
                continue
        return False

    # Iterate file_catalog deterministically (by directory path).
    for directory in sorted(file_catalog, key=lambda p: str(p)):
        lang_files = file_catalog[directory]
        # Iterate languages deterministically too (by name).
        for lang in sorted(lang_files, key=lambda l: l.name):
            if not lang.can_index_standalone:
                # JVM orphans: silently skipped. scip-java can't index
                # them. Layer C may track them as endpoint files.
                continue
            if _covered(directory, lang):
                continue
            entries.append(DiscoveryEntry(
                kind=lang.indexer_kind,  # type: ignore[arg-type]
                cwd=directory,
                markers=tuple(sorted(lang_files[lang])),
                entry_kind='scripts',
            ))

    return entries


__all__ = ['DiscoveryEntry', 'IndexerKind', 'discover']
