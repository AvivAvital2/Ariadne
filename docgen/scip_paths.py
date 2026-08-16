"""The doc↔SCIP path seam, owned in one place.

``documents.source_files`` and ``scip_symbols.file`` are in different path spaces. Measured
on the live store, **0 of 91 files retrieved at production width join as stored**: a spool
corpus is indexed once per repo with the repo as the indexer ``cwd``, so document paths
carry a prefix SCIP does not, and an ordinary source's document paths are absolute.

That join is what the ``(source, file, line_start)`` seek depends on, and it had no owner —
so three consumers reinvented it, differently. This module is the owner.

Two rules, both measured:

* **Verify against the index; do not trust a rule.** Preferring the longest matching ``cwd``
  over-strips ``spark/sql/core/...`` to ``core/...`` — 36 of 91 joined. Verifying each
  candidate: 59 of 91.
* **Least strip wins.** The right answer keeps the most path, so ``spark`` beats
  ``spark/sql`` when both are prefixes and both verify.

A path that resolves to nothing is returned as unresolved, never guessed. Of the live 91,
32 are genuinely absent — 20 markdown/html, plus 12 ``.java``/``.scala`` under a module that
was never indexed. **"Not indexed" and "no such symbol" are different answers**, and only
the first one is honest about a coverage gap.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pathlib import Path
    from sqlite3 import Connection


def scip_candidates(path: str, indexer_cwds: tuple[str, ...],
                    source_root: str | None) -> list[str]:
    """Every path this document path could be, least strip first.

    Ordered so that the candidate preserving the most path is tried first: that is the
    correct answer whenever more than one verifies.
    """
    bases = [path]
    root = (source_root or '').rstrip('/')
    if root and path.startswith(root + '/'):
        bases.append(path[len(root) + 1:])

    out: list[str] = []
    for base in bases:
        if base not in out:
            out.append(base)
        # shorter cwd first — least strip
        for cwd in sorted({c for c in indexer_cwds if c not in ('.', '')}, key=len):
            prefix = cwd.rstrip('/') + '/'
            if base.startswith(prefix):
                stripped = base[len(prefix):]
                if stripped and stripped not in out:
                    out.append(stripped)
    return out


def scip_paths_for(
    conn: 'Connection',
    paths: list[str],
    *,
    source: str,
    indexer_cwds: tuple[str, ...] = (),
    source_root: str | None = None,
) -> tuple[dict[str, str], tuple[str, ...]]:
    """Map document paths onto the paths ``scip_symbols`` stores for ``source``.

    Returns ``(document path -> scip path, unresolved document paths)``. Candidates come
    from the indexer manifest, never a hardcoded repo list, and each is checked against the
    index — so this cannot invent a path that looks plausible and matches nothing.
    """
    resolved: dict[str, str] = {}
    unresolved: list[str] = []
    known: dict[str, bool] = {}

    def indexed(candidate: str) -> bool:
        if candidate not in known:
            known[candidate] = conn.execute(
                'SELECT 1 FROM scip_symbols WHERE source_name = ? AND file = ? LIMIT 1',
                (source, candidate),
            ).fetchone() is not None
        return known[candidate]

    for path in paths:
        if path in resolved or path in unresolved:
            continue
        hit = next((c for c in scip_candidates(path, indexer_cwds, source_root)
                    if indexed(c)), None)
        if hit is None:
            unresolved.append(path)
        else:
            resolved[path] = hit
    return resolved, tuple(unresolved)


def indexer_cwds(source_root: 'str | Path') -> tuple[str, ...]:
    """The indexer ``cwd`` values from a source's manifest.

    A multi-package source is indexed once per package, each with its own ``cwd``, and
    ``scip_symbols.file`` is relative to that ``cwd`` rather than to the source root. So
    these are the prefixes a document path may carry that SCIP does not — the reason the
    seam exists at all. ``('.',)`` when the manifest is missing: the single-root default.

    Reading the manifest belongs here, with the rest of the path knowledge, rather than
    beside the staleness reporting it happened to live next to before.
    """
    import json
    from pathlib import Path as _Path

    try:
        manifest = json.loads(
            (_Path(source_root) / '.ariadne' / 'manifest.json').read_text(
                encoding='utf-8'))
    except (OSError, ValueError):
        return ('.',)
    found = tuple(ix.get('cwd', '.') for ix in manifest.get('indexers', ()))
    return found or ('.',)
