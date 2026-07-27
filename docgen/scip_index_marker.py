"""Completion marker for a source's SCIP index (``.ariadne/index.ok``).

The per-scope intermediates and the merged ``index.scip`` are written
NON-atomically by the external indexers and the merge step (the tool writes
straight to the final path). So an interrupted or killed run can leave a
truncated file at the canonical location. This marker makes reuse safe:

- it is published **atomically** (write to a temp file, then ``os.replace``)
  ONLY after a fully successful index;
- reuse is gated on it, and it records the corpus shas the index was built
  from, so a moved pin isn't trusted either.

An interrupted build therefore leaves no marker (or one whose recorded shas no
longer match), and the next run rebuilds instead of reusing a torn ``.scip``.
"""

from __future__ import annotations

import json
from pathlib import Path

MARKER_NAME = 'index.ok'
_CORPUS_SHA_MARKER = '.ariadne-corpus-sha'


def current_corpus_shas(source_root: Path) -> dict[str, str]:
    """Map each fetched corpus clone (one level under ``source_root``) to its
    pinned sha, read from the ``.ariadne-corpus-sha`` marker the spool fetch
    writes. Empty for non-spool sources (which have no such markers), so the
    match check below is vacuously satisfied for them.
    """
    shas: dict[str, str] = {}
    for marker in sorted(source_root.glob(f'*/{_CORPUS_SHA_MARKER}')):
        try:
            shas[marker.parent.name] = marker.read_text(encoding='utf-8').strip()
        except OSError:
            continue
    return shas


def read_marker(ariadne_dir: Path) -> dict | None:
    """Parse ``.ariadne/index.ok``; ``None`` if it is missing or unparseable
    (fail-closed — a corrupt marker is never trusted)."""
    try:
        return json.loads((ariadne_dir / MARKER_NAME).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write_marker(
    ariadne_dir: Path,
    *,
    indexer_versions: dict[str, str],
    corpus_shas: dict[str, str],
) -> None:
    """Atomically publish the completion marker: write a temp file in the same
    directory, then ``Path.replace`` it onto the final name. ``replace`` is an
    atomic rename on the same filesystem, so a crash mid-write can only leave a
    stray ``.tmp`` — never a torn ``index.ok`` at the canonical path."""
    ariadne_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            'version': 1,
            'indexer_versions': indexer_versions,
            'corpus_shas': corpus_shas,
        },
        indent=2,
        sort_keys=True,
    )
    tmp = ariadne_dir / (MARKER_NAME + '.tmp')
    tmp.write_text(payload, encoding='utf-8')
    tmp.replace(ariadne_dir / MARKER_NAME)


def invalidate_marker(ariadne_dir: Path) -> None:
    """Remove the marker before re-indexing, so an interrupted rebuild leaves no
    marker and the next run cannot reuse a half-written index."""
    (ariadne_dir / MARKER_NAME).unlink(missing_ok=True)


def index_complete(ariadne_dir: Path, source_root: Path) -> bool:
    """True iff a prior index finished cleanly AND still matches the corpus: the
    marker is present and parseable, and its recorded corpus shas equal the shas
    on disk now (so a changed pin is not trusted)."""
    marker = read_marker(ariadne_dir)
    if marker is None:
        return False
    return marker.get('corpus_shas', {}) == current_corpus_shas(source_root)
