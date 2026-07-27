"""Generation-completion marker for a source's docs (`.ariadne/generation.ok`).

A staleness-exempt source with a PINNED corpus (a spool) has a frozen file set:
once its docs are fully generated at a given corpus sha, a re-run has nothing to
discover or regenerate. This marker records that state so the generate step can
skip the discovery walk + staleness pass entirely — rather than re-walking an
immutable tree looking for changes that cannot exist.

It is written ONLY after a fully-successful generation, atomically, and keyed on
the corpus shas + the doc types generated. A re-pinned corpus (shas change) or a
request for a doc type not yet generated invalidates it, so regeneration
proceeds correctly.

Corpus-sha discovery is shared with the SCIP index marker.
"""

from __future__ import annotations

import json
from pathlib import Path

from docgen.scip_index_marker import current_corpus_shas

MARKER_NAME = 'generation.ok'

__all__ = [
    'MARKER_NAME',
    'current_corpus_shas',
    'generation_complete',
    'invalidate_marker',
    'read_marker',
    'write_marker',
]


def read_marker(ariadne_dir: Path) -> dict | None:
    """Parse `.ariadne/generation.ok`; None if missing or unparseable (fail-closed)."""
    try:
        return json.loads((Path(ariadne_dir) / MARKER_NAME).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def write_marker(
    ariadne_dir: Path,
    *,
    corpus_shas: dict[str, str],
    doc_types,
) -> None:
    """Atomically record that generation completed for this corpus + doc types."""
    ariadne_dir = Path(ariadne_dir)
    ariadne_dir.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(
        {
            'version': 1,
            'corpus_shas': dict(corpus_shas),
            'doc_types': sorted(set(doc_types or ())),
        },
        indent=2,
        sort_keys=True,
    )
    tmp = ariadne_dir / (MARKER_NAME + '.tmp')
    tmp.write_text(payload, encoding='utf-8')
    tmp.replace(ariadne_dir / MARKER_NAME)


def invalidate_marker(ariadne_dir: Path) -> None:
    """Remove the marker (before a run that will generate), so a partial or
    interrupted run leaves no marker that would wrongly skip next time."""
    (Path(ariadne_dir) / MARKER_NAME).unlink(missing_ok=True)


def generation_complete(
    ariadne_dir: Path,
    source_root: Path,
    *,
    requested_doc_types,
) -> bool:
    """True iff a prior generation finished fully for THIS pinned corpus AND
    already covers every requested doc type. False (regenerate) otherwise."""
    marker = read_marker(ariadne_dir)
    if marker is None:
        return False
    if marker.get('corpus_shas', {}) != current_corpus_shas(Path(source_root)):
        return False
    recorded = set(marker.get('doc_types', []))
    return set(requested_doc_types or ()).issubset(recorded)
