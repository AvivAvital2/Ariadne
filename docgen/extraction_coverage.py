"""Extraction-coverage version — a staleness *signal* for the SCIP extractors.

Ariadne's staleness is CONTENT-based (a file changing marks it stale). A
code-level change to what the SCIP extractors cover — a new language, a new
extension (e.g. ``.cjs``), a new sink — is invisible to that, so a source
already indexed under older coverage silently keeps incomplete ``library_scip``
data until something unrelated retriggers indexing.

To close that: **bump** :data:`EXTRACTION_COVERAGE_VERSION` whenever extractor
coverage changes. ``ariadne index`` stamps the current version into each
source's ``.ariadne/manifest.json``; ``sync --status``, ``check``, and spool
status compare the stamp to the current version and loudly tell the user to
re-run ``ariadne index`` — a SCIP-layer refresh with no LLM, embedding, or
doc-regeneration cost. The signal self-clears on the next index.

Deliberately surfaced regardless of a source's ``ignore_staleness`` setting:
that opt-out silences *content-changed* nagging, but this is a one-time,
self-clearing *correctness* signal after an Ariadne upgrade.
"""
from __future__ import annotations

import json
from pathlib import Path

# Bump when the SCIP extractors' COVERAGE changes (new language / extension /
# sink), so already-indexed sources are flagged to re-run ``ariadne index``.
#   v1 — baseline: JS grammar covers .cjs; doc-language detection covers
#        .vue / .conf / .css (all previously drifted / missing).
EXTRACTION_COVERAGE_VERSION = 1

_MANIFEST_KEY = 'extraction_coverage_version'


def _manifest_path(source_root) -> Path:
    return Path(source_root) / '.ariadne' / 'manifest.json'


def _stamped_version(source_root) -> int:
    """Coverage version a source was last indexed under — 0 if never stamped,
    no manifest, or unreadable/malformed."""
    try:
        data = json.loads(_manifest_path(source_root).read_text(encoding='utf-8'))
        return int(data.get(_MANIFEST_KEY, 0))
    except (OSError, ValueError, TypeError, AttributeError):
        return 0


def coverage_gap(source_root) -> 'tuple[int, int] | None':
    """``(stamped, current)`` when the source's persisted extraction is behind
    the current coverage version, else ``None``."""
    stamped = _stamped_version(source_root)
    if stamped < EXTRACTION_COVERAGE_VERSION:
        return (stamped, EXTRACTION_COVERAGE_VERSION)
    return None


def stamp_coverage(source_root) -> None:
    """Record that ``source_root`` was just (re)indexed under the current
    coverage version, in its ``.ariadne/manifest.json`` — created if needed,
    all other keys preserved."""
    path = _manifest_path(source_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if not isinstance(data, dict):
            data = {}
    except (OSError, ValueError):
        data = {}
    data[_MANIFEST_KEY] = EXTRACTION_COVERAGE_VERSION
    path.write_text(json.dumps(data, indent=2), encoding='utf-8')


def coverage_notice(source_name, source_root) -> 'str | None':
    """A one-line, actionable notice when ``source_name``'s extraction is
    behind the current coverage, else ``None``."""
    gap = coverage_gap(source_root)
    if gap is None:
        return None
    stamped, current = gap
    return (
        f"SCIP extraction for '{source_name}' is behind (indexed at v{stamped}, "
        f"current v{current}) — run `ariadne index --source {source_name}` to "
        f"refresh string-literal / route / config intelligence "
        f"(SCIP-layer only: no LLM, embedding, or doc-regen cost)."
    )
