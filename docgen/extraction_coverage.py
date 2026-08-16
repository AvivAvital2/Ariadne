"""Extraction-coverage version — a staleness *signal* for the SCIP ingest.

Ariadne's staleness is CONTENT-based (a file changing marks it stale). It watches
the *inputs*. This watches the *code*, because ``library_scip`` can go wrong
while every input is byte-identical.

The subject is anything that changes what ingest writes from an UNCHANGED
``.scip`` artifact — not only extractor coverage. Reading it as coverage alone
(a new language, extension, or sink) is what let the 2026-08-04 ingest rebuild
ship without a bump: it renamed no language and added no sink, it changed how
symbol identity, body extents, relationships and call-site attribution are
derived. The store then failed all four ingest invariants for two days while
``check`` correctly reported clean, comparing v1 to v1.

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

# Bump when the rows ingest writes from an unchanged ``.scip`` artifact would
# differ — new language / extension / sink, but equally a change to identity,
# extents, relationships, attribution, or a new column. Already-indexed sources
# are then flagged to re-run ``ariadne index`` (or ``--persist-only``, which
# rebuilds from the artifacts already on disk).
#   v1 — baseline: JS grammar covers .cjs; doc-language detection covers
#        .vue / .conf / .css (all previously drifted / missing).
#   v2 — the ingest rebuild and what followed it: ``local N`` ids namespaced per
#        document (one row had fused 4,446 files); body extents reconstructed so
#        a hop can be quoted; ``is_implementation`` ingested as ``implements``
#        edges; edges typed by what they point at; a call site attributed to the
#        enclosing callable rather than a local on the same line (34.65% of call
#        edges were owned by a variable); sources disk can no longer refresh
#        reconciled
#        away instead of surviving as unreachable rows.
EXTRACTION_COVERAGE_VERSION = 2

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
    """A one-line, actionable notice when ``source_name``'s ingest is behind, else
    ``None``.

    Names ``--persist-only`` because this drift is by definition code-side: the artifact
    is unchanged, only the rows written from it are behind. Re-indexing would rebuild a
    ``.scip`` that is already correct — on the databricks spool that is scip-java over 50
    packages to reproduce a byte-identical file.
    """
    gap = coverage_gap(source_root)
    if gap is None:
        return None
    stamped, current = gap
    return (
        f"SCIP ingest for '{source_name}' is behind (written at v{stamped}, "
        f"current v{current}) — run `ariadne index --persist-only --source "
        f"{source_name}` to rebuild library_scip from the .scip artifact already on "
        f"disk (no indexer, no LLM, no embedding, no doc-regen cost)."
    )
