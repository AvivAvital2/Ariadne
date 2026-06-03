"""SCIP source-config types and error hierarchy (SCIP plan, Phase A.3).

The error hierarchy is the centerpiece of the fail-loud contract: when
a source declares ``index_kinds.scala = "scip"`` (or ``.java``), any
problem with the SCIP artifact must surface as a structured error, not
a silent fallback to ast-grep. ``catalog-sync`` exits with a non-zero
code per ``ScipError`` subclass; CLI users see the reason and can
decide whether to use ``--allow-degraded``.
"""
from __future__ import annotations

from pathlib import Path

from attrs import field, frozen


@frozen
class SourceScipConfig:
    """Per-source SCIP configuration loaded from ``ariadne.yaml``.

    ``index_kinds`` maps language → indexer name (currently only
    ``"scip"`` is supported, but the field is structured this way so
    other indexers can be added without schema breakage). ``allow_degraded``
    is set per-call by the ``--allow-degraded`` CLI flag, NOT in
    ``ariadne.yaml`` — the file expresses *intent*, the runtime flag
    expresses *opt-in to known degradation*.
    """
    repo: str
    artifact_path: Path
    max_staleness_days: int = 7
    index_kinds: dict[str, str] = field(factory=dict)
    allow_degraded: bool = False
    # Merged Vue companion→.vue mapping across all of the source's
    # indexer entries (collected from the manifest's ``vue_mapping``
    # keys). Applied to the loaded index in ``resolve_index`` so catalog
    # extraction of ``.vue`` files sees ``.vue`` paths, mirroring what
    # the cross-source graph loader does. Empty for non-Vue sources.
    vue_mappings: dict[str, dict] = field(factory=dict)


class ScipError(Exception):
    """Base exception for any SCIP-artifact failure.

    Carries structured fields (``repo``, ``reason``) so the CLI can
    render a useful banner without parsing the message string.
    """

    def __init__(
        self,
        *,
        repo: str,
        reason: str,
        expected_commit: str | None = None,
        last_good_commit: str | None = None,
        last_good_age_days: int | None = None,
    ) -> None:
        super().__init__(f'{type(self).__name__}(repo={repo!r}, reason={reason!r})')
        self.repo = repo
        self.reason = reason
        self.expected_commit = expected_commit
        self.last_good_commit = last_good_commit
        self.last_good_age_days = last_good_age_days


class ScipUnavailableError(ScipError):
    """Raised when the SCIP artifact at ``artifact_path`` does not exist
    or cannot be opened. ``reason`` is typically ``"index_missing"``.
    """


class ScipTooStaleError(ScipError):
    """Raised when the SCIP artifact's mtime is older than
    ``max_staleness_days``. ``reason`` is typically
    ``"index_too_stale"``; ``last_good_age_days`` carries the actual age.
    """


class ScipCorruptError(ScipError):
    """Raised when the SCIP artifact exists and is fresh but cannot be
    decoded — protobuf parse failure, truncation, or wrong format.
    ``reason`` is typically ``"index_corrupt"``.
    """


# Module-level cache for loaded ScipIndex instances. Keyed by
# (artifact_path_str, mtime_ns) so a re-indexed artifact invalidates
# automatically on mtime change. Without this, every Scala/Java file
# in a sync re-reads + re-parses the same multi-MB .scip artifact.
_index_cache: dict[tuple[str, int], object] = {}


def resolve_index(
    config: SourceScipConfig | None, lang: str,
):
    """Resolve a ``ScipIndex`` for ``lang``, or return None if not declared.

    - ``config is None`` or ``index_kinds[lang] != "scip"`` → returns None
      (caller falls through to language-default extraction).
    - SCIP declared → returns a (possibly cached) ``ScipIndex``; raises a
      ``ScipError`` subclass on missing / stale / corrupt artifacts. The
      caller MUST NOT catch and silently fall back unless
      ``config.allow_degraded`` is True.

    The return type isn't annotated to avoid a circular import; callers
    can import ``ScipIndex`` from ``docgen.scip_extractor`` if they need
    the type for an isinstance check.
    """
    if config is None:
        return None
    if config.index_kinds.get(lang) != 'scip':
        return None

    artifact_str = str(config.artifact_path)
    try:
        mtime_ns = config.artifact_path.stat().st_mtime_ns
    except OSError:
        # Missing file — fall through to ScipIndex.load which raises the
        # canonical ScipUnavailableError.
        mtime_ns = -1

    cache_key = (artifact_str, mtime_ns)
    cached = _index_cache.get(cache_key)
    if cached is not None:
        return cached

    # Lazy import — scip_extractor imports back into this module for the
    # error classes; importing it at module load would be circular.
    from docgen.scip_extractor import ScipIndex, apply_vue_mapping
    index = ScipIndex.load(
        config.artifact_path,
        repo=config.repo,
        max_staleness_days=config.max_staleness_days,
    )
    # Translate Vue companion paths (Foo.vue.script.ts) back to the
    # original .vue files so catalog extraction of .vue sees .vue paths.
    if config.vue_mappings:
        index = apply_vue_mapping(index, config.vue_mappings)
    if mtime_ns >= 0:
        _index_cache[cache_key] = index
    return index


__all__ = [
    'ScipCorruptError',
    'ScipError',
    'ScipTooStaleError',
    'ScipUnavailableError',
    'SourceScipConfig',
    'resolve_index',
]
