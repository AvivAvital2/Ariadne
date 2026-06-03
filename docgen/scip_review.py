"""Discover review tooling — flag suspect entries and prompt the user.

After ``discover()`` produces entries, ``ariadne discover --review``
walks any flagged suspects (likely vendor bundles, mass-duplicate
dirs) and asks the user y/N. Rejected suspects' suggested patterns
get persisted to ``ariadne.yaml`` via :mod:`docgen.yaml_writer`.

Heuristics intentionally minimal:

- ``vendor_minified`` — every marker file ends with ``.min.js`` /
  ``.min.css`` / ``.bundle.*`` (a vendored, pre-built bundle is the
  whole content of the directory; mixed dirs are NOT flagged).
- ``mass_duplicate`` — ≥10 markers all sharing the same extension
  (catches large vendored script collections and auto-generated
  filesets).
- ``generated_output`` — reserved for future heuristics; currently
  unused since common build dirs are already excluded by
  ``DEFAULT_EXCLUDE_POLICY``.

Pattern-matching the suspects is the *only* place the heuristic
fires. The classifier never decides on its own; it just flags. The
user always has the final say.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Literal

from attrs import frozen

from docgen.scip_discovery import DiscoveryEntry


# Suffix sets for the vendor-minified heuristic. Order matters for
# pattern construction: longer / more specific suffixes first.
_MINIFIED_SUFFIXES: tuple[str, ...] = (
    '.min.js', '.min.css', '.min.mjs',
)
_BUNDLE_SUFFIXES: tuple[str, ...] = (
    '.bundle.js', '.bundle.css', '.bundle.mjs',
)
_VENDOR_SUFFIXES: tuple[str, ...] = _MINIFIED_SUFFIXES + _BUNDLE_SUFFIXES

# Threshold for the mass-duplicate heuristic. Below this count, even
# same-extension dirs aren't flagged — small clusters are usually
# legitimate.
_MASS_DUPLICATE_THRESHOLD: int = 10


@frozen
class Suspect:
    """One DiscoveryEntry flagged as likely-noise.

    ``suggested_pattern`` is what the user would paste into
    ``ariadne.yaml`` under the source's ``exclude:`` list — also what
    the auto-edit path writes if the user rejects the entry.
    ``description`` is one line shown at the prompt.
    """
    entry: DiscoveryEntry
    reason: Literal[
        'vendor_minified', 'mass_duplicate', 'generated_output',
    ]
    suggested_pattern: str
    description: str


def _matches_vendor_suffix(filename: str) -> str | None:
    """Return the matching vendor suffix, or None."""
    for suffix in _VENDOR_SUFFIXES:
        if filename.endswith(suffix):
            return suffix
    return None


def _is_vendor_minified(entry: DiscoveryEntry) -> bool:
    """All markers end with a vendor suffix (entirely-minified dir)."""
    if not entry.markers:
        return False
    return all(
        _matches_vendor_suffix(m.name) is not None
        for m in entry.markers
    )


def _is_mass_duplicate(entry: DiscoveryEntry) -> bool:
    """≥N markers, all sharing the same extension."""
    if len(entry.markers) < _MASS_DUPLICATE_THRESHOLD:
        return False
    extensions = {m.suffix.lower() for m in entry.markers}
    return len(extensions) == 1


def _classify(
    entry: DiscoveryEntry,
    *,
    source_root: Path,
) -> Suspect | None:
    """Apply heuristics to one entry. Returns the first matching
    Suspect, or None if the entry passes."""
    try:
        cwd_rel = entry.cwd.relative_to(source_root)
    except ValueError:
        # Entry's cwd isn't under source_root — shouldn't happen in
        # practice, but be defensive.
        return None
    cwd_rel_posix = cwd_rel.as_posix()

    if _is_vendor_minified(entry):
        # All markers share a vendor suffix; pick the suffix from the
        # first marker for the suggested pattern.
        first_suffix = _matches_vendor_suffix(entry.markers[0].name)
        # _is_vendor_minified guarantees this is non-None.
        assert first_suffix is not None
        pattern = f'{cwd_rel_posix}/*{first_suffix}'
        return Suspect(
            entry=entry,
            reason='vendor_minified',
            suggested_pattern=pattern,
            description=(
                f'{len(entry.markers)} minified vendor bundle(s) in '
                f'{cwd_rel_posix}'
            ),
        )

    if _is_mass_duplicate(entry):
        ext = entry.markers[0].suffix.lower()
        pattern = f'{cwd_rel_posix}/*{ext}'
        return Suspect(
            entry=entry,
            reason='mass_duplicate',
            suggested_pattern=pattern,
            description=(
                f'{len(entry.markers)} {ext} files in {cwd_rel_posix} — '
                f'mass-duplicate set'
            ),
        )

    return None


def classify_suspects(
    entries: list[DiscoveryEntry],
    *,
    source_root: Path,
) -> list[Suspect]:
    """Flag entries that look like vendor / generated / duplicated noise.

    ``source_root`` is used to compute paths for the suggested
    exclude patterns (always relative to the source root, since
    that's what ``SourceConfig.exclude`` patterns expect).
    """
    suspects: list[Suspect] = []
    for entry in entries:
        suspect = _classify(entry, source_root=source_root)
        if suspect is not None:
            suspects.append(suspect)
    return suspects


def prompt_keep_entry(
    suspect: Suspect,
    *,
    input_fn: Callable[[str], str] = input,
    output_fn: Callable[..., None] = print,
) -> bool:
    """Interactive prompt — does the user want to KEEP (index) the
    entry, or EXCLUDE it?

    Returns ``True`` to keep, ``False`` to exclude. Default (empty
    input) is exclude — conservative for noise removal. ``input_fn``
    and ``output_fn`` are dependency-injected so tests can stub them
    OS-independently.
    """
    output_fn(f'\n{suspect.description}')
    output_fn(f'Suggested exclude pattern: {suspect.suggested_pattern}')
    response = input_fn('Index this anyway? [y/N] ').strip().lower()
    return response in ('y', 'yes')


__all__ = [
    'Suspect',
    'classify_suspects',
    'prompt_keep_entry',
]
