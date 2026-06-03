from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from attrs import frozen


@frozen
class SourceEntry:
    """One configured Ariadne source as the bot advertises it to the agent."""

    name: str
    description: str | None
    aliases: tuple[str, ...]


def build_roster(
    source_names: Iterable[str],
    descriptions: Mapping[str, str] | None = None,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> list[SourceEntry]:
    """Build the source roster the system prompt advertises.

    ``source_names`` is the authoritative set (typically ``get_config().sources``
    keys), so every configured source is listed even if it has no curated
    description or aliases. Descriptions/aliases are overlaid from the bridge's
    own config. Sorted by name for a deterministic prompt.
    """
    descriptions = descriptions or {}
    aliases = aliases or {}
    return [
        SourceEntry(
            name=name,
            description=descriptions.get(name),
            aliases=tuple(aliases.get(name, ())),
        )
        for name in sorted(set(source_names))
    ]


def resolve_alias(term: str, roster: Iterable[SourceEntry]) -> str | None:
    """Map a colloquial term (or canonical name) to a canonical source name.

    Case-insensitive exact match against each source's name or aliases. Returns
    ``None`` when nothing matches — the agent then asks the user to disambiguate
    rather than guessing. This is a bridge-only convenience; the agent does the
    primary mapping from the roster rendered into its prompt.
    """
    needle = term.strip().lower()
    for entry in roster:
        if needle == entry.name.lower():
            return entry.name
        if any(needle == alias.lower() for alias in entry.aliases):
            return entry.name
    return None
