"""Config-getter call-site enumerator (Tier 2 Feature 3).

Builds the key↔code map that Tier 1's value-equality join only
approximates. The candidate set is the already-populated
``string_literals`` table — a Typesafe Config getter call necessarily
carries its key as a string-literal argument, so every literal is a
candidate getter-call site. For each candidate line we:

1. detect the file's language and read its source,
2. classify every candidate line's assignment RHS from one parse with
   :func:`inspect_definitions_at_lines` (chain-aware —
   ``getConfig("a").getString("b")`` → key ``a.b``), keeping only
   ``getter_call`` results (this drops literals that are not getter
   arguments, e.g. log messages),
3. resolve the key against ``config_values`` (value is ``None`` when the
   key isn't declared in any ``.conf`` — the read is still real),

and emit one :class:`ConfigRead` per confirmed getter call
(``confidence='config-resolved'``).

Files whose language the inspector can't read (or that can't be read off
disk) fall back to the Tier-1 string-match: a literal whose value equals
a known config key is recorded as an approximate read
(``confidence='string-match'``). Literals that match nothing are dropped.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from docgen.scip_config_index import ConfigRead, query_config_values_for_source
from docgen.scip_definition_inspector import inspect_definitions_at_lines
from docgen.scip_string_literal_extractor import _detect_lang

if TYPE_CHECKING:
    from sqlite3 import Connection


def _config_values_map(source_name: str, conn: 'Connection') -> dict[str, str]:
    """All config values for ``source_name`` as ``{key: value}``, first
    occurrence winning. One bulk fetch replaces a per-read key lookup
    (config_values.value is NOT NULL, so a missing key is the only way
    ``.get`` returns None)."""
    out: dict[str, str] = {}
    for cv in query_config_values_for_source(source_name=source_name, conn=conn):
        out.setdefault(cv.key, cv.value)
    return out


def extract_config_reads(
    *, source_name: str, conn: 'Connection',
) -> list[ConfigRead]:
    """Enumerate config-getter read sites for ``source_name`` from the
    populated ``string_literals`` + ``config_values`` tables. Reads
    source files off disk (``string_literals`` stores absolute paths).
    Returns one :class:`ConfigRead` per confirmed getter call, plus the
    string-match fallback rows. Empty when nothing matches."""
    values_by_key = _config_values_map(source_name, conn)
    cur = conn.execute(
        'SELECT file, line_start, col_start, value FROM string_literals '
        "WHERE source_name = ? AND kind = 'plain' ORDER BY file, line_start, col_start",
        (source_name,),
    )
    by_file: dict[str, list[tuple[int, int, str]]] = {}
    for file, line, col, value in cur.fetchall():
        by_file.setdefault(file, []).append((line, col, value))

    reads: list[ConfigRead] = []
    for file, literals in by_file.items():
        path = Path(file)
        lang = _detect_lang(path)
        text: str | None = None
        if lang is not None:
            try:
                text = path.read_text(encoding='utf-8', errors='replace')
            except OSError:
                text = None

        if lang is not None and text is not None:
            reads.extend(
                _resolved_reads(path, lang, text, literals, values_by_key),
            )
        else:
            reads.extend(
                _string_match_reads(path, literals, values_by_key),
            )
    return reads


def _resolved_reads(
    path: 'Path', lang: str, text: str,
    literals: list[tuple[int, int, str]], values_by_key: dict[str, str],
) -> list['ConfigRead']:
    """One read per candidate line that classifies as a getter call.
    Classifies all candidate lines from a SINGLE parse of the file
    (``inspect_definitions_at_lines``) — O(1) per line, not a re-parse
    per line."""
    lines: dict[int, list[int]] = {}
    for line, col, _value in literals:
        lines.setdefault(line, []).append(col)

    results = inspect_definitions_at_lines(
        source_text=text, lines=lines.keys(), language=lang,
    )
    out: list[ConfigRead] = []
    for line, cols in sorted(lines.items()):
        result = results.get(line)
        if result is None or result.kind != 'getter_call':
            continue
        out.append(ConfigRead(
            file=path,
            line=line,
            # Inspection is per-line, not per-column; record the leftmost
            # literal on the line as an approximate cursor for the read.
            col=min(cols),
            key=result.config_key,
            value=values_by_key.get(result.config_key),
            confidence='config-resolved',
        ))
    return out


def _string_match_reads(
    path: 'Path', literals: list[tuple[int, int, str]],
    values_by_key: dict[str, str],
) -> list['ConfigRead']:
    """Fallback for unreadable / unsupported-language files: a literal
    whose value equals a known config key is an approximate read."""
    out: list[ConfigRead] = []
    for line, col, value in literals:
        resolved = values_by_key.get(value)
        if resolved is not None:
            out.append(ConfigRead(
                file=path, line=line, col=col, key=value,
                value=resolved, confidence='string-match',
            ))
    return out


__all__ = ['ConfigRead', 'extract_config_reads']
