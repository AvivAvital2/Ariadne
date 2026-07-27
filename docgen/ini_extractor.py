"""INI-style `.conf` → list[ElementInfo].

`.conf` is overloaded: HOCON (Typesafe Config) is the canonical use, but
INI files — Sphinx ``theme.conf``, ``setup.cfg``, ``tox.ini``, systemd
units, git config — share the extension. The catalog dispatch sniffs a
``[section]`` header (via :func:`looks_like_ini`) and routes those here;
HOCON stays on the hocon extractor. HOCON reads ``[x]`` as an array, so a
section header is an unambiguous "this is INI, not HOCON" signal.

No ast-grep INI grammar is bundled, so this is a line scan: one
``ini_section`` element per ``[section]`` header and one ``ini_key`` per
``key = value`` / ``key : value`` (parented under its section). Comments
(``#`` / ``;`` / ``//``) and blanks are skipped. Multi-line continuation
values aren't split — a key element covers its declaring line. The scan
never raises; a file with no recognizable entries yields ``[]`` (the same
file-index fallback HOCON uses).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from docgen.catalog_extractor import ElementInfo

# A section name is an identifier-ish token — NOT quotes/commas (which would
# be a single-line HOCON root array like ``["a", "b"]``, not an INI section).
_SECTION_RE = re.compile(r'^\[(?P<name>[A-Za-z_][A-Za-z0-9_.\- ]*)\]\s*$')
# ``key = value`` / ``key : value``; the key precedes the FIRST separator so a
# URL value (``k = http://x``) keeps its ``:``. Empty values are allowed.
_KEY_RE = re.compile(r'^(?P<key>[^=:\s][^=:]*?)\s*[=:]\s*(?P<val>.*)$')
_COMMENT_PREFIXES = ('#', ';')
_SNIFF_STRUCTURAL_LINES = 5


def _is_comment_or_blank(line: str) -> bool:
    return (not line) or line[0] in _COMMENT_PREFIXES or line.startswith('//')


def looks_like_ini(src: str) -> bool:
    """True if ``src`` opens with an INI ``[section]`` header among its first
    structural (non-comment, non-blank) lines — the positive signal that a
    ``.conf`` is INI rather than HOCON."""
    checked = 0
    for raw in src.splitlines():
        line = raw.strip()
        if _is_comment_or_blank(line):
            continue
        if _SECTION_RE.match(line):
            return True
        checked += 1
        if checked >= _SNIFF_STRUCTURAL_LINES:
            break
    return False


def _module_qn(path: Path, source_root: Path) -> str:
    """Module-qualifier prefix from the file path (dots for slashes, no
    extension). Mirrors the HOCON/JSON/YAML extractors."""
    try:
        rel = path.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = Path(path.name)
    parts = list(rel.with_suffix('').parts)
    return '.'.join(parts) if parts else path.stem


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()


def _element(
    *, subtype: str, qn: str, parent: str | None, line: str, lineno: int,
    file: Path,
) -> ElementInfo:
    return ElementInfo(
        language='ini',
        subtype=subtype,
        file=str(file),
        qualified_name=qn,
        signature=line[:200],
        line_start=lineno,
        line_end=lineno,
        col_start=1,
        col_end=len(line) + 1,
        parent_qualified_name=parent,
        body_sha=_sha(line),
    )


def _extract_ini(
    src: str, path: Path, source_root: Path,
) -> list[ElementInfo]:
    """Emit one ``ini_section`` per header and one ``ini_key`` per key."""
    module_qn = _module_qn(path, source_root)
    out: list[ElementInfo] = []
    section_qn: str | None = None
    for i, raw in enumerate(src.splitlines()):
        line = raw.strip()
        if _is_comment_or_blank(line):
            continue
        sect = _SECTION_RE.match(line)
        if sect:
            name = sect.group('name').strip()
            section_qn = f'{module_qn}.{name}' if module_qn else name
            out.append(_element(
                subtype='ini_section', qn=section_qn,
                parent=module_qn or None, line=line, lineno=i + 1, file=path,
            ))
            continue
        key_match = _KEY_RE.match(line)
        if key_match:
            key = key_match.group('key').strip()
            parent = section_qn or (module_qn or None)
            qn = f'{parent}.{key}' if parent else key
            out.append(_element(
                subtype='ini_key', qn=qn, parent=parent,
                line=line, lineno=i + 1, file=path,
            ))
    return out


__all__ = ['looks_like_ini', '_extract_ini']
