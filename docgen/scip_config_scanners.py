"""Config-source scanner registry (Phase 2o / Layer B).

Walks a source tree and extracts ``(file, key, value, line_start)``
tuples from configuration files. The output is the raw material that
Layer C's resolution traversal (Phase 2s) consumes when walking SCIP
refs from a sink call site backward through config-key lookups.

Layer B does NOT directly produce ``(code_dir → env_dir)`` mappings —
that requires SCIP-ref traversal. Layer B's contract is just config
extraction.

MVP scope:

- HOCON (``*.conf``) — uses the in-tree Lark parser
- Dotenv (``.env``, ``.env.<suffix>``) — KEY=VALUE per line

Deferred (Phase 2o.b):

- Dockerfile (ENTRYPOINT/CMD/RUN/ENV directives)
- YAML (``*.yaml`` / ``*.yml`` for kustomize patches and similar)

Adding a new format = one scanner function + one entry in the dispatch
table at the bottom of this module.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path

from attrs import frozen


@frozen
class ConfigValue:
    """One ``(key, value)`` pair extracted from a config file.

    ``file`` is the absolute path. ``key`` uses dotted notation for
    nested structures (HOCON ``a { b = "c" }`` → ``a.b``). ``value``
    is always a string (numbers/booleans stringified for uniform
    downstream handling). ``line_start`` is 1-indexed for diagnostics.
    """
    file: Path
    key: str
    value: str
    line_start: int


# ---------------------------------------------------------------------------
# Dotenv scanner — KEY=VALUE per line
# ---------------------------------------------------------------------------


def scan_dotenv(file: Path) -> list[ConfigValue]:
    """Parse a ``.env``-style file (``KEY=VALUE`` per line).

    Surrounding single or double quotes on the value are stripped.
    Lines starting with ``#`` are comments. Blank lines are skipped
    but counted toward line numbers.
    """
    try:
        text = file.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []

    values: list[ConfigValue] = []
    for i, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' not in line:
            continue
        key, _, value = line.partition('=')
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and (
            (value[0] == '"' and value[-1] == '"')
            or (value[0] == "'" and value[-1] == "'")
        ):
            value = value[1:-1]
        if not key:
            continue
        values.append(ConfigValue(
            file=file, key=key, value=value, line_start=i,
        ))
    return values


# ---------------------------------------------------------------------------
# HOCON scanner — uses the in-tree Lark parser
# ---------------------------------------------------------------------------


def _key_path_text(key_path_node) -> str:
    """Reconstruct dotted key path from a ``key_path`` Tree node.

    Mirrors the existing helper in ``docgen/hocon_extractor.py`` —
    skips the ``.`` separator tokens, joins remaining KEY tokens.
    """
    parts: list[str] = []
    for child in key_path_node.children:
        token_text = str(child).strip()
        if token_text and token_text != '.':
            # Quoted keys (`"name"`) carry quotes in the token; strip them so
            # the dotted key reads `parent.name`, not `parent."name"`.
            parts.append(_strip_quotes(token_text))
    return '.'.join(parts)


def _strip_quotes(text: str) -> str:
    """Strip surrounding double quotes from a HOCON string token."""
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        # JSON-style — keep simple. Real HOCON allows escape sequences;
        # for Layer B's purposes we don't need to interpret them.
        return text[1:-1]
    return text


def _strip_triple_quotes(text: str) -> str:
    if len(text) >= 6 and text[:3] == '"""' and text[-3:] == '"""':
        return text[3:-3]
    return text


def _scalar_to_string(scalar_node) -> str | None:
    """Convert a ``scalar`` tree node to its string value.

    Returns None for substitutions (``${VAR}``) — those need Layer C
    resolution, not Layer B's flat extraction.
    """
    if not scalar_node.children:
        return None
    inner = scalar_node.children[0]
    # Token vs Tree
    if hasattr(inner, 'type'):
        # Token
        token_text = str(inner)
        if inner.type == 'STRING':
            return _strip_quotes(token_text)
        if inner.type == 'TRIPLE_STRING':
            return _strip_triple_quotes(token_text)
        # NUMBER, BOOL, NULL, UNQUOTED_VALUE — return as-is
        return token_text
    # Tree (substitution) → skip
    return None


def _value_to_string(value_node) -> str | None:
    """Extract a scalar string from a ``value`` tree node, or None
    if the value is structured (object/array), a substitution, or a
    concatenation."""
    if not value_node.children:
        return None
    # Value concatenation (`"http://"${host}`, `${base.dir}/log/events`)
    # yields multiple atoms. We can't flatten it to one resolved value, so
    # skip it rather than store a misleading first-atom truncation.
    if len(value_node.children) != 1:
        return None
    inner = value_node.children[0]
    if not hasattr(inner, 'data'):
        return None
    if inner.data == 'scalar':
        return _scalar_to_string(inner)
    if inner.data in ('duration_value', 'size_value'):
        # Two children: NUMBER, UNIT — concatenate
        return ' '.join(str(c) for c in inner.children)
    # object / array → not a leaf scalar, skip
    return None


def _walk_hocon(node, prefix: str, file: Path) -> Iterable[ConfigValue]:
    """Recursively walk a HOCON ``object_body`` Tree, yielding
    ConfigValue per assignment. Nested blocks contribute their key as
    a dotted prefix to inner assignments."""
    for child in node.children:
        if not hasattr(child, 'data'):
            continue  # token (separator), already filtered by grammar
        if child.data != 'entry':
            continue
        if not child.children:
            continue
        inner = child.children[0]
        if not hasattr(inner, 'data'):
            continue

        if inner.data == 'assignment':
            if len(inner.children) < 3:
                continue
            key = _key_path_text(inner.children[0])
            full_key = f'{prefix}.{key}' if prefix else key
            value_text = _value_to_string(inner.children[2])
            if value_text is None:
                continue
            line = (
                inner.meta.line
                if hasattr(inner, 'meta') and not inner.meta.empty
                else 0
            )
            yield ConfigValue(
                file=file,
                key=full_key,
                value=value_text,
                line_start=line,
            )

        elif inner.data == 'object_block':
            if len(inner.children) < 2:
                continue
            key = _key_path_text(inner.children[0])
            full_key = f'{prefix}.{key}' if prefix else key
            obj_node = inner.children[1]  # 'object' tree
            if not obj_node.children:
                continue
            inner_body = obj_node.children[0]  # 'object_body' tree
            yield from _walk_hocon(inner_body, full_key, file)

        # else: include / other — skip


def scan_hocon(file: Path) -> list[ConfigValue]:
    """Parse a HOCON file and yield one ConfigValue per assignment.

    Nested blocks flatten to dotted keys: ``resources { python = "..." }``
    yields ``key='resources.python'``. Substitutions (``${VAR}``) and
    structured values (objects/arrays) are skipped — they're not flat
    scalars Layer C can match against.

    Parse errors return ``[]`` rather than raising; one malformed
    config file shouldn't abort the entire scan.
    """
    try:
        text = file.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []

    try:
        from docgen.hocon_grammar import parse as hocon_parse
        tree = hocon_parse(text)
    except Exception:
        # Malformed HOCON → skip this file. Other scanners still get
        # to run on other files.
        return []

    return list(_walk_hocon(tree, prefix='', file=file))


# ---------------------------------------------------------------------------
# Aggregator — walks tree, dispatches by file shape
# ---------------------------------------------------------------------------


# Extension → scanner. Adding a new format = one entry here.
_SCANNER_BY_EXT: dict[str, Callable[[Path], list[ConfigValue]]] = {
    '.conf': scan_hocon,
}


def _scanner_for_file(file: Path) -> Callable[[Path], list[ConfigValue]] | None:
    """Resolve which scanner (if any) should handle this file.

    Extension match wins; otherwise basename pattern (``.env`` /
    ``.env.<suffix>``) routes to the dotenv scanner.
    """
    ext = file.suffix.lower()
    if ext in _SCANNER_BY_EXT:
        return _SCANNER_BY_EXT[ext]
    name = file.name
    if name == '.env' or name.startswith('.env.'):
        return scan_dotenv
    return None


def scan_config_sources(
    source_root: Path,
    *,
    exclude_dirs: frozenset[str] = frozenset(),
) -> list[ConfigValue]:
    """Walk ``source_root`` and aggregate config values from every
    file the registered scanners recognize.

    ``exclude_dirs`` is a set of directory NAMES skipped at any depth
    (matching the existing ``discover()`` semantics).

    Per-file errors are caught and the file is skipped — one bad file
    doesn't abort the whole scan.
    """
    source_root = source_root.resolve()
    if not source_root.is_dir():
        return []

    values: list[ConfigValue] = []
    seen: set[Path] = set()

    def walk(directory: Path) -> None:
        if directory.name in exclude_dirs:
            return
        try:
            resolved = directory.resolve()
        except (OSError, RuntimeError):
            return
        if resolved in seen:
            return
        seen.add(resolved)

        try:
            entries = list(directory.iterdir())
        except (OSError, PermissionError):
            return

        for entry in entries:
            if entry.is_file():
                scanner = _scanner_for_file(entry)
                if scanner is not None:
                    try:
                        values.extend(scanner(entry))
                    except Exception:
                        # Skip the broken file; keep scanning.
                        continue
            elif entry.is_dir():
                walk(entry)

    walk(source_root)
    return values


__all__ = [
    'ConfigValue',
    'scan_config_sources',
    'scan_dotenv',
    'scan_hocon',
]
