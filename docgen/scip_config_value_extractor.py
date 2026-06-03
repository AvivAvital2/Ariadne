"""Phase 2q — config-value index extractor.

Walks ``source_root`` for HOCON / YAML / dotenv files, parses each
format, and persists flattened ``(key, value, file, line_start)``
rows to ``config_values``. Phase 2s consumes these rows when
resolving sink-site arguments that route through config getters
(``config.getString("resources.python")``).

Format coverage (v1):

- **HOCON** (``.conf``) — uses Ariadne's existing Lark grammar
  (``docgen.hocon_grammar``). Supports nested objects (flattened to
  dot-paths), dotted key paths, scalar/number/boolean values.
  Skipped: arrays, substitutions (``${...}``), include directives.
- **YAML** (``.yaml`` / ``.yml``) — uses ``yaml.compose`` to keep
  line metadata on each node. Recursive walk; scalars emit, lists
  skip.
- **dotenv** (``.env``) — line-by-line scan. Strips ``export``
  prefix, surrounding quotes, comments, blanks.

Re-ingest semantics: clears prior rows for ``source_name`` before
inserting. Other sources untouched.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sqlite3 import Connection


# Files to scan, in dispatch order. ``.env.*`` variants (e.g.
# ``.env.production``) handled via the explicit name check below.
_HOCON_SUFFIXES: tuple[str, ...] = ('.conf',)
_YAML_SUFFIXES: tuple[str, ...] = ('.yaml', '.yml')


def _is_dotenv(path: Path) -> bool:
    """``.env`` and ``.env.production`` / ``.env.local`` etc."""
    return path.name == '.env' or path.name.startswith('.env.')


def _is_tree(node: object) -> bool:
    """Lark Tree vs Token. Trees have ``data`` and ``children``."""
    return hasattr(node, 'data') and hasattr(node, 'children')


# ---------------------------------------------------------------------------
# HOCON
# ---------------------------------------------------------------------------


def _key_path_text(key_path_node) -> str:
    parts: list[str] = []
    for child in key_path_node.children:
        text = str(child).strip()
        if text and text != '.':
            parts.append(text)
    return '.'.join(parts)


def _strip_string_quotes(text: str) -> str:
    """Strip outer quotes from a HOCON STRING / TRIPLE_STRING token's
    text. The grammar's STRING terminal carries surrounding ``"`` —
    triple-quoted carries ``\"\"\"``."""
    if text.startswith('"""') and text.endswith('"""') and len(text) >= 6:
        return text[3:-3]
    if (
        len(text) >= 2
        and text[0] == '"'
        and text[-1] == '"'
    ):
        return text[1:-1]
    return text


def _hocon_scalar_text(scalar_node) -> str | None:
    """Extract the string representation of a HOCON ``scalar`` node.
    Returns None for substitutions (``${var}``) — those are Phase 2s
    territory, not literal config values."""
    for child in scalar_node.children:
        if _is_tree(child):
            if child.data == 'substitution':
                return None
            # Other tree children inside scalar are unusual; ignore.
            continue
        # Token — TRIPLE_STRING / STRING / NUMBER / BOOL / NULL /
        # UNQUOTED_VALUE.
        text = str(child)
        # ``str(child)`` on a Token gives the text. Strip quotes for
        # STRING/TRIPLE_STRING; pass through for everything else.
        return _strip_string_quotes(text)
    return None


def _hocon_value_text(value_node) -> str | None:
    """Extract scalar text from a HOCON ``value`` node. Returns None
    for non-scalar values (object, array) — caller recurses into
    objects separately."""
    for child in value_node.children:
        if not _is_tree(child):
            continue
        if child.data == 'scalar':
            return _hocon_scalar_text(child)
        if child.data == 'duration_value':
            # NUMBER + TIME_UNIT children — concatenate
            parts = [
                str(c) for c in child.children if not _is_tree(c)
            ]
            return ' '.join(parts) if parts else None
        if child.data == 'size_value':
            parts = [
                str(c) for c in child.children if not _is_tree(c)
            ]
            return ' '.join(parts) if parts else None
        if child.data == 'object':
            return None  # caller recurses
        if child.data == 'array':
            return None  # arrays skipped in v1
    return None


def _hocon_walk_object_body(
    body_node,
    *,
    prefix: str,
    out: list[tuple[str, str, int]],
) -> None:
    """Walk an ``object_body`` node, emitting (key, value, line)
    triples for scalar leaves. Recurses into nested objects with
    ``parent.child`` key paths."""
    for child in body_node.children:
        if not _is_tree(child) or child.data != 'entry':
            continue
        inner = next(
            (c for c in child.children if _is_tree(c)),
            None,
        )
        if inner is None:
            continue

        if inner.data == 'assignment':
            key_path_node = next(
                (
                    c for c in inner.children
                    if _is_tree(c) and c.data == 'key_path'
                ),
                None,
            )
            value_node = next(
                (
                    c for c in inner.children
                    if _is_tree(c) and c.data == 'value'
                ),
                None,
            )
            if key_path_node is None or value_node is None:
                continue
            key_text = _key_path_text(key_path_node)
            full_key = (
                f'{prefix}.{key_text}' if prefix else key_text
            )
            meta = inner.meta
            line = meta.line if not meta.empty else 1
            scalar = _hocon_value_text(value_node)
            if scalar is not None:
                out.append((full_key, scalar, line))
                continue
            # Object value — recurse
            obj_node = next(
                (
                    c for c in value_node.children
                    if _is_tree(c) and c.data == 'object'
                ),
                None,
            )
            if obj_node is not None:
                inner_body = next(
                    (
                        c for c in obj_node.children
                        if _is_tree(c) and c.data == 'object_body'
                    ),
                    None,
                )
                if inner_body is not None:
                    _hocon_walk_object_body(
                        inner_body, prefix=full_key, out=out,
                    )

        elif inner.data == 'object_block':
            key_path_node = next(
                (
                    c for c in inner.children
                    if _is_tree(c) and c.data == 'key_path'
                ),
                None,
            )
            if key_path_node is None:
                continue
            key_text = _key_path_text(key_path_node)
            full_key = (
                f'{prefix}.{key_text}' if prefix else key_text
            )
            obj_node = next(
                (
                    c for c in inner.children
                    if _is_tree(c) and c.data == 'object'
                ),
                None,
            )
            if obj_node is None:
                continue
            inner_body = next(
                (
                    c for c in obj_node.children
                    if _is_tree(c) and c.data == 'object_body'
                ),
                None,
            )
            if inner_body is not None:
                _hocon_walk_object_body(
                    inner_body, prefix=full_key, out=out,
                )

        # 'include' entries are skipped — they aren't key/value pairs.


def _extract_hocon(text: str) -> list[tuple[str, str, int]]:
    """Return list of (key, value, line_1idx) tuples. Returns empty
    list on parse failure."""
    try:
        from docgen.hocon_grammar import parse as _parse
        tree = _parse(text)
    except Exception:
        return []
    out: list[tuple[str, str, int]] = []
    if _is_tree(tree) and tree.data == 'object_body':
        _hocon_walk_object_body(tree, prefix='', out=out)
    return out


# ---------------------------------------------------------------------------
# YAML
# ---------------------------------------------------------------------------


def _extract_yaml(text: str) -> list[tuple[str, str, int]]:
    """Return list of (key, value, line_1idx) for scalar leaves of a
    YAML document. List values are skipped — sequence resolution is
    Phase 2s territory. Multi-document YAML is not common in config;
    we walk only the first document."""
    try:
        import yaml
        node = yaml.compose(text)
    except Exception:
        return []
    if node is None:
        return []
    out: list[tuple[str, str, int]] = []
    _yaml_walk_node(node, prefix=[], out=out)
    return out


def _yaml_walk_node(
    node,
    *,
    prefix: list[str],
    out: list[tuple[str, str, int]],
) -> None:
    import yaml

    if isinstance(node, yaml.MappingNode):
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode):
                continue
            key = str(key_node.value)
            new_prefix = prefix + [key]
            if isinstance(value_node, yaml.ScalarNode):
                line = value_node.start_mark.line + 1
                out.append((
                    '.'.join(new_prefix),
                    str(value_node.value),
                    line,
                ))
            elif isinstance(value_node, yaml.MappingNode):
                _yaml_walk_node(
                    value_node, prefix=new_prefix, out=out,
                )
            # SequenceNode → skip


# ---------------------------------------------------------------------------
# .env
# ---------------------------------------------------------------------------


def _extract_dotenv(text: str) -> list[tuple[str, str, int]]:
    """Line-by-line scan of dotenv format. Skips comments / blanks.
    Strips ``export`` prefix and surrounding quotes."""
    out: list[tuple[str, str, int]] = []
    for line_idx, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith('#'):
            continue
        if stripped.startswith('export '):
            stripped = stripped[len('export '):].lstrip()
        if '=' not in stripped:
            continue
        key, _, value = stripped.partition('=')
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if (
            len(value) >= 2
            and value[0] == value[-1]
            and value[0] in ('"', "'")
        ):
            value = value[1:-1]
        out.append((key, value, line_idx))
    return out


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def _walk_config_files(source_root: Path):
    """Yield Path objects for every config file under source_root.
    Order is sorted for deterministic ingest."""
    if not source_root.exists():
        return
    for path in sorted(source_root.rglob('*')):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if (
            suffix in _HOCON_SUFFIXES
            or suffix in _YAML_SUFFIXES
            or _is_dotenv(path)
        ):
            yield path


def _extract_for_path(path: Path) -> list[tuple[str, str, int]]:
    suffix = path.suffix.lower()
    try:
        text = path.read_text(encoding='utf-8', errors='replace')
    except OSError:
        return []
    if suffix in _HOCON_SUFFIXES:
        return _extract_hocon(text)
    if suffix in _YAML_SUFFIXES:
        return _extract_yaml(text)
    if _is_dotenv(path):
        return _extract_dotenv(text)
    return []


def ingest_config_values(
    *,
    source_name: str,
    source_root: Path,
    conn: 'Connection',
) -> int:
    """Walk ``source_root`` for config files, extract key/value
    pairs per format, persist to ``config_values``.

    Re-ingest semantics: clears prior rows for ``source_name`` before
    inserting. Returns the number of rows inserted.
    """
    conn.execute(
        'DELETE FROM config_values WHERE source_name = ?',
        (source_name,),
    )

    rows: list[tuple] = []
    for path in _walk_config_files(source_root):
        try:
            triples = _extract_for_path(path)
        except Exception:
            continue
        file_str = str(path.resolve())
        for key, value, line in triples:
            rows.append((source_name, file_str, key, value, line))

    if rows:
        # The schema's UNIQUE(source_name, file, key, line_start)
        # admits one row per (key, line). Two values at the same
        # exact line/key would be a parser bug, but use INSERT OR
        # IGNORE to be safe rather than crash.
        conn.executemany(
            '''INSERT OR IGNORE INTO config_values
               (source_name, file, key, value, line_start)
               VALUES (?, ?, ?, ?, ?)''',
            rows,
        )
    conn.commit()
    return len(rows)


__all__ = ['ingest_config_values']
