"""HOCON parse tree → list[ElementInfo].

The extractor walks the Lark tree produced by `docgen.hocon_grammar.parse`
and emits one `ElementInfo` per key path:

  - `activation { ... }` produces an element at the block's full line range
    (the `key_path` text is `activation`, parent is the file's module qn).
  - `activation.pub = [...]` produces one element keyed on `activation.pub`,
    parent is `activation`, line range covers the assignment expression
    including any multi-line value (triple-quoted strings, arrays).
  - Dotted key paths like `a.b.c = 1` collapse to a single element with
    qualified_name `a.b.c` — same convention as JSON / YAML extractors.
  - `include "x.conf"` is recorded as a single element under the `include`
    pseudo-key (the runtime resolves it; the catalog just notes its
    presence).

Failures: parse errors return `[]`. The caller's `sync_file_catalog`
still creates a `file_index` doc, so the file remains discoverable in
search even when its content is malformed.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

from docgen.catalog_extractor import ElementInfo
from docgen.hocon_grammar import parse as _parse

if TYPE_CHECKING:
    from lark import Tree


def _module_qn(path: Path, source_root: Path) -> str:
    """Build the module-qualifier prefix from the file path. Mirrors
    `_py_module_qn` but works for any file: dots replace slashes; the
    extension is dropped."""
    try:
        rel = path.resolve().relative_to(source_root.resolve())
    except ValueError:
        rel = Path(path.name)
    parts = list(rel.with_suffix('').parts)
    return '.'.join(parts) if parts else path.stem


def _signature(text: str, max_len: int = 200) -> str:
    """Squash a (possibly multi-line) text fragment to a single line
    for catalog display. Matches the extractor convention used elsewhere."""
    first_line = text.splitlines()[0] if text else ''
    first_line = first_line.strip()
    if len(first_line) > max_len:
        first_line = first_line[: max_len - 1] + '…'
    return first_line


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8', errors='replace')).hexdigest()


def _key_path_text(key_path_node: Tree) -> str:
    """Reconstruct the dotted key path string from a `key_path` tree node.
    Handles unquoted KEY tokens (and ignores any whitespace tokens)."""
    parts: list[str] = []
    for child in key_path_node.children:
        # Each child is a Token (KEY) or '.' separator.
        token_text = str(child).strip()
        if token_text and token_text != '.':
            parts.append(token_text)
    return '.'.join(parts)


def _entry_text(entry_node: Tree, src_lines: list[str]) -> str:
    """Slice the source between the entry's start and end lines so we
    can compute body_sha / signature directly from the original text."""
    meta = entry_node.meta
    if meta.empty:
        return ''
    # Lark line numbers are 1-based; src_lines is 0-based.
    start = max(0, meta.line - 1)
    end = min(len(src_lines), meta.end_line)
    if start >= end:
        return src_lines[start] if start < len(src_lines) else ''
    return '\n'.join(src_lines[start:end])


def _emit_entry(
    entry_node: Tree,
    *,
    parent_qn: str,
    key_text: str,
    file: Path,
    src_lines: list[str],
) -> ElementInfo:
    """Build a single ElementInfo from an `entry` node with line metadata."""
    meta = entry_node.meta
    full_qn = f'{parent_qn}.{key_text}' if parent_qn else key_text
    body_text = _entry_text(entry_node, src_lines)
    return ElementInfo(
        language='hocon',
        subtype='hocon_key',
        file=str(file),
        qualified_name=full_qn,
        signature=_signature(body_text),
        line_start=meta.line if not meta.empty else 1,
        line_end=meta.end_line if not meta.empty else 1,
        col_start=meta.column if not meta.empty else 1,
        col_end=meta.end_column if not meta.empty else 1,
        parent_qualified_name=parent_qn or None,
        body_sha=_sha(body_text),
    )


def _walk_object_body(
    body_node: Tree,
    *,
    parent_qn: str,
    file: Path,
    src_lines: list[str],
    out: list[ElementInfo],
) -> None:
    """Walk an `object_body` node and emit ElementInfo for each entry.
    Recurses into nested object blocks."""
    for child in body_node.children:
        if not _is_tree(child):
            # Separators (NEWLINE / COMMA tokens). Skip.
            continue
        if child.data != 'entry':
            continue
        # An `entry` has exactly one child: include / assignment / object_block.
        inner = next(iter(child.children), None)
        if not _is_tree(inner):
            continue

        if inner.data == 'include':
            # `include "x.conf"` — synthesize a key path using the include
            # filename so each include has a unique qn.
            string_node = next(
                (c for c in inner.children if _is_tree(c) and c.data == 'string'),
                None,
            )
            if string_node is None:
                continue
            included = _string_value(string_node)
            key_text = f'include({included})'
            out.append(_emit_entry(
                child, parent_qn=parent_qn, key_text=key_text,
                file=file, src_lines=src_lines,
            ))
            continue

        if inner.data in ('assignment', 'object_block'):
            key_path_node = next(
                (c for c in inner.children if _is_tree(c) and c.data == 'key_path'),
                None,
            )
            if key_path_node is None:
                continue
            key_text = _key_path_text(key_path_node)
            element = _emit_entry(
                child, parent_qn=parent_qn, key_text=key_text,
                file=file, src_lines=src_lines,
            )
            out.append(element)

            if inner.data == 'object_block':
                # Nested object — recurse into its body.
                obj_node = next(
                    (c for c in inner.children if _is_tree(c) and c.data == 'object'),
                    None,
                )
                if obj_node is not None:
                    inner_body = next(
                        (c for c in obj_node.children
                         if _is_tree(c) and c.data == 'object_body'),
                        None,
                    )
                    if inner_body is not None:
                        _walk_object_body(
                            inner_body,
                            parent_qn=element.qualified_name,
                            file=file,
                            src_lines=src_lines,
                            out=out,
                        )


def _is_tree(node: object) -> bool:
    """Lark distinguishes Trees (rule matches) from Tokens (terminals).
    Only Trees have `.data` and `.children`."""
    return hasattr(node, 'data') and hasattr(node, 'children')


def _string_value(string_node: Tree) -> str:
    """Extract the literal text of a `string` node (the quoted content,
    quotes still included). Used only for the `include` directive."""
    if not string_node.children:
        return ''
    return str(string_node.children[0])


def _extract_hocon(
    src: str,
    path: Path,
    source_root: Path,
) -> list[ElementInfo]:
    """Parse HOCON source and emit catalog ElementInfo per key path.

    Returns an empty list on parse failure — `sync_file_catalog`
    still produces a `file_index` doc so the file is discoverable.
    """
    from lark.exceptions import UnexpectedInput

    try:
        tree = _parse(src)
    except UnexpectedInput:
        return []

    src_lines = src.splitlines()
    module_qn = _module_qn(path, source_root)
    elements: list[ElementInfo] = []

    # The top-level rule is `start -> object_body`, so `tree` IS the
    # object_body node.
    _walk_object_body(
        tree,
        parent_qn=module_qn,
        file=path,
        src_lines=src_lines,
        out=elements,
    )
    return elements


__all__ = ['_extract_hocon']
