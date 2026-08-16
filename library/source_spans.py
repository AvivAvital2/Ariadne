"""Bounded source spans: exact, hash-verified, association-justified.

Materialization's only unit used to be the indexed symbol extent, so a
gold-relevant region sitting just outside one — the doc comment above a
definition, its annotations, an associated file header — was structurally
unrepresentable. A SourceSpan carries such a region as byte-exact text
bound to the file's hash under a closed list of association reasons.
Bounds never truncate silently: an oversize block becomes a reported gap,
not a shortened span.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

_COMMENT_PREFIXES = ("#", "//", "/*", "*", "'''", '"""')


@dataclass(frozen=True)
class SourceSpan:
    source_name: str
    file: str
    line_start: int
    line_end: int
    text: str
    source_hash: str
    association_reason: str
    associated_symbol: str


@dataclass(frozen=True)
class SpanExtraction:
    spans: tuple = ()
    gaps: tuple = ()


def dedup_spans(spans) -> tuple:
    seen = set()
    ordered = []
    for span in spans:
        key = (span.file, span.line_start, span.line_end,
               span.association_reason)
        if key not in seen:
            seen.add(key)
            ordered.append(span)
    return tuple(ordered)


def definition_adjacent_spans(
        *, source_name: str, file: str, file_text: str, symbol: str,
        definition_line_start: int, max_blank_separation: int = 2,
        max_height: int = 40,
        include_file_header: bool = False) -> SpanExtraction:
    """Extract the structurally adjacent spans for one selected definition.

    Associations are a closed list: ``leading-documentation`` (the comment
    block contiguously above the definition, crossing only blank and
    annotation lines within the separation bound), ``annotation``
    (decorator/annotation lines attached to the definition), and — only on
    explicit request — ``file-header-documentation`` (the file's leading
    comment block up to the first non-comment line). Distant comments are
    excluded by contiguity, never included and trimmed.
    """
    lines = file_text.splitlines()
    source_hash = hashlib.sha256(file_text.encode()).hexdigest()
    spans = []
    gaps = []

    def make_span(reason: str, start: int, end: int) -> SourceSpan:
        return SourceSpan(
            source_name=source_name, file=file, line_start=start,
            line_end=end, text="\n".join(lines[start - 1:end]),
            source_hash=source_hash, association_reason=reason,
            associated_symbol=symbol)

    # Walk upward from the definition: annotations attach directly, then
    # up to max_blank_separation blank lines may separate the doc block.
    cursor = definition_line_start - 1
    annotation_end = 0
    annotation_start = 0
    while cursor >= 1 and _is_annotation(lines[cursor - 1]):
        annotation_start = cursor
        annotation_end = annotation_end or cursor
        cursor -= 1
    if annotation_end:
        spans.append(make_span(
            "annotation", annotation_start, annotation_end))

    blanks = 0
    while cursor >= 1 and not lines[cursor - 1].strip():
        blanks += 1
        cursor -= 1
    if blanks <= max_blank_separation:
        block_end = cursor
        while cursor >= 1 and _is_comment(lines[cursor - 1]):
            cursor -= 1
        block_start = cursor + 1
        if block_end >= block_start:
            height = block_end - block_start + 1
            if height > max_height:
                gaps.append({
                    "reason": "span-height-exceeded", "file": file,
                    "detail": f"leading documentation is {height} lines; "
                              f"bound is {max_height}"})
            else:
                spans.append(make_span(
                    "leading-documentation", block_start, block_end))

    if include_file_header:
        header_end = 0
        for index, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            if _is_comment(line):
                header_end = index
                continue
            break
        if header_end:
            header = make_span("file-header-documentation", 1, header_end)
            if all((header.line_start, header.line_end) !=
                   (span.line_start, span.line_end) for span in spans):
                spans.append(header)

    return SpanExtraction(spans=tuple(spans), gaps=tuple(gaps))


def _is_comment(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and (
        stripped.startswith(_COMMENT_PREFIXES) or stripped.endswith("*/"))


def _is_annotation(line: str) -> bool:
    return line.strip().startswith("@")
