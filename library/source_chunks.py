"""Deterministic exact-source chunks for narration placeholders.

SCIP decides what code is reachable; materialized bodies say what the code
literally says. These chunks isolate the causal statements inside selected
definition bodies so the narration model references ``{{X#}}`` instead of
retyping source, and expansion re-attaches the exact bytes with their
coordinates afterwards. Copying fidelity stops being the model's job.

Isolation is lexical and language-agnostic: a compiler-seeded line grows to
its complete statement by bracket balance and continuation syntax, pulls in
the local assignments feeding it, and keeps the branch predicates enclosing
it. When a statement cannot be isolated safely the complete body is retained
and the chunk says why — source proof is never silently discarded.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SourceChunk:
    """One exact, coordinate-anchored span of a selected definition body."""

    id: str
    source_name: str
    file: str
    line_start: int
    line_end: int
    #: ``((absolute line number, exact text), ...)`` — never re-rendered.
    lines: tuple
    #: Hash of the complete file the body was materialized from.
    sha256: str
    reason: str = "causal"


class _Unisolatable(Exception):
    """A causal statement could not be bounded safely inside the body."""


_STRINGS = re.compile(r"'(?:\\.|[^'\\])*'|\"(?:\\.|[^\"\\])*\"")
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_KEYWORDS = frozenset((
    "val", "var", "def", "lazy", "let", "const", "final", "new", "if",
    "else", "elif", "match", "case", "while", "for", "try", "catch",
    "except", "finally", "return", "yield", "null", "true", "false",
    "None", "True", "False", "this", "super", "import", "package",
    "private", "protected", "override", "implicit", "not", "and", "or",
    "is", "in", "async", "await", "with", "as", "assert", "raise",
    "throw", "then", "do",
))
_CONTINUATION_ENDINGS = (",", "=", "=>", "->", "&&", "||", "+", "\\")
_CONTINUATION_STARTS = (".",)
_BRANCH_KEYWORDS = (
    "if", "else", "elif", "match", "case", "while", "for", "try",
    "catch", "except", "finally",
)


def _code_text(line: str) -> str:
    """The line with string literals and trailing line comments removed."""
    without_strings = _STRINGS.sub("", line)
    for marker in ("//", "#"):
        index = without_strings.find(marker)
        if index != -1:
            without_strings = without_strings[:index]
    return without_strings


def _balance(line: str) -> int:
    code = _code_text(line)
    return (sum(code.count(opener) for opener in "([{")
            - sum(code.count(closer) for closer in ")]}"))


def _continues_down(line: str) -> bool:
    code = _code_text(line).rstrip()
    return any(code.endswith(ending) for ending in _CONTINUATION_ENDINGS)


def _continues_up(line: str) -> bool:
    code = _code_text(line).lstrip()
    return any(code.startswith(start) for start in _CONTINUATION_STARTS)


def _statement_span(lines, index: int, *, max_span: int) -> tuple[int, int]:
    """The complete lexical statement around ``index``, or refusal to guess."""
    start = end = index
    balance = _balance(lines[index])
    while True:
        grew = False
        while (start > 0
               and (balance < 0 or _continues_up(lines[start])
                    or _continues_down(lines[start - 1]))):
            start -= 1
            balance += _balance(lines[start])
            grew = True
        while (balance > 0
               or (end + 1 < len(lines) and _continues_up(lines[end + 1]))
               or _continues_down(lines[end])):
            if end + 1 >= len(lines):
                raise _Unisolatable(
                    f"statement at body line {index} never closes")
            end += 1
            balance += _balance(lines[end])
            grew = True
        if end - start + 1 > max_span:
            raise _Unisolatable(
                f"statement at body line {index} exceeds {max_span} lines")
        if not grew:
            return start, end


def _identifiers(line: str) -> set[str]:
    return {token for token in _IDENTIFIER.findall(_code_text(line))
            if token not in _KEYWORDS}


def _assignment_indices(lines, names) -> set[int]:
    found = set()
    for name in sorted(names):
        pattern = re.compile(
            rf"^\s*(?:(?:lazy\s+)?(?:val|var|let|const|final)\s+)?"
            rf"{re.escape(name)}\s*(?::[^=]*)?=(?![=>])"
            rf"|^\s*def\s+{re.escape(name)}\b")
        for index, line in enumerate(lines):
            if pattern.match(_code_text(line)):
                found.add(index)
    return found


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _enclosing_predicates(lines, index: int) -> set[int]:
    """Branch headers lexically enclosing ``index``, by indentation ladder."""
    found = set()
    ceiling = _indent(lines[index])
    for above in range(index - 1, 0, -1):
        line = lines[above]
        if not line.strip():
            continue
        indent = _indent(line)
        if indent >= ceiling:
            continue
        head = _code_text(line).lstrip().lstrip("} ").rstrip()
        first = head.split("(")[0].split(" ")[0] if head else ""
        if first in _BRANCH_KEYWORDS:
            found.add(above)
        ceiling = indent
        if indent == 0:
            break
    return found
def _is_comment_line(line: str) -> bool:
    stripped = line.strip()
    return bool(stripped) and stripped.startswith(
        ("//", "/*", "*", "#", '"""', "'''"))
def _body_chunk_spans(lines, seed_indices, *, max_statement_span):
    """Local kept indices and their reasons; raises when isolation is unsafe."""
    kept: dict[int, set[str]] = {}

    def keep(span, reason):
        for position in range(span[0], span[1] + 1):
            kept.setdefault(position, set()).add(reason)

    # The signature is one statement: a multiline parameter list and the
    # supertype clauses belong to it, through the line that opens the body.
    signature_end = 0
    for index in range(min(len(lines), 12)):
        code = _code_text(lines[index]).rstrip()
        if code.endswith(("{", "=", ":")):
            signature_end = index
            break
    for index in range(signature_end + 1):
        kept.setdefault(index, set()).add("signature")

    for index in sorted(seed_indices):
        keep(_statement_span(lines, index, max_span=max_statement_span),
             "call_site")
    identifiers = set()
    for index in sorted(kept):
        if kept[index] == {"signature"}:
            continue
        identifiers.update(_identifiers(lines[index]))
    for _ in range(3):
        new_indices = {
            index for index in _assignment_indices(lines, identifiers)
            if index not in kept}
        if not new_indices:
            break
        before = set(identifiers)
        for index in sorted(new_indices):
            span = _statement_span(lines, index, max_span=max_statement_span)
            keep(span, "assignment")
            for position in range(span[0], span[1] + 1):
                identifiers.update(_identifiers(lines[position]))
        if identifiers == before:
            break
    for index, line in enumerate(lines):
        if index in kept:
            continue
        code = _code_text(line).lstrip()
        if (code.startswith("return") and code != "return"
                and identifiers.intersection(_identifiers(line))):
            try:
                keep(_statement_span(
                    lines, index, max_span=max_statement_span), "return")
            except _Unisolatable:
                pass
    # Expression languages return the body's final statement without a
    # ``return`` keyword; when it consumes a causal identifier it is the
    # chain's product and must ride along.
    for index in range(len(lines) - 1, -1, -1):
        code = _code_text(lines[index]).strip()
        if not code or not code.strip(")]};"):
            continue
        if (index not in kept
                and identifiers.intersection(_identifiers(lines[index]))):
            try:
                keep(_statement_span(
                    lines, index, max_span=max_statement_span), "result")
            except _Unisolatable:
                pass
        break
    # A seeded call that assigns a value proves little until the value is
    # used: the first statement consuming that name rides along.
    assignment_head = re.compile(
        r"^\s*(?:(?:lazy\s+)?(?:val|var|let|const|final)\s+)?"
        r"([A-Za-z_]\w*)\s*(?::[^=]*)?=(?![=>])")
    for index in list(sorted(kept)):
        if "call_site" not in kept[index]:
            continue
        head = assignment_head.match(_code_text(lines[index]))
        if not head or head.group(1) in _KEYWORDS:
            continue
        name = head.group(1)
        for later in range(index + 1, len(lines)):
            if name not in _identifiers(lines[later]):
                continue
            if later not in kept:
                try:
                    keep(_statement_span(
                        lines, later, max_span=max_statement_span),
                         "consumer")
                except _Unisolatable:
                    pass
            break
    for index in sorted(seed_indices):
        for predicate in _enclosing_predicates(lines, index):
            kept.setdefault(predicate, set()).add("predicate")
    # The comment explaining a causal statement is part of its meaning: a
    # contiguous comment block directly above a kept line rides along.
    for index in sorted(kept):
        above = index - 1
        while (above >= 0 and index - above <= 30
               and _is_comment_line(lines[above])):
            kept.setdefault(above, set()).add("comment")
            above -= 1
    return kept
def derive_source_chunks(bodies, seeds, *, sites=(),
                         max_statement_span: int = 40,
                         short_body_lines: int = 48):
    """Compact causal chunks of the selected bodies, seeded by compiler sites.

    ``bodies`` are materialized definition excerpts; ``seeds`` are exact
    ``(file, line)`` coordinates the index recorded — call sites and body
    edges. A selected body containing no seed is still selected proof: a short
    one stays complete, a long one contributes its signature. ``sites`` are
    call-site and body-edge excerpts; one outside every body is already exact,
    hash-verified proof of a transition and becomes a chunk of its own. Chunk IDs are
    stable across identical inputs: bodies and spans are sorted, duplicates
    collapse, and no randomness or time enters anywhere.
    """
    seed_coordinates = {(str(file), int(line)) for file, line in seeds}
    unique_bodies = {}
    for body in bodies:
        key = (body.source_name, body.file, body.line_start,
               body.line_end, body.sha256)
        unique_bodies.setdefault(key, body)
    chunks = {}
    for key in sorted(unique_bodies):
        body = unique_bodies[key]
        lines = body.content.splitlines()
        if not lines:
            continue
        interior = {
            line - body.line_start
            for file, line in seed_coordinates
            if file == body.file
            and body.line_start <= line <= body.line_end}
        if interior:
            try:
                kept = _body_chunk_spans(
                    lines, interior, max_statement_span=max_statement_span)
            except _Unisolatable:
                kept = {index: {"unisolatable"} for index in range(len(lines))}
        elif len(lines) <= max(short_body_lines, 0):
            kept = {index: {"short_body"} for index in range(len(lines))}
        else:
            kept = {0: {"signature"}}
        ordered = sorted(kept)
        runs = []
        start = finish = ordered[0]
        for index in ordered[1:]:
            if index == finish + 1:
                finish = index
            else:
                runs.append((start, finish))
                start = finish = index
        runs.append((start, finish))
        for start, finish in runs:
            line_start = body.line_start + start
            line_end = body.line_start + finish
            chunk_key = (body.source_name, body.file, line_start, line_end,
                         body.sha256)
            if chunk_key in chunks:
                continue
            reasons = sorted(
                {reason for index in range(start, finish + 1)
                 for reason in kept.get(index, ())})
            chunks[chunk_key] = SourceChunk(
                id="", source_name=body.source_name, file=body.file,
                line_start=line_start, line_end=line_end,
                lines=tuple(
                    (body.line_start + index, lines[index])
                    for index in range(start, finish + 1)),
                sha256=body.sha256, reason=",".join(reasons))
    body_extents = tuple(
        (source_name, file, line_start, line_end)
        for source_name, file, line_start, line_end, _ in unique_bodies)
    for excerpt in sites:
        if excerpt.kind not in ("call_site", "body_edge", "doc_header"):
            continue
        if any(excerpt.source_name == source_name and excerpt.file == file
               and line_start <= excerpt.line_start
               and excerpt.line_end <= line_end
               for source_name, file, line_start, line_end in body_extents):
            continue
        lines = excerpt.content.splitlines()
        if not lines:
            continue
        chunk_key = (excerpt.source_name, excerpt.file, excerpt.line_start,
                     excerpt.line_end, excerpt.sha256)
        if chunk_key in chunks:
            continue
        chunks[chunk_key] = SourceChunk(
            id="", source_name=excerpt.source_name, file=excerpt.file,
            line_start=excerpt.line_start, line_end=excerpt.line_end,
            lines=tuple(
                (excerpt.line_start + index, text)
                for index, text in enumerate(lines)),
            sha256=excerpt.sha256, reason=excerpt.kind)
    ordered = sorted(chunks)
    return tuple(
        SourceChunk(
            id=f"X{position}", source_name=chunk.source_name,
            file=chunk.file, line_start=chunk.line_start,
            line_end=chunk.line_end, lines=chunk.lines,
            sha256=chunk.sha256, reason=chunk.reason)
        for position, chunk in enumerate(
            (chunks[key] for key in ordered), start=1))


def render_source_ledger(chunks) -> str:
    """The compact prompt section a narration references chunks from."""
    lines = [
        "SOURCE CHUNKS — exact code with coordinates. Reference a chunk as "
        "{{X#}} on its own line; never retype or paraphrase code."]
    for chunk in chunks:
        lines.append(
            f"  {{{{{chunk.id}}}}}: {chunk.file}:{chunk.line_start}-"
            f"{chunk.line_end} [{chunk.reason}]")
        for number, text in chunk.lines:
            lines.append(f"    {number} | {text}")
    return "\n".join(lines)


_SOURCE_PLACEHOLDER = re.compile(r"\{\{(X\d+)\}\}|(?<![\w{])(X\d+)(?![\w}])")


def source_chunk_values(chunks) -> dict:
    """Placeholder id -> the exact expansion block, one coordinate per line."""
    return {
        chunk.id: "\n".join(
            f"{chunk.file}:{number} `{text}`" for number, text in chunk.lines)
        for chunk in chunks}


def expand_source_placeholders(answer: str, chunks, *, strict: bool = True) -> str:
    """Expand known ``{{X#}}`` ids; unknown ids are never guessed at."""
    values = source_chunk_values(chunks)
    matches = list(_SOURCE_PLACEHOLDER.finditer(answer or ""))
    ids = {match.group(1) or match.group(2) for match in matches}
    unknown = sorted(ids - set(values))
    if unknown and strict:
        raise ValueError(
            "unknown evidence placeholder(s): " + ", ".join(unknown))
    return _SOURCE_PLACEHOLDER.sub(
        lambda match: values.get(
            match.group(1) or match.group(2),
            f"[unsupported evidence {match.group(1) or match.group(2)}]"),
        answer or "")
