"""Scaladoc/Javadoc parser → ``StructuredDoc`` (SCIP plan, Phase A.2).

Pure-Python parser for the shared ``@tag value`` grammar used by both
Scaladoc and Javadoc. The shape it produces (``StructuredDoc``) flows
through ``ElementInfo.documentation`` and onto LLM prompts as a single
JSON-serializable dict — so the downstream LLM has structured access to
``params`` / ``returns`` / ``throws`` / ``see_also`` instead of a raw
prose blob.

Differences between the two dialects:
- Scaladoc uses ``[[link]]`` for cross-references; Javadoc uses
  ``{@link target}`` inline blocks.
- Javadoc accepts ``@exception`` as an alias for ``@throws``.

Both dialects skip ``@tag`` lines that appear inside a ``{{{ ... }}}``
code fence (Scaladoc convention) so example code in docstrings doesn't
contaminate the structured output.
"""
from __future__ import annotations

import re

from attrs import field, frozen


@frozen
class StructuredDoc:
    """Parsed Scaladoc/Javadoc fields with sensible defaults.

    All fields are JSON-serializable (no datetime, no Path, etc.) so
    ``json.dumps(asdict(doc))`` works as the wire format for downstream
    consumers.
    """
    summary: str = ''
    body: str = ''
    params: dict[str, str] = field(factory=dict)
    returns: str | None = None
    throws: dict[str, str] = field(factory=dict)
    see_also: tuple[str, ...] = ()
    deprecated: str | None = None
    since: str | None = None


# A line that begins (after optional whitespace) with `@<word>` opens a
# tag section. The section ends at the next tag or end of input.
_TAG_RE = re.compile(r'^\s*@(\w+)\b(.*)$')

# Scaladoc code-fence delimiters; Javadoc uses HTML <pre> which we don't
# special-case. Inside a fence, lines beginning with `@` must NOT be
# parsed as tags.
_FENCE_OPEN = '{{{'
_FENCE_CLOSE = '}}}'


def _strip_comment_prefix(raw: str) -> str:
    """Remove the ``/**``, ``*/``, and per-line ``*`` markers from a
    raw doc comment so the inner lines are clean before tag parsing.
    """
    out: list[str] = []
    for line in raw.splitlines():
        stripped = line.lstrip()
        if stripped.startswith('/**'):
            stripped = stripped[3:].lstrip()
        elif stripped.startswith('*/'):
            stripped = stripped[2:].lstrip()
        elif stripped.startswith('*'):
            stripped = stripped[1:].lstrip()
        out.append(stripped)
    return '\n'.join(out)


def _split_sections(text: str) -> tuple[str, list[tuple[str, str]]]:
    """Return ``(body, [(tag, content), ...])``.

    ``body`` is the prose before the first tag; ``content`` is each tag's
    multi-line content joined with single spaces (so a description that
    wrapped onto continuation lines becomes a single string).
    """
    body_lines: list[str] = []
    tag_sections: list[tuple[str, list[str]]] = []
    current: list[str] | None = None
    in_fence = False

    for line in text.splitlines():
        # Toggle into-fence on open marker BEFORE matching tags, so a
        # ``@param`` on the same line as ``{{{`` is treated as fence content.
        if not in_fence and _FENCE_OPEN in line:
            in_fence = True
            (body_lines if current is None else current).append(line)
            if _FENCE_CLOSE in line[line.index(_FENCE_OPEN) + len(_FENCE_OPEN):]:
                in_fence = False
            continue
        if in_fence:
            (body_lines if current is None else current).append(line)
            if _FENCE_CLOSE in line:
                in_fence = False
            continue

        m = _TAG_RE.match(line)
        if m:
            tag = m.group(1)
            rest = m.group(2).strip()
            tag_sections.append((tag, [rest]))
            current = tag_sections[-1][1]
        else:
            (body_lines if current is None else current).append(line)

    body = '\n'.join(body_lines).strip()

    sections: list[tuple[str, str]] = []
    for tag, lines in tag_sections:
        parts = [ln.strip() for ln in lines if ln.strip()]
        sections.append((tag, ' '.join(parts)))
    return body, sections


def _first_sentence(text: str) -> str:
    """Return the first sentence of ``text`` (up to ``.``/``!``/``?`` +
    whitespace or EOL). Empty input yields an empty string.
    """
    text = text.strip()
    if not text:
        return ''
    m = re.match(r'^(.+?[.!?])(?:\s|$)', text, re.DOTALL)
    if m:
        return m.group(1).strip()
    return text


def _split_name_and_desc(content: str) -> tuple[str, str]:
    """For ``@param x desc`` and ``@throws Foo desc``, split into
    ``(name, desc)``. ``content`` is the post-tag string already joined.
    """
    parts = content.split(None, 1)
    if not parts:
        return '', ''
    if len(parts) == 1:
        return parts[0], ''
    return parts[0], parts[1].strip()


def _strip_brace_type(content: str) -> tuple[str | None, str]:
    """For JSDoc's ``{type}`` annotation: if ``content`` begins with a
    balanced ``{...}`` group, return ``(inner_type, rest)``; otherwise
    ``(None, content)``. Handles nested braces so ``{Record<string,T>}``
    style types are matched whole.
    """
    s = content.lstrip()
    if not s.startswith('{'):
        return None, content
    depth = 0
    for i, ch in enumerate(s):
        if ch == '{':
            depth += 1
        elif ch == '}':
            depth -= 1
            if depth == 0:
                return s[1:i].strip(), s[i + 1:].strip()
    # Unbalanced — leave content untouched.
    return None, content


def _parse(
    raw: str,
    *,
    exception_alias: bool,
    returns_alias: bool = False,
    brace_types: bool = False,
) -> StructuredDoc:
    text = _strip_comment_prefix(raw)
    body, sections = _split_sections(text)
    summary = _first_sentence(body)

    params: dict[str, str] = {}
    throws: dict[str, str] = {}
    see_also: list[str] = []
    returns: str | None = None
    deprecated: str | None = None
    since: str | None = None

    for tag, content in sections:
        # Javadoc-only alias.
        if exception_alias and tag == 'exception':
            tag = 'throws'
        # JSDoc spells it `@returns`.
        if returns_alias and tag == 'returns':
            tag = 'return'

        if tag == 'param':
            if brace_types:
                # `@param {type} name desc` — the type annotation is dropped
                # (params maps name→description).
                _, content = _strip_brace_type(content)
            name, desc = _split_name_and_desc(content)
            if name:
                params[name] = desc
        elif tag == 'throws':
            name = desc = ''
            if brace_types:
                # `@throws {ErrorType} desc` — the braced type IS the key.
                braced, rest = _strip_brace_type(content)
                if braced is not None:
                    name, desc = braced, rest
            if not name:
                name, desc = _split_name_and_desc(content)
            if name:
                throws[name] = desc
        elif tag == 'return':
            if brace_types:
                _, content = _strip_brace_type(content)
            returns = content.strip() or None
        elif tag == 'see':
            if content.strip():
                see_also.append(content.strip())
        elif tag == 'deprecated':
            deprecated = content.strip() or None
        elif tag == 'since':
            since = content.strip() or None
        # Unknown tags are silently dropped — they're rare and would only
        # add noise to the structured output.

    return StructuredDoc(
        summary=summary,
        body=body,
        params=params,
        returns=returns,
        throws=throws,
        see_also=tuple(see_also),
        deprecated=deprecated,
        since=since,
    )


def parse_scaladoc(raw: str) -> StructuredDoc:
    """Parse a Scaladoc comment body into a ``StructuredDoc``.

    ``raw`` may include or omit the surrounding ``/** ... */`` markers and
    per-line ``*`` prefixes — both are stripped.
    """
    return _parse(raw, exception_alias=False)


def parse_javadoc(raw: str) -> StructuredDoc:
    """Parse a Javadoc comment body into a ``StructuredDoc``.

    ``@exception`` is treated as an alias for ``@throws``. ``{@link target}``
    blocks are preserved verbatim in ``body``.
    """
    return _parse(raw, exception_alias=True)


def parse_jsdoc(raw: str) -> StructuredDoc:
    """Parse a JSDoc/TSDoc comment body into a ``StructuredDoc``.

    JSDoc shares the ``@tag value`` grammar but differs in two ways this
    handles: ``@returns`` is accepted as an alias for ``@return``, and
    ``@param`` / ``@returns`` / ``@throws`` may carry a leading ``{type}``
    annotation. For ``@param`` / ``@returns`` the type is dropped (the
    structured shape stores name→description); for ``@throws {ErrorType}``
    the braced type is used as the throws key.
    """
    return _parse(raw, exception_alias=False, returns_alias=True, brace_types=True)


__all__ = [
    'StructuredDoc',
    'parse_javadoc',
    'parse_jsdoc',
    'parse_scaladoc',
]
