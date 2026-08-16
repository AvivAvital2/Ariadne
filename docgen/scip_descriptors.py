"""SCIP symbol-string parsing for qualified-name extraction.

Pure-Python implementation; no protobuf dependency. Used by the SCIP
extractor to turn symbols like ::

    scip-java maven org.example my-lib 1.0 com/example/Foo#bar(I).

into qualified names like ``com.example.Foo.bar(int)`` (Java) or
``com.example.Foo.bar`` (Scala). The Scala branch keeps method names
untouched; the Java branch decodes the JVM signature disambiguator so
overloads round-trip to distinct qualified_names.
"""
from __future__ import annotations

import re
from typing import Literal

# Type-parameter markers (e.g. ``[T]``, ``[+T]``) are full descriptors but
# carry no semantic weight in the qualified name. We consume them and skip.
_TYPE_PARAM_RE = re.compile(r'\[[^\]]*\]')


# A descriptor is: name (optionally backtick-escaped) + optional disambiguator
# (parenthesized for methods) + suffix indicating the descriptor kind.
#   '/' = package, '#' = type, '.' = term/object/method, ':' = typealias.
_DESC_RE = re.compile(
    r"""
    (?:`(?P<bt>(?:``|[^`])+)`          # escaped identifier; doubled backticks are literal
        | (?P<plain>[^/.\#():\[\]`]+)  # plain identifier — no SCIP delimiters
    )
    (?:\((?P<disambig>[^)]*)\))?    # optional method disambiguator
    (?P<suffix>[/\#.:])             # descriptor kind suffix
    """,
    re.VERBOSE,
)


# Parameter descriptors: ``(name)`` following a method's ``().``. scip-python
# emits one SymbolInformation per parameter (including ``self``) — the
# parser must recognize this form, otherwise parameter symbols silently
# collapse to the same qualified_name as their enclosing method.
_PARAM_RE = re.compile(
    r"""
    \(
    (?:`(?P<bt>(?:``|[^`])+)`          # escaped parameter name
        | (?P<plain>[^)]+))          # plain parameter name
    \)
    """,
    re.VERBOSE,
)


_DescriptorKind = Literal[
    'package', 'type', 'term', 'method', 'typealias', 'parameter',
]


def _parse_descriptors(s: str) -> list[tuple[str, str, str]]:
    """Parse the descriptors substring into a list of ``(name, kind, disambig)``.

    ``kind`` is one of: ``package | type | term | method | typealias |
    parameter``. Type-parameter markers (``[T]``) are silently skipped;
    parameter descriptors (``(self)`` following a method's ``().``) are
    emitted as ``parameter`` entries so the parameter's qualified_name
    can be distinguished from its enclosing method's.
    """
    out: list[tuple[str, str, str]] = []
    pos = 0
    while pos < len(s):
        # Type-param markers are full descriptors carrying no QN content.
        tp = _TYPE_PARAM_RE.match(s, pos)
        if tp:
            pos = tp.end()
            continue

        # Parameter descriptors come AFTER the main loop's descriptor
        # match would otherwise bail (``(`` after ``)`` has no suffix
        # char). Handle them here so the parser can continue past them.
        pm = _PARAM_RE.match(s, pos)
        if pm:
            name = pm.group('bt').replace('``', '`') if pm.group('bt') is not None else pm.group('plain')
            out.append((name, 'parameter', ''))
            pos = pm.end()
            # SCIP wire format doesn't chain past a parameter in
            # practice, but be robust: skip a single trailing ``.``
            # separator so a subsequent descriptor can still match
            # without the parser bailing silently.
            if pos < len(s) and s[pos] == '.':
                pos += 1
            continue

        m = _DESC_RE.match(s, pos)
        if not m:
            # Unrecognized character — bail; partial input is preserved as
            # whatever we've accumulated so far.
            break

        name = m.group('bt').replace('``', '`') if m.group('bt') is not None else m.group('plain')
        sfx = m.group('suffix')
        disambig = m.group('disambig')

        if sfx == '/':
            kind: _DescriptorKind = 'package'
        elif sfx == '#':
            kind = 'type'
        elif sfx == ':':
            kind = 'typealias'
        elif sfx == '.':
            # Method if the optional `(disambig)` group fired (even when empty),
            # term otherwise. Distinguishing on group match (``is not None``)
            # rather than truthiness so that ``bar().`` is correctly classified
            # as a method even with empty disambig string.
            kind = 'method' if disambig is not None else 'term'
        else:
            pos = m.end()
            continue

        out.append((name, kind, disambig or ''))
        pos = m.end()
    return out


_PRIMITIVES = {
    'V': 'void', 'I': 'int', 'J': 'long', 'D': 'double', 'F': 'float',
    'Z': 'boolean', 'B': 'byte', 'S': 'short', 'C': 'char',
}


def _decode_java_descriptor(d: str) -> str:
    """Decode a JVM method-disambiguator string into ``(t1,t2,...)``.

    Handles primitives (``I`` → ``int``), boxed types (``Ljava/lang/String;``
    → ``java.lang.String``), arrays (``[I`` → ``int[]``, ``[[I`` →
    ``int[][]``), and multi-arg sequences. Unknown primitive letters round-
    trip as-is rather than crashing — defensive against future SCIP changes.
    """
    types: list[str] = []
    i, n = 0, len(d)
    while i < n:
        # Count leading ``[`` for array dimensions.
        arr = 0
        while i < n and d[i] == '[':
            arr += 1
            i += 1
        if i >= n:
            break

        c = d[i]
        if c == 'L':
            try:
                end = d.index(';', i)
            except ValueError:
                # Malformed object descriptor — preserve the rest as a literal
                # type so we don't crash a real-world index over a bad symbol.
                end = n
                t = d[i + 1:end].replace('/', '.')
                i = end
            else:
                t = d[i + 1:end].replace('/', '.')
                i = end + 1
        else:
            t = _PRIMITIVES.get(c, c)
            i += 1

        types.append(t + '[]' * arr)
    return '(' + ','.join(types) + ')'


_SIMPLE_IDENTIFIER_RE = re.compile(r"[_+$A-Za-z0-9-]+")


def _canonical_descriptor_spans(text: str) -> tuple[tuple[int, int, str], ...]:
    """Return exact spans for one complete SCIP descriptor chain.

    This parser is intentionally stricter than qualified-name rendering: plain
    identifiers follow SCIP's ASCII grammar, while escaped identifiers may contain
    spaces and doubled backticks. An incomplete parse returns no spans, allowing a
    caller to distinguish package metadata from the descriptor suffix.
    """
    spans = []
    position = 0
    while position < len(text):
        start = position
        if text[position] == "[":
            end = text.find("]", position + 1)
            if end < 0:
                return ()
            position = end + 1
            spans.append((start, position, "type-parameter"))
            continue
        if text[position] == "(":
            end = text.find(")", position + 1)
            if end < 0:
                return ()
            position = end + 1
            spans.append((start, position, "parameter"))
            continue
        if text[position] == "`":
            cursor = position + 1
            while cursor < len(text):
                if text[cursor] != "`":
                    cursor += 1
                    continue
                if cursor + 1 < len(text) and text[cursor + 1] == "`":
                    cursor += 2
                    continue
                break
            if cursor >= len(text):
                return ()
            name_end = cursor + 1
        else:
            match = _SIMPLE_IDENTIFIER_RE.match(text, position)
            if match is None:
                return ()
            name_end = match.end()
        if name_end < len(text) and text[name_end] == "(":
            close = text.find(")", name_end + 1)
            if close < 0 or close + 1 >= len(text) or text[close + 1] != ".":
                return ()
            position = close + 2
            kind = "method"
        elif name_end < len(text) and text[name_end] in "/#.:!":
            suffix = text[name_end]
            position = name_end + 1
            kind = {
                "/": "package", "#": "type", ".": "term",
                ":": "meta", "!": "macro",
            }[suffix]
        else:
            return ()
        spans.append((start, position, kind))
    return tuple(spans)


def _descriptor_suffix(symbol: str) -> tuple[int, tuple[tuple[int, int, str], ...]] | None:
    """Locate the canonical descriptor suffix without assuming package token count."""
    if symbol.startswith("local "):
        return None
    for offset in range(1, len(symbol)):
        if symbol[offset - 1] != " ":
            continue
        spans = _canonical_descriptor_spans(symbol[offset:])
        if spans:
            return offset, spans
    return None


def _symbol_descriptor_kind(symbol: str) -> str | None:
    """Return the final canonical descriptor kind, or ``None`` if malformed."""
    located = _descriptor_suffix(symbol)
    if located is None:
        return None
    return located[1][-1][2]


def _enclosing_symbol_from_symbol(symbol: str) -> str | None:
    """Return a global symbol's byte-exact canonical owner.

    SCIP requires global ownership to be encoded in the descriptor chain. Keeping the
    original prefix and descriptor bytes preserves distinctions such as ``Owner#``
    (type) versus ``Owner.`` (term/companion) that qualified names deliberately erase.
    Local symbols require explicit ``SymbolInformation.enclosing_symbol`` metadata.
    """
    located = _descriptor_suffix(symbol)
    if located is None:
        return None
    offset, spans = located
    if len(spans) < 2:
        return None
    return symbol[:offset + spans[-1][0]]
def _qualified_name_from_symbol(
    symbol: str, language: str,
) -> tuple[str, str | None]:
    """Turn a SCIP symbol into ``(qualified_name, parent_qualified_name)``.

    Ownership and display identity consume the same canonical descriptor suffix. This
    matters for escaped identifiers containing spaces and for package formats with a
    variable number of metadata tokens.
    """
    if symbol.startswith('local '):
        return symbol, None

    located = _descriptor_suffix(symbol)
    if located is None:
        return symbol, None
    descriptors = symbol[located[0]:]
    desc = _parse_descriptors(descriptors)
    if not desc:
        return symbol, None

    pkgs = [name for name, kind, _ in desc if kind == 'package']
    others = [item for item in desc if item[1] != 'package']
    pkg_qn = '.'.join(pkgs)

    if not others:
        return pkg_qn, None

    def render(name: str, kind: str, disambiguator: str) -> str:
        if kind == 'method' and language == 'java' and disambiguator:
            return f'{name}{_decode_java_descriptor(disambiguator)}'
        return name

    chain = [render(*item) for item in others]
    qualified = f"{pkg_qn}.{'.'.join(chain)}" if pkg_qn else '.'.join(chain)

    if len(others) > 1:
        parent_chain = [render(*item) for item in others[:-1]]
        parent = (
            f"{pkg_qn}.{'.'.join(parent_chain)}"
            if pkg_qn else '.'.join(parent_chain)
        )
    else:
        parent = pkg_qn or None

    return qualified, parent
def _parent_descriptor_kind(symbol: str) -> str | None:
    """Return the kind of the second-to-last canonical descriptor."""
    located = _descriptor_suffix(symbol)
    if located is None:
        return None
    descriptors = _parse_descriptors(symbol[located[0]:])
    if len(descriptors) < 2:
        return None
    return descriptors[-2][1]


__all__ = [
    '_decode_java_descriptor',
    '_enclosing_symbol_from_symbol',
    '_parent_descriptor_kind',
    '_parse_descriptors',
    '_qualified_name_from_symbol',
    '_symbol_descriptor_kind',
]
