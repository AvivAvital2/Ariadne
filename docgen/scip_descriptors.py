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
    r'''
    (?:`(?P<bt>[^`]+)`              # backtick-escaped identifier (handles `<init>`, etc.)
        | (?P<plain>[^/.\#():\[\]`]+)  # plain identifier — no SCIP delimiters
    )
    (?:\((?P<disambig>[^)]*)\))?    # optional method disambiguator (JVM signature on Java)
    (?P<suffix>[/\#.:])             # one of the four kind suffixes
    ''',
    re.VERBOSE,
)


# Parameter descriptors: ``(name)`` following a method's ``().``. scip-python
# emits one SymbolInformation per parameter (including ``self``) — the
# parser must recognize this form, otherwise parameter symbols silently
# collapse to the same qualified_name as their enclosing method.
_PARAM_RE = re.compile(
    r'''
    \(
    (?:`(?P<bt>[^`]+)`              # backtick-escaped param name
        | (?P<plain>[^)]+))          # plain param name
    \)
    ''',
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
            name = pm.group('bt') or pm.group('plain')
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

        name = m.group('bt') or m.group('plain')
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


def _qualified_name_from_symbol(
    symbol: str, language: str,
) -> tuple[str, str | None]:
    """Turn a SCIP symbol into ``(qualified_name, parent_qualified_name)``.

    SCIP symbols look like ``scip-java maven <group> <artifact> <version> <descriptors>``
    — six space-separated tokens for Maven artifacts. For ``local <id>``
    symbols (intra-document references) we just return the raw symbol with
    no parent.

    The parent is the symbol's enclosing scope: for a method, the enclosing
    class; for a class, the package; for a top-level package, ``None``.
    """
    if symbol.startswith('local '):
        return symbol, None

    # The descriptors are always the last whitespace-separated token of a
    # well-formed SCIP symbol, regardless of how many "package" tokens
    # precede it. Falling back to the raw symbol on malformed input keeps
    # the caller's logging useful.
    parts = symbol.split(' ')
    descriptors = parts[-1] if parts else ''
    if not descriptors:
        return symbol, None

    desc = _parse_descriptors(descriptors)
    if not desc:
        return symbol, None

    pkgs = [n for n, k, _ in desc if k == 'package']
    others = [(n, k, d) for n, k, d in desc if k != 'package']
    pkg_qn = '.'.join(pkgs)

    if not others:
        # Symbol IS a package — qn is the package, no parent.
        return pkg_qn, None

    def render(n: str, k: str, d: str) -> str:
        if k == 'method' and language == 'java' and d:
            return f'{n}{_decode_java_descriptor(d)}'
        return n

    chain = [render(*o) for o in others]
    qn = f"{pkg_qn}.{'.'.join(chain)}" if pkg_qn else '.'.join(chain)

    if len(others) > 1:
        parent_chain = [render(*o) for o in others[:-1]]
        parent = (
            f"{pkg_qn}.{'.'.join(parent_chain)}"
            if pkg_qn else '.'.join(parent_chain)
        )
    else:
        parent = pkg_qn or None

    return qn, parent


def _parent_descriptor_kind(symbol: str) -> str | None:
    """Return the kind of the second-to-last descriptor in the chain.

    Used for Python/JavaScript subtype dispatch: a SCIP ``Method`` symbol
    is a class method if the descriptor immediately before it is a type
    (``#``); otherwise it's a top-level function.

    Examples:
      - ``... licensing/LicenseService#validate_token().`` → ``'type'``
        (parent is the class)
      - ``... utils/compute_score().`` → ``'package'``
        (parent is the package; this is a top-level function)
      - ``local 1`` → None (intra-document refs have no chain)
    """
    if symbol.startswith('local '):
        return None
    parts = symbol.split(' ')
    descriptors = parts[-1] if parts else ''
    if not descriptors:
        return None
    desc = _parse_descriptors(descriptors)
    if len(desc) < 2:
        return None
    _, parent_kind, _ = desc[-2]
    return parent_kind


__all__ = [
    '_decode_java_descriptor',
    '_parent_descriptor_kind',
    '_parse_descriptors',
    '_qualified_name_from_symbol',
]
