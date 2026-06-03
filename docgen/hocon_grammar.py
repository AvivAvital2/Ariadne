"""HOCON grammar loader for Ariadne.

Loads the EBNF grammar from `docgen/hocon_grammar.lark` once at module
import time and exposes a `parse(src) -> lark.Tree` function. Every node
in the returned tree carries `meta.line` / `meta.column` / `meta.end_line`
/ `meta.end_column` thanks to `propagate_positions=True` — the load-bearing
property the rest of the catalog pipeline depends on for accurate
ElementInfo line ranges.

Failure mode: malformed HOCON raises `lark.exceptions.UnexpectedInput`.
The caller (`docgen.hocon_extractor._extract_hocon`) catches it and
returns `[]` so a single broken conf file never aborts catalog-sync.
"""
from __future__ import annotations

from pathlib import Path

from lark import Lark, Tree

_GRAMMAR_PATH = Path(__file__).parent / 'hocon_grammar.lark'

# Earley parses HOCON's gentle ambiguities (number-vs-unquoted-string,
# reserved-word-vs-key) more forgivingly than LALR. HOCON files are
# small enough that Earley's perf overhead is negligible.
_PARSER = Lark.open(
    str(_GRAMMAR_PATH),
    parser='earley',
    propagate_positions=True,
    maybe_placeholders=False,
)


def parse(src: str) -> Tree:
    """Parse a HOCON source string into a Lark tree.

    Raises:
        lark.exceptions.UnexpectedInput: when the input is not valid
            HOCON. Callers should catch this and degrade gracefully.
    """
    return _PARSER.parse(src)


__all__ = ['parse']
