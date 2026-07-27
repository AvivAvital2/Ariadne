"""Shared segment/word tokenizer.

Routing (lens_router) and suggestions (catalog_lookup) must split words
identically — a term that suggests one way and routes another is a bug, so
the tokenizer has exactly one home.
"""
import re

_CAMEL_WORD_RE = re.compile(r'[A-Z]+(?![a-z])|[A-Z][a-z0-9]*|[a-z0-9]+')


def segment_word_tokens(segment: str) -> frozenset:
    """Lowercased word tokens of one dotted-path segment or phrase — splits
    snake_case, camelCase, PascalCase, SCREAMING_CASE, and spaced words
    alike."""
    return frozenset(
        m.group(0).lower()
        for part in segment.split('_')
        for m in _CAMEL_WORD_RE.finditer(part)
    )
