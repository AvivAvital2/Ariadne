"""Single source of truth for how Ariadne fences and extracts Graphviz DOT diagrams.

A ``diagram`` doc embeds its DOT inside a ```dot fenced block. The producer
(``writer.add_diagram``) and the consumers (the validator, the generation
orchestrator, and the Slack-bridge renderer) all route through here, so the fence
grammar lives in exactly one place. Pure-text, no heavy deps, so any layer can
import it.
"""
from __future__ import annotations

import re

# A ```dot (or ```graphviz) fenced block; group 1 is the DOT source.
DOT_BLOCK_RE = re.compile(r'```(?:dot|graphviz)\n(.*?)```', re.DOTALL)


def fence_dot(dot_code: str) -> str:
    """Wrap DOT source in a ```dot fenced block — the canonical stored form."""
    return f'```dot\n{dot_code}\n```'


def extract_dot_blocks(text: str) -> list[str]:
    """The DOT source of every fenced block in ``text`` (stripped), in order."""
    return [m.group(1).strip() for m in DOT_BLOCK_RE.finditer(text or '')]


def split_description_and_dot(content: str) -> tuple[str, str]:
    """Split a diagram doc into ``(description, first_dot_block)``.

    The description is the content with all DOT blocks removed; the DOT is the
    first block's source (empty string if none). Inverse of ``fence_dot``.
    """
    blocks = extract_dot_blocks(content)
    description = DOT_BLOCK_RE.sub('', content or '').strip()
    return description, (blocks[0] if blocks else '')
