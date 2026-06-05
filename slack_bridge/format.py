"""Convert GitHub-flavored Markdown to Slack ``mrkdwn``.

The agent answers in Markdown, but Slack's ``text`` field is parsed as mrkdwn,
where bold is ``*one asterisk*`` (``**two**`` is NOT bold), links are
``<url|label>``, and ``#`` headings don't exist. ``to_mrkdwn`` rewrites the
constructs Slack mangles, while leaving code spans/blocks untouched (so literal
``**`` inside code survives) and leaving already-correct mrkdwn alone.

Deliberately conservative: it does not touch single-``*`` / single-``_`` emphasis
or bullet lists (Slack already renders ``- ``), so it can't mangle code-ish text.
"""
from __future__ import annotations

import re

# Code regions to shield from conversion: fenced ```...``` (multiline) or inline
# `...`. Captured so re.split keeps them; they land on odd indices.
_CODE = re.compile(r'(```.*?```|`[^`\n]+`)', re.DOTALL)

_LINK = re.compile(r'\[([^\]]+)\]\((\S+?)\)')        # [label](url) -> <url|label>
_BOLD_STAR = re.compile(r'\*\*(.+?)\*\*')            # **bold** -> *bold*
_BOLD_USCORE = re.compile(r'__(.+?)__')              # __bold__ -> *bold*
_HEADING = re.compile(r'(?m)^[ \t]{0,3}#{1,6}[ \t]+(.*?)[ \t]*#*[ \t]*$')  # # H -> *H*


def _convert(segment: str) -> str:
    segment = _LINK.sub(r'<\2|\1>', segment)
    segment = _BOLD_STAR.sub(r'*\1*', segment)
    segment = _BOLD_USCORE.sub(r'*\1*', segment)
    return _HEADING.sub(r'*\1*', segment)


def to_mrkdwn(text: str) -> str:
    """Return ``text`` with Markdown bold/links/headings rewritten as Slack
    mrkdwn. Code spans and fenced code blocks pass through verbatim.
    """
    parts = _CODE.split(text)
    # Even indices are non-code (convert); odd indices are code (leave as-is).
    return ''.join(
        part if i % 2 else _convert(part)
        for i, part in enumerate(parts)
    )
