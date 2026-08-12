"""Deliver completed agent replies to Slack, including requested Markdown files."""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from slack_bridge.diagram import PreparedReply, prepare_diagrams
from slack_bridge.format import to_mrkdwn

_logger = logging.getLogger(__name__)

# Used only if an explicitly requested file upload fails. It is not an automatic
# attachment threshold: answers remain inline at every length unless requested.
FALLBACK_CHUNK_MAX_CHARS = 12_000
_ATTACHMENT_FILENAME = 'ariadne-answer.md'
_ATTACHMENT_NOTICE = f'📎 I attached the complete answer as `{_ATTACHMENT_FILENAME}`.'

_SUBJECT = r'(?:answers?|responses?|plans?|prompts?|instructions?|outputs?|results?|this|it)'
_FORMAT = (
    r'(?:markdown(?:\s+(?:file|document|attachment))?'
    r'|[.]md(?:\s+file)?|(?:an?\s+)?file|file\s+attachment'
    r'|attached\s+(?:markdown|[.]md|file)(?:\s+file)?)'
)
_ACTION = r'(?:attach|upload|export|send|provide|return|deliver|give|put|place|create)'
_NEGATED_ATTACHMENT = re.compile(
    r"\b(?:do\s+not|don't|dont|no\s+need\s+to|without)\b"
    r'.{0,60}(?:attach|attachment|markdown|[.]md|file)\b',
    re.IGNORECASE | re.DOTALL,
)
_EXPLICIT_ATTACHMENT_PATTERNS = (
    re.compile(r'--attach-markdown\b', re.IGNORECASE),
    re.compile(
        rf'\b{_ACTION}\b.{{0,80}}\b{_SUBJECT}\b.{{0,80}}{_FORMAT}\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf'\b{_ACTION}\b.{{0,80}}{_FORMAT}\b.{{0,80}}\b{_SUBJECT}\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf'\b{_SUBJECT}\b.{{0,80}}\b(?:as|in|into)\b.{{0,30}}{_FORMAT}\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf'\b(?:respond|reply|output)\b.{{0,40}}\b(?:with|as|in)\b.{{0,20}}{_FORMAT}\b',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf'^\s*(?:please\s+)?(?:attach|upload|provide|return|send)\b.{{0,40}}{_FORMAT}\b'
        r'(?:\s*,?\s*please)?[.!]?\s*$',
        re.IGNORECASE | re.DOTALL,
    ),
    re.compile(
        rf'^\s*(?:an?\s+)?{_FORMAT}\s*,?\s*please[.!]?\s*$',
        re.IGNORECASE | re.DOTALL,
    ),
)

_MARKDOWN_FENCE_OPEN = re.compile(
    r'(?m)^[ \t]*(?P<fence>`{3,}|~{3,})[ \t]*(?:markdown|md)[ \t]*\r?\n',
    re.IGNORECASE,
)


@dataclass(frozen=True)
class MarkdownAttachment:
    """The file body and Slack-facing prose split from one agent reply."""

    content: str
    message: str


def markdown_attachment_requested(request_text: str) -> bool:
    """Return whether the user explicitly requested this answer as Markdown.

    False positives are more disruptive than false negatives, so ordinary mentions
    of files or Markdown do not count. The request must use an output action, an
    explicit ``--attach-markdown`` flag, or one of the terse request forms above.
    Explicit negation always wins.
    """
    text = request_text or ''
    if _NEGATED_ATTACHMENT.search(text):
        return False
    return any(pattern.search(text) for pattern in _EXPLICIT_ATTACHMENT_PATTERNS)


def partition_markdown_attachment(answer: str) -> MarkdownAttachment:
    """Separate one explicit Markdown document envelope from reply prose.

    Agents commonly put a requested document inside a four-backtick
    ``markdown`` fence so that the document can itself contain ordinary triple-
    backtick code blocks. That outer fence is a transport envelope, not part of
    the requested file. When exactly one well-formed envelope exists, upload its
    body and retain the surrounding text as the Slack-facing message. A clean
    answer with no envelope remains byte-for-byte intact apart from the delivery
    path's existing outer-whitespace trim.

    Multiple envelopes are deliberately left untouched: choosing one would lose
    content, while uploading the original answer is safe and reversible.
    """
    text = (answer or '').strip()
    candidates: list[tuple[int, int, str]] = []
    cursor = 0
    while opening := _MARKDOWN_FENCE_OPEN.search(text, cursor):
        fence = opening.group('fence')
        marker = re.escape(fence[0])
        # Some agents join their Slack-facing follow-up directly to the closing
        # fence (````The file is attached``). The fence still unambiguously ends
        # the outer envelope: consume only its marker so the joined prose remains
        # outside and is delivered in Slack, never inside the uploaded document.
        closing = re.compile(
            rf'(?m)^[ \t]*{marker}{{{len(fence)},}}'
        ).search(text, opening.end())
        if closing is None:
            break
        candidates.append((opening.start(), closing.end(), text[opening.end():closing.start()]))
        cursor = closing.end()

    if len(candidates) != 1:
        return MarkdownAttachment(content=text, message='')

    start, end, body = candidates[0]
    outside = [part.strip() for part in (text[:start], text[end:]) if part.strip()]
    content = body.strip()
    if not content:
        return MarkdownAttachment(content=text, message='')
    return MarkdownAttachment(content=content, message='\n\n'.join(outside))


def split_slack_message(
    text: str,
    *,
    max_chars: int = FALLBACK_CHUNK_MAX_CHARS,
) -> list[str]:
    """Split text losslessly, preferring paragraph and line boundaries."""
    if max_chars <= 0:
        raise ValueError('max_chars must be positive')
    if not text:
        return ['']

    chunks: list[str] = []
    start = 0
    while len(text) - start > max_chars:
        hard_end = start + max_chars
        split_at = text.rfind('\n\n', start, hard_end)
        separator_width = 2
        if split_at <= start:
            split_at = text.rfind('\n', start, hard_end)
            separator_width = 1
        if split_at <= start:
            split_at = hard_end
            separator_width = 0
        end = split_at + separator_width
        chunks.append(text[start:end])
        start = end
    chunks.append(text[start:])
    return chunks


async def deliver_reply(
    *,
    slack: Any,
    channel: str,
    thread_ts: str,
    placeholder_ts: str,
    answer: str,
    attach_markdown: bool,
    update_message: Callable[..., Awaitable[None]],
) -> PreparedReply:
    """Render diagrams and deliver inline unless the user requested Markdown.

    File upload is intentionally attempted only once because it is not
    idempotent. If a requested upload fails, the complete rendered response is
    posted in lossless bounded thread chunks instead of leaving the placeholder.
    """
    prepared = await asyncio.to_thread(prepare_diagrams, answer)
    rendered = to_mrkdwn(prepared.text)

    if not attach_markdown:
        await update_message(
            slack,
            channel=channel,
            ts=placeholder_ts,
            text=rendered,
        )
    else:
        attachment = partition_markdown_attachment(answer)
        try:
            await slack.files_upload_v2(
                channel=channel,
                thread_ts=thread_ts,
                content=attachment.content,
                filename=_ATTACHMENT_FILENAME,
                title='Ariadne full answer',
            )
        except Exception:
            _logger.exception(
                'Requested Markdown upload failed; falling back to thread chunks '
                '(channel=%s thread=%s)',
                channel,
                thread_ts,
            )
            chunks = split_slack_message(rendered)
            await update_message(
                slack,
                channel=channel,
                ts=placeholder_ts,
                text=chunks[0],
            )
            for chunk in chunks[1:]:
                await slack.chat_postMessage(
                    channel=channel,
                    thread_ts=thread_ts,
                    text=chunk,
                )
        else:
            message = to_mrkdwn(attachment.message) if attachment.message else ''
            if message:
                message = f'{message}\n\n{_ATTACHMENT_NOTICE}'
            else:
                message = _ATTACHMENT_NOTICE
            await update_message(
                slack,
                channel=channel,
                ts=placeholder_ts,
                text=message,
            )

    if prepared.images:
        await asyncio.gather(
            *(
                slack.files_upload_v2(
                    channel=channel,
                    thread_ts=thread_ts,
                    file=png,
                    filename=f'diagram-{index + 1}.png',
                    title='Diagram',
                )
                for index, png in enumerate(prepared.images)
            )
        )
    return prepared
