from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from attrs import frozen

# Prefix the bot uses for its "working…" placeholder message. Defined here so
# replay can drop placeholders and the handlers can post them consistently.
PLACEHOLDER_PREFIX = '🔎'


@frozen
class Turn:
    role: str  # 'user' | 'assistant'
    text: str


def _to_transcript(messages: Sequence[dict[str, Any]], bot_user_id: str) -> list[Turn]:
    """Reconstruct a compact conversation transcript from a Slack thread.

    Maps human messages → ``user`` (prefixed with the speaker id so multi-person
    threads stay attributable) and the bot's own messages → ``assistant``. Drops
    empty messages and the bot's "working…" placeholders so only real turns
    survive. Order is preserved.

    This deliberately keeps the *final answers*, not the bot's internal tool
    traces (which never reach Slack anyway), so re-feeding it on a cold turn is
    far lighter than resuming the SDK's full session transcript.
    """
    turns: list[Turn] = []
    for msg in messages:
        text = (msg.get('text') or '').strip()
        if not text:
            continue
        if msg.get('user') == bot_user_id:
            if text.startswith(PLACEHOLDER_PREFIX):
                continue
            turns.append(Turn('assistant', text))
        else:
            speaker = msg.get('user') or 'unknown'
            turns.append(Turn('user', f'[{speaker}] {text}'))
    return turns


async def reconstruct(slack_client: Any, channel: str, thread_ts: str, bot_user_id: str) -> list[Turn]:
    """Fetch a Slack thread and reconstruct its transcript (cold-path memory).

    Slack is the durable conversation store: on a cold follow-up we rebuild
    context from ``conversations.replies`` rather than relying on an on-disk SDK
    session (which is per-cwd/per-machine and may be pruned).
    """
    resp = await slack_client.conversations_replies(channel=channel, ts=thread_ts)
    return _to_transcript(resp.get('messages', []), bot_user_id)
