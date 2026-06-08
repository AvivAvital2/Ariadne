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


def _human_authors(messages: Sequence[dict[str, Any]], bot_user_id: str) -> frozenset[str]:
    """Distinct non-bot authors of real (non-empty) messages in a thread."""
    return frozenset(
        user
        for msg in messages
        if (user := msg.get('user')) and user != bot_user_id and (msg.get('text') or '').strip()
    )


@frozen
class ThreadContext:
    """A Slack thread reconstructed from its durable history (the cold-path memory).

    ``turns`` is the compact transcript for re-seeding a cold turn; ``bot_present``
    is the durable "the bot is engaged in this thread" signal — it has already
    answered here — which survives pool eviction and restarts because it lives in
    Slack, not in the in-memory pool; ``humans`` is the distinct set of human
    authors, which drives the 1:1-vs-multi-human follow-up gate.
    """

    turns: list[Turn]
    bot_present: bool
    humans: frozenset[str]


async def load_thread(slack_client: Any, channel: str, thread_ts: str, bot_user_id: str) -> ThreadContext:
    """Fetch a Slack thread once and derive everything the cold path needs.

    Slack is the durable conversation store: this is how a follow-up to a thread
    whose warm session was evicted (idle TTL, LRU, or a bridge restart) is still
    recognised as engaged — the bot's own prior reply lives in the thread, not in
    the pool — and how the participant set is recovered so the follow-up gate
    behaves the same cold as it did warm.
    """
    resp = await slack_client.conversations_replies(channel=channel, ts=thread_ts)
    messages = resp.get('messages', [])
    turns = _to_transcript(messages, bot_user_id)
    return ThreadContext(
        turns=turns,
        bot_present=any(t.role == 'assistant' for t in turns),
        humans=_human_authors(messages, bot_user_id),
    )


async def reconstruct(slack_client: Any, channel: str, thread_ts: str, bot_user_id: str) -> list[Turn]:
    """Fetch a Slack thread and reconstruct its transcript (cold-path memory).

    Slack is the durable conversation store: on a cold follow-up we rebuild
    context from ``conversations.replies`` rather than relying on an on-disk SDK
    session (which is per-cwd/per-machine and may be pruned).
    """
    return (await load_thread(slack_client, channel, thread_ts, bot_user_id)).turns
