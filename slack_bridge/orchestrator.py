from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slack_bridge.replay import Turn, reconstruct


def render_seed(turns: Sequence[Turn]) -> str:
    """Render prior thread turns as a compact context preamble for a cold turn.

    Kept lightweight on purpose: only the distilled Q&A (no tool traces), so
    re-feeding it is far cheaper than resuming the SDK's full transcript.
    """
    lines = [f'{"User" if t.role == "user" else "Assistant"}: {t.text}' for t in turns]
    body = '\n'.join(lines)
    return (
        'Earlier in this Slack thread:\n'
        f'{body}\n\n'
        'Continue the conversation. The latest message is:\n'
    )


async def answer_question(
    *,
    pool: Any,
    slack: Any,
    bot_user_id: str,
    channel: str,
    thread_ts: str,
    text: str,
) -> Any:
    """Run one turn for a thread, warming or cold-rebuilding context as needed.

    Warm thread (in the pool) → ask directly on the live session. Cold thread
    (evicted / post-restart / new) → rebuild context from the Slack thread and,
    if there are prior turns, seed them into the first prompt. Slack is the
    durable conversation store; we never rely on an on-disk SDK session.
    """
    cold = thread_ts not in pool
    session = await pool.get_or_create(thread_ts)
    if cold:
        turns = await reconstruct(slack, channel, thread_ts, bot_user_id)
        prior = turns[:-1]  # everything before the just-asked question (the last turn)
        prompt = render_seed(prior) + text if prior else text
        return await session.ask(prompt)
    return await session.ask(text)
