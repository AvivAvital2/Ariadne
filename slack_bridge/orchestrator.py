from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from slack_bridge.images import ImageRef, download_images, image_files_in
from slack_bridge.replay import Turn, load_thread


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
    token: str = '',
    trigger_files: Sequence[dict[str, Any]] = (),
    seed_turns: Sequence[Turn] | None = None,
    seed_images: Sequence[ImageRef] | None = None,
    image_fetch: Any = None,
) -> Any:
    """Run one turn for a thread, warming or cold-rebuilding context as needed.

    Warm thread (in the pool) → ask directly on the live session. Cold thread
    (evicted / post-restart / new) → rebuild context from the Slack thread and,
    if there are prior turns, seed them into the first prompt. Slack is the
    durable conversation store; we never rely on an on-disk SDK session.

    ``seed_turns``/``seed_images`` let a caller that has *already* loaded the
    thread (the follow-up gate fetches it to decide engagement) hand the
    transcript and its images in so the cold path doesn't fetch a second time.

    Images attached anywhere in the thread (the "correspondence") are downloaded
    with the bot ``token`` and sent to the model: the whole thread on the cold
    path, the triggering message's ``trigger_files`` on the warm path.
    ``image_fetch`` overrides the HTTP downloader (tests inject a fake).
    """
    cold = thread_ts not in pool
    session = await pool.get_or_create(thread_ts)
    if cold:
        if seed_turns is None:
            ctx = await load_thread(slack, channel, thread_ts, bot_user_id)
            turns, refs = ctx.turns, ctx.images
        else:
            turns, refs = seed_turns, list(seed_images or [])
        prior = turns[:-1]  # everything before the just-asked question (the last turn)
        prompt = render_seed(prior) + text if prior else text
        blobs = await download_images(refs, token=token, fetch=image_fetch)
        return await session.ask(prompt, images=blobs)
    refs = image_files_in([{'files': list(trigger_files)}]) if trigger_files else []
    blobs = await download_images(refs, token=token, fetch=image_fetch)
    return await session.ask(text, images=blobs)
