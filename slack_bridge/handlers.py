from __future__ import annotations

import asyncio
import random
import re
from typing import Any

from slack_bridge.diagram import prepare_diagrams
from slack_bridge.errors import to_user_message
from slack_bridge.format import to_mrkdwn
from slack_bridge.orchestrator import answer_question
from slack_bridge.replay import PLACEHOLDER_PREFIX, load_thread

_MENTION_RE = re.compile(r'<@[A-Z0-9]+>')
# The bot's name, matched as a plain word (any case) so a multi-human thread
# can summon it without a Slack @mention — e.g. "Ariadne, what do you think?".
# Deliberately a regex, NOT an LLM call, so the summon gate costs nothing.
_NAME_RE = re.compile(r'\bariadne\b', re.IGNORECASE)
_PLACEHOLDER_TEXT = f'{PLACEHOLDER_PREFIX} Searching the docs…'
_NOT_ALLOWED = (
    "Sorry — you're not set up to use this bot yet. "
    'Ask an admin to add you (or this channel) to the allowlist.'
)


_SLOW_BEFORE = [
    "This is taking longer than I expected",
    "This one's a bit involved",
    "This is a meatier question than usual",
    "There's a fair bit to sift through here",
    "This is running a little long",
    "Bigger than it first looked",
    "This one needs some real digging",
    "Still piecing this together",
    "Lots of ground to cover here",
    "This is a deep one",
    "This is taking more time than usual",
    "There's a lot to work through",
    "This one's keeping me busy",
    "Turns out this is non-trivial",
    "Still chasing this down",
]

_SLOW_AFTER = [
    "hang on, I'm still digging",
    "bear with me, I'm still on it",
    "give me a moment to finish",
    "still searching, almost there",
    "hang tight, nearly done",
    "I haven't forgotten you, still going",
    "stay with me, I'm getting there",
    "just need a little longer",
    "still pulling the pieces together",
    "won't be much longer",
    "let me keep at it",
    "I'm still on the case",
    "nearly there, thanks for waiting",
    "almost done now",
    "still crunching, hang on",
]


def _slow_notice() -> str:
    """A varied 'still working' line for the soft-timeout notice, mixed locally
    from two phrase pools (no LLM) so the user sees the turn is still alive."""
    return f'⏳ {random.choice(_SLOW_BEFORE)} — {random.choice(_SLOW_AFTER)}.'


def _clean_text(text: str) -> str:
    """Strip Slack ``<@USERID>`` mention tokens (and surrounding whitespace)."""
    return _MENTION_RE.sub('', text or '').strip()


def _name_invoked(text: str) -> bool:
    """True when the bot's name appears in ``text`` — the no-LLM summon gate for
    answering in a multi-human thread without a Slack @mention."""
    return bool(_NAME_RE.search(text or ''))


def _help_text(cfg: Any) -> str:
    """Usage shown for an empty question (bare ``/ariadne`` or a lone @mention).

    Sent immediately — no 'Searching…' placeholder, no agent turn — so the user
    learns how to interact instead of waiting on a no-op search.
    """
    lines = [
        '👋 *Ask me about the team’s codebases* — I answer from the Ariadne docs '
        '(read-only; I don’t change code).',
        '*Tell me which project you mean* — if you don’t, I’ll ask before answering.',
        '• `/ariadne in <project>, <your question>` — e.g. '
        '`/ariadne in <project>, how does the auth flow work?`',
        '• *Across projects:* `/ariadne how does <project-a> work with <project-b>?`',
        '• *Altitude is optional* — answers are in depth by default; add e.g. '
        '“for a product manager” or “from 10k feet” to change the level.',
        '• *Diagrams:* if the docs cover it, ask me to “diagram …” and I’ll render it inline.',
        '• You can also @mention me or DM me with a question.',
    ]
    sources = sorted(getattr(cfg, 'source_descriptions', {}) or {})
    if sources:
        lines.append('I can answer about: ' + ', '.join(f'`{s}`' for s in sources) + '.')
    return '\n'.join(lines)


async def handle_event(
    *, cfg: Any, pool: Any, slack: Any, bot_user_id: str, ack: Any, event: dict, seed_turns: Any = None
) -> None:
    """Transport-agnostic handler for one inbound Slack message.

    Acks immediately (Slack's 3s window), enforces the allowlist, posts a
    placeholder, runs the turn via the orchestrator (warming/cold-rebuilding the
    thread as needed), then edits the placeholder into the answer. Errors and
    timeouts are surfaced honestly rather than masked.

    ``seed_turns`` carries a thread transcript the caller already loaded (the
    follow-up gate fetches it to decide engagement) so the cold path reuses it
    instead of fetching the thread again.
    """
    await ack()

    user = event.get('user', '')
    channel = event.get('channel', '')
    thread_ts = event.get('thread_ts') or event.get('ts')

    if not cfg.is_allowed(user=user, channel=channel):
        await slack.chat_postMessage(channel=channel, thread_ts=thread_ts, text=_NOT_ALLOWED)
        return

    text = _clean_text(event.get('text', ''))
    if not text:
        # Empty question (bare `/ariadne` or a lone @mention): reply with usage
        # right away — no "Searching…" placeholder, no agent turn.
        await slack.chat_postMessage(channel=channel, thread_ts=thread_ts, text=_help_text(cfg))
        return

    placeholder = await slack.chat_postMessage(
        channel=channel, thread_ts=thread_ts, text=_PLACEHOLDER_TEXT
    )
    task = asyncio.ensure_future(
        answer_question(
            pool=pool,
            slack=slack,
            bot_user_id=bot_user_id,
            channel=channel,
            thread_ts=thread_ts,
            text=text,
            seed_turns=seed_turns,
        )
    )
    try:
        try:
            # Soft deadline: shield the task so a timeout HERE does not cancel it.
            # The same turn keeps running on the same session/context.
            reply = await asyncio.wait_for(asyncio.shield(task), cfg.soft_timeout_seconds)
        except asyncio.TimeoutError:
            # Still alive: tell the user, then give the running turn the rest of its
            # hard budget to finish. It is NOT restarted.
            await slack.chat_update(channel=channel, ts=placeholder["ts"], text=_slow_notice())
            reply = await asyncio.wait_for(
                task, max(0.0, cfg.turn_timeout_seconds - cfg.soft_timeout_seconds),
            )
        answer = (getattr(reply, "text", "") or "").strip() or "(no answer returned)"
    except Exception as exc:  # noqa: BLE001 -- surface honestly (timeout included); never mask
        answer = to_user_message(exc)

    # Render any DOT diagrams to PNGs off the event loop (dot is a subprocess).
    # Missing/invalid dot degrades to a warning + the DOT source inside prepared.text.
    prepared = await asyncio.to_thread(prepare_diagrams, answer)
    await slack.chat_update(
        channel=channel, ts=placeholder['ts'], text=to_mrkdwn(prepared.text),
    )
    if prepared.images:
        await asyncio.gather(*(
            slack.files_upload_v2(
                channel=channel,
                thread_ts=thread_ts,
                file=png,
                filename=f'diagram-{i + 1}.png',
                title='Diagram',
            )
            for i, png in enumerate(prepared.images)
        ))


def is_dm_message(event: dict) -> bool:
    """True only for a real user message in a DM.

    Filters out the bot's own messages (``bot_id``) and system/edit events
    (``subtype``) so the ``message`` listener doesn't loop on itself or react to
    channel noise.
    """
    return (
        event.get('channel_type') == 'im'
        and not event.get('bot_id')
        and not event.get('subtype')
    )


def command_to_event(command: dict, echo_ts: str) -> dict:
    """Normalize a slash-command payload into the event shape ``handle_event`` uses.

    ``echo_ts`` is the ts of the message the command handler posts first, so the
    bot threads its answer under it (commands have no triggering message of their
    own).
    """
    return {
        'user': command.get('user_id', ''),
        'channel': command.get('channel_id', ''),
        'ts': echo_ts,
        'text': command.get('text', ''),
    }


async def _noop_ack() -> None:
    pass


def make_listeners(cfg: Any, pool: Any, bot_user_id: str) -> dict[str, Any]:
    """Build the three Slack listeners (transport-agnostic; no ``slack_bolt`` import).

    ``app.py`` registers these on an ``AsyncApp``. Events auto-ack, so they pass a
    no-op ack; the slash command acks immediately, then posts an echo message to
    root a thread before delegating to the shared :func:`handle_event`.
    """

    # Distinct human participants per engaged thread_ts (in-memory; rides the
    # session's lifetime). Drives the 1:1-vs-multi-human follow-up decision.
    thread_humans: dict[str, set[str]] = {}

    def _note(thread_ts: str, user: str) -> None:
        if thread_ts and user and user != bot_user_id:
            thread_humans.setdefault(thread_ts, set()).add(user)

    async def _run(event: dict, client: Any, *, seed_turns: Any = None) -> None:
        _note(event.get('thread_ts') or event.get('ts'), event.get('user', ''))
        await handle_event(
            cfg=cfg, pool=pool, slack=client, bot_user_id=bot_user_id, ack=_noop_ack,
            event=event, seed_turns=seed_turns,
        )

    async def on_mention(event: dict, client: Any) -> None:
        await _run(event, client)

    async def on_message(event: dict, client: Any) -> None:
        # DMs are 1:1 by nature — answer (is_dm_message already drops the bot's
        # own messages and edit/system noise).
        if is_dm_message(event):
            await _run(event, client)
            return
        # Channel/group: never react to the bot's own posts or edit/system events.
        if event.get('bot_id') or event.get('subtype'):
            return
        thread_ts = event.get('thread_ts')
        # A fresh top-level topic (no thread) must @mention the bot — on_mention's job.
        if not thread_ts:
            return
        # An explicit @mention is on_mention's job too — don't answer twice.
        text = event.get('text', '')
        if f'<@{bot_user_id}>' in text:
            return
        # Engagement is durable in SLACK, not in the warm pool. A pool hit is
        # engaged by definition; on a miss (evicted by idle TTL / LRU, or a
        # restart) ask Slack whether the bot has already answered in this thread.
        # If so it stays engaged — the cold turn rebuilds from the thread and
        # re-warms the cache for another cycle; if not, it's a thread the bot
        # never joined, so leave it alone (a fresh topic must @mention).
        seed_turns = None
        if thread_ts not in pool:
            ctx = await load_thread(client, event.get('channel', ''), thread_ts, bot_user_id)
            if not ctx.bot_present:
                return
            # Recover the participant set from Slack so the 1:1-vs-multi-human
            # gate survives eviction (the in-memory tally died with the session).
            thread_humans[thread_ts] = set(ctx.humans)
            seed_turns = ctx.turns
        # Ingest the participant (replay keeps the full thread as context), then
        # answer only in a 1:1 correspondence, or when the bot is named.
        _note(thread_ts, event.get('user', ''))
        if len(thread_humans.get(thread_ts, set())) <= 1 or _name_invoked(text):
            await _run(event, client, seed_turns=seed_turns)

    async def on_command(ack: Any, command: dict, client: Any) -> None:
        await ack()
        text = _clean_text(command.get('text', ''))
        if not text:
            # Bare `/ariadne` -> usage help immediately (no echo, no thread, no agent).
            await client.chat_postMessage(channel=command['channel_id'], text=_help_text(cfg))
            return
        echo = await client.chat_postMessage(
            channel=command['channel_id'],
            text=f'<@{command.get("user_id", "")}> asked: {command.get("text", "")}',
        )
        await _run(command_to_event(command, echo['ts']), client)

    return {'app_mention': on_mention, 'message': on_message, 'command': on_command}
