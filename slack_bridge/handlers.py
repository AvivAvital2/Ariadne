from __future__ import annotations

import asyncio
import contextlib
import logging
import re
import time
from typing import Any

import slack_usage
import testimonials
from schema import _now_iso
from slack_bridge.budget import slow_notice
from slack_bridge.diagram import prepare_diagrams
from slack_bridge.errors import TurnBudgetExceeded, to_user_message
from slack_bridge.format import to_mrkdwn
from slack_bridge.images import image_files_in
from slack_bridge.orchestrator import answer_question
from slack_bridge.replay import PLACEHOLDER_PREFIX, load_thread
from slack_bridge.retry import retry_transient

_logger = logging.getLogger(__name__)

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


def _greet_text(cfg: Any) -> str:
    """The public 'Meet Ariadne' announcement for ``/ariadne greet``.

    A canned, no-LLM render (like :func:`_help_text`) meant to be posted into a
    channel at launch. The *Covers* list is config-driven: every advertised source
    is shown by its friendly ``source_titles`` label, falling back to the bare
    source key when none is set — so the announcement always matches what the bot
    can actually answer about. Posted by the bot, so it uses Slack mrkdwn.
    """
    titles = getattr(cfg, 'source_titles', {}) or {}
    names = set(getattr(cfg, 'source_descriptions', {}) or {}) | set(titles)
    covers = sorted((titles.get(name, name) for name in names), key=str.casefold)
    lines = [
        ':wave: *Meet Ariadne*',
        'Ask about the team’s codebases in plain English and get answers from a '
        'curated knowledge base, instead of digging through the repos. '
        'I’m *read-only*: I explain, I never change code.',
        '',
        '*How to ask*',
        '• `/ariadne <question>` — works anywhere',
        '• @-mention me in a channel I’m in, or DM me',
        '',
        '*Name the project* (I’ll ask if I can’t tell):',
        '`/ariadne in <project>, how does the auth flow work?`',
        '',
        '*Good to know*',
        '• Ask me to “diagram the … flow” and I’ll render it inline',
        '• Add “for a product manager” or “from 10k feet” to change the depth',
        '• Cross-project: “how does <A> talk to <B>?”',
        '• Just keep replying in the thread to follow up',
    ]
    if covers:
        lines += ['', '*Covers:*', *(f'• {label}' for label in covers)]
    lines += ['', 'Type `/ariadne` alone anytime for help.']
    return '\n'.join(lines)


async def _slack_update(slack: Any, *, channel: str, ts: str, text: str) -> None:
    """Edit a Slack message, retrying transient network blips with backoff.

    A message edit is idempotent (same ts+text twice is a no-op), so a one-off
    ``TimeoutError``/``ConnectionError`` shouldn't cost the user a computed answer.
    Rate-limit (429) handling is a Slack-client concern and belongs at the
    ``slack_sdk`` retry-handler level, not here (this module stays transport-agnostic).
    """
    await retry_transient(
        lambda: slack.chat_update(channel=channel, ts=ts, text=text),
        on=(TimeoutError, ConnectionError),
    )


async def handle_event(
    *, cfg: Any, pool: Any, slack: Any, bot_user_id: str, ack: Any, event: dict, seed_turns: Any = None, seed_images: Any = None
) -> None:
    """Transport-agnostic handler for one inbound Slack message.

    Acks immediately (Slack's 3s window), enforces the allowlist, posts a
    placeholder, runs the turn via the orchestrator (warming/cold-rebuilding the
    thread as needed), then edits the placeholder into the answer. Errors and
    timeouts are surfaced honestly rather than masked.

    ``seed_turns``/``seed_images`` carry a thread transcript (and its images) the
    caller already loaded (the follow-up gate fetches it to decide engagement) so
    the cold path reuses them instead of fetching the thread again.
    """
    await ack()

    user = event.get('user', '')
    channel = event.get('channel', '')
    thread_ts = event.get('thread_ts') or event.get('ts')

    if not cfg.is_allowed(user=user, channel=channel):
        await slack.chat_postMessage(channel=channel, thread_ts=thread_ts, text=_NOT_ALLOWED)
        return

    text = _clean_text(event.get('text', ''))
    trigger_files = event.get('files') or ()
    if not text and not image_files_in([event]):
        # Empty question (bare `/ariadne` or a lone @mention) with no image:
        # reply with usage right away — no "Searching…" placeholder, no agent turn.
        await slack.chat_postMessage(channel=channel, thread_ts=thread_ts, text=_help_text(cfg))
        return

    placeholder = await slack.chat_postMessage(
        channel=channel, thread_ts=thread_ts, text=_PLACEHOLDER_TEXT
    )
    turn_started = _now_iso()
    turn_t0 = time.monotonic()
    turn_score: int | None = None
    turn_outcome = 'error'
    task = asyncio.ensure_future(
        answer_question(
            pool=pool,
            slack=slack,
            bot_user_id=bot_user_id,
            channel=channel,
            thread_ts=thread_ts,
            text=text,
            token=cfg.slack_bot_token,
            trigger_files=trigger_files,
            seed_turns=seed_turns,
            seed_images=seed_images,
        )
    )
    budget = cfg.turn_budget
    try:
        # Soft deadline. Use asyncio.wait (NOT wait_for): it reports whether the
        # turn is still running and never turns the turn's OWN exception into our
        # timeout. wait_for would re-raise a TimeoutError the turn itself threw as
        # if it were our soft deadline — flashing the notice and failing fast,
        # bypassing the whole budget. We distinguish "still running" from "finished
        # (maybe with an error)" explicitly.
        done, _ = await asyncio.wait({task}, timeout=budget.soft_seconds)
        if not done:
            # Genuinely still running: tell the user, then give it the rest of the
            # hard budget — the SAME turn, not a restart.
            await _slack_update(slack, channel=channel, ts=placeholder['ts'], text=slow_notice())
            done, _ = await asyncio.wait(
                {task}, timeout=budget.extension_seconds,
            )
        if not done:
            # Hard cap exceeded — WE gave up (the turn was still running). Cancel,
            # log it, and surface a DISTINCT budget message (naming the limit) so a
            # too-low cap is visible — not the same text an inner timeout produces.
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
            _logger.warning(
                'Slack turn hit its %ss budget — cancelled (channel=%s thread=%s)',
                budget.total_seconds, channel, thread_ts,
            )
            answer = to_user_message(TurnBudgetExceeded(budget.total_seconds))
        else:
            reply = await task  # the real result — or re-raises the turn's real error
            answer = (getattr(reply, 'text', '') or '').strip() or '(no answer returned)'
            turn_score = getattr(reply, 'score', None)
            turn_outcome = getattr(reply, 'outcome', None) or 'answered'
    except Exception as exc:  # noqa: BLE001 -- surface honestly (timeout included); never mask
        _logger.exception('Slack turn failed (channel=%s thread=%s)', channel, thread_ts)
        answer = to_user_message(exc)

    # Render any DOT diagrams to PNGs off the event loop (dot is a subprocess).
    # Missing/invalid dot degrades to a warning + the DOT source inside prepared.text.
    prepared = await asyncio.to_thread(prepare_diagrams, answer)
    await _slack_update(
        slack, channel=channel, ts=placeholder['ts'], text=to_mrkdwn(prepared.text),
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
    if cfg.enable_feedback and turn_score is not None:
        # Best-effort: the answer is already posted, so capture must never surface to
        # the user. The permalink is a nice-to-have, fetched in its OWN suppress so a
        # transient Slack hiccup can't cost us the testimonial. Score came from the
        # agent's tool-call stream (AgentReply.score); the store is the separate,
        # swap-proof .ariadne/local/.
        permalink = None
        with contextlib.suppress(Exception):
            link = await slack.chat_getPermalink(channel=channel, message_ts=placeholder['ts'])
            permalink = link.get('permalink')
        with contextlib.suppress(Exception):
            testimonials.record(
                testimonials.local_dir(cfg.ariadne_dir),
                question=text,
                answer=answer,
                score=turn_score,
                duration_seconds=time.monotonic() - turn_t0,
                asked_at=turn_started,
                permalink=permalink,
                images=prepared.images,
            )

    with contextlib.suppress(Exception):
        name = await _resolve_user_name(slack, user)
        slack_usage.record(
            testimonials.local_dir(cfg.ariadne_dir),
            asked_at=turn_started,
            actor=user,
            name=name,
            outcome=turn_outcome,
            score=turn_score,
        )


async def _resolve_user_name(slack, user: str) -> str:
    """Best-effort Slack display name for a user id; the id itself on
    failure, so a transient Slack hiccup never costs the usage record its
    actor. No caching: names stay fresh and there is no cross-turn state."""
    if not user:
        return ''
    try:
        resp = await slack.users_info(user=user)
        profile = (resp.get('user') or {}).get('profile') or {}
        return (
            profile.get('display_name')
            or profile.get('real_name')
            or (resp.get('user') or {}).get('name')
            or user
        )
    except Exception:
        return user


def is_dm_message(event: dict) -> bool:
    """True only for a real user message in a DM.

    Filters out the bot's own messages (``bot_id``) and system/edit events
    (``subtype``) so the ``message`` listener doesn't loop on itself or react to
    channel noise.
    """
    return (
        event.get('channel_type') == 'im'
        and not event.get('bot_id')
        and event.get('subtype') in (None, 'file_share')
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


def _org_context(envelope: dict) -> dict:
    """Org identity + external-share flag from an event envelope or slash payload.

    Works for both the Events API envelope (``body``) and a slash-command
    payload — they share ``team_id``/``enterprise_id``; only events carry
    ``is_ext_shared_channel`` (a slash command from outside is caught by team).
    """
    auths = envelope.get('authorizations') or ()
    enterprise = (envelope.get('enterprise_id') or envelope.get('context_enterprise_id')
                  or (auths[0].get('enterprise_id') if auths else None) or '')
    return {
        'team_id': envelope.get('team_id') or envelope.get('context_team_id') or '',
        'enterprise_id': enterprise,
        'is_ext_shared': bool(envelope.get('is_ext_shared_channel')),
    }


async def _channel_is_shared(client: Any, channel_id: str, cache: dict[str, bool]) -> bool:
    """Authoritative externally-shared check via ``conversations.info`` (cached).

    The event envelope's ``is_ext_shared_channel`` covers mentions/channel
    messages, but a **slash** payload carries no such flag and a **Slack Connect
    DM** may not either — so for those surfaces we ask Slack directly. Fail-closed:
    if the lookup errors we treat the channel as shared (and don't cache the
    failure, so the next message retries) — never leak into an unverified channel.
    """
    if channel_id in cache:
        return cache[channel_id]
    try:
        ch = (await client.conversations_info(channel=channel_id)).get('channel') or {}
    except Exception:
        _logger.warning('conversations.info failed for %s — treating as externally shared', channel_id)
        return True
    shared = bool(ch.get('is_ext_shared') or ch.get('is_shared')
                  or ch.get('is_org_shared') or ch.get('is_pending_ext_shared'))
    cache[channel_id] = shared
    return shared


def make_listeners(cfg: Any, pool: Any, bot_user_id: str) -> dict[str, Any]:
    """Build the three Slack listeners (transport-agnostic; no ``slack_bolt`` import).

    ``app.py`` registers these on an ``AsyncApp``. Events auto-ack, so they pass a
    no-op ack; the slash command acks immediately, then posts an echo message to
    root a thread before delegating to the shared :func:`handle_event`.
    """

    # Distinct human participants per engaged thread_ts (in-memory; rides the
    # session's lifetime). Drives the 1:1-vs-multi-human follow-up decision.
    thread_humans: dict[str, set[str]] = {}
    shared_cache: dict[str, bool] = {}   # channel_id → externally-shared? (conversations.info, cached)

    def _note(thread_ts: str, user: str) -> None:
        if thread_ts and user and user != bot_user_id:
            thread_humans.setdefault(thread_ts, set()).add(user)

    async def _run(event: dict, client: Any, *, seed_turns: Any = None, seed_images: Any = None) -> None:
        _note(event.get('thread_ts') or event.get('ts'), event.get('user', ''))
        await handle_event(
            cfg=cfg, pool=pool, slack=client, bot_user_id=bot_user_id, ack=_noop_ack,
            event=event, seed_turns=seed_turns, seed_images=seed_images,
        )

    async def on_mention(event: dict, client: Any, body: dict | None = None) -> None:
        if not cfg.is_org_allowed(**_org_context(body or {})):
            return                                   # outside the org → silently ignore
        await _run(event, client)

    async def on_message(event: dict, client: Any, body: dict | None = None) -> None:
        if not cfg.is_org_allowed(**_org_context(body or {})):
            return                                   # outside the org → silently ignore
        # DMs are 1:1 by nature — answer (is_dm_message already drops the bot's
        # own messages and edit/system noise).
        if is_dm_message(event):
            if await _channel_is_shared(client, event.get('channel', ''), shared_cache):
                return                               # Slack Connect DM → ignore (#8)
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
        seed_images = None
        if thread_ts not in pool:
            ctx = await load_thread(client, event.get('channel', ''), thread_ts, bot_user_id)
            if not ctx.bot_present:
                return
            # Recover the participant set from Slack so the 1:1-vs-multi-human
            # gate survives eviction (the in-memory tally died with the session).
            thread_humans[thread_ts] = set(ctx.humans)
            seed_turns = ctx.turns
            seed_images = ctx.images
        # Ingest the participant (replay keeps the full thread as context).
        _note(thread_ts, event.get('user', ''))
        # Answer a 1:1 correspondence automatically, or whenever the bot is named.
        # But an @mention of another *user* (a bot @mention already returned above)
        # means the message is addressed to THEM, not Ariadne — don't barge in, even
        # while the thread still looks 1:1 (the tagged person hasn't posted yet, so
        # they're absent from the human tally).
        one_on_one = len(thread_humans.get(thread_ts, set())) <= 1
        addresses_another_user = bool(_MENTION_RE.search(text))
        if _name_invoked(text) or (one_on_one and not addresses_another_user):
            await _run(event, client, seed_turns=seed_turns, seed_images=seed_images)

    async def on_command(ack: Any, command: dict, client: Any) -> None:
        await ack()
        if not cfg.is_org_allowed(**_org_context(command)):
            return                                   # outside the org → silently ignore
        if await _channel_is_shared(client, command.get('channel_id', ''), shared_cache):
            return                                   # slash in a shared channel → ignore (#6)
        text = _clean_text(command.get('text', ''))
        if not text:
            # Bare `/ariadne` -> usage help immediately (no echo, no thread, no agent).
            await client.chat_postMessage(channel=command['channel_id'], text=_help_text(cfg))
            return
        if text.lower() == 'greet':
            # `/ariadne greet` -> public 'Meet Ariadne' announcement, canned (no
            # agent). Matched exactly so a real question ("greet the team…") still
            # routes to the agent; _clean_text already stripped surrounding space.
            await client.chat_postMessage(channel=command['channel_id'], text=_greet_text(cfg))
            return
        echo = await client.chat_postMessage(
            channel=command['channel_id'],
            text=f'<@{command.get("user_id", "")}> asked: {command.get("text", "")}',
        )
        await _run(command_to_event(command, echo['ts']), client)

    return {'app_mention': on_mention, 'message': on_message, 'command': on_command}
