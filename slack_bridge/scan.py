"""Backfill the local best-of store from a Slack channel's history.

``ariadne-slack scan`` walks the PUBLIC channels the bot belongs to — DMs and
private channels are never read — recovers each past question→answer pair (the
user's message and the bot's threaded reply), joins it to the quality score
Ariadne already logged for that turn (``usage_events.quality_score``, matched by
time), and records the scored pairs into ``.ariadne/local/`` via
:mod:`testimonials`. The store keeps the all-time top-20; re-scans are
idempotent (deduped by the originating message ts).

Scores live in ``ariadne.db`` (wiped on a DB swap) but the testimonials they
seed live in the regen-proof store, so a one-off scan snapshots them out.
"""
from __future__ import annotations

import contextlib
import re
import sqlite3
from datetime import UTC, datetime
from typing import Any, NamedTuple

import testimonials

_PAGE = 200                       # Slack list page size
_DEFAULT_WINDOW_SECONDS = 600.0   # max turn span to look forward for a score
_MENTION_RE = re.compile(r'^(?:<@[^>]+>\s*)+')


class _QA(NamedTuple):
    question: str
    answer: str
    ts: str       # the bot answer's Slack message ts — the dedup + permalink key


def _epoch(slack_ts: str) -> float:
    """A Slack message ts (``"1718000100.000050"``) as epoch seconds."""
    return float(slack_ts)


def _iso_epoch(iso: str) -> float:
    """A ``usage_events.timestamp`` (ISO-8601) as epoch seconds."""
    return datetime.fromisoformat(iso.replace('Z', '+00:00')).timestamp()


def _strip_mentions(text: str) -> str:
    """Drop leading ``<@U…>`` mentions so the question reads naturally."""
    return _MENTION_RE.sub('', text).strip()


def scored_events(conn: sqlite3.Connection) -> list[tuple[float, int]]:
    """All quality-scored usage events as ``(epoch, score)``, time-sorted.

    A row whose timestamp won't parse is skipped — never fatal.
    """
    rows = conn.execute(
        'SELECT timestamp, quality_score FROM usage_events '
        'WHERE quality_score IS NOT NULL'
    ).fetchall()
    out: list[tuple[float, int]] = []
    for ts, score in rows:
        try:
            out.append((_iso_epoch(ts), int(score)))
        except (TypeError, ValueError):
            continue
    return sorted(out)


def _score_for_turn(
    events: list[tuple[float, int]],
    answer_epoch: float,
    answer_epochs: list[float],
    window: float,
) -> int | None:
    """The score logged during the turn that produced this answer, or None.

    The turn runs from the answer's post-time until the next answer (the natural
    turn boundary), capped at ``window`` seconds — so a turn that was never
    scored can't borrow a neighbour's score. If several scores fall in the span
    (the agent logged more than one), the last wins.
    """
    next_epoch = min((e for e in answer_epochs if e > answer_epoch),
                     default=answer_epoch + window)
    hi = min(next_epoch, answer_epoch + window)
    in_span = [s for (e, s) in events if answer_epoch <= e < hi]
    return in_span[-1] if in_span else None


async def _drain(fetch: Any, key: str) -> list[dict]:
    """Collect every item from a cursor-paginated Slack list endpoint."""
    out: list[dict] = []
    cursor: str | None = None
    while True:
        resp = await fetch(cursor)
        out.extend(resp.get(key) or [])
        cursor = ((resp.get('response_metadata') or {}).get('next_cursor')) or None
        if cursor is None:
            return out


async def public_channels(slack: Any) -> list[str]:
    """Ids of the public channels the bot is a member of (privates excluded)."""
    chans = await _drain(
        lambda c: slack.conversations_list(
            types='public_channel', exclude_archived=True, cursor=c, limit=_PAGE),
        'channels',
    )
    return [ch['id'] for ch in chans if ch.get('is_member') and not ch.get('is_private')]


async def qa_pairs(slack: Any, channel: str, bot_user_id: str) -> list[_QA]:
    """Every ``_QA`` in a channel — a question and the bot's threaded reply."""
    history = await _drain(
        lambda c: slack.conversations_history(channel=channel, cursor=c, limit=_PAGE),
        'messages',
    )
    out: list[_QA] = []
    for root in history:
        if not root.get('reply_count'):
            continue
        replies = await _drain(
            lambda c: slack.conversations_replies(
                channel=channel, ts=root['ts'], cursor=c, limit=_PAGE),
            'messages',
        )
        question: str | None = None
        for msg in replies:
            is_bot = msg.get('user') == bot_user_id
            text = (msg.get('text') or '').strip()
            if is_bot and question and text:
                out.append(_QA(question=question, answer=text, ts=msg['ts']))
                question = None
            elif not is_bot and text:
                question = _strip_mentions(text)
    return out


async def scan(
    slack: Any,
    conn: sqlite3.Connection,
    *,
    store_dir: Any,
    bot_user_id: str,
    window_seconds: float = _DEFAULT_WINDOW_SECONDS,
    max_pairs: int | None = None,
    channels: list[str] | None = None,
) -> int:
    """Backfill ``store_dir`` from the bot's public channels. Returns #recorded.

    ``channels`` restricts the scan to those channel ids (still intersected with
    the public channels the bot is a member of, so it can never reach a private
    or non-member channel) — pin the backfill to one channel regardless of what
    else the bot has joined. ``None`` scans all public channels it's in.
    """
    events = scored_events(conn)
    targets = await public_channels(slack)
    if channels is not None:
        wanted = set(channels)
        targets = [c for c in targets if c in wanted]
    pairs: list[tuple[str, _QA]] = []
    for channel in targets:
        for qa in await qa_pairs(slack, channel, bot_user_id):
            pairs.append((channel, qa))

    answer_epochs = sorted(_epoch(qa.ts) for _, qa in pairs)   # turn boundaries (all pairs)
    pairs.sort(key=lambda cqa: _epoch(cqa[1].ts), reverse=True)   # newest first
    if max_pairs is not None:
        pairs = pairs[:max_pairs]

    recorded = 0
    for channel, qa in pairs:
        score = _score_for_turn(events, _epoch(qa.ts), answer_epochs, window_seconds)
        if score is None:
            continue
        permalink: str | None = None
        with contextlib.suppress(Exception):
            link = await slack.chat_getPermalink(channel=channel, message_ts=qa.ts)
            permalink = link.get('permalink')
        if testimonials.record(
            store_dir,
            question=qa.question,
            answer=qa.answer,
            score=score,
            duration_seconds=0.0,
            asked_at=datetime.fromtimestamp(_epoch(qa.ts), UTC).isoformat(),
            permalink=permalink,
            source_ts=qa.ts,
        ):
            recorded += 1
    return recorded


__all__ = ['public_channels', 'qa_pairs', 'scan', 'scored_events']
