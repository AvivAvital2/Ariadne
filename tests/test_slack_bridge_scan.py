"""Evolving tests for the channel backfill (``ariadne-slack scan``).

The scan walks the PUBLIC channels the bot belongs to, recovers each past
question→answer pair from the threads, joins it to the quality score Ariadne
logged for that turn (``usage_events``, by time), and records the scored pairs
into the swap-proof local store. Two tests grow:

* the core lifecycle — pair + public-only scope + score-join + idempotent
  re-scan + skip-the-unscored;
* robustness — odd history shapes, pagination, a bad DB timestamp, a permalink
  hiccup, and the ``max_pairs`` bound.

Everything is driven against a fake Slack surface + an in-memory ``usage_events``
table + a tmp store, so it never touches Slack, ariadne.db, or a real .ariadne/.
"""
from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

import testimonials
from slack_bridge.scan import scan

_BOT = 'UBOT'
_CHANS = [
    {'id': 'C_PUB', 'is_member': True, 'is_private': False},      # scanned
    {'id': 'C_PUB2', 'is_member': True, 'is_private': False},     # scanned (aggregated)
    {'id': 'C_PRIV', 'is_member': True, 'is_private': True},      # private → skipped
    {'id': 'C_NONMEM', 'is_member': False, 'is_private': False},  # not a member → skipped
]


def _iso(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, UTC).isoformat()


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(':memory:')
    conn.execute(
        'CREATE TABLE usage_events (id INTEGER PRIMARY KEY AUTOINCREMENT, '
        'timestamp TEXT, quality_score INTEGER)')
    return conn


class _FakeSlack:
    """Minimal async Slack surface: paged history, threaded replies, permalinks."""

    def __init__(self, channels: list[dict]) -> None:
        self.channels = channels
        self.history_pages: dict[str, list[list[dict]]] = {}
        self.replies: dict[tuple[str, str], list[dict]] = {}
        self.history_reads: list[str] = []
        self.permalink_fail = False

    # ---- test-side builders -------------------------------------------------
    def _page(self, channel: str, page: int) -> list[dict]:
        pages = self.history_pages.setdefault(channel, [[]])
        while len(pages) <= page:
            pages.append([])
        return pages[page]

    def add_message(self, channel: str, msg: dict, *, page: int = 0) -> None:
        self._page(channel, page).append(msg)

    def add_qa(self, conn: sqlite3.Connection, channel: str, root_ts: str, question: str,
               answer_ts: str, answer: str, *, score: int | None = None, page: int = 0) -> None:
        self._page(channel, page).append(
            {'ts': root_ts, 'user': 'U1', 'text': f'<@{_BOT}> {question}', 'reply_count': 1})
        self.replies[(channel, root_ts)] = [
            {'ts': root_ts, 'user': 'U1', 'text': f'<@{_BOT}> {question}'},
            {'ts': answer_ts, 'user': _BOT, 'text': answer},
        ]
        if score is not None:                       # score logged mid-turn, after the answer post
            conn.execute('INSERT INTO usage_events (timestamp, quality_score) VALUES (?, ?)',
                         (_iso(float(answer_ts) + 5.0), score))
            conn.commit()

    # ---- the async surface scan consumes ------------------------------------
    async def conversations_list(self, *, types, exclude_archived=True, cursor=None, limit=200):
        return {'channels': self.channels, 'response_metadata': {'next_cursor': ''}}

    async def conversations_history(self, *, channel, cursor=None, limit=200):
        self.history_reads.append(channel)
        pages = self.history_pages.get(channel, [[]])
        idx = int(cursor) if cursor else 0
        nxt = str(idx + 1) if idx + 1 < len(pages) else ''
        return {'messages': pages[idx], 'response_metadata': {'next_cursor': nxt}}

    async def conversations_replies(self, *, channel, ts, cursor=None, limit=200):
        return {'messages': self.replies.get((channel, ts), []),
                'response_metadata': {'next_cursor': ''}}

    async def chat_getPermalink(self, *, channel, message_ts):  # noqa: N802
        if self.permalink_fail:
            raise RuntimeError('slack hiccup')
        return {'permalink': f'https://slack.example/{channel}/{message_ts}'}


async def test_scan_backfills_scored_public_qa(tmp_path: Path) -> None:
    store = testimonials.local_dir(tmp_path)
    conn = _conn()
    slack = _FakeSlack([dict(c) for c in _CHANS])

    # Demand 1 — threads across BOTH public channels are paired (mention
    # stripped), joined to their DB scores by time, and recorded; the private
    # channel and the public channel the bot isn't a member of are never read.
    slack.add_qa(conn, 'C_PUB', '1718000100.000000', 'how does caching work?',
                 '1718000100.000050', 'It uses an LRU.', score=9)
    slack.add_qa(conn, 'C_PUB2', '1718000150.000000', 'how to deploy?',
                 '1718000150.000050', 'Run make deploy.', score=8)
    # No DB score — present only to prove a private / non-member channel is
    # never *read* from Slack.
    slack.add_qa(conn, 'C_PRIV', '1718000100.000000', 'secret?',
                 '1718000100.000050', 'private answer')
    slack.add_qa(conn, 'C_NONMEM', '1718000100.000000', 'lurking?',
                 '1718000100.000050', 'unseen answer')

    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT) == 2
    kept = testimonials.top(store)
    assert {(x.question, x.score) for x in kept} == {
        ('how does caching work?', 9), ('how to deploy?', 8)}
    cache = next(x for x in kept if x.score == 9)
    assert cache.permalink == 'https://slack.example/C_PUB/1718000100.000050'
    assert cache.source_ts == '1718000100.000050'
    assert 'C_PRIV' not in slack.history_reads and 'C_NONMEM' not in slack.history_reads

    # Demand 2 — a re-scan replays the same messages but records nothing new
    # (source_ts dedup), so the store stays stable.
    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT) == 0
    assert len(testimonials.top(store)) == 2

    # Demand 3 — a later scored turn is recorded, while a turn with no score in
    # its window is skipped (can't rank it) — an unscored turn must not borrow
    # the neighbouring turn's score.
    slack.add_qa(conn, 'C_PUB', '1718000200.000000', 'what is X?',
                 '1718000200.000050', 'X is Y.', score=7)
    slack.add_qa(conn, 'C_PUB', '1718000300.000000', 'unscored?',
                 '1718000300.000050', 'no score here.')          # no DB score
    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT) == 1
    kept = testimonials.top(store)
    assert {x.score for x in kept} == {9, 8, 7}
    assert all('unscored' not in x.question for x in kept)

    # Demand 4 — channels=[...] pins the backfill: only the named channel is read
    # and recorded, even though others are public + joined (so a later org-wide
    # rollout can't widen an explicitly-scoped scan).
    slack.add_qa(conn, 'C_PUB', '1718000400.000000', 'pinned q?',
                 '1718000400.000050', 'pinned a.', score=10)
    slack.add_qa(conn, 'C_PUB2', '1718009999.000000', 'other-chan q?',
                 '1718009999.000050', 'other a.', score=7)
    slack.history_reads.clear()
    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT, channels=['C_PUB']) == 1
    assert 'C_PUB2' not in slack.history_reads          # the other channel is never read
    kept = testimonials.top(store)
    assert next(x for x in kept if x.question == 'pinned q?').score == 10
    assert all(x.question != 'other-chan q?' for x in kept)   # filtered channel ignored


async def test_scan_is_robust_to_history_shape(tmp_path: Path) -> None:
    store = testimonials.local_dir(tmp_path)
    conn = _conn()
    slack = _FakeSlack([{'id': 'C_PUB', 'is_member': True, 'is_private': False}])

    # A standalone message (no thread) and a bot-started thread (no question)
    # carry no Q→A and must be ignored without error.
    slack.add_message('C_PUB', {'ts': '1718000010.000000', 'user': 'U1', 'text': 'just chatting'})
    slack.add_message('C_PUB', {'ts': '1718000020.000000', 'user': _BOT,
                                'text': 'an announcement', 'reply_count': 1})
    slack.replies[('C_PUB', '1718000020.000000')] = [
        {'ts': '1718000020.000000', 'user': _BOT, 'text': 'an announcement'},   # bot first → no question
        {'ts': '1718000020.000100', 'user': 'U1', 'text': 'thanks!'},
    ]
    # A real Q→A whose history sits on a SECOND page → pagination must be followed.
    slack.add_qa(conn, 'C_PUB', '1718000500.000000', 'paged question?',
                 '1718000500.000050', 'paged answer.', score=6, page=1)
    # A malformed DB timestamp must be skipped, not crash score parsing.
    conn.execute('INSERT INTO usage_events (timestamp, quality_score) VALUES (?, ?)', ('not-a-date', 5))
    conn.commit()

    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT) == 1
    assert testimonials.top(store)[0].question == 'paged question?'

    # A permalink failure must not cost the testimonial (the link is best-effort).
    slack.permalink_fail = True
    slack.add_qa(conn, 'C_PUB', '1718000600.000000', 'flaky link?',
                 '1718000600.000050', 'answer six.', score=8, page=1)
    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT) == 1
    flaky = next(x for x in testimonials.top(store) if x.answer == 'answer six.')
    assert flaky.score == 8 and flaky.permalink is None

    # max_pairs bounds how many pairs are processed, newest first.
    slack.permalink_fail = False
    slack.add_qa(conn, 'C_PUB', '1718000700.000000', 'newest?',
                 '1718000700.000050', 'newest answer.', score=10, page=1)
    before = len(testimonials.top(store))
    assert await scan(slack, conn, store_dir=store, bot_user_id=_BOT, max_pairs=1) == 1
    after = testimonials.top(store)
    assert len(after) == before + 1
    assert any(x.question == 'newest?' for x in after)   # only the newest pair was processed
