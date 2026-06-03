from __future__ import annotations

import types

from slack_bridge.orchestrator import answer_question


class _FakeSession:
    def __init__(self):
        self.asked = []

    async def ask(self, text):
        self.asked.append(text)
        return types.SimpleNamespace(text='answer', is_error=False, session_id='S')


class _FakePool:
    def __init__(self, *, contains):
        self._contains = contains
        self.session = _FakeSession()

    def __contains__(self, thread_ts):
        return self._contains

    async def get_or_create(self, thread_ts):  # noqa: ARG002 — mirrors SessionPool signature
        return self.session


class _FakeSlack:
    def __init__(self, messages):
        self._messages = messages
        self.replies_calls = 0

    async def conversations_replies(self, channel, ts):  # noqa: ARG002 — Slack client signature
        self.replies_calls += 1
        return {'messages': self._messages}


async def test_warm_thread_asks_directly_without_replay():
    pool = _FakePool(contains=True)
    slack = _FakeSlack([])

    await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T', text='hi'
    )

    assert pool.session.asked == ['hi']   # asked verbatim, no seed
    assert slack.replies_calls == 0        # warm → no Slack history fetch


async def test_cold_new_thread_has_no_prior_to_seed():
    pool = _FakePool(contains=False)
    # A brand-new mention: only the current question (+ a dropped placeholder).
    slack = _FakeSlack([
        {'user': 'UALICE', 'text': 'how does projecta work?'},
        {'user': 'UBOT', 'text': '🔎 working…'},
    ])

    await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T',
        text='how does projecta work?',
    )

    assert slack.replies_calls == 1                       # cold → fetch history
    assert pool.session.asked == ['how does projecta work?']   # nothing prior → no seed


async def test_cold_continuation_seeds_prior_turns():
    pool = _FakePool(contains=False)
    slack = _FakeSlack([
        {'user': 'UALICE', 'text': 'q1'},
        {'user': 'UBOT', 'text': 'a1'},
        {'user': 'UALICE', 'text': 'q2-current'},
        {'user': 'UBOT', 'text': '🔎 working…'},   # placeholder → dropped
    ])

    await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T',
        text='q2-current',
    )

    sent = pool.session.asked[0]
    assert 'q1' in sent and 'a1' in sent     # prior turns seeded as context
    assert sent.endswith('q2-current')        # current question appended last
