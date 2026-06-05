from __future__ import annotations

import types

from slack_bridge.handlers import (
    command_to_event,
    handle_event,
    is_dm_message,
    make_listeners,
)
from tests._slack_bridge_helpers import bridge_config


class _FakeSlack:
    def __init__(self, replies=None):
        self.posted = []   # list of (channel, thread_ts, text)
        self.updated = []  # list of (channel, ts, text)
        self._replies = replies or []
        self._n = 0

    async def chat_postMessage(self, *, channel, text, thread_ts=None):  # noqa: N802 — mirrors slack_sdk
        self._n += 1
        self.posted.append((channel, thread_ts, text))
        return {'ts': f'ph{self._n}'}

    async def chat_update(self, *, channel, ts, text):
        self.updated.append((channel, ts, text))

    async def conversations_replies(self, *, channel, ts):  # noqa: ARG002 — Slack client signature
        return {'messages': self._replies}


class _FakeSession:
    def __init__(self, reply=None, *, boom=None):
        self._reply = reply
        self._boom = boom
        self.asked = []

    async def ask(self, text):
        self.asked.append(text)
        if self._boom is not None:
            raise self._boom
        return self._reply


class _FakePool:
    def __init__(self, session, *, contains=True):
        self._session = session
        self._contains = contains

    def __contains__(self, thread_ts):
        return self._contains

    async def get_or_create(self, thread_ts):  # noqa: ARG002 — mirrors SessionPool
        return self._session


async def _noop_ack():
    pass


async def test_allowed_question_acks_placeholders_runs_agent_and_updates():
    acks = []

    async def ack():
        acks.append(True)

    reply = types.SimpleNamespace(text='the answer', is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(users=frozenset({'UALICE'}))
    event = {'user': 'UALICE', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> how does projecta work?'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=ack, event=event)

    assert acks == [True]                                    # acked within Slack's window
    assert slack.posted[0][2].startswith('🔎')               # placeholder posted in-thread
    assert pool._session.asked == ['how does projecta work?']     # @mention stripped before asking
    assert slack.updated[0][2] == 'the answer'               # placeholder edited to the answer


async def test_not_allowlisted_is_declined_without_invoking_the_agent():
    pool = _FakePool(_FakeSession(types.SimpleNamespace(text='x', is_error=False, session_id='S')))
    slack = _FakeSlack()
    cfg = bridge_config(users=frozenset({'UALICE'}))   # UBOB is not allowed
    event = {'user': 'UBOB', 'channel': 'CX', 'ts': 'T1', 'text': '<@UBOT> hi'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == []          # agent never invoked
    assert slack.updated == []                # no answer
    assert 'allowlist' in slack.posted[0][2].lower() or 'not set up' in slack.posted[0][2].lower()


async def test_agent_error_is_surfaced_honestly_in_the_thread():
    pool = _FakePool(_FakeSession(boom=RuntimeError('skopeo not found')), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    event = {'user': 'UANY', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> q'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert 'skopeo not found' in slack.updated[0][2]   # honest, not masked


def test_is_dm_message_accepts_only_real_user_dms():
    assert is_dm_message({'channel_type': 'im', 'user': 'U1', 'text': 'hi'})
    assert not is_dm_message({'channel_type': 'channel', 'text': 'hi'})        # not a DM
    assert not is_dm_message({'channel_type': 'im', 'bot_id': 'B1', 'text': 'x'})  # the bot itself
    assert not is_dm_message({'channel_type': 'im', 'subtype': 'message_changed'})  # edit/system


def test_command_to_event_normalizes_slash_command_payload():
    event = command_to_event(
        {'user_id': 'U1', 'channel_id': 'C1', 'text': 'how does projecta work?'},
        echo_ts='T9',
    )
    assert event == {'user': 'U1', 'channel': 'C1', 'ts': 'T9', 'text': 'how does projecta work?'}


async def test_command_listener_acks_echoes_then_answers():
    acks = []

    async def ack():
        acks.append(True)

    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    listeners = make_listeners(cfg, pool, 'UBOT')

    await listeners['command'](
        ack=ack,
        command={'user_id': 'U1', 'channel_id': 'C1', 'text': 'how does projecta work?'},
        client=slack,
    )

    assert acks == [True]                                   # acked within Slack's window
    assert any('asked:' in text for _, _, text in slack.posted)  # echo gives the thread a root
    assert pool._session.asked == ['how does projecta work?']
    assert slack.updated[0][2] == 'ans'


async def test_message_listener_answers_dms_and_ignores_noise():
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    cfg = bridge_config(users=frozenset({'U1'}))

    # A real DM is answered.
    dm_pool = _FakePool(_FakeSession(reply), contains=True)
    dm_slack = _FakeSlack()
    await make_listeners(cfg, dm_pool, 'UBOT')['message'](
        event={'channel_type': 'im', 'user': 'U1', 'channel': 'D1', 'ts': 'T1', 'text': 'hi'},
        client=dm_slack,
    )
    assert dm_slack.updated[0][2] == 'ans'

    # The bot's own DM message is ignored — no work, no loop.
    noise_pool = _FakePool(_FakeSession(reply), contains=True)
    noise_slack = _FakeSlack()
    await make_listeners(cfg, noise_pool, 'UBOT')['message'](
        event={'channel_type': 'im', 'bot_id': 'B1', 'text': 'x'},
        client=noise_slack,
    )
    assert noise_slack.posted == [] and noise_slack.updated == []


async def test_thread_followups_one_on_one_then_multiparty_name_gate():
    """1:1 thread → follow-ups answered without an @mention. Once a SECOND
    human joins, the bot still ingests every message (replay captures it) but
    only ANSWERS ones that name 'Ariadne' — a cheap regex gate, no LLM."""
    cfg = bridge_config(channels=frozenset({'C1'}))
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)   # thread is engaged
    slack = _FakeSlack()
    L = make_listeners(cfg, pool, 'UBOT')
    asked = pool._session.asked

    # Alice starts with an @mention → answered; thread now engaged.
    await L['app_mention'](
        event={'channel': 'C1', 'ts': 'TROOT', 'user': 'UALICE', 'text': '<@UBOT> how does X work?'},
        client=slack)
    assert asked == ['how does X work?']

    # Alice follows up in-thread, no @mention → 1:1 correspondence → answered.
    await L['message'](
        event={'channel': 'C1', 'thread_ts': 'TROOT', 'user': 'UALICE', 'ts': 'T2', 'text': 'and what about Y?'},
        client=slack)
    assert asked[-1] == 'and what about Y?'

    # Bob pitches in WITHOUT the name → ingested, NOT answered.
    n = len(asked)
    await L['message'](
        event={'channel': 'C1', 'thread_ts': 'TROOT', 'user': 'UBOB', 'ts': 'T3', 'text': 'I think it is fine'},
        client=slack)
    assert len(asked) == n   # multi-party + no name → silent

    # Bob names Ariadne → answered.
    await L['message'](
        event={'channel': 'C1', 'thread_ts': 'TROOT', 'user': 'UBOB', 'ts': 'T4', 'text': 'Ariadne, what do you think?'},
        client=slack)
    assert asked[-1] == 'Ariadne, what do you think?'


async def test_message_listener_channel_filters():
    """Channel messages the listener must NOT answer: a thread the bot isn't
    engaged in, the bot's own / system messages, and an explicit @mention
    (which app_mention already handles — no double answer)."""
    cfg = bridge_config(channels=frozenset({'C1'}))
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')

    # Not engaged (not in pool) → ignored; a fresh topic needs an @mention.
    p1 = _FakePool(_FakeSession(reply), contains=False)
    await make_listeners(cfg, p1, 'UBOT')['message'](
        event={'channel': 'C1', 'thread_ts': 'T', 'user': 'U1', 'ts': 'T2', 'text': 'hello'}, client=_FakeSlack())
    assert p1._session.asked == []

    # Bot's own + system messages in an engaged thread → ignored.
    p2 = _FakePool(_FakeSession(reply), contains=True)
    m2 = make_listeners(cfg, p2, 'UBOT')['message']
    await m2(event={'channel': 'C1', 'thread_ts': 'T', 'bot_id': 'B', 'ts': 'T2', 'text': 'x'}, client=_FakeSlack())
    await m2(event={'channel': 'C1', 'thread_ts': 'T', 'subtype': 'message_changed', 'ts': 'T2', 'text': 'x'}, client=_FakeSlack())
    assert p2._session.asked == []

    # An explicit @mention → deferred to app_mention (no double answer here).
    p3 = _FakePool(_FakeSession(reply), contains=True)
    await make_listeners(cfg, p3, 'UBOT')['message'](
        event={'channel': 'C1', 'thread_ts': 'T', 'user': 'U1', 'ts': 'T2', 'text': '<@UBOT> hi again'}, client=_FakeSlack())
    assert p3._session.asked == []


def test_name_invoked_regex_matches_the_name_not_arbitrary_text():
    from slack_bridge.handlers import _name_invoked

    assert _name_invoked('Ariadne, what do you think?')
    assert _name_invoked('hey ARIADNE can you help')   # case-insensitive
    assert _name_invoked('per the ariadne docs')       # any mention of the name
    assert not _name_invoked('what do you think?')
    assert not _name_invoked('')
