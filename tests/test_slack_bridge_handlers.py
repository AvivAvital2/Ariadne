from __future__ import annotations
import asyncio

import types

import pytest

from slack_bridge.diagram import dot_available
from slack_bridge.handlers import (
    command_to_event,
    handle_event,
    is_dm_message,
    make_listeners,
)
from tests._slack_bridge_helpers import bridge_config


class _FakeSlack:
    def __init__(self, replies=None):
        self.posted = []    # list of (channel, thread_ts, text)
        self.updated = []   # list of (channel, ts, text)
        self.uploaded = []  # list of files_upload_v2 call dicts
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

    async def files_upload_v2(self, *, channel, file, thread_ts=None, filename=None, title=None):  # noqa: N802
        self.uploaded.append(
            {'channel': channel, 'thread_ts': thread_ts, 'file': file, 'filename': filename}
        )


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

    # A thread the bot never joined (no bot message in Slack) → ignored; a fresh
    # topic needs an @mention. _FakeSlack() has no replies → no bot history.
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


async def test_followup_after_eviction_is_recognised_via_slack_not_the_pool():
    """The reported bug. A thread's warm session is evicted (idle TTL, LRU, or a
    bridge restart), then a human follows up — possibly hours later — WITHOUT an
    @mention. Engagement is durable in Slack (the bot's prior answer lives in the
    thread), so the bot still picks it up, cold-rebuilds from the thread, and
    re-warms the cache for another cycle. Pool membership is NOT the engagement
    test; the 1:1-vs-multi-human gate is recovered from Slack so it too survives
    eviction."""
    cfg = bridge_config(channels=frozenset({'C1'}))
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')

    # Evicted (contains=False) BUT the thread already holds the bot's earlier
    # answer → engaged. The late follow-up is answered, seeded from the thread.
    engaged = _FakeSlack(replies=[
        {'user': 'UALICE', 'text': 'how does X work?'},
        {'user': 'UBOT', 'text': 'X works like so.'},
        {'user': 'UALICE', 'text': 'and what about Y?'},   # the late follow-up
    ])
    p1 = _FakePool(_FakeSession(reply), contains=False)
    await make_listeners(cfg, p1, 'UBOT')['message'](
        event={'channel': 'C1', 'thread_ts': 'TROOT', 'user': 'UALICE', 'ts': 'T9', 'text': 'and what about Y?'},
        client=engaged)
    assert p1._session.asked                                  # picked up despite eviction
    assert 'X works like so.' in p1._session.asked[0]         # cold-rebuilt from Slack history

    # A thread the bot never joined (no bot message) → still ignored.
    stranger = _FakeSlack(replies=[
        {'user': 'UALICE', 'text': 'hey folks'},
        {'user': 'UBOB', 'text': 'what about Y?'},
    ])
    p2 = _FakePool(_FakeSession(reply), contains=False)
    await make_listeners(cfg, p2, 'UBOT')['message'](
        event={'channel': 'C1', 'thread_ts': 'TX', 'user': 'UBOB', 'ts': 'T9', 'text': 'what about Y?'},
        client=stranger)
    assert p2._session.asked == []                            # bot never spoke here → leave it alone

    # Multi-human thread, evicted: the participant tally is recovered from Slack,
    # so a follow-up that does NOT name the bot stays silent (gate preserved)…
    multi = _FakeSlack(replies=[
        {'user': 'UALICE', 'text': '<@UBOT> how does X work?'},
        {'user': 'UBOT', 'text': 'X works like so.'},
        {'user': 'UBOB', 'text': 'I disagree'},               # a second human
        {'user': 'UALICE', 'text': 'thoughts?'},
    ])
    p3 = _FakePool(_FakeSession(reply), contains=False)
    await make_listeners(cfg, p3, 'UBOT')['message'](
        event={'channel': 'C1', 'thread_ts': 'TM', 'user': 'UALICE', 'ts': 'T9', 'text': 'thoughts?'},
        client=multi)
    assert p3._session.asked == []                            # 2 humans + not named → silent, even cold

    # …but naming the bot summons it, evicted or not.
    p4 = _FakePool(_FakeSession(reply), contains=False)
    await make_listeners(cfg, p4, 'UBOT')['message'](
        event={'channel': 'C1', 'thread_ts': 'TM', 'user': 'UALICE', 'ts': 'T9', 'text': 'Ariadne, thoughts?'},
        client=multi)
    assert p4._session.asked                                  # named → answered despite eviction


def test_name_invoked_regex_matches_the_name_not_arbitrary_text():
    from slack_bridge.handlers import _name_invoked

    assert _name_invoked('Ariadne, what do you think?')
    assert _name_invoked('hey ARIADNE can you help')   # case-insensitive
    assert _name_invoked('per the ariadne docs')       # any mention of the name
    assert not _name_invoked('what do you think?')
    assert not _name_invoked('')


class _TrackingPool:
    """Pool with REAL keying: get_or_create records the key, __contains__ checks it.

    Unlike _FakePool(contains=True), this exercises the load-bearing invariant —
    that a same-thread follow-up's thread_ts equals the key the initiating
    @mention / slash command created the session under.
    """

    def __init__(self, session):
        self._session = session
        self.keys: set[str] = set()

    def __contains__(self, thread_ts):
        return thread_ts in self.keys

    async def get_or_create(self, thread_ts, *, seed=None):  # noqa: ARG002
        self.keys.add(thread_ts)
        return self._session


async def test_mention_then_same_thread_followup_reuses_session():
    """Scenario A: top-level @mention → bot answers in a thread → a follow-up in
    that thread (no mention) must reuse the session and answer. Real keying."""
    cfg = bridge_config(channels=frozenset({'C1'}))
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    pool = _TrackingPool(_FakeSession(reply))
    slack = _FakeSlack()
    L = make_listeners(cfg, pool, 'UBOT')
    asked = pool._session.asked

    await L['app_mention'](
        event={'channel': 'C1', 'ts': 'T1', 'user': 'UALICE', 'text': '<@UBOT> first?'},
        client=slack)
    assert asked == ['first?']
    assert 'T1' in pool.keys            # session keyed by the mention's ts

    await L['message'](
        event={'channel': 'C1', 'ts': 'T2', 'thread_ts': 'T1', 'user': 'UALICE', 'text': 'and second?'},
        client=slack)
    assert asked[-1] == 'and second?'   # follow-up answered


async def test_command_then_same_thread_followup_reuses_session():
    """Scenario B: /ariadne posts an echo, threads its answer under it → a
    follow-up in that thread must reuse the session. Real keying."""
    cfg = bridge_config(channels=frozenset({'C1'}))
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    pool = _TrackingPool(_FakeSession(reply))
    slack = _FakeSlack()
    L = make_listeners(cfg, pool, 'UBOT')
    asked = pool._session.asked

    async def _ack():
        pass

    await L['command'](
        ack=_ack, command={'user_id': 'U1', 'channel_id': 'C1', 'text': 'first?'}, client=slack)
    assert asked == ['first?']
    assert 'ph1' in pool.keys           # echo ts (first post) is the thread root + key

    await L['message'](
        event={'channel': 'C1', 'ts': 'T9', 'thread_ts': 'ph1', 'user': 'U1', 'text': 'second?'},
        client=slack)
    assert asked[-1] == 'second?'       # follow-up answered


@pytest.mark.skipif(not dot_available(), reason='requires graphviz `dot`')
async def test_diagram_in_reply_is_rendered_to_png_and_uploaded():
    """A reply carrying a ```dot block → the bridge renders it and uploads the
    PNG into the thread, and the edited text drops the raw DOT."""
    reply = types.SimpleNamespace(
        text='Here is the flow:\n\n```dot\ndigraph G { a -> b }\n```\n',
        is_error=False, session_id='S',
    )
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> show me the diagram'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert len(slack.uploaded) == 1
    assert slack.uploaded[0]['file'][:8] == b'\x89PNG\r\n\x1a\n'   # a real PNG
    assert slack.uploaded[0]['thread_ts'] == 'T1'
    assert '```dot' not in slack.updated[-1][2]                   # raw DOT replaced by the image


async def test_empty_question_replies_usage_without_placeholder_or_agent():
    """A bare summon (empty after stripping the @mention) gets usage help
    immediately — never the 'Searching the docs…' placeholder, never an agent turn."""
    pool = _FakePool(_FakeSession(types.SimpleNamespace(text='x', is_error=False, session_id='S')), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT>   '}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == []
    assert slack.updated == []
    assert len(slack.posted) == 1
    help_text = slack.posted[0][2]
    assert 'Searching' not in help_text and '🔎' not in help_text
    assert '/ariadne' in help_text
    low = help_text.lower()
    assert 'project' in low                                   # #1 name the project
    assert 'across' in low                                    # #4 cross-project asking
    assert 'product manager' in low or 'developer' in low     # #2 audience/scope
    assert 'diagram' in low                                   # #3 diagrams when docs allow


async def test_empty_slash_command_replies_usage_immediately():
    """Bare /ariadne with no text -> usage help right away: no 'asked:' echo, no
    placeholder, no agent turn."""
    pool = _FakePool(_FakeSession(types.SimpleNamespace(text='x', is_error=False, session_id='S')), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    L = make_listeners(cfg, pool, 'UBOT')

    await L['command'](ack=_noop_ack, command={'user_id': 'U1', 'channel_id': 'C1', 'text': '  '}, client=slack)

    assert pool._session.asked == []
    assert slack.updated == []
    assert len(slack.posted) == 1
    text = slack.posted[0][2]
    assert 'asked:' not in text
    assert 'Searching' not in text and '🔎' not in text
    assert '/ariadne' in text


class _SlowSession:
    """Session whose ask() sleeps a controllable amount before replying, to
    exercise the soft/hard timeout split. Records the call BEFORE sleeping, so a
    cancelled (hard-timed-out) turn still proves it ran once — not restarted."""

    def __init__(self, reply, *, delay):
        self._reply = reply
        self._delay = delay
        self.asked = []

    async def ask(self, text):
        self.asked.append(text)
        await asyncio.sleep(self._delay)
        return self._reply


async def test_slow_turn_posts_still_working_notice_then_answers_same_run():
    """Past the soft deadline the bot posts a 'still working' notice and lets the
    SAME turn finish on the same context (not a restart) within the hard cap."""
    reply = types.SimpleNamespace(text='the answer', is_error=False, session_id='S')
    pool = _FakePool(_SlowSession(reply, delay=0.3), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}), soft_timeout_seconds=0.05, turn_timeout_seconds=5.0)
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> how does it work?'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == ['how does it work?']        # ran once — same turn, not restarted
    notices = [t for _, _, t in slack.updated if t.startswith('⏳')]
    assert len(notices) == 1                                   # one 'still working' notice
    assert ' — ' in notices[0]                                 # composed shape, not a fixed string
    assert slack.updated[-1][2] == 'the answer'                # real answer still landed


async def test_turn_exceeding_hard_cap_notices_then_times_out():
    """Past the hard cap the bot cancels and surfaces the timeout message — but
    it still posted the 'still working' notice first, and ran the turn once."""
    reply = types.SimpleNamespace(text='late', is_error=False, session_id='S')
    pool = _FakePool(_SlowSession(reply, delay=5.0), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}), soft_timeout_seconds=0.05, turn_timeout_seconds=0.3)
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> big question'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == ['big question']             # ran once, not restarted
    notices = [t for _, _, t in slack.updated if t.startswith('⏳')]
    assert len(notices) == 1                                   # notice posted before giving up
    assert 'too long' in slack.updated[-1][2].lower()          # final = timeout message
