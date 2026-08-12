from __future__ import annotations

import asyncio
import logging
import re
import types

import pytest

import slack_usage
import testimonials
from slack_bridge.budget import TurnBudget, _SLOW_AFTER, _SLOW_BEFORE, slow_notice
from slack_bridge.diagram import dot_available
from slack_bridge.format import to_mrkdwn
from slack_bridge.handlers import (
    _channel_is_shared,
    _org_context,
    _resolve_user_name,
    command_to_event,
    handle_event,
    is_dm_message,
    make_listeners,
)
from tests._slack_bridge_helpers import bridge_config


class _FakeSlack:
    def __init__(self, replies=None, *, shared_channels=(), info_fail=False):
        self.posted = []    # list of (channel, thread_ts, text)
        self.updated = []   # list of (channel, ts, text)
        self.uploaded = []  # list of files_upload_v2 call dicts
        self.permalinks = []
        self._replies = replies or []
        self._n = 0
        self.shared_channels = set(shared_channels)   # channels conversations.info marks shared
        self.info_fail = info_fail                    # make conversations.info raise (fail-closed test)
        self.info_calls = []                          # to assert caching

    async def conversations_info(self, *, channel):  # noqa: N802 — mirrors slack_sdk
        self.info_calls.append(channel)
        if self.info_fail:
            raise RuntimeError('slack hiccup')
        return {'channel': {'id': channel, 'is_ext_shared': channel in self.shared_channels}}

    async def chat_postMessage(self, *, channel, text, thread_ts=None):  # noqa: N802 — mirrors slack_sdk
        self._n += 1
        self.posted.append((channel, thread_ts, text))
        return {'ts': f'ph{self._n}'}

    async def chat_update(self, *, channel, ts, text):
        self.updated.append((channel, ts, text))

    async def conversations_replies(self, *, channel, ts):  # noqa: ARG002 — Slack client signature
        return {'messages': self._replies}
    async def files_upload_v2(
        self,
        *,
        channel,
        file=None,
        content=None,
        thread_ts=None,
        filename=None,
        title=None,
    ):
        self.uploaded.append(
            {
                'channel': channel,
                'thread_ts': thread_ts,
                'file': file,
                'content': content,
                'filename': filename,
                'title': title,
            }
        )
    async def chat_getPermalink(self, *, channel, message_ts):  # noqa: N802
        self.permalinks.append((channel, message_ts))
        return {'ok': True, 'permalink': f'https://slack.example/{channel}/{message_ts}'}


class _FakeSession:
    def __init__(self, reply=None, *, boom=None):
        self._reply = reply
        self._boom = boom
        self.asked = []

    async def ask(self, text, images=()):  # noqa: ARG002 — accepts the new kwarg
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


def test_org_context_extracts_team_enterprise_and_shared_flag():
    assert _org_context({'team_id': 'T1'}) == {
        'team_id': 'T1', 'enterprise_id': '', 'is_ext_shared': False}
    assert _org_context({'enterprise_id': 'E1'})['enterprise_id'] == 'E1'
    ctx = _org_context({'authorizations': [{'enterprise_id': 'E9'}], 'is_ext_shared_channel': True})
    assert ctx == {'team_id': '', 'enterprise_id': 'E9', 'is_ext_shared': True}
    assert _org_context({'context_team_id': 'T2', 'context_enterprise_id': 'E2'}) == {
        'team_id': 'T2', 'enterprise_id': 'E2', 'is_ext_shared': False}


async def test_org_gate_ignores_out_of_org_and_shared_events_even_with_allow_all():
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    # Wide-open bot, but hard-gated to one org — the gate must win over allow_all.
    cfg = bridge_config(allow_all=True, allowed_orgs=frozenset({'T0HOME'}))
    dm = {'channel_type': 'im', 'user': 'U1', 'channel': 'D1', 'ts': 'T1', 'text': 'hi'}

    home = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['message'](
        event=dm, client=home, body={'team_id': 'T0HOME'})
    assert home.updated and home.updated[0][2] == 'ans'          # home org → answered

    ext = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['message'](
        event=dm, client=ext, body={'team_id': 'T0OTHER'})
    assert ext.posted == [] and ext.updated == []                # other org → ignored

    shared = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['app_mention'](
        event={'user': 'U1', 'channel': 'C1', 'ts': 'T3', 'text': '<@UBOT> hi'},
        client=shared, body={'team_id': 'T0HOME', 'is_ext_shared_channel': True})
    assert shared.posted == [] and shared.updated == []          # shared channel → ignored

    acks: list[bool] = []

    async def ack():
        acks.append(True)

    cmd = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['command'](
        ack=ack, client=cmd,
        command={'team_id': 'T0OTHER', 'channel_id': 'C9', 'user_id': 'UX', 'text': 'hi'})
    assert acks == [True]                                        # acked (Slack's 3s window)…
    assert cmd.posted == [] and cmd.updated == []                # …but a slash from another org is ignored


async def test_channel_is_shared_caches_and_fails_closed():
    cache: dict[str, bool] = {}
    s = _FakeSlack(shared_channels={'C_SHARED'})
    assert await _channel_is_shared(s, 'C_SHARED', cache) is True
    assert await _channel_is_shared(s, 'C_INTERNAL', cache) is False
    await _channel_is_shared(s, 'C_SHARED', cache)            # cached: no second lookup
    assert s.info_calls == ['C_SHARED', 'C_INTERNAL']
    # fail-closed: a conversations.info error → treated as shared, and not cached
    boom = _FakeSlack(info_fail=True)
    assert await _channel_is_shared(boom, 'C_X', {}) is True


async def test_slash_in_a_shared_channel_is_ignored_even_when_open():
    """#6: a slash in an externally-shared channel must not be answered there —
    the slash payload has no shared flag, so we verify via conversations.info."""
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    cfg = bridge_config(allow_all=True)            # open; the conversations.info check still blocks
    acks: list[bool] = []

    async def ack():
        acks.append(True)

    shared = _FakeSlack(shared_channels={'C_SHARED'})
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['command'](
        ack=ack, client=shared,
        command={'team_id': 'T0', 'channel_id': 'C_SHARED', 'user_id': 'U1', 'text': 'hi'})
    assert acks == [True] and shared.posted == [] and shared.updated == []   # no echo, no leak

    internal = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['command'](
        ack=ack, client=internal,
        command={'team_id': 'T0', 'channel_id': 'C_INTERNAL', 'user_id': 'U1', 'text': 'hi'})
    assert any('asked:' in t for _, _, t in internal.posted)   # internal slash still works


async def test_connect_dm_is_ignored():
    """#8: an externally-shared (Slack Connect) DM is ignored; a normal DM works."""
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    cfg = bridge_config(allow_all=True)

    connect = _FakeSlack(shared_channels={'D_CONNECT'})
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['message'](
        event={'channel_type': 'im', 'user': 'UX', 'channel': 'D_CONNECT', 'ts': 'T1', 'text': 'hi'},
        client=connect)
    assert connect.posted == [] and connect.updated == []

    internal = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['message'](
        event={'channel_type': 'im', 'user': 'U1', 'channel': 'D_INTERNAL', 'ts': 'T2', 'text': 'hi'},
        client=internal)
    assert internal.updated and internal.updated[0][2] == 'ans'


async def test_org_gate_is_the_floor_under_the_allowlist_and_covers_channel_threads():
    """Security invariant: the org filter sits *under* the channel allowlist (it
    can't be punched through by listing a channel), and it covers the channel
    thread-follow-up path (on_message), not just mentions/DMs."""
    reply = types.SimpleNamespace(text='ans', is_error=False, session_id='S')
    cfg = bridge_config(channels=frozenset({'C1'}), allowed_orgs=frozenset({'T0HOME'}))

    # A foreign-org @mention in the ALLOW-LISTED channel is still ignored —
    # the allowlist does not override the org filter.
    s1 = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['app_mention'](
        event={'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> hi'},
        client=s1, body={'team_id': 'T0FOREIGN'})
    assert s1.posted == [] and s1.updated == []

    # An externally-shared CHANNEL thread follow-up (home team) is ignored too —
    # the gate runs before any thread/engagement logic in on_message.
    s2 = _FakeSlack()
    await make_listeners(cfg, _FakePool(_FakeSession(reply), contains=True), 'UBOT')['message'](
        event={'channel': 'C1', 'user': 'U1', 'ts': 'T2', 'thread_ts': 'T0', 'text': 'follow up'},
        client=s2, body={'team_id': 'T0HOME', 'is_ext_shared_channel': True})
    assert s2.posted == [] and s2.updated == []


async def test_thread_followups_one_on_one_then_multiparty_name_gate():
    """1:1 thread → follow-ups answered without an @mention, UNLESS the message
    @mentions another *user* (then it's addressed to them, not Ariadne, so the
    bot stays out). Once a SECOND human joins, the bot still ingests every
    message (replay captures it) but only ANSWERS ones that name 'Ariadne' — a
    cheap regex gate, no LLM."""
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

    # Still 1:1, but Alice tags a COLLEAGUE (not the bot). The tagged person
    # hasn't posted, so the thread still LOOKS 1:1 — yet the message is addressed
    # to THEM, so the bot must stay out instead of barging in.
    n = len(asked)
    await L['message'](
        event={'channel': 'C1', 'thread_ts': 'TROOT', 'user': 'UALICE', 'ts': 'T2B',
               'text': '<@UCAROL> can you confirm this?'},
        client=slack)
    assert len(asked) == n   # @another-human → not for the bot → silent

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
    assert 'attach the answer as Markdown' in help_text


async def test_handle_event_forwards_token_and_trigger_files(monkeypatch):
    """An image on the triggering message reaches the orchestrator: handle_event
    passes the bot token (for the authenticated download) and the raw files."""
    captured = {}

    async def fake_answer_question(**kwargs):
        captured.update(kwargs)
        return types.SimpleNamespace(text='ok', is_error=False, session_id='S')

    monkeypatch.setattr('slack_bridge.handlers.answer_question', fake_answer_question)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    files = [{'id': 'F1', 'mimetype': 'image/png', 'url_private': 'u1'}]
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1',
             'text': '<@UBOT> what is this?', 'files': files}

    await handle_event(
        cfg=cfg, pool=_FakePool(_FakeSession(None)), slack=slack,
        bot_user_id='UBOT', ack=_noop_ack, event=event,
    )

    assert captured['token'] == cfg.slack_bot_token
    assert captured['trigger_files'] == files
    assert captured['text'] == 'what is this?'


async def test_image_only_message_is_answered_not_helped(monkeypatch):
    """A screenshot with no words (text empty after stripping the @mention) is a
    real question — it must run the agent, not be short-circuited to usage help."""
    called = []

    async def fake_answer_question(**kwargs):
        called.append(kwargs)
        return types.SimpleNamespace(text='ok', is_error=False, session_id='S')

    monkeypatch.setattr('slack_bridge.handlers.answer_question', fake_answer_question)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT>',
             'files': [{'id': 'F1', 'mimetype': 'image/png', 'url_private': 'u1'}]}

    await handle_event(
        cfg=cfg, pool=_FakePool(_FakeSession(None)), slack=slack,
        bot_user_id='UBOT', ack=_noop_ack, event=event,
    )

    assert len(called) == 1                            # ran the agent, not help
    assert slack.posted[0][2].startswith('🔎')          # placeholder, not usage text


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


async def test_greet_command_posts_config_driven_announcement_without_agent():
    """`/ariadne greet` posts a public 'Meet Ariadne' announcement rendered from
    config — no 'asked:' echo, no placeholder, no agent turn (no LLM cost). The
    Covers list reflects the advertised projects by friendly title, falling back
    to the bare source key when a project has no title."""
    pool = _FakePool(_FakeSession(types.SimpleNamespace(text='x', is_error=False, session_id='S')), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(
        channels=frozenset({'C1'}),
        source_descriptions={'src_one': 'first source', 'src_two': 'second source'},
        source_titles={'src_one': 'Source One (S1)'},   # src_two has no title → key fallback
    )
    L = make_listeners(cfg, pool, 'UBOT')

    await L['command'](ack=_noop_ack, command={'user_id': 'U1', 'channel_id': 'C1', 'text': 'greet'}, client=slack)

    # A single canned post — never the agent, the echo, or the placeholder.
    assert pool._session.asked == []
    assert slack.updated == []
    assert len(slack.posted) == 1
    text = slack.posted[0][2]
    assert 'asked:' not in text
    assert 'Searching' not in text and '🔎' not in text
    # Identity, the read-only promise, and the help pointer.
    assert 'Meet Ariadne' in text
    assert 'read-only' in text.lower()
    assert '/ariadne' in text
    # Covers list is config-driven: the titled project shows its label, the
    # untitled one falls back to its source key.
    assert 'Source One (S1)' in text
    assert 'src_two' in text

    # The keyword is matched EXACTLY (after strip/lower) — a real question that
    # merely starts with "greet" must still run the agent, not the announcement.
    real_pool = _FakePool(_FakeSession(types.SimpleNamespace(text='ans', is_error=False, session_id='S')), contains=True)
    real_slack = _FakeSlack()
    await make_listeners(cfg, real_pool, 'UBOT')['command'](
        ack=_noop_ack, command={'user_id': 'U1', 'channel_id': 'C1', 'text': 'greet the team for me'}, client=real_slack)
    assert real_pool._session.asked == ['greet the team for me']   # real question → agent
    assert all('Meet Ariadne' not in t for _, _, t in real_slack.posted)

    # …but surrounding whitespace / casing on the bare keyword still greets.
    loud_pool = _FakePool(_FakeSession(types.SimpleNamespace(text='x', is_error=False, session_id='S')), contains=True)
    loud_slack = _FakeSlack()
    await make_listeners(cfg, loud_pool, 'UBOT')['command'](
        ack=_noop_ack, command={'user_id': 'U1', 'channel_id': 'C1', 'text': '  GREET '}, client=loud_slack)
    assert loud_pool._session.asked == []
    assert 'Meet Ariadne' in loud_slack.posted[0][2]

    # With no projects configured yet (fresh launch), still greet — but omit the
    # Covers block entirely rather than leave a dangling 'Covers:' header.
    bare_cfg = bridge_config(channels=frozenset({'C1'}))
    bare_slack = _FakeSlack()
    await make_listeners(bare_cfg, pool, 'UBOT')['command'](
        ack=_noop_ack, command={'user_id': 'U1', 'channel_id': 'C1', 'text': 'greet'}, client=bare_slack)
    bare = bare_slack.posted[0][2]
    assert 'Meet Ariadne' in bare and 'Covers' not in bare


class _SlowSession:
    """Session whose ask() sleeps a controllable amount before replying, to
    exercise the soft/hard timeout split. Records the call BEFORE sleeping, so a
    cancelled (hard-timed-out) turn still proves it ran once — not restarted."""

    def __init__(self, reply, *, delay):
        self._reply = reply
        self._delay = delay
        self.asked = []

    async def ask(self, text, images=()):  # noqa: ARG002 — accepts the new kwarg
        self.asked.append(text)
        await asyncio.sleep(self._delay)
        return self._reply


async def test_slow_turn_posts_still_working_notice_then_answers_same_run():
    """Past the soft deadline the bot posts a 'still working' notice and lets the
    SAME turn finish on the same context (not a restart) within the hard cap."""
    reply = types.SimpleNamespace(text='the answer', is_error=False, session_id='S')
    pool = _FakePool(_SlowSession(reply, delay=0.3), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}), turn_budget=TurnBudget(soft_seconds=0.05, total_seconds=5.0))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> how does it work?'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == ['how does it work?']        # ran once — same turn, not restarted
    notices = [t for _, _, t in slack.updated if t.startswith('⏳')]
    assert len(notices) == 1                                   # one 'still working' notice
    before, sep, after = notices[0].removeprefix('⏳ ').removesuffix('.').partition(' — ')
    assert sep and before in _SLOW_BEFORE and after in _SLOW_AFTER   # a real pool draw
    assert slack.updated[-1][2] == 'the answer'                # real answer still landed


def test_slow_notice_is_randomized_from_the_phrase_pools():
    """The 'still working' notice is composed fresh on every draw from the two
    phrase pools: each draw is well-formed (⏳ before — after.), both halves come
    from their pools, and across draws both halves actually vary — a randomized
    line, not one canned string."""
    befores, afters = set(), set()
    for _ in range(60):
        notice = slow_notice()
        match = re.fullmatch(r'⏳ (?P<before>.+) — (?P<after>.+)\.', notice)
        assert match, f'malformed notice: {notice!r}'
        assert match['before'] in _SLOW_BEFORE
        assert match['after'] in _SLOW_AFTER
        befores.add(match['before'])
        afters.add(match['after'])
    assert len(befores) > 1 and len(afters) > 1   # randomized on both halves


async def test_turn_exceeding_hard_cap_gives_up_with_a_distinct_budget_message(caplog):
    """Past the hard cap the bot cancels and surfaces a DISTINCT 'I hit my time
    budget' message — it names the limit (so a too-low cap is visible) and is NOT
    the same 'narrow the question' text an inner timeout produces. It still posts
    the 'still working' notice first, ran the turn once, and logs the give-up."""
    reply = types.SimpleNamespace(text='late', is_error=False, session_id='S')
    pool = _FakePool(_SlowSession(reply, delay=5.0), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(channels=frozenset({'C1'}), turn_budget=TurnBudget(soft_seconds=0.05, total_seconds=0.3))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> big question'}

    with caplog.at_level(logging.WARNING):
        await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == ['big question']             # ran once, not restarted
    notices = [t for _, _, t in slack.updated if t.startswith('⏳')]
    assert len(notices) == 1                                   # notice posted before giving up
    final = slack.updated[-1][2]
    assert '0.3' in final                                      # names OUR limit → distinct, surfaces a low cap
    assert 'budget' in final.lower() or 'limit' in final.lower()
    assert '0.3' in caplog.text                                # give-up logged for the operator


class _TimeoutSession:
    """ask() raises a TimeoutError fast (no sleep) — a deeper SDK/MCP timeout that
    ENDS the turn quickly, NOT a slow turn hitting our soft deadline."""

    def __init__(self):
        self.asked = []

    async def ask(self, text, images=()):  # noqa: ARG002 — accepts the new kwarg
        self.asked.append(text)
        raise TimeoutError('mcp tool call timed out')


async def test_fast_internal_timeout_is_not_mistaken_for_soft_deadline(caplog):
    """A TimeoutError raised by the turn ITSELF (deeper layer) must NOT trigger the
    soft-deadline 'still working' notice/extension — and the real error must be
    logged, not silently swallowed. (The bug: wait_for conflated the two, so any
    inner TimeoutError flashed the notice then failed fast regardless of the cap.)"""
    pool = _FakePool(_TimeoutSession(), contains=True)
    slack = _FakeSlack()
    # Generous deadlines: if these were ever hit, the test would be slow. It isn't —
    # the turn fails instantly, proving the budget was never the gate.
    cfg = bridge_config(channels=frozenset({'C1'}), turn_budget=TurnBudget(soft_seconds=30.0, total_seconds=60.0))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> q'}

    with caplog.at_level(logging.ERROR):
        await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert pool._session.asked == ['q']                                  # ran once
    assert not [t for _, _, t in slack.updated if t.startswith('⏳')]     # NO false 'still working'
    assert 'too long' in slack.updated[-1][2].lower()                    # honest timeout message
    assert any(r.exc_info for r in caplog.records)                       # real error logged, not masked


class _FlakyUpdateSlack(_FakeSlack):
    """chat_update raises a transient ConnectionError on its first call, then
    succeeds — to prove the bridge retries an idempotent Slack edit instead of
    losing the computed answer to a one-off network blip."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.update_attempts = 0

    async def chat_update(self, *, channel, ts, text):
        self.update_attempts += 1
        if self.update_attempts == 1:
            raise ConnectionError('transient slack blip')
        await super().chat_update(channel=channel, ts=ts, text=text)


async def test_transient_slack_update_failure_is_retried_so_the_answer_lands():
    """A transient ConnectionError on the answer edit is retried (the edit is
    idempotent), so the computed answer still reaches the thread — not lost to a
    one-off blip. Post/upload are NOT retried (could double-post); edits only."""
    reply = types.SimpleNamespace(text='the answer', is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FlakyUpdateSlack()
    cfg = bridge_config(channels=frozenset({'C1'}))
    event = {'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> q'}

    await handle_event(cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT', ack=_noop_ack, event=event)

    assert slack.update_attempts == 2            # first edit blipped, then retried
    assert slack.updated[-1][2] == 'the answer'  # answer delivered despite the blip


def test_is_dm_message_accepts_file_share():
    """A Slack file upload is a `message` with subtype 'file_share' carrying
    files[] — a real user message, not edit/system noise. The DM gate must
    accept it, else an image attached in a DM never reaches the handler."""
    assert is_dm_message(
        {'channel_type': 'im', 'user': 'U1', 'subtype': 'file_share',
         'text': 'see this', 'files': [{'id': 'F1'}]}
    )
    # genuine noise (and the bot's own posts) still dropped
    assert not is_dm_message({'channel_type': 'im', 'subtype': 'message_changed'})
    assert not is_dm_message(
        {'channel_type': 'im', 'bot_id': 'B1', 'subtype': 'file_share'}
    )
async def test_bridge_records_a_testimonial_only_for_scored_turns(monkeypatch, tmp_path):
    """Evolving: a scored turn with feedback on writes the best-of testimonial to
    the local store with the right fields; a feedback-off turn and a no-score turn
    write nothing."""
    store = testimonials.local_dir(tmp_path)
    reply = types.SimpleNamespace(text='the answer', is_error=False, session_id='S', score=9)

    async def fake_aq(**kw):
        return reply

    monkeypatch.setattr('slack_bridge.handlers.answer_question', fake_aq)

    async def run(*, feedback, slack=None):
        cfg = bridge_config(channels=frozenset({'C1'}), enable_feedback=feedback, ariadne_dir=tmp_path)
        await handle_event(
            cfg=cfg, pool=_FakePool(_FakeSession(None)), slack=slack or _FakeSlack(),
            bot_user_id='UBOT', ack=_noop_ack,
            event={'user': 'U1', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> what is this?'},
        )

    # Demand 1 — scored turn, feedback on: testimonial written faithfully.
    await run(feedback=True)
    kept = testimonials.top(store)
    assert len(kept) == 1
    assert (kept[0].score, kept[0].question, kept[0].answer) == (9, 'what is this?', 'the answer')
    assert kept[0].duration_seconds >= 0
    assert kept[0].permalink                       # captured via chat.getPermalink

    # Demand 2 — feedback off: no capture.
    await run(feedback=False)
    assert len(testimonials.top(store)) == 1

    # Demand 3 — feedback on but the agent reported no score: no capture.
    reply.score = None
    await run(feedback=True)
    assert len(testimonials.top(store)) == 1

    # Demand 4 — a permalink-fetch failure must NOT cost the testimonial.
    reply.score = 8
    flaky = _FakeSlack()

    async def _boom(**kw):
        raise RuntimeError('slack hiccup')

    flaky.chat_getPermalink = _boom
    await run(feedback=True, slack=flaky)
    kept = testimonials.top(store)
    assert any(t.score == 8 and t.permalink is None for t in kept)


def _slack_with_names(names: dict[str, str]) -> _FakeSlack:
    """A fake Slack whose users_info resolves ids → display names."""
    s = _FakeSlack()

    async def users_info(*, user):  # noqa: N802 — mirrors slack_sdk
        return {'user': {'profile': {'display_name': names.get(user, '')}}}

    s.users_info = users_info
    return s


async def test_bridge_records_per_user_usage(monkeypatch, tmp_path):
    """Evolving: every answered turn appends a per-user usage record (counts +
    resolved name, no question text) regardless of enable_feedback; declined and
    empty-question turns (no agent run) record nothing."""
    store = testimonials.local_dir(tmp_path)
    reply = types.SimpleNamespace(
        text='the answer', is_error=False, session_id='S', score=8, outcome='hit')

    async def fake_aq(**kw):
        return reply

    monkeypatch.setattr('slack_bridge.handlers.answer_question', fake_aq)

    async def run(event, *, feedback=True):
        cfg = bridge_config(
            channels=frozenset({'C1'}), enable_feedback=feedback, ariadne_dir=tmp_path)
        await handle_event(
            cfg=cfg, pool=_FakePool(_FakeSession(None)),
            slack=_slack_with_names({'U_alice': 'alice', 'U_bob': 'bob'}),
            bot_user_id='UBOT', ack=_noop_ack, event=event)

    # D1 — an answered, hit turn records the id, resolved name, and outcome.
    await run({'user': 'U_alice', 'channel': 'C1', 'ts': 'T1', 'text': '<@UBOT> q?'})
    rows = slack_usage.aggregate(store)
    assert [(r.actor, r.name, r.questions, r.hits, r.misses) for r in rows] == [
        ('U_alice', 'alice', 1, 1, 0),
    ]

    # D2 — a not-allowlisted user is declined: no agent turn, nothing recorded.
    await run({'user': 'U_evil', 'channel': 'CX', 'ts': 'T2', 'text': '<@UBOT> q?'})
    assert [r.actor for r in slack_usage.aggregate(store)] == ['U_alice']

    # D3 — an empty question replies with help, runs no agent turn, records nothing.
    await run({'user': 'U_alice', 'channel': 'C1', 'ts': 'T3', 'text': '<@UBOT>'})
    assert [(r.actor, r.questions) for r in slack_usage.aggregate(store)] == [
        ('U_alice', 1),
    ]

    # D4 — feedback OFF still counts the question (outcome 'answered', not a hit).
    reply.outcome, reply.score = None, None
    await run({'user': 'U_bob', 'channel': 'C1', 'ts': 'T4', 'text': '<@UBOT> q?'},
              feedback=False)
    bob = next(r for r in slack_usage.aggregate(store) if r.actor == 'U_bob')
    assert (bob.name, bob.questions, bob.hits, bob.misses) == ('bob', 1, 0, 0)


async def test_resolve_user_name_handles_empty_and_lookup_failure():
    """Name resolution is best-effort: no id → empty; a Slack failure falls
    back to the id so the usage record always has an actor."""
    assert await _resolve_user_name(_slack_with_names({'U_a': 'alice'}), '') == ''
    assert await _resolve_user_name(_slack_with_names({'U_a': 'alice'}), 'U_a') == 'alice'

    class _Boom:
        async def users_info(self, *, user):  # noqa: N802 — mirrors slack_sdk
            raise RuntimeError('slack hiccup')

    assert await _resolve_user_name(_Boom(), 'U_x') == 'U_x'


async def test_long_answer_without_explicit_file_request_stays_inline():
    answer = '# Implementation plan\n\n' + ('- Preserve this exact Markdown.\n' * 700)
    reply = types.SimpleNamespace(text=answer, is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(users=frozenset({'UALICE'}))
    event = {
        'user': 'UALICE',
        'channel': 'C1',
        'ts': 'T1',
        'text': '<@UBOT> produce the complete implementation plan',
    }

    await handle_event(
        cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT',
        ack=_noop_ack, event=event,
    )

    assert slack.uploaded == []
    assert slack.updated[-1][2] == to_mrkdwn(answer.strip())


async def test_explicit_markdown_request_attaches_verbatim_answer():
    answer = '# Implementation plan\n\n- Preserve this exact Markdown.\n'
    reply = types.SimpleNamespace(text=answer, is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(users=frozenset({'UALICE'}))
    event = {
        'user': 'UALICE',
        'channel': 'C1',
        'ts': 'T1',
        'text': '<@UBOT> attach your complete answer as a Markdown file',
    }

    await handle_event(
        cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT',
        ack=_noop_ack, event=event,
    )

    assert len(slack.uploaded) == 1
    upload = slack.uploaded[0]
    assert upload['content'] == answer.strip()
    assert upload['file'] is None
    assert upload['filename'] == 'ariadne-answer.md'
    assert upload['title'] == 'Ariadne full answer'
    assert upload['thread_ts'] == 'T1'
    assert 'attached' in slack.updated[-1][2].lower()
    assert 'ariadne-answer.md' in slack.updated[-1][2]
    assert len(slack.updated[-1][2]) < 500


async def test_explicit_markdown_request_separates_fenced_document_from_reply():
    answer = (
        'I found the relevant implementation details.\n\n'
        '````markdown\n'
        '# Implementation plan\n\n'
        '```python\n'
        'print("preserve inner fences")\n'
        '```\n'
        '````'
        'The document is ready.'
    )
    reply = types.SimpleNamespace(text=answer, is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FakeSlack()
    cfg = bridge_config(users=frozenset({'UALICE'}))
    event = {
        'user': 'UALICE',
        'channel': 'C1',
        'ts': 'T1',
        'text': '<@UBOT> attach your complete answer as a Markdown file',
    }

    await handle_event(
        cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT',
        ack=_noop_ack, event=event,
    )

    assert slack.uploaded[0]['content'] == (
        '# Implementation plan\n\n'
        '```python\n'
        'print("preserve inner fences")\n'
        '```'
    )
    posted_reply = slack.updated[-1][2]
    assert 'I found the relevant implementation details.' in posted_reply
    assert 'The document is ready.' in posted_reply
    assert 'ariadne-answer.md' in posted_reply
    assert '# Implementation plan' not in posted_reply


class _FailingMarkdownUploadSlack(_FakeSlack):
    def __init__(self):
        super().__init__()
        self.markdown_upload_attempts = 0

    async def files_upload_v2(self, **kwargs):
        self.markdown_upload_attempts += 1
        raise RuntimeError('Slack file upload failed')


async def test_requested_markdown_upload_failure_falls_back_to_complete_thread_chunks():
    answer = 'Plan introduction.\n\n' + ('A bounded implementation step.\n' * 900)
    reply = types.SimpleNamespace(text=answer, is_error=False, session_id='S')
    pool = _FakePool(_FakeSession(reply), contains=True)
    slack = _FailingMarkdownUploadSlack()
    cfg = bridge_config(users=frozenset({'UALICE'}))
    event = {
        'user': 'UALICE',
        'channel': 'C1',
        'ts': 'T1',
        'text': '<@UBOT> return the full plan as an attached .md file',
    }

    await handle_event(
        cfg=cfg, pool=pool, slack=slack, bot_user_id='UBOT',
        ack=_noop_ack, event=event,
    )

    assert slack.markdown_upload_attempts == 1
    continuation_messages = [
        text
        for _, thread_ts, text in slack.posted[1:]
        if thread_ts == 'T1'
    ]
    assert slack.updated[-1][2] + ''.join(continuation_messages) == answer.strip()
