from __future__ import annotations

import types

from slack_bridge.images import ImageRef
from slack_bridge.orchestrator import answer_question
from tests._slack_bridge_helpers import PLACEHOLDER


class _FakeSession:
    def __init__(self):
        self.asked = []
        self.images = []   # per-call list of ImageBlobs forwarded to ask()

    async def ask(self, text, images=()):
        self.asked.append(text)
        self.images.append(list(images))
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


async def test_cold_thread_downloads_images_from_correspondence():
    # A screenshot attached earlier in the thread is discovered on the cold
    # path, downloaded with the bot token, and handed to the session.
    pool = _FakePool(contains=False)
    slack = _FakeSlack([
        {'user': 'UALICE', 'text': 'look at this', 'files': [
            {'id': 'F1', 'mimetype': 'image/png', 'url_private': 'u1'}]},
        {'user': 'UALICE', 'text': 'what is wrong?'},
    ])

    async def fake_fetch(url, token):
        assert (url, token) == ('u1', PLACEHOLDER)
        return b'PNG'

    await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T',
        text='what is wrong?', token=PLACEHOLDER, image_fetch=fake_fetch,
    )

    blobs = pool.session.images[0]
    assert [(b.media_type, b.data) for b in blobs] == [('image/png', b'PNG')]


async def test_warm_thread_downloads_trigger_image_without_thread_fetch():
    pool = _FakePool(contains=True)
    slack = _FakeSlack([])

    async def fake_fetch(url, token):
        return b'JPG'

    await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T',
        text='what is this?', token=PLACEHOLDER, image_fetch=fake_fetch,
        trigger_files=[{'id': 'F9', 'mimetype': 'image/jpeg', 'url_private': 'u9'}],
    )

    assert slack.replies_calls == 0                  # warm → no thread fetch
    assert [b.data for b in pool.session.images[0]] == [b'JPG']


async def test_cold_with_seed_uses_seed_images_without_refetch():
    # Caller already loaded the thread (follow-up gate) and handed images in —
    # the cold path must use them, not fetch the thread again.
    pool = _FakePool(contains=False)
    slack = _FakeSlack([])

    async def fake_fetch(url, token):
        return b'X'

    await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T',
        text='q', token=PLACEHOLDER, image_fetch=fake_fetch,
        seed_turns=[], seed_images=[ImageRef('F1', 'u1', 'image/png')],
    )

    assert slack.replies_calls == 0                  # seed provided → no refetch
    assert [b.data for b in pool.session.images[0]] == [b'X']
