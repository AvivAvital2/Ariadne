"""End-to-end: an image earlier in a thread reaches the model's query.

Wires the real orchestrator → ThreadSession → AgentRunner → images chain; only
the SDK client and the HTTP download are faked. Guards the whole assembled
feature, not one unit.
"""
from __future__ import annotations

import base64
import types

from slack_bridge.agent_runner import AgentRunner
from slack_bridge.orchestrator import answer_question
from slack_bridge.session import ThreadSession
from tests._slack_bridge_helpers import PLACEHOLDER


class _FakeSDKClient:
    def __init__(self, script):
        self._script = script
        self.queries = []

    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def query(self, prompt):
        self.queries.append(prompt)   # async iterable (images) or str (text-only)

    async def receive_response(self):
        for msg in self._script:
            yield msg


class _FakeSlack:
    def __init__(self, messages):
        self._messages = messages

    async def conversations_replies(self, channel, ts):  # noqa: ARG002
        return {'messages': self._messages}


class _ColdPool:
    def __init__(self, session):
        self._session = session

    def __contains__(self, thread_ts):  # always cold
        return False

    async def get_or_create(self, thread_ts):  # noqa: ARG002
        return self._session


async def test_image_in_correspondence_reaches_the_model_query():
    script = [
        types.SimpleNamespace(content=[types.SimpleNamespace(text='looks like a timeout')]),
        types.SimpleNamespace(is_error=False, result='', session_id='S1'),
    ]
    client = _FakeSDKClient(script)
    session = ThreadSession(AgentRunner(client))
    pool = _ColdPool(session)
    slack = _FakeSlack([
        {'user': 'UALICE', 'text': 'see this error', 'files': [
            {'id': 'F1', 'mimetype': 'image/png', 'url_private': 'u1'}]},
        {'user': 'UALICE', 'text': 'what is wrong?'},
    ])

    async def fake_fetch(url, token):
        assert (url, token) == ('u1', PLACEHOLDER)
        return b'\x89PNGscreenshot'

    reply = await answer_question(
        pool=pool, slack=slack, bot_user_id='UBOT', channel='C', thread_ts='T',
        text='what is wrong?', token=PLACEHOLDER, image_fetch=fake_fetch,
    )

    # The turn produced the model's answer…
    assert reply.text == 'looks like a timeout'
    # …and the screenshot rode the query as a single base64 image block.
    assert len(client.queries) == 1
    msgs = [m async for m in client.queries[0]]
    content = msgs[0]['message']['content']
    images = [b for b in content if b.get('type') == 'image']
    assert len(images) == 1
    assert base64.standard_b64decode(images[0]['source']['data']) == b'\x89PNGscreenshot'
    # The question text rode along too.
    text_blocks = [b for b in content if b.get('type') == 'text']
    assert text_blocks and 'what is wrong?' in text_blocks[0]['text']
