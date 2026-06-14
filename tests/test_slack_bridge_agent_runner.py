from __future__ import annotations

import base64
import types

from slack_bridge.agent_runner import AgentRunner, build_agent_options
from slack_bridge.images import ImageBlob


def _text_block(t):
    return types.SimpleNamespace(text=t)


def _assistant(*texts):
    # Shaped like claude_agent_sdk.AssistantMessage (has .content, no .is_error).
    return types.SimpleNamespace(content=[_text_block(t) for t in texts])


def _result(*, is_error=False, result='', session_id='S1'):
    # Shaped like claude_agent_sdk.ResultMessage (has .is_error/.result/.session_id).
    return types.SimpleNamespace(is_error=is_error, result=result, session_id=session_id)


class _FakeClient:
    def __init__(self, script):
        self._script = script
        self.connected = False
        self.disconnected = False
        self.queries = []

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True

    async def query(self, text):
        self.queries.append(text)

    async def receive_response(self):
        for msg in self._script:
            yield msg


def test_build_agent_options_is_read_only_dontask():
    opts = build_agent_options(
        system_prompt='SP',
        allowed_tools=['mcp__ariadne__ariadne_search'],
        mcp_servers={'ariadne': {'command': 'uv', 'args': []}},
        model='claude-x',
    )
    assert opts.permission_mode == 'dontAsk'
    assert opts.system_prompt == 'SP'
    assert 'mcp__ariadne__ariadne_search' in opts.allowed_tools
    assert 'ariadne' in opts.mcp_servers
    assert opts.model == 'claude-x'


async def test_ask_connects_queries_and_extracts_text_and_metadata():
    client = _FakeClient([
        _assistant('Hello, ', 'world.'),
        _result(is_error=False, result='(unused)', session_id='S9'),
    ])
    reply = await AgentRunner(client).ask('hi')

    assert client.connected is True
    assert client.queries == ['hi']
    assert reply.text == 'Hello, world.'   # assembled from assistant TextBlocks
    assert reply.is_error is False
    assert reply.session_id == 'S9'


async def test_ask_with_images_streams_content_blocks():
    client = _FakeClient([_assistant('ok'), _result(session_id='S2')])
    blob = ImageBlob(media_type='image/png', data=b'\x89PNGdata')
    reply = await AgentRunner(client).ask('what is this?', images=[blob])

    # Images can't ride the string path (text-only), so ask() switches to the
    # SDK streaming-dict path: query() gets an async iterable of message
    # envelopes, not a str.
    assert len(client.queries) == 1
    sent = client.queries[0]
    assert not isinstance(sent, str)
    msgs = [m async for m in sent]
    assert len(msgs) == 1
    content = msgs[0]['message']['content']
    assert msgs[0]['message']['role'] == 'user'
    assert {'type': 'text', 'text': 'what is this?'} in content
    img_blocks = [b for b in content if b.get('type') == 'image']
    assert len(img_blocks) == 1
    src = img_blocks[0]['source']
    assert src == {
        'type': 'base64',
        'media_type': 'image/png',
        'data': base64.standard_b64encode(b'\x89PNGdata').decode('ascii'),
    }
    # Response extraction is unchanged on the image path.
    assert reply.text == 'ok'
    assert reply.session_id == 'S2'


async def test_ask_image_only_omits_empty_text_block():
    # A screenshot with no words (e.g. a bare "@ariadne" + image) must not
    # send an empty text block — Anthropic rejects empty text content.
    client = _FakeClient([_result(session_id='S3')])
    blob = ImageBlob(media_type='image/jpeg', data=b'jpegbytes')
    await AgentRunner(client).ask('', images=[blob])

    msgs = [m async for m in client.queries[0]]
    content = msgs[0]['message']['content']
    assert [b['type'] for b in content] == ['image']


async def test_ask_flags_tool_error_and_falls_back_to_result_text():
    client = _FakeClient([_result(is_error=True, result='it broke', session_id='S1')])
    reply = await AgentRunner(client).ask('x')

    assert reply.is_error is True
    assert reply.text == 'it broke'   # no assistant text → fall back to result


async def test_aclose_disconnects_the_client():
    client = _FakeClient([])
    await AgentRunner(client).aclose()
    assert client.disconnected is True
