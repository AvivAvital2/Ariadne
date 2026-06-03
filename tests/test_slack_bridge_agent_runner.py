from __future__ import annotations

import types

from slack_bridge.agent_runner import AgentRunner, build_agent_options


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


async def test_ask_flags_tool_error_and_falls_back_to_result_text():
    client = _FakeClient([_result(is_error=True, result='it broke', session_id='S1')])
    reply = await AgentRunner(client).ask('x')

    assert reply.is_error is True
    assert reply.text == 'it broke'   # no assistant text → fall back to result


async def test_aclose_disconnects_the_client():
    client = _FakeClient([])
    await AgentRunner(client).aclose()
    assert client.disconnected is True
