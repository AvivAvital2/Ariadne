from __future__ import annotations

from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Any

from attrs import frozen
from claude_agent_sdk import ClaudeAgentOptions

from slack_bridge.images import ImageBlob


@frozen
class AgentReply:
    """One turn's outcome: the assistant text, plus metadata the bridge uses."""

    text: str
    is_error: bool
    session_id: str | None


def build_agent_options(
    *,
    system_prompt: str,
    allowed_tools: Sequence[str],
    mcp_servers: Mapping[str, Any],
    model: str | None = None,
    max_turns: int | None = None,
) -> ClaudeAgentOptions:
    """Build the SDK options for a read-only Ariadne agent.

    ``permission_mode="dontAsk"`` + an explicit ``allowed_tools`` means anything
    not on the list is denied silently — no interactive prompts in a server.
    """
    kwargs: dict[str, Any] = {
        'system_prompt': system_prompt,
        'allowed_tools': list(allowed_tools),
        'mcp_servers': mcp_servers,
        'permission_mode': 'dontAsk',
    }
    if model is not None:
        kwargs['model'] = model
    if max_turns is not None:
        kwargs['max_turns'] = max_turns
    return ClaudeAgentOptions(**kwargs)


class AgentRunner:
    """Wraps one ``ClaudeSDKClient`` (one conversation) and runs a turn.

    Connects lazily on first ask; ``aclose()`` disconnects (and reaps the client's
    Ariadne MCP subprocess). The pool serializes calls, so no internal locking.
    """

    def __init__(self, client: Any):
        self._client = client
        self._connected = False

    async def _ensure_connected(self) -> None:
        if not self._connected:
            await self._client.connect()
            self._connected = True

    async def ask(self, text: str, images: Sequence[ImageBlob] = ()) -> AgentReply:
        await self._ensure_connected()
        # Text-only rides the string path (cheapest; the SDK wraps it as a user
        # message). With images we must stream a user-message envelope whose
        # content is a block list — the string path is text-only — so switch to
        # the async-iterable query path.
        if images:
            blocks: list[dict[str, Any]] = []
            if text:
                blocks.append({'type': 'text', 'text': text})
            blocks.extend(img.to_content_block() for img in images)

            async def _stream() -> AsyncIterator[dict[str, Any]]:
                yield {
                    'type': 'user',
                    'message': {'role': 'user', 'content': blocks},
                    'parent_tool_use_id': None,
                }

            await self._client.query(_stream())
        else:
            await self._client.query(text)

        parts: list[str] = []
        result: Any = None
        # Duck-typed extraction (resilient to SDK field changes): assistant text
        # comes from TextBlocks inside a message's ``content``; the terminal
        # ResultMessage is the one carrying both ``is_error`` and ``result``.
        async for msg in self._client.receive_response():
            content = getattr(msg, 'content', None)
            if isinstance(content, list):
                for block in content:
                    block_text = getattr(block, 'text', None)
                    if isinstance(block_text, str):
                        parts.append(block_text)
            if hasattr(msg, 'is_error') and hasattr(msg, 'result'):
                result = msg

        text_out = ''.join(parts).strip()
        if not text_out and result is not None:
            text_out = getattr(result, 'result', '') or ''
        return AgentReply(
            text=text_out,
            is_error=bool(getattr(result, 'is_error', False)),
            session_id=getattr(result, 'session_id', None),
        )

    async def aclose(self) -> None:
        try:
            await self._client.disconnect()
        finally:
            self._connected = False
