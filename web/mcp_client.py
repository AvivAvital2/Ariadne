"""MCP client bridge for the web onboarding backend.

The web server is an MCP *client*. It connects to the Ariadne MCP server
(stdio subprocess locally; streamable-http remotely — future) and forwards
browser requests to MCP tool calls. One session is held for the server's
lifetime, mirroring how ``AriadneService`` keeps one ``Library`` open.

``MCPBridge`` is deliberately tiny and depends only on a ``call_tool``-shaped
session, so the aiohttp handlers can be tested against a fake session without
spawning a subprocess.
"""
from __future__ import annotations

import os
import sys
from contextlib import AsyncExitStack
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from mcp import StdioServerParameters


class MCPCallError(RuntimeError):
    """An MCP tool returned an error result."""


class MCPBridge:
    """Forward ``(tool, args)`` to an MCP session and return its structured dict."""

    def __init__(self, session) -> None:
        self._session = session

    async def call(
        self, tool: str, arguments: dict | None = None,
        *, progress_callback=None,
    ) -> dict:
        """Forward ``(tool, args)`` to an MCP session and return its structured dict.

        ``progress_callback`` (optional) receives the tool's progress
        notifications — ``(progress, total, message)`` — each time the server
        calls ``ctx.report_progress``; the onboarding backend uses it to stream
        build phases to the browser over SSE.
        """
        result = await self._session.call_tool(
            tool, arguments or {}, progress_callback=progress_callback)
        if getattr(result, 'isError', False):
            raise MCPCallError(_first_text(result) or f'{tool} failed')
        data = getattr(result, 'structuredContent', None)
        if data is None:
            # Tools without an output schema return plain text.
            return {'output': _first_text(result) or ''}
        return data


def _first_text(result) -> str | None:
    for block in getattr(result, 'content', None) or []:
        text = getattr(block, 'text', None)
        if text:
            return text
    return None


def stdio_server_params(config_path: str | None = None) -> StdioServerParameters:
    """Params to spawn the Ariadne MCP server as a stdio subprocess.

    The subprocess inherits the current environment so it resolves the same
    ariadne.yaml the user is onboarding into; ``config_path`` pins
    ``ARIADNE_CONFIG`` explicitly when given.
    """
    from mcp import StdioServerParameters

    env = dict(os.environ)
    if config_path:
        env['ARIADNE_CONFIG'] = config_path
    return StdioServerParameters(
        command=sys.executable,
        args=['-m', 'ariadne_mcp.server'],
        env=env,
    )


async def connect_stdio(stack: AsyncExitStack, params: StdioServerParameters) -> MCPBridge:
    """Open a stdio MCP session on ``stack`` and return a bridge over it.

    The caller owns ``stack`` and closes it on shutdown to tear down the
    session and the server subprocess.
    """
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client

    read, write = await stack.enter_async_context(stdio_client(params))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return MCPBridge(session)


async def connect_http(stack: AsyncExitStack, url: str) -> MCPBridge:
    """Open a streamable-http MCP session on ``stack`` and return a bridge.

    The remote counterpart of ``connect_stdio``: connects to a running
    ``ariadne mcp --http`` server at ``url`` instead of spawning one over
    stdio. The caller owns ``stack`` and closes it on shutdown.
    """
    from mcp import ClientSession
    from mcp.client.streamable_http import streamablehttp_client

    read, write, _ = await stack.enter_async_context(streamablehttp_client(url))
    session = await stack.enter_async_context(ClientSession(read, write))
    await session.initialize()
    return MCPBridge(session)
