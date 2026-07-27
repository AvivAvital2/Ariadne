"""The web server's MCP connection must open AND close inside one task.

aiohttp runs ``on_startup`` and ``on_cleanup`` in different tasks. The MCP
client (stdio / streamable-http) opens an anyio cancel scope that anyio requires
be exited in the same task it was entered. Opening it on startup and closing it
from the cleanup task raises::

    RuntimeError: Attempted to exit cancel scope in a different task than it
    was entered in

which makes ``ariadne serve`` crash on shutdown and survive SIGTERM (the MCP
subprocess leaks). Regression tests for that.
"""
from __future__ import annotations

import asyncio
import contextlib

import anyio
import pytest

from web import server as web_server


@pytest.mark.parametrize('transport', ['stdio', 'http'])
async def test_shutdown_does_not_cross_task_cancel_scope(monkeypatch, transport):
    closed = {'v': False}

    @contextlib.asynccontextmanager
    async def fake_transport():
        # A task group is the same task-affine cancel scope the real
        # stdio_client / streamablehttp_client open internally.
        async with anyio.create_task_group():
            try:
                yield ('read', 'write')
            finally:
                closed['v'] = True

    async def fake_connect(stack, *_args):
        await stack.enter_async_context(fake_transport())
        return 'bridge'

    if transport == 'stdio':
        monkeypatch.setattr(web_server, 'connect_stdio', fake_connect)
        app = web_server.make_app(server_params=object())
    else:
        monkeypatch.setattr(web_server, 'connect_http', fake_connect)
        app = web_server.make_app(mcp_url='http://mcp.test')

    # aiohttp invokes on_startup and on_cleanup in DIFFERENT tasks — reproduce
    # that by running each in its own task.
    await asyncio.ensure_future(web_server._startup_connect(app))
    assert app['bridge'] == 'bridge'
    assert closed['v'] is False          # connection held open after startup

    # Closing from a different task than startup must not raise the cancel-scope
    # RuntimeError — and must actually tear the connection down.
    await asyncio.ensure_future(web_server._cleanup_disconnect(app))
    assert closed['v'] is True


async def test_startup_failure_is_surfaced_not_masked(monkeypatch):
    """A connection failure during startup must propagate out of on_startup
    (so ``ariadne serve`` fails loudly), and cleanup afterward must be safe."""
    async def boom(stack, *_args):
        raise RuntimeError('cannot reach MCP server')

    monkeypatch.setattr(web_server, 'connect_stdio', boom)
    app = web_server.make_app(server_params=object())

    with pytest.raises(RuntimeError, match='cannot reach MCP server'):
        await asyncio.ensure_future(web_server._startup_connect(app))

    # cleanup after a failed startup must not hang or raise
    await asyncio.ensure_future(web_server._cleanup_disconnect(app))


async def test_cleanup_without_startup_is_safe():
    """on_cleanup may run even if on_startup never populated the lifecycle state
    (e.g. startup aborted before it ran) — it must be a no-op, not a crash."""
    app = web_server.make_app(server_params=object())
    # _startup_connect never ran → no _mcp_stop / _mcp_task set
    await web_server._cleanup_disconnect(app)
