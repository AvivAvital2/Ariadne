"""Contract test for the web onboarding backend (``web/server.py``).

The backend is a thin bridge: each ``/api`` endpoint forwards 1:1 to an MCP
tool and returns the structured result. We test that wiring against a FAKE
MCP bridge — exercising the pure ``dispatch_tool`` plus the route table —
without binding a socket (so it runs under the sandbox; the live HTTP round
trip is standard aiohttp).

A single evolving test:

  D1 — dispatch forwards args to the tool and returns its structured result.
  D2 — every /api route maps to its tool, and make_app registers them
       (POST) plus GET / for the page.
  D3 — an MCP tool error becomes HTTP 400 with an ``error`` field.
  D4 — the onboarding page is present to serve at /.
"""
from __future__ import annotations

import pytest

from web.mcp_client import MCPCallError
from web.server import (
    STATIC_DIR,
    TOOL_ROUTES,
    dispatch_tool,
    list_dirs,
    make_app,
    native_picker_command,
_onboard_command)


class _FakeBridge:
    """Records calls; returns canned structured content per tool."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_on: set[str] = set()

    async def call(self, tool: str, arguments: dict | None = None) -> dict:
        args = arguments or {}
        self.calls.append((tool, args))
        if tool in self.fail_on:
            raise MCPCallError(f'{tool} blew up')
        return {'tool': tool, 'echo': args}


async def test_backend_bridges_to_mcp_tools(tmp_path):
    bridge = _FakeBridge()

    # ---- D1: dispatch forwards args, returns the tool's structured result ----
    status, body = await dispatch_tool(
        bridge, 'ariadne_source_add', {'name': 'proj', 'path': '/x'})
    assert status == 200
    assert body == {'tool': 'ariadne_source_add',
                    'echo': {'name': 'proj', 'path': '/x'}}
    assert bridge.calls == [('ariadne_source_add', {'name': 'proj', 'path': '/x'})]

    # ---- D2: route table covers the onboarding tools; make_app wires them ----
    assert set(TOOL_ROUTES.values()) == {
        'ariadne_source_add', 'ariadne_list_sources',
        'ariadne_discover', 'ariadne_estimate'}
    for path, tool in TOOL_ROUTES.items():
        s, b = await dispatch_tool(bridge, tool, {'source': 'proj'})
        assert s == 200 and b['tool'] == tool

    app = make_app(bridge=bridge)
    registered = {(r.method, r.resource.canonical) for r in app.router.routes()}
    assert ('GET', '/') in registered
    for path in TOOL_ROUTES:
        assert ('POST', path) in registered
    assert app['bridge'] is bridge

    # ---- D3: a tool error maps to HTTP 400 + an error field -----------------
    bridge.fail_on = {'ariadne_estimate'}
    status, body = await dispatch_tool(bridge, 'ariadne_estimate', {'source': 'proj'})
    assert status == 400
    assert 'error' in body and 'ariadne_estimate' in body['error']

    # ---- D4: the onboarding page exists to serve at / -----------------------
    page = STATIC_DIR / 'onboarding.html'
    assert page.exists()
    assert page.read_text(encoding='utf-8').lower().lstrip().startswith('<!doctype html>')

    # ---- D5: the Browse directory picker lists local subdirs ---------------
    (tmp_path / 'alpha').mkdir()
    (tmp_path / 'beta').mkdir()
    (tmp_path / '.hidden').mkdir()
    listing = list_dirs(str(tmp_path))
    assert listing['path'] == str(tmp_path.resolve())
    assert listing['dirs'] == ['alpha', 'beta']  # sorted, dot-dirs omitted
    assert listing['parent'] == str(tmp_path.resolve().parent)
    assert ('GET', '/api/browse') in registered

    # ---- D6: the native OS folder picker is wired for this platform --------
    import sys
    cmd = native_picker_command(None)
    if sys.platform in ('darwin', 'win32') or sys.platform.startswith('linux'):
        assert cmd and cmd[0] in ('osascript', 'zenity', 'powershell')
    assert ('GET', '/api/pick-folder') in registered
    assert _onboard_command({}) is None  # no source → no command
    assert _onboard_command({'source': 'proj'}) == [
        'onboard', '--source', 'proj', '--approve', '--live']
    assert _onboard_command({
        'source': 'proj', 'model': 'claude-opus-4-8', 'batch': True,
        'types': ['explanation', 'qa'],
    }) == [
        'onboard', '--source', 'proj', '--approve',
        '--model', 'claude-opus-4-8', '--batch', '--types', 'explanation,qa']
    assert ('POST', '/api/onboard') in registered
    assert ('GET', '/api/onboard/status') in registered
