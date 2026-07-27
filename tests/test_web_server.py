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

import asyncio
import json

import pytest

from web.mcp_client import MCPCallError
from web.server import (
    STATIC_DIR,
    TOOL_ROUTES,
    dispatch_tool,
    list_dirs,
    make_app,
    native_picker_command,
    _feedback_call,
    _make_tool_handler,
    _onboard_start,
    _onboard_tool_args,
    _read_json,
    _run_onboard,
)


class _FakeBridge:
    """Records calls; returns canned structured content per tool."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []
        self.fail_on: set[str] = set()

    async def call(self, tool: str, arguments: dict | None = None,
                   *, progress_callback=None) -> dict:
        args = arguments or {}
        self.calls.append((tool, args))
        if tool in self.fail_on:
            raise MCPCallError(f'{tool} blew up')
        if progress_callback is not None:
            # mimic ariadne_onboard streaming ctx.report_progress
            await progress_callback(1, 3, 'Describing catalog elements')
            await progress_callback(2, 3, 'Generating documentation')
        return {'tool': tool, 'echo': args}


class _FakeReq:
    """Minimal stand-in for ``aiohttp.web.Request`` that drives a real handler
    without binding a socket (loopback bind is unavailable under the sandbox).

    Provides only what the handlers touch: ``.app`` (the bridge + jobs map), a
    JSON body, and ``can_read_body``. ``bad_json=True`` makes ``.json()`` raise,
    to exercise the malformed-body path.
    """

    def __init__(self, *, app, body=None, bad_json=False, can_read_body=True):
        self.app = app
        self._body = {} if body is None else body
        self._bad = bad_json
        self.can_read_body = can_read_body

    async def json(self):
        if self._bad:
            raise ValueError('malformed JSON')
        return self._body


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
        'ariadne_discover', 'ariadne_estimate',
        'ariadne_ask', 'ariadne_search'}
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

    # ---- D7: the "Generate" step maps the build request to ariadne_onboard
    # tool args, served as POST /api/onboard + an SSE events stream ----------
    assert _onboard_tool_args({'source': 'proj'}) == {'source': 'proj'}
    assert _onboard_tool_args({
        'source': 'proj', 'model': 'claude-opus-4-8', 'batch': True,
        'types': ['explanation', 'qa'], 'concurrency': 6,
    }) == {
        'source': 'proj', 'model': 'claude-opus-4-8', 'batch': True,
        'doc_types': ['explanation', 'qa'], 'concurrency': 6,
    }
    assert ('POST', '/api/onboard') in registered
    assert ('GET', '/api/onboard/events') in registered

    # ---- D8: _run_onboard relays MCP progress + the terminal result into a
    # queue as SSE-shaped events (progress… then exactly one done) ----------
    q: asyncio.Queue = asyncio.Queue()
    await _run_onboard(bridge, {'source': 'proj'}, q)
    events = []
    while not q.empty():
        events.append(q.get_nowait())
    assert [e['type'] for e in events] == ['progress', 'progress', 'done']
    assert events[0] == {'type': 'progress', 'current': 1, 'total': 3,
                         'message': 'Describing catalog elements'}
    assert events[-1]['result']['tool'] == 'ariadne_onboard'
    assert ('ariadne_onboard', {'source': 'proj'}) in bridge.calls

    # a tool failure surfaces as a single error event (the stream never hangs)
    bridge.fail_on = {'ariadne_onboard'}
    q_err: asyncio.Queue = asyncio.Queue()
    await _run_onboard(bridge, {'source': 'proj'}, q_err)
    err_events = []
    while not q_err.empty():
        err_events.append(q_err.get_nowait())
    assert err_events[-1]['type'] == 'error'
    assert 'ariadne_onboard' in err_events[-1]['error']


async def test_transport_selection(monkeypatch):
    """Phase B: the backend connects over stdio by default, or to a remote
    streamable-http MCP URL when one is given (``ariadne serve --mcp-url``)."""
    import web.server as server

    called: dict = {}

    async def fake_http(stack, url):
        called['http'] = url
        return 'http-bridge'

    async def fake_stdio(stack, params):
        called['stdio'] = params
        return 'stdio-bridge'

    monkeypatch.setattr(server, 'connect_http', fake_http)
    monkeypatch.setattr(server, 'connect_stdio', fake_stdio)

    # stdio path: no mcp_url → connect_stdio with the server params
    app = server.make_app(server_params='PARAMS')
    assert app.get('_mcp_url') is None
    await server._startup_connect(app)
    assert called == {'stdio': 'PARAMS'}
    assert app['bridge'] == 'stdio-bridge'

    # remote path: an mcp_url → connect_http with that url
    called.clear()
    app2 = server.make_app(mcp_url='http://remote:8000/mcp')
    assert app2['_mcp_url'] == 'http://remote:8000/mcp'
    await server._startup_connect(app2)
    assert called == {'http': 'http://remote:8000/mcp'}
    assert app2['bridge'] == 'http-bridge'


async def test_ask_search_feedback_endpoints():
    """Ask + Search are 1:1 tool forwards (auto-registered via TOOL_ROUTES);
    /api/feedback routes 👍 → log_hit, 👎 → log_miss by event_id."""
    bridge = _FakeBridge()
    app = make_app(bridge=bridge)
    registered = {(r.method, r.resource.canonical) for r in app.router.routes()}

    # ask + search are pure forwards, wired as POST routes
    assert TOOL_ROUTES['/api/ask'] == 'ariadne_ask'
    assert TOOL_ROUTES['/api/search'] == 'ariadne_search'
    assert ('POST', '/api/ask') in registered
    assert ('POST', '/api/search') in registered
    assert ('POST', '/api/feedback') in registered

    # dispatch forwards the browser body straight to the tool (question/role)
    status, body = await dispatch_tool(
        bridge, 'ariadne_ask', {'question': 'how does X work?', 'role': 'developer'})
    assert status == 200
    assert body['echo'] == {'question': 'how does X work?', 'role': 'developer'}

    # feedback tool-routing: helpful → log_hit; else → log_miss; feedback optional
    assert _feedback_call({'event_id': 5, 'helpful': True}) == (
        'ariadne_log_hit', {'event_id': 5})
    assert _feedback_call({'event_id': 7, 'helpful': False, 'feedback': 'wrong'}) == (
        'ariadne_log_miss', {'event_id': 7, 'feedback': 'wrong'})
    assert _feedback_call({'event_id': 9, 'helpful': False}) == (
        'ariadne_log_miss', {'event_id': 9})


# ---------------------------------------------------------------------------
# Direct connection: the UI talks to Ariadne straight over MCP, every time.
# ---------------------------------------------------------------------------
async def test_ask_is_a_direct_single_forward():
    """Every UI ask reaches Ariadne DIRECTLY: the ``/api/ask`` handler makes
    exactly ONE MCP call — to ``ariadne_ask`` — and returns its raw structured
    result verbatim. Nothing sits between the user and Ariadne deciding whether
    to consult it (contrast the Claude-Code path, where using Ariadne is at the
    agent's discretion)."""
    bridge = _FakeBridge()
    handler = _make_tool_handler(TOOL_ROUTES['/api/ask'])
    body = {'question': 'how are retries handled?', 'role': 'developer'}
    resp = await handler(_FakeReq(app={'bridge': bridge}, body=body))

    assert resp.status == 200
    # exactly one call — unconditional, no branch that could skip Ariadne
    assert bridge.calls == [('ariadne_ask', body)]
    # the tool's structured result is returned as-is (no summarising middle layer)
    assert json.loads(resp.body) == {'tool': 'ariadne_ask', 'echo': body}


# ---------------------------------------------------------------------------
# Attack surface: the browser cannot exploit Ariadne "over MCP" through the UI.
# ---------------------------------------------------------------------------
async def test_route_tool_is_authoritative_over_body():
    """A crafted request body can't redirect a route to a different MCP tool.
    Posting a body that NAMES another tool (plus mutating-tool arguments) to
    ``/api/ask`` still invokes ONLY ``ariadne_ask`` — the extra keys ride along
    as inert arguments, never as tool selection."""
    bridge = _FakeBridge()
    malicious = {'question': 'hi', 'tool': 'ariadne_generate',
                 'name': 'evil', 'path': '/etc', 'source': 'other'}
    handler = _make_tool_handler(TOOL_ROUTES['/api/ask'])
    await handler(_FakeReq(app={'bridge': bridge}, body=malicious))
    assert [t for t, _ in bridge.calls] == ['ariadne_ask']  # never ariadne_generate

    # Every route's tool is fixed by the route table, independent of the body.
    bridge2 = _FakeBridge()
    for _path, tool in TOOL_ROUTES.items():
        h = _make_tool_handler(tool)
        await h(_FakeReq(app={'bridge': bridge2}, body={'tool': 'ariadne_generate'}))
    invoked = {t for t, _ in bridge2.calls}
    assert invoked == set(TOOL_ROUTES.values())
    assert 'ariadne_generate' not in invoked


def test_browser_reachable_tools_are_a_safe_whitelist():
    """The browser can reach ONLY a fixed, safe set of MCP tools — the read +
    onboarding surface — never destructive or expensive maintenance tools. This
    is the boundary that keeps 'the UI does not expose MCP' true: wiring a route
    for e.g. ``ariadne_generate`` would trip this."""
    reachable = set(TOOL_ROUTES.values())
    reachable.add('ariadne_onboard')                      # POST /api/onboard
    for helpful in (True, False):                         # POST /api/feedback
        reachable.add(_feedback_call({'event_id': 1, 'helpful': helpful})[0])

    assert reachable == {
        'ariadne_source_add', 'ariadne_list_sources', 'ariadne_discover',
        'ariadne_estimate', 'ariadne_ask', 'ariadne_search',
        'ariadne_onboard', 'ariadne_log_hit', 'ariadne_log_miss',
    }
    # None of these destructive / expensive tools may be browser-reachable.
    forbidden = {'ariadne_generate', 'ariadne_generate_docs', 'ariadne_merge',
                 'ariadne_index_source', 'ariadne_contribute',
                 'ariadne_improve', 'ariadne_self_improve'}
    assert reachable.isdisjoint(forbidden)


async def test_malformed_body_is_contained():
    """A malformed / non-JSON / absent request body degrades to an empty args
    dict — the tool is still called (with ``{}``), never a 500 or a crash — and
    a tool-side error surfaces as a clean 400."""
    assert await _read_json(_FakeReq(app={}, bad_json=True)) == {}
    assert await _read_json(_FakeReq(app={}, can_read_body=False)) == {}

    bridge = _FakeBridge()
    handler = _make_tool_handler('ariadne_ask')
    resp = await handler(_FakeReq(app={'bridge': bridge}, bad_json=True))
    assert resp.status == 200
    assert bridge.calls == [('ariadne_ask', {})]

    # a tool that errors on the empty args yields a clean 400, not an unhandled 500
    bridge.fail_on = {'ariadne_ask'}
    bridge.calls.clear()
    resp = await handler(_FakeReq(app={'bridge': bridge}, bad_json=True))
    assert resp.status == 400
    assert 'error' in json.loads(resp.body)


async def test_onboard_start_requires_source_before_spawning_a_build():
    """A build cannot be kicked off without a source: ``/api/onboard`` returns
    400 and spawns no job / tool call when ``source`` is missing — so a crafted
    empty POST can't trigger a paid build."""
    bridge = _FakeBridge()
    app = {'bridge': bridge, 'jobs': {}}
    resp = await _onboard_start(_FakeReq(app=app, body={}))
    assert resp.status == 400
    assert app['jobs'] == {}       # no job registered
    assert bridge.calls == []      # ariadne_onboard never called


def test_browse_is_listing_only_and_never_reads_file_contents(tmp_path):
    """The Browse picker lists directory NAMES only — never file contents — and
    fails soft (never raises) on a bad / greedy path, so it can't be turned into
    an arbitrary file read through the UI."""
    (tmp_path / 'sub').mkdir()
    (tmp_path / 'secret.txt').write_text('TOP SECRET VALUE')
    listing = list_dirs(str(tmp_path))
    assert listing['dirs'] == ['sub']                       # dirs only; file absent
    assert 'TOP SECRET VALUE' not in json.dumps(listing)    # no content leak

    # a non-existent / traversal-y path fails soft to a real listing, no raise
    weird = list_dirs('/nonexistent/../../does-not-exist-xyz')
    assert set(weird) == {'path', 'parent', 'dirs'}
    assert isinstance(weird['dirs'], list)
