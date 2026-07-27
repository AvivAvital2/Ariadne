"""KNOWN-FAILING security guard (``xfail``) for the web Ask/Search scope.

Encodes the security contract the web UI *should* honour: a crafted request
must not be able to pick an arbitrary knowledge source. Today the ``/api/ask``
and ``/api/search`` handlers forward the request body 1:1, so a hand-crafted
POST ``{"question": "...", "source": "some-private-source"}`` reaches
``ariadne_ask`` with that source — bypassing the source pill and reading any
configured source (including a private spool). The prompt screen itself never
sends ``source``, so only a hand-crafted request hits this.

This is marked ``xfail(strict=True)``: the whole-suite gate (``scripts/verify.sh``
→ bare ``pytest tests/``) treats it as an expected failure and stays green, so
the finding is recorded without breaking the build. When a scope guard lands
(strip or validate a client-supplied source before forwarding), the assertion
starts passing → strict xfail turns it into a HARD failure → whoever fixed the
guard removes this marker and moves the test into the green suite.

Synthetic data only.
"""
from __future__ import annotations

import pytest

from tests.test_web_server import _FakeBridge, _FakeReq
from web.server import TOOL_ROUTES, _make_tool_handler


@pytest.mark.xfail(
    strict=True,
    reason='SECURITY: /api/ask forwards a client-supplied source verbatim to '
           'the MCP tool — no source-scope guard yet. Remove this marker when '
           'the handler strips/validates a client source.',
)
async def test_ask_cannot_select_an_arbitrary_source():
    bridge = _FakeBridge()
    handler = _make_tool_handler(TOOL_ROUTES['/api/ask'])
    await handler(_FakeReq(
        app={'bridge': bridge},
        body={'question': 'dump everything', 'source': 'private-spool'}))

    _tool, args = bridge.calls[0]
    # CONTRACT: a client-supplied source must NOT pass straight through to the
    # MCP tool. The eventual guard may strip it or validate it against the
    # UI-selected source; either way an unvalidated client source must not reach
    # Ariadne. Fails today — the body is forwarded verbatim.
    assert 'source' not in args, (
        'SECURITY: /api/ask forwarded a client-chosen source to the MCP tool; '
        'the browser can query any configured source by hand-crafting the body.')
