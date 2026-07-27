"""Transport selection for ``ariadne mcp`` — stdio by default, streamable-http
under ``--http`` (Phase B: what makes "remote" real).

``_serve_mcp`` is the pure seam: given a FastMCP-shaped object and the parsed
flags, it sets host/port + picks the transport, without binding a socket.
"""
from __future__ import annotations

from cli.integration import _serve_mcp


class _FakeSettings:
    host = None
    port = None


class _FakeMCP:
    """Records the transport run() was asked for + any host/port set."""

    def __init__(self) -> None:
        self.settings = _FakeSettings()
        self.ran: str | None = None

    def run(self, transport: str) -> None:
        self.ran = transport


def test_serve_mcp_defaults_to_stdio() -> None:
    mcp = _FakeMCP()
    _serve_mcp(mcp, http=False, host='0.0.0.0', port=9000)
    assert mcp.ran == 'stdio'
    # stdio never touches the http host/port
    assert mcp.settings.host is None
    assert mcp.settings.port is None


def test_serve_mcp_http_sets_host_port_and_runs_streamable() -> None:
    mcp = _FakeMCP()
    _serve_mcp(mcp, http=True, host='0.0.0.0', port=9000)
    assert mcp.ran == 'streamable-http'
    assert mcp.settings.host == '0.0.0.0'
    assert mcp.settings.port == 9000
