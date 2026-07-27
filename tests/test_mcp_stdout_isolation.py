"""An MCP **stdio** server must keep stdout clean for JSON-RPC.

The stdio transport speaks JSON-RPC on the process's stdout (mcp wraps
``sys.stdout.buffer``). Any stray text write to stdout — a ``print`` or a Rich
``console.print`` — lands on the wire and the client fails to parse it:

    ERROR:mcp.client.stdio:Failed to parse JSONRPC message from server
    ... input_value='Run log: /Users/.../generate-*.log'

That's exactly what broke manual onboarding under ``ariadne serve``: the
generate/onboard path prints ``Run log: ...`` to stdout via
``cli.generate.console`` (cli/generate.py:457), corrupting the stdio channel
the web server's MCP client reads. The server must reserve stdout for the
transport and route application output to stderr.
"""
from __future__ import annotations

import io
import subprocess
import sys
from pathlib import Path

from ariadne_mcp.server import reserve_stdout

_REPO_ROOT = Path(__file__).resolve().parent.parent


class _FakeStdout:
    """Stand-in for the process stdout the stdio transport speaks JSON-RPC on:
    its binary ``buffer`` is the wire; a text write onto it corrupts JSON-RPC."""

    def __init__(self) -> None:
        self.buffer = io.BytesIO()
        self.encoding = 'utf-8'

    def write(self, text: str) -> int:
        self.buffer.write(text.encode())
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def test_reserve_stdout_keeps_the_jsonrpc_channel_clean(monkeypatch):
    monkeypatch.setattr(sys, 'stdout', _FakeStdout())

    reserve_stdout()

    # The generate/onboard path prints exactly this to sys.stdout.
    from cli.generate import console
    console.print('Run log: /tmp/generate-20260725-220911.log')
    print('a stray print()')

    # None of it reached the JSON-RPC wire.
    assert sys.stdout.buffer.getvalue() == b''

    # The transport can still emit JSON-RPC on the reserved binary channel.
    sys.stdout.buffer.write(b'{"jsonrpc": "2.0", "id": 1}\n')
    assert sys.stdout.buffer.getvalue() == b'{"jsonrpc": "2.0", "id": 1}\n'


def test_stdout_isolation_holds_in_a_real_process_over_a_real_pipe():
    """The invariant that broke manual onboarding, checked over a real OS stdout
    pipe (not an in-memory fake): once the server reserves stdout, the exact
    generate/onboard console output goes to stderr, and ONLY JSON-RPC bytes
    reach stdout."""
    code = (
        'from ariadne_mcp.server import reserve_stdout\n'
        'reserve_stdout()\n'
        'import sys\n'
        'from cli.generate import console\n'
        "console.print('[dim]Run log: /tmp/generate-20260725-220911.log[/dim]')\n"
        "print('a stray print()')\n"
        'sys.stdout.buffer.write(b\'{"jsonrpc": "2.0", "id": 1}\\n\')\n'
        'sys.stdout.buffer.flush()\n'
    )
    r = subprocess.run(
        [sys.executable, '-c', code],
        capture_output=True, cwd=_REPO_ROOT,
    )
    assert r.returncode == 0, r.stderr.decode()
    # Only JSON-RPC reached the wire.
    assert r.stdout == b'{"jsonrpc": "2.0", "id": 1}\n'
    # The application output was routed to stderr instead.
    assert b'Run log' in r.stderr
    assert b'a stray print()' in r.stderr
