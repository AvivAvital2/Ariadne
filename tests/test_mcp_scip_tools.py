"""Contract for the SCIP MCP tools — Phase 2m.

Wraps ``ariadne discover`` and ``ariadne index`` as MCP tools so an
agent (Claude) can drive setup + indexing from inside a conversation.
Both tools shell out to the CLI via ``subprocess.run`` for parity with
``ariadne_improve`` (the existing pattern in mcp_server_admin).

Tests use a fake subprocess runner to verify the constructed CLI
commands without invoking the real CLI.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeSubprocess:
    """Replacement for ``subprocess.run``. Records calls; canned result."""

    def __init__(
        self,
        *,
        returncode: int = 0,
        stdout: str = '',
        stderr: str = '',
    ) -> None:
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr
        self.calls: list[dict] = []

    def __call__(self, cmd, *, capture_output=True, text=True,
                 timeout=None, **kwargs):
        self.calls.append({'cmd': list(cmd), 'timeout': timeout})
        return SimpleNamespace(
            returncode=self.returncode,
            stdout=self.stdout,
            stderr=self.stderr,
        )


# ---------------------------------------------------------------------------
# ariadne_discover_source
# ---------------------------------------------------------------------------


class TestDiscoverTool:
    @pytest.mark.asyncio
    async def test_invokes_discover_cli_with_source(self, monkeypatch):
        from ariadne_mcp.server_admin import ariadne_discover_source

        runner = FakeSubprocess(stdout='Wrote manifest.json\n')
        monkeypatch.setattr(
            'subprocess.run', runner,
        )
        response = await ariadne_discover_source(source='scalaproject')

        assert len(runner.calls) == 1
        cmd = runner.calls[0]['cmd']
        # Verify the CLI command shape
        assert 'ariadne' in cmd
        assert 'discover' in cmd
        assert 'scalaproject' in ' '.join(cmd)
        # Output captured
        assert 'manifest.json' in response.output

    @pytest.mark.asyncio
    async def test_passes_dry_run_flag(self, monkeypatch):
        from ariadne_mcp.server_admin import ariadne_discover_source

        runner = FakeSubprocess()
        monkeypatch.setattr('subprocess.run', runner)
        await ariadne_discover_source(source='scalaproject', dry_run=True)

        cmd = runner.calls[0]['cmd']
        assert '--dry-run' in cmd

    @pytest.mark.asyncio
    async def test_failure_returncode_surfaces_in_output(
        self, monkeypatch,
    ):
        from ariadne_mcp.server_admin import ariadne_discover_source

        runner = FakeSubprocess(
            returncode=1,
            stdout='', stderr='Source not found: ghost\n',
        )
        monkeypatch.setattr('subprocess.run', runner)
        response = await ariadne_discover_source(source='ghost')
        # Error output reaches the agent; the tool itself doesn't raise
        assert 'ghost' in response.output or 'Source not found' in response.output


# ---------------------------------------------------------------------------
# ariadne_index_source
# ---------------------------------------------------------------------------


class TestIndexTool:
    @pytest.mark.asyncio
    async def test_invokes_index_cli_with_source(self, monkeypatch):
        from ariadne_mcp.server_admin import ariadne_index_source

        runner = FakeSubprocess(stdout='Wrote .ariadne/index.scip\n')
        monkeypatch.setattr('subprocess.run', runner)
        response = await ariadne_index_source(source='scalaproject')

        assert len(runner.calls) == 1
        cmd = runner.calls[0]['cmd']
        assert 'index' in cmd
        assert 'scalaproject' in ' '.join(cmd)
        assert 'index.scip' in response.output

    @pytest.mark.asyncio
    async def test_passes_kind_filter(self, monkeypatch):
        from ariadne_mcp.server_admin import ariadne_index_source

        runner = FakeSubprocess()
        monkeypatch.setattr('subprocess.run', runner)
        await ariadne_index_source(source='scalaproject', kind='python')

        cmd = runner.calls[0]['cmd']
        assert '--kind' in cmd
        assert 'python' in cmd

    @pytest.mark.asyncio
    async def test_passes_dry_run_flag(self, monkeypatch):
        from ariadne_mcp.server_admin import ariadne_index_source

        runner = FakeSubprocess()
        monkeypatch.setattr('subprocess.run', runner)
        await ariadne_index_source(source='scalaproject', dry_run=True)

        cmd = runner.calls[0]['cmd']
        assert '--dry-run' in cmd
