"""Contract for ``ariadne trace-flow`` CLI command (Phase 9 wiring).

Pure-function ``trace_flow`` was tested in
``tests/test_scip_trace_flow.py``. This file tests the CLI handler:

- ``cmd_trace_flow(args)`` invokes the pure function with the right
  parameters and prints output.
- ``--json`` emits a structured object suitable for the MCP tool to
  parse and return verbatim.
- Default (non-JSON) output is human-readable; we don't pin the exact
  format (Rich rendering may evolve), only that something useful
  reaches stdout.

These tests are RED until ``cli_trace.py`` exists and registers
itself with the CLI dispatcher.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    """Create an ariadne.db with init_scip_schema applied. Tests
    seed it with whatever symbols/edges they need."""
    from library.scip import init_scip_schema

    p = tmp_path / 'ariadne.db'
    conn = sqlite3.connect(p)
    init_scip_schema(conn)
    conn.commit()
    conn.close()
    return p


def _seed_chain(
    db_path: Path,
    chain: list[tuple[str, str]],
    *,
    source_name: str = 'myapp',
) -> None:
    """Seed scip_symbols + scip_edges for a simple chain like
    [('A', 'B'), ('B', 'C')]. Each pair becomes one symbol pair +
    one edge."""
    conn = sqlite3.connect(db_path)
    seen: set[str] = set()
    for caller, callee in chain:
        for sym in (caller, callee):
            if sym not in seen:
                conn.execute(
                    'INSERT INTO scip_symbols VALUES '
                    '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                    (
                        sym, source_name, 'python', 'app.py',
                        1, 10, 'function', sym, sym, None,
                    ),
                )
                seen.add(sym)
        conn.execute(
            'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
            (caller, callee, 'reference', 'app.py', 5, 'scip'),
        )
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# JSON output — structured trace for MCP consumption
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_returns_zero_on_success(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        from cli.trace import cmd_trace_flow

        _seed_chain(db_path, [('A', 'B')])
        args = SimpleNamespace(
            symbol='A', depth=5, json=True, db=str(db_path),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0

    def test_output_is_valid_json(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        from cli.trace import cmd_trace_flow

        _seed_chain(db_path, [('A', 'B')])
        args = SimpleNamespace(
            symbol='A', depth=5, json=True, db=str(db_path),
        )
        cmd_trace_flow(args)
        out = capsys.readouterr().out
        # Must parse as JSON without error
        json.loads(out)

    def test_includes_start_metadata(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        from cli.trace import cmd_trace_flow

        _seed_chain(db_path, [('A', 'B')])
        args = SimpleNamespace(
            symbol='A', depth=5, json=True, db=str(db_path),
        )
        cmd_trace_flow(args)
        data = json.loads(capsys.readouterr().out)

        assert 'start' in data
        start = data['start']
        assert start['canonical_id'] == 'A'
        assert start['qualified_name'] == 'A'
        assert start['source'] == 'myapp'

    def test_includes_hops_with_tiers(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Each hop in the JSON output has a ``tier`` field plus
        caller/callee identifiers — what the MCP tool returns to
        agents."""
        from cli.trace import cmd_trace_flow

        _seed_chain(db_path, [('A', 'B'), ('B', 'C')])
        args = SimpleNamespace(
            symbol='A', depth=5, json=True, db=str(db_path),
        )
        cmd_trace_flow(args)
        data = json.loads(capsys.readouterr().out)

        assert 'hops' in data
        assert len(data['hops']) == 2
        for hop in data['hops']:
            assert 'tier' in hop
            assert 'caller_symbol_id' in hop
            assert 'callee_symbol_id' in hop
            assert 'file' in hop
            assert 'line' in hop
        assert all(h['tier'] == 'scip' for h in data['hops'])

    def test_truncated_flag_present(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Long chain + low depth → ``truncated: true`` in output."""
        from cli.trace import cmd_trace_flow

        _seed_chain(
            db_path,
            [('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E')],
        )
        args = SimpleNamespace(
            symbol='A', depth=2, json=True, db=str(db_path),
        )
        cmd_trace_flow(args)
        data = json.loads(capsys.readouterr().out)
        assert data['truncated'] is True

    def test_unknown_symbol_returns_empty_hops(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Unknown symbols don't error — return empty hops with
        start metadata reflecting the unknown lookup."""
        from cli.trace import cmd_trace_flow

        args = SimpleNamespace(
            symbol='nonexistent', depth=5, json=True,
            db=str(db_path),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        data = json.loads(capsys.readouterr().out)
        assert data['hops'] == []
        assert data['start']['canonical_id'] == 'nonexistent'


# ---------------------------------------------------------------------------
# Default (non-JSON) output — human-readable
# ---------------------------------------------------------------------------


class TestDefaultOutput:
    def test_default_renders_something_to_stdout(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Without ``--json`` the command still prints a trace.
        We don't pin the exact format (Rich rendering may evolve);
        we only verify the start symbol or 'no hops' message reaches
        stdout."""
        from cli.trace import cmd_trace_flow

        _seed_chain(db_path, [('A', 'B')])
        args = SimpleNamespace(
            symbol='A', depth=5, json=False, db=str(db_path),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        out = capsys.readouterr().out
        # Symbol or hop info should appear somewhere
        assert 'A' in out or 'B' in out

    def test_empty_trace_doesnt_crash(
        self, db_path: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Default output with no hops should still terminate
        gracefully (no errors, exit 0)."""
        from cli.trace import cmd_trace_flow

        # No symbols seeded; lookup is empty
        args = SimpleNamespace(
            symbol='nothing', depth=5, json=False, db=str(db_path),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0


# ---------------------------------------------------------------------------
# CLI registration — handler dispatch
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_register_commands_adds_trace_flow_subparser(self) -> None:
        """``cli_trace.register_commands`` adds a ``trace-flow``
        subparser following the existing CLI pattern (cli_callers,
        cli_graph)."""
        import argparse
        from cli.trace import register_commands

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest='cmd')
        register_commands(sub)

        # Parsing 'trace-flow A' shouldn't error
        ns = parser.parse_args(['trace-flow', 'A'])
        assert ns.cmd == 'trace-flow'
        assert ns.symbol == 'A'

    def test_handlers_dict_exposes_trace_flow(self) -> None:
        from cli.trace import HANDLERS

        assert 'trace-flow' in HANDLERS
        # The handler is callable
        assert callable(HANDLERS['trace-flow'])
