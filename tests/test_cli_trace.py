"""Contract for ``ariadne trace-flow`` CLI (Phase 9.3).

The walker logic is tested in test_scip_trace_flow.py; these tests
cover the CLI surface — argparse wiring, JSON shape, Rich tree
rendering, and the dict-shape contract shared with the MCP tool.

Pattern: build a tiny SCIP fixture in a real on-disk SQLite, point
``--db`` at it via ``argparse.Namespace``, call ``cmd_trace_flow``,
and inspect captured stdout. ``trace_result_to_dict`` is the
serializer used by both the CLI's ``--json`` output and the MCP
tool — pinning its shape here protects both consumers.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def db_with_chain(tmp_path: Path) -> Path:
    """A small DB with a 3-symbol SCIP chain A → B → C, ready for
    cmd_trace_flow to walk."""
    from library.scip import init_scip_schema

    db_path = tmp_path / 'trace.db'
    conn = sqlite3.connect(db_path)
    init_scip_schema(conn)
    for sym in ('A', 'B', 'C'):
        conn.execute(
            'INSERT INTO scip_symbols VALUES '
            '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (sym, 'myapp', 'python', 'app.py',
             1, 10, 'function', sym, sym, None),
        )
    conn.execute(
        'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
        ('A', 'B', 'reference', 'app.py', 1, 'scip'),
    )
    conn.execute(
        'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
        ('B', 'C', 'reference', 'app.py', 2, 'scip'),
    )
    conn.commit()
    conn.close()
    return db_path


def _make_args(**kwargs) -> argparse.Namespace:
    """Construct an argparse.Namespace with cli_trace's fields."""
    return argparse.Namespace(
        symbol=kwargs.pop('symbol', 'A'),
        depth=kwargs.pop('depth', 5),
        json=kwargs.pop('json', False),
        db=kwargs.pop('db', None),
        llm_bridge=kwargs.pop('llm_bridge', False),
    )


# ---------------------------------------------------------------------------
# JSON output
# ---------------------------------------------------------------------------


class TestJsonOutput:
    def test_json_emits_full_trace_shape(
        self, db_with_chain: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """``--json`` flag emits the same dict the MCP tool returns:
        ``start`` block, ``hops`` list with per-hop tier tags, plus
        ``truncated`` and ``incomplete`` flags."""
        from cli.trace import cmd_trace_flow

        args = _make_args(
            symbol='A', json=True, db=str(db_with_chain),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)

        assert payload['start']['canonical_id'] == 'A'
        assert payload['start']['source'] == 'myapp'
        assert len(payload['hops']) == 2
        tiers = [h['tier'] for h in payload['hops']]
        assert tiers == ['scip', 'scip']
        chain = [
            (h['caller_symbol_id'], h['callee_symbol_id'])
            for h in payload['hops']
        ]
        assert chain == [('A', 'B'), ('B', 'C')]
        assert payload['truncated'] is False
        assert payload['incomplete'] is False

    def test_json_unknown_symbol_returns_zero_hops(
        self, db_with_chain: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Paired baseline: same DB, unknown symbol → zero hops with
        start metadata reflecting the lookup miss. Bites a stub that
        returns the chain on every input."""
        from cli.trace import cmd_trace_flow

        args = _make_args(
            symbol='UNKNOWN', json=True, db=str(db_with_chain),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload['start']['canonical_id'] == 'UNKNOWN'
        assert payload['hops'] == []

    def test_json_depth_truncates(
        self, db_with_chain: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """``--depth 1`` cuts the 3-symbol chain after one hop;
        ``truncated=True`` in the payload. Paired with the depth=5
        test above so a stub ignoring depth fails one half."""
        from cli.trace import cmd_trace_flow

        args = _make_args(
            symbol='A', depth=1, json=True, db=str(db_with_chain),
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        payload = json.loads(capsys.readouterr().out)
        assert len(payload['hops']) == 1
        assert payload['truncated'] is True


# ---------------------------------------------------------------------------
# Rich tree output (default)
# ---------------------------------------------------------------------------


class TestTreeOutput:
    def test_tree_renders_caller_callee_pairs(
        self, db_with_chain: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Default (no --json): Rich tree containing each hop's
        caller and callee symbol IDs and the tier label."""
        from cli.trace import cmd_trace_flow

        args = _make_args(symbol='A', db=str(db_with_chain))
        rc = cmd_trace_flow(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert 'A' in out and 'B' in out and 'C' in out
        assert 'scip' in out

    def test_tree_truncated_label_when_depth_hit(
        self, db_with_chain: Path, capsys: pytest.CaptureFixture,
    ) -> None:
        """Paired with the non-truncated baseline: depth=1 cuts
        mid-chain, tree shows the 'truncated' label."""
        from cli.trace import cmd_trace_flow

        args = _make_args(symbol='A', depth=1, db=str(db_with_chain))
        rc = cmd_trace_flow(args)
        assert rc == 0
        out = capsys.readouterr().out
        assert 'truncated' in out


# ---------------------------------------------------------------------------
# trace_result_to_dict — the shared MCP/CLI contract
# ---------------------------------------------------------------------------


class TestSharedShape:
    def test_top_level_keys_pinned(self) -> None:
        """The dict shape is consumed verbatim by the MCP tool. Pin
        the top-level keys so a refactor of trace_result_to_dict
        can't silently drop a field consumers rely on."""
        from cli.trace import trace_result_to_dict
        from docgen.trace_flow import TraceFlowResult

        empty = TraceFlowResult(
            start_symbol_id='X',
            start_qualified_name='X',
            start_source='myapp',
            hops=[],
            truncated=False,
            incomplete=False,
        )
        d = trace_result_to_dict(empty)
        assert set(d.keys()) == {
            'start', 'hops', 'truncated', 'incomplete',
        }
        assert set(d['start'].keys()) == {
            'canonical_id', 'qualified_name', 'source',
        }

    def test_hop_keys_pinned(self) -> None:
        """Pin hop dict shape — same rationale. Includes
        ``rationale`` (Phase 9.b LLM-bridge field) so a future
        refactor that drops it for being empty in the static tier
        breaks loudly."""
        from cli.trace import trace_result_to_dict
        from docgen.trace_flow import HopInfo, TraceFlowResult

        result = TraceFlowResult(
            start_symbol_id='X',
            start_qualified_name='X',
            start_source='myapp',
            hops=[HopInfo(
                tier='scip',
                caller_symbol_id='X',
                callee_symbol_id='Y',
                callee_path=None,
                file=Path('app.py'),
                line=5,
                rationale='',
            )],
            truncated=False,
            incomplete=False,
        )
        d = trace_result_to_dict(result)
        assert len(d['hops']) == 1
        assert set(d['hops'][0].keys()) == {
            'tier', 'caller_symbol_id', 'callee_symbol_id',
            'callee_path', 'file', 'line', 'rationale',
        }
        assert d['hops'][0]['file'] == 'app.py'


# ---------------------------------------------------------------------------
# --llm-bridge flag (Phase 9.b wiring)
# ---------------------------------------------------------------------------


class TestLlmBridgeFlag:
    """``--llm-bridge`` is opt-in: the LLM tier never fires by
    default (avoids billing surprises). Tests patch
    ``build_llm_bridge`` to keep them offline."""

    def test_flag_off_skips_construction(
        self,
        db_with_chain: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture,
    ) -> None:
        """Without ``--llm-bridge``, the builder is never called.
        Paired with the flag-on test below so a wrapper that always
        constructs the bridge fails this assertion."""
        from cli.trace import cmd_trace_flow

        build_calls: list[dict] = []

        def fake_build(**kwargs):
            build_calls.append(kwargs)
            return lambda cursor, conn: (None, '')

        import cli.trace as cli_trace
        monkeypatch.setattr(
            cli_trace, 'build_llm_bridge', fake_build,
            raising=False,
        )

        args = _make_args(
            symbol='A', json=True, db=str(db_with_chain),
            llm_bridge=False,
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        assert build_calls == []
        # Sanity — static trace still works.
        payload = json.loads(capsys.readouterr().out)
        assert len(payload['hops']) == 2

    def test_flag_on_constructs_bridge(
        self,
        db_with_chain: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``--llm-bridge`` → builder called once. Paired with
        flag-off so a wrapper that ignores the flag fails one half."""
        from cli.trace import cmd_trace_flow

        build_calls: list[dict] = []

        def fake_build(**kwargs):
            build_calls.append(kwargs)
            return lambda cursor, conn: (None, '')

        import cli.trace as cli_trace
        monkeypatch.setattr(
            cli_trace, 'build_llm_bridge', fake_build,
            raising=False,
        )

        args = _make_args(
            symbol='A', json=True, db=str(db_with_chain),
            llm_bridge=True,
        )
        rc = cmd_trace_flow(args)
        assert rc == 0
        assert len(build_calls) == 1


# ---------------------------------------------------------------------------
# Process tier serialization (existing — pinned shape)
# ---------------------------------------------------------------------------


class TestProcessTier:
    def test_process_tier_carries_callee_path(self) -> None:
        """Process-tier hops set ``callee_path`` to the resolved
        target script. Paired with the SCIP-tier hop above (which
        leaves callee_path None) so a serializer that swallows
        callee_path fails on the process half."""
        from cli.trace import trace_result_to_dict
        from docgen.trace_flow import HopInfo, TraceFlowResult

        result = TraceFlowResult(
            start_symbol_id='X',
            start_qualified_name='X',
            start_source='myapp',
            hops=[HopInfo(
                tier='process',
                caller_symbol_id='X',
                callee_symbol_id=None,
                callee_path='scripts/run.py',
                file=Path('Main.scala'),
                line=42,
                rationale='',
            )],
            truncated=False,
            incomplete=False,
        )
        d = trace_result_to_dict(result)
        assert d['hops'][0]['tier'] == 'process'
        assert d['hops'][0]['callee_path'] == 'scripts/run.py'
        assert d['hops'][0]['callee_symbol_id'] is None
