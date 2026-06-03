"""Contract for ``ariadne_trace_flow`` MCP tool wrapper (Phase 9.2).

The walker logic lives in ``docgen/trace_flow.py`` and is tested in
``test_scip_trace_flow.py``; the dict shape is tested in
``test_cli_trace.py::TestSharedShape``. These tests pin the MCP
wrapper's contract:

- Reads ``db_path`` from the global config (so production callers
  don't need to pass it).
- Returns the same JSON dict the CLI's ``--json`` emits (consumers
  depend on identical shape across CLI and MCP surfaces).
- Honors the ``depth`` argument.
- Passes ``start_symbol`` through unchanged for unknown symbols.

The wrapper imports ``get_config`` lazily inside the function body,
so monkeypatching ``config.get_config`` works without import-order
gymnastics. Tests use ``asyncio.run`` to drive the async wrapper —
no pytest-asyncio plugin dependency.
"""
from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest


def _seed_chain_db(db_path: Path) -> None:
    """A → B SCIP edge in a fresh DB."""
    from library.scip import init_scip_schema

    conn = sqlite3.connect(db_path)
    init_scip_schema(conn)
    for sym in ('A', 'B'):
        conn.execute(
            'INSERT INTO scip_symbols VALUES '
            '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (sym, 'myapp', 'python', 'app.py',
             1, 10, 'function', sym, sym, None),
        )
    conn.execute(
        'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
        ('A', 'B', 'reference', 'app.py', 7, 'scip'),
    )
    conn.commit()
    conn.close()


@pytest.fixture
def db_with_simple_edge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Seed an A → B DB and patch ``config.get_config`` to point at
    it. The wrapper does ``from config import get_config`` inside its
    body, so the patch takes effect on the next call."""
    db_path = tmp_path / 'mcp_trace.db'
    _seed_chain_db(db_path)

    cfg_stub = SimpleNamespace(db_path=str(db_path))
    monkeypatch.setattr(
        'config.get_config', lambda: cfg_stub,
    )
    return db_path


# ---------------------------------------------------------------------------
# Wrapper contract
# ---------------------------------------------------------------------------


class TestMcpWrapper:
    def test_returns_dict_with_one_scip_hop(
        self, db_with_simple_edge: Path,
    ) -> None:
        """A → B in the DB → wrapper returns the dict shape with
        exactly one 'scip' hop. Asserts both the dict shape and the
        tier-correctness so a wrapper that silently mistags its
        hops fails."""
        from ariadne_mcp.server_admin import ariadne_trace_flow

        result = asyncio.run(
            ariadne_trace_flow(start_symbol='A', depth=5),
        )
        assert result['start']['canonical_id'] == 'A'
        assert result['start']['source'] == 'myapp'
        assert len(result['hops']) == 1
        hop = result['hops'][0]
        assert hop['tier'] == 'scip'
        assert hop['caller_symbol_id'] == 'A'
        assert hop['callee_symbol_id'] == 'B'
        assert hop['line'] == 7
        assert result['truncated'] is False

    def test_depth_zero_returns_no_hops(
        self, db_with_simple_edge: Path,
    ) -> None:
        """``depth=0`` → no walk, no hops. Paired with the depth=5
        case above to bite a wrapper that ignores its depth arg
        (e.g., always passes a hard-coded default)."""
        from ariadne_mcp.server_admin import ariadne_trace_flow

        result = asyncio.run(
            ariadne_trace_flow(start_symbol='A', depth=0),
        )
        assert result['hops'] == []

    def test_unknown_symbol_passes_through(
        self, db_with_simple_edge: Path,
    ) -> None:
        """Unknown symbol → empty hops; ``start.canonical_id``
        echoes the input so callers can correlate. Paired with
        the resolved-symbol case above so a stub returning the
        same canned response fails one half."""
        from ariadne_mcp.server_admin import ariadne_trace_flow

        result = asyncio.run(
            ariadne_trace_flow(start_symbol='NONEXISTENT', depth=3),
        )
        assert result['start']['canonical_id'] == 'NONEXISTENT'
        assert result['hops'] == []

    def test_enable_llm_bridge_default_false_no_construction(
        self,
        db_with_simple_edge: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``enable_llm_bridge`` defaults to False — the LLM tier is
        opt-in to avoid surprise LLM costs. Builder is never called.
        Paired with the True case below so a wrapper that always
        constructs the bridge fails this assertion."""
        from ariadne_mcp.server_admin import ariadne_trace_flow

        build_calls: list[dict] = []

        def fake_build(**kwargs):
            build_calls.append(kwargs)
            return lambda cursor, conn: (None, '')

        import ariadne_mcp.server_admin as mcp_server_admin
        monkeypatch.setattr(
            mcp_server_admin, 'build_llm_bridge', fake_build,
            raising=False,
        )

        result = asyncio.run(
            ariadne_trace_flow(start_symbol='A', depth=5),
        )
        assert build_calls == []
        # Sanity — static trace still produces the expected hop.
        assert len(result['hops']) == 1

    def test_enable_llm_bridge_true_constructs_bridge(
        self,
        db_with_simple_edge: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``enable_llm_bridge=True`` → builder called once."""
        from ariadne_mcp.server_admin import ariadne_trace_flow

        build_calls: list[dict] = []

        def fake_build(**kwargs):
            build_calls.append(kwargs)
            return lambda cursor, conn: (None, '')

        import ariadne_mcp.server_admin as mcp_server_admin
        monkeypatch.setattr(
            mcp_server_admin, 'build_llm_bridge', fake_build,
            raising=False,
        )

        asyncio.run(
            ariadne_trace_flow(
                start_symbol='A', depth=5, enable_llm_bridge=True,
            ),
        )
        assert len(build_calls) == 1

    def test_default_depth_walks_full_chain(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The wrapper's default depth=5 must actually let the
        walker run — a regression where the default became 0
        would silently break every production call. Builds a
        chain of 4 reachable hops; the default depth covers them."""
        from library.scip import init_scip_schema
        from ariadne_mcp.server_admin import ariadne_trace_flow

        db_path = tmp_path / 'mcp_chain.db'
        conn = sqlite3.connect(db_path)
        init_scip_schema(conn)
        for sym in ('A', 'B', 'C', 'D', 'E'):
            conn.execute(
                'INSERT INTO scip_symbols VALUES '
                '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (sym, 'myapp', 'python', 'app.py',
                 1, 10, 'function', sym, sym, None),
            )
        for caller, callee in (
            ('A', 'B'), ('B', 'C'), ('C', 'D'), ('D', 'E'),
        ):
            conn.execute(
                'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
                (caller, callee, 'reference', 'app.py', 1, 'scip'),
            )
        conn.commit()
        conn.close()

        cfg_stub = SimpleNamespace(db_path=str(db_path))
        monkeypatch.setattr(
            'config.get_config', lambda: cfg_stub,
        )

        # No depth kwarg → wrapper default (5)
        result = asyncio.run(
            ariadne_trace_flow(start_symbol='A'),
        )
        # 4 reachable hops within depth=5
        assert len(result['hops']) == 4
        assert result['truncated'] is False
