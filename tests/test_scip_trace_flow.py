"""Contract for ``trace_flow`` (Phase 9 / SCIP-everywhere user payoff).

Walks the combined cross-language graph from a starting symbol and
returns a structured trace with **per-hop provenance tags**:

- ``'scip'`` — within-language SCIP edge (compiler-grade)
- ``'swagger'`` — HTTP boundary resolved via Swagger ingestion
- ``'pattern'`` — HTTP boundary resolved via framework-pattern
  extraction (Phase 8a/8b/8c — populated when those phases land)
- ``'process'`` — subprocess/script invocation via process_invocations
  (Layer C / Phase 2t)
- ``'llm-inferred'`` — when static graph runs out (Phase 9.b)

Tier-priority cascade: each cursor first tries SCIP edges. If any
SCIP callees exist, the trace follows them and does NOT fall through
to other tiers (preserves within-language locality). Only when SCIP
is empty does the walker check HTTP / process tiers in order.

Algorithm guarantees:

- Cycle protection (each canonical_id visited at most once)
- Depth limit honored (``truncated=True`` flag when hit)
- ``incomplete=True`` when the graph runs out before the depth budget

MVP scope (this slice): static-graph traversal. Phase 9.b adds the
LLM bridge for graph-exhausted cases.

These tests are RED until ``docgen/trace_flow.py`` exists.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the SCIP schema applied."""
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _insert_symbol(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    source_name: str = 'myapp',
    file: str = 'app.py',
    line_start: int = 1,
    line_end: int = 10,
    qualified_name: str | None = None,
) -> None:
    conn.execute(
        'INSERT INTO scip_symbols VALUES '
        '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (
            canonical_id, source_name, 'python', file,
            line_start, line_end, 'function',
            canonical_id.rsplit('.', 1)[-1],
            qualified_name or canonical_id,
            None,
        ),
    )


def _insert_edge(
    conn: sqlite3.Connection,
    *,
    caller: str,
    callee: str,
    file: str = 'app.py',
    line: int = 5,
) -> None:
    conn.execute(
        'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
        (caller, callee, 'reference', file, line, 'scip'),
    )


# ---------------------------------------------------------------------------
# Value-class shapes
# ---------------------------------------------------------------------------


class TestValueShapes:
    def test_hop_info_is_frozen(self) -> None:
        from docgen.trace_flow import HopInfo

        h = HopInfo(
            tier='scip',
            caller_symbol_id='a',
            callee_symbol_id='b',
            callee_path=None,
            file=Path('app.py'),
            line=10,
            rationale='',
        )
        try:
            h.tier = 'swagger'  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError('HopInfo should be @frozen')

    def test_trace_flow_result_is_frozen(self) -> None:
        from docgen.trace_flow import TraceFlowResult

        r = TraceFlowResult(
            start_symbol_id='a',
            start_qualified_name='a',
            start_source='myapp',
            hops=[],
            truncated=False,
            incomplete=False,
        )
        try:
            r.truncated = True  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError(
            'TraceFlowResult should be @frozen',
        )


# ---------------------------------------------------------------------------
# Empty / unknown cases
# ---------------------------------------------------------------------------


class TestEmptyAndUnknown:
    def test_no_edges_returns_empty_hops(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Symbol exists but has no outgoing edges → no hops, the
        walker terminates cleanly."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='myapp.solo')
        conn.commit()

        result = trace_flow(
            start_symbol='myapp.solo', depth=5, conn=conn,
        )
        assert result.hops == []
        assert not result.truncated

    def test_unknown_symbol_returns_empty_with_start_metadata_blank(
        self, conn: sqlite3.Connection,
    ) -> None:
        """A start symbol not in scip_symbols → empty hops; start
        metadata reflects the unknown lookup."""
        from docgen.trace_flow import trace_flow

        result = trace_flow(
            start_symbol='nonexistent', depth=5, conn=conn,
        )
        assert result.start_symbol_id == 'nonexistent'
        assert result.hops == []


# ---------------------------------------------------------------------------
# SCIP tier — within-language traces
# ---------------------------------------------------------------------------


class TestScipTier:
    def test_single_scip_edge(self, conn: sqlite3.Connection) -> None:
        """One SCIP edge → one hop with tier='scip'."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='myapp.caller')
        _insert_symbol(conn, canonical_id='myapp.callee')
        _insert_edge(
            conn, caller='myapp.caller', callee='myapp.callee', line=7,
        )
        conn.commit()

        result = trace_flow(
            start_symbol='myapp.caller', depth=5, conn=conn,
        )
        assert len(result.hops) == 1
        h = result.hops[0]
        assert h.tier == 'scip'
        assert h.caller_symbol_id == 'myapp.caller'
        assert h.callee_symbol_id == 'myapp.callee'
        assert h.line == 7

    def test_multi_hop_chain(self, conn: sqlite3.Connection) -> None:
        """A → B → C → D produces three SCIP hops in order."""
        from docgen.trace_flow import trace_flow

        for sym in ('A', 'B', 'C', 'D'):
            _insert_symbol(conn, canonical_id=sym)
        _insert_edge(conn, caller='A', callee='B', line=1)
        _insert_edge(conn, caller='B', callee='C', line=2)
        _insert_edge(conn, caller='C', callee='D', line=3)
        conn.commit()

        result = trace_flow(start_symbol='A', depth=5, conn=conn)
        chain = [(h.caller_symbol_id, h.callee_symbol_id) for h in result.hops]
        assert chain == [('A', 'B'), ('B', 'C'), ('C', 'D')]
        assert all(h.tier == 'scip' for h in result.hops)

    def test_cycle_protection(self, conn: sqlite3.Connection) -> None:
        """A → B → A doesn't loop forever — each symbol visited at
        most once."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='A')
        _insert_symbol(conn, canonical_id='B')
        _insert_edge(conn, caller='A', callee='B')
        _insert_edge(conn, caller='B', callee='A')
        conn.commit()

        result = trace_flow(start_symbol='A', depth=10, conn=conn)
        # First hop: A → B. Second hop attempts B → A, but A is
        # already visited → skipped. Total: 1 hop.
        assert len(result.hops) == 1
        assert result.hops[0].caller_symbol_id == 'A'
        assert result.hops[0].callee_symbol_id == 'B'

    def test_branching_explores_all(
        self, conn: sqlite3.Connection,
    ) -> None:
        """A → B and A → C both surface as hops."""
        from docgen.trace_flow import trace_flow

        for sym in ('A', 'B', 'C'):
            _insert_symbol(conn, canonical_id=sym)
        _insert_edge(conn, caller='A', callee='B')
        _insert_edge(conn, caller='A', callee='C')
        conn.commit()

        result = trace_flow(start_symbol='A', depth=5, conn=conn)
        callees = {h.callee_symbol_id for h in result.hops}
        assert callees == {'B', 'C'}


# ---------------------------------------------------------------------------
# Depth limit
# ---------------------------------------------------------------------------


class TestDepthLimit:
    def test_depth_limit_truncates(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Chain of 5; depth=2 → only 2 hops, ``truncated=True``."""
        from docgen.trace_flow import trace_flow

        for sym in ('A', 'B', 'C', 'D', 'E', 'F'):
            _insert_symbol(conn, canonical_id=sym)
        _insert_edge(conn, caller='A', callee='B')
        _insert_edge(conn, caller='B', callee='C')
        _insert_edge(conn, caller='C', callee='D')
        _insert_edge(conn, caller='D', callee='E')
        _insert_edge(conn, caller='E', callee='F')
        conn.commit()

        result = trace_flow(start_symbol='A', depth=2, conn=conn)
        assert len(result.hops) == 2
        assert result.truncated is True

    def test_depth_zero_no_hops(self, conn: sqlite3.Connection) -> None:
        """depth=0 → no walking happens at all."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='A')
        _insert_symbol(conn, canonical_id='B')
        _insert_edge(conn, caller='A', callee='B')
        conn.commit()

        result = trace_flow(start_symbol='A', depth=0, conn=conn)
        assert result.hops == []


# ---------------------------------------------------------------------------
# HTTP tier (api_calls → api_endpoints) — Phase 7 surface
# ---------------------------------------------------------------------------


class TestSwaggerTier:
    def test_http_boundary_with_swagger_tier(
        self, conn: sqlite3.Connection,
    ) -> None:
        """A symbol with no SCIP callees but an api_calls row →
        the trace follows the HTTP boundary. ``resolution_source``
        on the endpoint is 'swagger' → hop tier is 'swagger'."""
        from docgen.trace_flow import trace_flow

        # Caller has no SCIP edges but is in api_calls
        _insert_symbol(
            conn, canonical_id='myapp.consumer', source_name='myapp',
        )
        _insert_symbol(
            conn, canonical_id='other.handler', source_name='other',
        )
        # Endpoint resolved via Swagger
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('ep1', 'other', 'POST', '/login',
             'other.handler', 'swagger'),
        )
        # Call site connects consumer → endpoint
        conn.execute(
            'INSERT INTO api_calls VALUES (?, ?, ?, ?, ?, ?)',
            ('myapp.consumer', 'ep1', 'app.py', 12,
             'pattern', 'exact-literal'),
        )
        conn.commit()

        result = trace_flow(
            start_symbol='myapp.consumer', depth=3, conn=conn,
        )
        assert len(result.hops) >= 1
        first = result.hops[0]
        assert first.tier == 'swagger'
        assert first.callee_symbol_id == 'other.handler'

    def test_http_endpoint_pattern_resolution_marks_pattern_tier(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Endpoints with ``resolution_source='pattern'`` (Phase 8)
        produce hops with tier='pattern' instead of 'swagger'."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(
            conn, canonical_id='myapp.consumer', source_name='myapp',
        )
        _insert_symbol(
            conn, canonical_id='other.handler', source_name='other',
        )
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('ep2', 'other', 'GET', '/users',
             'other.handler', 'pattern'),
        )
        conn.execute(
            'INSERT INTO api_calls VALUES (?, ?, ?, ?, ?, ?)',
            ('myapp.consumer', 'ep2', 'app.py', 5,
             'pattern', 'exact-literal'),
        )
        conn.commit()

        result = trace_flow(
            start_symbol='myapp.consumer', depth=3, conn=conn,
        )
        assert result.hops[0].tier == 'pattern'


# ---------------------------------------------------------------------------
# Process tier (process_invocations) — Layer C / Phase 2t
# ---------------------------------------------------------------------------


class TestProcessTier:
    def test_process_invocation_with_resolved_path(
        self, conn: sqlite3.Connection,
    ) -> None:
        """A symbol with no SCIP edges but a process_invocations row →
        hop with tier='process' carrying the resolved target_path."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(
            conn, canonical_id='scala.Runner.run', source_name='scalaproject',
            file='Runner.scala',
        )
        conn.execute(
            'INSERT INTO process_invocations '
            '(source_name, caller_symbol_id, target_path, '
            'target_symbol_id, confidence, file, line_start, line_end) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('scalaproject', 'scala.Runner.run',
             'models/azureml/local_scripts/train.py',
             None, 'resolved-constant', 'Runner.scala', 42, 42),
        )
        conn.commit()

        result = trace_flow(
            start_symbol='scala.Runner.run', depth=3, conn=conn,
        )
        assert len(result.hops) == 1
        h = result.hops[0]
        assert h.tier == 'process'
        assert h.callee_path == 'models/azureml/local_scripts/train.py'
        assert h.line == 42

    def test_process_invocation_unresolved_still_emits_hop(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Unresolved process_invocations (target_path NULL) still
        emit a hop — trace_flow can show 'attempted but not
        resolvable'. ``callee_symbol_id`` and ``callee_path`` both
        None."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(
            conn, canonical_id='scala.Runner.run', source_name='scalaproject',
        )
        conn.execute(
            'INSERT INTO process_invocations '
            '(source_name, caller_symbol_id, target_path, '
            'target_symbol_id, confidence, file, line_start, line_end) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('scalaproject', 'scala.Runner.run',
             None, None, 'unresolved', 'Runner.scala', 42, 42),
        )
        conn.commit()

        result = trace_flow(
            start_symbol='scala.Runner.run', depth=3, conn=conn,
        )
        assert len(result.hops) == 1
        h = result.hops[0]
        assert h.tier == 'process'
        assert h.callee_symbol_id is None
        assert h.callee_path is None


# ---------------------------------------------------------------------------
# Tier priority — SCIP wins over HTTP/process when both apply
# ---------------------------------------------------------------------------


class TestTierPriority:
    def test_scip_wins_over_http_when_both_exist(
        self, conn: sqlite3.Connection,
    ) -> None:
        """When a cursor has BOTH SCIP edges AND api_calls, only the
        SCIP edges are followed. Within-language locality is preferred;
        HTTP boundaries are reserved for when SCIP runs out."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(
            conn, canonical_id='myapp.consumer', source_name='myapp',
        )
        _insert_symbol(
            conn, canonical_id='myapp.helper', source_name='myapp',
        )
        # SCIP edge from consumer → helper
        _insert_edge(
            conn, caller='myapp.consumer', callee='myapp.helper',
        )
        # ALSO an api_call from consumer
        _insert_symbol(
            conn, canonical_id='other.handler', source_name='other',
        )
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('ep', 'other', 'GET', '/x', 'other.handler', 'swagger'),
        )
        conn.execute(
            'INSERT INTO api_calls VALUES (?, ?, ?, ?, ?, ?)',
            ('myapp.consumer', 'ep', 'app.py', 7,
             'pattern', 'exact-literal'),
        )
        conn.commit()

        result = trace_flow(
            start_symbol='myapp.consumer', depth=3, conn=conn,
        )
        # Only the SCIP hop; HTTP not followed
        assert len(result.hops) == 1
        assert result.hops[0].tier == 'scip'
        assert result.hops[0].callee_symbol_id == 'myapp.helper'

    def test_http_wins_over_process_when_both_exist(
        self, conn: sqlite3.Connection,
    ) -> None:
        """When SCIP is empty but BOTH api_calls AND process_invocations
        exist for a cursor, HTTP wins (more semantically rich than
        bare subprocess)."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(
            conn, canonical_id='myapp.consumer', source_name='myapp',
        )
        _insert_symbol(
            conn, canonical_id='other.handler', source_name='other',
        )
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('ep', 'other', 'GET', '/x', 'other.handler', 'swagger'),
        )
        conn.execute(
            'INSERT INTO api_calls VALUES (?, ?, ?, ?, ?, ?)',
            ('myapp.consumer', 'ep', 'app.py', 5,
             'pattern', 'exact-literal'),
        )
        conn.execute(
            'INSERT INTO process_invocations '
            '(source_name, caller_symbol_id, target_path, '
            'target_symbol_id, confidence, file, line_start, line_end) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            ('myapp', 'myapp.consumer', '/scripts/foo.py',
             None, 'resolved-constant', 'app.py', 10, 10),
        )
        conn.commit()

        result = trace_flow(
            start_symbol='myapp.consumer', depth=3, conn=conn,
        )
        assert len(result.hops) == 1
        assert result.hops[0].tier == 'swagger'


# ---------------------------------------------------------------------------
# Combined trace — multi-tier
# ---------------------------------------------------------------------------


class TestCombinedTrace:
    def test_scip_then_http_then_scip(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Realistic shape: function → function (SCIP), then HTTP
        boundary, then more SCIP within the producer service."""
        from docgen.trace_flow import trace_flow

        # Consumer-side: A.run → A.helper (SCIP)
        _insert_symbol(
            conn, canonical_id='A.run', source_name='A',
        )
        _insert_symbol(
            conn, canonical_id='A.helper', source_name='A',
        )
        _insert_edge(conn, caller='A.run', callee='A.helper', line=1)

        # A.helper → HTTP → B.handler
        _insert_symbol(
            conn, canonical_id='B.handler', source_name='B',
        )
        conn.execute(
            'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
            ('ep', 'B', 'POST', '/api', 'B.handler', 'swagger'),
        )
        conn.execute(
            'INSERT INTO api_calls VALUES (?, ?, ?, ?, ?, ?)',
            ('A.helper', 'ep', 'A.py', 5, 'pattern', 'exact-literal'),
        )

        # Producer-side: B.handler → B.svc (SCIP)
        _insert_symbol(
            conn, canonical_id='B.svc', source_name='B',
        )
        _insert_edge(conn, caller='B.handler', callee='B.svc', line=2)

        conn.commit()

        result = trace_flow(start_symbol='A.run', depth=5, conn=conn)
        tiers = [h.tier for h in result.hops]
        assert tiers == ['scip', 'swagger', 'scip']


# ---------------------------------------------------------------------------
# LLM bridge tier (Phase 9.b)
# ---------------------------------------------------------------------------


class TestLlmBridgeTier:
    """Phase 9.b — when SCIP, HTTP, and process tiers are all empty
    for a cursor, an injected ``llm_bridge`` callable is consulted to
    suggest the next hop. Bridges are injected (not constructed by
    the walker) so tests stay deterministic and production wiring
    can choose its own LLM client.

    Bridge contract: ``(cursor_symbol_id, conn) -> (next_symbol_id |
    None, rationale)``. ``None`` signals "no plausible bridge"; the
    walker sets ``result.incomplete=True``.
    """

    def test_bridge_invoked_when_static_graph_empty(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Cursor has no SCIP/HTTP/process edges → bridge gets called.
        Bridge returns (next, rationale) → llm-inferred hop added."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='A')
        _insert_symbol(conn, canonical_id='B')
        conn.commit()

        calls: list[str] = []

        def bridge(cursor: str, _conn) -> tuple[str | None, str]:
            calls.append(cursor)
            if cursor == 'A':
                return ('B', 'doc says A delegates to B')
            return (None, '')

        result = trace_flow(
            start_symbol='A', depth=5, conn=conn, llm_bridge=bridge,
        )
        # Bridge fires for both A (returns B) and B (returns None)
        assert calls == ['A', 'B']
        assert len(result.hops) == 1
        h = result.hops[0]
        assert h.tier == 'llm-inferred'
        assert h.caller_symbol_id == 'A'
        assert h.callee_symbol_id == 'B'
        assert h.rationale == 'doc says A delegates to B'
        # Bridge declined for B → trace incomplete
        assert result.incomplete is True

    def test_bridge_not_invoked_when_scip_fires(
        self, conn: sqlite3.Connection,
    ) -> None:
        """When SCIP has edges for a cursor, the bridge is NOT called
        for that cursor. Within-language locality preserved.

        Paired with the empty-graph test above so a stub that always
        calls the bridge fails this test, and a stub that never calls
        the bridge fails the other half. Asserts ``'B' in calls`` —
        B has no SCIP edges so the bridge SHOULD be called for B,
        even though it's not called for A."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='A')
        _insert_symbol(conn, canonical_id='B')
        _insert_edge(conn, caller='A', callee='B')
        conn.commit()

        calls: list[str] = []

        def bridge(cursor: str, _conn) -> tuple[str | None, str]:
            calls.append(cursor)
            return (None, '')

        result = trace_flow(
            start_symbol='A', depth=5, conn=conn, llm_bridge=bridge,
        )
        # SCIP fired for A → bridge NOT called for A
        assert 'A' not in calls
        # B has no edges → bridge IS called for B (this bites a stub
        # that never invokes the bridge)
        assert 'B' in calls
        # The single hop is the SCIP one
        assert len(result.hops) == 1
        assert result.hops[0].tier == 'scip'

    def test_bridge_returns_none_sets_incomplete(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Bridge declines → no hop added, ``incomplete=True`` set."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='A')
        conn.commit()

        def bridge(cursor: str, _conn) -> tuple[str | None, str]:
            return (None, '')

        result = trace_flow(
            start_symbol='A', depth=5, conn=conn, llm_bridge=bridge,
        )
        assert result.hops == []
        assert result.incomplete is True

    def test_no_bridge_no_incomplete(
        self, conn: sqlite3.Connection,
    ) -> None:
        """No bridge → exhausted graph still produces a clean
        (non-incomplete) result. Bites a walker that flags incomplete
        on every empty cursor regardless of bridge presence."""
        from docgen.trace_flow import trace_flow

        _insert_symbol(conn, canonical_id='A')
        conn.commit()

        result = trace_flow(start_symbol='A', depth=5, conn=conn)
        assert result.hops == []
        assert result.incomplete is False

    def test_bridge_chain_continues_until_decline(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Bridge can chain multiple llm-inferred hops within depth.
        A → B → C → (decline). Two llm-inferred hops; final cursor's
        decline sets incomplete=True."""
        from docgen.trace_flow import trace_flow

        for sym in ('A', 'B', 'C'):
            _insert_symbol(conn, canonical_id=sym)
        conn.commit()

        answers: dict[str, tuple[str | None, str]] = {
            'A': ('B', 'r1'),
            'B': ('C', 'r2'),
            'C': (None, ''),
        }

        def bridge(cursor: str, _conn) -> tuple[str | None, str]:
            return answers.get(cursor, (None, ''))

        result = trace_flow(
            start_symbol='A', depth=5, conn=conn, llm_bridge=bridge,
        )
        assert len(result.hops) == 2
        assert all(h.tier == 'llm-inferred' for h in result.hops)
        assert [h.rationale for h in result.hops] == ['r1', 'r2']
        assert result.incomplete is True

    def test_bridge_respects_depth_limit(
        self, conn: sqlite3.Connection,
    ) -> None:
        """LLM-inferred hops count toward depth like any other tier.
        Same chain as above but depth=1 → 1 hop, truncated=True.
        Paired with the chain test so a walker that ignores depth
        for LLM hops fails one half."""
        from docgen.trace_flow import trace_flow

        for sym in ('A', 'B', 'C'):
            _insert_symbol(conn, canonical_id=sym)
        conn.commit()

        answers: dict[str, tuple[str | None, str]] = {
            'A': ('B', 'r1'),
            'B': ('C', 'r2'),
        }

        def bridge(cursor: str, _conn) -> tuple[str | None, str]:
            return answers.get(cursor, (None, ''))

        result = trace_flow(
            start_symbol='A', depth=1, conn=conn, llm_bridge=bridge,
        )
        assert len(result.hops) == 1
        assert result.truncated is True

    def test_bridge_cycle_suggestion_skipped(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Bridge suggests a visited symbol → don't loop, don't add
        hop. Paired with the chain test so a walker that ignores
        ``visited`` would re-walk A and produce extra hops."""
        from docgen.trace_flow import trace_flow

        for sym in ('A', 'B'):
            _insert_symbol(conn, canonical_id=sym)
        conn.commit()

        # A → B (LLM), then bridge tries to send B back to A
        answers: dict[str, tuple[str | None, str]] = {
            'A': ('B', 'r1'),
            'B': ('A', 'cycle'),
        }

        def bridge(cursor: str, _conn) -> tuple[str | None, str]:
            return answers.get(cursor, (None, ''))

        result = trace_flow(
            start_symbol='A', depth=5, conn=conn, llm_bridge=bridge,
        )
        # Only A → B; A is already in visited so the cycle hop is
        # dropped silently.
        assert len(result.hops) == 1
        assert result.hops[0].callee_symbol_id == 'B'
