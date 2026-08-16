"""Cross-language flow tracer (Phase 9 / SCIP-everywhere user payoff).

Walks the combined cross-language graph from a starting symbol and
returns a structured trace with per-hop provenance tags. The graph is
the union of:

- SCIP edges (within-language, compiler-grade) — Wave 1 / Phase 2
- HTTP boundaries (api_calls + api_endpoints) — Phase 7 (Swagger) +
  Phase 8 (framework patterns)
- Subprocess invocations (process_invocations) — Layer C / Phase 2t
- LLM bridge — Phase 9.b, fires only when all static tiers are empty
  for a cursor and the caller injected an ``llm_bridge`` callable.

Tier-priority cascade per cursor:

1. SCIP edges — if any, follow them and don't fall through.
2. HTTP via api_calls (only when SCIP is empty for this cursor).
3. process_invocations (only when SCIP and HTTP are empty).
4. LLM bridge — only when all three static tiers are empty AND the
   caller provided one. Sets ``incomplete=True`` if the bridge
   declines (returns ``None``).

Within-language locality is preferred — boundary hops only fire when
the static within-language graph runs out. This matches the design
in ``designs/scip-everywhere-remaining.md:636-754``.

The function is sync. The MCP-tool wrapper (Phase 9 wiring slice)
awaits it from an async context — sync core is simpler to test and
the body is I/O-bound on SQLite, which doesn't benefit from asyncio.
``llm_bridge`` is also sync from the walker's perspective; production
implementations wrap async LLM calls behind a sync facade.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal

from attrs import field, frozen

if TYPE_CHECKING:
    from sqlite3 import Connection


# A bridge takes the cursor's canonical_id and a DB connection (so it
# can read documents / scip_symbols / etc. for context). Returns a
# tuple ``(next_symbol_id, rationale)`` — ``next_symbol_id=None``
# signals "no plausible bridge"; the rationale is shown to the user
# as the LLM's reasoning for the suggested hop.
LlmBridgeFn = Callable[[str, 'Connection'], 'tuple[str | None, str]']


_Tier = Literal['scip', 'swagger', 'pattern', 'process', 'llm-inferred']


@frozen
class HopInfo:
    """One hop in the trace.

    ``callee_symbol_id`` is the canonical_id of the next symbol when
    the hop has a known target; ``callee_path`` is the file-path
    string when the target is a script file (e.g., process_invocations
    where ``target_symbol_id`` hasn't been resolved yet via Phase
    2t.b). At least one of them is set; both can be ``None`` for an
    unresolved process_invocation.
    """
    tier: _Tier
    caller_symbol_id: str
    callee_symbol_id: str | None
    callee_path: str | None
    file: Path
    line: int
    rationale: str = ''


@frozen
class TraceFlowResult:
    """Top-level trace_flow output.

    ``truncated`` is True when the depth budget ran out with more to
    explore. ``incomplete`` is reserved for the LLM bridge (Phase 9.b)
    where the trace might end without exhausting the static graph.
    """
    start_symbol_id: str
    start_qualified_name: str
    start_source: str
    hops: list[HopInfo] = field(factory=list)
    truncated: bool = False
    incomplete: bool = False
    data_touches: dict = field(factory=dict)


def _lookup_start_metadata(
    conn: 'Connection',
    start_symbol: str,
) -> tuple[str, str]:
    """Return (qualified_name, source_name) for the start symbol, or
    fall back to the canonical_id and empty source if unknown."""
    cur = conn.execute(
        'SELECT qualified_name, source_name '
        'FROM scip_symbols WHERE canonical_id = ?',
        (start_symbol,),
    )
    row = cur.fetchone()
    if row is None:
        return (start_symbol, '')
    return (row[0], row[1])


def _lookup_symbol_location(
    conn: 'Connection',
    symbol_id: str,
) -> tuple[str, int]:
    """Return (file, line_start) for a symbol — used to anchor an
    LLM-inferred hop to the cursor's last-known location. Falls back
    to ('', 0) when the symbol isn't in scip_symbols (rare; mostly
    happens if the bridge suggests a string the walker can't resolve)."""
    cur = conn.execute(
        'SELECT file, line_start FROM scip_symbols WHERE canonical_id = ?',
        (symbol_id,),
    )
    row = cur.fetchone()
    if row is None:
        return ('', 0)
    return (row[0], row[1])


def _is_flow_edge(callee_id: str) -> bool:
    """True if a ``scip_edges`` callee is a genuine CALL (flow), not a type ref.

    Delegates to the ingest-time classifier so there is ONE rule. This used to
    sniff the suffix here, with a subtly different test (`().`) that dropped
    every overloaded method -- `foo(+1).` -- from flow traces.
    """
    from docgen.scip_cross_source import classify_edge

    return classify_edge(callee_id) == 'call'


def trace_flow(
    *,
    start_symbol: str,
    depth: int = 5,
    conn: 'Connection',
    llm_bridge: LlmBridgeFn | None = None,
) -> TraceFlowResult:
    """Walk from ``start_symbol`` through the combined cross-language
    graph, returning a depth-bounded trace with per-hop provenance.

    Cycle-protected: each canonical_id is visited at most once.

    ``llm_bridge`` (Phase 9.b) is consulted only when SCIP, HTTP, and
    process tiers are all empty for a cursor. If the bridge returns
    ``None``, ``result.incomplete`` is set to True. If no bridge is
    provided, the trace simply terminates on graph exhaustion without
    raising or setting flags.
    """
    start_qn, start_src = _lookup_start_metadata(conn, start_symbol)

    visited: set[str] = {start_symbol}
    queue: deque[tuple[str, int]] = deque([(start_symbol, depth)])
    hops: list[HopInfo] = []
    truncated = False
    incomplete = False

    while queue:
        cursor, remaining = queue.popleft()
        if remaining <= 0:
            # Hit the depth limit; mark truncated and skip processing.
            truncated = True
            continue

        # Tier 1: SCIP edges (within-language)
        scip_rows = conn.execute(
            'SELECT callee_canonical_id, file, line '
            'FROM scip_edges '
            "WHERE caller_canonical_id = ? AND edge_type = 'call'",
            (cursor,),
        ).fetchall()
        # Keep only genuine call edges — type/attribute refs and anonymous
        # locals aren't flow and blow the walk up combinatorially.
        scip_rows = [r for r in scip_rows if _is_flow_edge(r[0])]
        if scip_rows:
            for callee, file, line in scip_rows:
                if callee in visited:
                    continue
                hops.append(HopInfo(
                    tier='scip',
                    caller_symbol_id=cursor,
                    callee_symbol_id=callee,
                    callee_path=None,
                    file=Path(file),
                    line=line,
                ))
                visited.add(callee)
                queue.append((callee, remaining - 1))
            continue  # SCIP wins; don't fall through to other tiers

        # Tier 2: HTTP boundary via api_calls → api_endpoints
        api_rows = conn.execute(
            'SELECT ae.producer_symbol_id, ae.resolution_source, '
            '       ac.call_site_file, ac.call_site_line '
            'FROM api_calls ac '
            'JOIN api_endpoints ae '
            '  ON ae.endpoint_id = ac.endpoint_id '
            'WHERE ac.consumer_symbol_id = ?',
            (cursor,),
        ).fetchall()
        if api_rows:
            for producer_sym, res_source, file, line in api_rows:
                tier: _Tier = (
                    'swagger' if res_source == 'swagger' else 'pattern'
                )
                hops.append(HopInfo(
                    tier=tier,
                    caller_symbol_id=cursor,
                    callee_symbol_id=producer_sym,
                    callee_path=None,
                    file=Path(file),
                    line=line,
                ))
                if (
                    producer_sym is not None
                    and producer_sym not in visited
                ):
                    visited.add(producer_sym)
                    queue.append((producer_sym, remaining - 1))
            continue

        # Tier 3: process_invocations (Layer C subprocess edges)
        proc_rows = conn.execute(
            'SELECT target_path, target_symbol_id, file, line_start '
            'FROM process_invocations '
            'WHERE caller_symbol_id = ?',
            (cursor,),
        ).fetchall()
        if proc_rows:
            for target_path, target_sym, file, line in proc_rows:
                hops.append(HopInfo(
                    tier='process',
                    caller_symbol_id=cursor,
                    callee_symbol_id=target_sym,
                    callee_path=target_path,
                    file=Path(file),
                    line=line,
                ))
                if (
                    target_sym is not None
                    and target_sym not in visited
                ):
                    visited.add(target_sym)
                    queue.append((target_sym, remaining - 1))
            continue

        # Tier 4: LLM bridge (Phase 9.b). Only fires when all static
        # tiers were empty for this cursor AND a bridge was injected.
        if llm_bridge is None:
            continue
        next_sym, rationale = llm_bridge(cursor, conn)
        if next_sym is None:
            # Bridge declined → trace ends without a static answer.
            incomplete = True
            continue
        if next_sym in visited:
            # Bridge suggested a cycle; drop silently. Don't mark
            # incomplete — the bridge gave an answer, just one we
            # already know.
            continue
        loc_file, loc_line = _lookup_symbol_location(conn, cursor)
        hops.append(HopInfo(
            tier='llm-inferred',
            caller_symbol_id=cursor,
            callee_symbol_id=next_sym,
            callee_path=None,
            file=Path(loc_file),
            line=loc_line,
            rationale=rationale,
        ))
        visited.add(next_sym)
        queue.append((next_sym, remaining - 1))
    from docgen.sql_query_views import data_touched_by
    data_touches = {}
    for sym in visited:
        touched = data_touched_by(conn, sym)
        if touched:
            data_touches[sym] = touched
    return TraceFlowResult(
        start_symbol_id=start_symbol,
        start_qualified_name=start_qn,
        start_source=start_src,
        hops=hops,
        truncated=truncated,
        incomplete=incomplete,
        data_touches=data_touches,
    )


__all__ = [
    'HopInfo',
    'TraceFlowResult',
    'trace_flow',
]
