"""Render a cross-language trace as a Graphviz DOT sequence diagram.

DOT has no native ``sequenceDiagram`` keyword, so we build one from primitives:

- a header row of source "lifelines" (``rank=same``),
- a dashed vertical lifeline per source (one node per time step, rank-ordered),
- **activation bars** — thick segments overlaying a lifeline while that source is
  processing a call (opened by a ``call``, closed by its ``return``; they nest
  and repeat, so a participant can show several bars), and
- ``constraint=false`` message edges (solid call, dashed-grey return).

``dot -Tpng`` then lays this out as a real sequence diagram. The renderer is a
pure function over participant-level :class:`TraceMessage`s; :func:`trace_to_messages`
is the thin adapter that resolves a ``TraceFlowResult``'s symbol-ids to their
sources against the unified graph (``scip_symbols``). Both stay light and ride
the same DOT→PNG path the bridge already uses.
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

if TYPE_CHECKING:
    import sqlite3

    from docgen.trace_flow import TraceFlowResult

# Tier → call-edge style. SCIP is an in-process call (solid); the HTTP tiers
# (swagger/pattern) and the LLM bridge are network/inferred hops (dashed);
# a subprocess invocation is dotted.
_TIER_EDGE_STYLE: dict[str, str] = {
    'scip': 'solid',
    'swagger': 'dashed',
    'pattern': 'dashed',
    'process': 'dotted',
    'llm-inferred': 'dashed',
}


@frozen
class TraceMessage:
    """One ordered interaction in the trace, at the source (lifeline) level.

    ``caller``/``callee`` are source names — the sequence-diagram lifelines;
    ``label`` is shown on the arrow; ``tier`` is the ``trace_flow`` tier (drives
    the call-arrow style). ``kind`` is ``'call'`` (default — activates the
    callee) or ``'return'`` (the caller finishes and hands control back).
    """

    caller: str
    callee: str
    label: str
    tier: str = 'scip'
    kind: str = 'call'


def _participants(messages: list[TraceMessage]) -> list[str]:
    """Distinct sources in first-appearance order (entry point leftmost)."""
    seen: dict[str, None] = {}
    for m in messages:
        seen.setdefault(m.caller, None)
        seen.setdefault(m.callee, None)
    return list(seen)


def _node_id(participant: str, step: int) -> str:
    return f'{participant}__{step}'


def _activation_spans(
    messages: list[TraceMessage],
) -> dict[str, list[tuple[int, int]]]:
    """Per-source activation spans ``(start_step, end_step)``.

    A ``call`` to a source opens an activation on it; the matching ``return``
    from that source (stack-paired, so nesting works) closes it. Self-messages
    (caller == callee) run within the existing bar and open nothing. Activations
    still open at the end extend to the last step (an unreturned synchronous
    call — the natural shape of a forward-only trace).
    """
    open_stacks: dict[str, list[int]] = defaultdict(list)
    spans: dict[str, list[tuple[int, int]]] = defaultdict(list)
    n = len(messages)

    for step, m in enumerate(messages, start=1):
        if m.kind == 'return':
            if open_stacks[m.caller]:
                start = open_stacks[m.caller].pop()
                spans[m.caller].append((start, step))
        elif m.caller != m.callee:  # a 'call' to another source activates it
            open_stacks[m.callee].append(step)

    for participant, starts in open_stacks.items():
        for start in starts:
            spans[participant].append((start, n))

    return dict(spans)


def _active_segments(spans: list[tuple[int, int]]) -> set[int]:
    """Lower step-indices whose lifeline segment falls inside an activation."""
    active: set[int] = set()
    for start, end in spans:
        active.update(range(start, end))
    return active


def _wrap_label(text: str, width: int = 24) -> str:
    """Greedy word-wrap a message label to ~``width`` chars per line, joined
    with the DOT centered-newline escape (``\\n``) so long messages stack onto
    two-plus lines instead of forcing the whole diagram wider."""
    if len(text) <= width:
        return text
    lines: list[str] = []
    current = ''
    for word in text.split(' '):
        if current and len(current) + 1 + len(word) > width:
            lines.append(current)
            current = word
        else:
            current = f'{current} {word}' if current else word
    if current:
        lines.append(current)
    return '\\n'.join(lines)


def render_sequence_dot(messages: list[TraceMessage]) -> str:
    """Render ``messages`` as a Graphviz DOT sequence diagram.

    Returns a ``digraph`` string renderable by ``dot``.
    """
    participants = _participants(messages)
    n = len(messages)
    spans = _activation_spans(messages)

    lines: list[str] = [
        'digraph trace {',
        '  rankdir=TB;',
        # Straight-line edges so messages are horizontal (same-rank endpoints)
        # and lifelines vertical — no spline swoops.
        '  splines=line;',
        '  forcelabels=true;',
        '  ranksep=0.7;',
        '  nodesep=2.2;',
        '  node [shape=box, fontname="Helvetica", fontsize=12];',
        '  edge [fontname="Helvetica", fontsize=11];',
    ]

    # 1. Header row — one labeled box per source, all on the top rank, with an
    #    invisible chain pinning left→right order (else dot reorders the rank).
    if participants:
        decls = '; '.join(f'"{p}" [label="{p}"]' for p in participants)
        if len(participants) > 1:
            order = ' -> '.join(f'"{p}"' for p in participants)
            lines.append(f'  {{ rank=same; {decls}; {order} [style=invis]; }}')
        else:
            lines.append(f'  {{ rank=same; {decls}; }}')

    # 2. Lifelines with activation bars — a dashed vertical per source, drawn
    #    thick + solid over the steps where that source is active (a call in
    #    flight). Drawing the bar as consecutive lifeline segments keeps it
    #    straight down the column; the chain also forces the time ordering.
    lines.append('  node [shape=point, width=0.01, style=invis];')
    for p in participants:
        active = _active_segments(spans.get(p, []))
        prev = f'"{p}"'
        for i in range(1, n + 1):
            cur = f'"{_node_id(p, i)}"'
            if (i - 1) in active:
                seg = '[arrowhead=none, penwidth=11, color="gray60"]'
            else:
                seg = '[arrowhead=none, style=dashed, color="gray70"]'
            lines.append(f'  {prev} -> {cur} {seg};')
            prev = cur

    # 3. Time-slice rows — all sources' step-i nodes share a rank, so message i
    #    is drawn horizontally.
    for i in range(1, n + 1):
        row = '; '.join(f'"{_node_id(p, i)}"' for p in participants)
        lines.append(f'  {{ rank=same; {row}; }}')

    # 5. Messages — one non-constraining edge per hop at its time step. Calls
    #    are styled by tier (solid in-process, dashed HTTP); returns are dashed
    #    grey with an open arrowhead.
    for i, m in enumerate(messages, start=1):
        src = _node_id(m.caller, i)
        dst = _node_id(m.callee, i)
        if m.kind == 'return':
            attrs = (
                'constraint=false, style=dashed, color="gray40", '
                'fontcolor="gray40", arrowhead=vee'
            )
        else:
            style = _TIER_EDGE_STYLE.get(m.tier, 'solid')
            attrs = f'constraint=false, style={style}'
        lines.append(f'  "{src}" -> "{dst}" [{attrs}, xlabel="{_wrap_label(m.label)}"];')

    lines.append('}')
    return '\n'.join(lines) + '\n'


def _resolve(conn: sqlite3.Connection, symbol_id: str) -> tuple[str, str]:
    """Resolve a canonical_id to ``(qualified_name, source_name)`` via the
    unified graph. On a miss, fall back to the id itself with an empty source
    (mirrors ``trace_flow._lookup_start_metadata``)."""
    row = conn.execute(
        'SELECT qualified_name, source_name FROM scip_symbols WHERE canonical_id = ?',
        (symbol_id,),
    ).fetchone()
    if row is None:
        return symbol_id, ''
    return str(row[0]), str(row[1] or '')


def trace_to_messages(
    result: TraceFlowResult, conn: sqlite3.Connection,
) -> list[TraceMessage]:
    """Resolve a ``TraceFlowResult`` into source-level :class:`TraceMessage`s.

    Each hop becomes one ``call`` message: caller/callee mapped to their sources
    (the lifelines), labeled with the callee, tagged with the hop's tier. A
    process-tier hop with only a ``callee_path`` uses the script's filename as
    both the lifeline and the label.
    """
    messages: list[TraceMessage] = []
    for hop in result.hops:
        caller_qn, caller_src = _resolve(conn, hop.caller_symbol_id)
        if hop.callee_symbol_id is not None:
            callee_qn, callee_src = _resolve(conn, hop.callee_symbol_id)
        elif hop.callee_path is not None:
            callee_qn = Path(hop.callee_path).name
            callee_src = callee_qn
        else:
            callee_qn = callee_src = '<unresolved>'
        messages.append(
            TraceMessage(
                caller=caller_src or caller_qn,
                callee=callee_src or callee_qn,
                label=callee_qn,
                tier=hop.tier,
            )
        )
    return messages
