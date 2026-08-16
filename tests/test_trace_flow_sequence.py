"""``trace-flow --sequence`` renders a cross-repo trace as a Graphviz DOT
sequence diagram: lifelines = sources, ordered messages, HTTP hops dashed.

DOT has no high-level ``sequenceDiagram`` keyword, so the renderer builds one
from primitives — a header row, an invisible per-source lifeline chain, one
same-rank row per time step, and ``constraint=false`` message edges — which
``dot`` lays out as a real sequence diagram and renders straight to PNG.

Three pieces, three evolving tests: the pure renderer, the symbol→source
adapter, and the CLI wiring driven end-to-end through a real trace.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

from cli.trace import cmd_trace_flow
from docgen.trace_flow import HopInfo, TraceFlowResult
from docgen.trace_flow_sequence import (
    TraceMessage,
    _activation_spans,
    _wrap_label,
    render_sequence_dot,
    trace_to_messages,
)


def test_renders_cross_source_trace_as_a_dot_sequence_diagram() -> None:
    # alpha.login() calls into beta in-process (a SCIP edge); beta then reaches
    # gamma's auth over REST (an HTTP-tier hop, here `swagger`).
    messages = [
        TraceMessage(caller='alpha', callee='beta', label='login()', tier='scip'),
        TraceMessage(caller='beta', callee='gamma', label='authenticate()', tier='swagger'),
    ]

    dot = render_sequence_dot(messages)

    # A real Graphviz graph, renderable by `dot -Tpng`.
    assert dot.lstrip().startswith('digraph')

    # Lifelines: one per distinct source, in first-appearance order (the
    # entry point leftmost).
    assert dot.index('alpha') < dot.index('beta') < dot.index('gamma')

    # Laid out as a sequence diagram, not a flat call graph: time-slice rows
    # (rank=same), invisible lifelines (style=invis), and messages that don't
    # perturb the ranking (constraint=false).
    assert 'rank=same' in dot
    assert 'style=invis' in dot
    assert 'constraint=false' in dot

    # Each hop is a labeled message, emitted in call order.
    assert dot.index('login()') < dot.index('authenticate()')

    # The HTTP hop is dashed; the in-process hop is solid (not dashed).
    http_line = next(line for line in dot.splitlines() if 'authenticate()' in line)
    scip_line = next(line for line in dot.splitlines() if 'login()' in line)
    assert 'dashed' in http_line
    assert 'dashed' not in scip_line

    # Lifelines are drawn (dashed verticals, arrowhead-less), and at least one
    # activation bar (a thick segment) appears for the source that was called.
    assert 'arrowhead=none' in dot
    assert 'penwidth=' in dot


def test_activation_spans_open_on_call_close_on_return() -> None:
    # B is called (active), B calls C (C active), C returns (C closes),
    # B returns (B closes). Nested bars; the initiator A opens none.
    msgs = [
        TraceMessage('A', 'B', 'req', 'scip', 'call'),
        TraceMessage('B', 'C', 'sub', 'scip', 'call'),
        TraceMessage('C', 'B', 'resp', 'scip', 'return'),
        TraceMessage('B', 'A', 'done', 'scip', 'return'),
    ]

    spans = _activation_spans(msgs)

    assert spans['B'] == [(1, 4)]
    assert spans['C'] == [(2, 3)]
    assert 'A' not in spans


def test_long_labels_wrap_onto_multiple_lines() -> None:
    assert _wrap_label('short label') == 'short label'  # under width: untouched
    wrapped = _wrap_label('fetch public key / cert [back channel]')
    assert '\\n' in wrapped  # split onto >1 line with the DOT centered-newline
    assert all(len(line) <= 26 for line in wrapped.split('\\n'))


def _db_with_symbols(rows: list[tuple[str, str, str]]) -> sqlite3.Connection:
    """In-memory scip_symbols for adapter resolution (canonical_id → name/source)."""
    conn = sqlite3.connect(':memory:')
    conn.execute(
        'CREATE TABLE scip_symbols '
        '(canonical_id TEXT, qualified_name TEXT, source_name TEXT)'
    )
    conn.executemany('INSERT INTO scip_symbols VALUES (?, ?, ?)', rows)
    return conn


def test_adapter_resolves_hops_to_source_level_messages() -> None:
    # Hop symbol-ids carry no source/name; the adapter resolves them against
    # the unified graph (scip_symbols) into source-level messages.
    conn = _db_with_symbols([
        ('a.login', 'alpha.auth.login', 'alpha'),
        ('b.handle', 'beta.api.handle', 'beta'),
        ('c.authenticate', 'gamma.auth.authenticate', 'gamma'),
    ])
    result = TraceFlowResult(
        start_symbol_id='a.login',
        start_qualified_name='alpha.auth.login',
        start_source='alpha',
        hops=[
            HopInfo(
                tier='scip', caller_symbol_id='a.login',
                callee_symbol_id='b.handle', callee_path=None,
                file=Path('alpha/auth.py'), line=10,
            ),
            HopInfo(
                tier='swagger', caller_symbol_id='b.handle',
                callee_symbol_id='c.authenticate', callee_path=None,
                file=Path('beta/api.py'), line=20,
            ),
        ],
    )

    messages = trace_to_messages(result, conn)

    # One message per hop, caller/callee mapped to their sources, tier carried.
    assert [(m.caller, m.callee, m.tier) for m in messages] == [
        ('alpha', 'beta', 'scip'),
        ('beta', 'gamma', 'swagger'),
    ]
    # The label names the callee being invoked.
    assert 'handle' in messages[0].label
    assert 'authenticate' in messages[1].label


def _ins_sym(conn: sqlite3.Connection, cid: str, source: str, qn: str) -> None:
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, line_start, line_end, kind, display_name, qualified_name, parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (cid, source, 'python', 'app.py', 1, 10, 'function',
         cid.rsplit('.', 1)[-1], qn, None),
    )


def _ins_edge(conn: sqlite3.Connection, caller: str, callee: str) -> None:
    conn.execute(
        'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
        (caller, callee, 'reference', 'app.py', 5, 'scip'),
    )


def test_cli_sequence_flag_emits_a_dot_diagram_end_to_end(tmp_path, capsys) -> None:
    # Real trace through a real (temp-file) unified graph: alpha.login →
    # beta.handle. `trace-flow --sequence` should print a renderable DOT
    # sequence diagram with both source lifelines.
    from library.scip import init_scip_schema

    db = tmp_path / 'ariadne.db'
    conn = sqlite3.connect(db)
    init_scip_schema(conn)
    _ins_sym(conn, 'a.login', 'alpha', 'alpha.auth.login')
    _ins_sym(conn, 'b.handle', 'beta', 'beta.api.handle')
    _ins_edge(conn, 'a.login', 'b.handle')
    conn.commit()
    conn.close()

    args = argparse.Namespace(
        symbol='a.login', depth=5, json=False, sequence=True,
        db=str(db), llm_bridge=False,
    )
    rc = cmd_trace_flow(args)

    out = capsys.readouterr().out
    assert rc == 0
    assert out.lstrip().startswith('digraph')
    assert 'alpha' in out and 'beta' in out   # both source lifelines
    assert 'handle' in out                      # the message label
