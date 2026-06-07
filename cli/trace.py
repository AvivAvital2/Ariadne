"""``ariadne trace-flow`` CLI command (Phase 9 wiring).

Wraps :func:`docgen.trace_flow.trace_flow` with argparse + JSON/Rich
rendering. JSON output uses the same shape consumed by the
``ariadne_trace_flow`` MCP tool, so CLI users and agents see
consistent traces.

Tier colors in the Rich tree:

- ``scip`` — green (compiler-grade, within-language)
- ``swagger`` — cyan (HTTP via Swagger spec)
- ``pattern`` — yellow (HTTP via framework-pattern extraction)
- ``process`` — magenta (subprocess invocation, Layer C)
- ``llm-inferred`` — dim red (Phase 9.b; not yet emitted)
"""
from __future__ import annotations

import argparse
import json as _json
import sqlite3
from pathlib import Path

from rich.console import Console
from rich.tree import Tree

from config import get_config
from docgen.trace_flow import HopInfo, TraceFlowResult, trace_flow
from docgen.trace_flow_llm_bridge import build_llm_bridge
from docgen.trace_flow_sequence import render_sequence_dot, trace_to_messages


_console = Console()


_TIER_COLORS: dict[str, str] = {
    'scip': 'green',
    'swagger': 'cyan',
    'pattern': 'yellow',
    'process': 'magenta',
    'llm-inferred': 'dim red',
}


def register_commands(
    subparsers: argparse._SubParsersAction,
) -> None:
    """Register the ``trace-flow`` subcommand."""
    p = subparsers.add_parser(
        'trace-flow',
        help='Trace cross-language flow from a starting symbol',
    )
    p.add_argument(
        'symbol',
        help='Starting canonical_id (e.g., myapp.run)',
    )
    p.add_argument(
        '--depth', type=int, default=5,
        help='Maximum hop depth to walk (default: 5)',
    )
    p.add_argument(
        '--json', action='store_true',
        help='Emit structured JSON instead of a Rich tree',
    )
    p.add_argument(
        '--sequence', action='store_true',
        help=(
            'Emit a Graphviz DOT sequence diagram of the trace '
            '(lifelines = sources; render with `dot -Tpng`)'
        ),
    )
    p.add_argument(
        '--db',
        help='Path to ariadne.db (default: from config)',
    )
    p.add_argument(
        '--llm-bridge', action='store_true', dest='llm_bridge',
        help=(
            'Enable Phase 9.b LLM bridge — when the static graph '
            'runs out, ask Claude for the most likely next hop. '
            'Off by default (avoids LLM cost on every trace).'
        ),
    )


def _resolve_db_path(args: argparse.Namespace) -> Path:
    explicit = getattr(args, 'db', None)
    if explicit:
        return Path(explicit)
    cfg = get_config()
    return Path(cfg.db_path)


def _hop_to_dict(h: HopInfo) -> dict:
    return {
        'tier': h.tier,
        'caller_symbol_id': h.caller_symbol_id,
        'callee_symbol_id': h.callee_symbol_id,
        'callee_path': h.callee_path,
        'file': str(h.file),
        'line': h.line,
        'rationale': h.rationale,
    }


def trace_result_to_dict(result: TraceFlowResult) -> dict:
    """Serialize a TraceFlowResult to the dict shape used by both
    ``--json`` output and the MCP tool. Public so the MCP tool can
    import + reuse without duplication."""
    return {
        'start': {
            'canonical_id': result.start_symbol_id,
            'qualified_name': result.start_qualified_name,
            'source': result.start_source,
        },
        'hops': [_hop_to_dict(h) for h in result.hops],
        'truncated': result.truncated,
        'incomplete': result.incomplete,
    }


def _render_tree(result: TraceFlowResult) -> None:
    """Print a Rich tree of the trace, colored by tier."""
    label = (
        f'[bold]{result.start_qualified_name}[/bold] '
        f'(source: {result.start_source or "unknown"})'
    )
    tree = Tree(label)
    if not result.hops:
        tree.add('[dim](no hops)[/dim]')
    else:
        cursor = tree
        for h in result.hops:
            color = _TIER_COLORS.get(h.tier, 'white')
            target = h.callee_symbol_id or h.callee_path or '<unresolved>'
            line = (
                f'[{color}]{h.tier}[/{color}] '
                f'{h.caller_symbol_id} → {target} '
                f'[dim]@ {h.file}:{h.line}[/dim]'
            )
            cursor = cursor.add(line)

    if result.truncated:
        tree.add('[yellow](truncated — depth limit hit)[/yellow]')

    _console.print(tree)


def cmd_trace_flow(args: argparse.Namespace) -> int:
    """Trace cross-language flow from ``args.symbol`` and emit either
    Rich tree (default) or JSON (``--json``)."""
    db_path = _resolve_db_path(args)
    bridge = None
    if getattr(args, 'llm_bridge', False):
        bridge = build_llm_bridge()
    conn = sqlite3.connect(db_path)
    try:
        result = trace_flow(
            start_symbol=args.symbol,
            depth=args.depth,
            conn=conn,
            llm_bridge=bridge,
        )
        # The sequence renderer resolves hop symbol-ids → sources against the
        # graph, so build the messages while the connection is still open.
        messages = (
            trace_to_messages(result, conn)
            if getattr(args, 'sequence', False)
            else None
        )
    finally:
        conn.close()

    if getattr(args, 'sequence', False):
        # Plain print (not Rich) so the DOT is captured/piped verbatim.
        print(render_sequence_dot(messages))
    elif getattr(args, 'json', False):
        # Use plain print (not Rich) so capsys captures verbatim.
        print(_json.dumps(trace_result_to_dict(result), indent=2))
    else:
        _render_tree(result)

    return 0


HANDLERS: dict = {
    'trace-flow': cmd_trace_flow,
}


__all__ = [
    'HANDLERS',
    'cmd_trace_flow',
    'register_commands',
    'trace_result_to_dict',
]
