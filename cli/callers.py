"""``ariadne callers`` / ``ariadne callees`` — Phase 4.

Walks the cross-source graph from a starting symbol N levels deep and
renders the result as a Rich tree. Useful for understanding code:

- ``ariadne callers SYMBOL`` — who calls this?
- ``ariadne callees SYMBOL`` — what does this call?

The walk has cycle detection so recursive call patterns (A calls B,
B calls A) terminate cleanly.

Symbol resolution uses ``CrossSourceGraph.resolve_symbol`` per design
decision #3 — permissive (exact qualified_name → suffix → substring)
with disambiguation lists when multiple candidates tie.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.tree import Tree

if TYPE_CHECKING:
    from docgen.scip_cross_source import CrossSourceEdge, CrossSourceGraph


_console = Console()


# Tree node: (edge, sub_walks). edge is a CrossSourceEdge; sub_walks is
# a list of (edge, sub_walks) tuples representing the next level.
_WalkTree = list[tuple['CrossSourceEdge', list]]


def walk_callers(
    graph: 'CrossSourceGraph', start_id: str, depth: int,
) -> _WalkTree:
    """Walk reverse edges from ``start_id`` up to ``depth`` levels.
    Returns a nested list of ``(edge, sub_walks)`` tuples."""
    return _walk(graph, start_id, depth, direction='callers', visited=set())


def walk_callees(
    graph: 'CrossSourceGraph', start_id: str, depth: int,
) -> _WalkTree:
    """Walk forward edges from ``start_id`` up to ``depth`` levels."""
    return _walk(graph, start_id, depth, direction='callees', visited=set())


def _walk(
    graph: 'CrossSourceGraph',
    start_id: str,
    depth: int,
    *,
    direction: str,
    visited: set[str],
) -> _WalkTree:
    if depth <= 0 or start_id in visited:
        return []
    visited = visited | {start_id}
    edges = (
        graph.callers_of(start_id)
        if direction == 'callers'
        else graph.callees_of(start_id)
    )
    out: _WalkTree = []
    for edge in edges:
        next_id = (
            edge.caller.canonical_id if direction == 'callers'
            else edge.callee.canonical_id
        )
        sub = _walk(
            graph, next_id, depth - 1, direction=direction,
            visited=visited,
        )
        out.append((edge, sub))
    return out


def _render_walk(
    label: str, walk: _WalkTree, *, direction: str,
) -> Tree:
    """Build a Rich tree from the walk result."""
    root = Tree(f'[bold]{label}[/bold]')
    _attach(root, walk, direction)
    return root


def _attach(node: Tree, walk: _WalkTree, direction: str) -> None:
    for edge, sub in walk:
        if direction == 'callers':
            sym = edge.caller
        else:
            sym = edge.callee
        line = (
            f'[cyan]{sym.display_name or sym.canonical_id}[/cyan]  '
            f'[dim]({sym.source_name}/{sym.language})[/dim]  '
            f'[yellow]{edge.file}:{edge.line}[/yellow]'
        )
        child = node.add(line)
        _attach(child, sub, direction)


def _resolve_or_exit(graph, symbol_query: str) -> str | None:
    """Resolve a symbol query via graph.resolve_symbol. Prints
    diagnostics on failure. Returns the canonical_id, or None if
    the caller should exit non-zero."""
    resolution = graph.resolve_symbol(symbol_query)
    if resolution.symbol is not None:
        if resolution.match_tier != 'exact':
            _console.print(
                f'[dim](matched as {resolution.match_tier} of '
                f'{resolution.symbol.qualified_name})[/dim]',
            )
        return resolution.symbol.canonical_id

    if resolution.candidates:
        _console.print(
            f'[red]Ambiguous symbol "{symbol_query}" — '
            f'{len(resolution.candidates)} candidates at '
            f'{resolution.match_tier} tier:[/red]',
        )
        for c in resolution.candidates[:10]:
            _console.print(
                f'  [cyan]{c.qualified_name}[/cyan]  '
                f'[dim]({c.source_name}/{c.language}, {c.kind})[/dim]',
            )
        if len(resolution.candidates) > 10:
            _console.print(
                f'  [dim]... and {len(resolution.candidates) - 10} more[/dim]',
            )
        return None

    _console.print(
        f'[red]No symbol matches "{symbol_query}".[/red]',
    )
    return None


def _load_graph(db_path: Path | str | None):
    from config import get_config
    from docgen.scip_cross_source import CrossSourceGraph
    from library import Library

    if db_path is None:
        db_path = Path(get_config().db_path)
    library = Library(Path(db_path))
    graph = CrossSourceGraph()
    try:
        with library._conn_provider.acquire() as conn:
            graph.load_from(conn)
    finally:
        library.close()
    return graph


def cmd_callers(args: argparse.Namespace) -> int:
    """Render reverse-edge tree from a starting symbol."""
    graph = _load_graph(args.db)
    canonical_id = _resolve_or_exit(graph, args.symbol)
    if canonical_id is None:
        return 1

    walk = walk_callers(graph, canonical_id, depth=args.depth)
    if not walk:
        _console.print(
            f'[yellow]No callers found for "{args.symbol}".[/yellow]',
        )
        return 0

    sym = graph._symbols[canonical_id]
    tree = _render_walk(
        f'{sym.qualified_name}  ({sym.source_name}/{sym.language})',
        walk,
        direction='callers',
    )
    _console.print(tree)
    return 0


from attrs import field, frozen


@frozen
class ImpactReport:
    """Aggregated reverse-edge walk: which files and symbols are
    affected if the starting symbol changes."""
    start_symbol: object  # CrossSourceSymbol
    affected_symbols: list  # list[CrossSourceSymbol] in walk order
    files: set[str] = field(factory=set)


def compute_impact_radius(
    graph: 'CrossSourceGraph', start_id: str, depth: int,
) -> ImpactReport:
    """Walk reverse edges N-deep, collect every symbol and file that
    transitively depends on the starting symbol."""
    if start_id not in graph._symbols:
        # Empty report — caller decides whether to error
        return ImpactReport(
            start_symbol=None,
            affected_symbols=[],
            files=set(),
        )

    start_sym = graph._symbols[start_id]

    visited: set[str] = set()
    # The starting symbol IS affected — it's the thing being changed.
    affected: list = [start_sym]

    def _walk(sym_id: str, d: int) -> None:
        if d <= 0 or sym_id in visited:
            return
        visited.add(sym_id)
        for edge in graph.callers_of(sym_id):
            caller_id = edge.caller.canonical_id
            if caller_id not in visited:
                affected.append(edge.caller)
            _walk(caller_id, d - 1)

    _walk(start_id, depth)

    files = {sym.file for sym in affected}
    return ImpactReport(
        start_symbol=start_sym,
        affected_symbols=affected,
        files=files,
    )


def format_dead_code_report(
    graph: 'CrossSourceGraph', source_name: str,
) -> str | None:
    """Build a human-readable report of zero-reference symbols in a
    source. Returns None if the source has no dead code (or doesn't
    exist in the graph) — caller skips the section entirely.

    Used by ``cmd_improve --dead-code`` to inject a section into its
    existing multi-step output. The report is plain text (Rich-tagged)
    so it slots into either Rich console output or a plain dump.
    """
    if not graph.has_scip(source_name):
        return None
    unused = graph.symbols_with_zero_references(source_name)
    if not unused:
        return None

    lines: list[str] = [
        '[yellow]Dead code signals (require manual review):[/yellow]',
    ]
    for sym in sorted(unused, key=lambda s: (s.file, s.line_start)):
        lines.append(
            f'  [cyan]{sym.qualified_name}[/cyan] — '
            f'{sym.kind} in {sym.file}:{sym.line_start} '
            f'([dim]0 references in indexed sources[/dim])',
        )
    lines.append(
        '[dim]  False positives are common: tests, reflective access, '
        'public APIs called externally, and entry-point ``main`` '
        'functions. Manually review before deleting.[/dim]',
    )
    return '\n'.join(lines)


def cmd_impact_radius(args: argparse.Namespace) -> int:
    """Show what would be affected by a change to a symbol."""
    graph = _load_graph(args.db)
    canonical_id = _resolve_or_exit(graph, args.symbol)
    if canonical_id is None:
        return 1

    report = compute_impact_radius(graph, canonical_id, depth=args.depth)
    if not report.affected_symbols:
        _console.print(
            f'[yellow]No callers (no impact) for "{args.symbol}".[/yellow]',
        )
        return 0

    sym = report.start_symbol
    _console.print(
        f'[bold]Impact radius of {sym.qualified_name}[/bold] '
        f'(depth={args.depth})',
    )
    _console.print(
        f'  [cyan]{len(report.affected_symbols)}[/cyan] symbol(s) '
        f'affected across [cyan]{len(report.files)}[/cyan] file(s):',
    )
    for f in sorted(report.files):
        _console.print(f'  [yellow]{f}[/yellow]')
    return 0


def cmd_callees(args: argparse.Namespace) -> int:
    """Render forward-edge tree from a starting symbol."""
    graph = _load_graph(args.db)
    canonical_id = _resolve_or_exit(graph, args.symbol)
    if canonical_id is None:
        return 1

    walk = walk_callees(graph, canonical_id, depth=args.depth)
    if not walk:
        _console.print(
            f'[yellow]No callees found for "{args.symbol}".[/yellow]',
        )
        return 0

    sym = graph._symbols[canonical_id]
    tree = _render_walk(
        f'{sym.qualified_name}  ({sym.source_name}/{sym.language})',
        walk,
        direction='callees',
    )
    _console.print(tree)
    return 0


def register_commands(subparsers):
    """Register `ariadne callers`, `callees`, and `impact_radius`
    subcommands."""
    callers_parser = subparsers.add_parser(
        'callers',
        help='Show what calls a symbol (cross-source)',
    )
    callers_parser.add_argument('symbol', help='Symbol to query')
    callers_parser.add_argument(
        '--depth', type=int, default=5,
        help='Max depth to walk (default: 5)',
    )
    callers_parser.add_argument(
        '--source', '-s', default=None,
        help='Restrict to a specific source (optional)',
    )

    callees_parser = subparsers.add_parser(
        'callees',
        help='Show what a symbol calls (cross-source)',
    )
    callees_parser.add_argument('symbol', help='Symbol to query')
    callees_parser.add_argument(
        '--depth', type=int, default=5,
        help='Max depth to walk (default: 5)',
    )
    callees_parser.add_argument(
        '--source', '-s', default=None,
        help='Restrict to a specific source (optional)',
    )

    impact_parser = subparsers.add_parser(
        'impact_radius',
        help='Show what files/symbols are affected by a change to a symbol',
    )
    impact_parser.add_argument('symbol', help='Symbol to query')
    impact_parser.add_argument(
        '--depth', type=int, default=5,
        help='Max depth to walk reverse edges (default: 5)',
    )
    impact_parser.add_argument(
        '--source', '-s', default=None,
        help='Restrict to a specific source (optional)',
    )


HANDLERS = {
    'callers': lambda args: cmd_callers(args),
    'callees': lambda args: cmd_callees(args),
    'impact_radius': lambda args: cmd_impact_radius(args),
}


__all__ = [
    'HANDLERS',
    'ImpactReport',
    'cmd_callees',
    'cmd_callers',
    'cmd_impact_radius',
    'compute_impact_radius',
    'format_dead_code_report',
    'register_commands',
    'walk_callees',
    'walk_callers',
]


def format_stale_autodoc_report(dangling):
    """Report rst autodoc targets that no longer resolve to a code symbol --
    the docs reference code that was renamed or removed. ``dangling`` is the
    ``(rst_section, target)`` list from ``dangling_autodoc``. Returns None when
    empty so the caller skips the section.
    """
    if not dangling:
        return None
    lines = [
        '[yellow]\u26a0\ufe0f  Stale documentation (autodoc targets no longer resolve):[/yellow]',
    ]
    for section, target in sorted(dangling):
        lines.append(
            f'  [cyan]{section}[/cyan] documents [red]{target}[/red] '
            '-- not found in any indexed source',
        )
    lines.append(
        '[dim]  The rst references code that was renamed or removed; '
        'update the autodoc directive or the docs.[/dim]',
    )
    return '\n'.join(lines)
