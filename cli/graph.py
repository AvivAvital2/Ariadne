"""Graph CLI commands."""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console
from rich.table import Table

console = Console()


def get_library(db_path=None):
    from cli.main import get_library as _get_library
    return _get_library(db_path)


def register_commands(subparsers):
    # graph
    graph_parser = subparsers.add_parser('graph', help='Build, inspect, and visualize the dependency graph')
    graph_parser.add_argument('--build', action='store_true', help='Build/rebuild the graph from source + docs')
    graph_parser.add_argument('--stats', action='store_true', help='Show graph statistics')
    graph_parser.add_argument('--priorities', action='store_true', help='Show files ranked by priority score')
    graph_parser.add_argument('--related', metavar='DOC_ID', help='Show related docs for a specific document')
    graph_parser.add_argument('--export', metavar='PATH', help='Export interactive HTML visualization')
    graph_parser.add_argument('--report', metavar='PATH', help='Export CSV report of nodes with coverage and edges')
    graph_parser.add_argument('--source', '-s', default=None, help='Source name (default from config)')
    graph_parser.add_argument('--limit', type=int, default=20, help='Max results for priorities/related (default: 20)')


def cmd_graph(args: argparse.Namespace) -> int:
    """Build, inspect, and visualize the dependency graph."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source

    try:
        source_path = cfg.resolve_source(source_name) if source_name else None

        if args.build:
            if not source_path:
                console.print('[red]No source specified.[/red]')
                return 1
            console.print(f'Building graph for {source_name}...')
            # Phase 5: passing source_name enables atomic SCIP-edge
            # enrichment when the source has a current SCIP graph.
            counts = library.build_graph(
                source_path, source_name=source_name,
            )
            console.print('[green]Graph built:[/green]')
            for etype, count in sorted(counts.items()):
                console.print(f'  {etype}: {count} edges')
            return 0

        if args.stats:
            stats = library.get_graph_stats()
            table = Table(title='Graph Statistics')
            table.add_column('Metric', style='bold')
            table.add_column('Value', style='cyan')
            table.add_row('Total edges', str(stats['total_edges']))
            table.add_row('Source nodes', str(stats['source_nodes']))
            table.add_row('Target nodes', str(stats['target_nodes']))
            for etype, count in stats['by_type'].items():
                table.add_row(f'  {etype}', str(count))
            console.print(table)
            return 0

        if args.priorities:
            if not source_path:
                console.print('[red]No source specified.[/red]')
                return 1
            priorities = library.get_priorities(source_path)
            table = Table(title='File Priorities (undocumented first)')
            table.add_column('Priority', style='bold', justify='right')
            table.add_column('File')
            table.add_column('Edges', justify='right')
            table.add_column('Docs', justify='right')
            table.add_column('Coverage', justify='right')
            for p in priorities[:args.limit]:
                table.add_row(
                    f'{p["priority_score"]:.1f}',
                    p['file'],
                    str(p['total_edges']),
                    str(p['doc_count']),
                    f'{p["coverage_percent"]:.0f}%',
                )
            console.print(table)
            return 0

        if args.related:
            related = library.get_related(args.related, limit=args.limit)
            if not related:
                console.print(f'No related docs found for {args.related}')
                return 0
            table = Table(title='Related Documents')
            table.add_column('Distance', justify='right')
            table.add_column('Title')
            table.add_column('Type')
            for r in related:
                table.add_row(f'{r["distance"]:.2f}', r['title'], r['content_type'])
            console.print(table)
            return 0

        if args.export:
            from graph_viewer import generate_graph_html
            graph_data = library.export_graph_json()
            out = generate_graph_html(graph_data, Path(args.export))
            console.print(f'[green]Graph exported to {out}[/green]')
            console.print(f'  Nodes: {len(graph_data["nodes"])}, Edges: {len(graph_data["edges"])}')
            return 0

        if args.report:
            import csv
            if not source_path:
                console.print('[red]No source specified.[/red]')
                return 1
            priorities = library.get_priorities(source_path)
            report_path = Path(args.report)
            with open(report_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'file', 'inbound_edges', 'outbound_edges', 'total_edges',
                    'doc_count', 'coverage_percent', 'priority_score',
                ])
                writer.writeheader()
                writer.writerows(priorities)
            console.print(f'[green]Report exported to {report_path} ({len(priorities)} rows)[/green]')
            return 0

        # Default: show help
        console.print('Use --build, --stats, --priorities, --related, --export, or --report')
        return 0

    finally:
        library.close()


HANDLERS = {
    'graph': lambda args: cmd_graph(args),
}
