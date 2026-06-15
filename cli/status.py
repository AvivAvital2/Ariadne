"""Library reporting CLI commands (``stats``, ``status``, ``usage``, ``gaps``,
``vacuum``).

Extracted from cli/core.py. Read-only reporting over the doc library: stats,
the per-source x content-type status matrix with disk attribution, usage
analytics, gap analysis, and DB vacuum. Wired into the parser via this module's
``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

import testimonials
from config import get_config

if TYPE_CHECKING:
    from library import Library

DEFAULT_DB_PATH = Path('ariadne.db')

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register library reporting commands."""

    # stats
    stats_parser = subparsers.add_parser('stats', help='Show library statistics')
    stats_parser.add_argument('--by-source', action='store_true',
                              help='Show per-source document and size breakdown')

    # status — per-source × per-content-type matrix + disk attribution
    subparsers.add_parser(
        'status',
        help='Per-source content-type breakdown + disk attribution',
    )

    # usage
    usage_parser = subparsers.add_parser('usage', help='Show usage statistics')
    usage_parser.add_argument('--days', '-d', type=int, default=30,
                              help='Number of days to include (default: 30)')
    usage_parser.add_argument('--tool', '-t', default=None,
                              help='Filter by tool name')
    usage_parser.add_argument('--by-document', action='store_true',
                              help='Show per-document serve counts')
    usage_parser.add_argument('--top-served', type=int, default=None,
                              help='Show top N most-served documents')
    usage_parser.add_argument('--export-report', metavar='PATH', default=None,
                              help='Write a portable analytics report (usage + '
                                   'misses + doc signals) to PATH as JSON, for '
                                   'shipping off-box before replacing the database')

    # gaps
    gaps_parser = subparsers.add_parser('gaps', help='Show documentation gap analysis')
    gaps_parser.add_argument('--days', '-d', type=int, default=30,
                             help='Number of days to include (default: 30)')
    gaps_parser.add_argument('--analyze', action='store_true',
                             help='Run LLM-powered gap analysis')

    # vacuum
    subparsers.add_parser('vacuum', help='Optimize database file size')
    testimonials_parser = subparsers.add_parser(
        'testimonials',
        help='Show the top-scored Q&A testimonials (the local best-of store)')
    testimonials_parser.add_argument('--limit', '-n', type=int, default=20,
        help='Max testimonials to show (capped at 20)')
    testimonials_parser.add_argument('--dir', default=None,
        help='Ariadne working dir holding .ariadne/local/ (default: cwd)')
    testimonials_parser.add_argument('--export', metavar='PATH', default=None,
        help='Copy stored images into PATH (for the showcase)')
    testimonials_parser.add_argument('--export-html', metavar='FILE', default=None,
        help='Write a self-contained HTML showcase page to FILE (diagrams embedded)')


def cmd_stats(args: argparse.Namespace) -> int:
    """Show library statistics."""
    cfg = get_config()
    library = get_library(args.db)

    try:
        total = library.count_documents()
        chunks = library.count_chunks()

        table = Table(title='Library Statistics')
        table.add_column('Metric', style='bold')
        table.add_column('Value', style='cyan')

        table.add_row('Total documents', str(total))
        table.add_row('Total chunks', str(chunks))
        table.add_row('Database path', str(args.db or DEFAULT_DB_PATH))

        # Count by type
        for ct in ('explanation', 'architecture', 'qa', 'diagram', 'finding'):
            count = library.count_documents(content_type=ct)
            if count > 0:
                table.add_row(f'  {ct}', str(count))

        # Show sync state if available
        source_name = cfg.default_source
        if source_name:
            sync_state = library.get_sync_state(source_name)
            if sync_state:
                git_hash, synced_at = sync_state
                table.add_row('Last sync', f'{git_hash[:8]} ({synced_at[:19]})')

        console.print(table)

        if args.by_source:
            by_source = library.stats_by_source()
            total_meta = by_source.pop('_total', {})
            source_table = Table(title='Per-Source Breakdown')
            source_table.add_column('Source', style='bold')
            source_table.add_column('Docs', style='cyan', justify='right')
            source_table.add_column('Content Size', style='cyan', justify='right')
            source_table.add_column('Embedding Size', style='cyan', justify='right')

            for name, data in sorted(by_source.items()):
                source_table.add_row(
                    name,
                    str(data['doc_count']),
                    f"{data['content_size'] / 1024:.0f} KB",
                    f"{data['embedding_size'] / 1024:.0f} KB",
                )
            if total_meta:
                source_table.add_section()
                db_bytes = total_meta.get('db_size_bytes', 0)
                source_table.add_row('DB total', '', f'{db_bytes / 1024 / 1024:.1f} MB', '')

            console.print(source_table)

        return 0

    finally:
        library.close()


def _human_bytes(n: int) -> str:
    """Format a byte count compactly. Matches GNU ``ls -h`` style."""
    if n < 1024:
        return f'{n} B'
    for unit in ('KB', 'MB', 'GB', 'TB'):
        n_f = n / 1024
        if n_f < 1024 or unit == 'TB':
            return f'{n_f:.1f} {unit}'
        n = int(n_f)
    return f'{n} B'  # pragma: no cover


def _attributed_bytes(data: dict) -> int:
    """Sum a per-source stats dict's content + embedding bytes across
    documents, chunks, and sections — the per-source disk attribution."""
    return (
        data['doc_content'] + data['doc_embed']
        + data['chunk_content'] + data['chunk_embed']
        + data['section_content'] + data['section_embed']
    )


def _status_cache_path(library) -> Path:
    """Cache file lives next to the DB so each DB has its own cache.
    Filename pattern: ``{db_filename}.status-cache.json``."""
    return library.path.parent / f'{library.path.name}.status-cache.json'


def _load_status_cache(path: Path) -> dict[str, dict]:
    """Best-effort load. Missing or corrupt → empty cache (next run
    just recomputes)."""
    if not path.exists():
        return {}
    import json
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def _save_status_cache(path: Path, cache: dict) -> None:
    """Best-effort save. Failure to write is non-fatal — the cache is
    a performance hint, not load-bearing data."""
    import json
    try:
        path.write_text(json.dumps(cache, indent=2), encoding='utf-8')
    except OSError:
        pass


def cmd_status(args: argparse.Namespace) -> int:
    """Show per-source content-type breakdown plus disk attribution.

    Iterates sources one-by-one with a progress bar that shows the
    running total — the chunks-JOIN attribution can take a while on
    large DBs (scalaproject's 484K chunks read 5+ GB of embedding bytes),
    and a blank waiting screen is bad UX. The progress bar advances
    per source with descriptions like
    "scalaproject: 7.3 GB (running 7.4 GB)".

    Final output is a Rich table:
      - One row per source, sorted by attributed bytes descending.
      - Type-count columns — only the content_types actually present.
      - "Attributed" column = docs.content + docs.embed + chunks +
        sections + their embeddings, summed per source.
      - Footer: TOTAL attributed, DB file size, and overhead = the gap
        (SQLite indexes, page slack, freelists, WAL artifacts).
    """
    import os

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
    )

    library = get_library(args.db)

    try:
        # Phase 1 — enumerate sources (fast). Sizes the progress bar.
        sources = library.list_source_names()
        if not sources:
            console.print('[yellow]No documents in library.[/yellow]')
            return 0

        # Phase 2 — per-source attribution with running-total progress.
        # Cache: avoid the expensive chunks JOIN for sources whose
        # signature ({count}|{max_updated_at}) matches a prior run.
        cache_path = _status_cache_path(library)
        cache = _load_status_cache(cache_path)

        results: dict[str, dict] = {}
        running_total = 0

        progress_columns = (
            SpinnerColumn(),
            TextColumn('[bold cyan]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn('·'),
            TimeElapsedColumn(),
        )
        with Progress(
            *progress_columns, console=console, transient=True,
        ) as progress:
            task = progress.add_task(
                'Calculating per-source attribution',
                total=len(sources),
            )
            for src in sources:
                # Set the description BEFORE the work so users see
                # "Calculating scalaproject..." during the slow chunks JOIN
                # rather than a stale label from the previous source.
                progress.update(task, description=f'Calculating {src}…')

                sig = library.source_signature(src)
                cached = cache.get(src)
                if cached and cached.get('cache_key') == sig:
                    stats = cached['stats']
                    suffix = ' [cached]'
                else:
                    stats = library.stats_for_source(src)
                    cache[src] = {'cache_key': sig, 'stats': stats}
                    suffix = ''

                results[src] = stats
                attributed = _attributed_bytes(stats)
                running_total += attributed
                progress.update(
                    task, advance=1,
                    description=(
                        f'Calculating {src}: {_human_bytes(attributed)} '
                        f'(size {_human_bytes(running_total)}){suffix}'
                    ),
                )

        # Purge cache entries for sources that no longer exist (a
        # source got dropped from ariadne.yaml and its docs deleted).
        cache = {k: v for k, v in cache.items() if k in sources}
        _save_status_cache(cache_path, cache)

        # Phase 3 — pull DB file size for the overhead row.
        db_size = (
            os.path.getsize(library.path) if library.path.exists() else 0
        )

        # Phase 4 — render the final table (data-driven type columns).
        all_types: set[str] = set()
        for data in results.values():
            all_types.update(data['by_content_type'].keys())
        type_columns = sorted(all_types)

        table = Table(title='Library status — per source')
        table.add_column('Source', style='bold')
        table.add_column('Docs', style='cyan', justify='right')
        for ct in type_columns:
            table.add_column(ct, style='dim', justify='right')
        table.add_column('Attributed', style='cyan', justify='right')

        sorted_rows = sorted(
            results.items(),
            key=lambda kv: -_attributed_bytes(kv[1]),
        )
        sum_attributed = 0
        for src, data in sorted_rows:
            attributed = _attributed_bytes(data)
            sum_attributed += attributed
            row = [src, str(data['doc_count'])]
            for ct in type_columns:
                count = data['by_content_type'].get(ct, 0)
                row.append(str(count) if count else '·')
            row.append(_human_bytes(attributed))
            table.add_row(*row)

        overhead = max(db_size - sum_attributed, 0)
        table.add_section()
        table.add_row(
            'TOTAL',
            str(sum(d['doc_count'] for d in results.values())),
            *['' for _ in type_columns],
            _human_bytes(sum_attributed),
        )
        table.add_row(
            'DB file', '',
            *['' for _ in type_columns],
            _human_bytes(db_size),
        )
        table.add_row(
            'Overhead (indexes, slack, WAL)', '',
            *['' for _ in type_columns],
            _human_bytes(overhead),
            style='dim',
        )

        console.print(table)
        return 0
    finally:
        library.close()


def cmd_usage(args: argparse.Namespace) -> int:
    """Show Ariadne usage statistics."""
    library = get_library(args.db)

    try:
        if getattr(args, 'export_report', None):
            from analytics_report import build_analytics_report

            report = build_analytics_report(library, days=args.days)
            Path(args.export_report).write_text(report.to_json())
            console.print(
                f'[green]Wrote analytics report to {args.export_report}[/green] '
                f'({report.usage_summary["total_calls"]} calls, '
                f'{report.usage_summary["total_misses"]} misses, last {args.days}d)'
            )
            return 0

        stats = library.get_usage_stats(days=args.days, tool_name=args.tool)

        table = Table(title=f'Ariadne Usage (last {args.days} days)')
        table.add_column('Metric', style='bold')
        table.add_column('Value', style='cyan')

        table.add_row('Total calls', str(stats['total_calls']))
        table.add_row('Total hits', str(stats['total_hits']))
        table.add_row('Total misses', str(stats['total_misses']))
        table.add_row('Hit rate', f"{stats['hit_rate']:.1%}")
        table.add_row('Avg calls/day', f"{stats['avg_calls_per_day']:.1f}")

        if stats['by_tool']:
            table.add_section()
            for tool, data in stats['by_tool'].items():
                table.add_row(
                    tool,
                    f"{data['calls']} calls, {data['hits']} hits, "
                    f"{data['misses']} misses ({data['hit_rate']:.0%})",
                )

        console.print(table)

        if stats['recent_feedback']:
            fb_table = Table(title='Recent Feedback')
            fb_table.add_column('Time', style='dim')
            fb_table.add_column('Outcome', style='bold')
            fb_table.add_column('Feedback')

            for fb in stats['recent_feedback'][:10]:
                fb_table.add_row(
                    fb['timestamp'][:19],
                    fb['outcome'],
                    fb['feedback'],
                )
            console.print(fb_table)

        if args.by_document or args.top_served:
            limit = args.top_served or 20
            doc_usage = library.usage_by_document(days=args.days, limit=limit)
            if doc_usage:
                doc_table = Table(title=f'Top Served Documents (last {args.days} days)')
                doc_table.add_column('#', style='dim', justify='right')
                doc_table.add_column('Title', style='bold')
                doc_table.add_column('Type', style='dim')
                doc_table.add_column('Serves', style='cyan', justify='right')

                for i, d in enumerate(doc_usage, 1):
                    doc_table.add_row(
                        str(i), d['title'], d['content_type'], str(d['serve_count']),
                    )
                console.print(doc_table)
            else:
                console.print('[dim]No per-document tracking data yet (needs new searches).[/dim]')

        return 0

    finally:
        library.close()


def cmd_gaps(args: argparse.Namespace) -> int:
    """Show documentation gap analysis based on miss feedback."""
    library = get_library(args.db)

    try:
        report = library.get_gap_report(days=args.days)

        if report['total_misses'] == 0:
            console.print(f'[green]No misses recorded in the last {args.days} days.[/green]')
            return 0

        table = Table(title=f'Documentation Gaps (last {args.days} days)')
        table.add_column('#', style='dim')
        table.add_column('Gap', style='bold')
        table.add_column('Count', style='cyan')
        table.add_column('Last Seen', style='dim')

        for i, gap in enumerate(report['top_gaps'][:15], 1):
            table.add_row(
                str(i),
                gap['feedback'],
                str(gap['count']),
                gap['last_seen'][:10],
            )

        console.print(table)
        console.print(
            f"\nTotal misses: {report['total_misses']} "
            f"(miss rate: {report['miss_rate']:.1%})"
        )

        if args.analyze:
            import asyncio
            try:
                from gap_analysis import analyze_gaps
                console.print('\n[bold]Running LLM gap analysis...[/bold]')
                gap_report = asyncio.run(analyze_gaps(report['recent_misses']))
                console.print(f'\n[bold]{gap_report.summary}[/bold]\n')
                for rec in gap_report.recommendations:
                    console.print(
                        f'  [cyan]{rec.theme}[/cyan] ({rec.miss_count} misses)'
                    )
                    console.print(f'    {rec.description}')
                    console.print(f'    [green]→ {rec.recommendation}[/green]')
            except ImportError:
                console.print('[red]LLM analysis unavailable (gap_analysis module not found).[/red]')
            except Exception as e:
                console.print(f'[red]LLM analysis failed: {e}[/red]')

        return 0

    finally:
        library.close()


def cmd_vacuum(args: argparse.Namespace) -> int:
    """Optimize the database file size."""
    library = get_library(args.db)

    try:
        library.vacuum()
        console.print('[green]Database optimized.[/green]')
        return 0

    finally:
        library.close()
def cmd_testimonials(args: argparse.Namespace) -> int:
    """Show the local best-of Q&A testimonials (highest-scored, all-time)."""
    import shutil

    base = Path(args.dir) if getattr(args, 'dir', None) else Path.cwd()
    limit = min(getattr(args, 'limit', None) or testimonials.MAX_KEEP, testimonials.MAX_KEEP)
    entries = testimonials.top(testimonials.local_dir(base), limit=limit)

    export_html = getattr(args, 'export_html', None)
    if export_html:
        from testimonials_html import render_html
        dest = Path(export_html)
        dest.write_text(render_html(entries), encoding='utf-8')
        console.print(f'[green]Wrote {len(entries)} testimonial(s) to {dest}[/green]')
        return 0

    if not entries:
        console.print('[yellow]No testimonials recorded yet.[/yellow]')
        return 0
    for i, t in enumerate(entries, 1):
        console.print()
        console.print(
            f'[bold cyan]#{i}  score {t.score}/10[/bold cyan]  '
            f'[dim]{t.duration_seconds:.1f}s[/dim]')
        if t.permalink:
            console.print(f'[dim]{t.permalink}[/dim]')
        console.print(f'[bold]Q:[/bold] {t.question}')
        console.print(f'[bold]A:[/bold] {t.answer}')
        if t.images:
            console.print(f'[dim]({len(t.images)} image(s))[/dim]')
    export = getattr(args, 'export', None)
    if export:
        dest = Path(export)
        dest.mkdir(parents=True, exist_ok=True)
        copied = 0
        for t in entries:
            for img in t.images:
                shutil.copy(img, dest / f'{t.path.name}-{img.name}')
                copied += 1
        console.print(f'[green]Exported {copied} image(s) to {dest}[/green]')
    return 0


HANDLERS = {
    'stats': lambda args: cmd_stats(args),
    'status': lambda args: cmd_status(args),
    'usage': lambda args: cmd_usage(args),
    'gaps': lambda args: cmd_gaps(args),
    'vacuum': lambda args: cmd_vacuum(args),
    'testimonials': lambda args: cmd_testimonials(args),
}
