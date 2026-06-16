"""Documentation-generation CLI commands: topic, improve, docs, notify-changed."""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from config import get_config

if TYPE_CHECKING:
    from library import Library

# Default paths
DEFAULT_DB_PATH = Path('ariadne.db')

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


# Re-exports — the ``generate`` command's helpers and entry point now live
# in cli/generate.py. Kept importable from here for backwards-compatibility
# with tests and external callers.
from cli.generate import (  # noqa: E402
    cmd_generate,
)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register generation commands."""
    # generate — now lives in cli/generate.py
    from cli.generate import register_generate_parser
    register_generate_parser(subparsers)

    # batch — pending-batch management (lives in cli/batch.py).
    from cli.batch import register_batch_parser
    register_batch_parser(subparsers)

    # topic
    topic_parser = subparsers.add_parser('topic', help='Generate a cross-cutting topic doc from multiple source files')
    topic_parser.add_argument('title', help='Topic title (e.g., "Ingest Pipeline")')
    topic_parser.add_argument('--files', '-f', nargs='+', required=True,
                              help='Source files to include (relative to source root)')
    topic_parser.add_argument('--description', '-d', default='',
                              help='Brief description of the topic')
    topic_parser.add_argument('--source', '-s', default=None,
                              help='Source name (default from config)')
    topic_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be generated without saving')

    # improve
    improve_parser = subparsers.add_parser('improve', help='Run improvement cycle: analyze gaps, generate missing docs')
    improve_parser.add_argument('--source', '-s', default=None,
                                help='Source name (default from config)')
    improve_parser.add_argument('--max-files', type=int, default=10,
                                help='Max files to generate docs for (default: 10)')
    improve_parser.add_argument('--dry-run', action='store_true',
                                help='Show what would be done without generating')
    improve_parser.add_argument('--days', '-d', type=int, default=30,
                                help='Days of usage data to analyze (default: 30)')
    improve_parser.add_argument('--dead-code', action='store_true',
                                help='Include zero-reference symbol report '
                                     '(requires SCIP indexes)')

    # docs
    docs_parser = subparsers.add_parser('docs', help='Generate user-facing documentation from Ariadne knowledge')
    docs_parser.add_argument('--type', '-t', default='all',
                             help='Doc types to generate: readme, api, architecture, diagrams, all (default: all)')
    docs_parser.add_argument('--output', '-o', default='generated-docs',
                             help='Output directory (default: generated-docs)')
    docs_parser.add_argument('--source', '-s', default=None,
                             help='Filter to a specific source (e.g. pythonproject). Omit for all sources.')
    docs_parser.add_argument('--update-readme', metavar='PATH',
                             help='Update an existing README.md in-place instead of generating to output dir')
    docs_parser.add_argument('--serve', action='store_true',
                             help='Start a local MkDocs server after generating docs')

    # notify-changed
    notify_parser = subparsers.add_parser('notify-changed',
        help='Incremental catalog update for specific changed files')
    notify_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    notify_parser.add_argument('--files', nargs='+', default=None,
        help='Changed file paths relative to source root')
    notify_parser.add_argument('--regenerate', action='store_true', help='Regenerate LLM docs for changed files')


async def cmd_topic(args: argparse.Namespace) -> int:
    """Generate a cross-cutting topic doc from multiple source files."""
    cfg = get_config()
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None

    if not source_path:
        console.print('[red]Could not resolve source path.[/red]')
        return 1

    # Resolve file paths relative to source root
    file_paths = []
    for f in args.files:
        p = Path(f)
        if not p.is_absolute():
            p = source_path / f
        if p.exists():
            file_paths.append(p)
        else:
            console.print(f'[yellow]Warning: file not found: {f}[/yellow]')

    if not file_paths:
        console.print('[red]No valid source files found.[/red]')
        return 1

    console.print(f'[bold]Generating topic doc: "{args.title}"[/bold]')
    console.print(f'  Files: {len(file_paths)}')
    for p in file_paths:
        rel = p.relative_to(source_path) if p.is_relative_to(source_path) else p
        console.print(f'    - {rel}')

    if args.dry_run:
        console.print(f'\n[yellow]Dry run: would generate topic doc "{args.title}" from {len(file_paths)} files.[/yellow]')
        return 0

    import os

    from docgen.generator import DocGenerator, GeneratorConfig

    api_key = os.environ.get('OPENAI_API_KEY')
    gen_config = GeneratorConfig(
        model=cfg.model,
        api_key=api_key,
    )

    async with DocGenerator(config=gen_config) as generator:
        doc = await generator.generate_for_topic(
            title=args.title,
            description=args.description or f'How {args.title} works across multiple source files.',
            source_files=file_paths,
        )

    if not doc:
        console.print('[red]Topic doc generation failed.[/red]')
        return 1

    # Save to library
    library = get_library(args.db)
    try:
        from writer import LibraryWriter
        async with LibraryWriter(library) as writer:
            saved = await writer.add_document(
                content_type='explanation',
                title=doc.title,
                content=doc.content,
                source_files=list(doc.source_files),
                metadata=doc.metadata,
                source_name=source_name,
            )
        console.print(f'[green]Topic doc created: "{saved.title}" (id: {saved.id})[/green]')
        return 0
    finally:
        library.close()


async def cmd_improve(args: argparse.Namespace) -> int:
    """Run improvement cycle: analyze gaps and generate docs for undocumented files."""
    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source

    if not source_name:
        console.print('[red]No source specified and no default_source in config.[/red]')
        return 1

    try:
        # Step 1: Gap analysis
        console.print('[bold]Step 1: Analyzing gaps...[/bold]')
        gap_data = library.get_gap_report(days=args.days)
        total_misses = gap_data['total_misses']
        top_gaps = gap_data['top_gaps']

        if top_gaps:
            console.print(f'  Found {total_misses} misses in last {args.days} days:')
            for gap in top_gaps[:5]:
                console.print(f'    - "{gap["feedback"]}" ({gap["count"]}x)')
        else:
            console.print('  No misses recorded.')

        # Step 2: Find undocumented files
        console.print('[bold]Step 2: Finding undocumented files...[/bold]')
        source_path = cfg.resolve_source(source_name)
        if not source_path:
            console.print(f'[red]Source path not found for: {source_name}[/red]')
            return 1

        from docgen.staleness import find_catalog_files

        all_files = find_catalog_files(
            source_path,
            exclude_dir_names=cfg.resolve_excluded_dirs(source_name),
        )
        documented_files = set()
        for doc in library.list_documents():
            for sf in doc.source_files:
                documented_files.add(sf)

        undocumented = [
            f for f in all_files
            if str(f) not in documented_files
            and str(f.relative_to(source_path.parent) if f.is_relative_to(source_path.parent) else f) not in documented_files
            and f.name != '__init__.py'
        ]

        console.print(f'  Total catalog files: {len(all_files)}')
        console.print(f'  Documented: {len(all_files) - len(undocumented)}')
        console.print(f'  Undocumented: {len(undocumented)}')

        # Step 3: Per-document usage stats
        console.print('[bold]Step 3: Document usage stats...[/bold]')
        doc_usage = library.usage_by_document(days=args.days, limit=10)
        if doc_usage:
            console.print('  Top served docs:')
            for d in doc_usage[:5]:
                console.print(f'    - {d["title"]} ({d["serve_count"]}x)')
        else:
            console.print('  No per-document tracking data yet.')

        # Step 3c: Quality signals
        low_value = library.find_low_value_documents(min_serves=2, days=args.days)
        if low_value:
            console.print('[bold]Step 3c: Low-value docs (served but never hit):[/bold]')
            for d in low_value[:5]:
                console.print(f'    - "{d["title"]}" ({d["serve_count"]}x served, {d["content_size"]} chars)')

        oversized = library.find_oversized_documents(max_chars=15000)
        if oversized:
            console.print(f'[bold]Step 3d: Oversized docs (>{15000} chars, consider splitting):[/bold]')
            for d in oversized[:5]:
                console.print(f'    - "{d["title"]}" ({d["content_size"]:,} chars)')

        # Phase 4 slice C: dead-code signals from the cross-source SCIP
        # graph. Off by default (requires --dead-code) so improve runs
        # quickly when the graph is large.
        if getattr(args, 'dead_code', False):
            from cli.callers import format_dead_code_report
            from docgen.scip_cross_source import CrossSourceGraph

            console.print('[bold]Step 3e: Dead code signals[/bold]')
            graph = CrossSourceGraph()
            with library._conn_provider.acquire() as conn:
                graph.load_from(conn)
            dead_code = format_dead_code_report(graph, source_name)
            if dead_code:
                console.print(dead_code)
            else:
                console.print(
                    '  [dim]No dead code found (or source not in '
                    'SCIP graph — run `ariadne index --source X`).[/dim]'
                )
            src_path = cfg.get_source_path(source_name)
            if src_path is not None:
                from cli.callers import format_stale_autodoc_report
                from docgen.scip_persist import dangling_autodoc
                sc = cfg.get_source_config(source_name)
                ign = sc.ignore_staleness if sc else False
                stale = format_stale_autodoc_report(
                    dangling_autodoc(src_path, graph, ignore_staleness=ign))
                if stale:
                    console.print(stale)

        # Step 4: Prioritize using graph if available
        graph_stats = library.get_graph_stats()
        if graph_stats['total_edges'] > 0:
            console.print('[bold]Step 3b: Using graph priorities...[/bold]')
            priorities = library.get_priorities(source_path)
            # Filter to undocumented files only, keep priority order
            undoc_set = {str(f) for f in undocumented}
            priority_files = []
            for p in priorities:
                full_path = source_path / p['file']
                if str(full_path) in undoc_set or p['file'] in undoc_set:
                    priority_files.append(full_path)
            files_to_gen = priority_files[:args.max_files]
            if files_to_gen:
                console.print(f'  Prioritized by graph connectivity ({graph_stats["total_edges"]} edges)')
        else:
            files_to_gen = undocumented[:args.max_files]
        console.print(f'\n[bold]Step 4: Generate docs for {len(files_to_gen)} files[/bold]')

        if not files_to_gen:
            console.print('[green]All files are documented![/green]')
            return 0

        for f in files_to_gen:
            rel = f.relative_to(source_path) if f.is_relative_to(source_path) else f
            console.print(f'  {"[dim](dry-run)[/dim] " if args.dry_run else ""}{rel}')

        if args.dry_run:
            console.print(f'\n[yellow]Dry run: would generate docs for {len(files_to_gen)} files.[/yellow]')
            return 0

        # Actually generate
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        # SCIP source_config — required for Scala/Java to route through
        # the SCIP extractor. catalog_only_generator now defaults True,
        # so source_config alone is enough to wire the SCIP path.
        scip_config = cfg.get_source_scip_config(source_name)

        gen_config = OrchestratorConfig(
            source_path=source_path,
            db_path=Path(args.db or DEFAULT_DB_PATH),
            model=cfg.model,
            source_name=source_name,
            dry_run=False,
            source_config=scip_config,
        ignore_staleness=cfg.source_ignore_staleness(source_name))

        async with DocGenOrchestrator(gen_config) as orch:
            for f in files_to_gen:
                rel = f.relative_to(source_path) if f.is_relative_to(source_path) else f
                console.print(f'  Generating: {rel}...')
                try:
                    result = await orch._process_file(f)
                    console.print(f'    Created {result.docs_generated} docs')
                except Exception as e:
                    console.print(f'    [red]Failed: {e}[/red]')

        console.print('\n[green]Improvement cycle complete.[/green]')
        return 0

    finally:
        library.close()


async def cmd_docs(args: argparse.Namespace) -> int:
    """Generate user-facing documentation from Ariadne's knowledge base."""
    from doc_generator import DocGenerator

    library = get_library()
    try:
        output_dir = Path(args.output)

        if args.update_readme:
            gen = DocGenerator(library, output_dir, source=getattr(args, 'source', None))
            result = await gen.update_readme_in_place(Path(args.update_readme))
            console.print(f'[green]Updated README: {result}[/green]')
            return 0

        from doc_generator import DOC_TYPES
        doc_types = list(DOC_TYPES) if args.type == 'all' else [t.strip() for t in args.type.split(',')]

        gen = DocGenerator(library, output_dir, source=getattr(args, 'source', None))
        results = await gen.generate(doc_types)

        if results:
            console.print(f'[green]Generated {len(results)} doc(s) in {output_dir}/:[/green]')
            for dtype, path in results.items():
                console.print(f'  {dtype}: {path}')
        else:
            console.print('[yellow]No docs generated.[/yellow]')

        if args.serve and results:
            import subprocess
            console.print('\n[cyan]Starting MkDocs server...[/cyan]')
            # mkdocs expects docs dir to be named 'docs' — use config override
            subprocess.run(
                ['mkdocs', 'serve', '-f', str(output_dir / 'mkdocs.yml'), '--dev-addr', '127.0.0.1:8000'],
                cwd=output_dir,
            )

        return 0
    finally:
        library.close()




async def cmd_notify_changed(args: argparse.Namespace) -> int:
    """Notify Ariadne that a batch of files has changed in a source."""
    from docgen.catalog_writer import notify_changed
    from writer import LibraryWriter

    cfg = get_config()
    source_name = args.source or cfg.default_source
    if source_name is None:
        console.print('[red]No source specified and no default_source in config.[/red]')
        return 1
    source_path = cfg.resolve_source(source_name)
    if source_path is None or not source_path.exists():
        console.print(f'[red]Source path not found: {source_path}[/red]')
        return 1

    files = list(args.files or [])
    if not files:
        console.print('[yellow]No files specified. Nothing to do.[/yellow]')
        return 0

    # SCIP source_config for Scala/Java incremental sync.
    scip_config = cfg.get_source_scip_config(source_name)

    library = get_library(args.db)
    try:
        async with LibraryWriter(library) as writer:
            summary = await notify_changed(
                library, writer, source_name, files,
                source_root=source_path,
                source_config=scip_config,
            )

        total_added = sum(s.get('added', 0) for s in summary.values())
        total_modified = sum(s.get('modified', 0) for s in summary.values())
        total_removed = sum(s.get('removed', 0) for s in summary.values())
        total_moved = sum(s.get('moved', 0) for s in summary.values())
        total_deleted = sum(1 for s in summary.values() if s.get('deleted'))

        console.print()
        console.print(f'Notify-changed for [cyan]{source_name}[/cyan]:')
        console.print(f'  Files: {len(summary)} ({total_deleted} deleted)')
        console.print(
            f'  Elements: [green]+{total_added}[/green] added, '
            f'[yellow]~{total_modified}[/yellow] modified, '
            f'[red]-{total_removed}[/red] removed, '
            f'{total_moved} moved'
        )

        if getattr(args, 'regenerate', False):
            from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

            orch_config = OrchestratorConfig(
                source_path=source_path,
                db_path=Path(cfg.db_path),
                staleness_db_path=Path(cfg.staleness_db_path),
                model=cfg.model,
                doc_types=('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram'),
                force_regenerate=True,
            ignore_staleness=cfg.source_ignore_staleness(source_name))

            docs_created = 0
            docs_failed = 0
            async with DocGenOrchestrator(orch_config) as orchestrator:
                for rel in files:
                    file_path = source_path / rel if not Path(rel).is_absolute() else Path(rel)
                    if not file_path.exists():
                        continue
                    try:
                        gen = await orchestrator._process_file(file_path)
                        if gen is not None:
                            docs_created += getattr(gen, 'docs_generated', 0)
                            docs_failed += getattr(gen, 'docs_failed', 0)
                    except Exception as e:
                        console.print(f'[red]Regen failed for {rel}: {e}[/red]')
                        docs_failed += 1

            console.print()
            console.print(
                f'[green]Regenerated docs: {docs_created} created, '
                f'{docs_failed} failed[/green]'
            )

        return 0
    finally:
        library.close()


HANDLERS = {
    'generate': lambda args: asyncio.run(cmd_generate(args)),
    'improve': lambda args: asyncio.run(cmd_improve(args)),
    'topic': lambda args: asyncio.run(cmd_topic(args)),
    'docs': lambda args: asyncio.run(cmd_docs(args)),
    'notify-changed': lambda args: asyncio.run(cmd_notify_changed(args)),
    'batch': lambda args: __import__('cli.batch', fromlist=['cmd_batch']).cmd_batch(args),
}
