"""Core CRUD, stats, and metadata CLI commands."""
from __future__ import annotations

import argparse
import asyncio
import functools
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from config import get_config

if TYPE_CHECKING:
    from library import Library
    from schema import Document, SearchResult

# Default paths
DEFAULT_DB_PATH = Path('ariadne.db')
DEFAULT_EXPORT_PATH = Path('docs')

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    """Get or create a library instance."""
    from library import Library
    if db_path is None:
        cfg = get_config()
        db_path = Path(cfg.db_path)
    return Library(db_path)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register core CRUD commands."""
    # search
    search_parser = subparsers.add_parser('search', help='Search for documents')
    search_parser.add_argument('query', help='Search query')
    search_parser.add_argument('-k', type=int, default=5, help='Number of results')
    search_parser.add_argument('--type', choices=['explanation', 'architecture', 'qa', 'diagram', 'finding'],
                               help='Filter by content type')
    search_parser.add_argument('--chunks', action='store_true', help='Search at chunk level')
    search_parser.add_argument('--source', '-s', default=None,
                               help='Source scope for filtering results')
    search_parser.add_argument('--include-all', action='store_true',
                               help='Include all docs regardless of scope')

    # list
    list_parser = subparsers.add_parser('list', help='List documents')
    list_parser.add_argument('--type', choices=['explanation', 'architecture', 'qa', 'diagram', 'finding'],
                             help='Filter by content type')
    list_parser.add_argument('--status', choices=['stable', 'experimental', 'deprecated'],
                             help='Filter by document status')
    list_parser.add_argument('--branch', '-b', help='Filter to docs matching branch pattern')
    list_parser.add_argument('--limit', type=int, help='Maximum documents to show')

    # get
    get_parser = subparsers.add_parser('get', help='Get a document by ID')
    get_parser.add_argument('id', help='Document ID')
    get_parser.add_argument(
        '--source',
        help='Source name to scope the lookup (falls back to cwd, then default_source)',
    )

    # add
    add_parser = subparsers.add_parser('add', help='Add a new document')
    add_parser.add_argument('--type', choices=['explanation', 'architecture', 'qa', 'diagram', 'finding'],
                            default='explanation', help='Content type')
    add_parser.add_argument('--title', required=True, help='Document title')
    add_parser.add_argument('--file', '-f', help='Read content from file')
    add_parser.add_argument('--source-files', help='Comma-separated list of related source files')
    add_parser.add_argument(
        '--source',
        help='Source name attribution (falls back to cwd, then default_source)',
    )

    # finding (quick way to save session insights)
    finding_parser = subparsers.add_parser('finding', help='Save a finding/conclusion for future reference')
    finding_parser.add_argument('finding', help='The finding or conclusion to save')
    finding_parser.add_argument('--topic', '-t', help='Topic/title for the finding (auto-generated if omitted)')
    finding_parser.add_argument('--source-files', '-s', help='Comma-separated list of related source files')
    finding_parser.add_argument(
        '--source',
        help='Source name attribution (falls back to cwd, then default_source)',
    )
    finding_parser.add_argument('--no-embed', action='store_true',
                                help="Skip embedding generation (faster, but finding won't be searchable until rebuild)")

    # delete
    delete_parser = subparsers.add_parser('delete', help='Delete a document')
    delete_parser.add_argument('id', help='Document ID')
    delete_parser.add_argument('--force', '-f', action='store_true', help='Skip confirmation')
    delete_parser.add_argument(
        '--source',
        help='Source name to scope the lookup (falls back to cwd, then default_source)',
    )

    # export
    export_parser = subparsers.add_parser('export', help='Export to markdown')
    export_parser.add_argument('output', nargs='?', help='Output directory (default: docs_base/source)')
    export_parser.add_argument('--source', '-s', help='Source name for docs path (default from config)')

    # import
    import_parser = subparsers.add_parser('import', help='Import from markdown')
    import_parser.add_argument('input', nargs='?', help='Input directory (default: docs_base/source)')
    import_parser.add_argument('--source', '-s', help='Source name for docs path (default from config)')
    import_parser.add_argument('--skip-embeddings', action='store_true',
                               help='Skip embedding regeneration')

    # export-db (single-source standalone slice)
    from export_db import DEFAULT_EMBEDDING_MODEL
    exdb = subparsers.add_parser('export-db', help='Export a single source as a standalone slice DB')
    exdb.add_argument('--source', '-s', help='Source to slice (default from config)')
    exdb.add_argument('--out', '-o', required=True, help='Output bundle path (a standalone ariadne.db)')
    exdb.add_argument('--no-embeddings', action='store_true',
                      help='Omit embeddings; recipient runs `ariadne rebuild` on the bundle')
    exdb.add_argument('--with-scip', action='store_true',
                      help='Also carry the SCIP call graph (callers/impact_radius)')

    # import-db (merge a slice into this database)
    imdb = subparsers.add_parser('import-db', help='Merge a slice DB into this database')
    imdb.add_argument('bundle', help='Path to the slice DB to import')
    imdb.add_argument('--on-conflict', choices=['replace', 'skip', 'fail'], default='replace',
                      help='What to do when a document id already exists (default: replace)')
    imdb.add_argument('--embedding-model', default=DEFAULT_EMBEDDING_MODEL,
                      help='Embedding model this database uses (bundle must match)')

    # rebuild
    subparsers.add_parser('rebuild', help='Rebuild all embeddings')

    # stats
    stats_parser = subparsers.add_parser('stats', help='Show library statistics')
    stats_parser.add_argument('--by-source', action='store_true',
                              help='Show per-source document and size breakdown')

    # status — per-source × per-content-type matrix + disk attribution
    subparsers.add_parser(
        'status',
        help='Per-source content-type breakdown + disk attribution',
    )

    # discover — walk a source tree, identify SCIP indexer locations,
    # write .ariadne/manifest.json, ensure .gitignore excludes .ariadne/.
    discover_parser = subparsers.add_parser(
        'discover',
        help=(
            'Walk a source tree, detect SCIP indexer locations, write '
            'manifest.json'
        ),
    )
    discover_parser.add_argument(
        '--source', '-s', help='Source name to discover',
    )
    discover_parser.add_argument(
        '--all', action='store_true',
        help='Run discover on every source in ariadne.yaml',
    )
    discover_parser.add_argument(
        '--dry-run', action='store_true',
        help='Print the manifest without writing files',
    )
    discover_parser.add_argument(
        '--review', action='store_true',
        help=(
            'Walk suspect entries (vendor bundles, mass-duplicate dirs) '
            'and prompt y/N. Rejected entries get appended to the source\'s '
            '`exclude:` list in ariadne.yaml.'
        ),
    )
    discover_parser.add_argument(
        '--config-only', action='store_true',
        help=(
            'Write manifest + ariadne.yaml auto-managed block, but skip '
            'the indexer-run step. Used by sync when it detects a new '
            'language extension, so config stays current without paying '
            'the scip-X cost.'
        ),
    )

    # index — invoke per-language SCIP indexers, merge into one .scip
    index_parser = subparsers.add_parser(
        'index',
        help=(
            'Run SCIP indexers per the manifest, merge intermediates '
            'into <source>/.ariadne/index.scip'
        ),
    )
    index_parser.add_argument(
        '--source', '-s', help='Source name to index',
    )
    index_parser.add_argument(
        '--all', action='store_true',
        help='Run index on every source in ariadne.yaml',
    )
    index_parser.add_argument(
        '--dry-run', action='store_true',
        help='Print what would run without invoking indexers',
    )
    index_parser.add_argument(
        '--kind', choices=['python', 'typescript', 'java'],
        help='Run only entries of this indexer kind (skip the rest)',
    )
    index_parser.add_argument(
        '--force', '-f', action='store_true',
        help='Re-index even if the .scip artifact is still fresh (younger '
             'than the source max_staleness_days). Without this, a fresh '
             'index is reused so re-runs (e.g. onboard --approve) skip the '
             'slow indexers.',
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

    # gaps
    gaps_parser = subparsers.add_parser('gaps', help='Show documentation gap analysis')
    gaps_parser.add_argument('--days', '-d', type=int, default=30,
                             help='Number of days to include (default: 30)')
    gaps_parser.add_argument('--analyze', action='store_true',
                             help='Run LLM-powered gap analysis')

    # vacuum
    subparsers.add_parser('vacuum', help='Optimize database file size')

    # tag
    tag_parser = subparsers.add_parser('tag', help='Tag a document with metadata')
    tag_parser.add_argument('id', help='Document ID to tag')
    tag_parser.add_argument('--status', choices=['stable', 'experimental', 'deprecated'],
                            help='Set document status')
    tag_parser.add_argument('--branch', '-b', help='Add branch pattern')
    tag_parser.add_argument('--feature', '-f', help='Set feature name')
    tag_parser.add_argument('--alias', '-a', help='Add alias')
    tag_parser.add_argument('--remove-branch', help='Remove a branch pattern')
    tag_parser.add_argument(
        '--source',
        help='Source name to scope the lookup (falls back to cwd, then default_source)',
    )
    tag_parser.add_argument('--clear', action='store_true', help='Clear all branch/status metadata')
    
    # build-matrix
    build_matrix_parser = subparsers.add_parser('build-matrix', help='Pre-generate the embedding matrix artifact from the current DB (optional; copy it next to ariadne.db on the serving box)')
    build_matrix_parser.add_argument('--recreate', action='store_true', help='Remove the existing matrix and rebuild it (forces a fresh build; prompts unless --yes)')
    build_matrix_parser.add_argument('--yes', '-y', action='store_true', help='Skip the --recreate confirmation prompt')


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------


async def cmd_search(args: argparse.Namespace) -> int:
    """Search the library for documents matching a query."""
    from embedding import EmbeddingService

    cfg = get_config()
    library = get_library(args.db)

    try:
        async with EmbeddingService() as service:
            query_embedding = await service.embed(args.query)

        # Determine scope filtering
        source_paths: list[Path] | None = None
        source_name = None
        if not args.include_all:
            # Get source scope from args or auto-detect
            source_name = args.source or cfg.default_source
            if source_name:
                # Get source path and dependency paths
                scope_paths: list[Path] = []
                source_path = cfg.get_source_path(source_name)
                if source_path:
                    scope_paths.append(source_path)
                # Add dependency paths
                for dep in cfg.get_effective_dependencies(source_name):
                    dep_path = cfg.get_source_path(dep)
                    if dep_path:
                        scope_paths.append(dep_path)
                if scope_paths:
                    source_paths = scope_paths

        if args.chunks:
            results = library.search_chunks(query_embedding, k=args.k)
            # Filter chunk results by scope if needed
            if source_paths:
                results = [
                    r for r in results
                    if library.filter_documents_by_scope([r.document], source_paths)
                ]
        elif source_paths:
            results = library.search_with_scope(
                query_embedding,
                source_paths=source_paths,
                k=args.k,
                content_type=args.type,
            )
        else:
            results = library.search(query_embedding, k=args.k, content_type=args.type)

        # Apply conflict resolution to filter out superseded docs
        if results and source_name:
            source_precedence = [source_name] + cfg.get_effective_dependencies(source_name)
            resolved_docs = library.resolve_conflicts(
                [r.document for r in results],
                source_precedence=source_precedence,
            )
            resolved_ids = {d.id for d in resolved_docs}
            results = [r for r in results if r.document.id in resolved_ids]

        if not results:
            console.print('[yellow]No results found.[/yellow]')
            return 0

        _print_search_results(results)
        return 0

    finally:
        library.close()


def _print_search_results(results: list['SearchResult']) -> None:
    """Print search results in a formatted table."""
    table = Table(title='Search Results')
    table.add_column('Score', style='cyan', width=8)
    table.add_column('Type', style='magenta', width=12)
    table.add_column('Title', style='green')
    table.add_column('Preview', style='dim')

    for r in results:
        preview = r.document.content[:100].replace('\n', ' ') + '...'
        if r.chunk:
            preview = r.chunk.content[:100].replace('\n', ' ') + '...'

        table.add_row(
            f'{r.score:.4f}',
            r.document.content_type,
            r.document.title,
            preview,
        )

    console.print(table)


def cmd_list(args: argparse.Namespace) -> int:
    """List documents in the library."""
    library = get_library(args.db)

    try:
        docs = library.list_documents(content_type=args.type, limit=None)  # Get all, filter below

        # Filter by status if specified
        if hasattr(args, 'status') and args.status:
            docs = [d for d in docs if d.metadata.get('status', 'stable') == args.status]

        # Filter by branch pattern if specified
        if hasattr(args, 'branch') and args.branch:
            from library import filter_by_branch
            docs = filter_by_branch(docs, args.branch)

        # Apply limit after filtering
        if args.limit:
            docs = docs[:args.limit]

        if not docs:
            console.print('[yellow]No documents found.[/yellow]')
            return 0

        _print_document_list(docs, show_status=hasattr(args, 'status'))
        return 0

    finally:
        library.close()


def _print_document_list(docs: list['Document'], show_status: bool = False) -> None:
    """Print document list in a formatted table."""
    table = Table(title=f'Documents ({len(docs)})')
    table.add_column('ID', style='dim', width=36)
    table.add_column('Type', style='magenta', width=12)
    if show_status:
        table.add_column('Status', style='yellow', width=12)
    table.add_column('Title', style='green')
    table.add_column('Updated', style='cyan', width=20)

    for doc in docs:
        row = [
            doc.id,
            doc.content_type,
        ]
        if show_status:
            row.append(doc.metadata.get('status', 'stable') or 'stable')
        row.append(doc.title)
        row.append(doc.updated_at[:19])
        table.add_row(*row)

    console.print(table)


def cmd_get(args: argparse.Namespace) -> int:
    """Get a specific document by ID."""
    from config import get_config
    from scope_resolution import make_scoped_library

    library = get_library(args.db)

    try:
        try:
            scoped = make_scoped_library(
                get_config(), library,
                getattr(args, 'source', None),
            )
        except LookupError as e:
            console.print(
                f'[red]Cannot resolve source: {e}[/red]\n'
                '[dim]Pass --source explicitly, or set '
                'default_source in ariadne.yaml.[/dim]',
            )
            return 1
        doc = scoped.get_document(args.id)

        if doc is None:
            console.print(f'[red]Document not found: {args.id}[/red]')
            return 1

        console.print(f'[bold]Title:[/bold] {doc.title}')
        console.print(f'[bold]Type:[/bold] {doc.content_type}')
        console.print(f'[bold]ID:[/bold] {doc.id}')
        console.print(f'[bold]Created:[/bold] {doc.created_at}')
        console.print(f'[bold]Updated:[/bold] {doc.updated_at}')

        if doc.source_files:
            console.print('[bold]Source files:[/bold]')
            for sf in doc.source_files:
                console.print(f'  - {sf}')

        console.print()
        console.print('[bold]Content:[/bold]')
        console.print(doc.content)

        return 0

    finally:
        library.close()


async def cmd_add(args: argparse.Namespace) -> int:
    """Add a new document to the library."""
    from scope_resolution import resolve_source_name
    from writer import LibraryWriter

    library = get_library(args.db)

    try:
        # Read content from file or stdin
        if args.file:
            content = Path(args.file).read_text()
        else:
            console.print('Enter content (Ctrl+D to finish):')
            content = sys.stdin.read()

        source_files = args.source_files.split(',') if args.source_files else None
        cfg = get_config()
        source_name = resolve_source_name(
            cfg, getattr(args, 'source', None),
        )
        if source_name is None:
            console.print(
                '[red]Cannot resolve a source for this document. Pass '
                '--source explicitly or set default_source in '
                'ariadne.yaml.[/red]',
            )
            return 1

        async with LibraryWriter(library) as writer:
            doc = await writer.add_document(
                content_type=args.type,
                title=args.title,
                content=content,
                source_files=source_files,
                source_name=source_name,
            )

        console.print(f'[green]Document created: {doc.id}[/green]')
        return 0

    finally:
        library.close()


async def cmd_finding(args: argparse.Namespace) -> int:
    """Save a finding/conclusion to the library for future reference.

    This is a quick way to persist insights discovered during a session.
    """
    from scope_resolution import resolve_source_name

    library = get_library(args.db)

    try:
        # Build content from the finding text
        content = args.finding

        # If topic is provided, use it as title; otherwise extract from finding
        if args.topic:
            title = args.topic
        else:
            # Use first line or first N chars as title
            first_line = content.split('\n')[0].strip()
            title = first_line[:80] if len(first_line) > 80 else first_line

        # Add source files if provided
        source_files = args.source_files.split(',') if args.source_files else None

        cfg = get_config()
        source_name = resolve_source_name(
            cfg, getattr(args, 'source', None),
        )
        if source_name is None:
            console.print(
                '[red]Cannot resolve a source for this finding. Pass '
                '--source explicitly or set default_source in '
                'ariadne.yaml.[/red]',
            )
            return 1

        if args.no_embed:
            # Quick save without embeddings
            doc = library.add_document(
                content_type='finding',
                title=title,
                content=content,
                source_files=source_files,
                source_name=source_name,
            )
        else:
            # Full save with embeddings
            from writer import LibraryWriter
            async with LibraryWriter(library) as writer:
                doc = await writer.add_document(
                    content_type='finding',
                    title=title,
                    content=content,
                    source_files=source_files,
                    source_name=source_name,
                )

        console.print(f'[green]Finding saved: {doc.id}[/green]')
        console.print(f'[dim]Title: {title}[/dim]')
        if args.no_embed:
            console.print('[dim]Note: Saved without embedding. Run "ariadne rebuild" later to enable search.[/dim]')

        # Auto-export to markdown
        cfg = get_config()
        source_name = cfg.default_source
        if source_name:
            from export import LibraryExporter
            exporter = LibraryExporter(library)
            output_dir = cfg.resolve_docs_path(source_name)
            source_path = cfg.resolve_source(source_name)
            exporter.export_all(
                output_dir=output_dir,
                source_name=source_name,
                source_path=source_path,
            )
            console.print(f'[dim]Exported to {output_dir}[/dim]')

        return 0

    finally:
        library.close()


def cmd_delete(args: argparse.Namespace) -> int:
    """Delete a document from the library."""
    from config import get_config
    from scope_resolution import make_scoped_library

    library = get_library(args.db)

    try:
        if not args.force:
            try:
                scoped = make_scoped_library(
                    get_config(), library,
                    getattr(args, 'source', None),
                )
            except (LookupError, KeyError) as e:
                console.print(
                    f'[red]Cannot resolve source: {e}[/red]\n'
                    '[dim]Pass --source explicitly, or use --force '
                    'to delete by ID without scope check.[/dim]',
                )
                return 1
            doc = scoped.get_document(args.id)
            if doc is None:
                # Not found OR out of the resolved scope. Refuse to
                # silently delete a doc the user can't even see —
                # confirmation discipline must hold.
                console.print(
                    f'[red]Document not found in scope: {args.id}[/red]\n'
                    '[dim]The doc may belong to another source. Pass '
                    '--source <name> or --force to delete by ID.[/dim]',
                )
                return 1
            console.print(f'About to delete: [bold]{doc.title}[/bold]')
            if not console.input('Are you sure? [y/N] ').lower().startswith('y'):
                console.print('[yellow]Cancelled.[/yellow]')
                return 0

        if library.delete_document(args.id):
            console.print(f'[green]Document deleted: {args.id}[/green]')
            return 0
        else:
            console.print(f'[red]Document not found: {args.id}[/red]')
            return 1

    finally:
        library.close()


def cmd_export(args: argparse.Namespace) -> int:
    """Export library to markdown files."""
    from export import LibraryExporter

    cfg = get_config()
    source = args.source or cfg.default_source

    library = get_library(args.db)

    try:
        exporter = LibraryExporter(library)

        if args.output:
            output_dir = Path(args.output)
        elif source:
            output_dir = cfg.resolve_docs_path(source)
        else:
            output_dir = Path(DEFAULT_EXPORT_PATH)

        # Get source path for CLAUDE.md generation
        source_path = cfg.resolve_source(source) if source else None

        # Get effective dependencies (including parent if set)
        dependencies: list[str] | None = None
        if source:
            dependencies = cfg.get_effective_dependencies(source)

        paths = exporter.export_all(
            output_dir=output_dir,
            source_name=source,
            source_path=source_path,
            dependencies=dependencies,
        )

        console.print(f'[green]Exported {len(paths)} documents to {output_dir}[/green]')
        return 0

    finally:
        library.close()


def cmd_import_(args: argparse.Namespace) -> int:
    """Import documents from markdown files."""
    from export import import_from_markdown

    cfg = get_config()
    source = args.source or cfg.default_source

    library = get_library(args.db)

    try:
        if args.input:
            input_dir = Path(args.input)
        elif source:
            input_dir = cfg.resolve_docs_path(source)
        else:
            input_dir = Path(DEFAULT_EXPORT_PATH)

        if not input_dir.exists():
            console.print(f'[red]Directory not found: {input_dir}[/red]')
            return 1

        count = import_from_markdown(library, input_dir)

        console.print(f'[green]Imported {count} documents from {input_dir}[/green]')

        if not args.skip_embeddings:
            console.print('Regenerating embeddings...')
            asyncio.run(_rebuild_embeddings(library))
            console.print('[green]Embeddings regenerated.[/green]')

        return 0

    finally:
        library.close()


def cmd_export_db(args: argparse.Namespace) -> int:
    """Export a single source as a standalone slice database."""
    from export_db import export_source_db

    cfg = get_config()
    source = args.source or cfg.default_source
    if not source:
        console.print('[red]No source given and no default_source configured.[/red]')
        return 1
    source_db = args.db or Path(cfg.db_path)
    manifest = export_source_db(
        str(source_db), source, args.out,
        include_embeddings=not args.no_embeddings,
        include_scip=getattr(args, 'with_scip', False),
    )
    console.print(f'[green]Exported slice of {source!r} -> {args.out}[/green]')
    console.print(
        f'  documents={manifest.doc_count} chunks={manifest.chunk_count} '
        f'sections={manifest.section_count} themes={manifest.theme_count} '
        f'edges={manifest.edge_count} embeddings_included={manifest.includes_embeddings}'
    )
    return 0


def cmd_import_db(args: argparse.Namespace) -> int:
    """Merge a slice database into this database."""
    from export_db import import_source_db

    cfg = get_config()
    target_db = args.db or Path(cfg.db_path)
    report = import_source_db(
        str(target_db), args.bundle,
        on_conflict=args.on_conflict,
        expected_embedding_model=args.embedding_model,
    )
    console.print(
        f'[green]Imported {report.documents_merged} docs from {args.bundle} '
        f'(source={report.source_name!r}, conflicts={report.conflicts})[/green]'
    )
    return 0


def cmd_build_matrix(args: argparse.Namespace) -> int:
    """Pre-generate the shared embedding-matrix artifact from the current DB.

    Builds (or refreshes, if stale) ``.ariadne/doc_embeddings.npy`` next to the
    database — a cheap, local read of the embeddings already stored (no
    re-embedding, no API calls). Run it on the build box, then copy the artifact
    to the serving box alongside ``ariadne.db``; the server loads it instead of
    building. The matrix is optional — without it, ranking uses the SQLite path.

    ``--recreate`` removes the existing artifact and rebuilds unconditionally
    (forces a fresh build, and recovers a corrupt file); it prompts for
    confirmation unless ``--yes``.
    """
    from library.embedding_matrix import ARTIFACT_NAME, META_NAME, ensure_matrix, matrix_dir_for

    library = get_library(args.db)
    try:
        matrix_dir = matrix_dir_for(library)
        artifact = matrix_dir / ARTIFACT_NAME
        if getattr(args, 'recreate', False) and artifact.exists():
            if not getattr(args, 'yes', False) and not console.input(
                f'Remove the existing embedding matrix at {artifact} and rebuild it? [y/N] '
            ).strip().lower().startswith('y'):
                console.print('[yellow]Cancelled — existing matrix left in place.[/yellow]')
                return 0
            artifact.unlink()
            (matrix_dir / META_NAME).unlink(missing_ok=True)
        matrix = ensure_matrix(library)
        count = matrix.M.shape[0] if matrix is not None else 0
        if count == 0:
            console.print('[yellow]No embeddings in the database — nothing to build.[/yellow]')
            return 0
        size_mb = artifact.stat().st_size / 1_000_000
        console.print(f'[green]Embedding matrix ready:[/green] {artifact} ({size_mb:.0f} MB, {count} docs)')
        console.print('Copy this file to the serving box alongside ariadne.db.')
        return 0
    finally:
        library.close()


async def cmd_rebuild(args: argparse.Namespace) -> int:
    """Rebuild embeddings for all documents."""
    library = get_library(args.db)

    try:
        await _rebuild_embeddings(library)
        console.print('[green]All embeddings rebuilt.[/green]')
        return 0

    finally:
        library.close()


async def _rebuild_embeddings(library: 'Library') -> None:
    """Rebuild all embeddings, then refresh the shared embedding matrix."""
    from writer import LibraryWriter

    async with LibraryWriter(library) as writer:
        count = await writer.rebuild_all_embeddings()
        console.print(f'Updated {count} documents')

    from library.embedding_matrix import ensure_matrix
    ensure_matrix(library)


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
    return f'{n} B'


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


def cmd_tag(args: argparse.Namespace) -> int:
    """Tag a document with metadata (status, branches, feature, aliases)."""
    from config import get_config
    from scope_resolution import make_scoped_library

    library = get_library(args.db)

    try:
        try:
            scoped = make_scoped_library(
                get_config(), library,
                getattr(args, 'source', None),
            )
        except LookupError as e:
            console.print(
                f'[red]Cannot resolve source: {e}[/red]\n'
                '[dim]Pass --source explicitly, or set '
                'default_source in ariadne.yaml.[/dim]',
            )
            return 1
        doc = scoped.get_document(args.id)
        if doc is None:
            console.print(f'[red]Document not found: {args.id}[/red]')
            return 1

        # Build metadata updates
        metadata = dict(doc.metadata)

        if args.status:
            metadata['status'] = args.status

        if args.branch:
            branches = metadata.get('branches', [])
            if not isinstance(branches, list):
                branches = []
            if args.branch not in branches:
                branches.append(args.branch)
            metadata['branches'] = branches

        if args.feature:
            metadata['feature'] = args.feature

        if args.alias:
            aliases = metadata.get('aliases', [])
            if not isinstance(aliases, list):
                aliases = []
            if args.alias not in aliases:
                aliases.append(args.alias)
            metadata['aliases'] = aliases

        if args.remove_branch:
            branches = metadata.get('branches', [])
            if isinstance(branches, list) and args.remove_branch in branches:
                branches.remove(args.remove_branch)
                metadata['branches'] = branches

        if args.clear:
            # Clear all branch/status metadata
            metadata.pop('status', None)
            metadata.pop('branches', None)
            metadata.pop('feature', None)
            metadata.pop('aliases', None)

        # Update document
        library.update_document(args.id, metadata=metadata)

        console.print(f'[green]Updated metadata for: {doc.title}[/green]')

        # Show current metadata
        console.print('[bold]Current metadata:[/bold]')
        for key, value in metadata.items():
            console.print(f'  {key}: {value}')

        return 0

    finally:
        library.close()


# ---------------------------------------------------------------------------
# Indexer abstraction (Phase 2l)
# ---------------------------------------------------------------------------

# We avoid an ABC/Protocol class to keep the surface light — adapters
# duck-type a single ``run(*, cwd, output, env_hints) -> IndexerResult``
# method.

from attrs import frozen as _frozen


@_frozen
class IndexerResult:
    """What an IndexerAdapter returns from ``run``. Hard-fail semantics
    (decision #5) means any non-success aborts the index command.

    ``resolved_interpreter`` and ``resolution_source`` are populated by
    Python adapters (Phase 2n) so callers can log which interpreter was
    picked; they're optional / empty for non-Python indexers.

    ``vue_mapping_path`` is set by the TypeScript adapter when it ran the
    Vue extractor; ``cmd_index`` records it on the manifest entry as
    ``vue_mapping`` so the loader can translate companion paths back to
    ``.vue`` files. Empty for runs without Vue.
    """
    success: bool
    indexer_version: str = ''
    error_message: str = ''
    resolved_interpreter: Path | None = None
    resolution_source: str = ''
    vue_mapping_path: str = ''


class _SubprocessMerger:
    """Default merger — shells out to the ``scip`` CLI's ``merge``
    subcommand. Returns False if scip is not on PATH or the merge
    fails. Tests substitute a fake.

    Prints an actionable diagnostic on failure rather than swallowing
    it silently. ``scip merge failed`` with no context made debugging
    pointless — users now see whether the binary is missing (install
    hint), whether scip itself returned nonzero (with its stderr), or
    something else.
    """

    def merge(self, inputs: list[Path], output: Path) -> bool:
        import subprocess
        output.parent.mkdir(parents=True, exist_ok=True)
        cmd = (
            ['scip', 'merge', '--output', str(output)]
            + [str(p) for p in inputs]
        )
        try:
            subprocess.run(cmd, check=True, capture_output=True)
            return True
        except FileNotFoundError:
            console.print(
                '[red]scip CLI not on PATH — install with '
                '`brew install sourcegraph/scip/scip` (or '
                'see https://github.com/sourcegraph/scip).[/red]',
            )
            return False
        except subprocess.CalledProcessError as e:
            stderr = e.stderr
            if isinstance(stderr, bytes):
                stderr = stderr.decode('utf-8', errors='replace')
            console.print(
                f'[red]scip merge exited {e.returncode}.[/red]',
            )
            if stderr and stderr.strip():
                console.print(f'[red]{stderr.strip()}[/red]')
            return False


def _default_indexer_registry() -> dict:
    """Lazy-load adapters so cli_core's import doesn't pull docgen at
    module init (avoids circular imports — adapters import IndexerResult
    from this module). Tests bypass this by passing their own registry."""
    from docgen.scip_indexers import (
        JavaIndexerAdapter,
        PythonIndexerAdapter,
        TypescriptIndexerAdapter,
    )
    return {
        'java': JavaIndexerAdapter(),
        'python': PythonIndexerAdapter(),
        'typescript': TypescriptIndexerAdapter(),
    }


def _scope_label(entry: dict, kind: str) -> str:
    """Label for an intermediate .scip file. ``cwd='.'`` becomes just
    the kind; nested cwd flattens slashes for safe filenames."""
    cwd = entry.get('cwd', '.')
    if cwd == '.':
        return kind
    return f'{cwd.replace("/", "-")}-{kind}'


_LANGUAGE_LABELS = {
    'python': 'Python',
    'typescript': 'TypeScript',
    'java': 'Java',
}


def _streams_file_progress(kind: str) -> bool:
    """True if the adapter for ``kind`` emits per-file progress events.

    Only scip-python streams parseable ``current/total`` ticks. scip-java
    and scip-typescript are opaque subprocesses (they compile/index in one
    shot), so their bars can't track files — we show an animated
    indeterminate bar for them instead of a counter frozen at 0/N.
    """
    return kind == 'python'


def _pulse_bar(kind: str) -> bool:
    """True if ``kind``'s bar should be an animated indeterminate (pulse)
    bar rather than a determinate file counter.

    Only scip-java qualifies: it compiles the whole project in one
    monolithic, progress-less shot, so a ``completed/total`` bar would
    sit frozen at 0/N for minutes. Python streams per-file progress, and
    scip-typescript advances per scope — both get a real counter.
    """
    return kind == 'java'


def _index_detail_text(file_total: int, completed: int, *, pulse: bool) -> str:
    """Detail text for a per-language bar.

    Determinate bars (Python, TypeScript/JS) show a ``done/total files``
    counter. The pulse bar (Java) can't track progress, so it shows just
    the total — ``N files`` — without a meaningless ``0/``.
    """
    if pulse:
        return f'{int(file_total)} files'
    return f'{int(completed)}/{int(file_total)} files'


@functools.lru_cache(maxsize=1)
def _indexer_kind_extensions() -> dict[str, frozenset[str]]:
    """Map each indexer kind to the source-file extensions it covers,
    derived from the language registry so it can't drift.

    Cached: the registry is static, and ``_count_source_files`` is called
    once per scope, so without this the whole map is rebuilt N times per
    ``index`` run. Callers only read the returned dict."""
    from docgen.scip_languages import LANGUAGES

    out: dict[str, set[str]] = {}
    for lang in LANGUAGES:
        out.setdefault(lang.indexer_kind, set()).update(
            lang.source_extensions,
        )
    return {k: frozenset(v) for k, v in out.items()}


def _count_source_files(
    cwd: Path, kind: str, excluded_dir_names: frozenset[str],
) -> int:
    """Count indexable source files of ``kind`` under ``cwd``.

    Used only to size and order the per-language progress bars, so it
    needs to be a consistent estimate, not byte-exact to what the
    indexer ultimately reports. Skips excluded dirs, ``node_modules``,
    and minified JS bundles (the same things discovery ignores)."""
    import os

    from docgen.scip_discovery import _is_minified_js

    exts = _indexer_kind_extensions().get(kind, frozenset())
    if not exts:
        return 0
    skip = set(excluded_dir_names) | {'node_modules', '.git', '.ariadne'}
    count = 0
    for root, dirs, files in os.walk(cwd):
        dirs[:] = [d for d in dirs if d not in skip]
        for f in files:
            if Path(f).suffix.lower() in exts and not _is_minified_js(f):
                count += 1
    return count


def _plan_indexing(entries: list, count_fn) -> list[tuple]:
    """Group manifest indexer entries by language and order the groups
    by total file volume, smallest first.

    Returns ``[(kind, [entries...], total_files), ...]``. Entry order is
    preserved within each language. ``count_fn(entry) -> int`` is
    injected so this stays pure (no filesystem) for testing.
    """
    groups: dict[str, list] = {}
    totals: dict[str, int] = {}
    for entry in entries:
        kind = entry.get('kind')
        groups.setdefault(kind, []).append(entry)
        totals[kind] = totals.get(kind, 0) + count_fn(entry)
    # Smallest volume first; kind name breaks ties deterministically.
    ordered = sorted(groups, key=lambda k: (totals[k], k or ''))
    return [(k, groups[k], totals[k]) for k in ordered]


def cmd_index(
    args: argparse.Namespace,
    *,
    indexer_registry: dict | None = None,
    merger=None,
    phase_summary: list | None = None,
) -> int:
    """Run SCIP indexers per the manifest, merge intermediates into
    ``<source>/.ariadne/index.scip``, then run the 12-step persist
    chain that fills ``library_scip`` tables.

    Persist chain (dependency-correct order, end of every successful
    index):

    1. ``persist_all_sources`` → ``scip_symbols`` / ``scip_edges`` /
       ``scip_index_state`` (Tier 1 cross-source graph)
    2. ``persist_api_endpoints`` → ``api_endpoints`` from Swagger
       specs (only when source declares ``swagger_paths``)
    3. ``persist_string_literals`` → required by route extractors
       below to look up literal path values by SCIP position
    4. ``persist_config_values`` → ``config_values`` (HOCON / YAML /
       dotenv key→value defaults)
    5. ``persist_config_reads`` → ``config_reads`` (config-getter call
       sites resolved to their keys + values; needs steps 3 + 4)
    6. ``persist_akka_http_endpoints`` → ``api_endpoints`` (pattern,
       Scala Akka HTTP)
    7. ``persist_python_routes`` → ``api_endpoints`` (pattern,
       Flask / FastAPI)
    8. ``persist_express_routes`` → ``api_endpoints`` (pattern,
       Express / Koa)
    9. ``persist_python_http_clients`` → ``http_client_calls``
       (Tier 4 — ``httpx`` / ``requests`` / ``urllib``)
    10. ``persist_js_http_clients`` → ``http_client_calls``
        (``fetch`` / ``axios``)
    11. ``persist_scala_http_clients`` → ``http_client_calls``
        (Akka HTTP / sttp)
    12. ``persist_url_resolver`` → ``api_calls`` (joins client
        URLs to server endpoints, closing the cross-language tracing
        loop for ``ariadne_trace_flow``)

    Each persist_* helper is fail-soft on missing artifacts (a source
    without a current ``.scip`` returns 0 from its persist; the chain
    continues for the rest). The wiring contract is pinned in
    ``tests/test_cmd_index_wiring_chain.py``.

    Hard-fails on any indexer or merge error per decision #5. Updates
    each manifest entry with ``scip_path``, ``indexed_at``, and
    ``indexer_version`` so subsequent ``ariadne sync`` knows where to
    load from.

    ``indexer_registry`` and ``merger`` are dependency-injection
    points for tests; production uses the module-level defaults.
    """
    import json
    from datetime import datetime, timezone

    cfg = get_config()

    if args.all:
        sources_to_process = list(cfg.sources.keys())
        if not sources_to_process:
            console.print(
                '[red]No sources configured in ariadne.yaml[/red]',
            )
            return 1
    elif args.source:
        sources_to_process = [args.source]
    else:
        if cfg.default_source:
            sources_to_process = [cfg.default_source]
        else:
            console.print(
                '[red]No source specified — use --source X, --all, or '
                'set default_source in ariadne.yaml[/red]',
            )
            return 1

    if indexer_registry is None:
        indexer_registry = _default_indexer_registry()
    if merger is None:
        merger = _SubprocessMerger()

    for source_name in sources_to_process:
        sc = cfg.get_source_config(source_name)
        if sc is None:
            console.print(
                f'[red]Source not found in config: {source_name}[/red]',
            )
            return 1

        source_root = Path(sc.path).expanduser().resolve()

        # Freshness skip: if the merged .scip is still fresh (younger than
        # the source's max_staleness_days), reuse it rather than re-running
        # the slow per-language indexers. The artifact SHA only gates the
        # persist step, so the indexer skip is necessarily time-based. The
        # persist phase below walks every configured source, so it still
        # reloads this artifact — only the expensive indexer+merge is
        # skipped. ``--force`` bypasses. This is what makes a re-run of
        # ``onboard --approve`` (which re-enters the free phases) not
        # re-index when nothing has gone stale.
        if not getattr(args, 'force', False):
            merged_scip = source_root / '.ariadne' / 'index.scip'
            if merged_scip.exists():
                scip_cfg = cfg.get_source_scip_config(source_name)
                max_days = scip_cfg.max_staleness_days if scip_cfg else 7
                age_days = (
                    datetime.now(timezone.utc).timestamp()
                    - merged_scip.stat().st_mtime
                ) / 86400
                if age_days < max_days:
                    if not getattr(args, 'quiet', False):
                        console.print(
                            f'  [dim]Index — reusing fresh SCIP for '
                            f'{source_name} ({age_days:.1f}d < {max_days}d); '
                            f'pass --force to re-index[/dim]',
                        )
                    continue

        manifest_path = source_root / '.ariadne' / 'manifest.json'
        if not manifest_path.exists():
            console.print(
                f'[red]No manifest at {manifest_path} — run '
                f'`ariadne discover --source {source_name}` first[/red]',
            )
            return 1

        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        intermediates: list[Path] = []

        # Select the entries this invocation will touch (honor --kind and
        # skip kinds with no registered adapter).
        selected_entries = []
        for entry in manifest.get('indexers', []):
            kind = entry.get('kind')
            if args.kind and args.kind != kind:
                continue
            if indexer_registry.get(kind) is None:
                console.print(
                    f'[yellow]No adapter registered for kind={kind!r} '
                    f'— skipping[/yellow]',
                )
                continue
            selected_entries.append(entry)

        excluded_dir_names = frozenset(
            cfg.resolve_excluded_dirs(source_name),
        )

        def _entry_path(entry) -> Path:
            return (source_root / entry.get('cwd', '.')).resolve()

        # File counts size + order the per-language bars. Count once so
        # the volume sort and the bar totals agree, and we don't walk
        # each tree twice.
        entry_counts = {
            id(e): _count_source_files(
                _entry_path(e), e.get('kind'), excluded_dir_names,
            )
            for e in selected_entries
        }
        # Group by language, smallest-volume language first (so the quick
        # languages finish before the heavy JVM compile).
        plan = _plan_indexing(
            selected_entries, lambda e: entry_counts[id(e)],
        )

        if args.dry_run:
            for kind, kind_entries, total in plan:
                for entry in kind_entries:
                    output = (
                        source_root / '.ariadne' / 'intermediate'
                        / f'index-{_scope_label(entry, kind)}.scip'
                    )
                    console.print(
                        f'[bold]Would run {kind} adapter:[/bold] '
                        f'cwd={_entry_path(entry)} → {output} '
                        f'({entry_counts[id(entry)]} files)',
                    )
            continue

        # The source's effective exclusion set from ariadne.yaml — the
        # same set catalog-sync / discovery use. Directory names become
        # ``**/<name>`` globs so pyright matches them at any depth.
        user_excludes: list[str] = []
        for d in cfg.resolve_excluded_dirs(source_name):
            user_excludes.append(f'**/{d}')
        user_excludes.extend(sc.exclude)

        quiet = getattr(args, 'quiet', False)
        from rich.progress import (
            BarColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
        )

        progress_columns = (
            SpinnerColumn(),
            TextColumn('[bold cyan]{task.description}'),
            BarColumn(),
            TextColumn('[dim]{task.fields[detail]}'),
            TextColumn('·'),
            TimeElapsedColumn(),
        )

        # One Progress spanning all languages; one task (bar) per
        # language. Python advances smoothly via scip-python's per-file
        # stream; TypeScript/Java are opaque subprocesses, so their bar
        # bumps by a whole scope's file count on each scope's completion.
        with Progress(
            *progress_columns, console=console, transient=quiet,
        ) as progress:
            import time as _time

            for kind, kind_entries, kind_total in plan:
                label = _LANGUAGE_LABELS.get(kind, str(kind).title())
                lang_started = _time.monotonic()
                streams = _streams_file_progress(kind)
                pulse = _pulse_bar(kind)
                # Python streams per-file progress and TypeScript/JS bump
                # per scope — both get a determinate ``X/N files`` bar.
                # scip-java compiles in one progress-less shot, so it gets
                # an animated indeterminate (pulse) bar showing just ``N
                # files`` (a 0/N counter would sit frozen for minutes).
                task_id = progress.add_task(
                    f'  {label}',
                    total=None if pulse else (kind_total or None),
                    detail=_index_detail_text(kind_total, 0, pulse=pulse),
                )
                base = 0  # files completed by prior scopes of this lang

                for entry in kind_entries:
                    cwd = _entry_path(entry)
                    scope = _scope_label(entry, kind)
                    output = (
                        source_root / '.ariadne' / 'intermediate'
                        / f'index-{scope}.scip'
                    )
                    entry_count = entry_counts[id(entry)]
                    adapter = indexer_registry.get(kind)

                    run_kwargs: dict = {
                        'cwd': cwd,
                        'output': output,
                        'env_hints': dict(sc.env_hints),
                    }
                    if kind == 'python':
                        run_kwargs['entry_kind'] = entry.get(
                            'entry_kind', 'package',
                        )
                        run_kwargs['excludes'] = tuple(user_excludes)

                        def on_progress(
                            event, _base=base, _count=entry_count,
                        ) -> None:
                            if event.kind == 'tick' and event.total:
                                frac = min(
                                    1.0, event.current / event.total,
                                )
                                done = int(_base + frac * _count)
                                progress.update(
                                    task_id,
                                    completed=done,
                                    detail=_index_detail_text(
                                        kind_total, done, pulse=False,
                                    ),
                                )
                            elif event.kind == 'warning' and not quiet:
                                progress.console.print(
                                    f'[yellow]{event.text}[/yellow]',
                                )
                            # 'message'/'total' kinds: silenced — we own
                            # the totals via our own file count.

                        run_kwargs['progress_callback'] = on_progress

                    result = adapter.run(**run_kwargs)

                    if not result.success:
                        # Name the failing scope — a language can have
                        # many, and quiet mode hides per-scope detail.
                        progress.console.print(
                            f'[red]{kind} adapter failed (scope={scope}, '
                            f'cwd={cwd}): {result.error_message}[/red]',
                        )
                        return 1

                    if (
                        kind == 'python'
                        and result.resolved_interpreter is not None
                        and result.resolution_source
                        and not quiet
                    ):
                        progress.console.print(
                            f'  Python interpreter: '
                            f'{result.resolved_interpreter} '
                            f'(source: {result.resolution_source})',
                        )

                    base += entry_count
                    if not pulse:
                        # Determinate bars (Python, TypeScript/JS) advance
                        # to the running file total after each scope.
                        progress.update(
                            task_id,
                            completed=base,
                            detail=_index_detail_text(
                                kind_total, base, pulse=False,
                            ),
                        )

                    entry['scip_path'] = str(
                        output.relative_to(source_root / '.ariadne'),
                    )
                    entry['indexed_at'] = datetime.now(
                        timezone.utc,
                    ).isoformat()
                    entry['indexer_version'] = result.indexer_version
                    # Record the Vue companion→.vue mapping so the loader
                    # (and resolve_index for catalog extraction) can
                    # translate SCIP positions back to the .vue files.
                    if result.vue_mapping_path:
                        entry['vue_mapping'] = str(
                            Path(result.vue_mapping_path).relative_to(
                                source_root / '.ariadne',
                            ),
                        )
                    intermediates.append(output)

                # Finish as a full bar so a completed language reads as
                # done. For the pulse (Java) bar we set a real total now
                # to fill it, but keep the "N files" detail (no X/N).
                if kind_total:
                    progress.update(
                        task_id,
                        total=kind_total,
                        completed=kind_total,
                        detail=_index_detail_text(
                            kind_total, kind_total, pulse=pulse,
                        ),
                    )

                # Record a per-language summary so phase callers
                # (dry-run / onboard) can render it nested under their
                # "✓ Index" line.
                if phase_summary is not None:
                    phase_summary.append({
                        'language': label,
                        'files': kind_total,
                        'seconds': _time.monotonic() - lang_started,
                    })

        if intermediates:
            final = source_root / '.ariadne' / 'index.scip'
            if len(intermediates) == 1:
                # Single-language project (or a multi-language project
                # where only one indexer ran this invocation): no merge
                # needed. Copy the lone intermediate directly so the
                # external ``scip`` CLI isn't a dependency for the
                # common single-source case.
                import shutil
                final.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(intermediates[0], final)
            else:
                ok = merger.merge(intermediates, final)
                if not ok:
                    # _SubprocessMerger already printed the diagnostic.
                    return 1
            manifest['merged_at'] = datetime.now(
                timezone.utc,
            ).isoformat()
            manifest['merged_scip'] = 'index.scip'
            manifest_path.write_text(
                json.dumps(manifest, indent=2),
                encoding='utf-8',
            )
            if not getattr(args, 'quiet', False):
                console.print(f'[green]Wrote {final}[/green]')

    # End-of-index step: persist the cross-source graph into
    # ``library_scip`` so downstream readers (ariadne callers /
    # impact_radius / dead-code, plus the catalog generator's
    # architecture-prompt Dependents section) see the fresh data.
    # Walks every configured source — cross-source edges only resolve
    # when both endpoints sit in the same materialized graph.
    if not args.dry_run:
        from docgen.scip_persist import (
            persist_akka_http_endpoints,
            persist_all_sources,
            persist_api_endpoints,
            persist_config_reads,
            persist_config_values,
            persist_express_routes,
            persist_js_http_clients,
            persist_python_http_clients,
            persist_python_routes,
            persist_scala_http_clients,
            persist_string_literals,
            persist_url_resolver,
        )

        source_pairs: list[tuple[str, Path]] = []
        swagger_pairs: list[tuple[str, Path, list[str]]] = []
        for name in cfg.sources:
            other_sc = cfg.get_source_config(name)
            if other_sc is None:
                continue
            other_root = Path(other_sc.path).expanduser().resolve()
            source_pairs.append((name, other_root))
            if other_sc.swagger_paths:
                swagger_pairs.append((
                    name, other_root, list(other_sc.swagger_paths),
                ))

        persisted = persist_all_sources(Path(cfg.db_path), source_pairs)
        if persisted and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Persisted cross-source graph '
                f'({persisted} source(s)) → library_scip[/green]',
            )

        # Wave 4 Tier 2 — Swagger / OpenAPI ingestion. Populates
        # ``api_endpoints`` so ``ariadne_trace_flow``'s HTTP-tier
        # join (api_calls → api_endpoints) actually returns rows
        # once the client side is wired in a follow-up commit.
        if swagger_pairs:
            endpoints = persist_api_endpoints(
                Path(cfg.db_path), swagger_pairs,
            )
            if endpoints and not getattr(args, 'quiet', False):
                console.print(
                    f'[green]Ingested {endpoints} API endpoint(s) '
                    f'from Swagger specs → library_scip[/green]',
                )

        # Layer C prerequisite — populate ``string_literals``. The
        # route extractors below look up literal path values by SCIP
        # position; without this, every literal-arg endpoint silently
        # vanishes from the route walks.
        literals = persist_string_literals(Path(cfg.db_path), source_pairs)
        if literals and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Indexed {literals} string literal(s) → '
                f'library_scip[/green]',
            )

        # Phase 2q — populate ``config_values`` (HOCON / YAML / dotenv
        # key->value) so Layer C can resolve config-getter arguments to
        # their configured values. Previously never wired, so the table
        # stayed empty.
        config_vals = persist_config_values(Path(cfg.db_path), source_pairs)
        if config_vals and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Indexed {config_vals} config value(s) → '
                f'library_scip[/green]',
            )
        # Config↔code bridge (Tier 2) — enumerate config-getter call
        # sites into ``config_reads``. Needs ``string_literals`` and
        # ``config_values`` (both persisted just above).
        config_reads = persist_config_reads(Path(cfg.db_path), source_pairs)
        if config_reads and not getattr(args, 'quiet', False):
            console.print(
                f'[green]Indexed {config_reads} config read(s) → '
                f'library_scip[/green]',
            )

        # Wave 4 Tier 2 step 2 — Akka HTTP route extraction. Walks
        # SCIP-classified call sites for ``path`` / ``pathPrefix`` /
        # HTTP-verb combinators and persists detected routes to
        # ``api_endpoints`` with ``resolution_source='pattern'``
        # (preserves Swagger rows).
        quiet = getattr(args, 'quiet', False)
        akka_routes = persist_akka_http_endpoints(
            Path(cfg.db_path), source_pairs,
        )
        if akka_routes and not quiet:
            console.print(
                f'[green]Extracted {akka_routes} Akka HTTP route(s) '
                f'→ api_endpoints[/green]',
            )

        # Wave 4 Tier 2 step 3 — Flask / FastAPI route extraction.
        # Walks SCIP-classified ``@app.route`` / ``@app.<verb>`` /
        # ``@router.<verb>`` decorators and persists matched routes
        # to ``api_endpoints``. Same coexistence semantics as Akka.
        python_routes = persist_python_routes(
            Path(cfg.db_path), source_pairs,
        )
        if python_routes and not quiet:
            console.print(
                f'[green]Extracted {python_routes} Flask/FastAPI '
                f'route(s) → api_endpoints[/green]',
            )

        # Wave 4 Tier 2 step 4 — Express / Koa route extraction.
        # Walks ``app.<verb>(path, handler)`` and
        # ``router.<verb>(...)`` call sites and persists matched
        # routes to ``api_endpoints``.
        express_routes = persist_express_routes(
            Path(cfg.db_path), source_pairs,
        )
        if express_routes and not quiet:
            console.print(
                f'[green]Extracted {express_routes} Express/Koa '
                f'route(s) → api_endpoints[/green]',
            )

        # Wave 4 Tier 4 step 1 — Python HTTP client extraction.
        # Walks ``httpx.{verb}`` / ``requests.{verb}`` /
        # ``urllib.urlopen`` call sites and persists raw URL strings
        # to ``http_client_calls``. URL→endpoint joining (Phase 8c)
        # waits until all three client extractors are wired.
        py_http = persist_python_http_clients(
            Path(cfg.db_path), source_pairs,
        )
        if py_http and not quiet:
            console.print(
                f'[green]Extracted {py_http} Python HTTP call(s) '
                f'→ http_client_calls[/green]',
            )

        # Wave 4 Tier 4 step 2 — JS / TS HTTP client extraction.
        # Walks ``fetch(...)`` / ``axios.{verb}`` /
        # ``this.$http.{verb}`` (Vue 2 / Angular) call sites.
        js_http = persist_js_http_clients(
            Path(cfg.db_path), source_pairs,
        )
        if js_http and not quiet:
            console.print(
                f'[green]Extracted {js_http} JS/TS HTTP call(s) '
                f'→ http_client_calls[/green]',
            )

        # Wave 4 Tier 4 step 3 — Scala HTTP client extraction.
        # Walks Akka-HTTP ``Http().singleRequest`` / sttp
        # ``basicRequest.<verb>`` call sites.
        scala_http = persist_scala_http_clients(
            Path(cfg.db_path), source_pairs,
        )
        if scala_http and not quiet:
            console.print(
                f'[green]Extracted {scala_http} Scala HTTP call(s) '
                f'→ http_client_calls[/green]',
            )

        # Wave 4 Phase 8c — URL→endpoint resolution. The closing
        # step: matches client URLs (http_client_calls) against
        # server templates (api_endpoints), writes resolved edges
        # to ``api_calls``. Once this lands, ariadne_trace_flow's
        # HTTP-tier hops finally return cross-language chains.
        resolved = persist_url_resolver(
            Path(cfg.db_path), source_pairs,
        )
        if resolved and not quiet:
            console.print(
                f'[green]Resolved {resolved} URL→endpoint edge(s) '
                f'→ api_calls[/green]',
            )

    return 0


def _ensure_gitignore_entry(source_path: Path, line: str) -> None:
    """Append ``line`` to ``<source_path>/.gitignore`` if not already
    present. Creates the file if it doesn't exist. Idempotent."""
    gitignore = source_path / '.gitignore'
    if gitignore.exists():
        existing = gitignore.read_text(encoding='utf-8')
        if line in existing.splitlines():
            return
        with gitignore.open('a', encoding='utf-8') as f:
            if existing and not existing.endswith('\n'):
                f.write('\n')
            f.write(line + '\n')
    else:
        gitignore.write_text(line + '\n', encoding='utf-8')


def cmd_discover(args: argparse.Namespace) -> int:
    """Walk a source tree, detect SCIP indexer locations, write
    ``<source>/.ariadne/manifest.json``, AND auto-author the SCIP
    fields in ``ariadne.yaml``.

    UX principle: users author only ``path``, ``depends_on``,
    ``exclude``, ``exclude_dirs`` — Ariadne fills the rest.
    ``cmd_discover`` writes ``index_kinds`` (one entry per detected
    SCIP-routable language: ``javascript`` for any TS/JS file,
    ``scala``/``java`` for JVM sources) and the ``scip:`` block
    pointing at ``<source>/.ariadne/index.scip`` and a default
    ``max_staleness_days``. ``cmd_sync`` later detects new languages
    in changed files and triggers ``discover --config-only`` to keep
    these fields current automatically.

    Re-runs are idempotent — ``write_source_scip_config`` returns
    False (no rewrite) when the YAML already matches the detection
    state. Manual edits to the auto-managed block are regenerated
    on the next ``discover``.

    Adds ``.ariadne/`` to ``<source>/.gitignore`` so the artifact
    directory doesn't accidentally get committed.

    Per design decision #1 (artifacts gitignored, per-machine) and #6
    (auto-discovery via marker files: ``__init__.py``, ``package.json``,
    ``build.sbt``/``pom.xml``/``build.gradle*``).
    """
    import json

    from config import DEFAULT_EXCLUDE_POLICY
    from docgen.scip_discovery import discover

    cfg = get_config()

    if args.all:
        sources_to_process = list(cfg.sources.keys())
        if not sources_to_process:
            console.print(
                '[red]No sources configured in ariadne.yaml[/red]',
            )
            return 1
    elif args.source:
        sources_to_process = [args.source]
    else:
        if cfg.default_source:
            sources_to_process = [cfg.default_source]
        else:
            console.print(
                '[red]No source specified — use --source X, --all, or '
                'set default_source in ariadne.yaml[/red]',
            )
            return 1

    global_excl = set(
        cfg._config.get('exclude_policy') or DEFAULT_EXCLUDE_POLICY,
    )

    for source_name in sources_to_process:
        sc = cfg.get_source_config(source_name)
        if sc is None:
            console.print(
                f'[red]Source not found in config: {source_name}[/red]',
            )
            return 1

        source_path = Path(sc.path).expanduser().resolve()
        if not source_path.exists():
            console.print(
                f'[red]Source path does not exist: {source_path}[/red]',
            )
            return 1

        # Effective exclude set: global policy + source-level extras,
        # minus exempt-dirs.
        effective_excl = (
            global_excl | set(sc.exclude_dirs)
        ) - set(sc.exempt_dirs)

        # Per-source file-glob exclusions (e.g., '**/*.min.js').
        exclude_patterns = frozenset(sc.exclude or ())

        entries = discover(
            source_path,
            exclude_dirs=frozenset(effective_excl),
            exempt_dirs=frozenset(sc.exempt_dirs),
            exclude_patterns=exclude_patterns,
        )

        # --review: classify suspects, prompt user, persist decisions to
        # ariadne.yaml, and filter rejected entries from the manifest.
        if getattr(args, 'review', False) and entries:
            from docgen.scip_review import (
                classify_suspects,
                prompt_keep_entry,
            )
            from docgen.yaml_writer import append_source_excludes

            suspects = classify_suspects(
                entries, source_root=source_path,
            )
            if suspects:
                console.print(
                    f'\n[bold]Reviewing {len(suspects)} suspect '
                    f'entry/entries for {source_name}[/bold]',
                )
                rejected_entry_ids: set[int] = set()
                rejected_patterns: list[str] = []
                for suspect in suspects:
                    keep = prompt_keep_entry(suspect)
                    if not keep:
                        rejected_entry_ids.add(id(suspect.entry))
                        rejected_patterns.append(
                            suspect.suggested_pattern,
                        )

                if rejected_patterns:
                    entries = [
                        e for e in entries
                        if id(e) not in rejected_entry_ids
                    ]
                    config_path = cfg.config_path
                    if config_path is not None:
                        try:
                            added = append_source_excludes(
                                Path(config_path),
                                source_name,
                                rejected_patterns,
                            )
                            console.print(
                                f'[green]Added {added} exclude '
                                f'pattern(s) to {config_path}[/green]',
                            )
                        except (FileNotFoundError, KeyError) as e:
                            console.print(
                                f'[yellow]Could not update yaml: '
                                f'{e}[/yellow]',
                            )
                    else:
                        console.print(
                            '[yellow]No ariadne.yaml path resolved; '
                            'patterns not persisted. Add manually:'
                            '[/yellow]',
                        )
                        for p in rejected_patterns:
                            console.print(f'  - {p!r}')

        def _rel(p: Path) -> str:
            try:
                rel = p.relative_to(source_path)
                return '.' if str(rel) == '.' else str(rel)
            except ValueError:
                return str(p)

        manifest = {
            'ariadne_version': '1',
            'source_name': source_name,
            'indexers': [
                {
                    'kind': e.kind,
                    'cwd': _rel(e.cwd),
                    'markers': [_rel(m) for m in e.markers],
                    # Phase 2j.b: 'package' (has __init__.py / package.json
                    # / build file) vs 'scripts' (orphan dir). The Python
                    # adapter (Phase 2n) dispatches the transient
                    # pyrightconfig include pattern on this field.
                    'entry_kind': e.entry_kind,
                }
                for e in entries
            ],
        }

        if args.dry_run:
            console.print(
                f'[bold]Manifest for {source_name} (dry-run):[/bold]',
            )
            console.print(json.dumps(manifest, indent=2))
            continue

        manifest_dir = source_path / '.ariadne'
        manifest_dir.mkdir(exist_ok=True)
        # Pre-create the staging dir indexers write into. Each adapter's
        # underlying tool (e.g., scip-python) opens the output file
        # directly without mkdir-p, so without this the first
        # ``ariadne index`` run hits ENOENT. Creating it during discovery
        # — alongside ``.ariadne/`` itself — keeps the responsibility for
        # layout setup in one place.
        (manifest_dir / 'intermediate').mkdir(exist_ok=True)
        manifest_path = manifest_dir / 'manifest.json'
        manifest_path.write_text(
            json.dumps(manifest, indent=2),
            encoding='utf-8',
        )

        _ensure_gitignore_entry(source_path, '.ariadne/')

        if entries:
            table = Table(
                title=f'Discovered indexers for {source_name}',
            )
            table.add_column('Kind')
            table.add_column('cwd')
            table.add_column('Markers')
            for e in entries:
                table.add_row(
                    e.kind,
                    _rel(e.cwd),
                    ', '.join(_rel(m) for m in e.markers),
                )
            console.print(table)
        else:
            console.print(
                f'[yellow]No indexer-relevant clusters found in '
                f'{source_path}[/yellow]',
            )
        console.print(f'[green]Wrote {manifest_path}[/green]')

        # Auto-author the SCIP-related fields in ariadne.yaml so the
        # user only ever has to author path/depends_on/exclude/exclude_dirs.
        # Mapping: discovery kind → language(s) whose catalog extraction
        # routes through SCIP today (extract_elements supports
        # scala/java/javascript SCIP routing). Python is detected and
        # cross-source-indexed but its catalog stays on ast-grep — the
        # qualified_names already align with scip-python's, so no
        # routing layer is needed.
        catalog_scip_languages: set[str] = set()
        for e in entries:
            if e.kind == 'typescript':
                catalog_scip_languages.add('javascript')
            elif e.kind == 'java':
                # scip-java covers both Scala and Java in one indexer
                # output. Declare both index_kinds so catalog extraction
                # routes either file extension through scip_extractor.
                catalog_scip_languages.add('scala')
                catalog_scip_languages.add('java')

        config_path = cfg.config_path
        if catalog_scip_languages and config_path is not None:
            from docgen.yaml_writer import write_source_scip_config
            artifact_path = manifest_dir / 'index.scip'
            try:
                rewrote = write_source_scip_config(
                    Path(config_path),
                    source_name,
                    catalog_scip_languages=catalog_scip_languages,
                    artifact_path=artifact_path,
                )
                if rewrote:
                    console.print(
                        f'[green]Updated {config_path} '
                        f'(index_kinds: {sorted(catalog_scip_languages)})'
                        f'[/green]',
                    )
            except (FileNotFoundError, KeyError) as exc:
                console.print(
                    f'[yellow]Could not update yaml: {exc}[/yellow]',
                )

    return 0


# ---------------------------------------------------------------------------
# Handler dispatch table
# ---------------------------------------------------------------------------

HANDLERS = {
    'search': lambda args: asyncio.run(cmd_search(args)),
    'list': lambda args: cmd_list(args),
    'get': lambda args: cmd_get(args),
    'add': lambda args: asyncio.run(cmd_add(args)),
    'finding': lambda args: asyncio.run(cmd_finding(args)),
    'delete': lambda args: cmd_delete(args),
    'export': lambda args: cmd_export(args),
    'import': lambda args: cmd_import_(args),
    'export-db': lambda args: cmd_export_db(args),
    'import-db': lambda args: cmd_import_db(args),
    'rebuild': lambda args: asyncio.run(cmd_rebuild(args)),
    'build-matrix': lambda args: cmd_build_matrix(args),
    'stats': lambda args: cmd_stats(args),
    'status': lambda args: cmd_status(args),
    'usage': lambda args: cmd_usage(args),
    'gaps': lambda args: cmd_gaps(args),
    'vacuum': lambda args: cmd_vacuum(args),
    'tag': lambda args: cmd_tag(args),
    'discover': lambda args: cmd_discover(args),
    'index': lambda args: cmd_index(args),
}
