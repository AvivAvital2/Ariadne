"""Core CRUD, stats, and metadata CLI commands."""
from __future__ import annotations

import argparse
import asyncio
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

    # rebuild
    subparsers.add_parser('rebuild', help='Rebuild all embeddings')

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
    'rebuild': lambda args: asyncio.run(cmd_rebuild(args)),
    'build-matrix': lambda args: cmd_build_matrix(args),
    'tag': lambda args: cmd_tag(args),
    
}
