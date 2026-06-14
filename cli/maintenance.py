"""Documentation maintenance CLI commands (``check``, ``merge``, ``cleanup``,
``migrate``).

Extracted from cli/sync.py. Status checks, post-merge regeneration, cleanup of
expired branch docs, and metadata migrations. Wired into the parser via this
module's ``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from config import get_config

if TYPE_CHECKING:
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register documentation maintenance commands."""

    # check
    check_parser = subparsers.add_parser('check', help='Check documentation status')
    check_parser.add_argument('--source', '-s', default=None,
                               help='Source path or name (default from config)')
    check_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show list of stale/undocumented files')

    # merge
    merge_parser = subparsers.add_parser('merge', help='Regenerate docs after merging branches to main')
    merge_parser.add_argument('--source', '-s', default=None,
                               help='Source name (default from config)')
    merge_parser.add_argument('--since', default=None,
                               help='Git hash to compare from (default: last sync point)')
    merge_parser.add_argument('--dry-run', action='store_true',
                               help='Show what would be done')
    merge_parser.add_argument('--no-export', action='store_true',
                               help='Skip markdown export')
    merge_parser.add_argument('--skip-generate', action='store_true',
                               help='Only deprecate, skip regeneration')
    merge_parser.add_argument('--delete-consumed', action='store_true',
                               help='Delete branch docs instead of deprecating')

    # cleanup
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up expired or orphaned documents')
    cleanup_parser.add_argument('--expired', action='store_true',
                                 help='Remove branch-specific documents that have expired')
    cleanup_parser.add_argument('--dry-run', action='store_true',
                                 help='Show what would be removed without making changes')

    # migrate
    migrate_parser = subparsers.add_parser('migrate', help='Migrate documents to update metadata')
    migrate_parser.add_argument('--check', action='store_true',
                                 help='Show migration status without making changes')
    migrate_parser.add_argument('--source-files', action='store_true',
                                 help='Attempt to backfill missing source_files from document content')
    migrate_parser.add_argument('--fix-paths', action='store_true',
                                 help='Normalize paths in the staleness DB (fix ../prefix issues)')
    migrate_parser.add_argument('--fix-catalog-language', action='store_true',
                                 help='Re-detect and update language metadata on catalog file_index documents')
    migrate_parser.add_argument('--doc-ids', action='store_true',
                                 help=(
                                     'Backfill deterministic doc IDs '
                                     '(UUID5 from source+type+file) over '
                                     'legacy UUID4 docs. Collapses any '
                                     'duplicates that map to the same '
                                     'deterministic id (newest wins).'
                                 ))
    migrate_parser.add_argument('--infer-source-name', action='store_true',
                                 help=(
                                     'Backfill documents.source_name where NULL '
                                     'by matching source_files[0] against the '
                                     'configured source paths in ariadne.yaml. '
                                     'Run before --doc-ids to recover legacy '
                                     'docs that predate source_name plumbing.'
                                 ))
    migrate_parser.add_argument('--dry-run', action='store_true',
                                 help='Show planned changes without writing')
    migrate_parser.add_argument('--verbose', '-v', action='store_true',
                                 help='Show detailed output')


async def cmd_check(args: argparse.Namespace) -> int:
    """Check for stale or missing documentation."""
    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    cfg = get_config()

    # Resolve source via config
    source_path = cfg.resolve_source(args.source)
    if source_path is None:
        console.print('[red]No source specified and no default_source in config.[/red]')
        console.print('Use --source <path> or set default_source in ariadne.yaml')
        return 1

    if not source_path.exists():
        console.print(f'[red]Source path not found: {source_path}[/red]')
        return 1

    # SCIP source_config so check reports Scala/Java files for sources
    # that declare SCIP — without it, check_staleness uses find_python_files
    # and silently reports 0 files for Scala-only sources.
    source_name = args.source or cfg.default_source
    scip_config = (
        cfg.get_source_scip_config(source_name) if source_name else None
    )

    config = OrchestratorConfig(
        source_path=source_path,
        db_path=args.db or Path(cfg.db_path),
        staleness_db_path=Path(cfg.staleness_db_path),
        source_config=scip_config,
        catalog_only_generator=scip_config is not None,
    ignore_staleness=cfg.source_ignore_staleness(source_name))

    async with DocGenOrchestrator(config) as orchestrator:
        status = await orchestrator.check_staleness()

    # Print summary
    table = Table(title='Documentation Status')
    table.add_column('Metric', style='bold')
    table.add_column('Value', style='cyan')

    table.add_row(
        'Total catalog files' if scip_config else 'Total Python files',
        str(status['total_files']),
    )
    table.add_row('Up-to-date', f"[green]{status['up_to_date']}[/green]")
    table.add_row('Stale (need update)', f"[yellow]{status['stale_files']}[/yellow]")
    table.add_row('Undocumented', f"[red]{status['undocumented_files']}[/red]")

    console.print(table)

    if args.verbose and status['stale_paths']:
        console.print()
        console.print('[yellow]Stale files:[/yellow]')
        for path in status['stale_paths'][:20]:
            console.print(f'  - {path}')
        if len(status['stale_paths']) > 20:
            console.print(f'  ... and {len(status["stale_paths"]) - 20} more')

    if args.verbose and status['undocumented_paths']:
        console.print()
        console.print('[red]Undocumented files:[/red]')
        for path in status['undocumented_paths'][:20]:
            console.print(f'  - {path}')
        if len(status['undocumented_paths']) > 20:
            console.print(f'  ... and {len(status["undocumented_paths"]) - 20} more')

    return 0


async def cmd_merge(args: argparse.Namespace) -> int:
    """Regenerate stable docs after merging branches to main.

    This command:
    1. Detects experimental docs from merged branches (via GitHub API)
    2. Regenerates stable docs for the affected source files
    3. Deprecates or deletes the consumed experimental docs
    4. Exports and updates sync state
    """
    from docgen.merge import execute_merge, preview_merge

    cfg = get_config()

    # Resolve source
    source_name = args.source or cfg.default_source
    if source_name is None:
        console.print('[red]No source specified and no default_source in config.[/red]')
        return 1

    source_path = cfg.resolve_source(source_name)
    if source_path is None or not source_path.exists():
        console.print(f'[red]Source path not found: {source_path}[/red]')
        return 1

    # Warn if not on main
    from git_ops import get_current_branch
    current_branch = get_current_branch(source_path)
    if current_branch and current_branch != cfg.main_branch:
        console.print(f'[yellow]Warning: Not on {cfg.main_branch} (currently on {current_branch})[/yellow]')

    if args.dry_run:
        preview = preview_merge(source_name, args.delete_consumed)
        console.print(preview)
        return 0

    library = get_library(args.db)

    try:
        # Execute merge
        console.print()
        console.print('Executing merge...')

        merge_result = await execute_merge(
            library=library,
            cfg=cfg,
            source_name=source_name,
            source_path=source_path,
            db_path=args.db,
            since=args.since,
            skip_generate=args.skip_generate,
            no_export=args.no_export,
            delete_consumed=args.delete_consumed,
        )

        # Print results
        console.print()
        console.print('[green]Merge complete.[/green]')
        console.print(f'  Merged branches: {", ".join(merge_result.merged_branches)}')
        console.print(f'  Files regenerated: {merge_result.files_regenerated}')
        console.print(f'  Docs created: {merge_result.docs_created}')
        if merge_result.docs_failed:
            console.print(f'  [yellow]Docs failed: {merge_result.docs_failed}[/yellow]')
        if merge_result.docs_deprecated:
            console.print(f'  Docs deprecated: {merge_result.docs_deprecated}')
        if merge_result.docs_deleted:
            console.print(f'  Docs deleted: {merge_result.docs_deleted}')

        return 0

    finally:
        library.close()


def cmd_cleanup(args: argparse.Namespace) -> int:
    """Clean up expired or orphaned documents.

    With --expired: Remove branch-specific documents that have passed their expiration date.
    """
    from datetime import datetime

    library = get_library(args.db)

    try:
        docs = library.list_documents()
        now = datetime.now()
        removed_count = 0
        removed_titles: list[str] = []

        if args.expired:
            for doc in docs:
                expires_at_str = doc.metadata.get('expires_at')
                if expires_at_str:
                    try:
                        # Parse ISO format datetime
                        expires_at = datetime.fromisoformat(expires_at_str)
                        if expires_at < now:
                            if args.dry_run:
                                console.print(f'[dim]Would remove: {doc.title} (expired {expires_at_str[:10]})[/dim]')
                            else:
                                library.delete_document(doc.id)
                                removed_titles.append(doc.title)
                            removed_count += 1
                    except ValueError:
                        # Invalid date format, skip
                        pass

        if args.dry_run:
            console.print(f'[yellow]Dry run: would remove {removed_count} expired document(s)[/yellow]')
        elif removed_count > 0:
            console.print(f'[green]Removed {removed_count} expired branch document(s)[/green]')
            for title in removed_titles[:10]:
                console.print(f'  - {title}')
            if len(removed_titles) > 10:
                console.print(f'  ... and {len(removed_titles) - 10} more')
        else:
            console.print('[green]No expired documents found.[/green]')

        return 0

    finally:
        library.close()


def cmd_migrate(args: argparse.Namespace) -> int:
    """Migrate documents to update metadata or fill in missing fields.

    --check: Show migration status without making changes.
    --source-files: Attempt to backfill missing source_files from document content.
    """
    library = get_library(args.db)

    try:
        docs = library.list_documents()

        if args.check:
            # Count documents with and without source_files
            has_source_files = sum(1 for d in docs if d.source_files)
            missing_source_files = len(docs) - has_source_files

            console.print(f'Total documents: {len(docs)}')
            console.print(f'  With source_files: [green]{has_source_files}[/green]')
            console.print(f'  Missing source_files: [yellow]{missing_source_files}[/yellow]')

            if missing_source_files > 0:
                console.print()
                console.print('Run [bold]ariadne migrate --source-files[/bold] to attempt backfill.')
                console.print('Or run [bold]ariadne generate --force[/bold] to regenerate all docs.')

            return 0

        if args.source_files:
            # Attempt to infer source_files from document title/content
            updated = 0
            cfg = get_config()
            source_path = cfg.resolve_source(cfg.default_source)

            for doc in docs:
                if doc.source_files:
                    continue  # Already has source files

                # Try to infer from title
                # Common patterns: "Feature System" -> feature, "DuckDB Manager" -> duckdb
                title_parts = doc.title.lower().replace('-', ' ').split()

                # Search for matching Python files
                if source_path and source_path.exists():
                    matches: list[str] = []
                    for part in title_parts:
                        if len(part) > 3:  # Skip short words
                            # Search for files containing this word
                            for py_file in source_path.rglob('*.py'):
                                if part in py_file.stem.lower():
                                    rel_path = str(py_file.relative_to(source_path))
                                    if rel_path not in matches:
                                        matches.append(rel_path)

                    if matches and len(matches) <= 5:  # Only use if reasonably confident
                        library.update_document(doc.id, source_files=matches[:3])
                        updated += 1
                        if args.verbose:
                            console.print(f'Updated: {doc.title} -> {matches[:3]}')

            console.print(f'[green]Updated source_files for {updated} document(s)[/green]')
            return 0

        if args.fix_paths:
            from docgen.staleness import StalenessTracker

            cfg = get_config()
            source_path = cfg.resolve_source(cfg.default_source)
            if source_path is None:
                console.print('[red]No default_source configured \u2014 cannot resolve paths.[/red]')
                return 1

            staleness_db = Path(cfg.staleness_db_path)
            if not staleness_db.exists():
                console.print(f'[yellow]Staleness DB not found at {staleness_db}[/yellow]')
                return 1

            with StalenessTracker(staleness_db) as tracker:
                changes = tracker.normalize_paths(source_path)

            if changes:
                console.print(f'[green]Fixed {len(changes)} path(s) in staleness DB:[/green]')
                for old, new in changes:
                    console.print(f'  {old} -> {new}')
            else:
                console.print('[green]All paths already normalized.[/green]')
            return 0

        if args.fix_catalog_language:
            from docgen.catalog_extractor import _detect_language as _detect_lang_ext

            updated = 0
            unchanged = 0
            for doc in docs:
                if doc.metadata.get('kind') != 'file_index':
                    continue
                if not doc.source_files:
                    continue
                file_path = Path(doc.source_files[0])
                detected = _detect_lang_ext(file_path) or 'unknown'
                current = doc.metadata.get('language')
                if current == detected:
                    unchanged += 1
                    continue
                new_metadata = dict(doc.metadata)
                new_metadata['language'] = detected
                library.update_document(doc.id, metadata=new_metadata)
                updated += 1
                if args.verbose:
                    console.print(
                        f'[dim]{doc.source_files[0]}: {current!r} -> {detected!r}[/dim]'
                    )

            console.print(
                f'[green]Updated language on {updated} file_index doc(s);[/green] '
                f'{unchanged} already correct.'
            )
            return 0

        if args.infer_source_name:
            cfg = get_config()
            source_map: dict[str, Path] = {}
            for name, _ in (cfg.sources or {}).items():
                resolved = cfg.resolve_source(name)
                if resolved is not None:
                    source_map[name] = resolved
            if not source_map:
                console.print(
                    '[red]No sources configured in ariadne.yaml.[/red]'
                )
                return 1

            # Match each NULL-source doc against the longest matching
            # configured source path. Longest wins so subdirectory sources
            # (e.g. ao-research nested under pythonproject) bind correctly.
            sorted_sources = sorted(
                source_map.items(), key=lambda kv: -len(str(kv[1])),
            )

            updated = 0
            ambiguous = 0
            no_match = 0
            unmatched_samples: list[tuple[str, str]] = []
            with library._conn_provider.acquire() as conn:
                rows = conn.execute(
                    'SELECT id, source_files FROM documents '
                    'WHERE source_name IS NULL OR source_name = ""'
                ).fetchall()

                for doc_id, sf_json in rows:
                    sf = []
                    if sf_json:
                        try:
                            import json as _json
                            sf = _json.loads(sf_json)
                        except (ValueError, TypeError):
                            sf = []
                    if not sf:
                        no_match += 1
                        continue
                    primary = sf[0]
                    matched: list[str] = []
                    for name, path in sorted_sources:
                        if primary.startswith(str(path)):
                            matched.append(name)
                            break  # longest match wins; stop after first
                    if len(matched) == 1:
                        conn.execute(
                            'UPDATE documents SET source_name = ? WHERE id = ?',
                            (matched[0], doc_id),
                        )
                        updated += 1
                    elif len(matched) > 1:
                        ambiguous += 1
                    else:
                        no_match += 1
                        if len(unmatched_samples) < 10:
                            unmatched_samples.append((doc_id, primary))

            console.print(f'[green]Backfilled source_name on {updated} doc(s).[/green]')
            if ambiguous:
                console.print(f'[yellow]Ambiguous (multiple sources matched): {ambiguous}[/yellow]')
            if no_match:
                console.print(f'[yellow]No match: {no_match}[/yellow]')
                if args.verbose and unmatched_samples:
                    console.print('\n[dim]Unmatched samples (id, source_files[0]):[/dim]')
                    for sid, sf in unmatched_samples:
                        console.print(f'  {sid[:8]}... {sf!r}')
            return 0

        if args.doc_ids:
            cfg = get_config()
            source_map: dict[str, Path] = {}
            for name, sc in (cfg.sources or {}).items():
                resolved = cfg.resolve_source(name)
                if resolved is not None:
                    source_map[name] = resolved
            if not source_map:
                console.print(
                    '[red]No sources configured in ariadne.yaml — cannot '
                    'compute deterministic ids without source paths.[/red]'
                )
                return 1

            staleness_db = Path(cfg.staleness_db_path)
            result = library.migrate_doc_ids(
                source_name_to_path=source_map,
                dry_run=args.dry_run,
                staleness_db_path=staleness_db if staleness_db.exists() else None,
            )

            verb = 'Would migrate' if args.dry_run else 'Migrated'
            console.print(f'[bold]{verb} doc IDs[/bold]')
            console.print(f'  Inspected:            {result.inspected}')
            console.print(f'  Already deterministic: [green]{result.already_deterministic}[/green]')
            console.print(f'  Remapped:             [yellow]{result.remapped}[/yellow]')
            console.print(f'  Duplicates collapsed: [yellow]{result.duplicates_collapsed}[/yellow]')
            if result.skipped_no_source:
                console.print(
                    f'  Skipped (no source):  [red]{result.skipped_no_source}[/red] '
                    '(missing source_name or source not in ariadne.yaml)'
                )
            if args.verbose and result.sample:
                console.print('\n[dim]Sample remappings:[/dim]')
                for old, new, ct in result.sample:
                    console.print(f'  [dim]{ct}[/dim] {old} -> {new}')
            if args.verbose and result.skipped_no_source:
                console.print('\n[dim]Skipped breakdown by source_name:[/dim]')
                for sn, n in result.skipped_source_names:
                    console.print(f'  [yellow]{sn}[/yellow]: {n}')
                if result.skipped_sample:
                    console.print('\n[dim]Skipped sample (id, source_name, type, title):[/dim]')
                    for sid, sn, ct, title in result.skipped_sample:
                        sn_disp = sn or '<null>'
                        console.print(
                            f'  [dim]{ct}[/dim] [yellow]{sn_disp}[/yellow] '
                            f'{sid[:8]}... {title!r}'
                        )
            return 0

        console.print(
            '[yellow]No action specified. Use --check, --source-files, '
            '--fix-paths, --fix-catalog-language, --infer-source-name, '
            'or --doc-ids[/yellow]'
        )
        return 1

    finally:
        library.close()


HANDLERS = {
    'check': lambda args: asyncio.run(cmd_check(args)),
    'merge': lambda args: asyncio.run(cmd_merge(args)),
    'cleanup': lambda args: cmd_cleanup(args),
    'migrate': lambda args: cmd_migrate(args),
}
