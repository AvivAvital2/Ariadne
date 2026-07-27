"""Documentation sync & maintenance CLI commands (``check``, ``sync``,
``merge``, ``cleanup``, ``migrate``).

Extracted from cli/generation.py. These keep documentation in step with git
changes: incremental ``sync``, post-merge regeneration, ``cleanup`` of
expired branch docs, and metadata ``migrate``. Wired into the parser via this
module's ``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from config import get_config

if TYPE_CHECKING:
    from docgen.orchestrator import GenerationResult
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


_SCIP_ROUTABLE_LANGUAGES: frozenset[str] = frozenset({
    'javascript', 'scala', 'java',
})


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register documentation sync & maintenance commands."""

    # sync
    sync_parser = subparsers.add_parser('sync', help='Sync documentation with git changes')
    sync_parser.add_argument('--source', '-s', default=None,
                              help='Source name (default from config)')
    sync_parser.add_argument('--status', action='store_true',
                              help='Show sync status without making changes')
    sync_parser.add_argument('--force', '-f', action='store_true',
                              help='Force sync even if no previous state (sets initial sync point)')
    sync_parser.add_argument('--dry-run', action='store_true',
                              help='Show what would be done without making changes')
    sync_parser.add_argument('--skip-generate', action='store_true',
                              help='Skip document regeneration (just update sync state)')
    sync_parser.add_argument('--no-export', action='store_true',
                              help='Skip exporting to markdown after sync')
    sync_parser.add_argument('--vs-main', action='store_true',
                              help='Compare against main branch instead of last sync point')
    sync_parser.add_argument('--branch', action='store_true',
                              help='Tag regenerated docs as branch-specific (experimental, with TTL)')
    sync_parser.add_argument('--concurrency', '-c', type=int, default=3, help='Max concurrent LLM calls for regeneration and theme summarization')


def _maybe_auto_discover_for_new_language(
    cfg, source_name: str, source_path: Path,
    catalog_files: list[str],
) -> None:
    """If sync sees changed files in a SCIP-routable language that
    isn't yet declared in the source's ``index_kinds``, run
    ``ariadne discover --config-only`` to update ariadne.yaml so the
    next ``ariadne index`` picks the new language up.

    Cheap step (just walks the tree + writes yaml). The expensive
    scip-X invocation stays a separate user-invoked step — surfaced
    via a follow-up "run ariadne index" hint.

    Silent no-op when:
    - No catalog_files are SCIP-routable
    - All detected languages already declared
    - cfg has no config_path (yaml not loaded from disk — nothing to write)
    """
    from docgen.catalog_extractor import _detect_language

    detected: set[str] = set()
    for rel in catalog_files:
        full = source_path / rel
        lang = _detect_language(full)
        if lang in _SCIP_ROUTABLE_LANGUAGES:
            detected.add(lang)

    if not detected:
        return

    declared: set[str] = set()
    scip_cfg = cfg.get_source_scip_config(source_name)
    if scip_cfg is not None:
        declared = {
            lang for lang, idx in scip_cfg.index_kinds.items()
            if idx == 'scip'
        }

    missing = detected - declared
    if not missing:
        return

    if cfg.config_path is None:
        return  # nothing to write back to

    console.print(
        f'[yellow]New SCIP-routable language(s) detected in changed '
        f'files: {sorted(missing)}. Updating ariadne.yaml...[/yellow]',
    )
    from cli.index import cmd_discover
    discover_args = argparse.Namespace(
        source=source_name, all=False, dry_run=False,
        review=False, config_only=True,
    )
    cmd_discover(discover_args)
    console.print(
        f'[yellow]Run `ariadne index --source {source_name}` to '
        f'build the cross-source graph for the new language '
        f'(scip-X invocation, ~30s-10min depending on language).[/yellow]',
    )
    # Ordering note: now that index_kinds declares SCIP for this language,
    # catalog extraction for those files routes through resolve_index and
    # will fail-loud (ScipUnavailableError) until `ariadne index` produces
    # the .scip. Warn so the next `catalog-sync` isn't a surprise failure.
    console.print(
        '[yellow]Run it before the next `catalog-sync`: until the index '
        'exists, catalog extraction for the new language fails loud by '
        'design (no silent ast-grep fallback).[/yellow]',
    )


async def _refresh_themes_after_regen(library, cfg, *, concurrency: int) -> None:
    """Refresh cross-cutting theme clusters after a sync regeneration.

    Honors the user's ``--concurrency`` so theme summarization fans out at the
    same width as doc regeneration (and the generate/onboard paths), instead of
    ``generate_themes``' hardcoded default.
    """
    from docgen.themes import refresh_themes
    from writer import LibraryWriter as _LibraryWriter
    async with _LibraryWriter(library) as _writer:
        await refresh_themes(
            library, _writer,
            enabled=getattr(cfg, 'themes_enabled', True),
            summarize_kwargs={'concurrency': concurrency},
        )


async def cmd_sync(args: argparse.Namespace) -> int:
    """Sync documentation with git changes since last sync.

    Cheap incremental flow — reuses the cross-source graph and other
    ``library_scip`` data that ``ariadne index`` last persisted, only
    regenerates docs whose source files actually changed in git.
    Re-running ``ariadne index`` (the expensive scip-X step) is a
    separate user-invoked operation; sync never triggers it.

    This command:
    1. Gets the current git hash from the source repository
    2. Compares with the last synced hash (or main branch with --vs-main)
    3. Finds changed files via git diff
    4. **Auto-detect**: if any changed file is in a SCIP-routable
       language (.ts/.js/.scala/.java) not yet declared in the
       source's ``index_kinds``, runs ``ariadne discover --config-only``
       to update ``ariadne.yaml`` and prints a hint to re-run
       ``ariadne index`` for the new language. Cheap step (just walks
       the tree + writes yaml); expensive scip-X invocation stays
       separate.
    5. Identifies documents that reference those files
    6. Regenerates affected documents
    7. Imports and rebuilds embeddings

    With --vs-main:
    - Compares current branch against main branch
    - Useful for seeing what docs are affected by branch changes

    With --branch:
    - Tags regenerated docs as experimental with branch name
    - Sets expiration date based on branch_doc_ttl_days config
    """
    import subprocess
    from datetime import datetime, timedelta

    from rich.progress import Progress, SpinnerColumn, TextColumn

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

    library = get_library(args.db)

    # Read-side closure-scoped view; writes still go via raw ``library``
    # because ScopedLibrary doesn't expose mutation methods.
    from scope_resolution import make_scoped_library
    scoped = make_scoped_library(cfg, library, source_name)

    try:
        # Get current git hash
        try:
            result = subprocess.run(
                ['git', 'rev-parse', 'HEAD'],
                cwd=source_path,
                capture_output=True,
                text=True,
                check=True,
            )
            current_hash = result.stdout.strip()
        except subprocess.CalledProcessError:
            console.print(f'[red]Failed to get git hash from {source_path}[/red]')
            return 1

        # Get current branch name (needed for --branch and --vs-main)
        from git_ops import get_changed_files_vs_main, get_current_branch
        current_branch = get_current_branch(source_path)

        # Handle --vs-main mode: compare against main branch
        if args.vs_main:
            main_branch = cfg.main_branch

            if current_branch == main_branch:
                console.print(f'[yellow]Already on {main_branch} branch. No branch-specific docs needed.[/yellow]')
                return 0

            console.print(f'Branch: [cyan]{current_branch}[/cyan]')
            console.print(f'Comparing against: [cyan]{main_branch}[/cyan]')

            changed_files = get_changed_files_vs_main(source_path, main_branch)
            if changed_files is None:
                console.print(f'[red]Failed to get git diff vs {main_branch}[/red]')
                return 1

            if not changed_files:
                console.print('[green]No files changed from main.[/green]')
                return 0

            # Filter to catalog-covered extensions (multi-language: .py,
            # .scala, .java, .html, .js/.ts, .json, .yaml, .md, ...).
            from docgen.catalog_writer import CATALOG_EXTS
            catalog_files = [
                f for f in changed_files
                if any(f.endswith(ext) for ext in CATALOG_EXTS)
            ]
            console.print(f'Files changed from main: {len(changed_files)}')
            console.print(f'Catalog files: {len(catalog_files)}')

            # Find affected documents
            affected_docs = scoped.find_documents_by_source_files(changed_files)
            console.print(f'Affected documents: {len(affected_docs)}')

            if affected_docs:
                for doc in affected_docs[:10]:
                    console.print(f'  - "{doc.title}" ({doc.content_type})')
                if len(affected_docs) > 10:
                    console.print(f'  ... and {len(affected_docs) - 10} more')

            if args.dry_run:
                console.print('[yellow]Dry run - no changes made.[/yellow]')
                return 0

            # Regenerate affected documents if --branch is set
            if args.branch and catalog_files and not args.skip_generate:
                console.print()
                console.print('Regenerating branch-specific documentation...')

                from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

                # Calculate expiration date
                expires_at = datetime.now() + timedelta(days=cfg.branch_doc_ttl_days)
                branch_metadata = {
                    'status': 'experimental',
                    'branches': [current_branch],
                    'expires_at': expires_at.isoformat(),
                }

                _scip_cfg = cfg.get_source_scip_config(source_name)
                config = OrchestratorConfig(
                    source_path=source_path,
                    db_path=args.db or Path(cfg.db_path),
                    staleness_db_path=Path(cfg.staleness_db_path),
                    model=cfg.model,
                    doc_types=('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram'),
                    force_regenerate=True,
                    source_config=_scip_cfg,
                ignore_staleness=cfg.source_ignore_staleness(source_name), concurrency=args.concurrency)

                docs_created = 0
                docs_failed = 0
                created_doc_ids: list[str] = []

                with Progress(
                    SpinnerColumn(),
                    TextColumn('[progress.description]{task.description}'),
                    console=console,
                ) as progress:
                    task = progress.add_task('Regenerating...', total=len(catalog_files))

                    async with DocGenOrchestrator(config) as orchestrator:
                        semaphore = asyncio.Semaphore(orchestrator.config.concurrency)
                        errors: list[str] = []
                        completed_count = 0
                        progress_lock = asyncio.Lock()

                        async def process_one(rel_path: str) -> 'GenerationResult | None':
                            nonlocal completed_count
                            file_path = source_path / rel_path

                            if not file_path.exists():
                                async with progress_lock:
                                    completed_count += 1
                                    progress.update(task, completed=completed_count)
                                return None

                            async with semaphore:
                                try:
                                    result = await orchestrator._process_file(file_path)
                                    async with progress_lock:
                                        completed_count += 1
                                        progress.update(task, completed=completed_count,
                                                      description=f'({completed_count}/{len(catalog_files)}) Processing')
                                    return result
                                except Exception as e:
                                    errors.append(f'{rel_path}: {e}')
                                    async with progress_lock:
                                        completed_count += 1
                                        progress.update(task, completed=completed_count)
                                    return None

                        tasks = [process_one(rel_path) for rel_path in catalog_files]
                        results = await asyncio.gather(*tasks)

                        for result in results:
                            if result:
                                docs_created += result.docs_generated
                                docs_failed += result.docs_failed
                                created_doc_ids.extend(result.doc_ids)

                        docs_failed += len(errors)

                        for error in errors:
                            console.print(f'[red]Error: {error}[/red]')

                # Update created documents with branch metadata
                for doc_id in created_doc_ids:
                    doc = scoped.get_document(doc_id)
                    if doc:
                        updated_metadata = dict(doc.metadata)
                        updated_metadata.update(branch_metadata)
                        library.update_document(doc_id, metadata=updated_metadata)

                console.print(f'Generated {docs_created} branch-specific document(s) for {current_branch}')
                console.print(f'[dim](experimental, expires {expires_at.strftime("%Y-%m-%d")})[/dim]')

                if docs_failed > 0:
                    console.print(f'[yellow]{docs_failed} failed[/yellow]')

            # Refresh themes (cross-cutting clusters) — pull-based, gated by config.
            try:
                await _refresh_themes_after_regen(library, cfg, concurrency=args.concurrency)
            except Exception as e:
                console.print(f'[yellow]Themes refresh failed: {e}[/yellow]')

            # Export if requested
            if not args.no_export:
                console.print()
                console.print('Exporting to markdown...')
                from export import LibraryExporter
                exporter = LibraryExporter(library)
                output_dir = cfg.resolve_docs_path(source_name)
                paths = exporter.export_all(
                    output_dir=output_dir,
                    source_name=source_name,
                    source_path=source_path,
                )
                console.print(f'Exported {len(paths)} documents to {output_dir}')

            return 0

        # Standard sync mode: compare against last sync point
        sync_state = library.get_sync_state(source_name)

        if sync_state is None:
            if args.status:
                console.print(f'[yellow]Never synced. Current hash: {current_hash[:8]}[/yellow]')
                return 0

            console.print(f'[yellow]No previous sync state for "{source_name}".[/yellow]')
            console.print('This will be recorded as the initial sync point.')

            if not args.force:
                console.print('Run with --force to set the initial sync point, or use "ariadne generate" first.')
                return 0

            # Set initial sync point
            library.set_sync_state(source_name, current_hash)
            console.print(f'[green]Initial sync point set: {current_hash[:8]}[/green]')
            return 0

        last_hash, last_synced = sync_state

        if args.status:
            console.print(f'Last synced: {last_hash[:8]} at {last_synced[:19]}')
            console.print(f'Current:     {current_hash[:8]}')
            if last_hash == current_hash:
                console.print('[green]Up to date.[/green]')
            else:
                # Show number of commits
                try:
                    result = subprocess.run(
                        ['git', 'rev-list', '--count', f'{last_hash}..{current_hash}'],
                        cwd=source_path,
                        capture_output=True,
                        text=True,
                        check=True,
                    )
                    commit_count = result.stdout.strip()
                    console.print(f'[yellow]{commit_count} commits behind.[/yellow]')
                except subprocess.CalledProcessError:
                    console.print('[yellow]Changes detected.[/yellow]')
            # Code-level extractor-coverage staleness (invisible to the
            # git-diff staleness above); surfaced regardless of ignore_staleness.
            from docgen.extraction_coverage import coverage_notice
            _cov = coverage_notice(source_name, source_path)
            if _cov:
                console.print(f'[yellow]⚠ {_cov}[/yellow]')
            return 0

        if last_hash == current_hash:
            console.print('[green]Already up to date.[/green]')
            return 0

        # Get changed files
        console.print(f'Comparing {last_hash[:8]}..{current_hash[:8]}')

        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', f'{last_hash}..{current_hash}'],
                cwd=source_path,
                capture_output=True,
                text=True,
                check=True,
            )
            changed_files = [f for f in result.stdout.strip().split('\n') if f]
        except subprocess.CalledProcessError as e:
            console.print(f'[red]Failed to get git diff: {e}[/red]')
            return 1

        if not changed_files:
            console.print('[green]No files changed.[/green]')
            library.set_sync_state(source_name, current_hash)
            return 0

        console.print(f'Changed files: {len(changed_files)}')

        # Filter to catalog-covered extensions (multi-language).
        from docgen.catalog_writer import CATALOG_EXTS
        catalog_files = [
            f for f in changed_files
            if any(f.endswith(ext) for ext in CATALOG_EXTS)
        ]
        console.print(f'Catalog files: {len(catalog_files)}')

        # Auto-detect: did a new SCIP-routable language appear in
        # the changed set? If so, run discover (config-only) to add
        # it to ariadne.yaml so the next ariadne index picks it up.
        # Cheap step — just walks the tree and writes yaml. The
        # expensive scip-X invocation stays a separate user-invoked
        # step.
        _maybe_auto_discover_for_new_language(
            cfg, source_name, source_path, catalog_files,
        )

        if args.dry_run or args.status:
            for f in catalog_files[:15]:
                console.print(f'  [dim]{f}[/dim]')
            if len(catalog_files) > 15:
                console.print(f'  [dim]... and {len(catalog_files) - 15} more[/dim]')

        # Find affected documents
        affected_docs = scoped.find_documents_by_source_files(changed_files)
        console.print(f'Affected documents: {len(affected_docs)}')

        if affected_docs:
            for doc in affected_docs[:10]:
                console.print(f'  - {doc.title} ({doc.content_type})')
            if len(affected_docs) > 10:
                console.print(f'  ... and {len(affected_docs) - 10} more')

        if args.dry_run:
            console.print('[yellow]Dry run - no changes made.[/yellow]')
            return 0

        # Regenerate affected documents
        if catalog_files and not args.skip_generate:
            console.print()
            console.print('Regenerating documentation for changed files...')

            from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

            _scip_cfg = cfg.get_source_scip_config(source_name)
            config = OrchestratorConfig(
                source_path=source_path,
                db_path=args.db or Path(cfg.db_path),
                staleness_db_path=Path(cfg.staleness_db_path),
                model=cfg.model,
                # Without ``provider=``, OrchestratorConfig falls back
                # to ``'openai'`` regardless of what ``ariadne.yaml``
                # sets — silently routing Anthropic models to OpenAI's
                # endpoint and 404'ing every request. Pass it through.
                provider=cfg.provider,
                doc_types=('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram'),
                force_regenerate=True,
                source_config=_scip_cfg,
            ignore_staleness=cfg.source_ignore_staleness(source_name), concurrency=args.concurrency)

            docs_created = 0
            docs_failed = 0

            with Progress(
                SpinnerColumn(),
                TextColumn('[progress.description]{task.description}'),
                console=console,
            ) as progress:
                task = progress.add_task('Regenerating...', total=len(catalog_files))

                async with DocGenOrchestrator(config) as orchestrator:
                    semaphore = asyncio.Semaphore(orchestrator.config.concurrency)
                    errors: list[str] = []
                    completed_count = 0
                    progress_lock = asyncio.Lock()

                    async def process_one(rel_path: str) -> 'GenerationResult | None':
                        nonlocal completed_count
                        file_path = source_path / rel_path

                        if not file_path.exists():
                            async with progress_lock:
                                completed_count += 1
                                progress.update(task, completed=completed_count)
                            return None

                        async with semaphore:
                            try:
                                result = await orchestrator._process_file(file_path)
                                async with progress_lock:
                                    completed_count += 1
                                    progress.update(task, completed=completed_count,
                                                  description=f'({completed_count}/{len(catalog_files)}) Processing')
                                return result
                            except Exception as e:
                                errors.append(f'{rel_path}: {e}')
                                async with progress_lock:
                                    completed_count += 1
                                    progress.update(task, completed=completed_count)
                                return None

                    tasks = [process_one(rel_path) for rel_path in catalog_files]
                    results = await asyncio.gather(*tasks)

                    # Aggregate results
                    for result in results:
                        if result:
                            docs_created += result.docs_generated
                            docs_failed += result.docs_failed

                    # Count errors as failures
                    docs_failed += len(errors)

                    # Print errors at the end
                    for error in errors:
                        console.print(f'[red]Error: {error}[/red]')

            console.print(f'Regenerated: {docs_created} documents ({docs_failed} failed)')

        # Refresh themes (cross-cutting clusters) — pull-based, gated by config.
        try:
            await _refresh_themes_after_regen(library, cfg, concurrency=args.concurrency)
        except Exception as e:
            console.print(f'[yellow]Themes refresh failed: {e}[/yellow]')

        # Export by default (unless --no-export)
        if not args.no_export:
            console.print()
            console.print('Exporting to markdown...')
            from export import LibraryExporter
            exporter = LibraryExporter(library)
            output_dir = cfg.resolve_docs_path(source_name)
            paths = exporter.export_all(
                output_dir=output_dir,
                source_name=source_name,
                source_path=source_path,
            )
            console.print(f'Exported {len(paths)} documents to {output_dir}')

        # Deprecate stale gotchas
        deprecated_count = library.deprecate_stale_gotchas()
        if deprecated_count > 0:
            console.print(f'Deprecated {deprecated_count} stale gotcha(s).')

        # Update sync state
        library.set_sync_state(source_name, current_hash)
        console.print()
        console.print(f'[green]Sync complete: {current_hash[:8]}[/green]')

        return 0

    finally:
        library.close()


HANDLERS = {
    'sync': lambda args: asyncio.run(cmd_sync(args)),
    
}
