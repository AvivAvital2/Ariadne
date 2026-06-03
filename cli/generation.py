"""Documentation generation, sync, and maintenance CLI commands."""
from __future__ import annotations

import argparse
import asyncio
import io
import os
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from config import get_config

if TYPE_CHECKING:
    from docgen.orchestrator import GenerationResult
    from library import Library

# Default paths
DEFAULT_DB_PATH = Path('ariadne.db')

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


# Re-exports — the ``generate`` command's helpers and entry point now live
# in cli_generate.py. Kept importable from here for backwards-compatibility
# with tests and external callers.
from cli.generate import (  # noqa: E402
    cmd_generate,
)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register generation commands."""
    # generate — now lives in cli_generate.py
    from cli.generate import register_generate_parser
    register_generate_parser(subparsers)

    # batch — pending-batch management (lives in cli_batch.py).
    from cli.batch import register_batch_parser
    register_batch_parser(subparsers)

    # check
    check_parser = subparsers.add_parser('check', help='Check documentation status')
    check_parser.add_argument('--source', '-s', default=None,
                               help='Source path or name (default from config)')
    check_parser.add_argument('--verbose', '-v', action='store_true',
                               help='Show list of stale/undocumented files')

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

    # catalog-sync
    catalog_sync_parser = subparsers.add_parser('catalog-sync',
        help='Sync structural code catalog via ast-grep (per-element docs)')
    catalog_sync_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    catalog_sync_parser.add_argument('--allow-degraded', action='store_true',
        help=(
            'Permit ast-grep fallback when a SCIP index is unavailable. '
            'Without this flag a missing/stale/corrupt SCIP artifact causes '
            'catalog-sync to exit with code 2 (the fail-loud default).'
        ),
    )
    catalog_sync_parser.add_argument('--concurrency', '-c', type=int, default=4,
        help=(
            'Max concurrent files to process (default: 4). Catalog-sync is '
            'network-bound on the embedding API; raise to 8-16 if your tier '
            'tolerates it, lower to 1 for sequential safety.'
        ),
    )
    catalog_sync_parser.add_argument('--force', '-f', action='store_true',
        help=(
            'Bypass the file_sha short-circuit and re-extract every file. '
            'Use after changing extractor config (new SCIP index, updated '
            'exclude rules, etc.) to pick up changes the staleness check '
            "wouldn't otherwise see."
        ),
    )

    # catalog-describe
    catalog_desc_parser = subparsers.add_parser('catalog-describe',
        help='Generate LLM descriptions for catalog elements')
    catalog_desc_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    catalog_desc_parser.add_argument('--force', '-f', action='store_true',
        help='Regenerate descriptions even if already present')
    catalog_desc_parser.add_argument('--model', '-m', default=None,
        help='LLM model to use (default from config)')
    catalog_desc_parser.add_argument('--concurrency', '-c', type=int, default=4,
        help='Max concurrent LLM requests (default: 4)')
    catalog_desc_parser.add_argument('--max-calls', type=int, default=None,
        help='Cap on LLM calls')
    catalog_desc_parser.add_argument('--dry-run', action='store_true',
        help='Estimate cost (candidates × per-call tokens × model rate) '
             'and exit without making any LLM calls')
    catalog_desc_parser.add_argument('--batch', action='store_true',
        help='Use Anthropic Message Batches API (~50%% off, up to 24h '
             'SLA) instead of live per-element requests')
    catalog_desc_parser.add_argument('--resume', action='store_true',
        help='Resume a previously-submitted batch (fetch + apply '
             'results from the pending batch matching this source + '
             'model). Use after --batch was interrupted before '
             'fetching results, or on a fresh process to pick up an '
             'overnight batch.')

    # dry-run — wrapper that runs the free phases (discover, index,
    # catalog-sync) and estimates the LLM phases (catalog-describe,
    # generate, themes build) for a unified cost preview.
    dryrun_parser = subparsers.add_parser('dry-run',
        help='Run the free pipeline phases and estimate the cost of '
             'the remaining LLM-paid phases')
    dryrun_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    dryrun_parser.add_argument('--model', '-m', default=None,
        help='LLM model to use for estimates (default from config)')
    dryrun_parser.add_argument('--db', default=None,
        help='Override db_path for the Library')
    dryrun_parser.add_argument('--verbose', '-v', action='store_true',
        help='Show full per-phase output (indexer adapter detail, '
             'file change lists, progress bars). Default is a tight '
             'one-line-per-phase summary.')

    # onboard — full pipeline from discover through themes. Without
    # --approve it falls through to dry-run + a hint. With --approve
    # it runs all six phases (discover, index, catalog-sync,
    # catalog-describe, generate, themes build) and stops on the
    # first non-zero rc.
    onboard_parser = subparsers.add_parser('onboard',
        help='Onboard a source end-to-end in one run: free phases + cost '
             'preview, then (on --approve or a yes) the paid phases — '
             'without re-indexing.')
    onboard_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    onboard_parser.add_argument('--model', '-m', default=None,
        help='LLM model for paid phases (default from config)')
    onboard_parser.add_argument('--db', default=None,
        help='Override db_path for the Library')
    onboard_parser.add_argument('--verbose', '-v', action='store_true',
        help='Pass through to sub-phases')
    onboard_parser.add_argument('--approve', action='store_true',
        help='Skip the interactive proceed-prompt and run the paid '
             'phases after the preview. Without it, onboard prompts on a '
             'TTY (and stops at the preview when non-interactive).')
    onboard_parser.add_argument('--types', default=None,
        help='Comma-separated doc types for the generate phase '
             '(explanation,architecture,qa,gotcha,diagram). Omit to pick '
             'interactively after the cost preview (all on a non-TTY).')
    # --live / --batch skip the interactive picker (useful for CI /
    # scripted runs). Without either, onboard prompts. Names mirror
    # the picker labels so users don't have to map between "what they
    # see" and "what they type".
    mode_group = onboard_parser.add_mutually_exclusive_group()
    mode_group.add_argument('--live', dest='batch_mode',
        action='store_const', const='live',
        help='Live LLM dispatch — finishes in minutes, full price. '
             'Skips the interactive Live/Batch picker.')
    mode_group.add_argument('--batch', dest='batch_mode',
        action='store_const', const='batch',
        help='Batched LLM dispatch — up to 24h SLA, ~50%% off '
             '(Anthropic Message Batches API). Skips the picker.')
    onboard_parser.set_defaults(batch_mode=None)
    onboard_parser.add_argument('--concurrency', '-c', type=int,
        default=None,
        help='Max parallel LLM/embedding calls for catalog-sync, '
             'catalog-describe (live mode), and generate. Defaults '
             'to each phase\'s baked-in default if unset '
             '(catalog-sync=4, describe=4, generate=3).')

    # notify-changed
    notify_parser = subparsers.add_parser('notify-changed',
        help='Incremental catalog update for specific changed files')
    notify_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    notify_parser.add_argument('--files', nargs='+', default=None,
        help='Changed file paths relative to source root')
    notify_parser.add_argument('--regenerate', action='store_true', help='Regenerate LLM docs for changed files')

    # symbol
    symbol_parser = subparsers.add_parser('symbol',
        help='Look up a catalog element by qualified_name')
    symbol_parser.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    symbol_parser.add_argument('--name', required=True,
        help='Qualified name of the element')
    symbol_parser.add_argument('--file', default=None,
        help='Optional file path — helps scope suggestions')


    # list-file-scopes (scope-aware C)
    lfs_parser = subparsers.add_parser(
        'list-file-scopes',
        help='List qualified_names of all elements in a file',
    )
    lfs_parser.add_argument('--source', '-s', required=True)
    lfs_parser.add_argument('--file', '-f', required=True)
    # body subparser
    body_parser = subparsers.add_parser('body', help='Return current body text of a catalog element')
    body_parser.add_argument('--source', '-s', default=None)
    body_parser.add_argument('--file', '-f', default=None)
    body_parser.add_argument('--name', required=True)

    # themes (Phase 7) — nested subcommands: build / list / show
    themes_parser = subparsers.add_parser(
        'themes',
        help='Manage cross-cutting theme clusters (Leiden community detection)',
    )
    themes_sub = themes_parser.add_subparsers(
        dest='themes_action',
        help='Themes subcommand',
    )

    themes_build = themes_sub.add_parser(
        'build', help='Discover/refresh themes for the configured source',
    )
    themes_build.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')

    themes_list = themes_sub.add_parser(
        'list', help='List discovered themes',
    )
    themes_list.add_argument('--source', '-s', default=None,
        help='Filter by source name')
    themes_list.add_argument('--include-incoherent', action='store_true',
        help='Include themes the LLM marked INCOHERENT')
    themes_list.add_argument('--limit', type=int, default=50,
        help='Maximum number of themes to print (default: 50)')

    themes_show = themes_sub.add_parser(
        'show', help='Show full theme doc content for a cluster',
    )
    themes_show.add_argument('cluster_id',
        help='Stable cluster_id (printed by `themes list`)')

    # diff-docs (Catalog transition Phase 3.2) — side-by-side legacy vs
    # new prompts for spot-checking during the dual-run window.
    diff_parser = subparsers.add_parser(
        'diff-docs',
        help='Compare legacy vs catalog-driven generation prompts for a file',
    )
    diff_parser.add_argument('--file', '-f', required=True,
        help='Path to a Python file to diff (must exist)')
    diff_parser.add_argument('--type', '-t', default='explanation',
        help='Doc type to compare (default: explanation)')
    diff_parser.add_argument('--with-llm', action='store_true',
        help='Also run both pipelines with the real LLM and print outputs')


_SCIP_ROUTABLE_LANGUAGES: frozenset[str] = frozenset({
    'javascript', 'scala', 'java',
})


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
    from cli.core import cmd_discover
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
    )

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
                )

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
                from docgen.themes import refresh_themes
                from writer import LibraryWriter as _LibraryWriter
                async with _LibraryWriter(library) as _writer:
                    await refresh_themes(
                        library, _writer,
                        enabled=getattr(cfg, 'themes_enabled', True),
                    )
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
            )

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
            from docgen.themes import refresh_themes
            from writer import LibraryWriter as _LibraryWriter
            async with _LibraryWriter(library) as _writer:
                await refresh_themes(
                    library, _writer,
                    enabled=getattr(cfg, 'themes_enabled', True),
                )
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
        )

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



async def cmd_catalog_sync(args: argparse.Namespace) -> int:
    """Sync structural code catalog for a source.

    For Scala/Java sources with ``index_kinds.<lang> = "scip"``, extraction
    routes through SCIP. A missing/stale/corrupt SCIP artifact yields
    structured per-file ``scip_error`` summaries; this command exits with
    code 2 if any are present (the fail-loud contract). ``--allow-degraded``
    lets the user opt into ast-grep fallback for those files instead.
    """
    from docgen.catalog_writer import sync_source_catalog
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

    # Resolve SCIP config for this source (may be None for non-SCIP sources).
    # CLI flag --allow-degraded overrides the YAML default, which is False.
    scip_config = cfg.get_source_scip_config(source_name)
    if scip_config is not None and getattr(args, 'allow_degraded', False):
        from attrs import evolve
        scip_config = evolve(scip_config, allow_degraded=True)

    # Per-source excludes from ariadne.yaml — keep secrets / generated dirs
    # out of the catalog (and therefore out of embeddings + docs DB).
    exclude_patterns: tuple[str, ...] = ()
    sc = cfg.get_source_config(source_name)
    if sc is not None:
        exclude_patterns = sc.exclude
    # Resolve the full effective excluded-dir set (global policy
    # ∪ source.exclude_dirs − source.exempt_dirs).
    exclude_dir_names = cfg.resolve_excluded_dirs(source_name)

    library = get_library(args.db)
    try:
        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress_columns = (
            SpinnerColumn(),
            TextColumn('[bold cyan]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn('·'),
            TimeElapsedColumn(),
            TextColumn('eta'),
            TimeRemainingColumn(),
        )

        with Progress(
            *progress_columns, console=console,
            transient=getattr(args, 'quiet', False),
        ) as progress:
            task_id = progress.add_task(
                f'Syncing catalog for {source_name}', total=0,
            )

            def on_prog(current: int, total: int, current_file: str | None) -> None:
                # First event seeds the total; later events advance.
                if current_file:
                    short = Path(current_file).name
                    desc = f'Syncing {source_name}: [dim]{short}[/dim]'
                else:
                    desc = f'Syncing {source_name}: done'
                progress.update(
                    task_id, completed=current, total=total, description=desc,
                )

            async with LibraryWriter(library) as writer:
                summaries = await sync_source_catalog(
                    library, writer, source_name, source_path,
                    source_config=scip_config,
                    on_progress=on_prog,
                    concurrency=getattr(args, 'concurrency', 4),
                    exclude_patterns=exclude_patterns,
                    exclude_dir_names=exclude_dir_names,
                    force=getattr(args, 'force', False),
                )

        total_added = sum(s.added for s in summaries)
        total_modified = sum(s.modified for s in summaries)
        total_removed = sum(s.removed for s in summaries)
        total_unchanged = sum(s.unchanged for s in summaries)
        total_skipped = sum(1 for s in summaries if s.skipped)

        if not getattr(args, 'quiet', False):
            console.print()
            console.print(f'Catalog sync for [cyan]{source_name}[/cyan]:')
            console.print(f'  Files scanned: {len(summaries)} ({total_skipped} skipped)')
            console.print(
                f'  Elements: [green]+{total_added}[/green] added, '
                f'[yellow]~{total_modified}[/yellow] modified, '
                f'[red]-{total_removed}[/red] removed, {total_unchanged} unchanged'
            )

            changed = [s for s in summaries if s.added or s.modified or s.removed]
            if changed:
                console.print()
                console.print(f'Changed files ({len(changed)}):')
                for s in changed[:20]:
                    parts = []
                    if s.added: parts.append(f'+{s.added}')
                    if s.modified: parts.append(f'~{s.modified}')
                    if s.removed: parts.append(f'-{s.removed}')
                    console.print(f'  {s.file}: ' + ', '.join(parts))
                if len(changed) > 20:
                    console.print(f'  ... and {len(changed) - 20} more')

        # Surface SCIP errors per-file. Without --allow-degraded these
        # constitute a fail-loud failure (exit 2); with the flag we still
        # show them but exit 0 so the user knows what was skipped.
        scip_failures = [s for s in summaries if getattr(s, 'scip_error', None)]
        if scip_failures:
            console.print()
            console.print('[red]SCIP-backed extraction failed for some files:[/red]')
            for s in scip_failures[:20]:
                err = s.scip_error
                console.print(
                    f'  [red]{s.file}[/red] — '
                    f'{type(err).__name__}: '
                    f'reason={getattr(err, "reason", "?")}'
                )
            if len(scip_failures) > 20:
                console.print(f'  ... and {len(scip_failures) - 20} more')

            if not getattr(args, 'allow_degraded', False):
                console.print(
                    '\n[dim]Pass --allow-degraded to permit ast-grep fallback '
                    'for these files.[/dim]'
                )
                return 2
            console.print(
                '\n[yellow]--allow-degraded set; failures above did not '
                'block the run, but those files have no catalog entries.[/yellow]'
            )

        return 0
    finally:
        library.close()




# Empirical per-call token averages for catalog-describe. The prompt
# is a small fixed template (kind, qualified_name, parent, signature,
# file, lines) plus a system preamble, and the response is 1-2
# sentences. Sampled values across mixed Python/TS/Scala sources:
#   input  ~200 tokens (preamble + element metadata)
#   output ~60  tokens (concise description)
# These are conservative midpoints; the actual cost will land within
# roughly ±30% of the estimate.
_DESCRIBE_INPUT_TOKENS_PER_CALL = 200
_DESCRIBE_OUTPUT_TOKENS_PER_CALL = 60


def _describe_tokens_per_call(store, model: str) -> tuple[float, float]:
    """(input, output) tokens per catalog-describe call for ``model``.

    Uses real calibrated usage from the store when available, else the
    empirical 200/60 heuristic — so the describe estimate self-tunes
    after a run."""
    cal = store.mean_tokens(phase='describe', model=model) if store else None
    if cal is not None:
        return cal.mean_input, cal.mean_output
    return _DESCRIBE_INPUT_TOKENS_PER_CALL, _DESCRIBE_OUTPUT_TOKENS_PER_CALL


def _print_catalog_describe_cost_estimate(
    library: "Library",
    source_name: str,
    *,
    model: str,
    force: bool,
    max_calls: int | None,
) -> int:
    """Count candidates, multiply by per-call token estimates × model
    rate, print a cost preview, and exit. No LLM calls made.
    """
    from docgen.pricing import LLM_PRICING

    # Same candidate selection as describe_source_elements so the
    # estimate matches what the real run would actually call against.
    all_catalog = library.list_documents(
        content_type='catalog', limit=100_000,
    )
    candidates = [
        d for d in all_catalog
        if d.metadata.get('source_name') == source_name
        and d.metadata.get('kind') == 'element'
    ]
    if force:
        to_describe = list(candidates)
    else:
        to_describe = [
            d for d in candidates if not d.metadata.get('description')
        ]

    already_had = len(candidates) - len(to_describe)
    planned = (
        min(max_calls, len(to_describe))
        if max_calls is not None else len(to_describe)
    )

    total_input = planned * _DESCRIBE_INPUT_TOKENS_PER_CALL
    total_output = planned * _DESCRIBE_OUTPUT_TOKENS_PER_CALL

    rates = LLM_PRICING.get(model)
    if rates is None:
        cost_line = (
            f'[yellow]Cost: model {model!r} not in LLM_PRICING — '
            f'cannot compute. Configured prices in '
            f'docgen/pricing.py:LLM_PRICING.[/yellow]'
        )
    else:
        input_per_m, output_per_m = rates
        cost_usd = (
            total_input * input_per_m / 1_000_000
            + total_output * output_per_m / 1_000_000
        )
        cost_line = f'  Estimated cost: [bold]${cost_usd:.2f}[/bold]'

    console.print(
        f'[bold]Dry-run cost estimate for catalog-describe[/bold] '
        f'(source: {source_name})',
    )
    console.print(f'  Catalog elements: {len(candidates)}')
    console.print(f'  Already described: {already_had}')
    console.print(f'  Planned LLM calls: {planned}')
    console.print(
        f'  Tokens (est): {total_input:,} in / {total_output:,} out '
        f'({_DESCRIBE_INPUT_TOKENS_PER_CALL}/{_DESCRIBE_OUTPUT_TOKENS_PER_CALL} per call)',
    )
    console.print(f'  Model: {model}')
    console.print(cost_line)
    console.print(
        '[dim]Run without --dry-run to actually generate '
        'descriptions.[/dim]',
    )
    return 0


# Empirical per-theme token averages for the themes-build LLM step.
# The summarization prompt includes the cluster's member list + per-
# member descriptions; output is a coherent ~paragraph (or
# "INCOHERENT" terminator). Mid-range estimates:
_THEMES_INPUT_TOKENS_PER_THEME = 2000
_THEMES_OUTPUT_TOKENS_PER_THEME = 600


def _estimate_themes_cost(
    library: "Library", model: str,
) -> tuple[int, float | None, tuple[float, float] | None]:
    """Count existing themes and estimate the LLM cost to (re)summarize
    each one. Returns ``(theme_count, cost_usd_or_None, rates)``.

    If the themes table is empty (clustering hasn't run yet), returns
    ``(0, 0.0, rates)`` — there's nothing to summarize. We don't run
    Leiden here; that's the caller's choice, since it would mutate the
    DB. The estimate is only meaningful AFTER clustering has populated
    the themes table.
    """
    from docgen.pricing import LLM_PRICING

    themes = library.list_themes(coherent_only=False)
    n = len(themes)
    rates = LLM_PRICING.get(model)
    if rates is None:
        return n, None, None
    input_per_m, output_per_m = rates
    input_total = n * _THEMES_INPUT_TOKENS_PER_THEME
    output_total = n * _THEMES_OUTPUT_TOKENS_PER_THEME
    cost = (
        input_total * input_per_m / 1_000_000
        + output_total * output_per_m / 1_000_000
    )
    return n, cost, rates


class _PhaseUI:
    """Shared UX wrappers for multi-phase orchestrators (``dry-run`` and
    ``onboard``).

    Provides three operations:

    - ``silence_fast(label)`` — context that silences stdout/stderr and
      shows a transient spinner. Use for phases without a native
      progress bar (e.g., discover).
    - ``passthrough(label)`` — context that lets the sub-phase's own
      output through. Use for phases with native progress bars
      (index, catalog-sync, catalog-describe, generate, themes).
    - ``replay_captured(label)`` — dump the captured stdout/stderr for
      a previously-silenced phase to stderr. Call on failure so users
      see the diagnostic that explains the non-zero rc.

    In verbose mode, both context managers fall through to a
    ``Phase: <label>`` header and DON'T silence anything — useful for
    debugging pipelines.
    """

    def __init__(self, *, verbose: bool) -> None:
        import sys
        from rich.console import Console
        self.verbose = verbose
        self._sys = sys
        self._progress_console = Console(
            file=sys.__stderr__, force_terminal=True,
        )
        self._captured: dict[str, tuple[io.StringIO, io.StringIO]] = {}

    @contextmanager
    def silence_fast(self, label: str):
        from rich.progress import (
            Progress, SpinnerColumn, TextColumn, TimeElapsedColumn,
        )
        if self.verbose:
            console.print(f'\n[cyan]Phase: {label}[/cyan]')
            yield
            return
        progress = Progress(
            SpinnerColumn(),
            TextColumn('[cyan]{task.description}[/cyan]'),
            TimeElapsedColumn(),
            console=self._progress_console,
            transient=True,
        )
        stdout_buf = io.StringIO()
        stderr_buf = io.StringIO()
        self._captured[label] = (stdout_buf, stderr_buf)
        with progress:
            progress.add_task(f'{label}…', total=None)
            with redirect_stdout(stdout_buf):
                with redirect_stderr(stderr_buf):
                    yield

    def replay_captured(self, label: str) -> None:
        bufs = self._captured.get(label)
        if bufs is None:
            return
        stdout_text = bufs[0].getvalue()
        stderr_text = bufs[1].getvalue()
        if stdout_text:
            self._sys.stderr.write(stdout_text)
        if stderr_text:
            self._sys.stderr.write(stderr_text)
        self._sys.stderr.flush()

    @contextmanager
    def passthrough(self, label: str):
        if self.verbose:
            console.print(f'\n[cyan]Phase: {label}[/cyan]')
        yield


def _print_index_summary(summary: list) -> None:
    """Render the per-language Index summary nested under the caller's
    ``✓ Index`` line, e.g.::

        ✓ Index — SCIP graph persisted to library_scip
            Python      1500 files · 0:08
            Java         900 files · 1:12
    """
    for row in summary:
        secs = int(row.get('seconds', 0))
        elapsed = f'{secs // 60}:{secs % 60:02d}'
        console.print(
            f"    {row['language']:<11}{row['files']:>6} files · {elapsed}",
        )


async def cmd_dry_run(args: argparse.Namespace) -> int:
    """Wrapper: run the free phases (discover, index, catalog-sync)
    and estimate the cost of the LLM-paid phases (catalog-describe,
    generate, themes build).

    The free phases must actually run so subsequent estimates have
    something to count against (catalog-describe counts catalog rows
    that catalog-sync creates). The LLM phases are estimated inline
    — no API calls are made.
    """
    import warnings

    # Suppress SyntaxWarning from ast.parse() on the *analyzed* source. The
    # free phases below (discover/index/catalog-sync) parse target files that
    # may contain invalid escape sequences (e.g. an unescaped ``\[`` in a
    # regex). Those are the analyzed codebase's lint issues — not Ariadne's —
    # and would otherwise leak into dry-run output as "<unknown>:NNN
    # SyntaxWarning". Mirrors the filter cmd_generate installs for the same
    # reason (cli_generate.py); must be set before any parse path runs.
    warnings.filterwarnings('ignore', category=SyntaxWarning)

    from cli.core import cmd_discover, cmd_index
    from docgen.pricing import LLM_PRICING
    # Late import so monkeypatches to ``config.get_config`` in tests
    # take effect inside this command. Without this, the
    # module-level ``from config import get_config`` reference at the
    # top of cli_generation.py captures the original function at
    # import time and ignores monkeypatches done after that.
    import config as _config_module
    cfg = _config_module.get_config()
    source_name = args.source or cfg.default_source
    if source_name is None:
        console.print(
            '[red]No source specified and no default_source in '
            'config.[/red]',
        )
        return 1
    model = args.model or cfg.model

    verbose = getattr(args, 'verbose', False)

    console.print(
        f'[bold]ariadne dry-run[/bold] · source: {source_name} '
        f'· model: {model}',
    )

    def _ns(**overrides) -> argparse.Namespace:
        base = argparse.Namespace(**vars(args))
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    ui = _PhaseUI(verbose=verbose)
    _silence_fast_phase = ui.silence_fast
    _replay_captured = ui.replay_captured
    _passthrough_phase = ui.passthrough

    # ---- Free phases (run for real, no LLM cost) -----------------------
    discover_args = _ns(
        source=source_name, all=False, dry_run=False,
        review=False, config_only=False,
    )
    with _silence_fast_phase('Walking source tree and detecting languages'):
        rc = cmd_discover(discover_args)
    if rc != 0:
        _replay_captured('Walking source tree and detecting languages')
        return rc
    if not verbose:
        console.print('  [green]✓[/green] Discover — source tree walked, manifest written')

    # Index has its own per-language progress bar with file count +
    # ETA — let that through. But propagate quiet=True (when default
    # mode) so cmd_index skips its chatty boilerplate (Running X
    # adapter, cwd, output, Wrote, Persisted, Indexed N literals)
    # AND suppresses scip-X subprocess stderr (pyright warnings etc.)
    # while keeping the progress widget visible.
    index_args = _ns(
        source=source_name, all=False, dry_run=False, kind=None,
        quiet=not verbose,
    )
    index_summary: list = []
    with _passthrough_phase('Indexing symbols (per-language SCIP)'):
        rc = cmd_index(index_args, phase_summary=index_summary)
    if rc != 0:
        return rc
    if not verbose:
        console.print('  [green]✓[/green] Index — SCIP graph persisted to library_scip')
        _print_index_summary(index_summary)

    # Catalog-sync also has its own progress bar (files scanned /
    # total). Same quiet propagation rationale.
    # Honor an explicit --concurrency when present (onboard passes its
    # args through); `dry-run` itself has no such flag → default 4.
    _cc = getattr(args, 'concurrency', None)
    catalog_sync_args = _ns(
        source=source_name, allow_degraded=False,
        concurrency=_cc if _cc is not None else 4,
        force=False, quiet=not verbose,
    )
    with _passthrough_phase('Cataloging files'):
        rc = await cmd_catalog_sync(catalog_sync_args)
    if rc != 0:
        return rc
    if not verbose:
        console.print(
            '  [green]✓[/green] Catalog-sync — file_index and element '
            'docs written',
        )

    # ---- Paid phases (estimated, not executed) -------------------------
    library = get_library(args.db)
    try:
        # catalog-describe: reuse the existing helper, capture cost
        # via the LLM_PRICING table directly so we can sum.
        all_catalog = library.list_documents(
            content_type='catalog', limit=100_000,
        )
        candidates = [
            d for d in all_catalog
            if d.metadata.get('source_name') == source_name
            and d.metadata.get('kind') == 'element'
            and not d.metadata.get('description')
        ]
        # Calibration: use real per-call tokens from past runs when the
        # store has them, else the heuristic. Shares the library DB file.
        from docgen.calibration import CalibrationStore
        cal_store = CalibrationStore(args.db or cfg.db_path)
        desc_in, desc_out = _describe_tokens_per_call(cal_store, model)
        describe_input = int(len(candidates) * desc_in)
        describe_output = int(len(candidates) * desc_out)
        rates = LLM_PRICING.get(model)
        if rates is None:
            describe_cost: float | None = None
            describe_cost_batched: float | None = None
        else:
            ipm, opm = rates
            describe_cost = (
                describe_input * ipm / 1_000_000
                + describe_output * opm / 1_000_000
            )
            # Anthropic's Message Batches API applies a ~50% discount
            # on both input and output. catalog-describe gained batch
            # support, so mirror generate's dual-figure display.
            describe_cost_batched = describe_cost * 0.5

        # generate: walk the source tree and use the existing
        # estimate_cost helper. We don't actually run generate.
        source_path = cfg.resolve_source(source_name)
        from docgen.pricing import estimate_cost
        from docgen.staleness import find_catalog_files
        files = []
        if source_path is not None and source_path.exists():
            for p in find_catalog_files(source_path):
                try:
                    files.append((p, p.stat().st_size))
                except OSError:
                    continue
        if files and rates is not None:
            # Price the SAME doc types the generate phase will actually
            # produce (the default set) — estimating a subset silently
            # under-counts, since file content is sent once per doc type.
            from cli.generate import DEFAULT_GENERATE_DOC_TYPES
            # Caching is an Anthropic feature (and only discounts the
            # static scaffolding). Infer from the model family so this
            # matches `generate --dry-run` for Anthropic models without
            # crashing on a provider/model mismatch during a pure
            # estimate.
            caching_enabled = model.startswith('claude')

            # Calibrated output tokens per (doc_type, language) from real
            # runs, falling back to the per-doc-type phase average, then
            # to the flat heuristic inside estimate_cost.
            # Cache lookups by bucket: the estimate makes several passes
            # over the same files (baseline, batched, and the per-type
            # breakdown), and each (doc_type, language) bucket resolves to
            # the same calibration row every time — so query the store
            # once per distinct bucket instead of once per file per pass.
            _gen_output_cache: dict = {}

            def _gen_output(doc_type, language):
                # Narrow → broad: exact bucket, then per-type (any
                # language), then phase-wide. None → heuristic fallback.
                key = (doc_type, language)
                if key in _gen_output_cache:
                    return _gen_output_cache[key]
                result = None
                for kw in (
                    {'doc_type': doc_type, 'language': language},
                    {'doc_type': doc_type},
                    {},
                ):
                    c = cal_store.mean_tokens(
                        phase='generate', model=model, **kw,
                    )
                    if c is not None:
                        result = c.mean_output
                        break
                _gen_output_cache[key] = result
                return result

            # Two estimates: baseline (no --batch) and with Anthropic's
            # ~50% Message Batches discount applied. ``generate`` is the
            # only phase that batches today.
            generate_estimate = estimate_cost(
                files=files,
                doc_types=DEFAULT_GENERATE_DOC_TYPES,
                model=model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
            )
            generate_estimate_batched = estimate_cost(
                files=files,
                doc_types=DEFAULT_GENERATE_DOC_TYPES,
                model=model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
                batch_enabled=True,
            )
            generate_cost: float | None = generate_estimate.total_cost_usd
            generate_cost_batched: float | None = (
                generate_estimate_batched.total_cost_usd
            )
        else:
            generate_cost = 0.0 if rates is not None else None
            generate_cost_batched = generate_cost

        # themes build: count themes already in the table. If
        # clustering hasn't run, the count is 0 and the cost is 0 —
        # the user can re-invoke after running `themes build` (which
        # has a cheap clustering pass and an LLM summarization pass).
        themes_count, themes_cost, _ = _estimate_themes_cost(
            library, model,
        )

        # ---- Output ---------------------------------------------------
        # NOTE: any_unknown check uses these locals further down — keep
        # describe_cost_batched defined in all branches.
        console.print()
        console.print(
            '[bold]Cost estimate for the remaining LLM-paid phases[/bold]',
        )
        _print_phase(
            'catalog-describe',
            calls=len(candidates),
            unit='elements',
            in_tokens=describe_input,
            out_tokens=describe_output,
            cost=describe_cost,
            batched_cost=describe_cost_batched,
            verbose=verbose,
        )
        _print_phase(
            'generate',
            calls=len(files),
            unit='files',
            in_tokens=(
                generate_estimate.input_tokens if files and rates
                else 0
            ),
            out_tokens=(
                generate_estimate.output_tokens if files and rates
                else 0
            ),
            cost=generate_cost,
            batched_cost=generate_cost_batched,
            verbose=verbose,
        )
        # Per-doc-type breakdown so the user sees what each type costs
        # (and can drop the expensive ones). Only meaningful when we have
        # files + a known rate.
        if files and rates is not None:
            from cli.generate import DEFAULT_GENERATE_DOC_TYPES
            from docgen.pricing import estimate_generate_by_doc_type
            per_type = estimate_generate_by_doc_type(
                files, DEFAULT_GENERATE_DOC_TYPES, model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
            )
            per_type_batched = dict(estimate_generate_by_doc_type(
                files, DEFAULT_GENERATE_DOC_TYPES, model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
                batch_enabled=True,
            ))
            for dt, est in per_type:
                b = per_type_batched[dt]
                console.print(
                    f'      [dim]{dt:<13}'
                    f'${est.total_cost_usd:.2f} / '
                    f'${b.total_cost_usd:.2f} batched[/dim]',
                    soft_wrap=True,
                )
        # themes build: when nothing is clustered yet (first run), the
        # count is 0 — but onboard WILL cluster + summarize, so showing
        # "$0.00" implies it's free. Mark it not-estimated and keep it
        # out of the total (flagged below) instead.
        themes_unestimated = themes_count == 0
        if themes_unestimated:
            console.print(
                '  [cyan]themes build      [/cyan]'
                '[dim]not estimated (no clusters yet)[/dim]',
                soft_wrap=True,
            )
        else:
            _print_phase(
                'themes build',
                calls=themes_count,
                unit='themes',
                in_tokens=themes_count * _THEMES_INPUT_TOKENS_PER_THEME,
                out_tokens=themes_count * _THEMES_OUTPUT_TOKENS_PER_THEME,
                cost=themes_cost,
                verbose=verbose,
            )
        # Total.
        any_unknown = (
            describe_cost is None
            or generate_cost is None
            or (not themes_unestimated and themes_cost is None)
        )
        if any_unknown:
            console.print(
                f'  [yellow]Total: partial — model {model!r} not in '
                'LLM_PRICING for at least one phase.[/yellow]',
            )
        else:
            themes_part = 0.0 if themes_unestimated else themes_cost
            total_baseline = describe_cost + generate_cost + themes_part
            # Both catalog-describe AND generate now support --batch.
            # The batched total applies the discount to both phases.
            total_batched = (
                describe_cost_batched + generate_cost_batched + themes_part
            )
            console.print(
                f'  [bold]Total estimated cost: '
                f'${total_baseline:.2f} / '
                f'${total_batched:.2f} batched[/bold]',
                soft_wrap=True,
            )
            # Uncertainty band + omissions, so the figure reads as an
            # estimate (char-based heuristic), not a quote.
            omits = ['embedding cost']
            if themes_unestimated:
                omits.append('first-run themes summarization')
            console.print(
                f'  [dim]Rough estimate — actual typically within ±50% '
                f'(${total_baseline * 0.5:.2f}–${total_baseline * 1.5:.2f}); '
                f'excludes {", ".join(omits)}.[/dim]',
                soft_wrap=True,
            )
        if themes_unestimated:
            console.print(
                '[dim]Note: run `ariadne themes build` (clustering is '
                'free) and re-run dry-run for the themes estimate.[/dim]',
            )
        return 0
    finally:
        library.close()


def _print_phase(
    name: str, *, calls: int, in_tokens: int, out_tokens: int,
    cost: float | None,
    batched_cost: float | None = None,
    unit: str = 'calls',
    verbose: bool = False,
) -> None:
    cost_str = (
        f'${cost:.2f}' if cost is not None
        else '[yellow]?[/yellow] (model not in LLM_PRICING)'
    )
    if batched_cost is not None and cost is not None:
        cost_str = f'${cost:.2f} / ${batched_cost:.2f} batched'
    if verbose:
        console.print(
            f'  [cyan]{name:<20}[/cyan] {unit}={calls:>5}  '
            f'in={in_tokens:>9,}  out={out_tokens:>7,}  cost={cost_str}',
            soft_wrap=True,
        )
    else:
        # Default mode: phase name + the work size (e.g. how many
        # elements catalog-describe will process — distinct from the
        # file count) + cost, for a clear at-a-glance picture.
        console.print(
            f'  [cyan]{name:<18}[/cyan] {calls:>7,} {unit:<9} {cost_str}',
            soft_wrap=True,
        )


async def cmd_onboard(args: argparse.Namespace) -> int:
    """Onboard a source end-to-end in a SINGLE run.

    Runs the free phases + cost estimate once (the dry-run preview),
    then — with ``--approve`` or an interactive 'yes' — continues
    straight into the paid phases (catalog-describe → generate →
    themes) WITHOUT re-running discover/index/catalog-sync. This is the
    key property: you don't re-index (notably the slow scip-java
    compile) just to proceed past the preview.

    A paid phase returning non-zero stops the pipeline and propagates
    the rc; the paid sub-commands are idempotent, so a stopped run can
    be resumed by re-running ``onboard --approve``.
    """
    import config as _config_module

    cfg = _config_module.get_config()
    source_name = args.source or cfg.default_source
    if source_name is None:
        console.print(
            '[red]No source specified and no default_source in '
            'config.[/red]',
        )
        return 1
    model = args.model or cfg.model

    # ---- Preview: free phases (discover/index/catalog-sync) + cost
    # estimate, run exactly once. ----------------------------------------
    rc = await cmd_dry_run(args)
    if rc != 0:
        return rc

    # ---- Decide whether to run the paid phases ------------------------
    # --approve runs them unconditionally; otherwise prompt (a non-TTY
    # without --approve stops here). Either way the free phases above
    # are NOT re-run.
    if not (getattr(args, 'approve', False) or _prompt_proceed()):
        console.print()
        console.print(
            '[dim]Stopped after the cost preview. To run the paid '
            'phases (without re-indexing):[/dim]',
        )
        console.print(
            f'  [cyan]ariadne onboard --source {source_name} '
            '--approve[/cyan]',
        )
        return 0

    # ---- Paid phases --------------------------------------------------
    # Which doc types to generate (the generate-phase cost driver). An
    # explicit --types wins; otherwise let the user pick from the set
    # whose per-type cost the preview just showed (non-TTY → all).
    from cli.generate import DEFAULT_GENERATE_DOC_TYPES
    explicit_types = getattr(args, 'types', None)
    if explicit_types:
        selected_doc_types = tuple(
            t.strip() for t in explicit_types.split(',') if t.strip()
        )
    else:
        selected_doc_types = _select_generate_doc_types(
            DEFAULT_GENERATE_DOC_TYPES,
        )

    # Resolve batch mode. Explicit flag wins; otherwise prompt.
    batch_mode = getattr(args, 'batch_mode', None)
    if batch_mode is None:
        batch_mode = _prompt_for_batch_mode()
    use_batch = batch_mode == 'batch'

    verbose = getattr(args, 'verbose', False)
    mode_label = 'batched (24h SLA)' if use_batch else 'live'
    console.print(
        f'[bold]ariadne onboard[/bold] · source: {source_name} '
        f'· model: {model} · LLM mode: {mode_label}',
    )

    def _ns(**overrides) -> argparse.Namespace:
        base = argparse.Namespace(**vars(args))
        for k, v in overrides.items():
            setattr(base, k, v)
        return base

    # Concurrency override: a uniform --concurrency N supersedes each
    # phase's baked-in default. Unset → defaults (describe=4, generate=3;
    # catalog-sync=4 is honored inside the preview).
    cc = getattr(args, 'concurrency', None)
    catalog_describe_args = _ns(
        source=source_name, force=False, model=model,
        concurrency=cc if cc is not None else 4,
        max_calls=None,
        dry_run=False, batch=use_batch, resume=False,
        quiet=not verbose,
    )
    # Propagate the batch choice to generate too — without this, generate
    # would silently auto-batch when the prompt count crosses 200 even
    # when the user explicitly asked for live.
    generate_batch_mode = 'always' if use_batch else 'never'
    generate_args = _ns(
        source=source_name, model=model, provider=None,
        api_key=None, types=','.join(selected_doc_types),
        concurrency=cc if cc is not None else 3,
        force=False,
        dry_run=False, verbose=verbose,
        path=None, no_crossrefs=False,
        batch_mode=generate_batch_mode, auto_batch_threshold=200,
        confirm_yes=True,
        quiet=not verbose,
    )
    themes_args = _ns(
        source=source_name, themes_action='build',
        quiet=not verbose,
    )

    ui = _PhaseUI(verbose=verbose)
    phases: list[tuple[str, str, object]] = [
        (
            'Describing catalog elements',
            '  [green]✓[/green] Catalog-describe — descriptions persisted',
            lambda: cmd_catalog_describe(catalog_describe_args),
        ),
        (
            'Generating documentation',
            '  [green]✓[/green] Generate — explanation/architecture/qa docs written',
            lambda: cmd_generate(generate_args),
        ),
        (
            'Building themes',
            '  [green]✓[/green] Themes build — cluster summaries written',
            lambda: cmd_themes_build(themes_args),
        ),
    ]

    import inspect

    for label, success_line, invoke in phases:
        with ui.passthrough(label):
            result = invoke()
            rc = await result if inspect.isawaitable(result) else result
        if rc != 0:
            console.print(
                f'[red]Phase {label!r} failed (rc={rc}). Pipeline '
                'stopped.[/red]',
            )
            return rc
        if not verbose:
            console.print(success_line)

    console.print()
    console.print(
        f'[bold green]✓ Onboarding complete for source: '
        f'{source_name}[/bold green]',
    )
    return 0


def _prompt_proceed() -> bool:
    """Ask whether to continue past the cost preview into the paid
    phases.

    Returns True only on an explicit yes at an interactive prompt. A
    non-interactive context (no TTY) returns False, so a scripted
    ``onboard`` without ``--approve`` stops at the preview rather than
    spending money unattended.
    """
    import sys

    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    try:
        resp = input('Proceed with generation (paid phases)? [y/N]: ')
    except EOFError:
        return False
    return resp.strip().lower() in ('y', 'yes')


_BATCH_MODE_OPTIONS: list[tuple[str, str, str]] = [
    (
        'live',
        'Live',
        'finishes in minutes, full price',
    ),
    (
        'batch',
        'Batch',
        'up to 24h SLA, ~50% off (Anthropic Message Batches API)',
    ),
]


def _prompt_for_batch_mode() -> str:
    """Interactively pick live vs batched catalog-describe.

    Uses arrow-key selection on a TTY (no typing required). Falls back
    to a text prompt when stdin isn't a TTY (CI, piped input). Returns
    ``'live'`` or ``'batch'``.
    """
    import sys

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _arrow_key_select(
                _BATCH_MODE_OPTIONS,
                title='LLM mode (catalog-describe + generate)',
            )
        except Exception:
            # Terminal doesn't support raw mode (e.g., minimal containers).
            # Fall through to the typed prompt rather than crash.
            pass

    console.print()
    console.print(
        '[bold]LLM mode (catalog-describe + generate)[/bold]',
    )
    for _, label, desc in _BATCH_MODE_OPTIONS:
        console.print(f'  [cyan]{label.lower()}[/cyan] — {desc}')
    while True:
        choice = input(
            'Pick [l]ive / [b]atch (default: live): ',
        ).strip().lower()
        if choice in ('', 'l', 'live'):
            return 'live'
        if choice in ('b', 'batch'):
            return 'batch'
        console.print(
            f"[yellow]Didn't understand {choice!r}; please pick "
            "'l' or 'b'.[/yellow]",
        )


def _arrow_key_select(
    options: list[tuple[str, str, str]],
    *,
    title: str,
    initial_index: int = 0,
) -> str:
    """Single-line arrow-key picker over ``options`` (value, label, desc).

    Renders a vertical list with one row highlighted; ↑/↓ (or k/j)
    moves the cursor, Enter confirms, q/Ctrl-C cancels (raises
    ``KeyboardInterrupt``). Returns the chosen option's value string.

    Implementation: termios raw mode for single-key reads, Rich Live
    for flicker-free redraw. POSIX-only — the caller is responsible
    for falling back on non-TTY / non-POSIX environments.
    """
    import sys
    import termios
    import tty
    from rich.live import Live
    from rich.text import Text

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    idx = initial_index

    def _render() -> Text:
        body = Text()
        body.append(title, style='bold')
        body.append('\n')
        body.append('  (↑/↓ to move · Enter to select · q to cancel)\n',
                    style='dim')
        for i, (_, label, desc) in enumerate(options):
            cursor = '▶ ' if i == idx else '  '
            row = Text(cursor + label, style='cyan' if i == idx else '')
            row.append(f'  — {desc}', style='dim')
            body.append(row)
            body.append('\n')
        return body

    try:
        # cbreak (not raw) — gives us single-char reads but keeps the
        # terminal's line discipline. Under raw mode, ``\n`` doesn't
        # carriage-return, so Rich's multi-line render staircases to
        # the right; cbreak preserves CR-LF translation so each row
        # starts at column 0.
        tty.setcbreak(fd)
        with Live(
            _render(), console=console, refresh_per_second=30,
            transient=True,
        ) as live:
            while True:
                ch = sys.stdin.read(1)
                if ch == '\x1b':  # escape sequence (arrow keys)
                    seq = sys.stdin.read(2)
                    if seq == '[A':  # up
                        idx = (idx - 1) % len(options)
                    elif seq == '[B':  # down
                        idx = (idx + 1) % len(options)
                elif ch in ('k', 'K'):
                    idx = (idx - 1) % len(options)
                elif ch in ('j', 'J'):
                    idx = (idx + 1) % len(options)
                elif ch in ('\r', '\n'):
                    break
                elif ch in ('q', 'Q', '\x03'):  # q / Ctrl-C
                    raise KeyboardInterrupt
                live.update(_render())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    chosen = options[idx]
    console.print(
        f'[bold]{title}[/bold]: [cyan]{chosen[1].lower()}[/cyan] '
        f'[dim]— {chosen[2]}[/dim]',
    )
    return chosen[0]


def _arrow_key_multiselect(
    options: list[tuple[str, str]],
    *,
    title: str,
    selected: set[int],
) -> list[str]:
    """Checklist picker over ``options`` (value, label).

    ↑/↓ (or k/j) moves, Space toggles, Enter confirms, q/Ctrl-C cancels
    (raises ``KeyboardInterrupt``). ``selected`` is the set of initially
    checked indices. Returns the chosen values in option order. POSIX
    only — the caller falls back on non-TTY / non-POSIX.
    """
    import sys
    import termios
    import tty
    from rich.live import Live
    from rich.text import Text

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    idx = 0

    def _render() -> Text:
        body = Text()
        body.append(title, style='bold')
        body.append('\n')
        body.append(
            '  (↑/↓ move · Space toggle · Enter confirm · q cancel)\n',
            style='dim',
        )
        for i, (_, label) in enumerate(options):
            cursor = '▶ ' if i == idx else '  '
            box = '[x] ' if i in selected else '[ ] '
            body.append(
                Text(cursor + box + label, style='cyan' if i == idx else ''),
            )
            body.append('\n')
        return body

    try:
        tty.setcbreak(fd)
        with Live(
            _render(), console=console, refresh_per_second=30,
            transient=True,
        ) as live:
            while True:
                ch = sys.stdin.read(1)
                if ch == '\x1b':
                    seq = sys.stdin.read(2)
                    if seq == '[A':
                        idx = (idx - 1) % len(options)
                    elif seq == '[B':
                        idx = (idx + 1) % len(options)
                elif ch in ('k', 'K'):
                    idx = (idx - 1) % len(options)
                elif ch in ('j', 'J'):
                    idx = (idx + 1) % len(options)
                elif ch == ' ':
                    selected.discard(idx) if idx in selected else selected.add(idx)
                elif ch in ('\r', '\n'):
                    break
                elif ch in ('q', 'Q', '\x03'):
                    raise KeyboardInterrupt
                live.update(_render())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)

    return [v for i, (v, _) in enumerate(options) if i in selected]


def _select_generate_doc_types(default_types: tuple[str, ...]) -> tuple[str, ...]:
    """Let the user pick which doc types ``generate`` should produce.

    All types start checked. A non-interactive context (no TTY) or a
    cancelled/empty selection returns all ``default_types`` — so we never
    silently generate nothing.
    """
    import sys

    options = [(t, t) for t in default_types]
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return tuple(default_types)
    try:
        chosen = _arrow_key_multiselect(
            options,
            title='Doc types to generate (per-type cost shown above)',
            selected=set(range(len(options))),
        )
    except Exception:
        return tuple(default_types)
    return tuple(chosen) if chosen else tuple(default_types)


async def cmd_catalog_describe(args: argparse.Namespace) -> int:
    """Generate LLM descriptions for catalog elements of a source."""
    from docgen.catalog_describer import (
        describe_source_elements,
        describe_source_elements_batched,
    )
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

    library = get_library(args.db)
    try:
        if getattr(args, 'dry_run', False):
            if getattr(args, 'batch', False) or getattr(args, 'resume', False):
                console.print(
                    '[yellow]Note: --batch / --resume are ignored when '
                    '--dry-run is set. Use `ariadne dry-run` to see both '
                    'baseline and batched cost figures.[/yellow]',
                )
            return _print_catalog_describe_cost_estimate(
                library, source_name,
                model=args.model or cfg.model,
                force=args.force,
                max_calls=args.max_calls,
            )

        # Route to batched path when --batch OR --resume is set.
        # --batch submits a new batch; --resume fetches results from
        # an existing pending batch (same source+model) without
        # re-submitting.
        if getattr(args, 'batch', False) or getattr(args, 'resume', False):
            import os

            from cli.generate import resolve_provider
            from docgen.llm.factory import make_batch_strategy

            model = args.model or cfg.model
            # Batch dispatch picks the resolved provider's batch backend —
            # Anthropic's Message Batches or OpenAI's Batch API, both 24h /
            # ~50% off — so a gpt-* model batches via OpenAI instead of
            # crashing on a hardcoded AnthropicProvider. The matching API key
            # is required (make_batch_strategy raises for a provider with no
            # batch backend).
            batch_provider = resolve_provider(
                cli_provider=getattr(args, 'provider', None),
                cfg_provider=getattr(cfg, 'provider', None),
                model=model,
            )
            key_env = (
                'OPENAI_API_KEY' if batch_provider == 'openai'
                else 'ANTHROPIC_API_KEY'
            )
            api_key = os.environ.get(key_env, '')
            if not api_key:
                console.print(
                    f'[red]{key_env} is not set. Export it before using '
                    f'--batch or --resume with the {batch_provider} '
                    f'provider.[/red]',
                )
                return 1
            provider = make_batch_strategy(
                batch_provider, model=model, api_key=api_key,
            )
            provider_label = {
                'openai': 'OpenAI', 'anthropic': 'Anthropic',
            }.get(batch_provider, batch_provider)
            # Rich Progress widget driven by the batch poll callback.
            # Total isn't known until submit returns; show a spinner
            # until the first poll fires, then switch to a determinate
            # bar. transient=True so the bar self-clears when invoked
            # from a higher-level wrapper (e.g., onboard).
            from rich.progress import (
                BarColumn, MofNCompleteColumn, Progress,
                SpinnerColumn, TextColumn, TimeElapsedColumn,
            )
            quiet = getattr(args, 'quiet', False)
            progress = Progress(
                SpinnerColumn(),
                TextColumn('[cyan]{task.description}[/cyan]'),
                BarColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                transient=quiet,
            )
            task_id = progress.add_task(
                'Preparing batch…', total=None,
            )

            def _batched_progress(processing: int, succeeded: int,
                                  errored: int) -> None:
                # Processing-stage detail: the provider's per-poll counts.
                total = processing + succeeded + errored
                done = succeeded + errored
                if total == 0:
                    desc = (
                        f'Processing at {provider_label} — awaiting first poll'
                    )
                else:
                    desc = (
                        f'Processing at {provider_label} (processing '
                        f'{processing}, succeeded {succeeded}, '
                        f'errored {errored})'
                    )
                progress.update(
                    task_id,
                    description=desc,
                    completed=done,
                    total=total if total > 0 else None,
                )

            # Stage transitions for the rest of the batch lifecycle, so
            # the bar shows submit → process → download → apply rather
            # than freezing after processing while results are fetched
            # and tens of thousands of docs are re-embedded.
            _stage_labels = {
                'submit': f'Submitting requests to {provider_label}',
                'processing': f'Processing at {provider_label}',
                'download': 'Downloading results',
                'apply': 'Applying descriptions + re-embedding',
            }

            def _on_stage(stage: str, completed: int,
                          total: 'int | None') -> None:
                label = _stage_labels.get(stage, stage)
                if total:
                    desc = f'{label} ({completed:,}/{total:,})'
                else:
                    desc = label
                progress.update(
                    task_id, description=desc,
                    completed=completed, total=total,
                )

            from docgen.calibration import (
                CalibrationStore, set_usage_observer,
            )
            _cal_store = CalibrationStore(args.db or cfg.db_path)
            with progress, set_usage_observer(_cal_store.record):
                async with LibraryWriter(library) as writer:
                    result = await describe_source_elements_batched(
                        library, writer, source_name,
                        provider=provider,
                        model=args.model or cfg.model,
                        force=args.force,
                        resume=getattr(args, 'resume', False),
                        max_calls=args.max_calls,
                        concurrency=getattr(args, 'concurrency', 4) or 4,
                        on_progress=_batched_progress,
                        on_stage=_on_stage,
                    )
            if not quiet:
                console.print()
                mode = 'resumed' if result.get('resumed') else 'batched'
                console.print(
                    f'Catalog describe ({mode}) for [cyan]{source_name}[/cyan]:',
                )
                if 'note' in result:
                    console.print(f'  [yellow]{result["note"]}[/yellow]')
                console.print(f'  Submitted: {result.get("submitted", 0)}')
                console.print(
                    f'  [green]Described: {result.get("described", 0)}[/green]',
                )
                if result.get('skipped_already_applied'):
                    console.print(
                        f'  [dim]Already applied (from prior partial run): '
                        f'{result["skipped_already_applied"]}[/dim]',
                    )
                if result.get('failed'):
                    console.print(
                        f'  [red]Failed (Anthropic per-row error): '
                        f'{result["failed"]}[/red]',
                    )
                if result.get('batch_id') and not result.get('resumed'):
                    console.print(
                        f'  [dim]batch_id: {result["batch_id"]} (use '
                        '--resume if you need to re-fetch)[/dim]',
                    )
            return 0

        from rich.progress import (
            BarColumn,
            MofNCompleteColumn,
            Progress,
            SpinnerColumn,
            TextColumn,
            TimeElapsedColumn,
            TimeRemainingColumn,
        )

        progress_columns = (
            SpinnerColumn(),
            TextColumn('[bold cyan]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn('·'),
            TimeElapsedColumn(),
            TextColumn('eta'),
            TimeRemainingColumn(),
        )

        quiet = getattr(args, 'quiet', False)
        with Progress(
            *progress_columns, console=console, transient=quiet,
        ) as progress:
            task_id = progress.add_task(
                f'Describing catalog for {source_name}', total=0,
            )

            def _on_progress(
                described: int, failed: int, total: int,
                first_failure: str | None = None,
            ) -> None:
                # Single live bar — advance by completed (ok + failed) and
                # surface the running failure count in the description so a
                # non-zero count is visible without scrolling. When the
                # first failure happens, append a short reason so the
                # user can diagnose (auth, network, model) without
                # waiting for the full run to finish.
                desc = f'Describing {source_name}'
                if failed:
                    desc += f' [red]({failed} failed)[/red]'
                    if first_failure:
                        desc += f' [dim]· {first_failure}[/dim]'
                progress.update(
                    task_id, completed=described + failed, total=total,
                    description=desc,
                )

            from docgen.calibration import (
                CalibrationStore, set_usage_observer,
            )
            _cal_store = CalibrationStore(args.db or cfg.db_path)
            with set_usage_observer(_cal_store.record):
                async with LibraryWriter(library) as writer:
                    result = await describe_source_elements(
                        library, writer, source_name,
                        force=args.force,
                        concurrency=args.concurrency,
                        model=args.model,
                        max_calls=args.max_calls,
                        on_progress=_on_progress,
                    )

        if not quiet:
            console.print()
            console.print(
                f'Catalog describe for [cyan]{source_name}[/cyan]:',
            )
            console.print(f'  Candidates: {result["total_candidates"]}')
            console.print(
                f'  [green]Described: {result["described"]}[/green]',
            )
            console.print(
                f"  Already had description: "
                f"{result['already_had_description']}",
            )
            if result['failed']:
                console.print(f'  [red]Failed: {result["failed"]}[/red]')
                # Break the failure count down by category so the user
                # can act on it (re-run for empty_response, check
                # network for network, escalate for other).
                reasons = result.get('failure_reasons') or []
                category_counts: dict[str, int] = {}
                for _, category, _ in reasons:
                    category_counts[category] = (
                        category_counts.get(category, 0) + 1
                    )
                if category_counts:
                    parts = ', '.join(
                        f'{cat}={n}'
                        for cat, n in sorted(category_counts.items())
                    )
                    console.print(f'    [dim]by reason: {parts}[/dim]')
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
            )

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




def cmd_symbol(args: argparse.Namespace) -> int:
    """Look up a catalog element; prints JSON to stdout."""
    import json

    from docgen.catalog_lookup import lookup_symbol

    cfg = get_config()
    source_name = args.source or cfg.default_source
    if source_name is None:
        print(json.dumps({'error': 'no_source'}))
        return 1

    library = get_library(args.db)
    try:
        result = lookup_symbol(library, source_name, args.file, args.name)
        print(json.dumps(result))
        return 0
    finally:
        library.close()



def cmd_list_file_scopes(args: argparse.Namespace) -> int:
    """list-file-scopes: list qualified_names in a file."""
    import json

    from docgen.catalog_lookup import list_elements_in_file

    cfg = get_config()
    source_name = args.source or cfg.default_source
    if source_name is None:
        print(json.dumps({'error': 'no_source'}))
        return 1

    library = get_library(getattr(args, 'db', None))
    try:
        result = list_elements_in_file(library, source_name, args.file)
        print(json.dumps(result, indent=2))
        return 0
    finally:
        library.close()


def cmd_body(args: argparse.Namespace) -> int:
    """Return current body text of a catalog element; prints JSON."""
    import json

    from docgen.catalog_lookup import get_element_body

    cfg = get_config()
    source_name = args.source or cfg.default_source
    if source_name is None:
        print(json.dumps({'error': 'no_source'}))
        return 1

    library = get_library(args.db)
    try:
        result = get_element_body(library, source_name, args.file, args.name)
        print(json.dumps(result))
        return 0
    finally:
        library.close()


async def cmd_themes_build(args: argparse.Namespace) -> int:
    """`ariadne themes build` — run refresh_themes for the configured source.

    This is the awaitable core: it ``await``s ``refresh_themes`` directly
    rather than spinning its own event loop, so it composes into a caller
    that is already inside one (the async ``onboard`` pipeline awaits this
    as a phase). The standalone CLI dispatch goes through the sync
    ``_cmd_themes_build`` wrapper below, which supplies the event loop.
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from docgen.themes import refresh_themes
    from writer import LibraryWriter

    cfg = get_config()

    library = get_library(getattr(args, 'db', None))
    try:
        progress_columns = (
            SpinnerColumn(),
            TextColumn('[bold cyan]{task.description}'),
            BarColumn(),
            MofNCompleteColumn(),
            TextColumn('·'),
            TimeElapsedColumn(),
            TextColumn('eta'),
            TimeRemainingColumn(),
        )

        async def _build() -> dict:
            with Progress(*progress_columns, console=console) as progress:
                task_id = progress.add_task(
                    'Themes: clustering & summarizing', total=0,
                )

                def on_progress(
                    completed: int, total: int, cluster_id: str | None,
                ) -> None:
                    short = (cluster_id or '')[:14]
                    desc = (
                        f'Themes: summarizing [dim]{short}[/dim]'
                        if short else 'Themes: done'
                    )
                    progress.update(
                        task_id,
                        completed=completed,
                        total=total if total > 0 else None,
                        description=desc,
                    )

                async with LibraryWriter(library) as writer:
                    return await refresh_themes(
                        library, writer,
                        enabled=getattr(cfg, 'themes_enabled', True),
                        summarize_kwargs={'on_progress': on_progress},
                    )

        try:
            summary = await _build()
        except Exception as e:
            console.print(f'[red]Themes build failed: {e}[/red]')
            return 1

        path = summary.get('path', '?')
        console.print()
        console.print(f'Themes refresh: [cyan]{path}[/cyan]')
        console.print(f'  Changed:    {summary.get("changed", 0)}')
        console.print(f'  Summarized: [green]{summary.get("summarized", 0)}[/green]')
        console.print(f'  Incoherent: {summary.get("incoherent", 0)}')
        if summary.get('failed', 0):
            console.print(f'  [red]Failed: {summary["failed"]}[/red]')
        return 0
    finally:
        library.close()


def _cmd_themes_build(args: argparse.Namespace) -> int:
    """Synchronous wrapper for the standalone ``ariadne themes build``
    dispatch — supplies the event loop that :func:`cmd_themes_build`
    expects a caller to provide."""
    return asyncio.run(cmd_themes_build(args))


def _cmd_themes_list(args: argparse.Namespace) -> int:
    """`ariadne themes list` — print a table of discovered themes."""
    from ariadne_mcp.service_themes import themes_action

    library = get_library(getattr(args, 'db', None))
    try:
        coherent_only = not getattr(args, 'include_incoherent', False)
        result = themes_action(
            library,
            action='list',
            coherent_only=coherent_only,
            source=getattr(args, 'source', None),
            limit=getattr(args, 'limit', 50),
        )

        themes = result.get('themes', [])
        total = result.get('total', len(themes))
        if not themes:
            console.print('[yellow]No themes found.[/yellow] '
                          'Run [bold]ariadne themes build[/bold] to discover them.')
            return 0

        table = Table(title=f'Themes ({len(themes)} of {total})')
        table.add_column('cluster_id', style='cyan', no_wrap=True)
        table.add_column('title', style='bold')
        table.add_column('members', justify='right')
        table.add_column('coherent', justify='center')
        table.add_column('dirty', justify='center')

        for t in themes:
            table.add_row(
                t.get('cluster_id', '')[:14],
                str(t.get('title', '') or ''),
                str(t.get('member_count', 0)),
                'yes' if t.get('coherent') else 'no',
                'yes' if t.get('dirty') else 'no',
            )

        console.print(table)
        return 0
    finally:
        library.close()


def _cmd_themes_show(args: argparse.Namespace) -> int:
    """`ariadne themes show <cluster_id>` — print the full theme doc."""
    from ariadne_mcp.service_themes import themes_action

    cluster_id = getattr(args, 'cluster_id', None)
    if not cluster_id:
        console.print('[red]Missing cluster_id.[/red]')
        return 1

    library = get_library(getattr(args, 'db', None))
    try:
        result = themes_action(
            library, action='get', cluster_id=cluster_id,
        )
        if 'error' in result:
            console.print(f'[red]{result["error"]}[/red]')
            return 1

        title = result.get('title') or f'Theme {cluster_id[:12]}'
        content = result.get('content') or ''
        console.print(f'[bold cyan]{title}[/bold cyan] '
                      f'(cluster_id={cluster_id}, members={result.get("member_count", 0)})')
        console.print()
        console.print(content)
        return 0
    finally:
        library.close()


def cmd_diff_docs(args: argparse.Namespace) -> int:
    """Side-by-side comparison of legacy and catalog-driven prompts.

    Runs both pipelines (mocked LLM unless --with-llm) and prints the
    user prompts each path would have sent to the LLM. Useful for
    spot-checking parity during the dual-run window.
    """
    from unittest.mock import AsyncMock, patch

    from docgen._legacy_analyzer import SourceAnalyzer
    from docgen.catalog_enrich import enrich_file
    from docgen.generator import DocGenerator, GeneratorConfig

    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f'[red]File not found: {file_path}[/red]')
        return 1
    if file_path.suffix != '.py':
        console.print(
            f'[red]diff-docs only supports Python files (legacy path is '
            f'Python-only); got: {file_path.suffix}[/red]'
        )
        return 1

    doc_type = getattr(args, 'type', 'explanation')
    with_llm = getattr(args, 'with_llm', False)

    cfg = get_config()

    async def _run() -> tuple[str, str, str | None, str | None]:
        legacy_user: str = ''
        new_user: str = ''
        legacy_output: str | None = None
        new_output: str | None = None

        # Capture prompts (always) and optionally let the LLM run.
        async def fake_llm_legacy(system, user):
            nonlocal legacy_user
            legacy_user = user

        async def fake_llm_new(system, user):
            nonlocal new_user
            new_user = user

        # Legacy
        analyzer = SourceAnalyzer()
        try:
            metadata = analyzer.analyze_file(file_path)
        except SyntaxError as e:
            console.print(f'[red]Legacy path SyntaxError: {e}[/red]')
            return ('', '', None, None)

        legacy_gen = DocGenerator(
            config=GeneratorConfig(
                model=cfg.model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            ),
            analyzer=analyzer,
        )
        if with_llm:
            async with legacy_gen:
                docs = await legacy_gen.generate_for_module(
                    metadata, doc_types=(doc_type,),
                )
                legacy_output = docs[0].content if docs else None
                # Run again with mocked LLM just to capture the prompt.
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_legacy,
            ):
                legacy_gen2 = DocGenerator(analyzer=analyzer)
                async with legacy_gen2:
                    await legacy_gen2.generate_for_module(
                        metadata, doc_types=(doc_type,),
                    )
        else:
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_legacy,
            ):
                legacy_gen2 = DocGenerator(analyzer=analyzer)
                async with legacy_gen2:
                    await legacy_gen2.generate_for_module(
                        metadata, doc_types=(doc_type,),
                    )

        # New (catalog)
        bundle = enrich_file(file_path, source_root=file_path.parent)
        if bundle is None:
            console.print('[red]Catalog path could not build bundle.[/red]')
            return (legacy_user, '', legacy_output, None)

        new_gen = DocGenerator(
            config=GeneratorConfig(
                model=cfg.model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            ),
        )
        if with_llm:
            async with new_gen:
                docs = await new_gen.generate_from_elements(
                    bundle, doc_types=(doc_type,),
                )
                new_output = docs[0].content if docs else None
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_new,
            ):
                new_gen2 = DocGenerator()
                async with new_gen2:
                    await new_gen2.generate_from_elements(
                        bundle, doc_types=(doc_type,),
                    )
        else:
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_new,
            ):
                new_gen2 = DocGenerator()
                async with new_gen2:
                    await new_gen2.generate_from_elements(
                        bundle, doc_types=(doc_type,),
                    )

        return (legacy_user, new_user, legacy_output, new_output)

    try:
        legacy_user, new_user, legacy_out, new_out = asyncio.run(_run())
    except Exception as e:
        console.print(f'[red]diff-docs failed: {e}[/red]')
        return 1

    console.print(f'[bold]diff-docs[/bold] [cyan]{file_path}[/cyan] '
                  f'(doc_type={doc_type})')
    console.print()
    console.print('[bold yellow]== Legacy prompt ==[/bold yellow]')
    console.print(legacy_user or '[dim](empty)[/dim]')
    console.print()
    console.print('[bold green]== Catalog (new) prompt ==[/bold green]')
    console.print(new_user or '[dim](empty)[/dim]')

    if with_llm:
        console.print()
        console.print('[bold yellow]== Legacy LLM output ==[/bold yellow]')
        console.print(legacy_out or '[dim](no output)[/dim]')
        console.print()
        console.print('[bold green]== Catalog LLM output ==[/bold green]')
        console.print(new_out or '[dim](no output)[/dim]')

    return 0


def cmd_themes(args: argparse.Namespace) -> int:
    """Dispatcher for `ariadne themes <action>`."""
    action = getattr(args, 'themes_action', None)
    if action == 'build':
        return _cmd_themes_build(args)
    if action == 'list':
        return _cmd_themes_list(args)
    if action == 'show':
        return _cmd_themes_show(args)
    console.print('[yellow]Usage: ariadne themes (build|list|show <cluster_id>)[/yellow]')
    return 1


HANDLERS = {
    'generate': lambda args: asyncio.run(cmd_generate(args)),
    'check': lambda args: asyncio.run(cmd_check(args)),
    'sync': lambda args: asyncio.run(cmd_sync(args)),
    'merge': lambda args: asyncio.run(cmd_merge(args)),
    'cleanup': lambda args: cmd_cleanup(args),
    'migrate': lambda args: cmd_migrate(args),
    'improve': lambda args: asyncio.run(cmd_improve(args)),
    'topic': lambda args: asyncio.run(cmd_topic(args)),
    'docs': lambda args: asyncio.run(cmd_docs(args)),
    'catalog-sync': lambda args: asyncio.run(cmd_catalog_sync(args)),
    'catalog-describe': lambda args: asyncio.run(cmd_catalog_describe(args)),
    'dry-run': lambda args: asyncio.run(cmd_dry_run(args)),
    'onboard': lambda args: asyncio.run(cmd_onboard(args)),
    'notify-changed': lambda args: asyncio.run(cmd_notify_changed(args)),
    'symbol': lambda args: cmd_symbol(args),
    'list-file-scopes': lambda args: cmd_list_file_scopes(args),
    'body': lambda args: cmd_body(args),
    'themes': lambda args: cmd_themes(args),
    'diff-docs': lambda args: cmd_diff_docs(args),
    'batch': lambda args: __import__('cli.batch').cmd_batch(args),
}
