"""Structural code-catalog CLI commands (``catalog-sync``, ``catalog-describe``).

Extracted from cli/generation.py. ``catalog-sync`` extracts per-element
catalog docs (ast-grep/SCIP); ``catalog-describe`` generates LLM descriptions
for them (with a ``--dry-run`` cost estimate). Wired into the parser via this
module's ``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

from schema import CATALOG_KIND_ELEMENT

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from config import get_config

if TYPE_CHECKING:
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


_DESCRIBE_INPUT_TOKENS_PER_CALL = 200
_DESCRIBE_OUTPUT_TOKENS_PER_CALL = 60


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register structural code-catalog commands."""

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
        from cli.progress import make_progress

        with make_progress(
            console=console, transient=getattr(args, 'quiet', False),
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


def _print_catalog_describe_cost_estimate(
    library: "Library",
    source_name: str,
    *,
    model: str,
    force: bool,
    max_calls: int | None,
    batch: bool = False,
) -> int:
    """Count candidates, multiply by per-call token estimates × model
    rate, print a cost preview, and exit. No LLM calls made.

    ``batch`` prices the run at the Message Batches discount so the
    figure matches what a ``--batch`` run would actually cost; the
    live figure stays visible for comparison either way.
    """
    from docgen.pricing import _BATCH_DISCOUNT, LLM_PRICING

    # Same candidate selection as describe_source_elements so the
    # estimate matches what the real run would actually call against.
    all_catalog = library.list_documents(
        content_type='catalog', limit=100_000,
    )
    candidates = [
        d for d in all_catalog
        if d.metadata.get('source_name') == source_name
        and d.metadata.get('kind') == CATALOG_KIND_ELEMENT
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
        batched_usd = cost_usd * _BATCH_DISCOUNT
        off_pct = round((1 - _BATCH_DISCOUNT) * 100)
        if batch:
            cost_line = (
                f'  Estimated cost: [bold]${batched_usd:.2f}[/bold] '
                f'(batch, {off_pct}% off live ${cost_usd:.2f})'
            )
        else:
            cost_line = (
                f'  Estimated cost: [bold]${cost_usd:.2f}[/bold] '
                f'[dim](${batched_usd:.2f} with --batch)[/dim]'
            )

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
            if getattr(args, 'resume', False):
                console.print(
                    '[yellow]Note: --resume is ignored when --dry-run is '
                    'set — resuming fetches an already-submitted batch, so '
                    'there is nothing left to price.[/yellow]',
                )
            return _print_catalog_describe_cost_estimate(
                library, source_name,
                model=args.model or cfg.model,
                force=args.force,
                max_calls=args.max_calls,
                batch=getattr(args, 'batch', False),
            )

        # Route to batched path when --batch OR --resume is set.
        # --batch submits a new batch; --resume fetches results from
        # an existing pending batch (same source+model) without
        # re-submitting.
        if getattr(args, 'batch', False) or getattr(args, 'resume', False):
            from cli.generate import resolve_batch_strategy

            strategy, batch_provider, key_env = resolve_batch_strategy(
                args, cfg,
            )
            if strategy is None:
                console.print(
                    f'[red]{key_env} is not set. Export it before using '
                    f'--batch or --resume with the {batch_provider} '
                    f'provider.[/red]',
                )
                return 1
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
                        strategy=strategy,
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

        from cli.progress import make_progress

        quiet = getattr(args, 'quiet', False)
        with make_progress(console=console, transient=quiet) as progress:
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


HANDLERS = {
    'catalog-sync': lambda args: asyncio.run(cmd_catalog_sync(args)),
    'catalog-describe': lambda args: asyncio.run(cmd_catalog_describe(args)),
}
