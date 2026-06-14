"""``ariadne generate`` command — LLM-driven doc generation pipeline.

Owns:
- Argparse registration for the ``generate`` subcommand.
- Provider resolution (model → provider inference + mismatch detection).
- Dry-run cost estimation and printing.
- Dependency auto-detection prompt.
- The ``cmd_generate`` async entry point itself.
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from config import get_config
from cli.generate_cost import (  # noqa: E402
    _check_and_prompt_dependencies,
    _commit_scope,
    _library_for_estimate,
    _print_cost_estimate,
)

# LLM-driven doc types generated when --types is omitted. The dry-run
# cost estimate MUST price this exact set — pricing a subset silently
# under-estimates the generate phase (file content is sent once per
# doc type). ``catalog`` is excluded: it's structural (ast-grep), not
# an LLM call.
DEFAULT_GENERATE_DOC_TYPES = (
    'explanation', 'architecture', 'qa', 'gotcha', 'diagram',
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from rich.progress import Progress


console = Console()


# ---------------------------------------------------------------------------
# Provider resolution
# ---------------------------------------------------------------------------


def infer_provider_from_model(model: str) -> str | None:
    """Map a model name to its hosting provider.

    Returns "openai" for ``gpt-*``, "anthropic" for ``claude-*``, and
    ``None`` for unknown patterns (custom proxies, fine-tunes, future
    models). Callers default to "openai" when None comes back.
    """
    if model.startswith('gpt-'):
        return 'openai'
    if model.startswith('claude-'):
        return 'anthropic'
    return None


def resolve_provider(
    *,
    cli_provider: str | None,
    cfg_provider: str | None,
    model: str,
) -> str:
    """Resolve the LLM provider from CLI flag, config, and model name.

    Resolution order:
      1. ``--provider`` CLI flag wins.
      2. Then ``provider:`` from ``ariadne.yaml``.
      3. Then inferred from model family (``gpt-*`` → openai,
         ``claude-*`` → anthropic).
      4. Default to "openai" for unknown model patterns.

    If both an explicit provider AND an inferable model family are set
    and they disagree (e.g. ``provider: anthropic`` with
    ``model: gpt-5.4``), raises ``ValueError`` so the run fails fast
    instead of hitting a confusing 404 from the wrong endpoint.
    """
    inferred = infer_provider_from_model(model)
    explicit = cli_provider or cfg_provider

    if explicit and inferred and explicit != inferred:
        raise ValueError(
            f'Provider mismatch: model {model!r} is hosted by {inferred!r}, '
            f'but provider was set to {explicit!r}. Either change the model '
            f'to a {explicit!r}-hosted one, or update provider to {inferred!r}.'
        )

    return explicit or inferred or 'openai'


# ---------------------------------------------------------------------------
# Batch dispatch feature flag — apply post-resolution
# ---------------------------------------------------------------------------


def _resolve_batch_dispatch(
    *, batch_resolved: bool, batch_reason: str,
) -> tuple[bool, str]:
    """Backwards-compat wrapper around ``apply_dispatch_gate``.

    The real gate logic lives in ``docgen.batch_resolution`` so the
    orchestrator can consult the same downgrade rule. This wrapper
    preserves the kwargs-only signature that
    ``tests/test_batch_dispatch_flag.py`` and any external callers
    depend on.
    """
    from docgen.batch_resolution import apply_dispatch_gate

    return apply_dispatch_gate(batch_resolved, batch_reason)


# ---------------------------------------------------------------------------
# Confirm callbacks for batch first-run UX (#45.7)
# ---------------------------------------------------------------------------
#
# ``cli_confirm`` is the default — prompts via stdin so the user sees
# the 24h SLA disclosure on their first batch dispatch. ``yes_confirm``
# is the ``--yes`` short-circuit. Both are async because the
# orchestrator's ``run()`` is async; the default uses
# ``asyncio.to_thread(input, ...)`` so the blocking ``input()`` call
# doesn't stall the event loop.
#
# STUBS pending #45.7 impl. Real impls land in this same file.


async def yes_confirm(msg: str) -> bool:
    """Always-yes confirm callback for the ``--yes`` flag.

    No stdin read — caller has explicitly bypassed the confirmation
    prompt. ``msg`` is accepted for signature parity with
    :func:`cli_confirm` but ignored.
    """
    return True


async def cli_confirm(msg: str) -> bool:
    """Default confirm callback — prompts via stdin.

    Wraps the blocking ``input()`` in :func:`asyncio.to_thread` so
    the orchestrator's event loop isn't stalled while the user types.

    Default response is N (capital in the ``[y/N]`` suffix), so an
    empty enter declines. This is a deliberate safety net: users who
    hit enter by reflex shouldn't end up paying for a 24h batch run
    they didn't intend.
    """
    import asyncio

    response = await asyncio.to_thread(input, f'{msg} [y/N]: ')
    return response.strip().lower() in ('y', 'yes')


@asynccontextmanager
async def _progress_heartbeat(
    progress: 'Progress', interval: float = 0.5,
) -> 'AsyncIterator[None]':
    """Refresh the progress display on a fixed cadence for the duration of
    a long async phase.

    rich's auto-refresh doesn't reliably tick during batch's long awaits —
    the ~30s status polls and the single long results download — so the
    spinner and elapsed timer freeze between the sparse explicit updates.
    Ticking ``progress.refresh()`` ourselves keeps them moving.
    """
    import asyncio

    async def _tick() -> None:
        try:
            while True:
                await asyncio.sleep(interval)
                progress.refresh()
        except asyncio.CancelledError:
            pass

    task = asyncio.create_task(_tick())
    try:
        yield
    finally:
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass


def _generate_exit_code(result: object) -> int:
    """Exit code for a finished generation run.

    ``0`` = success or nothing-to-do. A *partial* failure (some docs
    generated, some failed) is a warning, not an error — so an ``onboard``
    pipeline keeps going to its later phases rather than halting on a
    single validation hiccup that batch mode doesn't retry. ``1`` is a
    HARD failure only: the run aborted, or every attempted doc failed.
    """
    if getattr(result, 'aborted', False):
        return 1
    if result.docs_created == 0 and result.docs_failed > 0:
        return 1
    return 0


# ---------------------------------------------------------------------------
# Argparse registration
# ---------------------------------------------------------------------------


def register_generate_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register the ``generate`` subcommand and its flags."""
    generate_parser = subparsers.add_parser(
        'generate', help='Generate documentation from source code',
    )
    generate_parser.add_argument(
        '--source', '-s', default=None,
        help='Source path or name (default from config)',
    )
    generate_parser.add_argument(
        '--model', '-m', default=None,
        help='LLM model to use (default from config: gpt-5.2)',
    )
    generate_parser.add_argument(
        '--provider', default=None,
        choices=['openai', 'anthropic'],
        help='LLM backend (default: openai). Anthropic uses '
             'ANTHROPIC_API_KEY; OpenAI uses OPENAI_API_KEY.',
    )
    generate_parser.add_argument(
        '--api-key',
        help='API key (or set OPENAI_API_KEY / ANTHROPIC_API_KEY env var)',
    )
    generate_parser.add_argument(
        '--types', default=None,
        help=(
            'Comma-separated doc types to generate. Defaults to ALL '
            'supported types (explanation,architecture,qa,gotcha,diagram). '
            'Pass a subset to scope the run, e.g. --types explanation.'
        ),
    )
    generate_parser.add_argument(
        '--concurrency', '-c', type=int, default=3,
        help='Max concurrent LLM requests (default: 3)',
    )
    generate_parser.add_argument(
        '--force', '-f', action='store_true',
        help='Regenerate even if up-to-date',
    )
    generate_parser.add_argument(
        '--dry-run', action='store_true',
        help='Preview file count and estimated cost without LLM calls',
    )
    generate_parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show detailed validation reports for failures',
    )
    generate_parser.add_argument(
        '--path', '-p', default=None,
        help='Subdirectory to generate docs for (relative to source root)',
    )
    generate_parser.add_argument(
        '--no-crossrefs', action='store_true',
        help='Skip cross-reference injection (post-processing).',
    )
    # Batch dispatch (Phase C / #45) live since #45.9. ``--batch always``
    # forces the Anthropic Message Batches API for the run (50% off,
    # up-to-24h SLA); ``--no-batch`` forces sync. ``auto`` (default)
    # picks batch when planned LLM calls hit ``--auto-batch-threshold``.
    batch_group = generate_parser.add_mutually_exclusive_group()
    batch_group.add_argument(
        '--batch', dest='batch_mode', action='store_const', const='always',
        help='Force batch dispatch (50%% off, up to 24h SLA, Anthropic only)',
    )
    batch_group.add_argument(
        '--no-batch', dest='batch_mode', action='store_const', const='never',
        help='Force synchronous dispatch (immediate, full price)',
    )
    generate_parser.set_defaults(batch_mode='auto')
    generate_parser.add_argument(
        '--auto-batch-threshold', type=int, default=200,
        help=(
            'In ``auto`` batch_mode, switch to batch when planned '
            'LLM calls >= this threshold (default: 200)'
        ),
    )
    # ``--yes`` skips the first-run confirmation prompt that warns
    # about the 24h SLA. CI runs and frequent-batch users want this.
    generate_parser.add_argument(
        '--yes', dest='confirm_yes', action='store_true',
        help='Skip the first-run batch confirmation prompt',
    )


# ---------------------------------------------------------------------------
# Main command handler
# ---------------------------------------------------------------------------


async def cmd_generate(args: argparse.Namespace) -> int:
    """Generate documentation from source code using LLM."""
    import warnings

    # Suppress SyntaxWarning from ast.parse() — set BEFORE any code path
    # that parses Python source (dependency check, catalog extraction).
    warnings.filterwarnings('ignore', category=SyntaxWarning)

    cfg = get_config()
    db_path = Path(args.db) if args.db else Path(cfg.db_path)
    # Scope the per-run log handler to this call (removed on exit so it
    # can't stack/leak), and capture real per-call token usage into the
    # calibration store so the next dry-run self-tunes its estimate.
    from docgen.calibration import CalibrationStore, set_usage_observer
    with _run_log_handlers(db_path, args.verbose), \
            set_usage_observer(CalibrationStore(db_path).record):
        return await _cmd_generate_inner(args)


@contextmanager
def _run_log_handlers(db_path: Path, verbose: bool):
    """Attach a per-run ``ariadne_runs/generate-*.log`` handler (and a
    DEBUG stderr stream in verbose mode) to the root logger for the
    duration of a generate run, then remove them.

    The handler captures details like Anthropic cache stats, per-call
    timings, and validation-retry messages — useful for postmortems even
    when the console only shows WARNING+. Handlers are tagged and any
    prior tagged handlers are dropped before attaching, so repeated
    calls in one process can't stack handlers and a crashed run can't
    leave one capturing unrelated logs.
    """
    import logging
    from datetime import UTC, datetime

    log_dir = db_path.parent / 'ariadne_runs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / (
        f'generate-{datetime.now(UTC).strftime("%Y%m%d-%H%M%S")}.log'
    )
    root = logging.getLogger()
    prior_level = root.level
    for h in list(root.handlers):
        if getattr(h, '_ariadne_run_log', False):
            root.removeHandler(h)

    fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)s %(message)s')
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler._ariadne_run_log = True
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(fmt)
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    console.print(f'[dim]Run log: {log_path}[/dim]')
    added = [file_handler]

    if verbose:
        stream_handler = logging.StreamHandler()
        stream_handler._ariadne_run_log = True
        stream_handler.setLevel(logging.DEBUG)
        stream_handler.setFormatter(fmt)
        root.addHandler(stream_handler)
        root.setLevel(logging.DEBUG)
        added.append(stream_handler)

    try:
        yield
    finally:
        for h in added:
            root.removeHandler(h)
            try:
                h.close()
            except Exception:
                pass
        root.setLevel(prior_level)


async def _cmd_generate_inner(args: argparse.Namespace) -> int:
    """Body of ``cmd_generate``, run inside the scoped run-log handler."""
    import logging  # noqa: F401 — used by the body below
    from datetime import UTC, datetime  # noqa: F401

    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

    cfg = get_config()

    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(args.source)
    if source_path is None:
        if source_name and source_name in cfg.sources:
            console.print(
                f"[red]Source '{source_name}' is serve-only — it has no 'path' in "
                f"ariadne.yaml (its docs live in the database). Generation needs a "
                f"source path; add `path:` under sources.{source_name}.[/red]"
            )
        else:
            console.print('[red]No source specified and no default_source in config.[/red]')
            console.print('Use --source <path> or set default_source in ariadne.yaml')
        return 1

    if not source_path.exists():
        console.print(f'[red]Source path not found: {source_path}[/red]')
        return 1
    from docgen.source_graph import persist_source_graph
    from library import Library as _GraphLibrary
    _graph_lib = _GraphLibrary(Path(args.db) if args.db else Path(cfg.db_path))
    try:
        persist_source_graph(cfg, _graph_lib)
    finally:
        _graph_lib.close()

    if source_name and source_name in cfg.sources and not args.path:
        _check_and_prompt_dependencies(source_name, source_path, cfg)

    model = args.model or cfg.model

    # When --types is omitted, default to all LLM-driven types. ``catalog`` is
    # excluded — it's structural (ast-grep) and runs through ``catalog-sync``,
    # not the LLM cost path the estimator and orchestrator share.
    doc_types = (
        tuple(t.strip() for t in args.types.split(',') if t.strip())
        if args.types
        else DEFAULT_GENERATE_DOC_TYPES
    )

    dependencies: tuple[str, ...] = ()
    if source_name and source_name in cfg.sources:
        dependencies = tuple(cfg.get_source_dependencies(source_name))

    exclude_patterns: tuple[str, ...] = ()
    if source_name and source_name in cfg.sources:
        sc = cfg.get_source_config(source_name)
        if sc is not None:
            exclude_patterns = sc.exclude
    # Resolve the full effective excluded-dir set: global policy
    # ∪ source.exclude_dirs − source.exempt_dirs. Falls back to the
    # default policy when no source is named.
    exclude_dir_names = cfg.resolve_excluded_dirs(source_name)

    scip_config = cfg.get_source_scip_config(source_name) if source_name else None

    provider = resolve_provider(
        cli_provider=args.provider,
        cfg_provider=getattr(cfg, 'provider', None),
        model=model,
    )

    if args.dry_run:
        return _print_cost_estimate(
            source_path=source_path,
            source_name=source_name,
            target_path=Path(args.path) if args.path else None,
            doc_types=doc_types,
            model=model,
            exclude_patterns=exclude_patterns,
            exclude_dir_names=exclude_dir_names,
            catalog_only=True,
            provider=provider,
            batch_mode=getattr(args, 'batch_mode', 'auto'),
            auto_batch_threshold=getattr(args, 'auto_batch_threshold', 200),
            staleness_db_path=Path(cfg.staleness_db_path),
            base_path=source_path,
            library=_library_for_estimate(args.db),
            force=args.force,
        )

    # Commit-diff gate: regenerate only files changed since the source's last
    # synced commit, then promote HEAD on success. ``restrict_to_files=None``
    # falls back to the staleness path (first run / non-git / --force).
    _db_path = Path(args.db or cfg.db_path)
    # Skip commit-scoping for subdirectory runs (--path): they document only
    # part of the tree, so promoting HEAD would wrongly mark the whole source
    # synced and skip changed files elsewhere next time.
    restrict_to_files, _promote_head = (
        _commit_scope(_db_path, source_name, source_path, args.force)
        if source_name and not args.dry_run and not getattr(args, 'path', None)
        else (None, None)
    )

    config = OrchestratorConfig(
        source_path=source_path,
        db_path=args.db or Path(cfg.db_path),
        staleness_db_path=Path(cfg.staleness_db_path),
        model=model,
        api_key=args.api_key,
        doc_types=doc_types,
        concurrency=args.concurrency,
        force_regenerate=args.force,
        dry_run=args.dry_run,
        source_name=source_name,
        dependencies=dependencies,
        target_path=Path(args.path) if args.path else None,
        source_config=scip_config,
        inject_crossrefs=not getattr(args, 'no_crossrefs', False),
        provider=provider,
        exclude_patterns=exclude_patterns,
        exclude_dir_names=exclude_dir_names,
        batch_mode=getattr(args, 'batch_mode', 'auto'),
        auto_batch_threshold=getattr(args, 'auto_batch_threshold', 200),
        restrict_to_files=restrict_to_files,
    ignore_staleness=cfg.source_ignore_staleness(source_name))

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

    with Progress(*progress_columns, console=console) as progress:
        task_id = progress.add_task(
            f'Generating docs for {source_name}', total=0,
        )
        crossref_task_id = progress.add_task(
            'Crossrefs (idle)', total=0, visible=False,
        )

        def on_progress(msg: str, current: int, total: int) -> None:
            progress.update(
                task_id,
                completed=current,
                total=total if total > 0 else None,
                description=f'{msg}',
            )

        def on_crossref_progress(msg: str, current: int, total: int) -> None:
            progress.update(
                crossref_task_id,
                completed=current,
                total=total if total > 0 else None,
                description=msg,
                visible=True,
            )

        # Set the progress hook BEFORE entering the context so __aenter__
        # phases (DB schema migration, embedding normalization, provider
        # init) emit visibly. Otherwise the bar sits on its initial label
        # for 10-30s on populated DBs.
        orchestrator = DocGenOrchestrator(config)
        orchestrator.progress_callback = on_progress
        # Wire the first-run batch confirmation prompt. ``--yes`` skips
        # via ``yes_confirm``; otherwise ``cli_confirm`` reads stdin.
        # Only matters when batch dispatch resolves; harmless otherwise.
        orchestrator.confirm_callback = (
            yes_confirm if getattr(args, 'confirm_yes', False)
            else cli_confirm
        )
        # Heartbeat keeps the spinner + elapsed timer moving through the
        # long batch awaits (polls + results download), which otherwise
        # leave the bar frozen between sparse progress updates.
        async with orchestrator, _progress_heartbeat(progress):
            result = await orchestrator.run(
                progress_callback=on_progress,
                crossref_progress=on_crossref_progress,
            )

            # Phase 3: cross-source reverse-augment. Re-runs generation
            # for files in this source whose symbols are consumed by
            # other indexed sources, with consumer context injected
            # into the prompt. Sources without manifests are silently
            # skipped — the phase produces meaningful output only
            # when SCIP indexes have been built (ariadne discover +
            # ariadne index for related sources).
            if not getattr(result, 'aborted', False):
                related_sources: dict[str, Path] = {}
                for other_name in cfg.sources:
                    if other_name == source_name:
                        continue
                    other_path = cfg.get_source_path(other_name)
                    if other_path is not None:
                        related_sources[other_name] = other_path
                try:
                    augmented_count = (
                        await orchestrator.run_reverse_augment(
                            related_sources=related_sources,
                        )
                    )
                    if augmented_count:
                        console.print(
                            f'[green]Reverse-augmented {augmented_count} '
                            f'doc(s) with consumer context.[/green]'
                        )
                except Exception as e:
                    # Fail-soft on reverse-augment errors — the main
                    # generation already succeeded; surface the issue
                    # without overwriting the result.
                    console.print(
                        f'[yellow]Reverse-augment phase encountered an '
                        f'error: {e}[/yellow]'
                    )

                # Promote the synced commit: the next generate/onboard then
                # regenerates only files changed AFTER this run. Done only on a
                # clean (non-aborted) run so an interrupted pass re-scopes from
                # the prior commit and retries the unfinished files.
                if _promote_head:
                    from library import Library
                    with Library(_db_path) as _lib:
                        _lib.set_sync_state(source_name, _promote_head)

    # Aborted: show resume guidance prominently before the standard table.
    if getattr(result, 'aborted', False):
        from docgen.pricing import estimate_cost
        unprocessed = list(getattr(result, 'unprocessed_files', ()))
        files_with_size: list[tuple[Path, int]] = []
        for p in unprocessed:
            try:
                files_with_size.append((p, p.stat().st_size))
            except OSError:
                continue
        resume_estimate = estimate_cost(
            files=tuple(files_with_size),
            doc_types=doc_types,
            model=model,
            caching_enabled=(provider == 'anthropic'),
            batch_enabled=False,
        ) if files_with_size else None

        console.print()
        console.print(
            f'[bold red]Run aborted:[/bold red] {result.abort_reason}'
        )
        console.print(
            f'  [yellow]{len(unprocessed)} file(s)[/yellow] queued but not processed.'
        )
        if resume_estimate is not None:
            console.print(
                f'  Estimated [bold]${resume_estimate.total_cost_usd:.2f}[/bold] '
                f'to resume (range '
                f'${resume_estimate.cost_lower_bound:.2f} – '
                f'${resume_estimate.cost_upper_bound:.2f}).'
            )
        console.print(
            '  [dim]Re-run the same command later — staleness tracking will '
            'pick up where you left off.[/dim]'
        )

    console.print()
    table = Table(title='Generation Results')
    table.add_column('Metric', style='bold')
    table.add_column('Value', style='cyan')
    table.add_row('Files processed', str(result.files_processed))
    table.add_row('Files skipped (up-to-date)', str(result.files_skipped))
    if getattr(result, 'aborted', False):
        table.add_row(
            'Files unprocessed (resume)',
            f'[red]{len(getattr(result, "unprocessed_files", ()))}[/red]',
        )
    table.add_row('Documents created', str(result.docs_created))
    table.add_row('Documents failed', str(result.docs_failed))
    console.print(table)

    # Cache stats — print a one-line "$X saved" summary when caching engaged.
    # Drops to a WARNING if caching was expected (Anthropic) but didn't fire.
    cs = getattr(result, 'cache_stats', None)
    if cs is not None and getattr(cs, 'total_calls', 0) > 0:
        from docgen.pricing import LLM_PRICING
        rates = LLM_PRICING.get(model)
        if rates is not None:
            in_rate, _ = rates
            # 0.9 = (1.0 - 0.1) reads cost 10% of base, so 90% saved per read.
            # 0.25 = writes cost 1.25× base, so 25% extra per write.
            saved_by_reads = cs.total_read_tokens * 0.9 * in_rate / 1_000_000
            paid_for_writes = cs.total_create_tokens * 0.25 * in_rate / 1_000_000
            net_savings = saved_by_reads - paid_for_writes

            console.print()
            if cs.cache_reads > 0:
                console.print(
                    f'[green]Prompt caching:[/green] saved [bold green]'
                    f'${net_savings:.2f}[/bold green] '
                    f'({cs.cache_reads}/{cs.total_calls} calls cached, '
                    f'{cs.total_read_tokens:,} tokens read from cache).'
                )
            elif cs.total_calls > 5:
                # No cache hits is normal on Opus 4.6 (4096-token min) for
                # typical Ariadne prompts (~500-3500 user tokens). Surface
                # quietly with the model context so it isn't misread as a bug.
                console.print(
                    f'[dim]Prompt caching: no hits ({cs.total_calls} calls). '
                    f'Expected on {model} when prompts fall below the '
                    'cache token minimum.[/dim]'
                )

    if result.validation_retry_attempts > 0 or result.validation_initial_failures > 0:
        unrecovered = (
            result.validation_initial_failures - result.validation_recovered
        )
        console.print()
        retry_table = Table(title='Validation Retry Summary')
        retry_table.add_column('Metric', style='bold')
        retry_table.add_column('Value', style='cyan')
        retry_table.add_row(
            'Initial validation failures',
            str(result.validation_initial_failures),
        )
        retry_table.add_row(
            'Total retry attempts', str(result.validation_retry_attempts),
        )
        retry_table.add_row(
            'Recovered after retry',
            f'[green]{result.validation_recovered}[/green]',
        )
        retry_table.add_row(
            'Failed after all retries', f'[red]{unrecovered}[/red]',
        )
        console.print(retry_table)

    if result.errors:
        console.print()
        console.print('[yellow]Errors:[/yellow]')
        for err in result.errors[:10]:
            console.print(f'  - {err}')
        if len(result.errors) > 10:
            console.print(f'  ... and {len(result.errors) - 10} more errors')

    if args.verbose and result.validation_results:
        from docgen.validator import format_validation_report

        failed_validations = [
            v for v in result.validation_results if not v.is_valid
        ]
        if failed_validations:
            console.print()
            console.print('[yellow]Validation Reports:[/yellow]')
            for validation in failed_validations[:10]:
                report = format_validation_report(
                    validation, validation.title or 'Document',
                )
                console.print()
                console.print(report)
            if len(failed_validations) > 10:
                console.print(
                    f'\n  ... and {len(failed_validations) - 10} '
                    f'more validation failures',
                )

    # Partial failure (some docs succeeded, some failed) is a warning, not
    # a hard error — surface it loudly but don't fail the run, so an
    # onboard pipeline continues to its later phases. Batch mode doesn't
    # retry validation failures, so a re-run picks up just the failed docs.
    if result.docs_failed > 0 and result.docs_created > 0:
        console.print(
            f'[yellow]⚠ {result.docs_failed} doc(s) failed[/yellow] '
            f'({result.docs_created} succeeded). Re-run `ariadne generate` '
            f'to retry just the failed ones (staleness skips the rest).'
        )

    return _generate_exit_code(result)
