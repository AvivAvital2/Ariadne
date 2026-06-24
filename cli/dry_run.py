"""Dry-run cost preview command (``dry-run``).

Extracted from cli/generation.py. Runs the free pipeline phases and estimates
the cost of the LLM-paid phases (catalog-describe, generate, themes); ``-i``
opens the interactive explorer. Wired into the parser via this module's
``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

from schema import CATALOG_KIND_ELEMENT

import argparse
import asyncio
import io
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from cli.catalog import (
    _DESCRIBE_INPUT_TOKENS_PER_CALL,
    _DESCRIBE_OUTPUT_TOKENS_PER_CALL,
    cmd_catalog_sync,
)
from cli.explorer_ui import run_explorer_tui

if TYPE_CHECKING:
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


_THEMES_INPUT_TOKENS_PER_THEME = 2000
_THEMES_OUTPUT_TOKENS_PER_THEME = 600


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the dry-run cost-preview command."""

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
    dryrun_parser.add_argument('-i', '--interactive', action='store_true',
        help='Open the full-screen explorer: a tree of per-directory generate '
             'cost (ranked, with bars) — toggle excludes, then write them to '
             'ariadne.yaml and re-estimate. Without a TTY, prints the static '
             'ranked per-directory cost table instead.')


def _describe_tokens_per_call(store, model: str) -> tuple[float, float]:
    """(input, output) tokens per catalog-describe call for ``model``.

    Uses real calibrated usage from the store when available, else the
    empirical 200/60 heuristic — so the describe estimate self-tunes
    after a run."""
    cal = store.mean_tokens(phase='describe', model=model) if store else None
    if cal is not None:
        return cal.mean_input, cal.mean_output
    return _DESCRIBE_INPUT_TOKENS_PER_CALL, _DESCRIBE_OUTPUT_TOKENS_PER_CALL


def _estimate_themes_cost(
    library: "Library", model: str,
) -> tuple[int, float | None, tuple[float, float] | None]:
    """Count the DIRTY themes and estimate the LLM cost to (re)summarize them.
    Returns ``(dirty_count, cost_usd_or_None, rates)``.

    Only dirty themes incur an LLM call: the Leiden rebuild is free and
    deterministic and re-summarizes only themes whose membership changed, so an
    unchanged tree estimates $0 (zero dirty). Mirrors the runtime, where
    generate_themes summarizes exactly the dirty set.
    """
    from docgen.pricing import LLM_PRICING

    dirty = set(library.get_dirty_themes())
    themes = [
        t for t in library.list_themes(coherent_only=False)
        if t.cluster_id in dirty
    ]
    n = len(themes)
    rates = LLM_PRICING.get(model)
    if rates is None:
        return n, None, None
    input_per_m, output_per_m = rates
    from docgen.themes import _build_theme_request
    from docgen.token_count import count_text_tokens
    input_total = 0
    _tt_ok = n > 0
    for _th in themes:
        _req = _build_theme_request(library, _th.cluster_id)
        if _req is None:
            continue
        _cnt = count_text_tokens(
            _req.system_prompt + "\n" + _req.user_prompt, model,
        )
        if _cnt is None:
            _tt_ok = False
            break
        input_total += _cnt
    if not _tt_ok or input_total == 0:
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


def _discover_files_for_estimate(cfg, source_name, source_path):
    """Files the generate phase will actually process, honoring the source's
    excludes — the SAME ``exclude_patterns`` / ``exclude_dir_names`` the real
    run (and the ``generate --dry-run`` table) apply.

    Without this the preview walks test suites and deploy configs the run
    skips (``test``/``testkit``/``cypress``/``kustomize``/…), over-counting
    and over-pricing them. Returns ``(path, size)`` pairs; unreadable files
    are skipped.
    """
    from config import DEFAULT_EXCLUDE_FILE_PATTERNS
    from docgen.staleness import find_catalog_files

    sc = cfg.get_source_config(source_name)
    files: list = []
    for p in find_catalog_files(
        source_path,
        exclude_patterns=(
            DEFAULT_EXCLUDE_FILE_PATTERNS + tuple((sc.exclude if sc else ()) or ())
        ),
        exclude_dir_names=cfg.resolve_excluded_dirs(source_name),
    ):
        try:
            files.append((p, p.stat().st_size))
        except OSError:
            continue
    return files


def _apply_explorer_staleness(cfg, source_name, *, chosen, currently_exempt) -> bool:
    """Persist the explorer staleness modal's choice to ariadne.yaml.

    Writes ``ignore_staleness: true`` only when the user accepted (``chosen``)
    and the source isn't already exempt — so it never re-writes a no-op or
    overrides an existing setting. ``set_source_config`` merges, so this
    preserves any excludes written alongside it. Returns whether it wrote.
    """
    if not chosen or currently_exempt:
        return False
    cfg.set_source_config(source_name, ignore_staleness=True)
    return True


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
    # reason (cli/generate.py); must be set before any parse path runs.
    warnings.filterwarnings('ignore', category=SyntaxWarning)

    from cli.index import cmd_discover, cmd_index
    from docgen.pricing import LLM_PRICING
    # Late import so monkeypatches to ``config.get_config`` in tests
    # take effect inside this command. Without this, the
    # module-level ``from config import get_config`` reference at the
    # top of cli/generation.py captures the original function at
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
            and d.metadata.get('kind') == CATALOG_KIND_ELEMENT
            and not d.metadata.get('description')
        ]
        # Calibration: use real per-call tokens from past runs when the
        # store has them, else the heuristic. Shares the library DB file.
        from docgen.calibration import CalibrationStore
        cal_store = CalibrationStore(args.db or cfg.db_path)
        desc_in, desc_out = _describe_tokens_per_call(cal_store, model)
        # Input: tiktoken the REAL describe prompt per element (its text is
        # known from the catalog metadata). Falls back to the calibrated/
        # flat per-call figure if tiktoken can't count any element. Output
        # can't be counted ahead of generation, so it stays calibrated.
        from docgen.catalog_describer import build_describe_prompt
        from docgen.token_count import count_text_tokens
        describe_input = 0
        _desc_tt_ok = bool(candidates)
        for _d in candidates:
            _n = count_text_tokens(build_describe_prompt(_d.metadata), model)
            if _n is None:
                _desc_tt_ok = False
                break
            describe_input += _n
        if not _desc_tt_ok:
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
        files = []
        if source_path is not None and source_path.exists():
            files = _discover_files_for_estimate(cfg, source_name, source_path)

        # Incremental scope: price what the real run will actually generate.
        # The real run (generate/onboard) uses the commit-diff gate — only
        # files changed since the source's last synced commit — so the preview
        # must scope the same way, else it over-reports a from-scratch cost.
        # When there's no commit gate (first run / non-git / --force) fall back
        # to the staleness subset. `files` stays the full set for the secondary
        # "full regeneration" figure.
        from cli.generate import (
            DEFAULT_GENERATE_DOC_TYPES)
        from cli.generate_cost import (
            _commit_scope, _stale_subset_for_estimate)
        requested_doc_types = (
            tuple(t.strip() for t in args.types.split(',') if t.strip())
            if getattr(args, 'types', None) else DEFAULT_GENERATE_DOC_TYPES
        )
        _restrict, _ = (
            _commit_scope(
                Path(getattr(args, 'db', None) or cfg.db_path),
                source_name, source_path, getattr(args, 'force', False),
            )
            if source_name and source_path is not None
            and not getattr(args, 'path', None)
            else (None, None)
        )
        if files and _restrict is not None:
            # Commit gate active: price exactly the changed files.
            def _rel(p: Path) -> str:
                try:
                    return p.relative_to(source_path).as_posix()
                except ValueError:
                    return p.as_posix()
            gen_files = [(p, s) for (p, s) in files if _rel(p) in _restrict]
        elif files and not getattr(args, 'force', False):
            gen_files = _stale_subset_for_estimate(
                files,
                staleness_db_path=Path(cfg.staleness_db_path),
                base_path=source_path,
                doc_types=requested_doc_types,
                library=library,
            doc_types_by_language=cfg.source_doc_types_by_language(source_name))
        else:
            gen_files = files
        gen_skipped = len(files) - len(gen_files)
        full_generate_cost: float | None = None
        full_generate_cost_batched: float | None = None

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

            # Exact input (file-content) tokens via tiktoken, cached per
            # path across the baseline / batched / per-type passes. Shared
            # factory (same one the generate --dry-run table uses); falls
            # back to the char heuristic when tiktoken is unavailable.
            from cli.generate_cost import _scaffold_overhead_counter
            from docgen.token_count import file_token_counter
            _gen_input = file_token_counter(model)
            _gen_overhead = _scaffold_overhead_counter(model)

            # Two estimates: baseline (no --batch) and with Anthropic's
            # ~50% Message Batches discount applied. ``generate`` is the
            # only phase that batches today.
            generate_estimate = estimate_cost(
                files=gen_files,
                doc_types=requested_doc_types,
                model=model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
                input_tokens_for=_gen_input,
                prompt_overhead_for=_gen_overhead,
            )
            generate_estimate_batched = estimate_cost(
                files=gen_files,
                doc_types=requested_doc_types,
                model=model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
                input_tokens_for=_gen_input,
                prompt_overhead_for=_gen_overhead,
                batch_enabled=True,
            )
            generate_cost: float | None = generate_estimate.total_cost_usd
            generate_cost_batched: float | None = (
                generate_estimate_batched.total_cost_usd
            )
            # Full-regeneration figures for the note — only when some
            # files were skipped (else they equal the headline cost).
            if gen_skipped > 0:
                full_generate_cost = estimate_cost(
                    files=files,
                    doc_types=requested_doc_types,
                    model=model,
                    caching_enabled=caching_enabled,
                    output_tokens_for=_gen_output,
                    input_tokens_for=_gen_input,
                    prompt_overhead_for=_gen_overhead,
                ).total_cost_usd
                full_generate_cost_batched = estimate_cost(
                    files=files,
                    doc_types=requested_doc_types,
                    model=model,
                    caching_enabled=caching_enabled,
                    output_tokens_for=_gen_output,
                    input_tokens_for=_gen_input,
                    prompt_overhead_for=_gen_overhead,
                    batch_enabled=True,
                ).total_cost_usd
        else:
            generate_cost = 0.0 if rates is not None else None
            generate_cost_batched = generate_cost
        themes_count, themes_cost, _ = _estimate_themes_cost(library, model)

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
            calls=len(gen_files),
            unit='files',
            in_tokens=(
                generate_estimate.input_tokens if gen_files and rates
                else 0
            ),
            out_tokens=(
                generate_estimate.output_tokens if gen_files and rates
                else 0
            ),
            cost=generate_cost,
            batched_cost=generate_cost_batched,
            verbose=verbose,
        )
        if gen_skipped > 0 and rates is not None:
            console.print(
                f'      [dim]{gen_skipped} of {len(files)} files already '
                f'generated (unchanged source) — skipped. Full '
                f'regeneration: ${full_generate_cost:.2f} / '
                f'${full_generate_cost_batched:.2f} batched; use '
                f'--force.[/dim]',
                soft_wrap=True,
            )
        # Per-doc-type breakdown so the user sees what each type costs
        # (and can drop the expensive ones). Only meaningful when we have
        # stale files to generate + a known rate.
        if gen_files and rates is not None:
            from cli.generate import DEFAULT_GENERATE_DOC_TYPES
            from docgen.pricing import estimate_generate_by_doc_type
            per_type = estimate_generate_by_doc_type(
                gen_files, requested_doc_types, model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
                input_tokens_for=_gen_input,
                prompt_overhead_for=_gen_overhead,
            )
            per_type_batched = dict(estimate_generate_by_doc_type(
                gen_files, requested_doc_types, model,
                caching_enabled=caching_enabled,
                output_tokens_for=_gen_output,
                input_tokens_for=_gen_input,
                prompt_overhead_for=_gen_overhead,
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
        # --interactive: explore per-directory cost and toggle excludes.
        # Scoped to the PENDING set (gen_files) — the same files the headline
        # prices and the real run will actually generate — so it never shows a
        # from-scratch price for an already-generated repo (the reported bug).
        # When nothing is pending the explorer is skipped (note below); any
        # excludes set here still persist to ariadne.yaml and apply to future
        # runs. On a TTY it opens the explorer; otherwise it prints the static
        # ranked table.
        if getattr(args, 'interactive', False) and gen_files and rates is not None:
            import sys as _sys

            from docgen.cost_by_dir import (
                cost_by_directory,
                format_by_dir_table,
            )
            _hooks = {
                'output_tokens_for': _gen_output,
                'input_tokens_for': _gen_input,
                'prompt_overhead_for': _gen_overhead,
            }
            cost_nodes = cost_by_directory(
                gen_files, source_path, requested_doc_types, model, **_hooks)
            leaves = {p.relative_to(source_path).as_posix() for p, _ in gen_files}
            if _sys.stdout.isatty() and _sys.stdin.isatty():
                from docgen.explorer_state import ExplorerState, apply_excludes
                from docgen.scan_tree import scan_tree
                excluded_dirs = set(cfg.resolve_excluded_dirs(source_name))
                tree = scan_tree(source_path, excluded_dirs=excluded_dirs)
                state = ExplorerState(
                    tree, cost_nodes, auto_excluded=sorted(excluded_dirs))

                def _recost(selected_types):
                    # Re-price the pending tree for a doc-type selection (the
                    # left-panel checkboxes drive this live).
                    return cost_by_directory(
                        gen_files, source_path, tuple(selected_types), model,
                        **_hooks)

                # Onboarding (only) asks the file browser to pop the staleness
                # modal on Apply, so the browser — not a CLI prompt — owns this.
                offer_staleness = getattr(args, 'offer_staleness', False)
                currently_exempt = bool(cfg.source_ignore_staleness(source_name))
                try:
                    app = await run_explorer_tui(
                        state, doc_types=requested_doc_types,
                        selected=requested_doc_types, recost=_recost,
                        offer_staleness=offer_staleness,
                        staleness_source=source_name,
                        staleness_exempt=currently_exempt)
                except KeyboardInterrupt:
                    console.print(
                        '[dim]Explorer cancelled — no changes written.[/dim]')
                else:
                    rules = state.excluded_rules()
                    chosen = app.selected_doc_types
                    if apply_excludes(state, cfg, source_name) and (
                        rules['exclude_dirs'] or rules['exclude']
                    ):
                        console.print(
                            f'[green]✓[/green] Wrote excludes to '
                            f'{cfg.config_path}: '
                            f'exclude_dirs={rules["exclude_dirs"]} '
                            f'exclude={rules["exclude"]}')
                    if _apply_explorer_staleness(
                        cfg, source_name,
                        chosen=getattr(app, 'staleness_exempt', False),
                        currently_exempt=currently_exempt,
                    ):
                        console.print(
                            f"[green]✓[/green] Marked '{source_name}' "
                            f'staleness-exempt in {cfg.config_path}')

                    _names = set(rules['exclude_dirs'])
                    _globs = rules['exclude']

                    def _kept(pair, _names=_names, _globs=_globs):
                        rel = pair[0].relative_to(source_path)
                        if any(part in _names for part in rel.parts[:-1]):
                            return False
                        return not any(pair[0].match(g) for g in _globs)

                    # Re-price THIS run (the pending set) for the chosen doc
                    # types + remaining files — what onboard is about to pay,
                    # consistent with the headline. Full-from-scratch stays in
                    # the "Full regeneration … use --force" note above.
                    kept = [pr for pr in gen_files if _kept(pr)]
                    dropped = len(gen_files) - len(kept)
                    before = estimate_cost(
                        files=gen_files, doc_types=requested_doc_types,
                        model=model, caching_enabled=caching_enabled,
                        **_hooks).total_cost_usd
                    after = estimate_cost(
                        files=kept, doc_types=chosen, model=model,
                        caching_enabled=caching_enabled, **_hooks).total_cost_usd
                    console.print(
                        f'  [green]this run (after explorer)[/green]  '
                        f'${after:.2f}  [dim](was ${before:.2f} for the default '
                        f'set; {dropped} of {len(gen_files)} files dropped)[/dim]',
                        soft_wrap=True,
                    )
                    # Doc-type choice isn't persisted (unlike excludes), so when
                    # it changed, surface the --types to pass to generate.
                    if tuple(chosen) != tuple(requested_doc_types):
                        hint = ','.join(chosen)
                        console.print(
                            f'  [dim]doc types:[/] {hint or "(none selected)"}'
                            + (f'   [dim]→ generate with[/] --types {hint}'
                               if hint else ''),
                            soft_wrap=True,
                        )
                    # Fold excludes + chosen types into the incremental total.
                    if gen_files:
                        kept_gen = [pr for pr in gen_files if _kept(pr)]
                        generate_cost = estimate_cost(
                            files=kept_gen, doc_types=chosen, model=model,
                            caching_enabled=caching_enabled, **_hooks).total_cost_usd
                        generate_cost_batched = estimate_cost(
                            files=kept_gen, doc_types=chosen, model=model,
                            caching_enabled=caching_enabled, batch_enabled=True,
                            **_hooks).total_cost_usd
            else:
                console.print()
                for _line in format_by_dir_table(
                    cost_nodes, leaves=leaves, model=model,
                ).splitlines():
                    console.print(_line, soft_wrap=True)
        elif (
            getattr(args, 'interactive', False) and files
            and not gen_files and rates is not None
        ):
            # Interactive was requested but there's nothing to generate — don't
            # open the explorer on a from-scratch price. The "Full regeneration
            # … use --force" note above already shows what a forced rebuild
            # costs.
            console.print(
                '  [dim]Nothing to generate — skipping the cost explorer. '
                'Re-run with --force to regenerate (see the full-regeneration '
                'figure above).[/dim]',
                soft_wrap=True,
            )
        # themes build: when nothing is clustered yet (first run), the
        # count is 0 — but onboard WILL cluster + summarize, so showing
        # "$0.00" implies it's free. Mark it not-estimated and keep it
        # out of the total (flagged below) instead.
        themes_unestimated = library.count_documents(content_type='theme') == 0
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
                unit='dirty themes',
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
            # Both catalog-describe AND generate support --batch; the batched
            # total applies the discount to both phases.
            total_batched = (
                describe_cost_batched + generate_cost_batched + themes_part
            )
            console.print(
                f'  [bold]Estimated minimum: ${total_baseline:.2f} sync / '
                f'${total_batched:.2f} batched[/bold]',
                soft_wrap=True,
            )
            # The figure is a FLOOR: a char-based token heuristic, and it
            # omits any phase that can't be priced yet (first-run themes
            # summarization, which onboard still runs). The real cost only
            # ever lands at or above it — so never show a lower bound
            # under the estimate; offer a ~+50% safety ceiling instead,
            # for BOTH the sync and batched figures.
            omits = []
            if themes_unestimated:
                omits.append('first-run themes summarization')
            ceiling = (
                f'  [dim]Add up to ~+50% to be safe: '
                f'${total_baseline * 1.5:.2f} sync / '
                f'${total_batched * 1.5:.2f} batched.'
            )
            if omits:
                ceiling += f' Excludes {", ".join(omits)}.'
            console.print(ceiling + '[/dim]', soft_wrap=True)
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


HANDLERS = {
    'dry-run': lambda args: asyncio.run(cmd_dry_run(args)),
}
