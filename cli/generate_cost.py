"""Generate-phase cost estimation + dependency-prompt helpers.

Extracted from cli/generate.py. Estimates the LLM cost of the generate phase
(per-source, stale-subset, commit-scope) and prompts about missing source
dependencies before generating. Pure helpers — imported by cmd_generate; not
a command module.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from docgen.batch_resolution import BATCH_ELIGIBLE_PROVIDERS

if TYPE_CHECKING:
    from collections.abc import Callable

    from config import Config
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


def _library_for_estimate(db_path: Path | None) -> object:
    """Open the library so the cost estimate's staleness check is
    type-aware — matching ``DocGenOrchestrator.run``. Opens (and creates
    if absent) the same library the real run uses; any failure here would
    block the real run too, so it is surfaced, not swallowed.
    """
    from cli.core import get_library
    return get_library(db_path)


def _commit_scope(
    db_path: Path,
    source_name: str,
    source_path: Path,
    force: bool,
) -> tuple[frozenset[str] | None, str | None]:
    """Resolve the incremental commit-diff scope for a generate/onboard run.

    The commit recorded in ``sync_state`` is the source of truth for "what's
    already documented": a run regenerates only files changed since it, then
    promotes HEAD. Returns ``(restrict_to_files, head)``:

    - ``head`` — the current HEAD to promote into ``sync_state`` after a
      successful run, or None when ``source_path`` isn't a git repo (then
      there's nothing to promote and the caller uses the staleness path).
    - ``restrict_to_files`` — source-relative paths changed since the last
      synced commit, or None meaning "no commit scoping; run the full /
      staleness pass" (never synced, ``force``, or a git/diff failure). An
      empty frozenset means "synced and nothing changed" → generate nothing.
    """
    from git_ops import get_changed_files_since, get_head_commit
    from library import Library

    head = get_head_commit(source_path)
    if head is None:
        return None, None            # not a git repo → legacy staleness, no promote
    if force:
        return None, head            # full pass, but still promote the baseline
    with Library(db_path) as lib:
        state = lib.get_sync_state(source_name)
    if state is None:
        return None, head            # never synced → full first pass, set baseline after
    changed = get_changed_files_since(state[0], source_path)
    if changed is None:
        return None, head            # diff failed → safe full fallback
    return frozenset(changed), head


def _stale_subset_for_estimate(
    files_with_size: list[tuple[Path, int]],
    *,
    staleness_db_path: Path,
    base_path: Path,
    doc_types: tuple[str, ...],
    library: object | None = None,
) -> list[tuple[Path, int]]:
    """Subset of ``files_with_size`` the next generate run would actually
    process — stale or never-documented — using the SAME staleness logic
    as :meth:`DocGenOrchestrator.run` (type-aware when ``library`` is
    given). Lets the dry-run estimate report the real incremental cost
    instead of a from-scratch upper bound: re-running with a populated
    staleness DB skips unchanged files, so pricing every discovered file
    overstates the cost.
    """
    from docgen.staleness import StalenessTracker

    paths = [p for p, _ in files_with_size]
    with StalenessTracker(staleness_db_path) as tracker:
        stale = set(
            tracker.get_stale_files(
                paths,
                base_path=base_path,
                requested_types=doc_types,
                library=library,
            ),
        )
    return [(p, size) for (p, size) in files_with_size if p in stale]


def _calibrated_generate_hooks(store, model, file_counter):
    """``(input_tokens_for, output_tokens_for)`` for the generate estimate,
    backed by recorded per-call usage (``phase='generate'``) when the store
    has data — finest ``(doc_type, language)`` bucket first, then coarser
    fallbacks — else the file-content / flat-output heuristics.

    Because the recorded ``mean_input`` is the *real* prompt input, this also
    captures the cross-source / dependency context the file-content count
    alone misses; the estimate self-tunes after one real run.
    """
    def _bucket(doc_type=None, language=None):
        if store is None:
            return None
        return store.mean_tokens(
            phase='generate', model=model,
            doc_type=doc_type, language=language,
        )

    def output_tokens_for(doc_type, language):
        cal = (_bucket(doc_type, language) or _bucket(language=language)
               or _bucket(doc_type=doc_type) or _bucket())
        return cal.mean_output if cal is not None else None

    def input_tokens_for(path):
        from docgen.pricing import _detect_language
        cal = _bucket(language=_detect_language(path)) or _bucket()
        return cal.mean_input if cal is not None else file_counter(path)

    return input_tokens_for, output_tokens_for


def _print_cost_estimate(
    *,
    source_path: Path,
    source_name: str | None,
    target_path: Path | None,
    doc_types: tuple[str, ...],
    model: str,
    exclude_patterns: tuple[str, ...],
    exclude_dir_names: tuple[str, ...],
    catalog_only: bool,
    provider: str = 'openai',
    batch_mode: str = 'auto',
    auto_batch_threshold: int = 200,
    staleness_db_path: Path | None = None,
    base_path: Path | None = None,
    library: object | None = None,
    force: bool = False,
) -> int:
    """Discover files, run the cost estimator, and print a Rich table.

    Used by ``--dry-run`` so users see expected cost before paying for
    LLM calls. Returns 0 always.
    """
    from docgen.pricing import estimate_cost
    from docgen.staleness import find_catalog_files, find_python_files

    discover = find_catalog_files if catalog_only else find_python_files
    from config import DEFAULT_EXCLUDE_FILE_PATTERNS
    excludes = DEFAULT_EXCLUDE_FILE_PATTERNS + tuple(exclude_patterns)

    search_root = source_path
    if target_path is not None:
        full_target = source_path / target_path
        if full_target.is_file():
            files_paths = [full_target]
        else:
            search_root = full_target
            files_paths = discover(
                search_root,
                exclude_patterns=excludes,
                exclude_dir_names=exclude_dir_names,
            )
    else:
        files_paths = discover(
            search_root,
            exclude_patterns=excludes,
            exclude_dir_names=exclude_dir_names,
        )

    full_files: list[tuple[Path, int]] = []
    for p in files_paths:
        try:
            full_files.append((p, p.stat().st_size))
        except OSError:
            continue

    # Incremental scope: a real run skips files whose source is unchanged
    # (DocGenOrchestrator.run filters via get_stale_files), so the headline
    # estimate must price only the stale/new subset. ``full_files`` is kept
    # for a secondary "full regeneration" figure. ``--force`` regenerates
    # everything, so it bypasses the filter.
    total_files = len(full_files)
    if staleness_db_path is not None and not force:
        files_with_size = _stale_subset_for_estimate(
            full_files,
            staleness_db_path=staleness_db_path,
            base_path=base_path or source_path,
            doc_types=doc_types,
            library=library,
        )
    else:
        files_with_size = full_files
    stale_count = len(files_with_size)

    # Caching kicks in only on Anthropic (explicit cache_control markers);
    # OpenAI's is automatic and not modeled in the estimate. Batch eligibility
    # reuses the authoritative set from docgen.batch_resolution (the same gate
    # the orchestrator consults just below), so the estimate can't diverge from
    # the actual dispatch decision.
    caching_enabled = provider == 'anthropic'
    batch_eligible = provider in BATCH_ELIGIBLE_PROVIDERS

    # Exact input (file-content) tokens via tiktoken, cached per path
    # across the estimate passes. Shared factory (same one onboard's
    # dry-run uses); falls back to the char heuristic when tiktoken is
    # unavailable.
    from docgen.token_count import file_token_counter
    _scaffold_overhead = _scaffold_overhead_counter(model)
    # Self-tuning input/output hooks: prefer recorded per-call usage — whose
    # input INCLUDES the cross-source / dependency prompt context the
    # file-content count alone misses — falling back to the tiktoken file
    # count + flat-output heuristic when the store has no data yet.
    _cal_store = None
    if library is not None:
        try:
            from docgen.calibration import CalibrationStore
            _cal_store = CalibrationStore(library._conn_provider.path)
        except Exception:
            _cal_store = None
    _input_tokens, _output_tokens = _calibrated_generate_hooks(
        _cal_store, model, file_token_counter(model),
    )
    # When calibration drives the input, the recorded mean_input already
    # includes the prompt scaffold — so don't add the scaffold heuristic on
    # top (it would double-count). Cold runs (no data) still need it.
    _has_cal = (
        _cal_store is not None
        and _cal_store.mean_tokens(phase='generate', model=model) is not None
    )
    _scaffold = (lambda _dt: 0) if _has_cal else _scaffold_overhead

    # For auto mode we need a planned-calls count to compare against
    # the threshold. ``estimate_cost(batch_enabled=False)`` is the
    # cheapest way to get that count without recomputing pricing.
    # always/never/non-anthropic short-circuit before consulting
    # planned_calls, so a 0 here is harmless.
    if batch_mode == 'auto' and batch_eligible:
        provisional = estimate_cost(
            files=tuple(files_with_size),
            doc_types=doc_types,
            model=model,
            caching_enabled=caching_enabled,
            batch_enabled=False,
        )
        planned_calls = provisional.total_calls
    else:
        planned_calls = 0

    from docgen.batch_resolution import (
        apply_dispatch_gate,
        resolve_batch_decision,
    )

    batch_resolved, batch_reason = resolve_batch_decision(
        provider=provider,
        batch_mode=batch_mode,
        planned_calls=planned_calls,
        auto_threshold=auto_batch_threshold,
    )

    # Phase C-fix: gate the 50% batch claim behind a feature flag until
    # the orchestrator's run() actually dispatches via submit_batch
    # (#45). Becomes a no-op once #45.9 flips the flag True.
    batch_resolved, batch_reason = apply_dispatch_gate(
        batch_resolved, batch_reason,
    )

    estimate = estimate_cost(
        files=tuple(files_with_size),
        doc_types=doc_types,
        model=model,
        caching_enabled=caching_enabled,
        batch_enabled=batch_resolved,
        input_tokens_for=_input_tokens,
        output_tokens_for=_output_tokens,
        prompt_overhead_for=_scaffold,
    )

    # Full-regeneration figure for the note — only recomputed when some
    # files were skipped (otherwise it equals the headline estimate).
    skipped = total_files - stale_count
    if skipped > 0:
        full_estimate = estimate_cost(
            files=tuple(full_files),
            doc_types=doc_types,
            model=model,
            caching_enabled=caching_enabled,
            batch_enabled=batch_resolved,
            input_tokens_for=_input_tokens,
            prompt_overhead_for=_scaffold,
        )
    else:
        full_estimate = estimate

    console.print()
    table = Table(
        title=(
            f'Dry-run cost estimate for '
            f'{source_name or source_path.name}'
        ),
    )
    table.add_column('Metric', style='bold')
    table.add_column('Value', style='cyan')
    table.add_row('Source path', str(source_path))
    table.add_row('Model', model)
    table.add_row('Doc types requested', ', '.join(doc_types))
    table.add_row('Files discovered', str(total_files))
    table.add_row(
        'Stale / new (this run)',
        f'{stale_count} stale/new of {total_files} files'
        + (f'  ({skipped} up-to-date, skipped)' if skipped > 0 else ''),
    )
    table.add_row('Total LLM calls', f'{estimate.total_calls:,}')
    table.add_row('Estimated input tokens', f'{estimate.input_tokens:,}')
    table.add_row('Estimated output tokens', f'{estimate.output_tokens:,}')
    table.add_row('Estimated embedding tokens', f'{estimate.embedding_tokens:,}')

    if estimate.rates is None:
        table.add_row(
            '[yellow]Cost[/yellow]',
            f'[yellow]unknown — model {model!r} not in pricing table[/yellow]',
        )
    else:
        in_rate, out_rate = estimate.rates
        table.add_row(
            'Pricing (per 1M tokens)',
            f'input ${in_rate:.2f}, output ${out_rate:.2f}',
        )
        table.add_row(
            'Baseline LLM cost (no cache, no batch)',
            f'${estimate.baseline_cost_usd:.2f}',
        )
        if estimate.cache_savings_usd > 0:
            table.add_row(
                '  - Prompt-caching savings',
                f'[green]-${estimate.cache_savings_usd:.2f}[/green]',
            )
        if estimate.batch_savings_usd > 0:
            table.add_row(
                '  - Batch API savings (50% off, batch mode active)',
                f'[green]-${estimate.batch_savings_usd:.2f}[/green]',
            )
        table.add_row(
            'LLM cost (with selected discounts)',
            f'[green]${estimate.llm_cost_usd:.2f}[/green]',
        )
        table.add_row(
            'Embedding cost',
            f'${estimate.embedding_cost_usd:.2f}',
        )
        # Floor, not a midpoint: the char-based heuristic and any skipped
        # phases mean the real cost only lands at or above this. Show the
        # estimate as a minimum with a ~+50% ceiling — no lower bound.
        table.add_row(
            '[bold]Estimated minimum[/bold]',
            f'[bold green]${estimate.total_cost_usd:.2f}[/bold green]'
            f' (up to ~+50%: ${estimate.cost_upper_bound:.2f})',
        )

    table.add_row(
        'Caching',
        '[green]on (Anthropic)[/green]' if caching_enabled else '[dim]off (OpenAI)[/dim]',
    )
    if batch_resolved:
        batch_display = f'[green]using batch (50% off)[/green] — {batch_reason}'
    elif batch_eligible:
        batch_display = f'using sync — {batch_reason}'
    else:
        batch_display = f'[yellow]using sync[/yellow] — {batch_reason}'
    table.add_row('Batch mode', batch_display)

    console.print(table)
    if skipped > 0 and estimate.rates is not None:
        console.print(
            f'[dim]{skipped} of {total_files} files are already generated '
            f'(unchanged source) — skipped. Full regeneration would be '
            f'${full_estimate.total_cost_usd:.2f}; use --force to '
            f'regenerate everything.[/dim]',
        )
    console.print(
        '[dim]Output length is estimated (it can not be counted before '
        'generation); real cost can run up to ~+50% higher.[/dim]',
    )
    return 0


def _excluded_evidence_note(dep, excluded_dirs: set[str]) -> str:
    """Explain evidence found inside a directory excluded from documentation.

    Import analysis intentionally scans every import — including dirs like
    ``.venv``/``site-packages`` that doc generation excludes — so it can
    detect dependencies that surface only through an installed package. That
    scan runs locally (no LLM, no cost). Return an explanatory note when a
    dependency's evidence comes from such a directory; otherwise ``''``.
    """
    from pathlib import Path as _P

    matched = next(
        (
            part
            for ev in dep.evidence
            for part in _P(ev.file_path).parts
            if part in excluded_dirs
        ),
        None,
    )
    if matched is None:
        return ''
    return (
        f'[dim]ⓘ Found via an installed package under {matched}/ — a '
        f'directory excluded from documentation, but intentionally still '
        f'scanned for imports so Ariadne can catch links that first-party '
        f"code alone wouldn't reveal. This scan runs locally; no file content "
        f'is sent to the LLM, so it adds no cost.[/dim]'
    )


def _check_and_prompt_dependencies(
    source_name: str,
    source_path: Path,
    cfg: 'Config',
) -> None:
    """Check for dependencies and prompt user to save if detected.

    Args:
        source_name: Name of the source being generated.
        source_path: Path to the source code.
        cfg: Configuration instance.
    """
    from rich.panel import Panel

    from docgen.dependency import detect_dependencies

    # Check if dependencies are already configured
    existing_deps = cfg.get_source_dependencies(source_name)
    if existing_deps:
        return

    # Get all known source paths
    known_sources = cfg.get_all_source_paths()
    if len(known_sources) < 2:
        return  # No other sources to depend on
    # Ariadne's own source is never a real dependency target (its name also
    # collides with common packages, e.g. the `ariadne` GraphQL lib), so
    # ignore any configured source whose path is this Ariadne repo.
    from config import _PACKAGE_ROOT
    _tool_root = Path(_PACKAGE_ROOT).resolve()
    ignore = frozenset(name for name, p in known_sources.items() if Path(p).resolve() == _tool_root)
    detected = detect_dependencies(source_path, known_sources, ignore=ignore)
    if not detected:
        return

    # Show each detected dependency
    for dep in detected:
        evidence_lines = []
        for ev in dep.evidence:
            evidence_lines.append(f'  [cyan]{ev.file_path}:{ev.line_number}[/cyan]')
            evidence_lines.append(f'    [dim]{ev.import_statement}[/dim]')
            evidence_lines.append('')
        evidence_text = '\n'.join(evidence_lines).rstrip()
        note = _excluded_evidence_note(dep, set(cfg.resolve_excluded_dirs(source_name)))
        if note:
            evidence_text = f'{evidence_text}\n\n{note}'

        panel_content = (
            f'No dependency was explicitly configured between [bold]{source_name}[/bold]\n'
            f'and [bold]{dep.source_name}[/bold], but Ariadne detected a relationship based on\n'
            f'import analysis.\n\n'
            f'[bold]Evidence:[/bold]\n\n'
            f'{evidence_text}\n'
            f'If this is accurate and you wish to save this dependency for\n'
            f'future documentation generation, please approve.'
        )

        console.print()
        console.print(Panel(
            panel_content,
            title='Dependency Detection',
            border_style='yellow',
        ))
        console.print()

        if _prompt_yes_no(
            f'Save dependency [bold]{source_name}[/bold] -> '
            f'[bold]{dep.source_name}[/bold] to config?'
        ):
            current_deps = cfg.get_source_dependencies(source_name)
            if dep.source_name not in current_deps:
                new_deps = current_deps + [dep.source_name]
                if cfg.set_source_dependencies(source_name, new_deps):
                    console.print('[green]Saved dependency to ariadne.yaml[/green]')
                else:
                    console.print('[yellow]Could not save to config file[/yellow]')


def _prompt_yes_no(question: str) -> bool:
    """Prompt for a yes/no answer, returning True for ``y``/``Y`` and
    False for ``n``/``N``. Any other response is invalid; the prompt is
    shown again until one of those four is given.
    """
    while True:
        match console.input(f'{question} [y/n] '):
            case 'y' | 'Y':
                return True
            case 'n' | 'N':
                return False


def _scaffold_overhead_counter(model: str) -> 'Callable[[str], int | None]':
    """Per-doc-type prompt-scaffolding token counter (tiktoken of the real
    templates), cached per doc_type — for ``estimate_cost(prompt_overhead_for=)``.

    Shared by both dry-run surfaces. Returns ``None`` per doc_type when
    tiktoken is unavailable, so estimate_cost falls back to its flat
    PROMPT_OVERHEAD_TOKENS heuristic.
    """
    from docgen.prompts import static_scaffold
    from docgen.token_count import count_text_tokens

    cache: dict = {}

    def counter(doc_type: str) -> int | None:
        if doc_type not in cache:
            cache[doc_type] = count_text_tokens(static_scaffold(doc_type), model)
        return cache[doc_type]

    return counter
