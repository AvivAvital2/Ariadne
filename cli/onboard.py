"""End-to-end onboarding command (``onboard``) and its interactive prompts.

Extracted from cli/generation.py. Runs the free phases + cost preview (via
cmd_dry_run), then — on --approve or a yes — the paid phases (catalog-describe,
generate, themes), with interactive doc-type / dependency prompts. Excludes and
the staleness-exemption choice are made in the file browser (the exclude
explorer), which onboard always opens on a TTY.
Wired into the parser via this module's ``register_commands`` + ``HANDLERS``
(assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console

from cli.catalog import cmd_catalog_describe
from cli.dry_run import _PhaseUI, cmd_dry_run
from cli.generate import cmd_generate
from cli.themes_cmd import cmd_themes_build

if TYPE_CHECKING:
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


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


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register the onboard command."""

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
    onboard_parser.add_argument('--doc-types-off', default=None,
        help='Comma-separated doc types to leave UNchecked by default in '
             'the interactive picker (still selectable; excluded on a '
             'non-TTY). Spool builds pass architecture,qa,diagram.')
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

    # ---- Mark which other configured sources this project depends on
    # (TTY only, not --approve). Persists depends_on to ariadne.yaml so the
    # paid generate phase loads those sources' docs as context. ----
    if not getattr(args, 'approve', False):
        _offer_dependency_detection(cfg, source_name)

    # ---- Always open the file browser before the preview (TTY, not --approve)
    # The browser owns ariadne.yaml configuration: the user reviews + excludes
    # expensive paths there, and answers the staleness question as an Apply-time
    # pop-up — rather than via pre-browser CLI prompts. The preview the
    # proceed-prompt then acts on already reflects the chosen excludes, and the
    # paid generate phase honors them via ariadne.yaml.
    import sys
    if not getattr(args, 'approve', False) and (
        sys.stdin.isatty() and sys.stdout.isatty()
    ):
        args.interactive = True       # always review in the browser
        args.offer_staleness = True   # explorer pops the staleness modal on Apply

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
    # whose per-type cost the preview just showed. --doc-types-off pre-
    # unchecks types in the picker (opt-in) and drops them on a non-TTY;
    # spool builds pass architecture,qa,diagram to default a leaner pack.
    from cli.generate import DEFAULT_GENERATE_DOC_TYPES
    explicit_types = getattr(args, 'types', None)
    if explicit_types:
        selected_doc_types = tuple(
            t.strip() for t in explicit_types.split(',') if t.strip()
        )
    else:
        off_raw = getattr(args, 'doc_types_off', None)
        off_types = frozenset(
            t.strip() for t in off_raw.split(',') if t.strip()
        ) if off_raw else frozenset()
        selected_doc_types = _select_generate_doc_types(
            DEFAULT_GENERATE_DOC_TYPES, off_types,
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
    # (label, success_line, invoke, fatal). fatal=False phases warn and
    # continue on failure — themes are a semantic-clustering augmentation, so
    # a themes failure must not discard a completed (and, for a spool, paid)
    # generate + embed run at the last step.
    phases: list[tuple[str, str, object, bool]] = [
        (
            'Describing catalog elements',
            '  [green]✓[/green] Catalog-describe — descriptions persisted',
            lambda: cmd_catalog_describe(catalog_describe_args),
            True,
        ),
        (
            'Generating documentation',
            '  [green]✓[/green] Generate — explanation/architecture/qa docs written',
            lambda: cmd_generate(generate_args),
            True,
        ),
    ]
    phases.append((
        'Building themes',
        '  [green]✓[/green] Themes build — cluster summaries written',
        lambda: cmd_themes_build(themes_args),
        False,
    ))

    import inspect

    for label, success_line, invoke, fatal in phases:
        with ui.passthrough(label):
            result = invoke()
            rc = await result if inspect.isawaitable(result) else result
        if rc != 0:
            if fatal:
                console.print(
                    f'[red]Phase {label!r} failed (rc={rc}). Pipeline '
                    'stopped.[/red]',
                )
                return rc
            # Non-fatal (e.g. themes): surface loudly but keep the completed
            # docs/embeddings — they're persisted and the pack can still build.
            console.print(
                f'[yellow]Phase {label!r} failed (rc={rc}) — continuing; it '
                'augments the docs and the generated docs/pack are '
                'unaffected.[/yellow]',
            )
            continue
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


def _prompt_for_batch_mode(
    options: tuple = _BATCH_MODE_OPTIONS,
    title: str = 'LLM mode (catalog-describe + generate)',
) -> str:
    """Interactively pick live vs batched work.

    Uses arrow-key selection on a TTY (no typing required). Falls back
    to a text prompt when stdin isn't a TTY (CI, piped input). Returns
    the selected option value (``'live'`` or ``'batch'``).
    """
    import sys

    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            return _arrow_key_select(options, title=title)
        except Exception:
            # Terminal doesn't support raw mode (e.g., minimal containers).
            # Fall through to the typed prompt rather than crash.
            pass

    console.print()
    console.print(f'[bold]{title}[/bold]')
    for _, label, desc in options:
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
            row.append(f'  — {desc}')
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
    console.print(f'[bold]{title}[/bold]')
    for i, (_, label, desc) in enumerate(options):
        if i == idx:
            console.print(f'  ▶ [cyan]{label.lower()}[/cyan] — {desc}')
        else:
            console.print(f'    {label.lower()} — {desc}')
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


def _select_generate_doc_types(
    default_types: tuple[str, ...],
    off_types: frozenset[str] = frozenset(),
) -> tuple[str, ...]:
    """Let the user pick which doc types ``generate`` should produce.

    All ``default_types`` are listed; those in ``off_types`` start
    UNchecked (opt-in) — spool builds default architecture/qa/diagram off
    for a leaner pack. A non-interactive context (no TTY) or a
    cancelled/empty selection returns the on-by-default set
    (``default_types`` minus ``off_types``) — so we never silently
    generate nothing.
    """
    import sys

    on_default = tuple(t for t in default_types if t not in off_types)
    options = [(t, t) for t in default_types]
    preselected = {i for i, t in enumerate(default_types) if t not in off_types}
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return on_default
    try:
        chosen = _arrow_key_multiselect(
            options,
            title='Doc types to generate (per-type cost shown above)',
            selected=preselected,
        )
    except Exception:
        return on_default
    return tuple(chosen) if chosen else on_default


def _dependency_candidates(cfg, source_name: str) -> list[str]:
    """Other configured sources that ``source_name`` could depend on.

    Excludes the source itself and any source whose path is Ariadne's own
    repo — its name collides with common packages and a documented project
    never depends on the doc tool (the same rule the import auto-detector
    uses). Sorted for a stable prompt order.
    """
    from config import _PACKAGE_ROOT

    tool_root = Path(_PACKAGE_ROOT).resolve()
    candidates = []
    for name, path in cfg.get_all_source_paths().items():
        if name == source_name:
            continue
        try:
            if Path(path).resolve() == tool_root:
                continue
        except OSError:
            pass
        candidates.append(name)
    return sorted(candidates)


def _select_onboard_dependencies(cfg, source_name: str) -> None:
    """Interactively mark which other sources ``source_name`` depends on and
    persist the choice to ariadne.yaml.

    Shows a checklist of the other configured sources, pre-checked with the
    source's current ``depends_on`` so it doubles as an editor; the result is
    written back via ``set_source_dependencies`` (so the paid generate phase
    loads those sources' docs as context). No-op when there are no eligible
    sources, or in a non-TTY / non-POSIX context, or if the picker is
    cancelled (q / Ctrl-C) — onboarding continues either way.
    """
    import sys

    candidates = _dependency_candidates(cfg, source_name)
    if not candidates:
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    current = list(cfg.get_source_dependencies(source_name))
    options = [(name, name) for name in candidates]
    preselected = {i for i, name in enumerate(candidates) if name in current}
    try:
        chosen = _arrow_key_multiselect(
            options,
            title=f'Which sources does {source_name} depend on?',
            selected=preselected,
        )
    except (KeyboardInterrupt, Exception):
        return  # cancelled / non-POSIX terminal — leave config untouched

    if chosen == current:
        return
    if not cfg.set_source_dependencies(source_name, chosen):
        console.print('[yellow]Could not save dependencies to config.[/yellow]')
    elif chosen:
        console.print(
            f'[green]Saved dependencies for {source_name}: '
            f'{", ".join(chosen)}[/green]',
        )
    else:
        console.print(f'[green]Cleared dependencies for {source_name}.[/green]')


def _offer_dependency_detection(cfg, source_name: str) -> None:
    """Interactive onboard gate that makes cross-source dependency
    detection optional.

    Detection scans the onboarded repo for imports referencing the
    project's OTHER configured sources to infer a hidden ``depends_on``.
    This asks once whether to do that: a 'no' persists
    ``skip_dependency_detection: true`` (so the later generate phase's
    import scan is skipped too) and returns; a 'yes' opens the manual
    picker. Skipped entirely when the source already opted out, has no
    candidate sources, or in a non-TTY context.
    """
    import sys

    if cfg.source_skip_dependency_detection(source_name):
        return
    if not _dependency_candidates(cfg, source_name):
        return
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return

    if _prompt_detect_dependencies(source_name):
        _select_onboard_dependencies(cfg, source_name)
        return
    if cfg.set_source_config(source_name, skip_dependency_detection=True):
        console.print(
            f'[green]Skipping dependency detection for {source_name} '
            f'(saved skip_dependency_detection: true).[/green]'
        )
    else:
        console.print(
            '[yellow]Could not save skip_dependency_detection to config.[/yellow]'
        )


def _prompt_detect_dependencies(source_name: str) -> bool:
    """Ask whether to scan for hidden dependencies linking
    ``source_name`` to the other configured sources. Defaults to yes; a
    closed stdin (``EOFError``) also returns yes, preserving prior
    behavior."""
    try:
        resp = input(
            f'Scan for hidden dependencies linking {source_name} to your '
            f'other sources? [Y/n]: '
        )
    except EOFError:
        return True
    return resp.strip().lower() in ('', 'y', 'yes')


HANDLERS = {
    'onboard': lambda args: asyncio.run(cmd_onboard(args)),
}
