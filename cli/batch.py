"""``ariadne batch`` command — manage pending Anthropic batches and local runs.

A pending batch is a Message Batches API submission whose results
haven't been fetched yet (typically because the user's machine
crashed mid-poll, or they ``Ctrl-C``'d). The orchestrator's
``run()`` auto-resumes when ``find_pending_batch`` matches, so this
CLI is for the cases where auto-resume isn't enough:

- ``list`` — see what's in flight or orphaned.
- ``clear`` — drop a row by ``batch_id`` (e.g. before re-submitting
  with different config).

A hosted ``status`` subcommand (single-shot poll of the Anthropic side) is
deliberately deferred — it requires a single-poll method on
``AnthropicProvider`` that doesn't exist yet, and ``ariadne generate``
auto-resumes anyway.

Local runs support offline list/status, bounded resume, explicit
cancellation and terminal cleanup in the bound main database.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from inference import InferenceContext

import argparse
from pathlib import Path

from rich.console import Console

from config import get_config
from docgen.staleness import StalenessTracker

console = Console()


def register_batch_parser(subparsers: argparse._SubParsersAction) -> None:
    """Register ``ariadne batch`` and its sub-subparsers."""
    batch_parser = subparsers.add_parser(
        'batch',
        help='Manage pending batches (list / clear) and durable local runs',
    )
    batch_subparsers = batch_parser.add_subparsers(
        dest='batch_action', required=True,
    )
    batch_subparsers.add_parser(
        'list', help='List all pending batches',
    )
    clear_parser = batch_subparsers.add_parser(
        'clear', help='Drop a pending batch by id',
    )
    clear_parser.add_argument('batch_id', help='Provider batch ID or local: run ID')
    for action in ('status', 'resume', 'cancel'):
        parser = batch_subparsers.add_parser(action, help=f'{action.title()} a local run')
        parser.add_argument('batch_id', help='local: run ID')


def cmd_batch(args: argparse.Namespace) -> int:
    """Dispatcher for ``ariadne batch <action>``."""
    action = getattr(args, 'batch_action', None)
    from inference import current_context

    context = current_context()
    if (context is not None and context.local) or str(getattr(args, 'batch_id', '')).startswith(
        'local:'
    ):
        return _local_batch(args, context)
    if action == 'list':
        return cmd_batch_list(args)
    if action == 'clear':
        return cmd_batch_clear(args)
    console.print(
        '[yellow]Usage: ariadne batch (list | clear <id>)[/yellow]',
    )
    return 1


def _local_batch(args: argparse.Namespace, context: InferenceContext | None) -> int:
    from cli.runtime import run_async
    from docgen.local_pipeline import resume_run
    from docgen.spool import SpoolStore

    if context is None or not context.local:
        raise ValueError(
            'Select the local library and its model configuration to manage a local run'
        )
    action = args.batch_action
    library = context.open_library(operation='write') if action in {'resume', 'clear'} else None
    store = SpoolStore(context, library)
    if action == 'list':
        runs = store.list_runs()
        if not runs:
            console.print('No local runs.')
        for run in runs:
            console.print(
                f'{run["run_id"]}  {run["operation_kind"]}  {run["state"]}  phase={run["phase"]}'
            )
        return 0
    run_id = args.batch_id
    if not run_id.startswith('local:'):
        raise ValueError('Local configurations require a local: run ID')
    if action == 'cancel':
        store.cancel(run_id)
        console.print(f'Cancelled {run_id}; already applied documents are preserved.')
        return 0
    if action == 'clear':
        store.clear(run_id)
        console.print(f'Cleared {run_id}; applied documents are preserved.')
        return 0
    if action == 'resume':
        result = run_async(resume_run(library, run_id))
    else:
        result = {**store.counts(run_id), 'state': store.get_run(run_id)['state']}
    console.print(
        f'{run_id}: {result["state"]}; {result["responses"]} responses saved; {result["applied"]}/{result["total"]} documents applied; {result["failed"]} failed; {result["obsolete"]} obsolete'
    )
    return int(action == 'resume' and result['state'] != 'completed')


def cmd_batch_list(args: argparse.Namespace) -> int:
    """List all pending batches in the staleness DB.

    One row per batch: ``batch_id  submitted_at  config=<prefix>``.
    Empty state prints a friendly message rather than nothing — so
    a typo or fresh setup doesn't look like the command silently
    failed.
    """
    cfg = get_config()
    path = Path(getattr(args, 'staleness_db', None) or cfg.staleness_db_path)
    if not path.exists():
        console.print('[dim]No pending batches.[/dim]')
        return 0
    staleness = StalenessTracker(path, read_only=True)
    try:
        batches = staleness.list_pending_batches()
    finally:
        staleness.close()

    if not batches:
        console.print('[dim]No pending batches.[/dim]')
        return 0

    for b in batches:
        # Truncate config_hash for display — full SHA256 is noise.
        console.print(
            f'[cyan]{b.batch_id}[/cyan]  '
            f'submitted={b.submitted_at}  '
            f'config={b.config_hash[:12]}...',
        )
    return 0


def cmd_batch_clear(args: argparse.Namespace) -> int:
    """Drop a pending batch row by ``batch_id``.

    Returns 0 on success (row existed and was removed), 1 on miss
    (no such row). The non-zero exit-on-miss is deliberate: a typo
    or already-cleared id should look different from a successful
    no-op so users notice.
    """
    cfg = get_config()
    staleness = StalenessTracker(Path(getattr(args, 'staleness_db', None) or cfg.staleness_db_path))
    try:
        cleared = staleness.clear_pending_batch(args.batch_id)
    finally:
        staleness.close()

    if cleared:
        console.print(
            f'[green]Cleared pending batch {args.batch_id}[/green]',
        )
        return 0
    console.print(
        f'[yellow]No such pending batch: {args.batch_id}[/yellow]',
    )
    return 1
