"""Command-line interface for Ariadne.

This module provides CLI commands for managing the documentation library.
Commands live in per-domain submodules of this package (cli/core.py,
cli/generation.py, cli/integration.py, …); each exposes ``register_commands``
and ``HANDLERS``, which ``create_parser()`` and ``main()`` assemble into the
full parser and dispatch table.

Usage:
    ariadne search "How does caching work?"
    ariadne list --type explanation
    ariadne generate
    ariadne config
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from rich.console import Console

from config import get_config

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


def create_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser, delegating to submodules."""
    parser = argparse.ArgumentParser(
        prog='ariadne',
        description='Documentation library management with config-based source paths.',
    )
    parser.add_argument(
        '--db',
        type=Path,
        default=None,
        help='Path to database file (default from config or ariadne.db)',
    )
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Verbose logging: API retry/backoff chatter, request details',
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register commands from each submodule
    from cli.analysis import register_commands as register_analysis
    from cli.callers import register_commands as register_callers
    from cli.catalog import register_commands as register_catalog
    from cli.core import register_commands as register_core
    from cli.debug import register_commands as register_debug
    from cli.dry_run import register_commands as register_dry_run
    from cli.generation import register_commands as register_generation
    from cli.graph import register_commands as register_graph
    from cli.health import register_commands as register_health
    from cli.index import register_commands as register_index
    from cli.integration import register_commands as register_integration
    from cli.lookup import register_commands as register_lookup
    from cli.maintenance import register_commands as register_maintenance
    from cli.onboard import register_commands as register_onboard
    from cli.status import register_commands as register_status
    from cli.sync import register_commands as register_sync
    from cli.themes_cmd import register_commands as register_themes_cmd
    from cli.trace import register_commands as register_trace
    from cli.spools_cmd import register_commands as register_spools

    register_core(subparsers)
    register_index(subparsers)
    register_generation(subparsers)
    register_dry_run(subparsers)
    register_onboard(subparsers)
    register_themes_cmd(subparsers)
    register_catalog(subparsers)
    register_status(subparsers)
    register_sync(subparsers)
    register_maintenance(subparsers)
    register_integration(subparsers)
    register_lookup(subparsers)
    register_analysis(subparsers)
    register_debug(subparsers)
    register_health(subparsers)
    register_graph(subparsers)
    register_callers(subparsers)
    register_trace(subparsers)
    register_spools(subparsers)

    return parser


def _load_env() -> None:
    """Best-effort load of a local ``.env``.

    python-dotenv is only a convenience for picking up API keys from a
    ``.env`` file. If the environment running the ``ariadne`` entry point
    doesn't have it installed (e.g. a ``uv tool install`` shim), skip
    silently rather than crashing the whole CLI before any command runs.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _configure_logging(debug: bool) -> None:
    """Root logging: WARNING by default; --debug reveals retry/backoff chatter."""
    level = logging.DEBUG if debug else logging.WARNING
    logging.basicConfig(level=level)
    logging.getLogger().setLevel(level)
    # httpx/httpcore log every request at INFO ("HTTP Request: POST
    # .../embeddings 200 OK"). Under --debug that per-request stream floods the
    # console and tears up the Rich progress bar; Ariadne's own retry/backoff
    # logs are separate, so keep these two at WARNING regardless of --debug.
    for _noisy in ('httpx', 'httpcore'):
        logging.getLogger(_noisy).setLevel(logging.WARNING)


def main() -> int:
    """Main entry point for the CLI."""
    _load_env()

    parser = create_parser()
    args = parser.parse_args()
    _configure_logging(args.debug)

    if args.command is None:
        parser.print_help()
        return 0

    # Assemble handler dict from submodules
    from cli.analysis import HANDLERS as ANALYSIS_HANDLERS
    from cli.callers import HANDLERS as CALLERS_HANDLERS
    from cli.catalog import HANDLERS as CATALOG_HANDLERS
    from cli.core import HANDLERS as CORE_HANDLERS
    from cli.debug import HANDLERS as DEBUG_HANDLERS
    from cli.dry_run import HANDLERS as DRY_RUN_HANDLERS
    from cli.generation import HANDLERS as GEN_HANDLERS
    from cli.graph import HANDLERS as GRAPH_HANDLERS
    from cli.health import HANDLERS as HEALTH_HANDLERS
    from cli.index import HANDLERS as INDEX_HANDLERS
    from cli.integration import HANDLERS as INTEGRATION_HANDLERS
    from cli.lookup import HANDLERS as LOOKUP_HANDLERS
    from cli.maintenance import HANDLERS as MAINTENANCE_HANDLERS
    from cli.onboard import HANDLERS as ONBOARD_HANDLERS
    from cli.status import HANDLERS as STATUS_HANDLERS
    from cli.sync import HANDLERS as SYNC_HANDLERS
    from cli.themes_cmd import HANDLERS as THEMES_CMD_HANDLERS
    from cli.trace import HANDLERS as TRACE_HANDLERS
    from cli.spools_cmd import HANDLERS as SPOOLS_HANDLERS

    handlers: dict = {}
    handlers.update(CORE_HANDLERS)
    handlers.update(INDEX_HANDLERS)
    handlers.update(GEN_HANDLERS)
    handlers.update(DRY_RUN_HANDLERS)
    handlers.update(ONBOARD_HANDLERS)
    handlers.update(THEMES_CMD_HANDLERS)
    handlers.update(CATALOG_HANDLERS)
    handlers.update(STATUS_HANDLERS)
    handlers.update(SYNC_HANDLERS)
    handlers.update(MAINTENANCE_HANDLERS)
    handlers.update(INTEGRATION_HANDLERS)
    handlers.update(LOOKUP_HANDLERS)
    handlers.update(ANALYSIS_HANDLERS)
    handlers.update(DEBUG_HANDLERS)
    handlers.update(HEALTH_HANDLERS)
    handlers.update(GRAPH_HANDLERS)
    handlers.update(CALLERS_HANDLERS)
    handlers.update(TRACE_HANDLERS)
    handlers.update(SPOOLS_HANDLERS)

    handler = handlers.get(args.command)
    if handler is None:
        console.print(f'[red]Unknown command: {args.command}[/red]')
        return 1

    try:
        return handler(args)
    except KeyboardInterrupt:
        console.print('\n[yellow]Interrupted.[/yellow]')
        return 130
    except Exception as e:
        console.print(f'[red]Error: {e}[/red]')
        return 1


if __name__ == '__main__':
    sys.exit(main())
