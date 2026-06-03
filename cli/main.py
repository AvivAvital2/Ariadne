"""Command-line interface for Ariadne.

This module provides CLI commands for managing the documentation library.
Commands are organized into domain modules:
- cli_core: CRUD, stats, metadata
- cli_generation: generate, sync, merge, improve
- cli_integration: init, manifest, config, MCP
- cli_analysis: analysis, review, explain commands
- cli_debug: diagnose, debug, test, teach commands
- cli_health: lint, debt, doctor, trends, ROI commands
- cli_graph: dependency graph commands

Usage:
    ariadne search "How does caching work?"
    ariadne list --type explanation
    ariadne generate
    ariadne config
"""
from __future__ import annotations

import argparse
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

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Register commands from each submodule
    from cli.analysis import register_commands as register_analysis
    from cli.callers import register_commands as register_callers
    from cli.core import register_commands as register_core
    from cli.debug import register_commands as register_debug
    from cli.generation import register_commands as register_generation
    from cli.graph import register_commands as register_graph
    from cli.health import register_commands as register_health
    from cli.integration import register_commands as register_integration
    from cli.trace import register_commands as register_trace

    register_core(subparsers)
    register_generation(subparsers)
    register_integration(subparsers)
    register_analysis(subparsers)
    register_debug(subparsers)
    register_health(subparsers)
    register_graph(subparsers)
    register_callers(subparsers)
    register_trace(subparsers)

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


def main() -> int:
    """Main entry point for the CLI."""
    _load_env()

    parser = create_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        return 0

    # Assemble handler dict from submodules
    from cli.analysis import HANDLERS as ANALYSIS_HANDLERS
    from cli.callers import HANDLERS as CALLERS_HANDLERS
    from cli.core import HANDLERS as CORE_HANDLERS
    from cli.debug import HANDLERS as DEBUG_HANDLERS
    from cli.generation import HANDLERS as GEN_HANDLERS
    from cli.graph import HANDLERS as GRAPH_HANDLERS
    from cli.health import HANDLERS as HEALTH_HANDLERS
    from cli.integration import HANDLERS as INTEGRATION_HANDLERS
    from cli.trace import HANDLERS as TRACE_HANDLERS

    handlers: dict = {}
    handlers.update(CORE_HANDLERS)
    handlers.update(GEN_HANDLERS)
    handlers.update(INTEGRATION_HANDLERS)
    handlers.update(ANALYSIS_HANDLERS)
    handlers.update(DEBUG_HANDLERS)
    handlers.update(HEALTH_HANDLERS)
    handlers.update(GRAPH_HANDLERS)
    handlers.update(CALLERS_HANDLERS)
    handlers.update(TRACE_HANDLERS)

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
