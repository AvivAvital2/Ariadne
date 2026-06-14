"""Read-only catalog inspection CLI commands (``symbol``, ``list-file-scopes``,
``body``).

Extracted from cli/generation.py. Each prints a JSON lookup of catalog
elements; wired into the parser via this module's ``register_commands`` +
``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import TYPE_CHECKING

from config import get_config

if TYPE_CHECKING:
    from library import Library


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register read-only catalog inspection commands."""

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


HANDLERS = {
    'symbol': lambda args: cmd_symbol(args),
    'list-file-scopes': lambda args: cmd_list_file_scopes(args),
    'body': lambda args: cmd_body(args),
}
