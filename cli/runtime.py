"""CLI operation boundaries shared by nested command helpers."""

from __future__ import annotations

import argparse
from collections.abc import Awaitable, Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config import Config
    from library import Library

import asyncio
from contextlib import nullcontext
from pathlib import Path

from inference import InferenceContext, current_context

INSPECTION_COMMANDS = frozenset(
    {
        'list',
        'get',
        'export',
        'stats',
        'status',
        'config',
        'doctor',
        'usage',
        'symbol',
        'body',
        'list-file-scopes',
        'discover',
        'manifest',
    }
)
WRITE_COMMANDS = frozenset(
    {
        'add',
        'finding',
        'delete',
        'import',
        'rebuild',
        'tag',
        'generate',
        'sync',
        'merge',
        'improve',
        'topic',
        'onboard',
        'catalog-sync',
        'catalog-describe',
        'index',
        'migrate',
        'cleanup',
        'notify-changed',
        'vacuum',
        'teach',
    }
)


class ConcurrencyAction(argparse.Action):
    """Retain hosted parser defaults while distinguishing explicit overrides."""

    def __call__(
        self,
        parser: argparse.ArgumentParser,
        namespace: argparse.Namespace,
        values: Any,
        option_string: str | None = None,
    ) -> None:
        del parser, option_string
        setattr(namespace, self.dest, values)
        namespace.concurrency_explicit = True


def operation_for(args: argparse.Namespace) -> str:
    if getattr(args, 'dry_run', False) or args.command in {'dry-run', 'check'}:
        return 'plan'
    if args.command == 'onboard' and not getattr(args, 'approve', False):
        return 'plan'
    if args.command == 'graph':
        return 'write' if getattr(args, 'build', False) else 'inspect'
    if args.command == 'sync' and getattr(args, 'status', False):
        return 'inspect'
    if args.command == 'themes':
        return 'write' if getattr(args, 'themes_action', None) == 'build' else 'inspect'
    if args.command == 'batch':
        return 'write' if getattr(args, 'batch_action', None) in {'resume', 'clear'} else 'inspect'
    if args.command in INSPECTION_COMMANDS:
        return 'inspect'
    return 'write' if args.command in WRITE_COMMANDS else 'semantic'


def run_async(awaitable: Awaitable[Any]) -> Any:
    """Close operation-owned HTTP pools before this event loop goes away."""

    async def execute() -> Any:
        try:
            return await awaitable
        finally:
            context = current_context()
            if context is not None:
                await context.close_clients()

    return asyncio.run(execute())


def run_command(
    handler: Callable[[argparse.Namespace], int], args: argparse.Namespace, config: Config
) -> int:
    if args.command in {'export-db', 'import-db'}:
        # Slice transfer checks both databases itself. An inference context
        # binds a single library and must not initialize or stamp either file.
        return handler(args)
    concurrency = getattr(args, 'concurrency', None)
    if getattr(args, 'concurrency_explicit', None) is False:
        concurrency = None
    context = InferenceContext.from_config(
        config,
        db=getattr(args, 'db', None),
        staleness_db=getattr(args, 'staleness_db', None),
        model=getattr(args, 'model', None),
        provider=getattr(args, 'provider', None),
        concurrency=concurrency,
    )
    operation = operation_for(args)
    if not context.local and args.command in {'dry-run', 'onboard'}:
        operation = 'write'  # Hosted previews retain structural preparation.
    if hasattr(args, 'concurrency') and concurrency is None:
        args.concurrency = (
            1
            if context.local
            else (4 if args.command in {'catalog-sync', 'catalog-describe'} else 3)
        )
    context.operation = operation
    args.db = context.db_path
    args.staleness_db = context.staleness_db_path
    with context.activate():
        try:
            if context.local and operation == 'plan':
                return local_preview(args, config, context)
            if (
                context.local
                and args.command == 'batch'
                and getattr(args, 'batch_action', None) in {'resume', 'clear'}
            ):
                from docgen.spool import SpoolStore

                store = SpoolStore(context)
                store.get_run(args.batch_id)
                if args.batch_action == 'resume':
                    store.jobs(args.batch_id)  # Version checks precede schema writes.
            with context.write_session() if operation == 'write' else nullcontext():
                if operation == 'write':
                    context.open_library(operation='write').close()
                return handler(args)
        finally:
            asyncio.run(context.aclose())


def get_library(db_path: Path | str | None = None) -> Library:
    from config import get_config
    from library import Library

    context = current_context()
    if context is not None:
        if db_path is not None and Path(db_path).resolve() != context.db_path:
            raise ValueError('Nested command requested a different library path')
        return context.open_library(
            operation=getattr(context, 'operation', 'write' if context.session else 'semantic')
        )
    return Library(Path(db_path or get_config().db_path))


def local_preview(args: argparse.Namespace, config: Config, context: InferenceContext) -> int:
    """Deterministic, read-only local estimate. Never open a writer or tokenizer."""
    from rich.console import Console

    from docgen.staleness import find_catalog_files

    console = Console()
    source = getattr(args, 'source', None) or config.default_source
    root = config.resolve_source(source)
    if root is None or not root.exists():
        raise ValueError('A valid source is required for a local estimate')
    sc = config.get_source_config(source)
    files = find_catalog_files(
        root,
        exclude_patterns=sc.exclude if sc else (),
        exclude_dir_names=config.resolve_excluded_dirs(source),
    )
    target = getattr(args, 'path', None)
    if target:
        target = (root / target).resolve()
        files = [path for path in files if path == target or target in path.parents]
    types = tuple(
        (getattr(args, 'types', None) or 'explanation,architecture,qa,gotcha,diagram').split(',')
    )
    from docgen.catalog_extractor import _detect_language
    from docgen.prompts import filter_doc_types_for_language

    calls = sum(
        len(filter_doc_types_for_language(types, _detect_language(path), override=config.source_doc_types_by_language(source)))
        for path in files
    )
    tokens = sum(len(path.read_bytes()) // 4 for path in files)
    console.print(
        f'Local preview: {len(files)} source files; approximately {calls} generation calls; {tokens} source tokens (character estimate).'
    )
    console.print('Provider token charges: $0. Hardware/runtime cost: unestimated.')
    console.print(
        'Catalog, dependency and theme work is an incomplete estimate until structural artifacts are available.'
    )
    console.print(f'Library: {context.db_path}\nStaleness: {context.staleness_db_path}')
    return 0
