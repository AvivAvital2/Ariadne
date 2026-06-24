"""Theme-cluster (``themes build|list|show``) and prompt-diff (``diff-docs``)
CLI commands.

Extracted from cli/generation.py to keep that module focused. These are
independent leaf commands wired into the parser via this module's
``register_commands`` + ``HANDLERS`` (assembled in cli/main.py).
"""
from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path
from typing import TYPE_CHECKING

from rich.console import Console
from rich.table import Table

from config import get_config

if TYPE_CHECKING:
    from library import Library

console = Console()


def get_library(db_path: Path | None = None) -> 'Library':
    from cli.core import get_library
    return get_library(db_path)


def register_commands(subparsers: argparse._SubParsersAction) -> None:
    """Register theme-cluster and prompt-diff commands."""

    # themes (Phase 7) — nested subcommands: build / list / show
    themes_parser = subparsers.add_parser(
        'themes',
        help='Manage cross-cutting theme clusters (Leiden community detection)',
    )
    themes_sub = themes_parser.add_subparsers(
        dest='themes_action',
        help='Themes subcommand',
    )

    themes_build = themes_sub.add_parser(
        'build', help='Discover/refresh themes for the configured source',
    )
    themes_build.add_argument('--source', '-s', default=None,
        help='Source name (default from config)')
    themes_build.add_argument('--batch', action='store_true',
        help='Summarize dirty themes via the provider batch API '
             '(~50%% off, up to 24h) instead of the live per-theme path')
    themes_build.add_argument('--concurrency', '-c', type=int, default=3, help='Max concurrent LLM calls for theme summarization')

    themes_list = themes_sub.add_parser(
        'list', help='List discovered themes',
    )
    themes_list.add_argument('--source', '-s', default=None,
        help='Filter by source name')
    themes_list.add_argument('--include-incoherent', action='store_true',
        help='Include themes the LLM marked INCOHERENT')
    themes_list.add_argument('--limit', type=int, default=50,
        help='Maximum number of themes to print (default: 50)')

    themes_show = themes_sub.add_parser(
        'show', help='Show full theme doc content for a cluster',
    )
    themes_show.add_argument('cluster_id',
        help='Stable cluster_id (printed by `themes list`)')

    # diff-docs (Catalog transition Phase 3.2) — side-by-side legacy vs
    # new prompts for spot-checking during the dual-run window.
    diff_parser = subparsers.add_parser(
        'diff-docs',
        help='Compare legacy vs catalog-driven generation prompts for a file',
    )
    diff_parser.add_argument('--file', '-f', required=True,
        help='Path to a Python file to diff (must exist)')
    diff_parser.add_argument('--type', '-t', default='explanation',
        help='Doc type to compare (default: explanation)')
    diff_parser.add_argument('--with-llm', action='store_true',
        help='Also run both pipelines with the real LLM and print outputs')


async def cmd_themes_build(args: argparse.Namespace) -> int:
    """`ariadne themes build` — run refresh_themes for the configured source.

    This is the awaitable core: it ``await``s ``refresh_themes`` directly
    rather than spinning its own event loop, so it composes into a caller
    that is already inside one (the async ``onboard`` pipeline awaits this
    as a phase). The standalone CLI dispatch goes through the sync
    ``_cmd_themes_build`` wrapper below, which supplies the event loop.
    """
    from cli.progress import make_progress

    from docgen.themes import refresh_themes
    from writer import LibraryWriter

    cfg = get_config()
    batch_strategy = None
    if getattr(args, 'batch', False):
        from cli.generate import resolve_batch_strategy

        batch_strategy, batch_provider, key_env = resolve_batch_strategy(
            args, cfg,
        )
        if batch_strategy is None:
            console.print(
                f'[red]{key_env} is not set. Export it before using '
                f'`themes build --batch` with the {batch_provider} '
                f'provider.[/red]',
            )
            return 1

    library = get_library(getattr(args, 'db', None))
    try:
        async def _build() -> dict:
            with make_progress(console=console) as progress:
                task_id = progress.add_task(
                    'Themes: clustering & summarizing', total=0,
                )

                def on_progress(
                    completed: int, total: int, cluster_id: str | None,
                ) -> None:
                    short = (cluster_id or '')[:14]
                    desc = (
                        f'Themes: summarizing [dim]{short}[/dim]'
                        if short else 'Themes: done'
                    )
                    progress.update(
                        task_id,
                        completed=completed,
                        total=total if total > 0 else None,
                        description=desc,
                    )

                def on_stage(stage: str, completed: int, total: int) -> None:
                    # Batch path emits coarse stage transitions (submit ->
                    # processing -> download -> apply) so the long poll wait
                    # isn't a frozen 0/0 bar.
                    labels = {
                        'submit': 'Themes: submitting batch',
                        'processing': 'Themes: processing at provider',
                        'download': 'Themes: downloading results',
                        'apply': 'Themes: applying summaries',
                    }
                    progress.update(
                        task_id,
                        description=labels.get(stage, f'Themes: {stage}'),
                        completed=completed,
                        total=total if total > 0 else None,
                    )

                summarize_kwargs = {'on_progress': on_progress}
                summarize_kwargs['concurrency'] = getattr(args, 'concurrency', 3)
                if batch_strategy is not None:
                    summarize_kwargs['on_stage'] = on_stage

                async with LibraryWriter(library) as writer:
                    return await refresh_themes(
                        library, writer,
                        enabled=getattr(cfg, 'themes_enabled', True),
                        summarize_kwargs=summarize_kwargs,
                        batch_strategy=batch_strategy,
                    )

        try:
            summary = await _build()
        except Exception as e:
            # Surface the full traceback, not just the message — a bare
            # ``{e}`` (e.g. "'<' not supported between 'NoneType' and 'int'")
            # hides which call raised it, leaving the failure undiagnosable.
            import traceback
            console.print(f'[red]Themes build failed: {e}[/red]')
            console.print('[red]Traceback:[/red]')
            console.print(traceback.format_exc(), soft_wrap=True)
            return 1

        path = summary.get('path', '?')
        console.print()
        console.print(f'Themes refresh: [cyan]{path}[/cyan]')
        console.print(f'  Changed:    {summary.get("changed", 0)}')
        console.print(f'  Summarized: [green]{summary.get("summarized", 0)}[/green]')
        console.print(f'  Incoherent: {summary.get("incoherent", 0)}')
        if summary.get('failed', 0):
            console.print(f'  [red]Failed: {summary["failed"]}[/red]')
        if summary.get('quota_exhausted'):
            console.print(
                '  [yellow]⚠ Theme summaries stopped — Anthropic API usage '
                'cap reached:[/yellow]',
            )
            console.print(f'    [dim]{summary.get("quota_message", "")}[/dim]')
            console.print(
                '    [dim]Docs were generated; re-run [bold]ariadne themes '
                'build[/bold] once the cap resets to finish summaries.[/dim]',
            )
        return 0
    finally:
        library.close()


def _cmd_themes_build(args: argparse.Namespace) -> int:
    """Synchronous wrapper for the standalone ``ariadne themes build``
    dispatch — supplies the event loop that :func:`cmd_themes_build`
    expects a caller to provide."""
    return asyncio.run(cmd_themes_build(args))


def _cmd_themes_list(args: argparse.Namespace) -> int:
    """`ariadne themes list` — print a table of discovered themes."""
    from ariadne_mcp.service_themes import themes_action

    library = get_library(getattr(args, 'db', None))
    try:
        coherent_only = not getattr(args, 'include_incoherent', False)
        result = themes_action(
            library,
            action='list',
            coherent_only=coherent_only,
            source=getattr(args, 'source', None),
            limit=getattr(args, 'limit', 50),
        )

        themes = result.get('themes', [])
        total = result.get('total', len(themes))
        if not themes:
            console.print('[yellow]No themes found.[/yellow] '
                          'Run [bold]ariadne themes build[/bold] to discover them.')
            return 0

        table = Table(title=f'Themes ({len(themes)} of {total})')
        table.add_column('cluster_id', style='cyan', no_wrap=True)
        table.add_column('title', style='bold')
        table.add_column('members', justify='right')
        table.add_column('coherent', justify='center')
        table.add_column('dirty', justify='center')

        for t in themes:
            table.add_row(
                t.get('cluster_id', '')[:14],
                str(t.get('title', '') or ''),
                str(t.get('member_count', 0)),
                'yes' if t.get('coherent') else 'no',
                'yes' if t.get('dirty') else 'no',
            )

        console.print(table)
        return 0
    finally:
        library.close()


def _cmd_themes_show(args: argparse.Namespace) -> int:
    """`ariadne themes show <cluster_id>` — print the full theme doc."""
    from ariadne_mcp.service_themes import themes_action

    cluster_id = getattr(args, 'cluster_id', None)
    if not cluster_id:
        console.print('[red]Missing cluster_id.[/red]')
        return 1

    library = get_library(getattr(args, 'db', None))
    try:
        result = themes_action(
            library, action='get', cluster_id=cluster_id,
        )
        if 'error' in result:
            console.print(f'[red]{result["error"]}[/red]')
            return 1

        title = result.get('title') or f'Theme {cluster_id[:12]}'
        content = result.get('content') or ''
        console.print(f'[bold cyan]{title}[/bold cyan] '
                      f'(cluster_id={cluster_id}, members={result.get("member_count", 0)})')
        console.print()
        console.print(content)
        return 0
    finally:
        library.close()


def cmd_diff_docs(args: argparse.Namespace) -> int:
    """Side-by-side comparison of legacy and catalog-driven prompts.

    Runs both pipelines (mocked LLM unless --with-llm) and prints the
    user prompts each path would have sent to the LLM. Useful for
    spot-checking parity during the dual-run window.
    """
    from unittest.mock import AsyncMock, patch

    from docgen._legacy_analyzer import SourceAnalyzer
    from docgen.catalog_enrich import enrich_file
    from docgen.generator import DocGenerator, GeneratorConfig

    file_path = Path(args.file)
    if not file_path.exists():
        console.print(f'[red]File not found: {file_path}[/red]')
        return 1
    if file_path.suffix != '.py':
        console.print(
            f'[red]diff-docs only supports Python files (legacy path is '
            f'Python-only); got: {file_path.suffix}[/red]'
        )
        return 1

    doc_type = getattr(args, 'type', 'explanation')
    with_llm = getattr(args, 'with_llm', False)

    cfg = get_config()

    async def _run() -> tuple[str, str, str | None, str | None]:
        legacy_user: str = ''
        new_user: str = ''
        legacy_output: str | None = None
        new_output: str | None = None

        # Capture prompts (always) and optionally let the LLM run.
        async def fake_llm_legacy(system, user):
            nonlocal legacy_user
            legacy_user = user

        async def fake_llm_new(system, user):
            nonlocal new_user
            new_user = user

        # Legacy
        analyzer = SourceAnalyzer()
        try:
            metadata = analyzer.analyze_file(file_path)
        except SyntaxError as e:
            console.print(f'[red]Legacy path SyntaxError: {e}[/red]')
            return ('', '', None, None)

        legacy_gen = DocGenerator(
            config=GeneratorConfig(
                model=cfg.model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            ),
            analyzer=analyzer,
        )
        if with_llm:
            async with legacy_gen:
                docs = await legacy_gen.generate_for_module(
                    metadata, doc_types=(doc_type,),
                )
                legacy_output = docs[0].content if docs else None
                # Run again with mocked LLM just to capture the prompt.
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_legacy,
            ):
                legacy_gen2 = DocGenerator(analyzer=analyzer)
                async with legacy_gen2:
                    await legacy_gen2.generate_for_module(
                        metadata, doc_types=(doc_type,),
                    )
        else:
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_legacy,
            ):
                legacy_gen2 = DocGenerator(analyzer=analyzer)
                async with legacy_gen2:
                    await legacy_gen2.generate_for_module(
                        metadata, doc_types=(doc_type,),
                    )

        # New (catalog)
        bundle = enrich_file(file_path, source_root=file_path.parent)
        if bundle is None:
            console.print('[red]Catalog path could not build bundle.[/red]')
            return (legacy_user, '', legacy_output, None)

        new_gen = DocGenerator(
            config=GeneratorConfig(
                model=cfg.model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            ),
        )
        if with_llm:
            async with new_gen:
                docs = await new_gen.generate_from_elements(
                    bundle, doc_types=(doc_type,),
                )
                new_output = docs[0].content if docs else None
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_new,
            ):
                new_gen2 = DocGenerator()
                async with new_gen2:
                    await new_gen2.generate_from_elements(
                        bundle, doc_types=(doc_type,),
                    )
        else:
            with patch.object(
                DocGenerator, '_call_llm',
                new_callable=AsyncMock, side_effect=fake_llm_new,
            ):
                new_gen2 = DocGenerator()
                async with new_gen2:
                    await new_gen2.generate_from_elements(
                        bundle, doc_types=(doc_type,),
                    )

        return (legacy_user, new_user, legacy_output, new_output)

    try:
        legacy_user, new_user, legacy_out, new_out = asyncio.run(_run())
    except Exception as e:
        console.print(f'[red]diff-docs failed: {e}[/red]')
        return 1

    console.print(f'[bold]diff-docs[/bold] [cyan]{file_path}[/cyan] '
                  f'(doc_type={doc_type})')
    console.print()
    console.print('[bold yellow]== Legacy prompt ==[/bold yellow]')
    console.print(legacy_user or '[dim](empty)[/dim]')
    console.print()
    console.print('[bold green]== Catalog (new) prompt ==[/bold green]')
    console.print(new_user or '[dim](empty)[/dim]')

    if with_llm:
        console.print()
        console.print('[bold yellow]== Legacy LLM output ==[/bold yellow]')
        console.print(legacy_out or '[dim](no output)[/dim]')
        console.print()
        console.print('[bold green]== Catalog LLM output ==[/bold green]')
        console.print(new_out or '[dim](no output)[/dim]')

    return 0


def cmd_themes(args: argparse.Namespace) -> int:
    """Dispatcher for `ariadne themes <action>`."""
    action = getattr(args, 'themes_action', None)
    if action == 'build':
        return _cmd_themes_build(args)
    if action == 'list':
        return _cmd_themes_list(args)
    if action == 'show':
        return _cmd_themes_show(args)
    console.print('[yellow]Usage: ariadne themes (build|list|show <cluster_id>)[/yellow]')
    return 1


HANDLERS = {
    'themes': lambda args: cmd_themes(args),
    'diff-docs': lambda args: cmd_diff_docs(args),
}
