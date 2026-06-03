"""Debug and testing CLI commands."""
from __future__ import annotations

import argparse
from pathlib import Path

from rich.console import Console

console = Console()


def get_library(db_path=None):
    from cli.main import get_library as _get_library
    return _get_library(db_path)


def register_commands(subparsers):
    # teach
    teach_parser = subparsers.add_parser('teach', help='Interactive teaching mode for a concept')
    teach_parser.add_argument('concept', help='Concept to teach')

    # diagnose
    diag_parser = subparsers.add_parser('diagnose', help='Diagnose an error using Ariadne docs')
    diag_parser.add_argument('error', nargs='?', help='Error message or stack trace (or pipe from stdin)')

    # debug-context
    dbg_parser = subparsers.add_parser('debug-context', help='Get complete debugging context for a file')
    dbg_parser.add_argument('file', help='File path to debug')
    dbg_parser.add_argument('--source', '-s', default=None, help='Source name')

    # test-for
    testfor_parser = subparsers.add_parser('test-for', help='Find tests for a file')
    testfor_parser.add_argument('file', help='Source file path')

    # test-topic
    testtopic_parser = subparsers.add_parser('test-topic', help='Find all tests related to a topic')
    testtopic_parser.add_argument('topic', help='Topic name (e.g., "ingest", "temporal")')

    # gotchas
    gotchas_parser = subparsers.add_parser('gotchas', help='Show documented pitfalls for a file')
    gotchas_parser.add_argument('file', help='Source file path')

    # stack-explain
    stack_parser = subparsers.add_parser('stack-explain', help='Annotate a stack trace with Ariadne doc context')
    stack_parser.add_argument('traceback', nargs='?', help='Stack trace text (or pipe from stdin)')


def cmd_teach(args: argparse.Namespace) -> int:
    """Interactive teaching mode for a concept."""
    import random
    library = get_library(args.db)
    try:
        docs = library._text_search(args.concept, limit=5)
        if not docs:
            console.print(f'[yellow]No docs found about "{args.concept}".[/yellow]')
            return 0

        console.print(f'[bold]Learning: {args.concept}[/bold]\n')

        # Present the concept
        main_doc = docs[0]
        paragraphs = [p.strip() for p in main_doc.content.split('\n\n') if len(p.strip()) > 50 and not p.strip().startswith('#')]
        if paragraphs:
            console.print(f'[cyan]{paragraphs[0][:500]}[/cyan]\n')

        # Generate a quiz question
        if len(paragraphs) > 1:
            answer_para = random.Random().choice(paragraphs[1:4] if len(paragraphs) > 3 else paragraphs[1:])
            # Find a key term in the answer
            import re
            terms = re.findall(r'`(\w+)`', answer_para)
            if terms:
                term = random.Random().choice(terms)
                console.print(f'[bold]Question:[/bold] What role does `{term}` play in {args.concept}?')
                console.print('[dim]Think about it, then press Enter to see the answer...[/dim]')
                try:
                    input()
                except EOFError:
                    pass
                console.print('\n[bold]Answer:[/bold]')
                console.print(f'{answer_para[:400]}\n')

        console.print(f'[dim]Learn more: ariadne explain <file> or ariadne get {main_doc.id}[/dim]')
        return 0
    finally:
        library.close()


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Diagnose an error using Ariadne docs."""
    import sys
    library = get_library(args.db)
    try:
        error = args.error
        if not error and not sys.stdin.isatty():
            error = sys.stdin.read()
        if not error:
            console.print('[red]Provide an error message as argument or pipe from stdin.[/red]')
            return 1
        result = library.diagnose(error)
        console.print(f'[bold]Extracted: {len(result["extracted_files"])} files, {len(result["extracted_errors"])} errors[/bold]')
        if result['extracted_errors']:
            console.print(f'Errors: {", ".join(result["extracted_errors"])}')
        if result['matched_docs']:
            console.print(f'\n[bold]Relevant docs ({len(result["matched_docs"])}):[/bold]')
            for d in result['matched_docs']:
                console.print(f'  - {d["title"]} ({d["match"]})')
        else:
            console.print('[yellow]No matching docs found.[/yellow]')
        return 0
    finally:
        library.close()


def cmd_debug_context(args: argparse.Namespace) -> int:
    """Get complete debugging context for a file."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_path = cfg.resolve_source(args.source or cfg.default_source)
    try:
        ctx = library.debug_context(args.file, source_path)
        gotchas = library.extract_gotchas(args.file)
        tests = library.find_tests_for(args.file)
        console.print(f'[bold]Debug Context: {args.file}[/bold]\n')
        console.print(f'Docs: {ctx["docs"]["total_documents"]}')
        console.print(f'Tests: {", ".join(t["path"].split("/")[-1] for t in tests) or "none"}')
        if ctx['known_issues']:
            console.print(f'Known Issues: {", ".join(ctx["known_issues"])}')
        if gotchas:
            console.print('\n[bold]Gotchas:[/bold]')
            for g in gotchas[:10]:
                console.print(f'  [yellow]- {g["text"]}[/yellow]')
        if ctx['recent_changes']:
            console.print('\n[bold]Recent Changes:[/bold]')
            for c in ctx['recent_changes']:
                console.print(f'  {c}')
        return 0
    finally:
        library.close()


def cmd_test_for(args: argparse.Namespace) -> int:
    """Find tests for a file."""
    library = get_library(args.db)
    try:
        tests = library.find_tests_for(args.file)
        if not tests:
            console.print(f'[dim]No tests found for {args.file}[/dim]')
            return 0
        console.print(f'[bold]Tests for {args.file} ({len(tests)}):[/bold]')
        for t in tests:
            console.print(f'  {t["path"]} ({t["match_type"]})')
        return 0
    finally:
        library.close()


def cmd_test_topic(args: argparse.Namespace) -> int:
    """Find all tests related to a topic."""
    library = get_library(args.db)
    try:
        tests = library.find_tests_for_topic(args.topic)
        if not tests:
            console.print(f'[dim]No tests found for topic "{args.topic}"[/dim]')
            return 0
        console.print(f'[bold]Tests for topic "{args.topic}" ({len(tests)}):[/bold]')
        for t in tests:
            console.print(f'  {t["path"].split("/")[-1]:40s} {t["relevance"]}')
        return 0
    finally:
        library.close()


def cmd_gotchas(args: argparse.Namespace) -> int:
    """Show documented pitfalls for a file."""
    library = get_library(args.db)
    try:
        gotchas = library.extract_gotchas(args.file)
        if not gotchas:
            console.print(f'[green]No gotchas found for {args.file}[/green]')
            return 0
        console.print(f'[bold]Gotchas for {args.file} ({len(gotchas)}):[/bold]')
        for g in gotchas:
            console.print(f'  [yellow]- {g["text"]}[/yellow]')
            console.print(f'    [dim]Source: {g["source"]}[/dim]')
        return 0
    finally:
        library.close()


def cmd_stack_explain(args: argparse.Namespace) -> int:
    """Annotate a stack trace with Ariadne doc context."""
    import sys
    library = get_library(args.db)
    try:
        text = args.traceback
        if not text and not sys.stdin.isatty():
            text = sys.stdin.read()
        if not text:
            console.print('[red]Provide a stack trace as argument or pipe from stdin.[/red]')
            return 1
        frames = library.stack_explain(text)
        if not frames:
            console.print('[dim]No stack frames could be parsed.[/dim]')
            return 0
        for f in frames:
            if f['file'] == '(exception)':
                console.print(f'\n[bold red]{f["function"]}: {f["context"]}[/bold red]')
                continue
            basename = Path(f['file']).name
            doc_info = f' [cyan]docs: {", ".join(f["docs"][:2])}[/cyan]' if f['docs'] else ' [dim]no docs[/dim]'
            console.print(f'  {basename}:{f["line"]} in [bold]{f["function"]}[/bold]{doc_info}')
            if f['context']:
                console.print(f'    [dim]{f["context"][:150]}[/dim]')
        return 0
    finally:
        library.close()


HANDLERS = {
    'teach': lambda args: cmd_teach(args),
    'diagnose': lambda args: cmd_diagnose(args),
    'debug-context': lambda args: cmd_debug_context(args),
    'test-for': lambda args: cmd_test_for(args),
    'test-topic': lambda args: cmd_test_topic(args),
    'gotchas': lambda args: cmd_gotchas(args),
    'stack-explain': lambda args: cmd_stack_explain(args),
}
