"""Health and maintenance CLI commands."""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

console = Console()


def get_library(db_path=None):
    from cli.main import get_library as _get_library
    return _get_library(db_path)


def register_commands(subparsers):
    # lint-docs
    subparsers.add_parser('lint-docs', help='Check document quality issues')

    # shrink
    subparsers.add_parser('shrink', help='Suggest how to split oversized documents')

    # trends
    trends_parser = subparsers.add_parser('trends', help='Show metric trends over time')
    trends_parser.add_argument('--days', '-d', type=int, default=30, help='Number of days')

    # roi
    roi_parser = subparsers.add_parser('roi', help='Estimate return on investment')
    roi_parser.add_argument('--days', '-d', type=int, default=30, help='Number of days')

    # suggest-topics
    suggest_parser = subparsers.add_parser('suggest-topics', help='Suggest new cross-cutting topic documents')
    suggest_parser.add_argument('--source', '-s', default=None, help='Source name')

    # quiz
    quiz_parser = subparsers.add_parser('quiz', help='Generate Q&A pairs from docs for testing understanding')
    quiz_parser.add_argument('--count', '-n', type=int, default=5, help='Number of questions')
    quiz_parser.add_argument('--source', '-s', default=None, help='Source name')

    # debt
    debt_parser = subparsers.add_parser('debt', help='Calculate documentation debt score')
    debt_parser.add_argument('--source', '-s', default=None, help='Source name (default from config)')

    # diff-impact
    diff_parser = subparsers.add_parser('diff-impact', help='Show which docs are affected by uncommitted changes')
    diff_parser.add_argument('--source', '-s', default=None, help='Source name (default from config)')
    diff_parser.add_argument('--staged', action='store_true', help='Check staged changes only')

    # doctor
    subparsers.add_parser('doctor', help='Run health check on the Ariadne library')


def cmd_lint_docs(args: argparse.Namespace) -> int:
    """Check document quality issues."""
    library = get_library(args.db)
    try:
        issues = library.lint_docs()
        if not issues:
            console.print('[green]No document quality issues found.[/green]')
            return 0
        table = Table(title=f'Document Quality Issues ({len(issues)})')
        table.add_column('Type', style='bold')
        table.add_column('Document')
        table.add_column('Detail')
        for issue in issues[:30]:
            table.add_row(issue['issue_type'], issue['title'][:40], issue['detail'][:60])
        console.print(table)
        return 0
    finally:
        library.close()


def cmd_shrink(args: argparse.Namespace) -> int:
    """Suggest how to split oversized documents."""
    library = get_library(args.db)
    try:
        suggestions = library.shrink_suggestions()
        if not suggestions:
            console.print('[green]No oversized documents found.[/green]')
            return 0
        for s in suggestions:
            console.print(f'\n[bold]{s["title"]}[/bold] ({s["content_size"]:,} chars, {s["sections"]} sections)')
            console.print('  Proposed splits:')
            for split in s['proposed_splits']:
                console.print(f'    - {split}')
        return 0
    finally:
        library.close()


def cmd_trends(args: argparse.Namespace) -> int:
    """Show metric trends over time."""
    library = get_library(args.db)
    try:
        trends = library.get_trends(days=args.days)
        if not trends['daily_usage']:
            console.print('[dim]No usage data for this period.[/dim]')
            return 0
        table = Table(title=f'Usage Trends (last {args.days} days)')
        table.add_column('Date', style='dim')
        table.add_column('Calls', justify='right')
        table.add_column('Hits', justify='right', style='green')
        table.add_column('Misses', justify='right', style='red')
        table.add_column('Hit Rate', justify='right')
        for day in trends['daily_usage']:
            table.add_row(day['date'], str(day['calls']), str(day['hits']),
                         str(day['misses']), f'{day["hit_rate"]:.0%}')
        console.print(table)
        console.print(f'\nCurrent library: {trends["current_doc_count"]} documents')
        return 0
    finally:
        library.close()


def cmd_roi(args: argparse.Namespace) -> int:
    """Estimate return on investment."""
    library = get_library(args.db)
    try:
        roi = library.estimate_roi(days=args.days)
        table = Table(title=f'Ariadne ROI Estimate (last {args.days} days)')
        table.add_column('Metric', style='bold')
        table.add_column('Value', style='cyan')
        table.add_row('Total calls', str(roi['total_calls']))
        table.add_row('Total hits', str(roi['total_hits']))
        table.add_row('Hit rate', f'{roi["hit_rate"]:.1%}')
        table.add_row('Documents', str(roi['doc_count']))
        table.add_section()
        table.add_row('Est. time saved', f'{roi["estimated_time_saved_minutes"]} min ({roi["estimated_time_saved_hours"]}h)')
        table.add_row('Est. generation cost', f'${roi["estimated_generation_cost_usd"]}')
        table.add_row('ROI ratio', f'{roi["roi_ratio"]} min saved per $1 spent')
        console.print(table)
        return 0
    finally:
        library.close()


def cmd_suggest_topics(args: argparse.Namespace) -> int:
    """Suggest new cross-cutting topic documents."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None

    if not source_path:
        console.print('[red]No source specified.[/red]')
        return 1

    try:
        suggestions = library.suggest_topics(source_path)
        if not suggestions:
            console.print('[green]No topic suggestions — all clusters are covered.[/green]')
            return 0

        console.print(f'[bold]Suggested Topic Documents ({len(suggestions)}):[/bold]\n')
        for i, s in enumerate(suggestions, 1):
            console.print(f'[bold cyan]{i}. {s["title"]}[/bold cyan]')
            console.print(f'   {s["rationale"]}')
            console.print(f'   Files: {", ".join(s["files"][:5])}')
            if len(s['files']) > 5:
                console.print(f'   ... and {len(s["files"]) - 5} more')
            console.print()

        console.print('[dim]Generate a topic: ariadne topic "Title" --files file1.py file2.py[/dim]')
        return 0
    finally:
        library.close()


def cmd_quiz(args: argparse.Namespace) -> int:
    """Generate Q&A pairs from docs for testing understanding."""
    import random

    library = get_library(args.db)

    try:
        docs = library.list_documents()
        if not docs:
            console.print('[yellow]No documents in library.[/yellow]')
            return 0

        # Pick random docs and generate questions from their content
        rng = random.Random()
        sample = rng.sample(docs, min(args.count * 2, len(docs)))

        console.print(f'[bold]Ariadne Knowledge Quiz ({args.count} questions):[/bold]\n')

        questions_shown = 0
        for doc in sample:
            if questions_shown >= args.count:
                break
            if not doc.content or len(doc.content) < 100:
                continue

            # Extract a question from the doc title/content
            title = doc.title
            # Get first meaningful paragraph as context
            paragraphs = [p.strip() for p in doc.content.split('\n\n') if len(p.strip()) > 50]
            if not paragraphs:
                continue

            answer_preview = paragraphs[0][:200]
            questions_shown += 1

            console.print(f'[bold]Q{questions_shown}:[/bold] What does {title} do/explain?')
            console.print(f'[dim]   Hint: {answer_preview}...[/dim]')
            console.print(f'   [cyan]Full answer: ariadne get {doc.id}[/cyan]\n')

        return 0
    finally:
        library.close()


def cmd_debt(args: argparse.Namespace) -> int:
    """Calculate documentation debt score."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None

    if not source_path:
        console.print('[red]No source specified.[/red]')
        return 1

    try:
        result = library.doc_debt_score(source_path)

        grade_colors = {'A': 'green', 'B': 'cyan', 'C': 'yellow', 'D': 'red', 'F': 'bold red'}
        color = grade_colors.get(result['grade'], 'white')

        console.print(f'\n[bold]Documentation Debt Score: [{color}]{result["score"]}/100 (Grade: {result["grade"]})[/{color}][/bold]\n')

        table = Table(title='Debt Components (lower = better)')
        table.add_column('Component', style='bold')
        table.add_column('Debt', justify='right')
        table.add_column('Max', justify='right', style='dim')

        for name, debt in result['components'].items():
            max_val = {'coverage_debt': 40, 'embedding_debt': 15, 'duplicate_debt': 15, 'graph_debt': 15, 'hit_rate_debt': 15}
            table.add_row(name.replace('_', ' ').title(), f'{debt}', f'/{max_val.get(name, "?")}')

        console.print(table)

        if result['suggestions']:
            console.print('\n[bold]Suggestions:[/bold]')
            for s in result['suggestions']:
                console.print(f'  - {s}')

        return 0
    finally:
        library.close()


def cmd_diff_impact(args: argparse.Namespace) -> int:
    """Show which Ariadne docs are affected by uncommitted git changes."""
    import subprocess

    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None

    if not source_path or not source_path.exists():
        console.print(f'[red]Source path not found for: {source_name}[/red]')
        return 1

    try:
        # Get changed files from git
        diff_cmd = ['git', 'diff', '--name-only']
        if args.staged:
            diff_cmd.append('--cached')
        result = subprocess.run(diff_cmd, capture_output=True, text=True, cwd=source_path)
        changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        if not changed_files:
            console.print('[green]No uncommitted changes found.[/green]')
            return 0

        # Resolve to absolute paths
        abs_files = [str(source_path / f) for f in changed_files]

        # Find affected docs
        affected = library.find_documents_by_source_files(abs_files)

        console.print(f'[bold]{len(changed_files)} files changed, {len(affected)} docs affected[/bold]\n')

        if affected:
            table = Table(title='Affected Documents')
            table.add_column('Type', style='dim')
            table.add_column('Title', style='bold')
            table.add_column('Source File')

            for doc in affected:
                # Find which changed file this doc references
                matching_file = 'unknown'
                for sf in doc.source_files:
                    for cf in abs_files:
                        if cf in sf or sf in cf:
                            matching_file = cf.replace(str(source_path) + '/', '')
                            break
                table.add_row(doc.content_type, doc.title, matching_file)

            console.print(table)
            console.print(f'\n[yellow]These docs may be stale. Run "ariadne sync -s {source_name}" to update.[/yellow]')
        else:
            console.print('[dim]No Ariadne docs reference the changed files.[/dim]')
            # Check if any changed files are undocumented (single scan)
            all_source_files = set()
            for d in library.list_documents():
                all_source_files.update(d.source_files)
            undoc = [f for f in changed_files if f.endswith('.py') and not any(
                f in sf for sf in all_source_files
            )]
            if undoc:
                console.print(f'\n[yellow]{len(undoc)} changed Python files have no docs:[/yellow]')
                for f in undoc[:10]:
                    console.print(f'  - {f}')

        return 0

    finally:
        library.close()


def cmd_doctor(args: argparse.Namespace) -> int:
    """Run health check on the Ariadne library."""
    library = get_library(args.db)

    try:
        result = library.health_check()

        status_colors = {'ok': 'green', 'warning': 'yellow', 'error': 'red'}
        status_icons = {'ok': 'PASS', 'warning': 'WARN', 'error': 'FAIL'}

        table = Table(title=f'Ariadne Health Check — {result["summary"]}')
        table.add_column('Check', style='bold')
        table.add_column('Status')
        table.add_column('Detail')

        for check in result['checks']:
            color = status_colors.get(check['status'], 'white')
            icon = status_icons.get(check['status'], '?')
            table.add_row(
                check['name'],
                f'[{color}]{icon}[/{color}]',
                check['detail'],
            )

        console.print(table)
        return 0 if result['errors'] == 0 else 1

    finally:
        library.close()


HANDLERS = {
    'lint-docs': lambda args: cmd_lint_docs(args),
    'shrink': lambda args: cmd_shrink(args),
    'trends': lambda args: cmd_trends(args),
    'roi': lambda args: cmd_roi(args),
    'suggest-topics': lambda args: cmd_suggest_topics(args),
    'quiz': lambda args: cmd_quiz(args),
    'debt': lambda args: cmd_debt(args),
    'diff-impact': lambda args: cmd_diff_impact(args),
    'doctor': lambda args: cmd_doctor(args),
}
