"""Analysis CLI commands."""
from __future__ import annotations

import argparse

from rich.console import Console
from rich.table import Table

console = Console()


def get_library(db_path=None):
    from cli.main import get_library as _get_library
    return _get_library(db_path)


def register_commands(subparsers):
    # freshen
    freshen_parser = subparsers.add_parser('freshen', help='Check which docs for a file need regeneration')
    freshen_parser.add_argument('file', help='Source file path')

    # auto-tag
    autotag_parser = subparsers.add_parser('auto-tag', help='Cluster docs by embedding similarity and suggest labels')
    autotag_parser.add_argument('--clusters', '-n', type=int, default=10, help='Number of clusters')

    # rechunk
    subparsers.add_parser('rechunk-check', help='Find docs with poor chunk quality')

    # relate
    relate_parser = subparsers.add_parser('relate', help='Show how a document relates to its neighbors')
    relate_parser.add_argument('doc_id', help='Document ID')

    # decision-log
    subparsers.add_parser('decision-log', help='Extract all design decisions into a single log')

    # review-checklist
    review_parser = subparsers.add_parser('review-checklist', help='Generate PR review checklist')
    review_parser.add_argument('files', nargs='+', help='Changed file paths')

    # file_impact_radius
    impact_parser = subparsers.add_parser(
        'file_impact_radius', help='What files/tests/docs are affected by changing a file')
    impact_parser.add_argument('file', help='File to analyze')

    # coupling
    coupling_parser = subparsers.add_parser('coupling', help='Find highly-coupled file pairs')
    coupling_parser.add_argument('--source', '-s', default=None, help='Source name')

    # summarize
    summarize_parser = subparsers.add_parser('summarize', help='One-paragraph summary of a file')
    summarize_parser.add_argument('file', help='File path')

    # semantic-dupes
    dupes_parser = subparsers.add_parser('semantic-dupes', help='Find semantically similar documents that could be merged')
    dupes_parser.add_argument('--threshold', type=float, default=0.95, help='Similarity threshold (0-1)')

    # compare
    compare_parser = subparsers.add_parser('compare', help='Show how two files relate')
    compare_parser.add_argument('file1', help='First file path')
    compare_parser.add_argument('file2', help='Second file path')

    # patterns
    patterns_parser = subparsers.add_parser('patterns', help='Detect recurring code patterns in the codebase')
    patterns_parser.add_argument('--source', '-s', default=None, help='Source name')

    # complexity
    complex_parser = subparsers.add_parser('complexity', help='Score files by complexity vs doc coverage')
    complex_parser.add_argument('--source', '-s', default=None, help='Source name')
    complex_parser.add_argument('--limit', type=int, default=20, help='Max results')

    # who-knows
    who_parser = subparsers.add_parser('who-knows', help='Find developers with context on a topic')
    who_parser.add_argument('topic', help='Topic to search for')
    who_parser.add_argument('--source', '-s', default=None, help='Source name')

    # near-misses
    near_parser = subparsers.add_parser('near-misses', help='Predict what docs will be needed next')
    near_parser.add_argument('--days', '-d', type=int, default=7, help='Days to analyze')

    # explain
    explain_parser = subparsers.add_parser('explain', help='Show everything Ariadne knows about a file')
    explain_parser.add_argument('file', help='Path to the source file')


def cmd_freshen(args: argparse.Namespace) -> int:
    """Check which docs for a file need regeneration."""
    library = get_library(args.db)
    try:
        result = library.freshen_file(args.file)
        console.print(f'[bold]{result["file"]}[/bold]: {result["total_docs"]} docs, {result["stale_docs"]} stale')
        for d in result['docs']:
            status = '[red]STALE[/red]' if d['is_stale'] else '[green]OK[/green]'
            console.print(f'  {status} {d["title"]} ({d["content_type"]})')
        console.print(f'\n{result["recommendation"]}')
        return 0
    finally:
        library.close()


def cmd_auto_tag(args: argparse.Namespace) -> int:
    """Cluster docs by embedding similarity and suggest labels."""
    library = get_library(args.db)
    try:
        clusters = library.auto_tag_clusters(args.clusters)
        if not clusters:
            console.print('[dim]Not enough docs with embeddings for clustering.[/dim]')
            return 0
        console.print(f'[bold]Document Clusters ({len(clusters)}):[/bold]\n')
        for c in clusters:
            console.print(f'[bold cyan]{c["suggested_label"]}[/bold cyan] ({c["doc_count"]} docs)')
            for t in c['sample_titles'][:3]:
                console.print(f'  - {t}')
            console.print()
        return 0
    finally:
        library.close()


def cmd_rechunk_check(args: argparse.Namespace) -> int:
    """Find docs with poor chunk quality."""
    library = get_library(args.db)
    try:
        suggestions = library.suggest_rechunk()
        if not suggestions:
            console.print('[green]All chunks look good.[/green]')
            return 0
        console.print(f'[bold]Chunk Quality Issues ({len(suggestions)} docs):[/bold]\n')
        for s in suggestions:
            console.print(f'[bold]{s["title"]}[/bold] (quality: {s["quality_score"]}, {s["chunk_count"]} chunks)')
            for issue in s['issues'][:3]:
                console.print(f'  [yellow]- {issue}[/yellow]')
            console.print()
        return 0
    finally:
        library.close()


def cmd_relate(args: argparse.Namespace) -> int:
    """Show how a document relates to its neighbors."""
    library = get_library(args.db)
    try:
        result = library.relate_docs(args.doc_id)
        if 'error' in result:
            console.print(f'[red]{result["error"]}[/red]')
            return 1
        console.print(f'[bold]{result["title"]}[/bold] ({result["content_type"]})')
        console.print(f'Source: {", ".join(f.split("/")[-1] for f in result["source_files"])}')
        for rel_type, items in result['neighbors'].items():
            if items:
                console.print(f'\n[bold]{rel_type}:[/bold]')
                for item in items[:5]:
                    label = item.get('title') or item.get('file') or item.get('doc', '')
                    console.print(f'  - {label}')
        return 0
    finally:
        library.close()


def cmd_decision_log(args: argparse.Namespace) -> int:
    """Extract all design decisions into a log."""
    library = get_library(args.db)
    try:
        decisions = library.decision_log()
        if not decisions:
            console.print('[dim]No design decisions found in architecture docs.[/dim]')
            return 0
        console.print(f'[bold]Design Decision Log ({len(decisions)} decisions):[/bold]\n')
        for d in decisions:
            console.print(f'[bold cyan]{d["decision"]}[/bold cyan]')
            console.print(f'  [dim]Source: {d["source"]}[/dim]')
            if d['rationale']:
                console.print(f'  {d["rationale"][:200]}')
            console.print()
        return 0
    finally:
        library.close()


def cmd_review_checklist(args: argparse.Namespace) -> int:
    """Generate PR review checklist."""
    library = get_library(args.db)
    try:
        checklist = library.review_checklist(args.files)
        if not checklist:
            console.print('[green]No specific checks needed for these files.[/green]')
            return 0
        console.print(f'[bold]Review Checklist ({len(checklist)} items):[/bold]\n')
        for item in checklist:
            icon = {'gotcha': '!', 'thread_safety': 'T', 'temporal': 'L',
                    'validation': 'V', 'missing_tests': '?'}.get(item['type'], '-')
            console.print(f'  [{icon}] {item["file"]}: {item["check"]}')
        return 0
    finally:
        library.close()


def cmd_impact_radius(args: argparse.Namespace) -> int:
    """Calculate change impact radius."""
    library = get_library(args.db)
    try:
        result = library.impact_radius(args.file)
        console.print(f'[bold]Impact Radius: {args.file}[/bold]\n')
        table = Table()
        table.add_column('Metric', style='bold')
        table.add_column('Value', style='cyan')
        table.add_row('Direct dependents', str(result['direct_dependents']))
        table.add_row('Transitive dependents', str(result['transitive_dependents']))
        table.add_row('Total affected files', str(result['total_affected_files']))
        table.add_row('Affected docs', str(result['affected_docs']))
        table.add_row('Affected tests', str(result['affected_tests']))
        table.add_row('Radius score', str(result['radius_score']))
        console.print(table)
        if result['top_dependents']:
            console.print(f'\nTop dependents: {", ".join(result["top_dependents"])}')
        return 0
    finally:
        library.close()


def cmd_coupling(args: argparse.Namespace) -> int:
    """Find highly-coupled file pairs."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_path = cfg.resolve_source(args.source or cfg.default_source)
    if not source_path:
        console.print('[red]No source specified.[/red]')
        return 1
    try:
        pairs = library.coupling_report(source_path)
        if not pairs:
            console.print('[green]No highly-coupled pairs found.[/green]')
            return 0
        table = Table(title='Coupling Report')
        table.add_column('File 1')
        table.add_column('File 2')
        table.add_column('Mutual', justify='center')
        table.add_column('Edges', justify='right')
        for p in pairs:
            mutual = '[red]Yes[/red]' if p['mutual_imports'] else 'No'
            table.add_row(p['file1'], p['file2'], mutual, str(p['total_edges']))
        console.print(table)
        return 0
    finally:
        library.close()


def cmd_summarize(args: argparse.Namespace) -> int:
    """One-paragraph summary of a file."""
    library = get_library(args.db)
    try:
        summary = library.summarize_file(args.file)
        console.print(summary)
        return 0
    finally:
        library.close()


def cmd_semantic_dupes(args: argparse.Namespace) -> int:
    """Find semantically similar documents that could be merged."""
    library = get_library(args.db)

    try:
        console.print(f'Scanning for semantic duplicates (threshold: {args.threshold})...')
        dupes = library.find_semantic_duplicates(args.threshold)

        if not dupes:
            console.print('[green]No semantic duplicates found.[/green]')
            return 0

        table = Table(title=f'Semantic Duplicates ({len(dupes)} pairs)')
        table.add_column('Similarity', justify='right')
        table.add_column('Document 1')
        table.add_column('Document 2')

        for d in dupes:
            table.add_row(
                f'{d["similarity"]:.3f}',
                d['doc1_title'],
                d['doc2_title'],
            )

        console.print(table)
        console.print('\n[yellow]Consider merging highly similar docs to reduce library bloat.[/yellow]')
        return 0
    finally:
        library.close()


def cmd_compare(args: argparse.Namespace) -> int:
    """Show how two files relate."""
    library = get_library(args.db)

    try:
        result = library.compare_files(args.file1, args.file2)

        console.print(f'\n[bold]Comparing {args.file1} and {args.file2}[/bold]')
        console.print(f'Relationship: [cyan]{result["relationship"]}[/cyan]\n')

        if result['direct_edges']:
            console.print(f'[bold]Direct connections:[/bold] {", ".join(result["direct_edges"])}')

        if result['shared_imports']:
            console.print(f'[bold]Shared imports ({len(result["shared_imports"])}):[/bold]')
            for imp in result['shared_imports']:
                console.print(f'  - {imp}')

        if result['shared_docs']:
            console.print(f'[bold]Shared documentation ({len(result["shared_docs"])}):[/bold]')
            for title in result['shared_docs']:
                console.print(f'  - {title}')

        console.print(f'\n{args.file1}: {result["file1_imports"]} imports')
        console.print(f'{args.file2}: {result["file2_imports"]} imports')

        return 0
    finally:
        library.close()


def cmd_patterns(args: argparse.Namespace) -> int:
    """Detect recurring code patterns."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None
    if not source_path:
        console.print('[red]No source specified.[/red]')
        return 1
    try:
        patterns = library.detect_patterns(source_path)
        if not patterns:
            console.print('[dim]No patterns detected.[/dim]')
            return 0
        table = Table(title='Code Patterns')
        table.add_column('Pattern', style='bold')
        table.add_column('Count', justify='right')
        table.add_column('Example Files')
        for p in patterns:
            table.add_row(p['pattern'], str(p['count']), ', '.join(p['files'][:3]))
        console.print(table)
        return 0
    finally:
        library.close()


def cmd_complexity(args: argparse.Namespace) -> int:
    """Score files by complexity vs doc coverage."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None
    if not source_path:
        console.print('[red]No source specified.[/red]')
        return 1
    try:
        results = library.file_complexity(source_path)
        table = Table(title='File Complexity vs Documentation (highest risk first)')
        table.add_column('Risk', justify='right', style='bold')
        table.add_column('File')
        table.add_column('Lines', justify='right')
        table.add_column('Classes', justify='right')
        table.add_column('Functions', justify='right')
        table.add_column('Documented')
        for r in results[:args.limit]:
            doc_status = '[green]Yes[/green]' if r['documented'] else '[red]No[/red]'
            table.add_row(f'{r["risk_score"]:.0f}', r['file'], str(r['lines']),
                         str(r['classes']), str(r['functions']), doc_status)
        console.print(table)
        return 0
    finally:
        library.close()


def cmd_who_knows(args: argparse.Namespace) -> int:
    """Find developers with context on a topic."""
    from config import get_config

    cfg = get_config()
    library = get_library(args.db)
    source_name = args.source or cfg.default_source
    source_path = cfg.resolve_source(source_name) if source_name else None
    if not source_path:
        console.print('[red]No source specified.[/red]')
        return 1
    try:
        results = library.who_knows(args.topic, source_path)
        if not results:
            console.print(f'[dim]No authorship data found for "{args.topic}"[/dim]')
            return 0
        table = Table(title=f'Who Knows About "{args.topic}"')
        table.add_column('Author', style='bold')
        table.add_column('Lines', justify='right')
        table.add_column('Share', justify='right')
        for r in results:
            table.add_row(r['author'], str(r['lines']), f'{r["percentage"]}%')
        console.print(table)
        return 0
    finally:
        library.close()


def cmd_near_misses(args: argparse.Namespace) -> int:
    """Predict what docs will be needed next."""
    library = get_library(args.db)
    try:
        predictions = library.near_misses(days=args.days)
        if not predictions:
            console.print('[green]No emerging gaps detected.[/green]')
            return 0
        console.print(f'[bold]Predicted Documentation Needs (last {args.days} days):[/bold]')
        for p in predictions:
            status = '[green]has docs[/green]' if p['has_docs'] else '[red]no docs[/red]'
            console.print(f'  {p["topic"]:20s} ({p["miss_count"]}x) {status}')
            console.print(f'    [dim]{p["suggestion"]}[/dim]')
        return 0
    finally:
        library.close()


def cmd_explain(args: argparse.Namespace) -> int:
    """Show everything Ariadne knows about a specific file."""
    library = get_library(args.db)

    try:
        result = library.explain(args.file)

        if result['total_documents'] == 0:
            console.print(f'[yellow]No documentation found for {args.file}[/yellow]')
            console.print('Try generating docs: ariadne generate --path <directory>')
            return 0

        console.print(f'[bold]{result["summary"]}[/bold]\n')

        # Print documents grouped by type, explanation first
        type_order = ['explanation', 'architecture', 'finding', 'qa', 'diagram']
        for ct in type_order:
            docs = result['documents'].get(ct, [])
            if not docs:
                continue
            console.print(f'[bold cyan]--- {ct.upper()} ---[/bold cyan]')
            for doc in docs:
                console.print(f'[bold]{doc["title"]}[/bold]')
                # Print content with reasonable truncation for CLI
                content = doc['content']
                if len(content) > 3000:
                    content = content[:3000] + '\n\n[dim]... (truncated, use ariadne get <id> for full content)[/dim]'
                console.print(content)
                console.print()

        # Show graph neighbors
        if result['graph_neighbors']:
            console.print('[bold cyan]--- RELATED FILES ---[/bold cyan]')
            for n in result['graph_neighbors'][:10]:
                rel = n['relationship']
                console.print(f'  {rel}: {n["file"]}')

        return 0

    finally:
        library.close()


HANDLERS = {
    'freshen': lambda args: cmd_freshen(args),
    'auto-tag': lambda args: cmd_auto_tag(args),
    'rechunk-check': lambda args: cmd_rechunk_check(args),
    'relate': lambda args: cmd_relate(args),
    'decision-log': lambda args: cmd_decision_log(args),
    'review-checklist': lambda args: cmd_review_checklist(args),
    'file_impact_radius': lambda args: cmd_impact_radius(args),
    'coupling': lambda args: cmd_coupling(args),
    'summarize': lambda args: cmd_summarize(args),
    'semantic-dupes': lambda args: cmd_semantic_dupes(args),
    'compare': lambda args: cmd_compare(args),
    'patterns': lambda args: cmd_patterns(args),
    'complexity': lambda args: cmd_complexity(args),
    'who-knows': lambda args: cmd_who_knows(args),
    'near-misses': lambda args: cmd_near_misses(args),
    'explain': lambda args: cmd_explain(args),
}
