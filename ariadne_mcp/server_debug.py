"""Debug and code intelligence tools for the Ariadne MCP server."""
from __future__ import annotations

from ariadne_mcp.models import (
    AdminActionResponse,
    IssueAnalysisResponse,
)


def ariadne_diagnose(
    error_message: str,
) -> AdminActionResponse:
    """Diagnose an error using Ariadne's documentation.

    Paste a stack trace or error message. Ariadne extracts file paths,
    function names, and error types, then finds relevant docs to explain
    what likely went wrong.

    Args:
        error_message: Stack trace, error text, or symptom description.
    """
    from ariadne_mcp.service import AriadneService

    result = AriadneService.get().diagnose(error_message)
    lines = [f'Extracted: {len(result["extracted_files"])} files, {len(result["extracted_functions"])} functions, {len(result["extracted_errors"])} error types']
    if result['extracted_errors']:
        lines.append(f'Errors: {", ".join(result["extracted_errors"])}')
    if result['matched_docs']:
        lines.append(f'\nRelevant docs ({len(result["matched_docs"])}):')
        for d in result['matched_docs']:
            lines.append(f'  - {d["title"]} (matched by {d["match"]})')
    else:
        lines.append('\nNo matching docs found. Consider contributing a finding with ariadne_contribute.')
    return AdminActionResponse(output='\n'.join(lines))


def ariadne_debug_context(
    file_path: str,
) -> AdminActionResponse:
    """Get complete debugging context for a file in one call.

    Returns: file docs, dependencies, test files, recent git changes,
    known issues, and gotchas. Everything needed to start debugging.

    Args:
        file_path: Path to the file being debugged.
    """
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    ctx = svc.full_debug_context(file_path)
    gotchas = ctx.get('gotchas', [])
    tests = ctx.get('test_files', [])

    lines = [f'# Debug Context: {file_path}\n']
    lines.append(f'**Docs:** {ctx["docs"]["total_documents"]} documents')
    lines.append(f'**Tests:** {", ".join(t["path"].split("/")[-1] for t in tests) or "none found"}')

    if ctx['known_issues']:
        lines.append(f'\n**Known Issues:** {", ".join(ctx["known_issues"])}')

    if gotchas:
        lines.append(f'\n**Gotchas ({len(gotchas)}):**')
        for g in gotchas[:5]:
            lines.append(f'  - {g["text"]}')

    if ctx['recent_changes']:
        lines.append('\n**Recent Changes:**')
        for c in ctx['recent_changes']:
            lines.append(f'  {c}')

    neighbors = ctx.get('graph_neighbors', [])
    if neighbors:
        lines.append(f'\n**Dependencies:** {len(neighbors)} connected files')

    return AdminActionResponse(output='\n'.join(lines))


def ariadne_find_tests(
    target: str,
    mode: str = 'file',
) -> AdminActionResponse:
    """Find tests for a specific file or topic.

    Args:
        target: File path (mode='file') or topic name (mode='topic').
        mode: 'file' to find tests for a specific file, 'topic' to find
              all tests related to a topic (e.g., 'ingest', 'temporal').
    """
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    if mode == 'topic':
        tests = svc.find_tests_for_topic(target)
        if not tests:
            return AdminActionResponse(output=f'No tests found for topic "{target}"')
        lines = [f'Tests for topic "{target}" ({len(tests)}):']
        for t in tests:
            lines.append(f'  {t["path"].split("/")[-1]:40s}  {t["relevance"]}')
        return AdminActionResponse(output='\n'.join(lines))
    else:
        tests = svc.find_tests_for(target)
        if not tests:
            return AdminActionResponse(output=f'No tests found for {target}')
        lines = [f'Tests for {target} ({len(tests)}):']
        for t in tests:
            lines.append(f'  {t["path"].split("/")[-1]:40s}  ({t["match_type"]})')
        return AdminActionResponse(output='\n'.join(lines))


def ariadne_diff_explain(
    commit: str = 'HEAD',
    source: str | None = None,
) -> AdminActionResponse:
    """Explain what a commit changed using Ariadne docs for context.

    Args:
        commit: Git commit hash or ref (default: HEAD).
        source: Source name for path resolution.
    """
    import subprocess

    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    source_name = svc._resolve_source(source) or ''
    source_path = svc._source_path(source_name)

    if not source_path:
        return AdminActionResponse(output='No source configured.')

    try:
        result = subprocess.run(
            ['git', 'diff', f'{commit}~1..{commit}', '--name-only'],
            capture_output=True, text=True, timeout=10, cwd=source_path,
        )
        changed_files = [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]

        msg_result = subprocess.run(
            ['git', 'log', '--format=%s', '-1', commit],
            capture_output=True, text=True, timeout=5, cwd=source_path,
        )
        commit_msg = msg_result.stdout.strip()
    except Exception as e:
        return AdminActionResponse(output=f'Git error: {e}')

    # Find docs for changed files
    abs_files = [str(source_path / f) for f in changed_files]
    affected_docs = svc.find_documents_by_source_files(abs_files)

    lines = [f'# Commit: {commit_msg}\n', f'**Changed files:** {len(changed_files)}']
    for f in changed_files[:10]:
        lines.append(f'  - {f}')

    if affected_docs:
        lines.append(f'\n**Affected documentation ({len(affected_docs)}):**')
        for doc in affected_docs[:10]:
            lines.append(f'  - [{doc.content_type}] {doc.title}')
            # Extract first relevant paragraph
            for para in doc.content.split('\n\n')[:3]:
                if len(para.strip()) > 50:
                    lines.append(f'    {para.strip()[:150]}...')
                    break
    else:
        lines.append('\nNo Ariadne docs reference the changed files.')

    return AdminActionResponse(output='\n'.join(lines))


async def ariadne_refactor_plan(
    file_path: str,
    goal: str = '',
) -> AdminActionResponse:
    """Propose a safe refactoring plan for a file using Ariadne's knowledge.

    Analyzes the file's docs, dependencies, and graph neighbors to suggest
    refactoring steps with impact analysis.

    Args:
        file_path: Path to the file to refactor.
        goal: What the refactoring should achieve (optional).
    """
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    explain_data = svc.explain(file_path)
    graph_neighbors = explain_data.get('graph_neighbors', [])

    # Build context from docs + graph
    doc_context = ''
    for ct, docs in explain_data.get('documents', {}).items():
        for d in docs[:1]:
            doc_context += f'\n## {d["title"]}\n{d["content"][:2000]}\n'

    imports = [n['file'] for n in graph_neighbors if n.get('relationship') == 'imports']
    imported_by = [n['file'] for n in graph_neighbors if n.get('relationship') == 'imported_by']

    output = f'# Refactoring Plan: {file_path}\n\n'
    output += f'**Goal:** {goal or "General improvement"}\n\n'
    output += f'## Current State\n{explain_data["summary"]}\n\n'
    output += f'## Dependencies\n- Imports from: {len(imports)} files\n- Imported by: {len(imported_by)} files\n\n'
    output += f'## Impact Analysis\nChanges to this file may affect {len(imported_by)} dependent files:\n'
    for f in imported_by[:10]:
        output += f'  - {f.split("/")[-1]}\n'
    output += f'\n## Documentation Context\n{doc_context[:3000]}\n'

    return AdminActionResponse(output=output)


async def ariadne_test_suggestions(
    file_path: str,
) -> AdminActionResponse:
    """Suggest what test cases are missing for a file based on its documentation.

    Analyzes the file's Ariadne docs to identify testable behaviors,
    edge cases, and integration points that should have tests.

    Args:
        file_path: Path to the source file.
    """
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    explain_data = svc.explain(file_path)

    if explain_data['total_documents'] == 0:
        return AdminActionResponse(output=f'No documentation found for {file_path}. Generate docs first.')

    # Extract testable behaviors from docs
    output = f'# Test Suggestions for {file_path}\n\n'
    output += f'Based on {explain_data["total_documents"]} Ariadne documents:\n\n'

    for ct, docs in explain_data.get('documents', {}).items():
        for d in docs:
            content = d['content']
            # Look for patterns that suggest test cases
            import re

            # Public methods/functions
            methods = re.findall(r'`(\w+)\(', content)
            if methods:
                output += f'## From {d["title"]}\n'
                unique_methods = sorted(set(methods))[:15]
                for m in unique_methods:
                    output += f'- [ ] Test `{m}()` — verify expected behavior\n'

            # Edge cases mentioned
            edge_patterns = re.findall(r'(?:edge case|corner case|error|raise|fail|empty|none|null|zero|invalid|missing)', content.lower())
            if edge_patterns:
                output += f'- [ ] Test edge cases: {", ".join(sorted(set(edge_patterns))[:5])}\n'

            output += '\n'

    return AdminActionResponse(output=output)


def ariadne_list_issues(
    repo: str,
    state: str = 'open',
    limit: int = 10,
    labels: str | None = None,
) -> AdminActionResponse:
    """List GitHub issues for a repository.

    Use this to browse issues before selecting one for analysis
    with ariadne_analyze_issue.

    Args:
        repo: GitHub repo in "owner/name" format.
        state: Issue state filter ('open', 'closed', 'all').
        limit: Max issues to return.
        labels: Comma-separated label filter (optional).
    """
    import subprocess

    cmd = ['gh', 'issue', 'list', '--repo', repo, '--state', state,
           '--limit', str(limit), '--json', 'number,title,labels,updatedAt']
    if labels:
        cmd.extend(['--label', labels])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        if result.returncode != 0:
            return AdminActionResponse(output=f'Error: {result.stderr.strip()}')

        import json as json_mod
        issues = json_mod.loads(result.stdout)
        lines = []
        for issue in issues:
            label_str = ', '.join(l['name'] for l in issue.get('labels', []))
            lines.append(f'#{issue["number"]:>4d}  {issue["title"][:60]:60s}  {label_str}')
        return AdminActionResponse(output='\n'.join(lines) if lines else 'No issues found.')
    except FileNotFoundError:
        return AdminActionResponse(output='GitHub CLI (gh) not found. Install: https://cli.github.com/')
    except Exception as e:
        return AdminActionResponse(output=f'Error: {e}')


async def ariadne_analyze_issue(
    repo: str,
    issue_number: int,
) -> IssueAnalysisResponse:
    """Read a GitHub issue and propose implementation based on Ariadne's knowledge.

    Fetches the issue (title, body, comments), searches Ariadne for relevant
    docs and files, then synthesizes an implementation proposal.

    Args:
        repo: GitHub repo in "owner/name" format (e.g., "ExampleCorp/myproject").
        issue_number: Issue number.
    """
    from ariadne_mcp.service import AriadneService

    return await AriadneService.get().analyze_issue(repo, issue_number)


def register_tools(mcp) -> None:
    """Register debug tools with the MCP server."""
    from mcp.types import ToolAnnotations

    mcp.tool(annotations=ToolAnnotations(
        title='Diagnose Error',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_diagnose)

    mcp.tool(annotations=ToolAnnotations(
        title='Debug Context',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_debug_context)

    mcp.tool(annotations=ToolAnnotations(
        title='Find Tests',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_find_tests)

    mcp.tool(annotations=ToolAnnotations(
        title='Diff Explain',
        readOnlyHint=True,
        openWorldHint=True,
    ))(ariadne_diff_explain)

    mcp.tool(annotations=ToolAnnotations(
        title='Refactor Plan',
        readOnlyHint=True,
        openWorldHint=True,
    ))(ariadne_refactor_plan)

    mcp.tool(annotations=ToolAnnotations(
        title='Test Suggestions',
        readOnlyHint=True,
        openWorldHint=True,
    ))(ariadne_test_suggestions)

    mcp.tool(annotations=ToolAnnotations(
        title='List Issues',
        readOnlyHint=True,
        openWorldHint=True,
    ))(ariadne_list_issues)

    mcp.tool(annotations=ToolAnnotations(
        title='Analyze Issue',
        readOnlyHint=True,
        openWorldHint=True,
    ))(ariadne_analyze_issue)
