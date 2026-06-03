"""Knowledge & analysis tools for the Ariadne MCP server."""
from __future__ import annotations

from ariadne_mcp.models import (
    AdminActionResponse,
    AskResponse,
    ExplainDocument,
    ExplainResponse,
    TaskContextResponse,
)


def ariadne_review_checklist(
    files: list[str],
) -> AdminActionResponse:
    """Generate a PR review checklist from Ariadne's knowledge of changed files.

    Checks for: documented gotchas, thread safety concerns, temporal leak
    risks, validation requirements, and missing tests.

    Args:
        files: List of changed file paths.
    """
    from ariadne_mcp.service import AriadneService

    checklist = AriadneService.get().review_checklist(files)
    if not checklist:
        return AdminActionResponse(output='No specific checks needed.')
    lines = [f'Review Checklist ({len(checklist)} items):']
    for item in checklist:
        lines.append(f'  [{item["type"]}] {item["file"]}: {item["check"]}')
    return AdminActionResponse(output='\n'.join(lines))


async def ariadne_review(
    task_description: str,
    changed_files: list[str] | None = None,
) -> 'ReviewResponse':
    """Composite architecture review for design deliberation.

    Combines semantic search, per-file explanation, impact analysis,
    and safety checklist into a single structured response. Used by
    the Code Inspector's deliberation agents.

    Args:
        task_description: What the task/feature is about.
        changed_files: List of files being modified.
    """
    from ariadne_mcp.service import AriadneService

    return await AriadneService.get().review(task_description, changed_files)


async def ariadne_task_context(
    task_description: str,
    file_paths: list[str],
) -> 'TaskContextResponse':
    """One-shot briefing for a worker starting a task.

    Bundles search, per-file explain, review checklist, and find_tests
    into a single response — collapses 4+ separate calls into one.

    Args:
        task_description: What the task is about.
        file_paths: Files the worker will modify or create.
    """
    from ariadne_mcp.service import AriadneService

    return await AriadneService.get().task_context(task_description, file_paths)


def ariadne_impact_radius(
    file_path: str,
) -> AdminActionResponse:
    """Calculate how many files/tests/docs would be affected by changing a file.

    Args:
        file_path: Path to the file to analyze.
    """
    from ariadne_mcp.service import AriadneService

    result = AriadneService.get().impact_radius(file_path)
    lines = [
        f'Impact Radius: {file_path}',
        f'  Direct dependents: {result["direct_dependents"]}',
        f'  Transitive dependents: {result["transitive_dependents"]}',
        f'  Affected docs: {result["affected_docs"]}',
        f'  Affected tests: {result["affected_tests"]}',
        f'  Radius score: {result["radius_score"]}',
    ]
    if result['top_dependents']:
        lines.append(f'  Top dependents: {", ".join(result["top_dependents"])}')
    return AdminActionResponse(output='\n'.join(lines))


def ariadne_summarize(
    file_path: str,
) -> AdminActionResponse:
    """Get a one-paragraph summary of what a file does.

    Shorter than ariadne_explain — suitable for PR descriptions,
    changelogs, or quick context.

    Args:
        file_path: Path to the source file.
    """
    from ariadne_mcp.service import AriadneService

    summary = AriadneService.get().summarize_file(file_path)
    return AdminActionResponse(output=summary)


def ariadne_explain(
    file_path: str,
) -> ExplainResponse:
    """Get everything Ariadne knows about a specific source file.

    Assembles all document types (explanation, architecture, topic, finding)
    related to the file into a single composite response. More efficient than
    multiple search calls — use this when you need to understand a specific file.

    Args:
        file_path: Path to the source file (absolute or relative to source root).
    """
    from ariadne_mcp.service import AriadneService, _trim_related_documents

    svc = AriadneService.get()
    raw = svc.explain(file_path)

    return ExplainResponse(
        file=raw['file'],
        summary=raw['summary'],
        total_documents=raw['total_documents'],
        types_found=raw['types_found'],
        documents={
            ct: [ExplainDocument(id=d['id'], title=d['title'], content=_trim_related_documents(d['content']))
                 for d in docs]
            for ct, docs in raw['documents'].items()
        },
        graph_neighbors=raw['graph_neighbors'],
    )


async def ariadne_ask(
    question: str,
    branch: str | None = None,
) -> AskResponse:
    """Ask a natural language question and get a synthesized answer from Ariadne docs.

    Unlike ariadne_search which returns raw documents, this tool:
    1. Searches for relevant docs
    2. Assembles the most relevant content
    3. Uses an LLM to synthesize a direct answer with citations

    Use this for "how does X work?" or "what is the pattern for Y?" questions.

    Args:
        question: Natural language question about the codebase.
        branch: Branch filter (optional, auto-detected).
    """
    from ariadne_mcp.service import AriadneService

    return await AriadneService.get().ask(question, branch)


def register_tools(mcp) -> None:
    """Register knowledge tools with the MCP server."""
    from mcp.types import ToolAnnotations

    mcp.tool(annotations=ToolAnnotations(
        title='Review Checklist',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_review_checklist)

    mcp.tool(annotations=ToolAnnotations(
        title='Architecture Review',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_review)

    mcp.tool(annotations=ToolAnnotations(
        title='Task Context',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_task_context)

    mcp.tool(annotations=ToolAnnotations(
        title='Impact Radius',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_impact_radius)

    mcp.tool(annotations=ToolAnnotations(
        title='Summarize File',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_summarize)

    mcp.tool(annotations=ToolAnnotations(
        title='Explain File',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_explain)

    mcp.tool(annotations=ToolAnnotations(
        title='Ask Question',
        readOnlyHint=True,
        openWorldHint=True,
    ))(ariadne_ask)
