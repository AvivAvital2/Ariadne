"""Analytics and coverage tools for the Ariadne MCP server."""
from __future__ import annotations

from ariadne_mcp.models import (
    ContributeResponse,
    CoverageResponse,
    DocumentUsageResponse,
    GapReportResponse,
    GraphResponse,
    GraphStatsResponse,
    PriorityEntry,
    ProjectStatsResponse,
    UsageStatsResponse,
)


def ariadne_graph(
    action: str = 'stats',
    source: str | None = None,
    limit: int = 20,
) -> GraphResponse:
    """Interact with the Ariadne dependency graph.

    Actions:
    - 'build': Build/rebuild the graph from source files and documents.
    - 'stats': Show graph statistics (node/edge counts).
    - 'priorities': Show undocumented files ranked by connectivity (most-connected first).
    - 'export': Generate self-contained HTML visualization file.

    Args:
        action: One of 'build', 'stats', 'priorities', 'export'.
        source: Source name for build/priorities (optional, uses default).
        limit: Max results for priorities (default: 20).
    """
    from pathlib import Path as _Path

    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    source_name = svc._resolve_source(source) or ''

    if action == 'build':
        source_path = svc._source_path(source_name)
        if not source_path or not source_path.exists():
            return GraphResponse(action='build', output=f'Source path not found: {source_name}')
        counts = svc.build_graph(source_path)
        return GraphResponse(action='build', output=f'Graph built: {counts}')

    elif action == 'stats':
        stats = svc.get_graph_stats()
        return GraphResponse(
            action='stats',
            stats=GraphStatsResponse(**stats),
        )

    elif action == 'priorities':
        source_path = svc._source_path(source_name)
        if not source_path:
            return GraphResponse(action='priorities', output='No source specified')
        raw = svc.get_priorities(source_path)
        return GraphResponse(
            action='priorities',
            priorities=[PriorityEntry(**{k: r[k] for k in ('file', 'total_edges', 'doc_count', 'coverage_percent', 'priority_score')}) for r in raw[:limit]],
        )

    elif action == 'export':
        from graph_viewer import generate_graph_html

        graph_data = svc.export_graph_json()
        out_path = _Path.cwd() / 'ariadne-graph.html'
        generate_graph_html(graph_data, out_path)
        return GraphResponse(
            action='export',
            file_path=str(out_path),
            output=f'Graph exported: {len(graph_data["nodes"])} nodes, {len(graph_data["edges"])} edges',
        )

    return GraphResponse(action=action, output=f'Unknown action: {action}')


def ariadne_coverage(
    source: str | None = None,
) -> CoverageResponse:
    """Check documentation coverage for a source.

    Returns documented vs undocumented file counts and lists undocumented files.
    Use this instead of multiple ariadne_search calls when checking coverage.

    Args:
        source: Source name (optional, uses default if not specified).
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().coverage(source)


def ariadne_source_path(
    source: str | None = None,
) -> dict:
    """Get the filesystem path for a named source.

    Returns the resolved absolute path for a source configured in ariadne.yaml.
    Use this to find where a project's code lives on disk.

    Args:
        source: Source name (e.g. 'store', 'myproject'). Uses default if not specified.
    """
    from config import get_config

    cfg = get_config()
    source_name = source or cfg.default_source
    path = cfg.get_source_path(source_name) if source_name else None
    if path is None:
        return {'error': f"Source '{source_name}' not found", 'sources': list(cfg.sources.keys())}
    return {'source': source_name, 'path': str(path)}


async def ariadne_contribute(
    title: str,
    content: str,
    source_files: list[str] | None = None,
    content_type: str = 'finding',
) -> ContributeResponse:
    """Save a session insight to the Ariadne library.

    Call this when you've discovered or explained something significant about
    the codebase that isn't already in Ariadne — architecture insights,
    debugging findings, design rationale, or cross-cutting concerns.

    The document gets an embedding immediately and is searchable in the
    same session.

    Args:
        title: Short descriptive title for the insight.
        content: The insight content in markdown format.
        source_files: Related source file paths (optional).
        content_type: "finding" (default), "explanation", "architecture", or "gotcha".
    """
    from ariadne_mcp.service import AriadneService

    return await AriadneService.get().contribute(
        title=title,
        content=content,
        source_files=source_files,
        content_type=content_type,
    )


def ariadne_usage_stats(
    days: int = 30,
    tool_name: str | None = None,
) -> UsageStatsResponse:
    """Show Ariadne usage statistics.

    Use this when asked about Ariadne's usage, value, or effectiveness.

    Args:
        days: Number of days to include (default: 30).
        tool_name: Filter to a specific tool (optional).
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().usage_stats(days, tool_name)


def ariadne_project_stats() -> ProjectStatsResponse:
    """Show per-project document count and size statistics.

    Reports document count, content size, embedding size, and chunk count
    grouped by source (e.g., pythonproject, benchmark). Also shows total DB size.

    Use this when asked about Ariadne's library size or per-project coverage.
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().project_stats()


def ariadne_document_usage(
    days: int = 30,
    limit: int = 20,
) -> DocumentUsageResponse:
    """Show which documents are served most frequently.

    Returns per-document serve counts, sorted by most-served first.
    Use this to identify popular docs (candidates for improvement) and
    unused docs (candidates for pruning).

    Args:
        days: Number of days to include (default: 30).
        limit: Max documents to return (default: 20).
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().document_usage(days, limit)


async def ariadne_gaps(
    days: int = 30,
    analyze: bool = False,
) -> GapReportResponse:
    """Generate a gap/miss report showing what documentation is missing.

    Use this when asked about Ariadne's coverage gaps or improvement opportunities.

    Args:
        days: Number of days to include (default: 30).
        analyze: Run LLM-powered analysis for deeper recommendations.
    """
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    response = svc.gap_report(days)

    if analyze and response.total_misses > 0:
        analysis_text = await svc.gap_analysis(days)
        response.analysis = analysis_text

    return response


def register_tools(mcp) -> None:
    """Register analytics tools with the MCP server."""
    from mcp.types import ToolAnnotations

    mcp.tool(annotations=ToolAnnotations(
        title='Dependency Graph',
        readOnlyHint=False,
        destructiveHint=False,
        openWorldHint=False,
    ))(ariadne_graph)

    mcp.tool(annotations=ToolAnnotations(
        title='Coverage Check',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_coverage)

    mcp.tool(annotations=ToolAnnotations(
        title='Source Path',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_source_path)

    mcp.tool(annotations=ToolAnnotations(
        title='Contribute Insight',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ))(ariadne_contribute)

    mcp.tool(annotations=ToolAnnotations(
        title='Usage Stats',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_usage_stats)

    mcp.tool(annotations=ToolAnnotations(
        title='Project Stats',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_project_stats)

    mcp.tool(annotations=ToolAnnotations(
        title='Document Usage',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_document_usage)

    mcp.tool(annotations=ToolAnnotations(
        title='Gap Report',
        readOnlyHint=True,
        openWorldHint=False,
    ))(ariadne_gaps)
