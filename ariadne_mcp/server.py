"""MCP server for Ariadne documentation retrieval.

Thin tool definitions that delegate to AriadneService.
All business logic lives in ariadne_mcp/service.py; response models in ariadne_mcp/models.py.

Tool modules:
- ariadne_mcp.server_knowledge: review, explain, ask, task context
- ariadne_mcp.server_debug: diagnose, debug context, tests, diff explain
- ariadne_mcp.server_analytics: graph, coverage, contribute, usage stats
- ariadne_mcp.server_admin: sync, generate, improve, merge, docs

Usage:
    python -m ariadne_mcp.server

Configured automatically by ``ariadne init``. Manual setup in .mcp.json::

    "mcpServers": {
        "ariadne": {
            "command": "uv",
            "args": ["run", "--directory", "/path/to/Ariadne", "ariadne", "mcp"]
        }
    }
"""
from __future__ import annotations

import logging

# Best-effort .env loading — python-dotenv is a convenience for local
# API-key pickup. Don't let its absence crash MCP server startup.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations

from ariadne_mcp.models import (
    BranchStatusResponse,
    FeedbackResponse,
    ListResponse,
    SearchResponse,
    SyncStatusResponse,
)

_logger = logging.getLogger(__name__)


def _warn_if_no_sources() -> None:
    """Fail loud if the server starts with no sources configured.

    The #1 silent failure mode: launched (e.g. via ``uv run --directory``)
    from a cwd where no ``ariadne.yaml`` resolves, so every source-scoped tool
    fails with a cryptic ``configured sources: []``. Name where we looked and
    how to fix it instead of serving an empty list silently.
    """
    from config import CONFIG_FILENAME, config_search_paths, get_config

    if get_config().sources:
        return
    searched = ', '.join(str(p) for p in config_search_paths())
    _logger.warning(
        'Ariadne MCP starting with NO sources configured — every '
        'source-scoped query will fail. No %s with sources was found. '
        'Searched: %s. Fix: set ARIADNE_CONFIG=/abs/path/%s, or launch from '
        'the project directory.',
        CONFIG_FILENAME,
        searched,
        CONFIG_FILENAME,
    )


# ---------------------------------------------------------------------------
# Server
# ---------------------------------------------------------------------------

mcp = FastMCP(
    'ariadne',
    instructions=(
        'Ariadne is your PRIMARY knowledge source for conceptual and '
        'architectural questions about the codebase. It contains curated, '
        'LLM-generated documentation with context that raw code searches '
        'cannot provide.\n\n'
        'When to use which tool:\n'
        '- ariadne_search: conceptual/architectural questions ("how does X work?", '
        '"what pattern does Y use?")\n'
        '- ariadne_list_all: structural queries (what docs exist, coverage audits) '
        '— call ONCE and filter locally, do NOT search one-by-one per file\n'
        '- ariadne_coverage: check which source files have docs vs undocumented\n'
        '- Grep/Glob/Read: file-level lookups, specific code searches\n\n'
        'Workflow:\n'
        '1. ariadne_search first for conceptual questions\n'
        '2. Use search results to guide follow-up code reads\n'
        '3. Report feedback: ariadne_log_hit (useful) or ariadne_log_miss '
        '(not useful) using the event_id from the search response\n'
        '4. Contribute back: when you discover or explain something significant '
        "about the codebase that Ariadne didn't already have, save it with "
        'ariadne_contribute so future sessions benefit\n\n'
        'Fall back to direct codebase tools when Ariadne lacks coverage '
        'for your specific query.\n\n'
        'If Myproject is also available, use this decision tree:\n'
        '- Question about the codebase? → ariadne_search\n'
        '- Small task (1-2 files)? → myproject_submit\n'
        '- Larger task (3+ files)? → myproject_decompose (breaks it into planned subtasks)\n'
        '- Already working on a task? → myproject_worker for status updates\n\n'
        'Combined workflow for implementation:\n'
        '1. ariadne_search — understand the domain first\n'
        '2. myproject_decompose — break down and plan the task\n'
        '3. myproject_plan — generate file manifests via Ariadne\n'
        '4. Direct codebase reads only to fill gaps\n\n'
        'Note: Search results include full document content for best precision. '
        'If responses seem large, set MAX_MCP_OUTPUT_TOKENS=50000 in your '
        'Claude Code project settings to suppress the warning.\n\n'
        'SOURCE RESOLUTION (load-bearing for cross-source isolation):\n'
        "Most Ariadne tools take a ``source=`` argument that names the project/\n"
        "component the user is asking about. Ariadne uses it to scope reads to\n"
        "that source's dependency closure — without it, the user can see docs\n"
        "from sibling projects they didn't ask about.\n\n"
        'Resolution rules:\n'
        "1. EXTRACT source from the user's free-text framing. Common patterns:\n"
        '   - "In PROJECT, how does X work?" → source=\'PROJECT\'\n'
        '   - "How does X work in PROJECT?" → source=\'PROJECT\'\n'
        '   - "For PROJECT, …" / "PROJECT side, …" → source=\'PROJECT\'\n'
        '2. If the user names multiple projects ("How does PROJECT_A use\n'
        '   PROJECT_B?"), the OUTER project (the one the user is reasoning\n'
        "   from) is the source. PROJECT_B's content will surface naturally\n"
        "   if PROJECT_A depends on it (closure semantics handle this).\n"
        '3. If the user does NOT name a project, ASK them which project they\n'
        '   mean before calling. Do not guess from the question content; do\n'
        '   not rely on default_source — silently picking a default exposes\n'
        "   the wrong project's docs.\n"
        '4. If you call without ``source=`` and the tool returns a\n'
        '   ``LookupError`` (or an error mentioning "no source context"),\n'
        "   that's the fail-closed signal: ask the user to name the project."
    ),
)


# ---------------------------------------------------------------------------
# Retrieval tools (read-only)
# ---------------------------------------------------------------------------

@mcp.tool(annotations=ToolAnnotations(
    title='Search Docs',
    readOnlyHint=True,
    openWorldHint=False,
))
async def ariadne_search(
    query: str | None = None,
    feature: str | None = None,
    branch: str | None = None,
    status: str | None = None,
    context_file: str | None = None,
    limit: int = 10,
    sections_only: bool = False,
    source: str | None = None,
    role: str = 'developer',
) -> SearchResponse:
    """Search Ariadne documentation with optional filters for branch-aware retrieval.

    Use this FIRST before any direct codebase exploration (Grep, Glob, Read, ls, find).

    ``source`` is the load-bearing scope argument: it identifies which
    project/component the user is asking about. Extract it from the
    user's free-text framing (e.g., "in PROJECT, how does X work?"
    → ``source='PROJECT'``). If the user's question doesn't name a
    project AND no default_source is configured, this call will
    fail-closed with a ``LookupError`` — DO NOT guess; instead ask the
    user which project they mean.

    Args:
        query: Search query for documentation content.
        feature: Filter by feature name or alias (e.g., "combinatorial-sql-builder").
        branch: Filter docs matching this branch pattern (default: auto-detect current branch).
        status: Filter by status (stable, experimental, deprecated).
        context_file: File you're currently working on — boosts docs for related files.
        limit: Maximum results to return.
        sections_only: Return only relevant sections instead of full documents. Saves tokens.
        source: The project/component the user is asking about. Extract
            from the user's question; if not present, ask the user
            rather than guessing.
        role: Audience for results — 'developer' (default) or
            'product_manager' (audience-adapted retrieval).
    """
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    if branch is None:
        branch = svc.get_branch()
    return await svc.search(query, feature, branch, status, limit, context_file=context_file, sections_only=sections_only, source=source, role=role)


@mcp.tool(annotations=ToolAnnotations(
    title='List All Docs',
    readOnlyHint=True,
    openWorldHint=False,
))
def ariadne_list_all(
    include_expired: bool = False,
    source: str | None = None,
) -> ListResponse:
    """List documents within the resolved source's closure.

    ``source`` identifies which project the user is asking about
    (extracted from free-text framing). If unset and no default_source
    is configured, the call fails-closed — ask the user rather than
    guessing.

    Shows document status, branches, and expiration info.

    Args:
        include_expired: Include documents past their expiration date.
        source: The project/component the user is asking about. Extract
            from the user's question; if not present, ask the user
            rather than guessing.
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().list_all(include_expired, source=source)


@mcp.tool(annotations=ToolAnnotations(
    title='Branch Status',
    readOnlyHint=True,
    openWorldHint=False,
))
def ariadne_branch_status(source: str | None = None) -> BranchStatusResponse:
    """Show documents affected by current branch changes vs main.

    Use this to see which docs may be stale due to branch modifications.

    Args:
        source: Source name (optional, uses default if not specified).
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().branch_status(source)


@mcp.tool(annotations=ToolAnnotations(
    title='Sync Status',
    readOnlyHint=True,
    openWorldHint=False,
))
def ariadne_sync_status(source: str | None = None) -> SyncStatusResponse:
    """Check when Ariadne documentation was last synced.

    Use this when asked "when was Ariadne last updated?" or similar.

    Args:
        source: Source name (optional, uses default if not specified).
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().sync_status(source)


# ---------------------------------------------------------------------------
# Feedback tools (additive)
# ---------------------------------------------------------------------------

@mcp.tool(annotations=ToolAnnotations(
    title='Log Hit',
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
))
def ariadne_log_hit(
    event_id: int,
    feedback: str | None = None,
) -> FeedbackResponse:
    """Report that an Ariadne tool result was useful.

    Call this after using information from an Ariadne tool result.

    Args:
        event_id: The usage event ID from a previous tool call.
        feedback: Optional feedback about what was helpful or what could be better.
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().log_hit(event_id, feedback)


@mcp.tool(annotations=ToolAnnotations(
    title='Log Miss',
    readOnlyHint=False,
    destructiveHint=False,
    idempotentHint=True,
    openWorldHint=False,
))
def ariadne_log_miss(
    event_id: int,
    feedback: str = '',
) -> FeedbackResponse:
    """Report that an Ariadne tool result was not useful.

    Call this when Ariadne was consulted but didn't have what was needed.
    The feedback helps identify documentation gaps for future improvement.

    Args:
        event_id: The usage event ID from a previous tool call.
        feedback: Description of what was missing or needed.
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().log_miss(event_id, feedback)
                                                                                                                                                                        
                
@mcp.tool()
def ariadne_expand(
    event_id: int,
) -> dict:
    """Return the full content of documents referenced by a prior search event.                                                                                         
 
    Use when a previous `ariadne_search` response was truncated to sections                                                                                             
    (due to response_token_budget) and sections turned out to be insufficient.
    The caller supplies the `event_id` from the truncated response and receives                                                                                         
    the full content of the same documents.
                                                                                                                                                                        
    Args:
        event_id: The usage event ID from the prior truncated `ariadne_search`.
                                                                                                                                                                        
    Returns:
        {                                                                                                                                                               
            "event_id": int,
            "original_query": str | None,
            "documents": list[dict],
            "missing_document_ids": list[str],  # only if some source docs deleted
        }                                                                                                                                                               
        Or on error: {"error": str, "event_id": int}
    """                                                                                                                                                                 
    from ariadne_mcp.service import AriadneService

    # Delegates to the service so the spool-scope gate (CRIT-12) lives
    # in one place: a disabled spool's docs are not re-surfaced by id.
    return AriadneService.get().expand(event_id)


# ---------------------------------------------------------------------------
# Register tools from submodules
# ---------------------------------------------------------------------------

from ariadne_mcp.server_admin import register_tools as _register_admin
from ariadne_mcp.server_analytics import register_tools as _register_analytics
from ariadne_mcp.server_debug import register_tools as _register_debug
from ariadne_mcp.server_knowledge import register_tools as _register_knowledge

_register_knowledge(mcp)
_register_debug(mcp)
_register_analytics(mcp)
_register_admin(mcp)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import asyncio
    import atexit
    import os

    def _cleanup() -> None:
        """Clean up AriadneService and LLM client resources on shutdown."""
        try:
            import llm
            from ariadne_mcp.service import AriadneService

            loop = asyncio.new_event_loop()
            if AriadneService._instance is not None:
                loop.run_until_complete(AriadneService._instance.close())
            loop.run_until_complete(llm.close())
            loop.close()
        except Exception:
            pass  # Best-effort cleanup at shutdown

    atexit.register(_cleanup)
    _warn_if_no_sources()

    # The lowlevel MCP server logs an INFO line per request ("Processing
    # request of type CallToolRequest") — one per UI click under `ariadne
    # serve`, and per tool call in the agent's MCP logs. Quiet it to WARNING
    # so real problems still surface but the per-call chatter doesn't. Set
    # ARIADNE_MCP_VERBOSE=1 to restore it.
    if not os.environ.get('ARIADNE_MCP_VERBOSE'):
        logging.getLogger('mcp.server.lowlevel.server').setLevel(logging.WARNING)

    mcp.run(transport='stdio')
                                                                                                                                                                  
                
@mcp.tool(annotations=ToolAnnotations(
    title='Notify Changed Files',
    readOnlyHint=False,
    destructiveHint=False,
    openWorldHint=False,
))
async def ariadne_notify_changed(                                                                                                                                       
    source: str,                                                                                                                                                        
    files: list[str],
    regenerate: bool = False,                                                                                                                                           
) -> dict:      
    """Notify Ariadne that a batch of files in `source` has changed.
                                                                                                                                                                        
    Incrementally updates the catalog: adds new elements, updates modified
    ones, removes deleted elements, detects cross-file moves, and deletes                                                                                               
    file_index docs for files that no longer exist.                                                                                                                     
                                                                                                                                                                        
    If ``regenerate=True``, after the catalog is synced the LLM-written docs                                                                                            
    for the changed files are also refreshed (explanation/architecture/etc.).                                                                                           
    Default False preserves prior behavior (structural-only update).                                                                                                    
                                                                                                                                                                        
    Returns a per-file summary dict:                                                                                                                                    
        {rel_path: {added, modified, removed, moved, unchanged, deleted}}                                                                                               
    plus, if regenerate was requested:                                                                                                                                  
        {"_regen": {"docs_created": int, "docs_failed": int}}                                                                                                           
    """                                                                                                                                                                 
    from pathlib import Path as _P

    from config import get_config
    from docgen.catalog_writer import notify_changed as _notify
    from library import Library
    from writer import LibraryWriter
                                                                                                                                                                        
    cfg = get_config()
    library = Library(cfg.db_path)                                                                                                                                      
    try:                                                                                                                                                                
        async with LibraryWriter(library) as writer:
            result = await _notify(library, writer, source, files)                                                                                                      
                
        if regenerate:
            from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig
 
            source_path = cfg.resolve_source(source)                                                                                                                    
            if source_path is None or not source_path.exists():                                                                                                         
                result['_regen'] = {'error': f'source path not found: {source_path}'}                                                                                   
                return result                                                                                                                                           
                                                                                                                                                                        
            orch_config = OrchestratorConfig(                                                                                                                           
                source_path=source_path,
                db_path=cfg.db_path,
                staleness_db_path=_P(cfg.staleness_db_path),                                                                                                            
                model=cfg.model,
                doc_types=('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram'),                                                                        
                force_regenerate=True,                                                                                                                                  
            )                                                                                                                                                           
                                                                                                                                                                        
            docs_created = 0
            docs_failed = 0                                                                                                                                             
            async with DocGenOrchestrator(orch_config) as orchestrator:
                for rel in files:                                                                                                                                       
                    file_path = source_path / rel if not _P(rel).is_absolute() else _P(rel)                                                                             
                    if not file_path.exists():                                                                                                                          
                        continue                                                                                                                                        
                    try:                                                                                                                                                
                        gen = await orchestrator._process_file(file_path)
                        if gen is not None:                                                                                                                             
                            docs_created += getattr(gen, 'docs_generated', 0)
                            docs_failed += getattr(gen, 'docs_failed', 0)                                                                                               
                    except Exception:                                                                                                                                   
                        docs_failed += 1
                                                                                                                                                                        
            result['_regen'] = {'docs_created': docs_created, 'docs_failed': docs_failed}
                                                                                                                                                                        
        return result
    finally:                                                                                                                                                            
        library.close()
@mcp.tool(annotations=ToolAnnotations(
    title='Read File',
    readOnlyHint=True,
    openWorldHint=False,
))
async def ariadne_read(
    file_path: str,
    offset: int | None = None,                                                                                                                                          
    limit: int | None = None,
) -> dict:                                                                                                                                                              
    """Read a file's content, optionally sliced by line range.

    Mirrors Claude Code's Read tool parameter shape (file_path/offset/limit)                                                                                            
    so a PreToolUse mcp_tool hook can redirect Read -> ariadne_read.
                                                                                                                                                                        
    Args:
        file_path: Absolute or relative path to the file.                                                                                                               
        offset: 1-indexed starting line (default: 1).                                                                                                                   
        limit: Max number of lines to return (default: all).
                                                                                                                                                                        
    Returns:
        {"content": str, "file": str, "line_start": int, "line_end": int}                                                                                               
        or {"error": str} on failure.                                                                                                                                   
    """
    from pathlib import Path as _P
    p = _P(file_path)                                                                                                                                                   
    if not p.is_absolute():
        p = p.resolve()                                                                                                                                                 
    if not p.exists():
        return {'error': f'file not found: {file_path}'}
    if p.is_dir():                                                                                                                                                      
        return {'error': f'is a directory: {file_path}'}
    try:                                                                                                                                                                
        text = p.read_text(encoding='utf-8')                                                                                                                            
    except (OSError, UnicodeDecodeError) as e:
        return {'error': f'read failed: {e}'}                                                                                                                           
    lines = text.splitlines(keepends=True)
    total = len(lines)                                                                                                                                                  
    start = max(1, offset) if offset else 1
    end = min(total, start + limit - 1) if limit else total                                                                                                             
    sliced = ''.join(lines[start - 1:end])
    return {                                                                                                                                                            
        'content': sliced,
        'file': str(p),                                                                                                                                                 
        'line_start': start,
        'line_end': end,
        'total_lines': total,
    }


@mcp.tool(annotations=ToolAnnotations(
    title='Look up catalog symbol',                                                                                                                                                  
    readOnlyHint=True,                                                                                                                                                               
    openWorldHint=False,                                                                                                                                                             
))                                                                                                                                                                                   
async def ariadne_symbol(
    qualified_name: str,                                                                                                                                                             
    source: str | None = None,
    file: str | None = None,                                                                                                                                                         
) -> dict:      
    """Look up a single catalog element by qualified_name.
                                                                                                                                                                                     
    Direct DB lookup (no LLM). Returns the element's structural info when found,
    or fuzzy suggestions when not. Use this to verify a symbol exists before                                                                                                         
    editing it, or to find nearest matches for a misremembered name.                                                                                                                 
    """                                                                                                                                                                              
    from config import get_config
    from docgen.catalog_lookup import lookup_symbol
    from library import Library
 
    cfg = get_config()                                                                                                                                                               
    source_name = source or cfg.default_source
    if source_name is None:                                                                                                                                                          
        return {'error': 'no_source', 'message': 'No source specified and no default_source in config'}
                                                                                                                                                                                     
    library = Library(cfg.db_path)
    try:                                                                                                                                                                             
        return lookup_symbol(library, source_name, file, qualified_name)
    finally:                                                                                                                                                                         
        library.close()


@mcp.tool()
async def ariadne_body(
    qualified_name: str,
    source: str | None = None,
    file: str | None = None,
) -> dict:
    """Return current body text of a catalog element by qualified_name."""
    from config import get_config
    from docgen.catalog_lookup import get_element_body
    from library import Library

    cfg = get_config()
    source_name = source or cfg.default_source
    if source_name is None:
        return {'error': 'no_source', 'message': 'No source specified and no default_source in config'}

    library = Library(cfg.db_path)
    try:
        return get_element_body(library, source_name, file, qualified_name)
    finally:
        library.close()


@mcp.tool(annotations=ToolAnnotations(
    title='Find where a config key is read',
    readOnlyHint=True,
    openWorldHint=False,
))
async def ariadne_config_usage(
    key: str,
    source: str | None = None,
) -> dict:
    """Bridge a config key to the code that reads it.

    Returns the key's literal default (from the catalog) plus the code sites
    that read it. Read sites come from the call-site-verified ``config_reads``
    index when available — each with its resolved value and per-site
    'config-resolved'/'string-match' confidence, including split-path
    ``getConfig("a").getString("b")`` reads — falling back to a literal
    string-match for keys not yet in that index. Use for 'where is <lever>
    read / what is its default'.
    """
    from config import get_config
    from docgen.catalog_lookup import config_usage
    from library import Library

    cfg = get_config()
    source_name = source or cfg.default_source
    if source_name is None:
        return {'error': 'no_source', 'message': 'No source specified and no default_source in config'}

    library = Library(cfg.db_path)
    try:
        return config_usage(library, source_name, key)
    finally:
        library.close()


@mcp.tool(annotations=ToolAnnotations(
    title='Cross-cutting Themes',
    readOnlyHint=True,
    openWorldHint=False,
))
async def ariadne_themes(
    action: str = 'list',
    cluster_id: str | None = None,
    coherent_only: bool = True,
    source: str | None = None,
    limit: int = 50,
) -> dict:
    """Inspect cross-cutting themes (cluster-level theme docs).

    Themes are clusters of catalog elements discovered by community detection
    over the hybrid (structural + semantic) graph. See docs/cross-cutting-themes-leiden-plan.md.

    Args:
        action: 'list' (default — list themes), 'get' (full theme doc by cluster_id),
                or 'members' (member elements with weights).
        cluster_id: Required for 'get' and 'members'.
        coherent_only: For 'list' — exclude themes the LLM judged INCOHERENT.
        source: Optional source name filter.
        limit: For 'list' — max number of themes to return.
    """
    from config import get_config
    from library import Library
    from ariadne_mcp.service_themes import themes_action

    cfg = get_config()
    library = Library(cfg.db_path)
    try:
        # Themes are library-internal cross-source — themes_action takes
        # the raw library, not a ScopedLibrary. The ``source`` argument
        # is forwarded to the SQL JOIN filter (single-source match,
        # legacy behavior) without closure expansion.
        return themes_action(
            library,
            action=action,  # type: ignore[arg-type]
            cluster_id=cluster_id,
            coherent_only=coherent_only,
            source=source,
            limit=limit,
        )
    finally:
        library.close()

