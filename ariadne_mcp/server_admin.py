"""Admin tools for the Ariadne MCP server."""
from __future__ import annotations

from mcp.server.fastmcp import Context

from docgen.trace_flow_llm_bridge import build_llm_bridge
from ariadne_mcp.models import (
    AdminActionResponse,
    DiscoverResponse,
    EstimateResponse,
    SourceAddResponse,
    SourceListResponse,
)


async def ariadne_branch_sync(
    source: str | None = None,
    dry_run: bool = True,
    ctx: Context | None = None,
) -> AdminActionResponse:
    """Regenerate affected documents for current branch.

    Creates branch-specific experimental docs with TTL.

    Args:
        source: Source name (optional).
        dry_run: If true, only show what would be done.
    """
    from ariadne_mcp.service import AriadneService

    if ctx and not dry_run:
        await ctx.info('Regenerating branch-specific documents...')
    return AriadneService.get().branch_sync(source, dry_run)


async def ariadne_generate(
    file_path: str,
    source_name: str | None = None,
    ctx: Context | None = None,
) -> AdminActionResponse:
    """Generate documentation for a source file or directory on demand.

    Uses the DocGenOrchestrator directly (in-process) rather than shelling
    out to the CLI. Generates explanation and architecture docs.

    Args:
        file_path: File or directory path (relative to source root).
        source_name: Source name (optional, uses default if not specified).
    """
    from ariadne_mcp.service import AriadneService

    if ctx:
        await ctx.info(f'Generating docs for {file_path}')
    return await AriadneService.get().generate_file(file_path, source_name)


async def ariadne_improve(
    source: str | None = None,
    max_files: int = 10,
    days: int = 30,
    dry_run: bool = True,
    ctx: Context | None = None,
) -> AdminActionResponse:
    """Run improvement cycle: analyze gaps and generate docs for undocumented files.

    Analyzes usage gaps, finds undocumented source files, and generates
    documentation for the highest-priority gaps. Use dry_run=True to preview.

    Args:
        source: Source name (optional, uses default).
        max_files: Max files to generate docs for (default: 10).
        days: Days of usage data to analyze (default: 30).
        dry_run: If true, only show what would be done.
    """
    import subprocess

    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    source_name = svc._resolve_source(source) or ''

    if ctx:
        await ctx.info(f'Running improvement cycle (source={source_name}, dry_run={dry_run})')

    cmd = [
        'uv', 'run', 'ariadne', 'improve',
        '-s', source_name,
        '--max-files', str(max_files),
        '--days', str(days),
    ]
    if dry_run:
        cmd.append('--dry-run')

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=600,
        )
        output = result.stdout + result.stderr
        return AdminActionResponse(output=output or '(no output)')
    except subprocess.TimeoutExpired:
        return AdminActionResponse(output='Improvement cycle timed out after 600s.')
    except Exception as e:
        return AdminActionResponse(output=f'Error: {e}')


async def ariadne_discover_source(
    source: str | None = None,
    all: bool = False,
    dry_run: bool = False,
    ctx: Context | None = None,
) -> AdminActionResponse:
    """Walk a source tree and write ``.ariadne/manifest.json`` with the
    detected SCIP indexer plan. Adds ``.ariadne/`` to the source's
    ``.gitignore``. Idempotent — re-runs preserve the existing
    ``.gitignore`` line.

    Use this once per source after adding it to ariadne.yaml; ariadne
    auto-detects which SCIP indexers (python/typescript/java) apply
    based on filesystem markers (``__init__.py``, ``package.json``,
    ``build.sbt``/``pom.xml``/``build.gradle*``).

    Args:
        source: Source name (optional, uses default if not specified).
        all: Discover every source in ariadne.yaml.
        dry_run: Preview manifest without writing files.
    """
    import subprocess

    cmd = ['uv', 'run', 'ariadne', 'discover']
    if source:
        cmd.extend(['--source', source])
    if all:
        cmd.append('--all')
    if dry_run:
        cmd.append('--dry-run')

    if ctx and not dry_run:
        await ctx.info(f'Running ariadne discover (source={source})')

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=120,
        )
        output = (result.stdout or '') + (result.stderr or '')
        return AdminActionResponse(output=output or '(no output)')
    except subprocess.TimeoutExpired:
        return AdminActionResponse(
            output='ariadne discover timed out after 120s.',
        )
    except Exception as e:
        return AdminActionResponse(output=f'Error: {e}')


async def ariadne_index_source(
    source: str | None = None,
    all: bool = False,
    dry_run: bool = False,
    kind: str | None = None,
    ctx: Context | None = None,
) -> AdminActionResponse:
    """Run SCIP indexers per the source's manifest, merge intermediates
    into ``<source>/.ariadne/index.scip``.

    Hard-fails on any indexer or merge error (no fallback). Each
    indexer entry's manifest gets ``scip_path``, ``indexed_at``, and
    ``indexer_version`` populated; subsequent ariadne sync reads these
    to load the cross-source graph.

    First run on a fresh source can be minutes-long (especially for
    JVM projects where scip-java triggers compilation). Re-runs are
    incremental where the indexer supports it.

    Args:
        source: Source name (optional, uses default).
        all: Index every source in ariadne.yaml.
        dry_run: Show what would run without invoking indexers.
        kind: Run only entries of this indexer kind
            (python/typescript/java); skip the rest.
    """
    import subprocess

    cmd = ['uv', 'run', 'ariadne', 'index']
    if source:
        cmd.extend(['--source', source])
    if all:
        cmd.append('--all')
    if dry_run:
        cmd.append('--dry-run')
    if kind:
        cmd.extend(['--kind', kind])

    if ctx and not dry_run:
        await ctx.info(
            f'Running ariadne index (source={source}, kind={kind})',
        )

    # Long timeout — scip-java on a large Scala codebase can run for
    # several minutes during initial compilation.
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=1800,
        )
        output = (result.stdout or '') + (result.stderr or '')
        return AdminActionResponse(output=output or '(no output)')
    except subprocess.TimeoutExpired:
        return AdminActionResponse(
            output='ariadne index timed out after 1800s.',
        )
    except Exception as e:
        return AdminActionResponse(output=f'Error: {e}')


def ariadne_self_improve(
    event_id: int,
) -> AdminActionResponse:
    """Diagnose a low-score usage event and regenerate docs to fill the gap.

    Call this with an event_id from a search that scored ≤5. Ariadne will:
    1. Parse the reason from the feedback
    2. Check if the reason is specific and actionable
    3. Identify which source files need better docs
    4. Regenerate documentation targeting the identified gap

    Skips vague reasons ("not helpful") and areas without source files.

    Args:
        event_id: The usage event ID that received a low score.
    """
    from ariadne_mcp.service import AriadneService

    result = AriadneService.get().self_improve(event_id)
    action = result.get('action_taken', False)
    detail = result.get('detail', '')
    diagnosis = result.get('diagnosis', {})

    lines = [f'Self-improve for event {event_id}:']
    lines.append(f'  Action taken: {"yes" if action else "no"}')
    lines.append(f'  Detail: {detail}')
    if diagnosis.get('reason'):
        lines.append(f'  Reason: {diagnosis["reason"]}')
    if result.get('regenerated_files'):
        lines.append(f'  Regenerated: {", ".join(result["regenerated_files"])}')
    return AdminActionResponse(output='\n'.join(lines))


async def ariadne_merge(
    source: str | None = None,
    dry_run: bool = True,
    delete_consumed: bool = False,
    ctx: Context | None = None,
) -> AdminActionResponse:
    """Detect merged branches and regenerate stable docs.

    Use after PRs are merged to main.

    Args:
        source: Source name (optional, uses default if not specified).
        dry_run: Preview only — show what would be done.
        delete_consumed: Delete branch docs instead of deprecating. Destructive — requires user approval.
    """
    from ariadne_mcp.service import AriadneService

    if ctx and not dry_run:
        await ctx.info('Processing merged branch documents...')
    return AriadneService.get().merge(source, dry_run, delete_consumed)


async def ariadne_generate_docs(
    types: str = 'all',
    output_dir: str = 'generated-docs',
    source: str | None = None,
    update_readme: str | None = None,
) -> dict:
    """Generate user-facing documentation from Ariadne's knowledge base.

    Assembles curated knowledge into polished markdown docs using LLM.

    Args:
        types: Comma-separated doc types: readme, api, architecture, all (default: all).
        output_dir: Output directory for generated docs (default: generated-docs).
        source: Filter to a specific source (e.g. "pythonproject"). Omit for all sources.
        update_readme: Path to existing README.md to update in-place (instead of generating fresh).
    """
    from pathlib import Path

    from doc_generator import DOC_TYPES, DocGenerator
    from ariadne_mcp.service import AriadneService

    svc = AriadneService.get()
    gen = DocGenerator(svc.library, Path(output_dir), source=source)

    if update_readme:
        result = await gen.update_readme_in_place(Path(update_readme))
        return {'updated': str(result)}

    doc_types = list(DOC_TYPES) if types == 'all' else [t.strip() for t in types.split(',')]
    results = await gen.generate(doc_types)
    return {'generated': {k: str(v) for k, v in results.items()}}


async def ariadne_trace_flow(
    start_symbol: str,
    depth: int = 5,
    enable_llm_bridge: bool = False,
    include_diagram: bool = False,
    ctx: Context | None = None,
) -> dict:
    """Trace cross-language flow from a starting symbol — the SCIP-
    everywhere user payoff.

    Walks the combined graph (SCIP edges + api_calls / api_endpoints +
    process_invocations) from ``start_symbol`` and returns hops with
    per-tier provenance:

    - ``'scip'`` — within-language SCIP edge (compiler-grade)
    - ``'swagger'`` — HTTP boundary resolved from a Swagger spec
    - ``'pattern'`` — HTTP boundary resolved from framework patterns
    - ``'process'`` — subprocess/script invocation (Layer C)

    Tier priority: SCIP first, then HTTP, then process. Within-language
    locality is preferred — boundary hops only fire when the within-
    language graph runs out for a given cursor. Cycle-protected;
    depth-bounded with a ``truncated`` flag in the response.

    Args:
        start_symbol: Starting canonical_id (e.g., from
            ``ariadne_callers``/``ariadne_callees`` output).
        depth: Maximum hop depth to walk (default: 5).
        include_diagram: When True, add a ``diagram`` field to the response —
            a fenced Graphviz DOT sequence diagram of the trace (lifelines =
            sources, dashed = HTTP hops). Include that ```dot block verbatim
            in a chat reply to render the cross-repo flow as an image where
            supported (e.g. the Slack bridge renders it in-thread).
    """
    import sqlite3

    from cli.trace import trace_result_to_dict
    from config import get_config
    from docgen.trace_flow import trace_flow

    if ctx:
        await ctx.info(
            f'Tracing flow from {start_symbol} (depth={depth})',
        )

    bridge = None
    if enable_llm_bridge:
        bridge = build_llm_bridge()

    cfg = get_config()
    conn = sqlite3.connect(cfg.db_path)
    try:
        result = trace_flow(
            start_symbol=start_symbol,
            depth=depth,
            conn=conn,
            llm_bridge=bridge,
        )
        response = trace_result_to_dict(result)
        if include_diagram:
            # Resolve hop symbol-ids → sources against the open graph, then
            # emit a fenced DOT sequence diagram the bridge renders in-thread.
            from diagram_format import fence_dot
            from docgen.trace_flow_sequence import (
                render_sequence_dot,
                trace_to_messages,
            )

            messages = trace_to_messages(result, conn)
            response['diagram'] = fence_dot(render_sequence_dot(messages))
    finally:
        conn.close()

    return response


async def ariadne_docs_read(
    doc_type: str = 'readme',
    output_dir: str = 'generated-docs',
) -> dict:
    """Read a previously generated doc file — zero LLM cost.

    Use this as a preliminary step before any codebase exploration to get
    project context cheaply. Available types: readme, api, architecture,
    faq, decisions, diff, diagrams, notebooks.

    Args:
        doc_type: Which doc to read (default: readme).
        output_dir: Directory where generated docs live (default: generated-docs).
    """
    from pathlib import Path

    from doc_generator import DOC_TYPE_TO_FILE

    filename = DOC_TYPE_TO_FILE.get(doc_type)
    if filename is None:
        return {'error': f'Unknown doc type: {doc_type}', 'available': list(DOC_TYPE_TO_FILE.keys())}

    path = Path(output_dir) / filename
    if not path.exists():
        return {'error': f'Doc not generated yet. Run: ariadne docs generate --type {doc_type}', 'path': str(path)}

    content = path.read_text(encoding='utf-8')
    return {'type': doc_type, 'path': str(path), 'content': content}


async def ariadne_source_add(
    name: str,
    path: str | None = None,
    depends_on: list[str] | None = None,
    parent: str | None = None,
    branches: list[str] | None = None,
    ref: str | None = None,
    exclude: list[str] | None = None,
    exclude_dirs: list[str] | None = None,
    exempt_dirs: list[str] | None = None,
    ignore_staleness: bool | None = None,
    set_default: bool | None = None,
    doc_types_by_language: dict | None = None, ctx: Context | None = None,
) -> SourceAddResponse:
    """Add or update a source in ariadne.yaml — the onboarding "Connect"/"Scope" step.

    Mirrors ``ariadne source add``: bootstraps ariadne.yaml when a project
    has none, makes the first source the default, and on re-runs updates
    only the fields you pass (idempotent). Returns the persisted source
    config plus detected git metadata (branch / file count / size / last
    commit) for the connect screen.

    Run ``ariadne_discover`` next to detect languages and the index plan.

    Args:
        name: Source name — scopes queries; lowercase, no spaces.
        path: Filesystem path to the source root. Required for a new
            source; optional when updating an existing one.
        depends_on: Source names to load as context.
        parent: Parent source, for a subdirectory source.
        branches: Git branch patterns this source is active on.
        ref: Pin indexing to a fixed git ref instead of the working tree.
        exclude: Glob patterns to exclude (e.g. ``**/*.min.js``).
        exclude_dirs: Directory names to exclude (e.g. ``build``).
        exempt_dirs: Directory names to FORCE-INCLUDE even though a default
            policy would skip them (e.g. ``vendor``, ``tests``) — the
            override knob for the default excludes.
        ignore_staleness: Exempt this source from staleness checks.
        set_default: Force or refuse default-source status; defaults to the
            "first source becomes default" rule.
    """
    from ariadne_mcp.service import AriadneService

    if ctx:
        await ctx.info(f'Writing source {name!r} to ariadne.yaml')
    return AriadneService.get().source_add(
        name,
        path=path,
        depends_on=depends_on,
        parent=parent,
        branches=branches,
        ref=ref,
        exclude=exclude,
        exclude_dirs=exclude_dirs,
        exempt_dirs=exempt_dirs,
        ignore_staleness=ignore_staleness,
        set_default=set_default,
    doc_types_by_language=doc_types_by_language)


async def ariadne_estimate(
    source: str | None = None,
    model: str | None = None,
    doc_types: list[str] | None = None,
    ctx: Context | None = None,
) -> EstimateResponse:
    """Estimate the LLM cost of documenting a source — the onboarding "Preview" step.

    Pure preview, no LLM calls: walks the source (honoring its excludes) and
    runs the same cost model ``ariadne dry-run`` uses. Returns totals (live
    + ~50%-off batched), a per-directory/per-file cost tree (the explorer),
    a per-doc-type split, the detected language histogram, the model price
    list, and which doc types apply per language — everything the preview
    needs so a user can trim scope and pick a model/speed before paying for
    ``ariadne_onboard``.

    Args:
        source: Source name (defaults to the configured default_source).
        model: Model to price against (defaults to the configured model).
        doc_types: Doc types to include (defaults to the standard set).
    """
    from ariadne_mcp.service import AriadneService

    if ctx:
        await ctx.info(f'Estimating documentation cost for {source or "(default)"}')
    return AriadneService.get().estimate(source, model=model, doc_types=doc_types)


async def ariadne_discover(
    source: str | None = None,
    ctx: Context | None = None,
) -> DiscoverResponse:
    """Detect languages and the SCIP index plan for a source — the onboarding "Discover" step.

    Walks the source tree, detects which SCIP indexers apply
    (python/typescript/java) from filesystem markers, writes
    ``<source>/.ariadne/manifest.json``, and auto-authors the
    ``index_kinds``/``scip`` block in ariadne.yaml. Returns the structured
    plan: the detected-language histogram, the per-language indexer plan,
    ``index_kinds``, and file/dir counts.

    This is the structured, in-process counterpart of
    ``ariadne_discover_source`` (which shells out and returns plain text).
    Sequence: ``ariadne_source_add`` → ``ariadne_discover`` →
    ``ariadne_estimate`` (cost preview) → ``ariadne_onboard`` (build).

    Args:
        source: Source name (defaults to the configured default_source).
    """
    from ariadne_mcp.service import AriadneService

    if ctx:
        await ctx.info(
            f'Discovering languages and index plan for {source or "(default)"}')
    return AriadneService.get().discover(source)


def ariadne_list_sources() -> SourceListResponse:
    """List the sources configured in ariadne.yaml.

    Returns each source's name, path, default flag, and declared
    dependencies. Used by the onboarding dependency picker to offer the
    project's existing sources, and useful any time you need to know what's
    configured without reading ariadne.yaml directly.
    """
    from ariadne_mcp.service import AriadneService

    return AriadneService.get().list_sources()


def register_tools(mcp) -> None:
    """Register admin tools with the MCP server."""
    from mcp.types import ToolAnnotations

    mcp.tool(annotations=ToolAnnotations(
        title='Branch Sync',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_branch_sync)

    mcp.tool(annotations=ToolAnnotations(
        title='Generate Docs',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_generate)

    mcp.tool(annotations=ToolAnnotations(
        title='Improve Library',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_improve)

    mcp.tool(annotations=ToolAnnotations(
        title='Self-Improve',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_self_improve)

    mcp.tool(annotations=ToolAnnotations(
        title='Discover SCIP Indexer Locations',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_discover_source)

    mcp.tool(annotations=ToolAnnotations(
        title='Run SCIP Indexers',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_index_source)

    mcp.tool(annotations=ToolAnnotations(
        title='Merge Branches',
        readOnlyHint=False,
        destructiveHint=True,
        idempotentHint=False,
        openWorldHint=False,
    ))(ariadne_merge)

    mcp.tool(annotations=ToolAnnotations(
        title='Generate Docs',
        readOnlyHint=False,
    ))(ariadne_generate_docs)

    mcp.tool(annotations=ToolAnnotations(
        title='Read Generated Docs',
        readOnlyHint=True,
    ))(ariadne_docs_read)

    mcp.tool(annotations=ToolAnnotations(
        title='Add Source',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_source_add)

    mcp.tool(annotations=ToolAnnotations(
        title='Estimate Documentation Cost',
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_estimate)

    mcp.tool(annotations=ToolAnnotations(
        title='Discover Languages and Index Plan',
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_discover)

    mcp.tool(annotations=ToolAnnotations(
        title='List Configured Sources',
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_list_sources)

    mcp.tool(annotations=ToolAnnotations(
        title='Trace Cross-Language Flow',
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ))(ariadne_trace_flow)
