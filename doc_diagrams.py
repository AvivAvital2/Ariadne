"""Extracted doc generation handlers (diagrams, architecture, diff).

Each function takes a ``DocGenerator`` instance as its first argument and
returns the output ``Path``.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from doc_helpers import (
    _assemble_notebook,
    _build_code_cell,
    _build_excalidraw_elements,
    _build_markdown_cell,
    _build_svg_diagram,
    _components_to_layers,
    _extract_components_from_docs,
    _extract_flow_from_docs,
    _format_dict,
    _is_source_module,
    _parse_flow_output,
    _to_relative_path,
)

if TYPE_CHECKING:
    from doc_generator import DocGenerator

_logger = logging.getLogger(__name__)

async def generate_arch_diagrams(gen: DocGenerator) -> Path:
    """Generate architecture diagrams: system overview + component interaction."""
    import json as _json

    arch_docs = gen._list_docs('architecture', limit=30)

    components, interactions = _extract_components_from_docs(arch_docs)

    arch_dir = gen.output_dir / 'arch-diagrams'
    arch_dir.mkdir(parents=True, exist_ok=True)

    diagram_pages: list[tuple[str, str]] = []

    # 1. System Overview
    if components:
        overview_layers, overview_edges = _components_to_layers(components, interactions)
        svg = _build_svg_diagram(overview_layers, overview_edges)
        (arch_dir / 'system-overview.svg').write_text(svg, encoding='utf-8')
        elements = _build_excalidraw_elements(overview_layers, overview_edges)
        excalidraw = {'type': 'excalidraw', 'version': 2, 'elements': elements,
                      'appState': {'viewBackgroundColor': '#ffffff'}}
        (arch_dir / 'system-overview.excalidraw').write_text(
            _json.dumps(excalidraw, indent=2), encoding='utf-8')

        overview_md = ['# System Overview\n']
        overview_md.append('![System Overview](arch-diagrams/system-overview.svg)\n')
        overview_md.append('## Components\n')
        for comp_name, comp_desc in sorted(components.items()):
            overview_md.append(f'### `{comp_name}`\n{comp_desc}\n')
        (arch_dir / 'system-overview.md').write_text('\n'.join(overview_md), encoding='utf-8')
        diagram_pages.append(('System Overview', 'arch-diagrams/system-overview.md'))

    # 2. Data Flow
    pipeline_docs = [d for d in arch_docs if any(
        kw in d.content.lower() for kw in ('pipeline', 'data flow', 'sequence', 'step 1', 'step 2')
    )]
    if pipeline_docs:
        flow_components, flow_edges = _extract_flow_from_docs(pipeline_docs[:5])
        if flow_components:
            flow_layers, flow_edge_list = _components_to_layers(flow_components, flow_edges)
            svg = _build_svg_diagram(flow_layers, flow_edge_list)
            (arch_dir / 'data-flow.svg').write_text(svg, encoding='utf-8')
            elements = _build_excalidraw_elements(flow_layers, flow_edge_list)
            excalidraw = {'type': 'excalidraw', 'version': 2, 'elements': elements,
                          'appState': {'viewBackgroundColor': '#ffffff'}}
            (arch_dir / 'data-flow.excalidraw').write_text(
                _json.dumps(excalidraw, indent=2), encoding='utf-8')

            flow_md = ['# Data Flow\n']
            flow_md.append('![Data Flow](arch-diagrams/data-flow.svg)\n')
            flow_md.append('## Pipeline Steps\n')
            for comp_name, comp_desc in flow_components.items():
                flow_md.append(f'- **{comp_name}**: {comp_desc}')
            (arch_dir / 'data-flow.md').write_text('\n'.join(flow_md), encoding='utf-8')
            diagram_pages.append(('Data Flow', 'arch-diagrams/data-flow.md'))

    # 3. Deep-dive diagrams for complex concepts
    concept_keywords = {
        'threading': ('thread', 'async', 'concurrent', 'parallel', 'to_thread'),
        'feature-pipeline': ('feature', 'generation', 'selection', 'enrichment', 'analyze'),
        'type-system': ('dtype', 'schema', 'polars', 'duckdb', 'type system', 'bridge'),
    }
    for concept_name, keywords in concept_keywords.items():
        concept_docs = [d for d in arch_docs if any(kw in d.content.lower() for kw in keywords)]
        if not concept_docs:
            continue
        concept_comps, concept_edges = _extract_components_from_docs(concept_docs[:5])
        if len(concept_comps) < 2:
            continue

        layers, edges = _components_to_layers(concept_comps, concept_edges)
        svg = _build_svg_diagram(layers, edges)
        (arch_dir / f'{concept_name}.svg').write_text(svg, encoding='utf-8')
        elements = _build_excalidraw_elements(layers, edges)
        excalidraw = {'type': 'excalidraw', 'version': 2, 'elements': elements,
                      'appState': {'viewBackgroundColor': '#ffffff'}}
        (arch_dir / f'{concept_name}.excalidraw').write_text(
            _json.dumps(excalidraw, indent=2), encoding='utf-8')

        title = concept_name.replace('-', ' ').title()
        page_md = [f'# {title}\n', f'![{title}](arch-diagrams/{concept_name}.svg)\n']
        page_md.append('## Components\n')
        for comp_name, comp_desc in sorted(concept_comps.items()):
            page_md.append(f'- **{comp_name}**: {comp_desc}')
        (arch_dir / f'{concept_name}.md').write_text('\n'.join(page_md), encoding='utf-8')
        diagram_pages.append((title, f'arch-diagrams/{concept_name}.md'))

    # Index page
    index_lines = ['# Architecture Diagrams\n',
                   'Visual guides to system structure and key concepts.\n']
    if not diagram_pages:
        index_lines.append('No architecture documentation available to generate diagrams from.\n')
    else:
        for title, filename in diagram_pages:
            index_lines.append(f'- [{title}]({filename})')

    index_path = gen.output_dir / 'architecture-diagrams.md'
    index_path.write_text('\n'.join(index_lines), encoding='utf-8')
    return index_path

async def generate_user_flows(gen: DocGenerator) -> Path:
    """Generate user flow diagrams showing data paths through the system."""
    import json as _json

    arch_docs = gen._list_docs('architecture', limit=30)
    expl_docs = gen._list_docs('explanation', limit=20)
    all_docs = arch_docs + expl_docs

    flow_keywords = ('pipeline', 'flow', 'step 1', 'step 2', 'sequence', 'workflow',
                     'process', '\u2192', 'then', 'next', 'finally', 'returns')
    flow_docs = [d for d in all_docs if sum(
        1 for kw in flow_keywords if kw in (d.content or '').lower()
    ) >= 2]

    flows_dir = gen.output_dir / 'user-flows'
    flows_dir.mkdir(parents=True, exist_ok=True)

    flow_pages: list[tuple[str, str, str]] = []

    if flow_docs:
        doc_summaries = []
        for doc in flow_docs[:15]:
            doc_summaries.append(f'### {doc.title}\n{doc.content[:600]}')
        context = '\n\n'.join(doc_summaries)

        flows_text = await gen._llm(
            system_prompt=(
                'From the architecture documentation below, identify 3-5 distinct user flows '
                '(end-to-end data paths through the system). For each flow, output:\n'
                'FLOW: <short title>\n'
                'STEPS: <step1> -> <step2> -> <step3> -> ...\n'
                'DESCRIPTION: <one-line description>\n\n'
                'Focus on the most important flows a developer would need to understand. '
                'Keep step names short (2-4 words). Use ONLY information from the docs.'
            ),
            user_prompt=context,
            max_tokens=1500,
        )

        parsed_flows = _parse_flow_output(flows_text)

        for i, (title, steps, description) in enumerate(parsed_flows):
            if len(steps) < 2:
                continue

            step_components = dict.fromkeys(steps, '')
            step_edges = [(steps[j], steps[j + 1]) for j in range(len(steps) - 1)]

            layers, edges = _components_to_layers(step_components, step_edges)
            svg = _build_svg_diagram(layers, edges)
            slug = re.sub(r'[^a-z0-9]+', '-', title.lower()).strip('-')[:30]

            (flows_dir / f'{slug}.svg').write_text(svg, encoding='utf-8')
            elements = _build_excalidraw_elements(layers, edges)
            excalidraw = {'type': 'excalidraw', 'version': 2, 'elements': elements,
                          'appState': {'viewBackgroundColor': '#ffffff'}}
            (flows_dir / f'{slug}.excalidraw').write_text(
                _json.dumps(excalidraw, indent=2), encoding='utf-8')

            page_md = [f'# {title}\n', f'{description}\n',
                       f'![{title}](user-flows/{slug}.svg)\n',
                       '## Steps\n']
            for j, step in enumerate(steps, 1):
                page_md.append(f'{j}. **{step}**')
            (flows_dir / f'{slug}.md').write_text('\n'.join(page_md), encoding='utf-8')
            flow_pages.append((title, f'user-flows/{slug}.md', description))

    # Index page
    index_lines = ['# User Flows\n',
                   'End-to-end data paths through the system.\n']
    if not flow_pages:
        index_lines.append('No user flows could be extracted from the documentation.\n')
    else:
        for title, filename, description in flow_pages:
            index_lines.append(f'### [{title}]({filename})\n{description}\n')

    index_path = gen.output_dir / 'user-flows.md'
    index_path.write_text('\n'.join(index_lines), encoding='utf-8')
    return index_path

def _get_import_edges(gen: DocGenerator) -> list[tuple[str, str]]:
    """Get import edges from Ariadne's graph, using short filenames."""
    try:
        with gen.lib._conn_provider.acquire() as conn:
            rows = conn.execute(
                "SELECT source_id, target_id FROM doc_graph WHERE edge_type = 'imports'"
            ).fetchall()
        return [(r[0].split('/')[-1], r[1].split('/')[-1]) for r in rows]
    except Exception:
        _logger.debug('Failed to read graph edges', exc_info=True)
        return []

async def generate_diagrams(gen: DocGenerator) -> Path:
    """Generate dependency diagrams: Excalidraw (editable), SVG (embeddable), markdown (doc page)."""
    import json as _json

    edges = _get_import_edges(gen)
    if not edges:
        md_path = gen.output_dir / 'dependency-diagram.md'
        md_path.write_text('# Dependency Diagram\n\nNo import graph data available.\n', encoding='utf-8')
        return md_path

    all_nodes: set[str] = set()
    for src, tgt in edges:
        all_nodes.add(src)
        all_nodes.add(tgt)

    in_degree: dict[str, int] = dict.fromkeys(all_nodes, 0)
    out_edges_map: dict[str, list[str]] = {n: [] for n in all_nodes}
    for src, tgt in edges:
        in_degree[tgt] = in_degree.get(tgt, 0) + 1
        out_edges_map[src].append(tgt)

    layers: list[list[str]] = []
    remaining = set(all_nodes)
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = sorted(remaining)[:5]
        layers.append(sorted(layer))
        for n in layer:
            remaining.discard(n)
            for tgt in out_edges_map.get(n, []):
                in_degree[tgt] = max(0, in_degree.get(tgt, 1) - 1)

    # 1. Excalidraw
    elements = _build_excalidraw_elements(layers, edges)
    diagram = {
        'type': 'excalidraw',
        'version': 2,
        'elements': elements,
        'appState': {'viewBackgroundColor': '#ffffff'},
    }
    excalidraw_path = gen.output_dir / 'dependency-diagram.excalidraw'
    excalidraw_path.write_text(_json.dumps(diagram, indent=2), encoding='utf-8')

    # 2. SVG
    svg_content = _build_svg_diagram(layers, edges)
    svg_path = gen.output_dir / 'dependency-diagram.svg'
    svg_path.write_text(svg_content, encoding='utf-8')

    # 3. Markdown wrapper
    md_path = gen.output_dir / 'dependency-diagram.md'
    md_path.write_text(
        f'# Dependency Diagram\n\n'
        f'{len(all_nodes)} modules, {len(edges)} import edges\n\n'
        f'![Dependency Diagram](dependency-diagram.svg)\n\n'
        f'*Editable version: [dependency-diagram.excalidraw](dependency-diagram.excalidraw)*\n',
        encoding='utf-8',
    )
    return md_path

async def generate_notebooks(gen: DocGenerator) -> Path:
    """Generate a Jupyter notebook with runnable code examples from Ariadne."""
    import json as _json

    docs_lite = gen._list_docs_lite('explanation')
    source_files: list[str] = []
    for doc in docs_lite:
        source_files.extend(doc.source_files)
    source_files = sorted(set(source_files))[:20]

    cells: list[dict] = []

    cells.append(_build_markdown_cell('# Code Examples\n\nGenerated from Ariadne documentation.'))

    for fp in source_files:
        rel_fp = _to_relative_path(fp)
        if not _is_source_module(rel_fp):
            continue
        explain_data = gen.lib.explain(fp)
        if not explain_data or not explain_data.get('documents'):
            continue

        summary = explain_data.get('summary', '')
        doc_content = ''
        for docs in explain_data.get('documents', {}).values():
            for doc in docs:
                doc_content += doc.get('content', '')[:500] + '\n'

        if not doc_content.strip():
            continue

        import_path = rel_fp.replace('/', '.').removesuffix('.py')

        code = await gen._llm(
            system_prompt=(
                "Generate a SHORT, runnable Python code example for the module described below. "
                f"The module's import path is: {import_path}\n"
                "Use this import path (not filesystem paths). Include 2-3 lines of inline comments. "
                "Output ONLY the Python code, no markdown fences, no explanation."
            ),
            user_prompt=f'Module: {import_path} ({rel_fp})\nSummary: {summary}\nDocumentation:\n{doc_content}',
            max_tokens=500,
        )

        cells.append(_build_markdown_cell(f'## `{rel_fp}`\n\n{summary}'))
        cells.append(_build_code_cell(code.strip()))

    notebook = _assemble_notebook(cells)
    path = gen.output_dir / 'examples.ipynb'
    path.write_text(_json.dumps(notebook, indent=1), encoding='utf-8')
    return path

async def generate_architecture(gen: DocGenerator) -> Path:
    """Generate architecture guide from Ariadne's knowledge."""
    arch_docs = gen._list_docs('architecture')
    graph_stats = gen.lib.get_graph_stats()

    scored_docs: list[tuple[int, Any]] = []
    for doc in arch_docs:
        score = 0
        for sf in doc.source_files:
            ir = gen.lib.impact_radius(sf)
            if ir:
                score += ir.get('radius_score', 0)
        scored_docs.append((score, doc))
    scored_docs.sort(key=lambda x: x[0], reverse=True)

    top_docs = [doc for _, doc in scored_docs[:15]]
    key_files: list[str] = []
    for doc in top_docs:
        key_files.extend(doc.source_files)

    impact_data: list[str] = []
    for fp in sorted(set(key_files))[:10]:
        ir = gen.lib.impact_radius(fp)
        if ir and ir.get('top_dependents'):
            impact_data.append(f"**{_to_relative_path(fp)}** impacts: {', '.join(ir['top_dependents'][:5])}")

    context_parts: list[str] = []
    if top_docs:
        context_parts.append('## Architecture Documents (ranked by impact)')
        for doc in top_docs:
            context_parts.append(f'### {doc.title}\n{doc.content[:800]}')
    if graph_stats:
        context_parts.append(f'## Dependency Graph Stats\n{_format_dict(graph_stats)}')
    if impact_data:
        context_parts.append('## Key File Dependencies\n' + '\n'.join(impact_data))

    context = '\n\n'.join(context_parts)

    system_prompt = (
        'You are generating an architecture guide for a software project. Structure it as:\n'
        '- System Overview (what the system does at a high level)\n'
        '- Component Architecture (key components and their responsibilities)\n'
        '- Data Flow (how data moves through the system)\n'
        '- Key Dependencies (what depends on what)\n'
        '- Design Decisions (notable patterns and why they were chosen)\n\n'
        'Use ONLY the Ariadne documentation provided. Write for developers joining the team. '
        'Use markdown with clear headings.'
    )

    arch_content = await gen._llm(
        system_prompt=system_prompt,
        user_prompt=f'Generate an architecture guide from this Ariadne documentation:\n\n{context}',
        max_tokens=4096,
    )

    path = gen.output_dir / 'architecture.md'
    path.write_text(arch_content, encoding='utf-8')
    return path

async def generate_diff(gen: DocGenerator) -> Path:
    """Generate a summary of changes between current branch and main."""
    from git_ops import get_changed_files_vs_main, get_current_branch

    branch = get_current_branch()
    changed_files = get_changed_files_vs_main(gen.output_dir.parent)
    if not changed_files:
        from config import get_config
        cfg = get_config()
        for name, src_path in cfg.get_all_source_paths().items():
            changed_files = get_changed_files_vs_main(src_path, cfg.main_branch)
            if changed_files:
                break

    if not changed_files:
        path = gen.output_dir / 'branch-diff.md'
        path.write_text(
            f"# Branch Diff Summary\n\nBranch: `{branch or 'unknown'}`\n\nNo changes detected vs main.\n",
            encoding='utf-8',
        )
        return path

    file_context: list[str] = []
    for fp in changed_files[:30]:
        explain_data = gen.lib.explain(fp)
        summary = explain_data.get('summary', '') if explain_data else ''
        ir = gen.lib.impact_radius(fp)
        impact_count = ir.get('total_affected_files', 0) if ir else 0
        test_count = ir.get('affected_tests', 0) if ir else 0
        entry = f'- **`{fp}`**'
        if summary:
            entry += f': {summary[:150]}'
        if impact_count > 0:
            entry += f' (impacts {impact_count} files, {test_count} tests)'
        file_context.append(entry)

    context = '\n'.join(file_context)

    diff_content = await gen._llm(
        system_prompt=(
            'Summarize the changes on this branch for a developer being brought up to speed. '
            'Structure as:\n'
            '- **What changed**: high-level summary of the changes\n'
            '- **Why it matters**: architectural impact based on the Ariadne context\n'
            '- **Files to review**: most important files to look at first\n'
            '- **Tests affected**: which test areas may need attention\n\n'
            'Be concise. Use ONLY the provided file context.'
        ),
        user_prompt=f'Branch: {branch}\nChanged files ({len(changed_files)} total):\n{context}',
        max_tokens=2000,
    )

    path = gen.output_dir / 'branch-diff.md'
    path.write_text(
        f"# Branch Diff Summary\n\nBranch: `{branch or 'unknown'}` | "
        f"{len(changed_files)} files changed\n\n{diff_content}",
        encoding='utf-8',
    )
    return path
