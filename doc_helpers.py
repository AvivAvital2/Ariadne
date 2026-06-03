"""Standalone helper functions for the documentation generator."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

_EXCLUDED_PATH_PREFIXES = ('test_', 'tests/', 'benchmark/', 'conftest', '.venv/', '__pycache__/')

def _to_relative_path(fp: str) -> str:
    """Strip absolute path prefixes, keeping only the relative project path."""
    for marker in ('/Ariadne/', '/myproject/', '/myproject/'):
        idx = fp.find(marker)
        if idx >= 0:
            return fp[idx + len(marker):]
    return Path(fp).name

def _is_source_module(fp: str) -> bool:
    """Return True if the file is a source module (not test/benchmark/config)."""
    if not fp.endswith('.py'):
        return False
    if fp.startswith('..'):
        return False
    basename = Path(fp).name
    return not any(fp.startswith(p) or basename.startswith(p) for p in _EXCLUDED_PATH_PREFIXES)

def _format_func_sig(node: Any) -> str:
    """Format an AST FunctionDef/AsyncFunctionDef into a readable signature."""
    import ast

    prefix = 'async ' if isinstance(node, ast.AsyncFunctionDef) else ''
    args = node.args
    params: list[str] = []

    defaults_offset = len(args.args) - len(args.defaults)
    for i, arg in enumerate(args.args):
        name = arg.arg
        if name == 'self' or name == 'cls':
            continue
        annotation = ast.unparse(arg.annotation) if arg.annotation else ''
        default_idx = i - defaults_offset
        default = f'={ast.unparse(args.defaults[default_idx])}' if default_idx >= 0 else ''
        param = f'{name}: {annotation}' if annotation else name
        params.append(f'{param}{default}')

    if args.vararg:
        params.append(f'*{args.vararg.arg}')
    if args.kwarg:
        params.append(f'**{args.kwarg.arg}')

    ret = f' -> {ast.unparse(node.returns)}' if node.returns else ''
    return f"{prefix}{node.name}({', '.join(params)}){ret}"

def _strip_markdown_fences(text: str) -> str:
    """Strip wrapping ```markdown ... ``` fences that LLMs sometimes add."""
    text = text.strip()
    if text.startswith('```markdown'):
        text = text[len('```markdown'):].strip()
    elif text.startswith('```md'):
        text = text[len('```md'):].strip()
    elif text.startswith('```'):
        text = text[3:].strip()
    if text.endswith('```'):
        text = text[:-3].strip()
    return text

def _format_dict(d: dict[str, Any]) -> str:
    """Format a dict as markdown key-value pairs."""
    return '\n'.join(f'- **{k}**: {v}' for k, v in d.items())

def _format_explain(file_path: str, data: dict[str, Any]) -> str:
    """Format explain data into a markdown section."""
    parts = [f'### `{file_path}`']
    if data.get('summary'):
        parts.append(data['summary'])
    for doc_type, docs in data.get('documents', {}).items():
        for doc in docs:
            parts.append(f"**{doc.get('title', doc_type)}**: {doc.get('content', '')[:500]}")
    return '\n\n'.join(parts)

def _parse_sections(markdown: str) -> list[tuple[str, str]]:
    """Parse markdown into (heading, full_section_text) tuples.

    Splits on ## headings. The first section (before any ##) gets heading "".
    """
    sections: list[tuple[str, str]] = []
    current_heading = ''
    current_lines: list[str] = []

    for line in markdown.split('\n'):
        if line.startswith('## '):
            if current_lines:
                sections.append((current_heading, '\n'.join(current_lines)))
            current_heading = line
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, '\n'.join(current_lines)))

    return sections

def _normalize_heading(heading: str) -> str:
    """Normalize a heading for matching: lowercase, strip markdown and whitespace."""
    return re.sub(r'[^a-z0-9]+', ' ', heading.lower()).strip()

def _find_matching_section(
    normalized_heading: str,
    sections: list[tuple[str, str]],
) -> tuple[str, str] | None:
    """Find a section whose normalized heading matches."""
    for heading, content in sections:
        if _normalize_heading(heading) == normalized_heading:
            return (heading, content)
    return None

def _build_package_api_page(pkg: str, files: list[tuple[str, str]]) -> tuple[str, int]:
    """Build a single package API reference page. Returns (content, module_count)."""
    import ast

    sections: list[str] = [f'# `{pkg}` API Reference\n']

    by_subpkg: dict[str, list[tuple[str, str]]] = {}
    for rel_fp, abs_fp in sorted(files):
        parts = Path(rel_fp).parts
        sub = str(Path(*parts[1:-1])) if len(parts) > 2 else ''
        by_subpkg.setdefault(sub, []).append((rel_fp, abs_fp))

    module_count = 0
    for sub, sub_files in sorted(by_subpkg.items()):
        if sub:
            sections.append(f'## `{sub}/`\n')

        for rel_fp, abs_fp in sub_files:
            abs_path = Path(abs_fp)
            if not abs_path.exists():
                continue
            try:
                source = abs_path.read_text(encoding='utf-8')
                tree = ast.parse(source)
            except Exception:
                continue

            module_name = Path(rel_fp).stem
            import_path = rel_fp.replace('/', '.').removesuffix('.py')
            module_doc = ast.get_docstring(tree) or ''
            parts_list: list[str] = [f'### `{module_name}`']
            if module_doc:
                parts_list.append(f'_{module_doc.split(chr(10))[0]}_')
            parts_list.append(f'Import: `{import_path}`\n')

            for node in ast.iter_child_nodes(tree):
                if isinstance(node, ast.ClassDef):
                    class_doc = ast.get_docstring(node) or ''
                    parts_list.append(f'\n**class `{node.name}`**')
                    if class_doc:
                        parts_list.append(f'> {class_doc.split(chr(10))[0]}')
                    for item in node.body:
                        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            sig = _format_func_sig(item)
                            if not item.name.startswith('_') or item.name == '__init__':
                                parts_list.append(f'- `{sig}`')
                                fn_doc = ast.get_docstring(item)
                                if fn_doc:
                                    parts_list.append(f'  {fn_doc.split(chr(10))[0]}')

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name.startswith('_'):
                        continue
                    sig = _format_func_sig(node)
                    fn_doc = ast.get_docstring(node) or ''
                    parts_list.append(f'\n`{sig}`')
                    if fn_doc:
                        parts_list.append(f'> {fn_doc.split(chr(10))[0]}')

            if len(parts_list) > 2:  # More than just header + import
                sections.append('\n'.join(parts_list))
                module_count += 1

    return '\n\n---\n\n'.join(sections), module_count

def _build_api_nav(api_dir: Path) -> list[dict[str, str]]:
    """Build MkDocs nav entries for per-package API pages."""
    if not api_dir.exists():
        return []
    entries: list[dict[str, str]] = []
    for md_file in sorted(api_dir.glob('*.md')):
        pkg_name = md_file.stem
        entries.append({pkg_name: f'api/{md_file.name}'})
    return entries

def _build_module_tree(pkg: str, files: list[tuple[str, str]]) -> str:
    """Build an indented module tree for the index page."""
    lines: list[str] = []
    by_subpkg: dict[str, list[str]] = {}
    for rel_fp, _ in sorted(files):
        parts = Path(rel_fp).parts
        sub = str(Path(*parts[1:-1])) if len(parts) > 2 else ''
        module_name = Path(rel_fp).stem
        by_subpkg.setdefault(sub, []).append(module_name)

    for sub, modules in sorted(by_subpkg.items()):
        if sub:
            lines.append(f'- **{sub}/**')
            for m in modules:
                lines.append(f'    - `{m}`')
        else:
            for m in modules:
                lines.append(f'- `{m}`')

    return '\n'.join(lines)

def _parse_flow_output(text: str) -> list[tuple[str, list[str], str]]:
    """Parse LLM flow output into (title, steps, description) tuples."""
    flows: list[tuple[str, list[str], str]] = []
    current_title = ''
    current_steps: list[str] = []
    current_desc = ''

    for line in text.split('\n'):
        stripped = line.strip()
        upper = stripped.upper()
        if upper.startswith('FLOW:'):
            if current_title and current_steps:
                flows.append((current_title, current_steps, current_desc))
            current_title = stripped[5:].strip().strip('*_')
            current_steps = []
            current_desc = ''
        elif upper.startswith('STEPS:'):
            raw = stripped[6:].strip()
            current_steps = [s.strip().strip('*_`') for s in re.split(r'\s*->\s*', raw) if s.strip()]
        elif upper.startswith('DESCRIPTION:'):
            current_desc = stripped[12:].strip()

    if current_title and current_steps:
        flows.append((current_title, current_steps, current_desc))

    return flows

def _extract_components_from_docs(docs: list[Any]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Extract component names and relationships from architecture docs."""
    components: dict[str, str] = {}
    interactions: list[tuple[str, str]] = []

    for doc in docs:
        title = doc.title or ''
        content = doc.content or ''

        for sf in getattr(doc, 'source_files', []):
            rel = _to_relative_path(sf)
            parts = Path(rel).parts
            if len(parts) >= 2:
                pkg = f'{parts[0]}.{parts[1]}'
            elif parts:
                pkg = parts[0].removesuffix('.py')
            else:
                continue

            if pkg not in components:
                first_line = content.split('\n')[0].strip('# ').strip()
                if not first_line or first_line.startswith('```'):
                    first_line = title
                components[pkg] = first_line[:120]

        for line in content.split('\n'):
            line_lower = line.lower()
            if 'depends on' in line_lower or 'imports' in line_lower or '\u2192' in line or '-->' in line:
                for comp_a in components:
                    short_a = comp_a.split('.')[-1]
                    if short_a.lower() in line_lower:
                        for comp_b in components:
                            if comp_a == comp_b:
                                continue
                            short_b = comp_b.split('.')[-1]
                            if short_b.lower() in line_lower:
                                interactions.append((comp_a, comp_b))

    return components, interactions

def _extract_flow_from_docs(docs: list[Any]) -> tuple[dict[str, str], list[tuple[str, str]]]:
    """Extract sequential flow steps from pipeline-like docs."""
    steps: dict[str, str] = {}
    edges: list[tuple[str, str]] = []
    prev_step: str | None = None

    for doc in docs:
        content = doc.content or ''
        for line in content.split('\n'):
            stripped = line.strip()
            if not stripped:
                continue
            step_match = re.match(r'(?:\d+[\.\)]\s*|step\s+\d+[:\s]+)(.+)', stripped, re.IGNORECASE)
            if step_match:
                step_name = step_match.group(1).strip('*_` ').split('(')[0].strip()[:40]
                if step_name and step_name not in steps:
                    desc = stripped[:100]
                    steps[step_name] = desc
                    if prev_step:
                        edges.append((prev_step, step_name))
                    prev_step = step_name

    return steps, edges

def _components_to_layers(components: dict[str, str], interactions: list[tuple[str, str]]) -> tuple[list[list[str]], list[tuple[str, str]]]:
    """Convert components + interactions into layers for diagram rendering."""
    all_nodes = set(components.keys())
    if not all_nodes:
        return [], []

    in_degree: dict[str, int] = dict.fromkeys(all_nodes, 0)
    out_edges: dict[str, list[str]] = {n: [] for n in all_nodes}
    valid_edges: list[tuple[str, str]] = []

    for src, tgt in interactions:
        if src in all_nodes and tgt in all_nodes and src != tgt:
            in_degree[tgt] = in_degree.get(tgt, 0) + 1
            out_edges[src].append(tgt)
            valid_edges.append((src, tgt))

    layers: list[list[str]] = []
    remaining = set(all_nodes)
    while remaining:
        layer = [n for n in remaining if in_degree.get(n, 0) == 0]
        if not layer:
            layer = sorted(remaining)[:5]
        layers.append(sorted(layer))
        for n in layer:
            remaining.discard(n)
            for tgt in out_edges.get(n, []):
                in_degree[tgt] = max(0, in_degree.get(tgt, 1) - 1)

    return layers, valid_edges

_LAYER_COLORS = ('#a5d8ff', '#b2f2bb', '#ffd8a8', '#d0bfff', '#c3fae8', '#fff3bf', '#ffc9c9', '#eebefa')

_NODE_W = 160
_NODE_H = 50
_X_GAP = 40
_Y_GAP = 80

def _build_excalidraw_elements(
    layers: list[list[str]],
    edges: list[tuple[str, str]],
) -> list[dict[str, Any]]:
    """Build Excalidraw JSON elements from layered graph nodes and edges."""
    elements: list[dict[str, Any]] = []
    node_positions: dict[str, tuple[int, int]] = {}

    y = 60
    max_x = 0
    for layer_idx, layer in enumerate(layers):
        x = 40
        color = _LAYER_COLORS[layer_idx % len(_LAYER_COLORS)]
        for node_name in layer:
            node_id = f"n_{node_name.replace('.', '_')}"
            elements.append({
                'type': 'rectangle',
                'id': node_id,
                'x': x,
                'y': y,
                'width': _NODE_W,
                'height': _NODE_H,
                'backgroundColor': color,
                'fillStyle': 'solid',
                'roundness': {'type': 3},
                'label': {'text': node_name, 'fontSize': 14},
            })
            node_positions[node_name] = (x, y)
            x += _NODE_W + _X_GAP
            max_x = max(max_x, x)
        y += _NODE_H + _Y_GAP

    for i, (src, tgt) in enumerate(edges):
        if src not in node_positions or tgt not in node_positions:
            continue
        sx, sy = node_positions[src]
        tx, ty = node_positions[tgt]
        start_x = sx + _NODE_W // 2
        start_y = sy + _NODE_H
        end_x = tx + _NODE_W // 2
        end_y = ty
        dx = end_x - start_x
        dy = end_y - start_y

        src_id = f"n_{src.replace('.', '_')}"
        tgt_id = f"n_{tgt.replace('.', '_')}"

        elements.append({
            'type': 'arrow',
            'id': f'e_{i}',
            'x': start_x,
            'y': start_y,
            'width': abs(dx),
            'height': abs(dy),
            'points': [[0, 0], [dx, dy]],
            'endArrowhead': 'arrow',
            'strokeColor': '#757575',
            'strokeWidth': 1,
            'startBinding': {'elementId': src_id, 'fixedPoint': [0.5, 1]},
            'endBinding': {'elementId': tgt_id, 'fixedPoint': [0.5, 0]},
        })

    return elements

_SVG_STROKE_COLORS = ('#4a9eed', '#22c55e', '#f59e0b', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16', '#ef4444')

def _build_svg_diagram(
    layers: list[list[str]],
    edges: list[tuple[str, str]],
) -> str:
    """Build an SVG dependency diagram from layered graph nodes and edges."""
    node_positions: dict[str, tuple[int, int]] = {}

    y = 40
    max_x = 0
    for layer_idx, layer in enumerate(layers):
        x = 20
        for node_name in layer:
            node_positions[node_name] = (x, y)
            x += _NODE_W + _X_GAP
            max_x = max(max_x, x)
        y += _NODE_H + _Y_GAP

    total_w = max(max_x + 20, 400)
    total_h = y + 20

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {total_w} {total_h}" '
        f'width="{total_w}" height="{total_h}" style="font-family: -apple-system, sans-serif; font-size: 12px;">',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="10" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#888"/></marker></defs>',
    ]

    for src, tgt in edges:
        if src not in node_positions or tgt not in node_positions:
            continue
        sx, sy = node_positions[src]
        tx, ty = node_positions[tgt]
        x1 = sx + _NODE_W // 2
        y1 = sy + _NODE_H
        x2 = tx + _NODE_W // 2
        y2 = ty
        parts.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#ccc" stroke-width="1" marker-end="url(#arrow)"/>'
        )

    for layer_idx, layer in enumerate(layers):
        fill = _LAYER_COLORS[layer_idx % len(_LAYER_COLORS)]
        stroke = _SVG_STROKE_COLORS[layer_idx % len(_SVG_STROKE_COLORS)]
        for node_name in layer:
            x, y_pos = node_positions[node_name]
            parts.append(
                f'<rect x="{x}" y="{y_pos}" width="{_NODE_W}" height="{_NODE_H}" '
                f'rx="6" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
            )
            label = node_name if len(node_name) <= 20 else node_name[:18] + '..'
            parts.append(
                f'<text x="{x + _NODE_W // 2}" y="{y_pos + _NODE_H // 2 + 4}" '
                f'text-anchor="middle" fill="#333">{label}</text>'
            )

    parts.append('</svg>')
    return '\n'.join(parts)

def _build_markdown_cell(source: str) -> dict[str, Any]:
    """Build a Jupyter markdown cell dict."""
    return {
        'cell_type': 'markdown',
        'metadata': {},
        'source': source.splitlines(keepends=True),
    }

def _build_code_cell(source: str) -> dict[str, Any]:
    """Build a Jupyter code cell dict."""
    return {
        'cell_type': 'code',
        'metadata': {},
        'source': source.splitlines(keepends=True),
        'outputs': [],
        'execution_count': None,
    }

def _assemble_notebook(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Assemble cells into a valid nbformat v4 notebook."""
    return {
        'nbformat': 4,
        'nbformat_minor': 5,
        'metadata': {
            'kernelspec': {
                'display_name': 'Python 3',
                'language': 'python',
                'name': 'python3',
            },
            'language_info': {
                'name': 'python',
                'version': '3.12.0',
            },
        },
        'cells': cells,
    }
