"""On-demand documentation generator from Ariadne's knowledge base.

Assembles curated knowledge (docs, sections, gotchas, architecture) into
user-facing documentation files. Uses Ariadne's LLM provider for editorial
polish — Ariadne supplies facts, LLM synthesizes prose.

Usage:
    from doc_generator import DocGenerator
    gen = DocGenerator(library, output_dir)
    results = await gen.generate(["readme", "api", "architecture"])
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from doc_diagrams import (
    generate_arch_diagrams,
    generate_architecture,
    generate_diagrams,
    generate_diff,
    generate_notebooks,
    generate_user_flows,
)
from doc_helpers import (
    _assemble_notebook as _assemble_notebook,
)
from doc_helpers import (
    _build_api_nav,
    _build_module_tree,
    _build_package_api_page,
    _find_matching_section,
    _format_dict,
    _is_source_module,
    _normalize_heading,
    _parse_sections,
    _strip_markdown_fences,
    _to_relative_path,
)
from doc_helpers import (
    _build_code_cell as _build_code_cell,
)
from doc_helpers import (
    _build_excalidraw_elements as _build_excalidraw_elements,
)
from doc_helpers import (
    _build_markdown_cell as _build_markdown_cell,
)
from doc_helpers import (
    # Re-exports for backward compatibility (tests import from doc_generator)
    _build_svg_diagram as _build_svg_diagram,
)
from doc_helpers import (
    _components_to_layers as _components_to_layers,
)
from doc_helpers import (
    _extract_components_from_docs as _extract_components_from_docs,
)
from doc_helpers import (
    _parse_flow_output as _parse_flow_output,
)

_logger = logging.getLogger(__name__)

# Doc types and their handlers
DOC_TYPES = ('readme', 'api', 'architecture', 'arch_diagrams', 'user_flows', 'diagrams', 'notebooks', 'faq', 'decisions', 'diff', 'patterns')

DOC_TYPE_TO_FILE = {
    'readme': 'README.md',
    'api': 'api-reference.md',
    'architecture': 'architecture.md',
    'faq': 'faq.md',
    'decisions': 'design-decisions.md',
    'diff': 'branch-diff.md',
    'diagrams': 'dependency-diagram.md',
    'notebooks': 'examples.ipynb',
    'patterns': 'patterns-gotchas.md',
    'arch_diagrams': 'architecture-diagrams.md',
    'user_flows': 'user-flows.md',
}

class DocGenerator:
    """Generate user-facing documentation from Ariadne's knowledge base."""

    def __init__(self, library: Any, output_dir: Path, source: str | None = None) -> None:
        self.lib = library
        self.output_dir = output_dir
        self.source = source  # Filter docs to this source name (e.g. "pythonproject")

    def _list_docs(self, content_type: str, limit: int | None = None) -> list:
        """List documents filtered by source if set."""
        docs = self.lib.list_documents(content_type=content_type, limit=limit * 3 if limit else None)
        if self.source:
            docs = [d for d in docs if (d.source_name or '') == self.source]
        elif self.source is not None:
            docs = [d for d in docs if not d.source_name]
        if limit:
            docs = docs[:limit]
        return docs

    @staticmethod
    async def _llm(system_prompt: str, user_prompt: str, max_tokens: int = 3000) -> str:
        """Call LLM and strip markdown fences from output."""
        from llm import chat_complete
        result = await chat_complete(system_prompt=system_prompt, user_prompt=user_prompt, max_tokens=max_tokens)
        return _strip_markdown_fences(result)

    def _list_docs_lite(self, content_type: str) -> list:
        """List doc metadata filtered by source if set."""
        docs = self.lib.list_documents_lite(content_type=content_type)
        if self.source:
            docs = [d for d in docs if (d.source_name or '') == self.source]
        elif self.source is not None:
            docs = [d for d in docs if not d.source_name]
        return docs

    async def generate(self, doc_types: list[str] | None = None) -> dict[str, Path]:
        """Generate docs for requested types. Returns {type: output_path}."""
        if doc_types is None:
            doc_types = list(DOC_TYPES)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results: dict[str, Path] = {}

        for dtype in doc_types:
            handler = _HANDLERS.get(dtype)
            if handler is None:
                _logger.warning('Unknown doc type: %s (available: %s)', dtype, ', '.join(DOC_TYPES))
                continue
            try:
                path = await handler(self)
                results[dtype] = path
                _logger.info('Generated %s → %s', dtype, path)
            except Exception:
                _logger.exception('Failed to generate %s', dtype)

        # Generate mkdocs.yml for the doc site
        if results:
            self._generate_mkdocs_config(results)

        return results

    def _generate_mkdocs_config(self, generated: dict[str, Path]) -> None:
        """Create mkdocs.yml for MkDocs Material site."""
        import yaml

        nav_map = {
            'readme': ('Home', 'README.md'),
            'api': ('API Reference', 'api-reference.md'),
            'architecture': ('Architecture', 'architecture.md'),
            'faq': ('FAQ', 'faq.md'),
            'decisions': ('Design Decisions', 'design-decisions.md'),
            'diagrams': ('Dependency Diagram', 'dependency-diagram.md'),
            'diff': ('Branch Changes', 'branch-diff.md'),
            'patterns': ('Patterns & Gotchas', 'patterns-gotchas.md'),
            'arch_diagrams': ('Architecture Diagrams', 'architecture-diagrams.md'),
            'user_flows': ('User Flows', 'user-flows.md'),
        }

        nav: list[dict[str, Any]] = []
        for dtype in DOC_TYPES:
            if dtype in generated and dtype in nav_map:
                label, filename = nav_map[dtype]
                if dtype == 'api':
                    api_nav = _build_api_nav(self.output_dir / 'api')
                    if api_nav:
                        nav.append({'API Reference': [{'Overview': filename}] + api_nav})
                        continue
                nav.append({label: filename})

        site_name = 'Documentation'
        if self.source:
            site_name = f'{self.source} Documentation'

        config = {
            'site_name': site_name,
            'docs_dir': '.',
            'site_dir': '../_site',
            'theme': {
                'name': 'material',
                'features': ['navigation.sections', 'search.suggest', 'search.highlight'],
                'palette': {'scheme': 'default'},
            },
            'nav': nav,
        }

        config_path = self.output_dir / 'mkdocs.yml'
        config_path.write_text(yaml.dump(config, default_flow_style=False, sort_keys=False), encoding='utf-8')

    async def _generate_readme(self) -> Path:
        """Generate a README.md from Ariadne's knowledge."""
        self.output_dir.mkdir(parents=True, exist_ok=True)

        arch_docs = self._list_docs('architecture')
        expl_docs = self._list_docs('explanation', limit=20)
        finding_docs = self._list_docs('finding', limit=10)
        gotcha_docs = self._list_docs('gotcha', limit=10)
        graph_stats = self.lib.get_graph_stats()

        context_parts: list[str] = []

        if arch_docs:
            context_parts.append('## Architecture Documents')
            for doc in arch_docs[:5]:
                context_parts.append(f'### {doc.title}\n{doc.content[:1000]}')

        if expl_docs:
            context_parts.append('## Key Module Explanations')
            for doc in expl_docs[:10]:
                source = ', '.join(doc.source_files) if doc.source_files else 'N/A'
                context_parts.append(f'- **{doc.title}** ({source}): {doc.content[:300]}')

        if finding_docs:
            context_parts.append('## Notable Findings')
            for doc in finding_docs[:5]:
                context_parts.append(f'- {doc.title}: {doc.content[:200]}')

        if gotcha_docs:
            context_parts.append('## Known Gotchas')
            for doc in gotcha_docs[:5]:
                context_parts.append(f'- {doc.title}: {doc.content[:200]}')

        if graph_stats:
            context_parts.append(f'## Project Stats\n{_format_dict(graph_stats)}')

        context = '\n\n'.join(context_parts)

        system_prompt = (
            'You are a technical writer generating a README.md for a software project. '
            'Use ONLY the Ariadne documentation context provided — do not invent information. '
            'Structure the README with these sections:\n'
            '- Project title and one-line description\n'
            '- Overview (what the project does, key concepts)\n'
            '- Architecture (high-level component overview)\n'
            '- Installation / Getting Started\n'
            '- Key Modules (brief description of main components)\n\n'
            'Do NOT include patterns or gotchas — those live on a separate page. '
            'Keep it concise and practical. Use markdown formatting.'
        )

        readme_content = await self._llm(
            system_prompt=system_prompt,
            user_prompt=f'Generate a README.md from this Ariadne documentation:\n\n{context}',
            max_tokens=4096,
        )

        path = self.output_dir / 'README.md'
        path.write_text(readme_content, encoding='utf-8')
        return path

    async def _generate_api_reference(self) -> Path:
        """Generate API reference: index page + per-package pages via AST extraction."""
        docs_lite = self._list_docs_lite('explanation')
        rel_to_abs: dict[str, str] = {}
        for doc in docs_lite:
            for sf in doc.source_files:
                rel = _to_relative_path(sf)
                if _is_source_module(rel):
                    rel_to_abs[rel] = sf

        by_package: dict[str, list[tuple[str, str]]] = {}
        for rel_fp, abs_fp in sorted(rel_to_abs.items()):
            parts = Path(rel_fp).parts
            pkg = parts[0] if len(parts) > 1 else '_root'
            by_package.setdefault(pkg, []).append((rel_fp, abs_fp))

        api_dir = self.output_dir / 'api'
        api_dir.mkdir(parents=True, exist_ok=True)

        package_summaries: list[tuple[str, str, int]] = []
        for pkg, files in sorted(by_package.items()):
            page_content, module_count = _build_package_api_page(pkg, files)
            if module_count == 0:
                continue
            filename = f'{pkg}.md'
            (api_dir / filename).write_text(page_content, encoding='utf-8')
            package_summaries.append((pkg, filename, module_count))

        index_lines: list[str] = ['# API Reference\n']
        for pkg, filename, count in package_summaries:
            index_lines.append(f'## [`{pkg}`](api/{filename})')
            pkg_files = by_package[pkg]
            sub_tree = _build_module_tree(pkg, pkg_files)
            index_lines.append(sub_tree)

        path = self.output_dir / 'api-reference.md'
        path.write_text('\n\n'.join(index_lines), encoding='utf-8')
        return path

    async def _generate_faq(self) -> Path:
        """Generate FAQ from gotchas, findings, and error patterns."""
        gotcha_docs = self._list_docs('gotcha', limit=20)
        finding_docs = self._list_docs('finding', limit=20)

        context_parts: list[str] = []
        if gotcha_docs:
            context_parts.append('## Gotchas (common pitfalls)')
            for doc in gotcha_docs:
                context_parts.append(f'- **{doc.title}**: {doc.content[:300]}')
        if finding_docs:
            context_parts.append('## Findings (discovered patterns)')
            for doc in finding_docs:
                context_parts.append(f'- **{doc.title}**: {doc.content[:300]}')

        if not context_parts:
            path = self.output_dir / 'faq.md'
            path.write_text('# FAQ\n\nNo gotchas or findings documented yet.\n', encoding='utf-8')
            return path

        faq_content = await self._llm(
            system_prompt=(
                'Generate a FAQ document from the gotchas and findings below. '
                'Format as:\n'
                '## Q: <question>\n\n<answer>\n\n'
                'Group related questions together. Use ONLY the provided content.'
            ),
            user_prompt='\n\n'.join(context_parts),
            max_tokens=3000,
        )

        path = self.output_dir / 'faq.md'
        path.write_text(f'# Frequently Asked Questions\n\n{faq_content}', encoding='utf-8')
        return path

    async def _generate_decisions(self) -> Path:
        """Generate design considerations — why X not Y."""
        arch_docs = self._list_docs('architecture')
        finding_docs = self._list_docs('finding', limit=15)

        decision_keywords = ('decision', 'chose', 'tradeoff', 'alternative', 'why', 'instead', 'rationale', 'considered')
        context_parts: list[str] = []

        matched = 0
        for doc in arch_docs:
            if matched >= 15:
                break
            lower = doc.content.lower()
            if any(kw in lower for kw in decision_keywords):
                context_parts.append(f'### {doc.title}\n{doc.content[:600]}')
                matched += 1

        for doc in finding_docs:
            lower = doc.content.lower()
            if any(kw in lower for kw in decision_keywords):
                context_parts.append(f'### {doc.title} (finding)\n{doc.content[:400]}')

        if not context_parts:
            path = self.output_dir / 'design-decisions.md'
            path.write_text('# Design Decisions\n\nNo design rationale documented yet.\n', encoding='utf-8')
            return path

        decisions_content = await self._llm(
            system_prompt=(
                "Generate a 'Design Decisions' document explaining the key architectural "
                "choices made in this project. For each decision:\n"
                "- What was decided\n"
                "- Why this approach was chosen\n"
                "- What alternatives were considered (ONLY if mentioned in the source; "
                "do NOT write 'Not specified' — simply omit the alternatives section)\n"
                "- Any tradeoffs or limitations (ONLY if mentioned; omit if not)\n\n"
                "Use ONLY the provided content. Write for developers joining the team. "
                "If information is not available for a section, skip it entirely rather "
                "than using placeholder text."
            ),
            user_prompt='\n\n'.join(context_parts),
            max_tokens=3000,
        )

        path = self.output_dir / 'design-decisions.md'
        path.write_text(f'# Design Decisions\n\n{decisions_content}', encoding='utf-8')
        return path

    async def _generate_patterns(self) -> Path:
        """Generate a standalone patterns and gotchas page."""
        gotcha_docs = self._list_docs('gotcha', limit=30)
        finding_docs = self._list_docs('finding', limit=20)
        arch_docs = self._list_docs('architecture', limit=10)

        context_parts: list[str] = []

        if gotcha_docs:
            context_parts.append('## Gotchas (common pitfalls)')
            for doc in gotcha_docs:
                context_parts.append(f'- **{doc.title}**: {doc.content[:500]}')

        pattern_keywords = ('pattern', 'convention', 'practice', 'rule', 'always', 'never', 'must')
        for doc in arch_docs:
            lower = doc.content.lower()
            if any(kw in lower for kw in pattern_keywords):
                context_parts.append(f'### {doc.title}\n{doc.content[:600]}')

        if finding_docs:
            context_parts.append('## Discovered Patterns')
            for doc in finding_docs[:10]:
                context_parts.append(f'- **{doc.title}**: {doc.content[:300]}')

        if not context_parts:
            path = self.output_dir / 'patterns-gotchas.md'
            path.write_text('# Notable Patterns & Gotchas\n\nNo patterns or gotchas documented yet.\n', encoding='utf-8')
            return path

        content = await self._llm(
            system_prompt=(
                "Generate a 'Notable Patterns & Gotchas' reference page. Structure as:\n"
                "## Patterns\nGroup related coding patterns, conventions, and best practices. "
                "For each pattern, explain WHAT it is and WHY it matters.\n\n"
                "## Gotchas\nList common pitfalls and surprises. For each, explain the problem "
                "and how to avoid it.\n\n"
                "Use ONLY the provided content. Write for developers joining the team."
            ),
            user_prompt='\n\n'.join(context_parts),
            max_tokens=3000,
        )

        path = self.output_dir / 'patterns-gotchas.md'
        path.write_text(f'# Notable Patterns & Gotchas\n\n{content}', encoding='utf-8')
        return path

    async def update_readme_in_place(self, readme_path: Path) -> Path:
        """Update an existing README.md by replacing matched sections.

        Parses the existing README by ## headings, generates fresh content,
        and replaces sections that have a matching heading. Unmatched manual
        sections are preserved.
        """
        if not readme_path.exists():
            return await self._generate_readme()

        existing = readme_path.read_text(encoding='utf-8')
        existing_sections = _parse_sections(existing)

        generated_path = await self._generate_readme()
        generated = generated_path.read_text(encoding='utf-8')
        generated_sections = _parse_sections(generated)

        merged_parts: list[str] = []
        replaced_headings: set[str] = set()

        for heading, content in existing_sections:
            normalized = _normalize_heading(heading)
            match = _find_matching_section(normalized, generated_sections)
            if match is not None:
                merged_parts.append(match[1])
                replaced_headings.add(normalized)
            else:
                merged_parts.append(content)

        for heading, content in generated_sections:
            if _normalize_heading(heading) not in replaced_headings:
                merged_parts.append(content)

        readme_path.write_text('\n\n'.join(merged_parts), encoding='utf-8')
        return readme_path

# Handler registry — maps doc type names to async callables (DocGenerator -> Path)
_HANDLERS: dict[str, Any] = {
    'readme': DocGenerator._generate_readme,
    'api': DocGenerator._generate_api_reference,
    'architecture': generate_architecture,
    'diagrams': generate_diagrams,
    'notebooks': generate_notebooks,
    'faq': DocGenerator._generate_faq,
    'decisions': DocGenerator._generate_decisions,
    'diff': generate_diff,
    'patterns': DocGenerator._generate_patterns,
    'arch_diagrams': generate_arch_diagrams,
    'user_flows': generate_user_flows,
}
