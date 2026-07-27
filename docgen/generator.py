"""LLM-based documentation generator.

This module provides the DocGenerator class that uses LLMs to generate
documentation from source code analysis.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from attrs import define, field

from docgen._legacy_analyzer import SourceAnalyzer
from docgen._legacy_metadata import ModuleGroup, ModuleMetadata
from docgen.catalog_enrich import EnrichedFileBundle
from docgen.llm.anthropic import QuotaExhaustedError
from docgen.prompts import (
    DocType,
    PromptTemplate,
    filter_doc_types_for_language,
    format_classes_functions,
    format_dependencies,
    format_module_info,
    format_related_modules,
    get_template,
    render_user_template,
)

_logger = logging.getLogger(__name__)

# Maximum source code length before chunking
MAX_SOURCE_LENGTH = 500000
# Maximum tokens the LLM may emit per call. Set to 8192 (was 4096) because
# 4096 truncated long architecture/qa docs on big files (e.g., 10K-line
# generated source), producing UNCLOSED_CODE_BLOCK validation failures.
# Note: this is a CEILING, not a target — Anthropic bills only actual tokens
# generated, so raising it costs nothing on docs that fit under 4096. Independent
# of the cache token minimum (which is about INPUT, not output).
MAX_OUTPUT_TOKENS = 8192


# Subtype taxonomy spanning Python + SCIP-extracted Scala/Java. The
# generator's bundle filters use these to keep prompt rendering
# language-agnostic.
_CLASS_LIKE_SUBTYPES = frozenset({
    # Python
    'class',
    # Scala
    'scala_class', 'scala_object', 'scala_trait',
    # Java
    'java_class', 'java_interface', 'java_enum',
    # Go — struct (aggregate) + interface (contract) are the class-like types.
    'go_struct', 'go_interface',
})

_CALLABLE_SUBTYPES = frozenset({
    # Python
    'function', 'async_function', 'method',
    # Scala — the SCIP "scala_def" applies to both top-level defs and
    # methods inside a class; "method-vs-function" is determined by
    # parent_qualified_name, not subtype.
    'scala_def', 'scala_implicit',
    # Java
    'java_method', 'java_constructor',
    # Go — top-level funcs and receiver methods (method-vs-function is on
    # parent_qualified_name, same as Scala).
    'go_function', 'go_method',
})


@define
class GeneratorConfig:
    """Configuration for the documentation generator.

    Attributes:
        model: The LLM model to use (e.g., 'gpt-5.2', 'gpt-5.2-mini', 'claude-3-5-sonnet').
        api_key: API key for the LLM provider.
        base_url: Base URL for the API endpoint.
        max_retries: Maximum number of retry attempts for failed requests.
        retry_delay: Base delay between retries in seconds.
        timeout: Request timeout in seconds.
        doc_types: Types of documentation to generate.
    """

    model: str = 'gpt-5.2'
    api_key: str | None = None
    base_url: str = 'https://api.openai.com/v1'
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 120.0
    doc_types: tuple[DocType, ...] = ('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram')
    # LLM backend selector — "openai" or "anthropic". When "anthropic",
    # the generator routes through the native /v1/messages API with
    # x-api-key auth; api_key/base_url should be Anthropic's. Default
    # "openai" preserves historical behavior.
    provider: str = 'openai'
    withhold_source_prose: bool = False


def _dangling_autodoc_meta(bundle):
    """{'stale_autodoc': True} when the bundle's rst autodoc references a symbol
    that did not resolve (a dangling target), else {}. Spread into a generated
    doc's metadata so search down-ranks it.
    """
    if bundle.scip and any(not link.resolved for link in bundle.scip.autodoc_links):
        return {'stale_autodoc': True}
    return {}


@define
class GeneratedDoc:
    """A generated documentation item."""

    title: str
    content: str
    doc_type: DocType
    source_files: tuple[str, ...]
    metadata: dict


@define
class PromptBundle:
    """Pre-built prompt for one ``(file, doc_type)`` pair, ready to be
    sent through either streaming ``_call_llm`` or
    ``provider.submit_batch``.

    Built up-front by ``build_prompts_for_module`` /
    ``build_prompts_for_bundle`` so the orchestrator can collect
    prompts across all files before deciding to dispatch sync vs
    batch. The pre-computed ``title`` and ``metadata`` let
    ``assemble_doc`` wrap a batch response into a ``GeneratedDoc``
    without re-deriving them from the source path or analyzer —
    important because the analyzer state may not be available by
    the time a batch result arrives (potentially hours later, or in
    a different process during resume).
    """

    file: Path
    doc_type: DocType
    system_prompt: str
    user_prompt: str
    title: str
    metadata: dict


@define
class DocGenerator:
    """Generates documentation from source code using LLMs.

    This class analyzes Python source code and generates various types of
    documentation including explanations, architecture docs, Q&A pairs,
    and diagrams.

    Attributes:
        config: Generator configuration.
        analyzer: Source code analyzer.
        _client: HTTP client for API requests.
    """

    config: GeneratorConfig = field(factory=GeneratorConfig)
    analyzer: SourceAnalyzer = field(factory=SourceAnalyzer)
    intent_filler: object | None = None
    _provider: object | None = field(default=None, init=False)

    async def __aenter__(self) -> DocGenerator:
        """Construct the LLM provider via the factory.

        The provider encapsulates the per-backend HTTP shape (OpenAI's
        ``/chat/completions`` vs Anthropic's ``/v1/messages``); the
        generator interacts with it through ``LLMProvider.call``.
        """
        from docgen.llm.factory import make_llm_provider

        self._provider = make_llm_provider(
            provider=self.config.provider,
            model=self.config.model,
            api_key=self.config.api_key or '',
            base_url=self.config.base_url,
            max_retries=self.config.max_retries,
            timeout=self.config.timeout,
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release the LLM provider's HTTP client."""
        if self._provider is not None:
            await self._provider.aclose()
            self._provider = None

    async def generate_for_file(
        self,
        path: Path,
        doc_types: tuple[DocType, ...] | None = None,
    ) -> list[GeneratedDoc]:
        """Generate documentation for a single Python file.

        Args:
            path: Path to the Python file.
            doc_types: Types of documentation to generate. Defaults to config.doc_types.

        Returns:
            List of generated documentation items.
        """
        metadata = self.analyzer.analyze_file(path)
        return await self.generate_for_module(metadata, doc_types)

    async def generate_for_module(
        self,
        metadata: ModuleMetadata,
        doc_types: tuple[DocType, ...] | None = None,
        extra_prompt_context: str = '',
    ) -> list[GeneratedDoc]:
        """Generate documentation for an analyzed module.

        Args:
            metadata: Module metadata from analyzer.
            doc_types: Types of documentation to generate.
            extra_prompt_context: Optional markdown block appended to
                the LLM prompt. Used by the reverse-augment phase
                (Phase 3) to inject consumer-context information so
                the regenerated docs describe how the module is
                consumed by other indexed sources. Empty default
                preserves existing callers' behavior.

        Returns:
            List of generated documentation items.
        """
        doc_types = doc_types or self.config.doc_types
        docs: list[GeneratedDoc] = []

        _logger.info('Generating docs for %s (types: %s)', metadata.path, ', '.join(doc_types))
        source_code = metadata.path.read_text(encoding='utf-8')

        for doc_type in doc_types:
            try:
                _logger.info('  [%s] %s — starting', doc_type, metadata.module_name)
                content = await self._generate_doc(
                    metadata, source_code, doc_type,
                    extra_prompt_context=extra_prompt_context,
                )
                if content:
                    title = self._generate_title(metadata, doc_type)
                    docs.append(
                        GeneratedDoc(
                            title=title,
                            content=content,
                            doc_type=doc_type,
                            source_files=(str(metadata.path),),
                            metadata={
                                'module_name': metadata.module_name,
                                'source_hash': metadata.source_hash,
                            },
                        )
                    )
            except QuotaExhaustedError:
                # Re-raise so the orchestrator's abort-coordination
                # (run():427) sees it. Without this, a per-doc-type
                # blanket Exception catch would swallow quota errors
                # and the orchestrator would never set abort_event,
                # causing users to burn compute on doomed retries
                # instead of gracefully aborting.
                raise
            except Exception as e:
                _logger.error('Failed to generate %s for %s: %s', doc_type, metadata.path, e)

        return docs

    async def generate_for_group(
        self,
        group: ModuleGroup,
        doc_types: tuple[DocType, ...] | None = None,
        concurrency: int = 3,
    ) -> list[GeneratedDoc]:
        """Generate documentation for a module group (package).

        Args:
            group: Module group from analyzer.
            doc_types: Types of documentation to generate.
            concurrency: Maximum concurrent LLM requests.

        Returns:
            List of generated documentation items.
        """
        doc_types = doc_types or self.config.doc_types
        docs: list[GeneratedDoc] = []

        _logger.info('Generating group docs: %d modules, concurrency=%d', len(group.all_modules), concurrency)
        # Generate docs for each module in the group
        semaphore = asyncio.Semaphore(concurrency)

        async def generate_with_limit(metadata: ModuleMetadata) -> list[GeneratedDoc]:
            async with semaphore:
                return await self.generate_for_module(metadata, doc_types)

        tasks = [generate_with_limit(mod) for mod in group.all_modules]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                _logger.error('Generation failed: %s', result)
            else:
                docs.extend(result)

        # Generate group-level architecture doc if requested
        if 'architecture' in doc_types and len(group.all_modules) > 1:
            try:
                group_doc = await self._generate_group_architecture(group)
                if group_doc:
                    docs.append(group_doc)
            except Exception as e:
                _logger.error('Failed to generate group architecture: %s', e)

        return docs

    async def generate_for_topic(
        self,
        title: str,
        description: str,
        source_files: list[Path],
    ) -> GeneratedDoc | None:
        """Generate a cross-cutting topic doc from multiple source files.

        Args:
            title: Topic title (e.g., "Ingest Pipeline").
            description: Brief description of what the topic covers.
            source_files: Paths to the source files that make up this topic.

        Returns:
            Generated topic documentation, or None if generation failed.
        """
        from docgen.prompts import TOPIC_SYSTEM, TOPIC_TEMPLATE

        # Read and summarize each file
        file_summaries = []
        for path in source_files:
            if not path.exists():
                _logger.warning('Topic source file not found: %s', path)
                continue
            source = path.read_text(encoding='utf-8')
            # Truncate large files to fit in prompt
            if len(source) > MAX_SOURCE_LENGTH // len(source_files):
                source = self._chunk_source_simple(source, MAX_SOURCE_LENGTH // len(source_files))
            file_summaries.append(f'### {path.name}\n**Path**: `{path}`\n```python\n{source}\n```')

        if not file_summaries:
            _logger.error('No valid source files for topic: %s', title)
            return None

        prompt = TOPIC_TEMPLATE.format(
            topic_title=title,
            topic_description=description,
            file_summaries='\n\n'.join(file_summaries),
        )

        content = await self._call_llm(TOPIC_SYSTEM, prompt)
        if content:
            return GeneratedDoc(
                title=title,
                content=content,
                doc_type='explanation',
                source_files=tuple(str(p) for p in source_files if p.exists()),
                metadata={'topic': True, 'topic_title': title},
            )
        return None

    @staticmethod
    def _chunk_source_simple(source: str, max_length: int) -> str:
        """Truncate source code to fit within a budget, keeping structure."""
        if len(source) <= max_length:
            return source
        # Keep first portion (imports + top-level defs) and note truncation
        return source[:max_length] + '\n# ... (truncated)\n'

    def _build_prompt_for_doc(
        self,
        metadata: ModuleMetadata,
        source_code: str,
        doc_type: DocType,
        extra_prompt_context: str = '',
    ) -> tuple[str, str]:
        """Build the ``(system, user)`` prompt pair for one doc_type.

        Shared between ``_generate_doc`` (streaming dispatch) and
        ``build_prompts_for_module`` (batch dispatch). Centralizing
        the construction keeps batch and streaming inputs byte-for-byte
        identical — divergence would silently regress quality on every
        batch run.
        """
        template = get_template(doc_type)
        if len(source_code) > MAX_SOURCE_LENGTH:
            source_code = self._chunk_source(source_code, metadata)
        prompt = self._format_prompt(template, metadata, source_code)
        if extra_prompt_context:
            prompt = f'{prompt}\n\n{extra_prompt_context}'
        return template.system_prompt, prompt

    async def _generate_doc(
        self,
        metadata: ModuleMetadata,
        source_code: str,
        doc_type: DocType,
        extra_prompt_context: str = '',
    ) -> str | None:
        """Generate a single documentation item.

        Args:
            metadata: Module metadata.
            source_code: The source code text.
            doc_type: Type of documentation to generate.
            extra_prompt_context: Optional markdown block appended to
                the prompt — used by reverse-augment to inject
                consumer-context info.

        Returns:
            Generated documentation content, or None if generation failed.
        """
        system_prompt, user_prompt = self._build_prompt_for_doc(
            metadata, source_code, doc_type, extra_prompt_context,
        )
        from docgen.calibration import usage_context
        with usage_context(
            phase='generate', doc_type=doc_type, language='python',
        ):
            return await self._call_llm(system_prompt, user_prompt)

    async def _generate_group_architecture(self, group: ModuleGroup) -> GeneratedDoc | None:
        """Generate architecture documentation for a module group.

        Args:
            group: The module group to document.

        Returns:
            Generated documentation, or None if generation failed.
        """
        template = get_template('architecture')

        # Prepare component info
        component_info = f'**Package**: `{group.name}`\n'
        component_info += f'**Modules**: {len(group.all_modules)}\n'
        component_info += f'**Classes**: {len(group.all_public_classes)}\n'
        component_info += f'**Functions**: {len(group.all_public_functions)}\n'

        # Prepare source overview (summaries of each module)
        source_parts = []
        for mod in group.all_modules[:10]:  # Limit to avoid token overflow
            source_parts.append(f'### {mod.module_name}')
            if mod.docstring:
                source_parts.append(mod.docstring.split('\n')[0])
            source_parts.append(f"Classes: {', '.join(c.name for c in mod.public_classes)}")
            source_parts.append(f"Functions: {', '.join(f.name for f in mod.public_functions)}")
            source_parts.append('')
        source_code = '\n'.join(source_parts)

        # Collect dependencies
        all_deps = set()
        internal_deps = set()
        for mod in group.all_modules:
            all_deps.update(mod.dependencies)
            internal_deps.update(self.analyzer.get_internal_dependencies(mod, group.name))

        dependencies = format_dependencies(list(all_deps), list(internal_deps))

        prompt = render_user_template(
            template, language='python',
            component_info=component_info,
            source_code=source_code,
            dependencies=dependencies,
            dependents='(Not analyzed at group level)',
        )

        content = await self._call_llm(template.system_prompt, prompt)
        if content:
            return GeneratedDoc(
                title=f'{group.name} Architecture',
                content=content,
                doc_type='architecture',
                source_files=tuple(str(mod.path) for mod in group.all_modules),
                metadata={'package_name': group.name},
            )
        return None

    def _format_prompt(
        self,
        template: PromptTemplate,
        metadata: ModuleMetadata,
        source_code: str,
    ) -> str:
        """Format a prompt template with module information.

        Args:
            template: The prompt template.
            metadata: Module metadata.
            source_code: Source code text.

        Returns:
            Formatted prompt string.
        """
        # Common info
        module_info = format_module_info(
            metadata.module_name,
            metadata.docstring,
            [c.name for c in metadata.public_classes],
            [f.name for f in metadata.public_functions],
        )

        # Template-specific formatting (legacy path is Python-only).
        if template.doc_type == 'explanation':
            related = format_related_modules(
                [(dep, None) for dep in list(metadata.dependencies)[:5]]
            )
            return render_user_template(
                template, language='python',
                module_info=module_info,
                source_code=source_code,
                related_modules=related,
            )

        elif template.doc_type == 'architecture':
            deps = format_dependencies(
                list(metadata.dependencies),
                list(self.analyzer.get_internal_dependencies(metadata, metadata.module_name.split('.')[0])),
            )
            return render_user_template(
                template, language='python',
                component_info=module_info,
                source_code=source_code,
                dependencies=deps,
                dependents='(Analysis not performed)',
            )

        elif template.doc_type == 'qa':
            return render_user_template(
                template, language='python',
                module_info=module_info,
                source_code=source_code,
                existing_docs='(No existing documentation)',
            )

        elif template.doc_type == 'catalog':
            return render_user_template(
                template, language='python',
                module_info=module_info,
                source_code=source_code,
            )

        elif template.doc_type == 'gotcha':
            return render_user_template(
                template, language='python',
                module_info=module_info,
                source_code=source_code,
            )

        elif template.doc_type == 'diagram':
            classes = [
                {
                    'name': c.name,
                    'bases': list(c.bases),
                    'methods': [m.name for m in c.public_methods],
                }
                for c in metadata.public_classes
            ]
            functions = [
                {
                    'name': f.name,
                    'args': [a.name for a in f.arguments if a.name != 'self'],
                }
                for f in metadata.public_functions
            ]
            classes_funcs = format_classes_functions(classes, functions)
            return render_user_template(
                template, language='python',
                component_info=module_info,
                classes_functions=classes_funcs,
                relationships='(Derived from imports and inheritance)',
            )

        return render_user_template(
            template, language='python',
            module_info=module_info,
            source_code=source_code,
        )

    def _chunk_source(self, source_code: str, metadata: ModuleMetadata) -> str:
        """Chunk large source code while preserving structure.

        Args:
            source_code: Full source code.
            metadata: Module metadata for structure info.

        Returns:
            Chunked source code that fits within limits.
        """
        lines = source_code.split('\n')

        # Strategy: Keep module docstring, imports, class/function signatures
        chunks = []
        current_length = 0
        max_length = MAX_SOURCE_LENGTH

        # Always include docstring
        if metadata.docstring:
            doc_lines = ['"""' + metadata.docstring + '"""', '']
            chunks.extend(doc_lines)
            current_length += sum(len(line) for line in doc_lines)

        # Include imports (abbreviated)
        import_section = []
        for imp in metadata.imports[:15]:
            if imp.is_from_import:
                names = ', '.join(imp.names[:3])
                if len(imp.names) > 3:
                    names += ', ...'
                import_section.append(f'from {imp.module} import {names}')
            else:
                import_section.append(f'import {imp.module}')
        if import_section:
            chunks.extend(import_section)
            chunks.append('')
            current_length += sum(len(line) for line in import_section)

        # Include class definitions with signatures
        for cls in metadata.classes:
            if current_length > max_length * 0.8:
                chunks.append('# ... (truncated)')
                break

            class_def = f'class {cls.name}'
            if cls.bases:
                class_def += f"({', '.join(cls.bases)})"
            class_def += ':'
            chunks.append(class_def)

            if cls.docstring:
                first_line = cls.docstring.split('\n')[0]
                chunks.append(f'    """{first_line}..."""')

            # Include method signatures
            for method in cls.methods[:10]:
                sig = f'    {method.signature}: ...'
                chunks.append(sig)
                current_length += len(sig)

            if len(cls.methods) > 10:
                chunks.append(f'    # ... and {len(cls.methods) - 10} more methods')

            chunks.append('')

        # Include function signatures
        for func in metadata.functions:
            if current_length > max_length:
                chunks.append('# ... (truncated)')
                break

            chunks.append(f'{func.signature}:')
            if func.docstring:
                first_line = func.docstring.split('\n')[0]
                chunks.append(f'    """{first_line}..."""')
            chunks.append('    ...')
            chunks.append('')
            current_length += 50  # Rough estimate

        return '\n'.join(chunks)

    def _generate_title(self, metadata: ModuleMetadata, doc_type: DocType) -> str:
        """Generate a title for the documentation.

        Args:
            metadata: Module metadata.
            doc_type: Type of documentation.

        Returns:
            Generated title string.
        """
        module_name = metadata.module_name.split('.')[-1]

        if doc_type == 'explanation':
            if metadata.docstring:
                first_line = metadata.docstring.split('\n')[0].strip()
                if len(first_line) < 50:
                    return first_line
            return f"{module_name.replace('_', ' ').title()} Module"

        elif doc_type == 'architecture':
            return f"{module_name.replace('_', ' ').title()} Architecture"

        elif doc_type == 'qa':
            return f"{module_name.replace('_', ' ').title()} FAQ"

        elif doc_type == 'diagram':
            return f"{module_name.replace('_', ' ').title()} Diagram"

        elif doc_type == 'catalog':
            return f"{module_name.replace('_', ' ').title()} Function Catalog"

        elif doc_type == 'gotcha':
            return f"{module_name.replace('_', ' ').title()} Gotchas"

        return module_name.title()

    # -------------------------------------------------------------------
    # Catalog-driven generation path (Phase 2.3) — consumes
    # EnrichedFileBundle instead of ModuleMetadata. Coexists with the
    # legacy ModuleMetadata path; the orchestrator picks via a flag.
    # -------------------------------------------------------------------

    async def generate_from_elements(
        self,
        bundle: EnrichedFileBundle,
        doc_types: tuple[DocType, ...] | None = None,
    ) -> list[GeneratedDoc]:
        """Generate documentation from a catalog-derived ``EnrichedFileBundle``.

        Mirror of :meth:`generate_for_module` but driven by the multi-language
        catalog extractor + Python enrichment. Non-Python bundles still flow
        through here; doc_types that depend on Python-only fields gracefully
        degrade (empty class/function lists, no docstring, no imports).
        Doc types are also filtered per-language so JSON/YAML/MD only get
        ``explanation`` even when the caller requests the full set.
        """
        requested = doc_types or self.config.doc_types
        doc_types = filter_doc_types_for_language(requested, bundle.language)
        bundle, source_code = self._prepare_bundle_source(bundle)

        _logger.info(
            'Generating docs from bundle %s (lang=%s, types: %s)',
            bundle.path, bundle.language, ', '.join(doc_types),
        )

        async def _gen_one(doc_type: DocType) -> GeneratedDoc | None:
            try:
                content = await self._generate_doc_from_bundle(
                    bundle, source_code, doc_type,
                )
                if not content:
                    return None
                title = self._generate_title_from_bundle(bundle, doc_type)
                return GeneratedDoc(
                    title=title,
                    content=content,
                    doc_type=doc_type,
                    source_files=(str(bundle.path),),
                    metadata={
                        'module_name': bundle.module_name,
                        'language': bundle.language,**_dangling_autodoc_meta(bundle)
                    },
                )
            except QuotaExhaustedError:
                # Re-raise so the orchestrator's abort-coordination
                # (run():427) sees it. See generate_for_module above
                # for the rationale.
                raise
            except Exception as e:
                _logger.error(
                    'Failed to generate %s for %s: %s',
                    doc_type, bundle.path, e,
                )
                return None

        results = await asyncio.gather(*[_gen_one(dt) for dt in doc_types])
        return [r for r in results if r is not None]

    # -------------------------------------------------------------------
    # Build-only path (#45.2) — produce prompts WITHOUT calling the LLM.
    # The orchestrator's batch dispatch (#45) collects prompts upfront
    # via these helpers, submits them through provider.submit_batch,
    # then wraps responses with assemble_doc to land identical
    # GeneratedDocs to the streaming path.
    # -------------------------------------------------------------------

    async def build_prompts_for_module(
        self,
        metadata: ModuleMetadata,
        doc_types: tuple[DocType, ...] | None = None,
        extra_prompt_context: str = '',
    ) -> list[PromptBundle]:
        """Build PromptBundles for a module without calling the LLM.

        Mirror of ``generate_for_module``'s setup phase. Reuses
        ``_build_prompt_for_doc`` so the prompts produced here are
        byte-identical to what the streaming path sends — that
        identity is the contract that lets batch and streaming
        produce equivalent docs for the same source.

        Pre-computes the per-doc-type ``title`` and packs the
        ``module_name``/``source_hash`` metadata so a batch result
        (which may arrive hours later) can be wrapped into a
        ``GeneratedDoc`` without re-running the analyzer.
        """
        doc_types = doc_types or self.config.doc_types
        source_code = metadata.path.read_text(encoding='utf-8')
        bundles: list[PromptBundle] = []
        for doc_type in doc_types:
            system_prompt, user_prompt = self._build_prompt_for_doc(
                metadata, source_code, doc_type, extra_prompt_context,
            )
            title = self._generate_title(metadata, doc_type)
            bundles.append(
                PromptBundle(
                    file=metadata.path,
                    doc_type=doc_type,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    title=title,
                    metadata={
                        'module_name': metadata.module_name,
                        'source_hash': metadata.source_hash,
                    },
                )
            )
        return bundles

    async def build_prompts_for_bundle(
        self,
        bundle: EnrichedFileBundle,
        doc_types: tuple[DocType, ...] | None = None,
    ) -> list[PromptBundle]:
        """Build PromptBundles for a catalog bundle without calling the
        LLM. Catalog twin of :meth:`build_prompts_for_module`.

        Applies ``filter_doc_types_for_language`` so non-Python files
        only emit the doc_types they support — burning batch tokens
        on a doc_type the language can't produce is wasted spend.
        """
        requested = doc_types or self.config.doc_types
        doc_types = filter_doc_types_for_language(requested, bundle.language)
        bundle, source_code = self._prepare_bundle_source(bundle)

        bundles: list[PromptBundle] = []
        for doc_type in doc_types:
            system_prompt, user_prompt = self._build_prompt_for_bundle_doc(
                bundle, source_code, doc_type,
            )
            title = self._generate_title_from_bundle(bundle, doc_type)
            bundles.append(
                PromptBundle(
                    file=bundle.path,
                    doc_type=doc_type,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    title=title,
                    metadata={
                        'module_name': bundle.module_name,
                        'language': bundle.language,**_dangling_autodoc_meta(bundle)
                    },
                )
            )
        return bundles

    def assemble_doc(
        self, prompt: PromptBundle, content: str,
    ) -> GeneratedDoc:
        """Wrap a batch response into a ``GeneratedDoc`` using the
        pre-computed title + metadata from the prompt.

        Symmetry point with the build helpers: downstream validation
        and storage cannot tell whether a doc came through batch or
        streaming. The batch path calls this for every successful
        response from ``provider.fetch_batch_results``.
        """
        return GeneratedDoc(
            title=prompt.title,
            content=content,
            doc_type=prompt.doc_type,
            source_files=(str(prompt.file),),
            metadata=prompt.metadata,
        )

    def _build_prompt_for_bundle_doc(
        self,
        bundle: EnrichedFileBundle,
        source_code: str,
        doc_type: DocType,
    ) -> tuple[str, str]:
        """Bundle-path twin of ``_build_prompt_for_doc``. Same role:
        keep batch and streaming prompts identical for the catalog
        path."""
        template = get_template(doc_type)
        if len(source_code) > MAX_SOURCE_LENGTH:
            source_code = self._chunk_source_from_bundle(source_code, bundle)
        prompt = self._format_prompt_from_bundle(template, bundle, source_code)
        return template.system_prompt, prompt

    async def _generate_doc_from_bundle(
        self,
        bundle: EnrichedFileBundle,
        source_code: str,
        doc_type: DocType,
    ) -> str | None:
        system_prompt, user_prompt = self._build_prompt_for_bundle_doc(
            bundle, source_code, doc_type,
        )
        # Tag the call's token usage for calibration (no-op without an
        # active observer; the provider emits usage mid-call).
        from docgen.calibration import usage_context
        with usage_context(
            phase='generate', doc_type=doc_type, language=bundle.language,
        ):
            return await self._call_llm(system_prompt, user_prompt)
    def _prepare_bundle_source(self, bundle):
        """Raw source for user repos; the filler swap for Spool builds (§18.4).

    Withhold mode renders the structural spine (no comments, no
    docstrings) and scrubs the module docstring so no tier-3 prose
    reaches the prompt; ``intent_filler`` is the only admitted prose.
    """
        if not self.config.withhold_source_prose:
            try:
                return bundle, bundle.path.read_text(encoding='utf-8')
            except (OSError, UnicodeDecodeError):
                return bundle, ''
        import attrs

        from docgen.filler_swap import structural_source

        scrubbed = attrs.evolve(bundle, module_docstring=None)
        return scrubbed, structural_source(
            bundle, intent_filler=self.intent_filler,
        )

    @staticmethod
    def _public_classes_from_bundle(
        bundle: EnrichedFileBundle,
    ) -> list[str]:
        names: list[str] = []
        for e in bundle.elements:
            if e.element.subtype not in _CLASS_LIKE_SUBTYPES:
                continue
            short = e.element.qualified_name.rsplit('.', 1)[-1]
            if short.startswith('_'):
                continue
            # Only top-level classes (parent is the module itself).
            parent = e.element.parent_qualified_name
            if parent is None or parent == bundle.module_name:
                names.append(short)
        return names

    @staticmethod
    def _public_functions_from_bundle(
        bundle: EnrichedFileBundle,
    ) -> list[str]:
        names: list[str] = []
        for e in bundle.elements:
            if e.element.subtype not in _CALLABLE_SUBTYPES:
                continue
            short = e.element.qualified_name.rsplit('.', 1)[-1]
            if short.startswith('_'):
                continue
            parent = e.element.parent_qualified_name
            # Top-level only — under the module/package, not under a class.
            if parent is None or parent == bundle.module_name:
                names.append(short)
        return names

    @staticmethod
    def _methods_for_class(
        bundle: EnrichedFileBundle, class_qn: str,
    ) -> list[str]:
        out: list[str] = []
        for e in bundle.elements:
            if e.element.subtype not in _CALLABLE_SUBTYPES:
                continue
            if e.element.parent_qualified_name != class_qn:
                continue
            short = e.element.qualified_name.rsplit('.', 1)[-1]
            if short.startswith('_'):
                continue
            out.append(short)
        return out

    def _format_prompt_from_bundle(
        self,
        template: PromptTemplate,
        bundle: EnrichedFileBundle,
        source_code: str,
    ) -> str:
        """Build the LLM user prompt from an EnrichedFileBundle."""
        public_classes = self._public_classes_from_bundle(bundle)
        public_functions = self._public_functions_from_bundle(bundle)

        module_info = format_module_info(
            bundle.module_name,
            bundle.module_docstring,
            public_classes,
            public_functions,
        )

        lang = bundle.language

        if template.doc_type == 'explanation':
            deps = sorted({imp.module.split('.')[0] for imp in bundle.imports})
            related = format_related_modules(
                [(d, None) for d in deps[:5]]
            )
            from docgen.prompts import format_rst_crossref
            return render_user_template(
                template, language=lang,
                module_info=module_info,
                source_code=source_code,
                related_modules=related,
                rst_crossref=format_rst_crossref(bundle.scip),
            )

        if template.doc_type == 'architecture':
            top_level_deps = sorted({
                imp.module.split('.')[0] for imp in bundle.imports
            })
            top_pkg = bundle.module_name.split('.')[0]
            internal = sorted({
                imp.module for imp in bundle.imports
                if imp.module.startswith(top_pkg)
            })
            deps = format_dependencies(top_level_deps, internal)
            # SCIP-derived dependents — populated when a CrossSourceGraph
            # was loaded at orchestrator startup (Phase 2 Change 2).
            # Without it, the LLM sees the legacy "(Analysis not
            # performed)" placeholder.
            if bundle.scip is not None and bundle.scip.callers:
                from docgen.prompts import format_cross_source_callers
                dependents = format_cross_source_callers(bundle.scip.callers)
            else:
                dependents = '(Analysis not performed)'
            if bundle.scip is not None and bundle.scip.callees:
                from docgen.prompts import format_cross_source_callees
                cross_source_calls = format_cross_source_callees(bundle.scip.callees)
            else:
                cross_source_calls = '(None detected)'
            return render_user_template(
                template, language=lang,
                component_info=module_info,
                source_code=source_code,
                dependencies=deps,
                dependents=dependents,
                cross_source_calls=cross_source_calls,
            )

        if template.doc_type == 'qa':
            return render_user_template(
                template, language=lang,
                module_info=module_info,
                source_code=source_code,
                existing_docs='(No existing documentation)',
            )

        if template.doc_type == 'catalog':
            return render_user_template(
                template, language=lang,
                module_info=module_info,
                source_code=source_code,
            )

        if template.doc_type == 'gotcha':
            return render_user_template(
                template, language=lang,
                module_info=module_info,
                source_code=source_code,
            )

        if template.doc_type == 'diagram':
            classes_dicts = []
            for e in bundle.elements:
                if e.element.subtype not in _CLASS_LIKE_SUBTYPES:
                    continue
                short = e.element.qualified_name.rsplit('.', 1)[-1]
                if short.startswith('_'):
                    continue
                parent = e.element.parent_qualified_name
                if parent is not None and parent != bundle.module_name:
                    continue
                bases = e.python.bases if e.python else ()
                classes_dicts.append({
                    'name': short,
                    'bases': list(bases),
                    'methods': self._methods_for_class(
                        bundle, e.element.qualified_name,
                    ),
                })
            functions_dicts = []
            for e in bundle.elements:
                if e.element.subtype not in _CALLABLE_SUBTYPES:
                    continue
                short = e.element.qualified_name.rsplit('.', 1)[-1]
                if short.startswith('_'):
                    continue
                parent = e.element.parent_qualified_name
                if parent is not None and parent != bundle.module_name:
                    continue
                args = []
                if e.python is not None:
                    args = [n for n in e.python.arg_names if n != 'self']
                functions_dicts.append({
                    'name': short,
                    'args': args,
                })
            classes_funcs = format_classes_functions(classes_dicts, functions_dicts)
            return render_user_template(
                template, language=lang,
                component_info=module_info,
                classes_functions=classes_funcs,
                relationships='(Derived from imports and inheritance)',
            )

        return render_user_template(
            template, language=lang,
            module_info=module_info,
            source_code=source_code,
        )

    def _chunk_source_from_bundle(
        self, source_code: str, bundle: EnrichedFileBundle,
    ) -> str:
        """Trim a large source down to a budget while keeping structure."""
        lines: list[str] = []
        budget = MAX_SOURCE_LENGTH

        if bundle.module_docstring:
            lines.extend([f'"""{bundle.module_docstring}"""', ''])
            budget -= sum(len(line) for line in lines)

        # Imports — keep the first 15.
        for imp in bundle.imports[:15]:
            if imp.is_from_import:
                names = ', '.join(imp.names[:3])
                if len(imp.names) > 3:
                    names += ', ...'
                lines.append(f'from {imp.module} import {names}')
            elif imp.alias:
                lines.append(f'import {imp.module} as {imp.alias}')
            else:
                lines.append(f'import {imp.module}')
        if bundle.imports:
            lines.append('')

        running = sum(len(line) for line in lines)
        # Class / function signatures.
        for e in bundle.elements:
            if running > budget * 0.8:
                lines.append('# ... (truncated)')
                break
            if e.element.subtype in _CLASS_LIKE_SUBTYPES:
                short = e.element.qualified_name.rsplit('.', 1)[-1]
                # Only Python carries parsed bases; non-Python uses the
                # element's signature line as a fallback.
                if e.python and e.python.bases:
                    base_str = f"({', '.join(e.python.bases)})"
                    lines.append(f'class {short}{base_str}:')
                elif e.element.signature:
                    lines.append(e.element.signature)
                else:
                    lines.append(f'class {short}')
                if e.python and e.python.docstring:
                    first = e.python.docstring.split('\n')[0]
                    lines.append(f'    """{first}..."""')
                running += 80
                lines.append('')
            elif e.element.subtype in _CALLABLE_SUBTYPES:
                # Skip if this is a method under a class — the class's
                # block above prints its methods. We surface only top-level
                # callables here.
                parent = e.element.parent_qualified_name
                if parent is not None and parent != bundle.module_name:
                    continue
                short = e.element.qualified_name.rsplit('.', 1)[-1]
                if e.python is not None:
                    # Python branch — reconstruct from arg_names.
                    arg_str = ', '.join(e.python.arg_names)
                    async_prefix = (
                        'async ' if e.element.subtype == 'async_function' else ''
                    )
                    lines.append(f'{async_prefix}def {short}({arg_str}):')
                    if e.python.docstring:
                        first = e.python.docstring.split('\n')[0]
                        lines.append(f'    """{first}..."""')
                    lines.append('    ...')
                elif e.element.signature:
                    # Non-Python branch — use the SCIP/extractor signature.
                    lines.append(e.element.signature)
                else:
                    lines.append(f'def {short}')
                running += 80
                lines.append('')
        return '\n'.join(lines)

    async def generate_for_directory(
        self,
        directory: Path,
        doc_types: tuple[DocType, ...] | None = None,
        concurrency: int = 3,
    ) -> list[GeneratedDoc]:
        """Generate docs for every catalog file in ``directory``.

        Walks ``iter_catalog_files`` (multi-language), builds an
        ``EnrichedFileBundle`` for each, and dispatches to
        ``generate_from_elements`` per file. When ``doc_types`` includes
        ``"architecture"`` and the directory has 2+ files, also emits a
        package-level architecture doc that aggregates the per-file
        summaries — the catalog-driven replacement for
        ``_generate_group_architecture``.
        """
        from docgen.catalog_enrich import enrich_file
        from docgen.catalog_writer import iter_catalog_files

        doc_types = doc_types or self.config.doc_types

        files = iter_catalog_files(directory)
        bundles: list[EnrichedFileBundle] = []
        for f in files:
            bundle = enrich_file(f, source_root=directory)
            if bundle is not None:
                bundles.append(bundle)

        docs: list[GeneratedDoc] = []

        # Per-file docs.
        sem = asyncio.Semaphore(concurrency)

        async def per_file(bundle: EnrichedFileBundle) -> list[GeneratedDoc]:
            async with sem:
                return await self.generate_from_elements(bundle, doc_types)

        results = await asyncio.gather(
            *[per_file(b) for b in bundles], return_exceptions=True,
        )
        for res in results:
            if isinstance(res, Exception):
                _logger.error('per-file generation failed: %s', res)
            else:
                docs.extend(res)

        # Group-level architecture (only if we have multiple files and
        # architecture was requested).
        if 'architecture' in doc_types and len(bundles) > 1:
            try:
                group_doc = await self._generate_group_arch_from_bundles(
                    directory, bundles,
                )
                if group_doc is not None:
                    docs.append(group_doc)
            except Exception as e:
                _logger.error('group architecture generation failed: %s', e)

        return docs

    async def _generate_group_arch_from_bundles(
        self,
        directory: Path,
        bundles: list[EnrichedFileBundle],
    ) -> GeneratedDoc | None:
        """Build the package-level architecture doc from a list of bundles."""
        template = get_template('architecture')
        group_name = directory.name

        component_info = (
            f'**Package**: `{group_name}`\n'
            f'**Files**: {len(bundles)}\n'
        )

        source_parts: list[str] = []
        for bundle in bundles[:10]:
            source_parts.append(f'### {bundle.module_name}')
            if bundle.module_docstring:
                source_parts.append(bundle.module_docstring.split('\n')[0])
            public_classes = self._public_classes_from_bundle(bundle)
            public_funcs = self._public_functions_from_bundle(bundle)
            if public_classes:
                source_parts.append(f"Classes: {', '.join(public_classes)}")
            if public_funcs:
                source_parts.append(f"Functions: {', '.join(public_funcs)}")
            source_parts.append('')
        source_code = '\n'.join(source_parts)

        all_deps: set[str] = set()
        internal_deps: set[str] = set()
        for bundle in bundles:
            for imp in bundle.imports:
                top = imp.module.split('.')[0]
                all_deps.add(top)
                if imp.module.startswith(group_name):
                    internal_deps.add(imp.module)
        dependencies = format_dependencies(sorted(all_deps), sorted(internal_deps))

        prompt = render_user_template(
            template, language='python',
            component_info=component_info,
            source_code=source_code,
            dependencies=dependencies,
            dependents='(Not analyzed at group level)',
        )

        content = await self._call_llm(template.system_prompt, prompt)
        if not content:
            return None
        return GeneratedDoc(
            title=f'{group_name} Architecture',
            content=content,
            doc_type='architecture',
            source_files=tuple(str(b.path) for b in bundles),
            metadata={
                'package_name': group_name,
                'group': True,
            },
        )

    def _generate_title_from_bundle(
        self, bundle: EnrichedFileBundle, doc_type: DocType,
    ) -> str:
        """Title generation that mirrors ``_generate_title`` for the new path."""
        module_name = bundle.module_name.split('.')[-1]
        readable = module_name.replace('_', ' ').title()

        if doc_type == 'explanation':
            if bundle.module_docstring:
                first_line = bundle.module_docstring.split('\n')[0].strip()
                if first_line and len(first_line) < 50:
                    return first_line
            return f'{readable} Module'
        if doc_type == 'architecture':
            return f'{readable} Architecture'
        if doc_type == 'qa':
            return f'{readable} FAQ'
        if doc_type == 'diagram':
            return f'{readable} Diagram'
        if doc_type == 'catalog':
            return f'{readable} Function Catalog'
        if doc_type == 'gotcha':
            return f'{readable} Gotchas'
        return readable

    async def _call_llm(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> str | None:
        """Delegate to the configured provider.

        The provider encapsulates retry, error handling, and per-backend
        request/response shape. See ``docgen/llm/openai.py`` and
        ``docgen/llm/anthropic.py``.
        """
        if self._provider is None:
            msg = 'DocGenerator must be used as async context manager'
            raise RuntimeError(msg)
        return await self._provider.call(
            system_prompt,
            user_prompt,
            max_tokens=MAX_OUTPUT_TOKENS,
        )

async def generate_docs(
    source_path: Path,
    config: GeneratorConfig | None = None,
    doc_types: tuple[DocType, ...] | None = None,
) -> list[GeneratedDoc]:
    """Convenience function to generate documentation for a path.

    Args:
        source_path: Path to a Python file or directory.
        config: Generator configuration.
        doc_types: Types of documentation to generate.

    Returns:
        List of generated documentation items.
    """
    config = config or GeneratorConfig()
    analyzer = SourceAnalyzer()

    async with DocGenerator(config=config, analyzer=analyzer) as generator:
        if source_path.is_file():
            return await generator.generate_for_file(source_path, doc_types)
        else:
            group = analyzer.analyze_directory(source_path)
            return await generator.generate_for_group(group, doc_types)
