"""Catalog enrichment — Python-specific data on top of ElementInfo.

(Catalog transition Phase 2.1+2.2; SCIP cross-source enrichment Phase 2 Change 2.)

The catalog extractor (``docgen.catalog_extractor``) produces a flat list of
``ElementInfo`` from any supported file. That shape is sufficient for
language-agnostic structural data, but Python-only fields the legacy
``SourceAnalyzer`` extracted (decorators, parsed args, return annotations,
docstrings, base classes, dataclass/attrs/abstract flags, structured
imports, module docstring) used to flow into the doc-generation prompts.

This module adds those fields back as a *sidecar* to ``ElementInfo`` —
parsed via ``ast.parse`` once per file. Non-Python languages return a
bundle with empty enrichment, so the new generator path can run uniformly
across languages.

Phase 2 Change 2 adds an *additional* sidecar — ``ScipFileMetadata`` —
populated when ``enrich_file`` is given a loaded ``CrossSourceGraph``.
This surfaces cross-source callers/callees for each function/method/class
in the file so the LLM prompt can describe how the file connects to the
rest of the codebase, not just its in-file structure.
"""
from __future__ import annotations

import ast
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from attrs import frozen

from docgen.catalog_extractor import (
    ElementInfo,
    Language,
    _detect_language,
    _py_module_qn,
    extract_elements,
)
from docgen.scip_config import ScipError, ScipIndexNotReadyError
from ast_utils import safe_ast_parse

if TYPE_CHECKING:
    from docgen.scip_cross_source import CrossSourceGraph

ArgKind = Literal['positional', 'keyword', 'var_positional', 'var_keyword']


@frozen
class PythonEnrichment:
    """Python-specific data extracted via ``ast.parse``.

    Mirrors the parts of ``FunctionInfo``/``ClassInfo``/``ArgumentInfo``
    that actually reach a generator prompt (per the Phase 2 spec).
    Defaults are empty/None so a non-Python element with no enrichment
    is still constructible.
    """
    decorators: tuple[str, ...] = ()
    arg_names: tuple[str, ...] = ()
    arg_kinds: tuple[ArgKind, ...] = ()
    arg_annotations: tuple[str | None, ...] = ()
    arg_defaults: tuple[str | None, ...] = ()
    return_annotation: str | None = None
    bases: tuple[str, ...] = ()
    docstring: str | None = None
    is_dataclass: bool = False
    is_attrs: bool = False
    is_abstract: bool = False


@frozen
class StructuredImport:
    """A single import statement as the legacy ``ImportInfo`` carried it.

    Reaches the architecture prompt via ``format_dependencies`` and the
    explanation/qa/etc. prompts via the imports-section of ``_chunk_source``.
    """
    module: str
    names: tuple[str, ...] = ()
    is_from_import: bool = False
    alias: str | None = None
    lineno: int = 0


@frozen
class EnrichedElementInfo:
    """An ``ElementInfo`` with optional language-specific enrichment.

    For Python elements (function/class/method/async_function), ``python``
    is populated. For non-Python languages or for variable-subtype Python
    elements, ``python`` may be None.
    """
    element: ElementInfo
    python: PythonEnrichment | None = None


@frozen
class ScipCaller:
    """An external (cross-file) symbol that calls a symbol in this file.

    ``local_qualified_name`` matches an ``ElementInfo.qualified_name``
    in this file; the remote fields locate the caller. The prompt
    template renders these so the LLM knows who depends on this file's
    public API.
    """
    local_qualified_name: str
    remote_qualified_name: str
    remote_source_name: str
    remote_file: str
    remote_line: int = 0


@frozen
class ScipCallee:
    """An external (cross-file) symbol called from a symbol in this file.

    Inverse of ``ScipCaller``. ``local_qualified_name`` is the function/
    method in this file that initiates the outbound call.
    """
    local_qualified_name: str
    remote_qualified_name: str
    remote_source_name: str
    remote_file: str
    remote_line: int = 0


@frozen
class ScipFileMetadata:
    """SCIP-derived cross-source data for one file.

    Populated by ``enrich_file`` when a ``CrossSourceGraph`` is provided.
    Edges are restricted to those crossing a file boundary — same-file
    references add nothing the in-file element list doesn't already
    describe.
    """
    callers: tuple[ScipCaller, ...] = ()
    callees: tuple[ScipCallee, ...] = ()
    autodoc_links: tuple[AutodocLink, ...] = ()
    documented_by_rst: tuple[ReverseAutodocLink, ...] = ()


@frozen
class EnrichedFileBundle:
    """Per-file aggregate consumed by the new generator path."""
    path: Path
    language: Language
    module_name: str
    module_docstring: str | None = None
    imports: tuple[StructuredImport, ...] = ()
    elements: tuple[EnrichedElementInfo, ...] = ()
    line_count: int = 0
    scip: ScipFileMetadata | None = None


# ---------------------------------------------------------------------------
# AST helpers (mirror docgen.analyzer to avoid coupling)
# ---------------------------------------------------------------------------


def _annotation_str(node: ast.expr | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _decorator_names(decorator_list: list[ast.expr]) -> tuple[str, ...]:
    names: list[str] = []
    for dec in decorator_list:
        if isinstance(dec, ast.Name):
            names.append(dec.id)
        elif isinstance(dec, ast.Attribute):
            names.append(ast.unparse(dec))
        elif isinstance(dec, ast.Call):
            if isinstance(dec.func, ast.Name):
                names.append(dec.func.id)
            elif isinstance(dec.func, ast.Attribute):
                names.append(ast.unparse(dec.func))
            else:
                names.append(ast.unparse(dec))
        else:
            names.append(ast.unparse(dec))
    return tuple(names)


_DATACLASS_DECORATORS = ('dataclass', 'dataclasses.dataclass')
_ATTRS_DECORATORS = {
    'define', 'frozen', 'attrs', 'attr.s',
    'attrs.define', 'attrs.frozen',
}


def _is_dataclass(decorators: tuple[str, ...]) -> bool:
    return any(d in _DATACLASS_DECORATORS for d in decorators)


def _is_attrs_class(decorators: tuple[str, ...]) -> bool:
    return any(d in _ATTRS_DECORATORS for d in decorators)


def _is_abstract_class(bases: tuple[str, ...], decorators: tuple[str, ...]) -> bool:
    if 'ABC' in bases or 'abc.ABC' in bases:
        return True
    return 'abstractmethod' in decorators or 'abc.abstractmethod' in decorators


def _parse_arguments(
    args: ast.arguments,
) -> tuple[
    tuple[str, ...], tuple[ArgKind, ...], tuple[str | None, ...], tuple[str | None, ...],
]:
    """Return (names, kinds, annotations, defaults) — parallel tuples."""
    names: list[str] = []
    kinds: list[ArgKind] = []
    annotations: list[str | None] = []
    defaults: list[str | None] = []

    num_positional = len(args.posonlyargs) + len(args.args)
    num_defaults = len(args.defaults)
    default_offset = num_positional - num_defaults

    for i, arg in enumerate(args.posonlyargs):
        idx = i - default_offset
        default = ast.unparse(args.defaults[idx]) if 0 <= idx < len(args.defaults) else None
        names.append(arg.arg)
        kinds.append('positional')
        annotations.append(_annotation_str(arg.annotation))
        defaults.append(default)

    for i, arg in enumerate(args.args):
        idx = i + len(args.posonlyargs) - default_offset
        default = ast.unparse(args.defaults[idx]) if 0 <= idx < len(args.defaults) else None
        names.append(arg.arg)
        kinds.append('positional')
        annotations.append(_annotation_str(arg.annotation))
        defaults.append(default)

    if args.vararg:
        names.append(args.vararg.arg)
        kinds.append('var_positional')
        annotations.append(_annotation_str(args.vararg.annotation))
        defaults.append(None)

    for i, arg in enumerate(args.kwonlyargs):
        default = (
            ast.unparse(args.kw_defaults[i])
            if i < len(args.kw_defaults) and args.kw_defaults[i] is not None
            else None
        )
        names.append(arg.arg)
        kinds.append('keyword')
        annotations.append(_annotation_str(arg.annotation))
        defaults.append(default)

    if args.kwarg:
        names.append(args.kwarg.arg)
        kinds.append('var_keyword')
        annotations.append(_annotation_str(args.kwarg.annotation))
        defaults.append(None)

    return tuple(names), tuple(kinds), tuple(annotations), tuple(defaults)


def _function_enrichment(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> PythonEnrichment:
    decorators = _decorator_names(node.decorator_list)
    names, kinds, annotations, defaults = _parse_arguments(node.args)
    return PythonEnrichment(
        decorators=decorators,
        arg_names=names,
        arg_kinds=kinds,
        arg_annotations=annotations,
        arg_defaults=defaults,
        return_annotation=_annotation_str(node.returns),
        docstring=ast.get_docstring(node),
    )


def _class_enrichment(node: ast.ClassDef) -> PythonEnrichment:
    decorators = _decorator_names(node.decorator_list)
    bases = tuple(ast.unparse(b) for b in node.bases)
    return PythonEnrichment(
        decorators=decorators,
        bases=bases,
        docstring=ast.get_docstring(node),
        is_dataclass=_is_dataclass(decorators),
        is_attrs=_is_attrs_class(decorators),
        is_abstract=_is_abstract_class(bases, decorators),
    )


def _walk_python(
    tree: ast.Module, module_name: str,
) -> dict[str, PythonEnrichment]:
    """Walk the AST building qualified_name → enrichment for classes/functions."""
    out: dict[str, PythonEnrichment] = {}

    def _visit(node: ast.AST, parent_qn: str) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, ast.ClassDef):
                qn = f'{parent_qn}.{child.name}'
                out[qn] = _class_enrichment(child)
                _visit(child, qn)
            elif isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef):
                qn = f'{parent_qn}.{child.name}'
                out[qn] = _function_enrichment(child)
                # Functions can contain nested defs/classes; recurse for completeness.
                _visit(child, qn)

    _visit(tree, module_name)
    return out


def _extract_imports(tree: ast.Module) -> tuple[StructuredImport, ...]:
    out: list[StructuredImport] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                out.append(StructuredImport(
                    module=alias.name,
                    alias=alias.asname,
                    is_from_import=False,
                    lineno=node.lineno,
                ))
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ''
            names = tuple(a.name for a in node.names)
            out.append(StructuredImport(
                module=module,
                names=names,
                is_from_import=True,
                lineno=node.lineno,
            ))
    return tuple(out)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def enrich_python_elements(
    file: Path, src: str, *, module_name: str,
) -> dict[str, PythonEnrichment]:
    """Parse Python ``src`` once, return a qualified_name → PythonEnrichment map.

    The keys match ``ElementInfo.qualified_name`` produced by
    ``catalog_extractor`` for class/function/method elements, so callers
    can attach the enrichment by name.
    """
    try:
        tree = safe_ast_parse(src, filename=str(file))
    except SyntaxError:
        return {}
    return _walk_python(tree, module_name)


def enrich_file(
    path: Path,
    *,
    source_root: Path,
    src: str | None = None,
    source_config: object | None = None,
    cross_source_graph: 'CrossSourceGraph | None' = None,
) -> EnrichedFileBundle | None:
    """Build a full ``EnrichedFileBundle`` for ``path``.

    For Python files, parses with both ast-grep (via ``extract_elements``)
    and ``ast.parse`` (for the enrichment), then merges them. For other
    supported languages, returns a bundle with elements but no Python
    enrichment. Returns None for unsupported extensions / unreadable files.

    ``source_config`` is forwarded to ``extract_elements``: required for
    Scala/Java to route through the SCIP-backed extractor; ignored by the
    other languages.

    ``cross_source_graph`` (optional, Phase 2 Change 2) — when provided,
    the resulting bundle's ``scip`` field carries cross-file callers and
    callees resolved against the materialized SCIP graph. The graph is
    expected to already be loaded (via ``load_from(conn)`` or
    ``add_source`` + ``materialize``); ``enrich_file`` does no I/O.
    """
    language = _detect_language(path)
    if language is None:
        return None

    if src is None:
        try:
            src = path.read_text(encoding='utf-8')
        except (OSError, UnicodeDecodeError):
            return None

    line_count = len(src.splitlines())
    try:
        elements = extract_elements(
            path, source_root=source_root, source_config=source_config,
        )
    except ScipError as exc:
        # A SCIP-routed language whose index was never built raises a terse
        # ScipError here. Translate it into one actionable, fail-loud error
        # (source, language, artifact, remedy) rather than letting it escape the
        # generate path raw. Only reached when a real file of that language is
        # actually extracted through SCIP.
        raise ScipIndexNotReadyError(
            repo=exc.repo,
            reason=exc.reason,
            language=language,
            artifact=getattr(source_config, 'artifact_path', None),
            remedy_cmd=f'ariadne index --source {exc.repo}',
        ) from exc

    scip_metadata = _compute_scip_metadata(
        elements, path, source_root, cross_source_graph,
    )

    if language == 'python':
        module_name = _py_module_qn(path, source_root)
        try:
            tree = safe_ast_parse(src, filename=str(path))
            module_docstring = ast.get_docstring(tree)
            imports = _extract_imports(tree)
            enrichment_map = _walk_python(tree, module_name)
        except SyntaxError:
            module_docstring = None
            imports = ()
            enrichment_map = {}

        enriched = tuple(
            EnrichedElementInfo(
                element=el,
                python=enrichment_map.get(el.qualified_name),
            )
            for el in elements
        )
        return EnrichedFileBundle(
            path=path,
            language=language,
            module_name=module_name,
            module_docstring=module_docstring,
            imports=imports,
            elements=enriched,
            line_count=line_count,
            scip=scip_metadata,
        )

    # Non-Python: empty enrichment, but elements still flow. JVM imports
    # (Scala/Java) populate the bundle's imports tuple via regex on the
    # source — SCIP doesn't surface them as structured intermediates and
    # the prompt's Dependencies section needs *something*.
    enriched = tuple(EnrichedElementInfo(element=el) for el in elements)
    module_name = _module_name_for_non_python(elements, path, source_root)
    imports: tuple[StructuredImport, ...] = ()
    if language in ('scala', 'java'):
        imports = _extract_jvm_imports(src)
    return EnrichedFileBundle(
        path=path,
        language=language,
        module_name=module_name,
        elements=enriched,
        imports=imports,
        line_count=line_count,
        scip=scip_metadata,
    )


_SCIP_RESOLVABLE_SUBTYPES = frozenset({
    'function', 'async_function', 'method', 'class',
})


@frozen
class AutodocLink:
    """A Sphinx autodoc directive resolved (or not) against the SCIP graph.

    ``resolved=False`` is the "docs reference X, but X is not in the indexed
    code" signal -- recorded explicitly, never silently dropped.
    """
    section_qualified_name: str
    target: str
    symbol_qualified_name: str | None = None
    symbol_file: str | None = None
    resolved: bool = False


def _resolve_autodoc_links(
    elements: list[ElementInfo],
    graph: 'CrossSourceGraph | None',
) -> tuple[AutodocLink, ...]:
    """Resolve each rst section's ``autodoc_targets`` against the SCIP graph.

    A hit links the section to the real code symbol; a miss is recorded with
    ``resolved=False`` (the "documented but not in the code" finding).
    Returns ``()`` when no graph is available -- the source simply isn't
    SCIP-indexed; the file is still documented.
    """
    if graph is None:
        return ()
    links: list[AutodocLink] = []
    for el in elements:
        for target in el.autodoc_targets:
            res = graph.resolve_symbol(target)
            sym = res.symbol
            links.append(AutodocLink(
                section_qualified_name=el.qualified_name,
                target=target,
                symbol_qualified_name=sym.qualified_name if sym else None,
                symbol_file=sym.file if sym else None,
                resolved=sym is not None,
            ))
    return tuple(links)


@frozen
class ReverseAutodocLink:
    """A code symbol documented by an rst section -- the reverse of
    :class:`AutodocLink`. Lets a code doc surface its human rst rationale.
    """
    symbol_qualified_name: str
    rst_section_qualified_name: str


def _compute_scip_metadata(
    elements: list[ElementInfo],
    path: Path,
    source_root: Path,
    graph: 'CrossSourceGraph | None',
) -> ScipFileMetadata | None:
    """Resolve each catalog element against the SCIP graph, return its
    cross-file callers and callees.

    Returns ``None`` when ``graph`` is None (caller didn't ask for SCIP
    enrichment). Returns an empty ``ScipFileMetadata`` when the graph
    is loaded but the file's symbols aren't present in it (e.g., the
    file post-dates the last index run).

    Same-file edges are excluded — the in-file element list already
    captures intra-file structure. Only edges crossing a file boundary
    contribute, since those are the ones the prompt's Cross-Source
    section is meant to surface.
    """
    if graph is None:
        return None

    callers: list[ScipCaller] = []
    callees: list[ScipCallee] = []
    documented: list[ReverseAutodocLink] = []

    for el in elements:
        if el.subtype not in _SCIP_RESOLVABLE_SUBTYPES:
            continue

        resolution = graph.resolve_symbol(el.qualified_name)
        sym = resolution.symbol
        if sym is None:
            # Not in the graph, or ambiguous (multiple matches at the
            # best tier). Either way, skip — surfacing an unverified
            # candidate to the LLM would be worse than silence.
            continue

        local_file = sym.file

        for edge in graph.callers_of(sym.canonical_id):
            if edge.caller.file == local_file:
                continue
            callers.append(ScipCaller(
                local_qualified_name=el.qualified_name,
                remote_qualified_name=edge.caller.qualified_name,
                remote_source_name=edge.caller.source_name,
                remote_file=edge.caller.file,
                remote_line=edge.line,
            ))

        for edge in graph.callees_of(sym.canonical_id):
            if edge.callee.file == local_file:
                continue
            callees.append(ScipCallee(
                local_qualified_name=el.qualified_name,
                remote_qualified_name=edge.callee.qualified_name,
                remote_source_name=edge.callee.source_name,
                remote_file=edge.callee.file,
                remote_line=edge.line,
            ))

        for section_qn in graph.rst_sections_documenting(sym.qualified_name):
            documented.append(ReverseAutodocLink(
                symbol_qualified_name=sym.qualified_name,
                rst_section_qualified_name=section_qn,
            ))

    return ScipFileMetadata(
        callers=tuple(callers),
        callees=tuple(callees),
        documented_by_rst=tuple(documented),
        autodoc_links=_resolve_autodoc_links(elements, graph),
    )


def _module_name_for_non_python(
    elements: list[ElementInfo], path: Path, source_root: Path,
) -> str:
    """Return a stable, file-unique module identifier for non-Python files.

    For SCIP-extracted Scala/Java elements, ``parent_qualified_name`` on a
    top-level element IS the package — combine it with the file stem so
    each file gets a unique identity (``com.example.Foo`` for Foo.scala).
    Without the file stem, two files in the same package would produce
    identical bundle ``module_name`` values, and the generator would
    create duplicate docs sharing the same title.

    For HTML/JS/etc. (no SCIP enrichment) fall back to the path-relative
    stem, which is also file-unique.
    """
    # Find the package via the first element with a parent.
    package = None
    for el in elements:
        if el.parent_qualified_name:
            package = el.parent_qualified_name
            break

    if package:
        return f'{package}.{path.stem}'

    try:
        rel = path.relative_to(source_root).with_suffix('')
    except ValueError:
        rel = Path(path.stem)
    parts = list(rel.parts)
    return '.'.join(parts) if parts else path.stem


import re as _re

# Match Scala / Java import statements. The captured group is the
# dotted path. Trailing brace-list (``{Bar, Baz}``), wildcard (``._``
# / ``.*``), or trailing ``.`` is stripped after the match.
_JVM_IMPORT_RE = _re.compile(
    r'^\s*import\s+(?:static\s+)?([\w][\w.]*)',
    _re.MULTILINE,
)


def _extract_jvm_imports(src: str) -> tuple[StructuredImport, ...]:
    """Pull import statements out of Scala/Java source text.

    Lossless enough for prompt rendering — the architecture doc just
    needs to know what packages a file pulls in. Brace-form and
    wildcard imports surface as the prefix package.
    """
    out: list[StructuredImport] = []
    seen: set[str] = set()
    for m in _JVM_IMPORT_RE.finditer(src):
        path = m.group(1).rstrip('._')
        if not path or path in seen:
            continue
        seen.add(path)
        out.append(StructuredImport(module=path, is_from_import=False))
    return tuple(out)


__all__ = [
    'ArgKind',
    'EnrichedElementInfo',
    'EnrichedFileBundle',
    'PythonEnrichment',
    'ScipCallee',
    'ScipCaller',
    'ScipFileMetadata',
    'StructuredImport',
    'enrich_file',
    'enrich_python_elements',
]
