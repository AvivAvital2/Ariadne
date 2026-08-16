"""Catalog projection over the SCIP model.

The SCIP *reading* half of this module became ``docgen/scip_index.py``, where identity,
extents, relationships and empty-vs-absent are settled once. What remains here is the other
concern it used to carry: turning a document into catalog ``ElementInfo`` — scaladoc/jsdoc
parsing, subtype mapping, signature fallback.

That projection is **restored verbatim**, deliberately. It is not where any of the defects
lived, and its behaviour is pinned by 36 test files; rewriting it from memory would risk
regressions for no gain. Mixing it with index reading is what made the old module hard to
reason about, so the split is the fix, not a rewrite of both halves.

The model classes are re-exported under their historical private names so existing fixtures
construct them unchanged.
"""
from __future__ import annotations

from __future__ import annotations
import hashlib
import time
from pathlib import Path
from attrs import asdict, field, frozen
from docgen.catalog_extractor import ElementInfo
from docgen.doc_parser import parse_javadoc, parse_jsdoc, parse_scaladoc
from docgen.scip_config import (
    ScipCorruptError,
    ScipTooStaleError,
    ScipUnavailableError,
)
from docgen.scip_descriptors import (
    _parent_descriptor_kind,
    _qualified_name_from_symbol,
)


# The model, re-exported under the names callers already import.
from docgen.scip_index import ScipDocument as _ScipDoc  # noqa: E402,F401
from docgen.scip_index import ScipIndex  # noqa: E402,F401
from docgen.scip_index import ScipOccurrence as _ScipOccurrence  # noqa: E402,F401
from docgen.scip_index import ScipSymbolInfo as _ScipSymbol  # noqa: E402,F401
from docgen.scip_index import document_from_proto as _proto_to_doc  # noqa: E402,F401

__all__ = ['ScipIndex', '_ScipDoc', '_ScipOccurrence', '_ScipSymbol', 'extract']

def _offset_range(range_tuple, offset: int):
    """Add ``offset`` to the line components of a SCIP range tuple.
    Handles 3-tuple (same-line) and 4-tuple (multi-line) shapes, and an empty
    tuple (an absent ``enclosing_range``), which stays empty."""
    r = list(range_tuple)
    if not r:
        return ()
    if len(r) == 3:
        # (line, start_col, end_col)
        return (r[0] + offset, r[1], r[2])
    # (start_line, start_col, end_line, end_col)
    return (r[0] + offset, r[1], r[2] + offset, r[3])


def apply_vue_mapping(index: "ScipIndex", mapping: dict) -> "ScipIndex":
    """Return a new ScipIndex with ``.vue.script.{js,ts}`` paths
    translated back to original ``.vue`` files and line numbers offset
    by the mapping's ``line_offset``.

    Documents whose ``relative_path`` is not in the mapping pass
    through unchanged. The transformation is purely structural — no
    SCIP semantic interpretation needed. Shared by the cross-source
    graph loader and ``scip_config.resolve_index`` so both the
    structural graph and catalog extraction see ``.vue`` paths.
    """
    translated_docs = []
    for doc in index.documents:
        m = mapping.get(doc.relative_path)
        if m is None:
            translated_docs.append(doc)
            continue
        offset = int(m.get('line_offset', 0))
        new_occurrences = tuple(
            _ScipOccurrence(
                symbol=o.symbol,
                range=_offset_range(o.range, offset),
                is_definition=o.is_definition,
                enclosing_range=_offset_range(o.enclosing_range, offset),
            )
            for o in doc.occurrences
        )
        translated_docs.append(_ScipDoc(
            relative_path=m.get('original', doc.relative_path),
            occurrences=new_occurrences,
            symbols=doc.symbols,
        ))
    return ScipIndex(
        documents=tuple(translated_docs),
        source_root=index.source_root,
    )


def _range(occ: _ScipOccurrence) -> tuple[int, int, int, int]:
    """Convert SCIP's 0-indexed range to Ariadne's 1-indexed convention."""
    r = list(occ.range)
    if len(r) == 3:
        # Same-line range — (start_line, start_col, end_col).
        return (r[0] + 1, r[1], r[0] + 1, r[2])
    return (r[0] + 1, r[1], r[2] + 1, r[3])


def _subtype(
    symbol: _ScipSymbol,
    language: str,
    *,
    parent_descriptor_kind: str | None = None,
) -> str | None:
    """Map SCIP ``kind`` × properties × language → Ariadne ``Subtype``.

    Returns None for kinds we don't track (the caller silently skips them).

    For Python/JavaScript, ``parent_descriptor_kind`` disambiguates a
    ``Method`` symbol: if the parent is a type/class (``'type'``), it's a
    class method (``method``); otherwise it's a top-level function
    (``function`` for Python, ``js_function`` for JS/TS).
    """
    kind = symbol.kind

    if language == 'scala':
        if kind == 'Class':
            return 'scala_class'
        if kind == 'Object':
            return 'scala_object'
        if kind == 'Trait':
            return 'scala_trait'
        if kind == 'Method':
            return 'scala_implicit' if symbol.is_implicit else 'scala_def'
        if kind == 'Field':
            return 'scala_var' if symbol.is_var else 'scala_val'
        if kind == 'TypeAlias':
            return 'scala_type'
        if kind == 'PackageObject':
            return 'scala_package_object'
    elif language == 'java':
        if kind == 'Class':
            return 'java_class'
        if kind == 'Interface':
            return 'java_interface'
        if kind == 'Enum':
            return 'java_enum'
        if kind == 'Method':
            if symbol.display_name == '<init>':
                return 'java_constructor'
            return 'java_method'
        if kind == 'Field':
            return 'java_field'
    elif language == 'python':
        if kind == 'Class':
            return 'class'
        if kind in ('Method', 'Function'):
            # SCIP indexers vary on kind for callables: scip-python may
            # emit Method for any callable (Pyright treats modules as
            # objects), but a future indexer or version may emit Function
            # for top-level defs. Handle both; parent kind disambiguates.
            return 'method' if parent_descriptor_kind == 'type' else 'function'
        if kind in ('Variable', 'Field'):
            return 'variable'
    elif language == 'javascript':
        if kind == 'Class':
            return 'js_class'
        if kind in ('Method', 'Function'):
            # scip-typescript routinely emits Function for top-level
            # functions, Method for class members. Both must dispatch
            # correctly via parent kind.
            return (
                'method' if parent_descriptor_kind == 'type' else 'js_function'
            )
        if kind in ('Variable', 'Field'):
            return 'variable'
    elif language == 'go':
        # scip-go emits standard SCIP kinds. Go has no classes: a struct type
        # is the class-like aggregate, an interface the abstract contract.
        # Kind names track scip-go's output; a receiver (parent type) marks a
        # method vs a top-level func, mirroring the python/js dispatch.
        if kind == 'Struct':
            return 'go_struct'
        if kind == 'Interface':
            return 'go_interface'
        if kind in ('Method', 'Function'):
            return 'go_method' if parent_descriptor_kind == 'type' else 'go_function'
        if kind == 'Field':
            return 'go_field'
        if kind == 'Constant':
            return 'go_const'
        if kind == 'Variable':
            return 'go_var'
        if kind in ('Type', 'TypeAlias'):
            return 'go_type'
    return None


def extract(
    file: Path,
    *,
    source_root: Path,
    index: ScipIndex,
) -> list[ElementInfo]:
    """Build ``ElementInfo`` records for ``file`` from the SCIP index.

    Only emits one record per symbol with a ``Definition`` occurrence
    in the document — references to symbols defined elsewhere are skipped.
    """
    doc = index.document_for(file, source_root=source_root)
    if doc is None:
        return []

    suffix = file.suffix.lower()
    parse_doc = None
    if suffix == '.java':
        language = 'java'
        parse_doc = parse_javadoc
    elif suffix in ('.scala', '.sbt'):
        language = 'scala'
        parse_doc = parse_scaladoc
    elif suffix == '.py':
        # Python doc parsing (Pythondoc) not yet wired; SCIP signature_text
        # provides the typed signature, doc comments stay as raw strings.
        language = 'python'
    elif suffix in ('.ts', '.tsx', '.js', '.jsx', '.mjs', '.vue'):
        # JSDoc/TSDoc doc-comments parse into the same StructuredDoc as
        # Java/Scala. .vue resolves here because the index is vue-mapped
        # before this runs, so document_for('Foo.vue') finds the doc.
        language = 'javascript'
        parse_doc = parse_jsdoc
    elif suffix == '.go':
        # Godoc doc-comment parsing not yet wired; SCIP signature_text
        # provides the typed signature, doc comments stay as raw strings
        # (same posture as python).
        language = 'go'
    else:
        return []

    # Read the file once for signature-fallback (first non-blank line of
    # the definition slice). Empty list if unreadable — the fallback then
    # produces ``signature=""``.
    try:
        src_lines = file.read_text(encoding='utf-8').splitlines()
    except (OSError, UnicodeDecodeError):
        src_lines = []

    symbol_map = {s.symbol: s for s in doc.symbols}
    file_str = str(file)
    out: list[ElementInfo] = []

    for occ in doc.occurrences:
        if not occ.is_definition:
            continue
        sym = symbol_map.get(occ.symbol)
        if sym is None:
            continue
        parent_kind = _parent_descriptor_kind(occ.symbol)
        subtype = _subtype(sym, language, parent_descriptor_kind=parent_kind)
        if subtype is None:
            continue

        qn, parent_qn = _qualified_name_from_symbol(occ.symbol, language)
        line_start, col_start, line_end, col_end = _range(occ)

        documentation: dict | None = None
        if sym.documentation and parse_doc is not None:
            documentation = asdict(parse_doc(sym.documentation))

        # Signature: prefer SemanticDB's typed signature; fall back to
        # the first non-blank line of the file slice.
        signature = ''
        if sym.signature_text:
            for line in sym.signature_text.split('\n'):
                stripped = line.strip()
                if stripped:
                    signature = stripped
                    break
        if not signature and src_lines:
            line_idx = line_start - 1  # 1-indexed → 0-indexed
            if 0 <= line_idx < len(src_lines):
                signature = src_lines[line_idx].strip()

        # Body SHA from the file slice — incremental sync uses this to
        # detect content changes between runs. Without it, edits to a
        # Scala file silently classify as "unchanged" because the stored
        # empty string matches the new empty string.
        body_sha = ''
        if src_lines:
            slice_start = max(0, line_start - 1)
            slice_end = min(len(src_lines), line_end)
            body_text = '\n'.join(src_lines[slice_start:slice_end])
            body_sha = hashlib.sha256(
                body_text.encode('utf-8', errors='replace'),
            ).hexdigest()

        out.append(ElementInfo(
            language=language,  # type: ignore[arg-type]
            subtype=subtype,    # type: ignore[arg-type]
            file=file_str,
            qualified_name=qn,
            signature=signature,
            line_start=line_start,
            line_end=line_end,
            col_start=col_start,
            col_end=col_end,
            parent_qualified_name=parent_qn,
            body_sha=body_sha,
            documentation=documentation,
        ))
    if language == 'scala':
        from docgen.scip_scala_test_extractor import (
            extract_scalatest_cases,
            relabel_suites,
        )
        cases = extract_scalatest_cases(
            doc, file=file, source_text='\n'.join(src_lines))
        out = relabel_suites(out, cases)
        out.extend(cases)

    return out
