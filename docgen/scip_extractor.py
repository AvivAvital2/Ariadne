"""SCIP-driven extractor for Scala/Java (SCIP plan, Phase A.5).

Replaces ast-grep for ``.scala``/``.sbt``/``.java`` when the source has
``index_kinds.scala = "scip"`` (or ``.java``) declared in ``ariadne.yaml``.
SCIP gives us the typed-tree facts the compiler itself sees — resolved
implicits, JVM-decoded overload signatures, structured scaladoc/javadoc.

The extractor is structured so its core logic operates on small frozen
duck-typed intermediates (``_ScipDoc`` / ``_ScipSymbol`` / ``_ScipOccurrence``).
``ScipIndex.load`` reads a real ``.scip`` file and converts the protobuf
into those intermediates; tests can synthesize them directly without
needing the protobuf bindings.
"""
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

# ---------------------------------------------------------------------------
# Test-friendly intermediates — populated either from real SCIP protobuf
# (in load()) or directly in tests.
# ---------------------------------------------------------------------------


@frozen
class _ScipOccurrence:
    """One occurrence of a symbol in a SCIP document.

    ``range`` is ``(start_line, start_col, end_col)`` (3-tuple, same line)
    or ``(start_line, start_col, end_line, end_col)`` (4-tuple).
    ``enclosing_range`` is the definition's *body* span (SCIP's separate field,
    same tuple shapes; ``()`` when absent — references, locals, and indexers
    that do not emit it). Lines/cols are 0-indexed in the wire format.
    """
    symbol: str
    range: tuple[int, ...]
    is_definition: bool
    enclosing_range: tuple[int, ...] = ()


@frozen
class _ScipSymbol:
    """SCIP ``SymbolInformation`` flattened to the bits the extractor uses."""
    symbol: str
    kind: str  # SCIP SymbolKind enum name: "Class", "Object", "Method", ...
    documentation: str = ''
    is_implicit: bool = False
    is_var: bool = False
    display_name: str = ''
    # SemanticDB's typed signature for this symbol (e.g.
    # "def bar(x: Int): String"). Falls back to a file-slice read in
    # extract() when empty.
    signature_text: str = ''


@frozen
class _ScipDoc:
    relative_path: str
    occurrences: tuple[_ScipOccurrence, ...] = ()
    symbols: tuple[_ScipSymbol, ...] = ()


# ---------------------------------------------------------------------------
# ScipIndex
# ---------------------------------------------------------------------------


@frozen
class ScipIndex:
    """Loaded SCIP index, keyed by relative path for ``document_for`` lookups.

    Constructed either via ``ScipIndex.load(path)`` (production) or
    directly with synthetic ``_ScipDoc`` tuples (tests).
    """
    documents: tuple[_ScipDoc, ...] = ()
    source_root: Path = field(factory=Path)

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        repo: str,
        max_staleness_days: int | None,
    ) -> "ScipIndex":
        """Read and parse a ``.scip`` artifact, raising structured errors.

        Failure modes (in order):
          1. ``ScipUnavailableError`` — file does not exist.
          2. ``ScipTooStaleError`` — mtime older than ``max_staleness_days``.
          3. ``ScipCorruptError`` — protobuf parse failure or scip_pb2
             bindings absent (run ``tests/setup_scip_pb2.py`` once).
        """
        if not path.exists():
            raise ScipUnavailableError(repo=repo, reason='index_missing')

        mtime = path.stat().st_mtime
        age_days = (time.time() - mtime) / 86400
        if max_staleness_days is not None and age_days > max_staleness_days:
            raise ScipTooStaleError(
                repo=repo,
                reason='index_too_stale',
                last_good_age_days=int(age_days),
            )

        # Parse protobuf — bindings may not be present.
        try:
            from docgen.scip import scip_pb2  # type: ignore[import-not-found]
        except ImportError as e:
            raise ScipCorruptError(
                repo=repo, reason='index_corrupt',
            ) from e

        try:
            wire = path.read_bytes()
            pb_index = scip_pb2.Index()
            pb_index.ParseFromString(wire)
        except Exception as e:
            raise ScipCorruptError(
                repo=repo, reason='index_corrupt',
            ) from e

        documents = tuple(_proto_to_doc(d) for d in pb_index.documents)
        return cls(documents=documents, source_root=path.parent)

    def document_for(
        self, file: Path, source_root: Path | None = None,
    ) -> _ScipDoc | None:
        """Look up the SCIP document for ``file``.

        SCIP records ``relative_path`` from the project root (where the
        scip-java build was invoked) — NOT relative to wherever the
        ``.scip`` artifact happens to live. Callers should pass
        ``source_root`` explicitly (the project root) for correct lookups.
        The instance's stored ``source_root`` is used only as a fallback;
        it is wrong whenever the artifact lives under a subdir like
        ``target/``.
        """
        root = source_root if source_root is not None else self.source_root
        try:
            rel = file.relative_to(root).as_posix()
        except ValueError:
            return None
        for doc in self.documents:
            if doc.relative_path == rel:
                return doc
        return None


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


def _proto_to_doc(pb_doc) -> _ScipDoc:
    """Translate a ``scip_pb2.Document`` into our intermediate ``_ScipDoc``.

    Lazy-imports the protobuf bindings so this module imports cleanly
    when scip_pb2 is missing (tests don't go through this path).
    """
    from docgen.scip import scip_pb2  # type: ignore[import-not-found]

    occurrences: list[_ScipOccurrence] = []
    for o in pb_doc.occurrences:
        is_def = bool(
            o.symbol_roles & scip_pb2.SymbolRole.Definition
        ) if hasattr(o, 'symbol_roles') else False
        occurrences.append(_ScipOccurrence(
            symbol=o.symbol,
            range=tuple(o.range),
            is_definition=is_def,
            enclosing_range=tuple(getattr(o, 'enclosing_range', ())),
        ))

    symbols: list[_ScipSymbol] = []
    for s in pb_doc.symbols:
        kind_name = scip_pb2.SymbolInformation.Kind.Name(s.kind) if hasattr(
            s, 'kind') else ''
        documentation = '\n'.join(s.documentation) if hasattr(
            s, 'documentation') else ''
        properties = getattr(s, 'properties', 0)
        is_implicit = bool(
            properties & scip_pb2.SymbolInformation.Property.IMPLICIT
        ) if properties else False
        is_var = bool(
            properties & scip_pb2.SymbolInformation.Property.VAR
        ) if properties else False
        # signature_documentation is a Document message with a `text` field
        # carrying the typed signature (e.g. "def bar(x: Int): String").
        signature_text = ''
        sig_doc = getattr(s, 'signature_documentation', None)
        if sig_doc is not None:
            signature_text = getattr(sig_doc, 'text', '') or ''
        symbols.append(_ScipSymbol(
            symbol=s.symbol,
            kind=kind_name,
            documentation=documentation,
            is_implicit=is_implicit,
            is_var=is_var,
            display_name=getattr(s, 'display_name', ''),
            signature_text=signature_text,
        ))

    return _ScipDoc(
        relative_path=pb_doc.relative_path,
        occurrences=tuple(occurrences),
        symbols=tuple(symbols),
    )


# ---------------------------------------------------------------------------
# extract — the entry point used by catalog_extractor
# ---------------------------------------------------------------------------


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


__all__ = [
    'ScipIndex',
    '_ScipDoc',
    '_ScipOccurrence',
    '_ScipSymbol',
    'extract',
]
