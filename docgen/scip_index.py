"""The SCIP index, modelled once.

Stage one of the north star travels this model, so what SCIP means is decided **here**,
at load, rather than reconstructed at each point of use. Every defect in the module this
replaces came from the latter: the meaning of a local id, of a definition's extent, of an
empty-versus-absent field, and of a relationship was re-derived by each consumer, slightly
differently, and the disagreements were invisible.

Four invariants hold by construction:

* **Identity.** SCIP numbers local bindings per document, so the first local in every file
  is ``local 0``. A bare id is therefore one row that every document points at — measured
  on the live store: 7,910 bare rows, edges from 4,446 files aimed at ``local 5``, 16,061
  in-edges on ``local 0``, and 499,249 edges joining symbols from different sources. Ids
  are scoped to their document at load, keeping the ``local `` prefix so every existing
  detector still fires.
* **Extent.** A definition's ``range`` is its *identifier*; the body is ``enclosing_range``,
  and scip-java emits none at all. Extent is decided **per definition** — supplied range
  when there is one, positional reconstruction when there is not — because gating that per
  document leaves the ones that lack it collapsed onto a single line. ``line_start`` stays
  the identifier line: it is the key that resolves 95-100% of callables.
* **Relationships.** ``is_implementation`` answers *which implementation runs* at a
  polymorphic dispatch. It is first-class here; it previously had zero readers, which is
  why the edge-type set was only ``call`` and ``type_ref``.
* **Empty is absent.** scip-python supplies ``kind`` as ``UnspecifiedKind`` and
  ``display_name`` as ``''`` — present but empty, on 96-99% of rows. One rule, applied
  once, instead of a fallback that fires only on absence and so never fires.
"""
from __future__ import annotations

import re
import time
from pathlib import Path

from attrs import field, frozen

#: A local id SCIP numbered per document, before it is scoped. Two shapes exist and both
#: are document-local: ``local 5`` and ``local 5(self)`` — scip-python appends the binding's
#: name. Measured over the live store's 7,910 local rows: 6,847 bare, 1,063 named, and this
#: pattern matches all of them. Requiring the id to END at the digits missed every named
#: one, so ``local 19(self)`` stayed the same string in every file that had a ``self`` at
#: that index — fusing documents AND sources, which is what left 12 cross-source edges in
#: an otherwise clean rebuild.
#:
#: The character after the space is a digit for an unscoped id and never one once scoped,
#: which is what lets SQL distinguish them with ``GLOB 'local [0-9]*'``.
_BARE_LOCAL = re.compile(r'^local \d+(\(.*\))?$')

#: What scip-python puts in ``kind`` when it has nothing to say.
_UNSPECIFIED_KIND = 'UnspecifiedKind'


def _last_descriptor(symbol: str) -> str:
    """The trailing name in a SCIP symbol — ``…/run().`` → ``run``."""
    return symbol.rsplit('/', 1)[-1].rstrip('.#)(').strip('`')


def _line_range(wire: tuple[int, ...]) -> tuple[int, int]:
    """A SCIP wire range (0-indexed) as 1-indexed ``(start_line, end_line)``.

    Three-tuple ``(line, start_col, end_col)`` is a single line; four-tuple
    ``(start_line, start_col, end_line, end_col)`` spans lines.
    """
    if not wire:
        return (0, 0)
    if len(wire) == 3:
        return (wire[0] + 1, wire[0] + 1)
    return (wire[0] + 1, wire[2] + 1)


@frozen
class ScipRelationship:
    """One ``SymbolInformation.Relationship`` — what this symbol is to another."""

    symbol: str
    is_implementation: bool = False
    is_reference: bool = False
    is_type_definition: bool = False
    is_definition: bool = False


@frozen
class ScipOccurrence:
    """One appearance of a symbol in a document."""

    symbol: str
    range: tuple[int, ...]
    is_definition: bool = False
    enclosing_range: tuple[int, ...] = ()

    @property
    def identifier_lines(self) -> tuple[int, int]:
        return _line_range(self.range)

    @property
    def is_local(self) -> bool:
        return self.symbol.startswith('local ')

    @property
    def is_parameter(self) -> bool:
        """``Foo#bar().(self)`` is not a standalone definition."""
        return self.symbol.endswith(')') and ').(' in self.symbol


@frozen
class ScipSymbolInfo:
    """``SymbolInformation`` — the metadata SCIP carries about a symbol."""

    symbol: str
    kind: str = ''
    documentation: str = ''
    is_implicit: bool = False
    is_var: bool = False
    display_name: str = ''
    signature_text: str = ''
    relationships: tuple[ScipRelationship, ...] = ()

    @property
    def effective_display_name(self) -> str:
        """The name to show. Empty is absent — see the module docstring."""
        return self.display_name or _last_descriptor(self.symbol)

    @property
    def effective_kind(self) -> str | None:
        """The kind, or ``None`` when SCIP declined to say.

        ``UnspecifiedKind`` is not a kind. Returning it as one is what let
        kind-gated logic silently no-op across every scip-python corpus.
        """
        if not self.kind or self.kind == _UNSPECIFIED_KIND:
            return None
        return self.kind


@frozen
class ScipDocument:
    """One file's occurrences and symbol metadata."""

    relative_path: str
    occurrences: tuple[ScipOccurrence, ...] = ()
    symbols: tuple[ScipSymbolInfo, ...] = ()

    @property
    def last_line(self) -> int:
        """The last line this document mentions at all.

        The bound for a trailing definition's body, so an extent is never a sentinel a
        body read would have to clamp.
        """
        return max((occ.identifier_lines[1] for occ in self.occurrences), default=0)

    def definitions(self) -> tuple[ScipOccurrence, ...]:
        return tuple(occ for occ in self.occurrences if occ.is_definition)

    def _boundaries(self) -> list[int]:
        """Start lines that may end a preceding definition's body.

        Locals and parameters are excluded: scip-java emits them densely, and letting
        one end a method's body would cut the method off at its first local.
        """
        return sorted({
            occ.identifier_lines[0] for occ in self.definitions()
            if not occ.is_local and not occ.is_parameter
        })

    def extent_of(self, occurrence: ScipOccurrence) -> tuple[int, int]:
        """``(line_start, line_end)`` for a definition — identifier start, body end.

        Decided per definition, so a document that mixes supplied and missing
        ``enclosing_range`` gets a real extent for both kinds.
        """
        start, identifier_end = occurrence.identifier_lines
        if occurrence.enclosing_range:
            return (start, max(identifier_end, _line_range(occurrence.enclosing_range)[1]))
        nxt = next((line for line in self._boundaries() if line > start), None)
        end = (nxt - 1) if nxt is not None else self.last_line
        return (start, max(identifier_end, end))

    def implementations(self) -> tuple[tuple[str, str], ...]:
        """``(implementor, interface)`` pairs this document declares."""
        return tuple(
            (info.symbol, rel.symbol)
            for info in self.symbols
            for rel in info.relationships
            if rel.is_implementation
        )

    def scoped_to(self, source_name: str) -> 'ScipDocument':
        """This document with every bare local id scoped to it."""
        if not any(_BARE_LOCAL.match(occ.symbol) for occ in self.occurrences):
            return self
        prefix = f'local {source_name}:{self.relative_path}:'
        return ScipDocument(
            relative_path=self.relative_path,
            occurrences=tuple(
                occ if not _BARE_LOCAL.match(occ.symbol) else
                ScipOccurrence(
                    symbol=prefix + occ.symbol[len('local '):],
                    range=occ.range,
                    is_definition=occ.is_definition,
                    enclosing_range=occ.enclosing_range,
                )
                for occ in self.occurrences
            ),
            symbols=self.symbols,
        )


@frozen
class ScipIndex:
    """A loaded SCIP index, with identity already settled."""

    documents: tuple[ScipDocument, ...] = ()
    source_root: Path = field(factory=Path)

    def scoped_to(self, source_name: str) -> 'ScipIndex':
        """Every bare local id scoped to its document. Idempotent."""
        return ScipIndex(
            documents=tuple(doc.scoped_to(source_name) for doc in self.documents),
            source_root=self.source_root,
        )

    def document_for(self, file: Path,
                     source_root: Path | None = None) -> 'ScipDocument | None':
        """The document for ``file``, or ``None``."""
        root = source_root or self.source_root
        try:
            wanted = Path(file).resolve().relative_to(Path(root).resolve()).as_posix()
        except (OSError, ValueError):
            wanted = Path(file).as_posix()
        for doc in self.documents:
            if doc.relative_path == wanted:
                return doc
        return None

    @classmethod
    def load(cls, path: Path, *, repo: str,
             max_staleness_days: int | None) -> 'ScipIndex':
        """Read an index from disk. Absent, stale and corrupt all raise.

        Never returns a degraded index: a silently empty one is exactly how an unwired
        pipeline reported success for months.
        """
        from docgen.scip_config import (
            ScipCorruptError,
            ScipTooStaleError,
            ScipUnavailableError,
        )

        path = Path(path)
        if not path.exists():
            raise ScipUnavailableError(repo=repo, reason='index_missing')

        age_days = (time.time() - path.stat().st_mtime) / 86400
        if max_staleness_days is not None and age_days > max_staleness_days:
            raise ScipTooStaleError(repo=repo, reason='index_too_stale',
                                    last_good_age_days=int(age_days))

        try:
            from docgen.scip import scip_pb2  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ScipCorruptError(repo=repo, reason='index_corrupt') from exc

        try:
            pb_index = scip_pb2.Index()
            pb_index.ParseFromString(path.read_bytes())
        except Exception as exc:  # protobuf raises a family of decode errors
            raise ScipCorruptError(repo=repo, reason='index_corrupt') from exc

        documents = tuple(document_from_proto(pb) for pb in pb_index.documents)
        # Identity is settled before anything downstream sees the index.
        return cls(documents=documents, source_root=path.parent).scoped_to(repo)


def document_from_proto(pb_doc) -> ScipDocument:
    """Translate a ``scip_pb2.Document``. Lazy-imports the bindings."""
    from docgen.scip import scip_pb2  # type: ignore[import-not-found]

    definition_role = scip_pb2.SymbolRole.Definition

    occurrences = tuple(
        ScipOccurrence(
            symbol=occ.symbol,
            range=tuple(occ.range),
            is_definition=bool(getattr(occ, 'symbol_roles', 0) & definition_role),
            enclosing_range=tuple(getattr(occ, 'enclosing_range', ())),
        )
        for occ in pb_doc.occurrences
    )

    symbols = []
    for info in pb_doc.symbols:
        signature = getattr(info, 'signature_documentation', None)
        symbols.append(ScipSymbolInfo(
            symbol=info.symbol,
            kind=(scip_pb2.SymbolInformation.Kind.Name(info.kind)
                  if getattr(info, 'kind', 0) else ''),
            documentation='\n'.join(info.documentation),
            # The vendored proto declares no `Property` enum and no `properties`
            # field -- SymbolInformation carries only symbol, documentation,
            # relationships, kind, display_name, signature_documentation and
            # enclosing_symbol. The module this replaces read
            # `SymbolInformation.Property.IMPLICIT`, which resolves to the *Kind*
            # value named `Property` (41) and would raise; it never did, because the
            # expression sat behind `if properties`, and `properties` is always
            # absent. So these two are structurally False here, and that is stated
            # rather than computed from a field that does not exist.
            is_implicit=False,
            is_var=False,
            display_name=getattr(info, 'display_name', ''),
            signature_text=getattr(signature, 'text', '') if signature else '',
            relationships=tuple(
                ScipRelationship(
                    symbol=rel.symbol,
                    is_implementation=bool(getattr(rel, 'is_implementation', False)),
                    is_reference=bool(getattr(rel, 'is_reference', False)),
                    is_type_definition=bool(
                        getattr(rel, 'is_type_definition', False)),
                    is_definition=bool(getattr(rel, 'is_definition', False)),
                )
                for rel in getattr(info, 'relationships', ())
            ),
        ))
    return ScipDocument(relative_path=pb_doc.relative_path,
                        occurrences=occurrences, symbols=tuple(symbols))
