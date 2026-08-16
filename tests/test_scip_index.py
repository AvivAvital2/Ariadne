"""The SCIP index model — what the rebuild makes true by construction.

Four defects lived in the module this replaces, and each one was a case of reconstructing
SCIP's meaning ad hoc at the point of use instead of modelling it once:

* ``local N`` ids are numbered **per document**, so a bare id is one row every document
  points at. Measured: 7,910 bare rows, 4,446 files aiming at ``local 5``, 16,061 in-edges
  on ``local 0``, 499,249 edges joining different sources.
* a definition's ``range`` is its **identifier**; its body is ``enclosing_range``, and
  scip-java emits none. Measured: 0 of 306,473 named symbols had a multi-line extent, so
  nothing a chain cited could be quoted.
* ``SymbolInformation.relationships`` carries ``is_implementation`` and had **zero readers**,
  so the edge-type set was only ``call`` and ``type_ref``.
* ``kind`` and ``display_name`` arrive **present but empty** from scip-python (0% and 96-99%),
  and a fallback that fired only on absence never fired.

So this model normalises at load: every id is unique, every definition carries a body
extent, relationships are first-class, and empty-vs-absent is one rule. Stage one of the
north star travels this model, so it has to be true here rather than checked later.

Synthetic fixtures only.
"""
from __future__ import annotations

import re

import pytest

from docgen.scip_index import (
    ScipDocument,
    ScipIndex,
    ScipOccurrence,
    ScipRelationship,
    ScipSymbolInfo,
)


def _doc(path='a.py', *, occurrences=(), symbols=()):
    return ScipDocument(relative_path=path, occurrences=tuple(occurrences),
                        symbols=tuple(symbols))


def _occ(symbol, line, *, definition=False, enclosing=None):
    return ScipOccurrence(
        symbol=symbol, range=(line, 4, line, 12), is_definition=definition,
        enclosing_range=enclosing or (),
    )


class TestLocalIdentity:
    """A local binding belongs to the document that emitted it."""

    def test_the_same_bare_id_in_two_documents_becomes_two_symbols(self):
        index = ScipIndex(documents=(
            _doc('a.py', occurrences=[_occ('local 1', 5, definition=True)]),
            _doc('b.py', occurrences=[_occ('local 1', 5, definition=True)]),
        )).scoped_to('src1')

        ids = {occ.symbol for doc in index.documents for occ in doc.occurrences}
        assert len(ids) == 2
        assert all(not re.fullmatch(r'local \d+', i) for i in ids)

    def test_a_scoped_local_still_announces_itself_as_local(self):
        """Every existing detector must keep firing: `LIKE 'local %'`, `startswith`."""
        index = ScipIndex(documents=(
            _doc('a.py', occurrences=[_occ('local 7', 2, definition=True)]),
        )).scoped_to('src1')

        only = index.documents[0].occurrences[0].symbol
        assert only.startswith('local ')
        assert 'a.py' in only and 'src1' in only

    def test_a_named_symbol_is_untouched_by_scoping(self):
        named = 'scip-python python src1 0.1 `m`/f().'
        index = ScipIndex(documents=(
            _doc('a.py', occurrences=[_occ(named, 1, definition=True)]),
        )).scoped_to('src1')

        assert index.documents[0].occurrences[0].symbol == named

    def test_scoping_is_idempotent(self):
        """Re-scoping an already-scoped index must not double-wrap the id."""
        once = ScipIndex(documents=(
            _doc('a.py', occurrences=[_occ('local 1', 1, definition=True)]),
        )).scoped_to('src1')
        twice = once.scoped_to('src1')

        assert ([o.symbol for d in twice.documents for o in d.occurrences]
                == [o.symbol for d in once.documents for o in d.occurrences])


class TestBodyExtent:
    """A definition's extent is its body, and it is decided per definition."""

    def test_a_supplied_enclosing_range_is_used(self):
        doc = _doc('m.py', occurrences=[
            _occ('sym', 4, definition=True, enclosing=(4, 0, 12, 0))])

        assert doc.extent_of(doc.occurrences[0]) == (5, 13)

    def test_identifier_line_is_the_start_even_when_the_body_starts_earlier(self):
        """`line_start` is the join key that resolves 95-100% of callables; it must
        stay the identifier line even when the body extent begins above it."""
        doc = _doc('m.py', occurrences=[
            _occ('sym', 6, definition=True, enclosing=(4, 0, 12, 0))])

        assert doc.extent_of(doc.occurrences[0])[0] == 7

    def test_a_definition_with_no_enclosing_range_runs_to_the_next_definition(self):
        doc = _doc('C.scala', occurrences=[
            _occ('first', 9, definition=True),
            _occ('second', 29, definition=True),
            _occ('first', 40),
        ])

        assert doc.extent_of(doc.occurrences[0]) == (10, 29)

    def test_the_last_definition_runs_to_the_last_line_the_document_mentions(self):
        """Never a 1<<30 sentinel: a stored extent is read by a body fetch."""
        doc = _doc('C.scala', occurrences=[
            _occ('only', 9, definition=True),
            _occ('ref', 40),
        ])

        assert doc.extent_of(doc.occurrences[0]) == (10, 41)

    def test_a_mixed_document_reconstructs_only_what_is_missing(self):
        """The defect I introduced once: gating per document starved 221 of 360."""
        doc = _doc('m.py', occurrences=[
            _occ('has', 4, definition=True, enclosing=(4, 0, 9, 0)),
            _occ('lacks', 19, definition=True),
            _occ('ref', 30),
        ])

        assert doc.extent_of(doc.occurrences[0]) == (5, 10)
        assert doc.extent_of(doc.occurrences[1]) == (20, 31)


class TestSymbolMetadata:
    """Present-but-empty is the same as absent — one rule, applied once."""

    def test_an_empty_display_name_falls_back_to_the_descriptor(self):
        info = ScipSymbolInfo(symbol='scip-python python src1 0.1 `m`/run().',
                              kind='', display_name='')

        assert info.effective_display_name == 'run'

    def test_a_supplied_display_name_wins(self):
        info = ScipSymbolInfo(symbol='x/run().', kind='Method', display_name='run')

        assert info.effective_display_name == 'run'

    @pytest.mark.parametrize('kind', ['', 'UnspecifiedKind'])
    def test_an_unspecified_kind_is_reported_as_unknown_not_as_a_kind(self, kind):
        """scip-python sends the enum NAME `UnspecifiedKind`, not an empty string —
        treating it as a kind is what let kind-gated logic silently no-op."""
        info = ScipSymbolInfo(symbol='x/run().', kind=kind, display_name='')

        assert info.effective_kind is None


class TestRelationships:
    """`is_implementation` is what answers *which implementation runs*."""

    def test_implementation_relationships_are_exposed(self):
        doc = _doc('m.py', symbols=[
            ScipSymbolInfo(
                symbol='x/Impl#', kind='Class',
                relationships=(ScipRelationship(symbol='x/Proto#',
                                                is_implementation=True),),
            )])

        assert doc.implementations() == (('x/Impl#', 'x/Proto#'),)

    def test_a_non_implementation_relationship_is_not_reported_as_one(self):
        doc = _doc('m.py', symbols=[
            ScipSymbolInfo(
                symbol='x/A#', kind='Class',
                relationships=(ScipRelationship(symbol='x/B#', is_reference=True),),
            )])

        assert doc.implementations() == ()


class TestLoadingContract:
    def test_a_missing_index_is_an_error_not_an_empty_index(self, tmp_path):
        """Silent success is the failure mode this rebuild exists to end."""
        from docgen.scip_config import ScipUnavailableError

        with pytest.raises(ScipUnavailableError):
            ScipIndex.load(tmp_path / 'absent.scip', repo='src1',
                           max_staleness_days=None)


class TestProtoTranslation:
    """The wire format, exercised for real — a protobuf built here, written, loaded.

    This path was verified by hand against a corpus index and left untested, which is the
    habit that produced every defect above: confirmed once, then trusted forever.
    """

    @staticmethod
    def _write(tmp_path):
        from docgen.scip import scip_pb2

        index = scip_pb2.Index()
        doc = index.documents.add()
        doc.relative_path = 'm.py'

        definition = doc.occurrences.add()
        definition.symbol = 'scip-python python src1 0.1 `m`/Impl#'
        definition.range.extend([4, 6, 4, 10])
        definition.symbol_roles = scip_pb2.SymbolRole.Definition
        definition.enclosing_range.extend([4, 0, 18, 0])

        reference = doc.occurrences.add()
        reference.symbol = 'scip-python python src1 0.1 `m`/helper().'
        reference.range.extend([9, 8, 9, 14])

        bare_local = doc.occurrences.add()
        bare_local.symbol = 'local 3'
        bare_local.range.extend([6, 4, 6, 9])
        bare_local.symbol_roles = scip_pb2.SymbolRole.Definition

        info = doc.symbols.add()
        info.symbol = 'scip-python python src1 0.1 `m`/Impl#'
        info.kind = scip_pb2.SymbolInformation.Kind.Value('Class')
        info.display_name = 'Impl'
        info.documentation.append('What Impl is for.')
        rel = info.relationships.add()
        rel.symbol = 'scip-python python src1 0.1 `m`/Proto#'
        rel.is_implementation = True

        path = tmp_path / 'index.scip'
        path.write_bytes(index.SerializeToString())
        return path

    def test_a_written_index_round_trips_through_load(self, tmp_path):
        index = ScipIndex.load(self._write(tmp_path), repo='src1',
                               max_staleness_days=None)
        doc = index.documents[0]

        assert doc.relative_path == 'm.py'
        assert len(doc.occurrences) == 3
        assert index.source_root == tmp_path

    def test_the_definition_role_and_enclosing_range_survive(self, tmp_path):
        doc = ScipIndex.load(self._write(tmp_path), repo='src1',
                             max_staleness_days=None).documents[0]
        impl = next(o for o in doc.occurrences if o.symbol.endswith('Impl#'))

        assert impl.is_definition
        assert doc.extent_of(impl) == (5, 19)

    def test_a_reference_is_not_a_definition(self, tmp_path):
        doc = ScipIndex.load(self._write(tmp_path), repo='src1',
                             max_staleness_days=None).documents[0]
        ref = next(o for o in doc.occurrences if 'helper' in o.symbol)

        assert not ref.is_definition

    def test_load_scopes_locals_so_no_bare_id_ever_enters_the_process(self, tmp_path):
        doc = ScipIndex.load(self._write(tmp_path), repo='src1',
                             max_staleness_days=None).documents[0]
        locals_ = [o.symbol for o in doc.occurrences if o.symbol.startswith('local ')]

        assert locals_ == ['local src1:m.py:3']

    def test_metadata_and_relationships_translate(self, tmp_path):
        doc = ScipIndex.load(self._write(tmp_path), repo='src1',
                             max_staleness_days=None).documents[0]
        info = doc.symbols[0]

        assert info.effective_kind == 'Class'
        assert info.effective_display_name == 'Impl'
        assert info.documentation == 'What Impl is for.'
        assert doc.implementations() == (
            ('scip-python python src1 0.1 `m`/Impl#',
             'scip-python python src1 0.1 `m`/Proto#'),
        )

    def test_a_stale_index_raises_rather_than_loading(self, tmp_path):
        import os
        import time

        from docgen.scip_config import ScipTooStaleError

        path = self._write(tmp_path)
        old = time.time() - 30 * 86400
        os.utime(path, (old, old))

        with pytest.raises(ScipTooStaleError):
            ScipIndex.load(path, repo='src1', max_staleness_days=7)


class TestLocalShapes:
    """scip-python emits two local shapes and both are document-local.

    Measured over the live store's 7,910 local rows: 6,847 ``local N`` and 1,063
    ``local N(name)``. Requiring the id to END at the digits missed every named one, so
    ``local 19(self)`` stayed one string across every file with a ``self`` at that index —
    fusing documents *and* sources, and leaving 12 cross-source edges in a rebuild that
    was otherwise clean. A verification run caught it; the unit tests had not.
    """

    @pytest.mark.parametrize('bare', ['local 0', 'local 19', 'local 1(self)',
                                      'local 12(bundle)', 'local 3(_d)'])
    def test_every_local_shape_is_scoped(self, bare):
        index = ScipIndex(documents=(
            ScipDocument('a.py', occurrences=(
                ScipOccurrence(symbol=bare, range=(1, 0, 1, 5), is_definition=True),)),
        )).scoped_to('src1')

        scoped = index.documents[0].occurrences[0].symbol
        assert scoped != bare
        assert scoped.startswith('local src1:a.py:')

    def test_the_binding_name_survives_scoping(self):
        """`local 1(self)` keeps `(self)` — it is part of what SCIP said."""
        index = ScipIndex(documents=(
            ScipDocument('a.py', occurrences=(
                ScipOccurrence(symbol='local 1(self)', range=(1, 0, 1, 5),
                               is_definition=True),)),
        )).scoped_to('src1')

        assert index.documents[0].occurrences[0].symbol == 'local src1:a.py:1(self)'

    def test_a_named_local_in_two_files_does_not_fuse(self):
        index = ScipIndex(documents=(
            ScipDocument('a.py', occurrences=(
                ScipOccurrence(symbol='local 1(self)', range=(1, 0, 1, 5),
                               is_definition=True),)),
            ScipDocument('b.py', occurrences=(
                ScipOccurrence(symbol='local 1(self)', range=(1, 0, 1, 5),
                               is_definition=True),)),
        )).scoped_to('src1')

        ids = {o.symbol for d in index.documents for o in d.occurrences}
        assert len(ids) == 2
class TestAContainerBodyExtendsPastItsMembers:
    """A class's body does not end where its first member begins.

    scip-java emits no ``enclosing_range``, so an extent is reconstructed positionally:
    a definition ends where the next definition starts. For a *container* the next
    definition is its own first member, one line down, so every class collapsed to a
    single line. Measured on the live databricks store: 47.5% of ``Class`` and 60.7% of
    ``Object`` symbols recorded a one-line body, against 0.1% of ``StaticMethod`` — and
    292 of 292 sampled single-line containers had their next definition within two lines.

    An unquotable hop is the failure this closes: stage two quotes a hop from its extent,
    so a container with a one-line body has nothing to show.

    Descendants are recognised by the moniker, which is hierarchical — a member's symbol
    string starts with its container's — so this needs no language-specific rule.
    """

    def test_a_class_extends_past_its_own_members(self):
        cls = 'scip-java java src1 0.1 `pkg`/Widget#'
        member = 'scip-java java src1 0.1 `pkg`/Widget#run().'
        sibling = 'scip-java java src1 0.1 `pkg`/Other#'
        doc = ScipDocument('Widget.java', occurrences=(
            _occ(cls, 10, definition=True),        # class Widget {
            _occ(member, 11, definition=True),     #   void run() {
            _occ(sibling, 40, definition=True),    # class Other {
        ))

        start, end = doc.extent_of(doc.definitions()[0])

        assert (start, end) == (11, 40), (
            'the class must run to the next NON-descendant definition, not to its own '
            f'first member; got {(start, end)}'
        )

    def test_a_member_still_ends_at_the_next_sibling(self):
        """The original rule is right for a leaf — only containers were wrong."""
        cls = 'scip-java java src1 0.1 `pkg`/Widget#'
        first = 'scip-java java src1 0.1 `pkg`/Widget#run().'
        second = 'scip-java java src1 0.1 `pkg`/Widget#stop().'
        doc = ScipDocument('Widget.java', occurrences=(
            _occ(cls, 10, definition=True),
            _occ(first, 11, definition=True),
            _occ(second, 25, definition=True),
        ))

        start, end = doc.extent_of(doc.definitions()[1])

        assert (start, end) == (12, 25), f'got {(start, end)}'
def test_proto_preserves_canonical_enclosing_symbol(tmp_path):
    from docgen.scip import scip_pb2

    owner = "semanticdb maven . . pkg/Owner#"
    child = "semanticdb maven . . pkg/Owner#run()."
    index = scip_pb2.Index()
    document = index.documents.add()
    document.relative_path = "Owner.scala"
    info = document.symbols.add()
    info.symbol = child
    info.kind = scip_pb2.SymbolInformation.Kind.Value("Method")
    info.enclosing_symbol = owner
    path = tmp_path / "ownership.scip"
    path.write_bytes(index.SerializeToString())

    loaded = ScipIndex.load(
        path, repo="src1", max_staleness_days=None)

    assert loaded.documents[0].symbols[0].enclosing_symbol == owner
