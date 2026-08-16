"""The menu a chain offers, and resolving what came back.

``index -> fetch document -> curate bundle -> formulate -> respond`` sent the whole bundle
to the model. Measured at production width that is 2,645 hop lines and 883 descriptions —
240,945 tokens, $1.20 a question — and 68% of it is coordinates for hops the answer never
mentions.

So the bundle is offered before it is spent. The model sees one line per thing it could
read; it names the ones it wants; only those bodies are fetched. Measured on the same
chain: 973 symbols and ~1,060 sections is about $0.13 against $1.20, and the second call
carries only what was asked for.

Two halves, deliberately labelled, because they are not equally reliable:

* **definitions** — one ``catalog`` entry per symbol the walk reached, each anchored to an
  exact ``file:line`` from the index. This is what an answer cites.
* **sections** — headings of the ``explanation`` documents covering the same files.
  Generated prose about a module, with no line-level anchor. Background, not citation.

Selection is by number, not by name: a number either resolves or it does not, so a model
cannot invent a symbol by misspelling one, and nothing needs fuzzy matching. Unknown
numbers are reported, never guessed at.

Synthetic fixtures only: source ``src1``.
"""
from __future__ import annotations

import pytest

from library import Library
from library.chain_answer import AnswerEvidence, locations_for
from library.chain_bundle import BundleHop
from library.chain_menu import (
    Fetched, Selection, fetch_selected, menu_for, render_selected, resolve_selection, _occurrence_key, complete_selection_with_body_dependencies, project_selected_evidence)
from library.source_materialization import SourceExcerpt
from library.structural_assembly import StructuralCitation

SOURCE = 'src1'


def _hop(qualified_name, *, file, line, evidence=None, title=None):
    return BundleHop(
        citation=StructuralCitation(
            qualified_name=qualified_name, file=file, line_start=line,
            source_name=SOURCE, relation='calls', hop=1, call_site_file='m.py',
            call_site_line=line + 100, line_end=line + 6),
        document_id=f'doc-{qualified_name}', title=title, evidence=evidence)


@pytest.fixture
def library(tmp_path):
    """A store with a catalog entry per symbol and one sectioned explanation document."""
    with Library(tmp_path / 'library.db') as lib:
        for qualified_name, description in (
            ('pkg.merge.Classic.writeAllChanges', 'Writes every change to the target.'),
            ('pkg.merge.InsertOnly.writeOnlyInserts', 'Appends rows matching nothing.'),
        ):
            lib.add_document(
                content_type='catalog', title=qualified_name.rsplit('.', 1)[-1],
                content=description, source_files=['pkg/merge.py'],
                doc_id=f'doc-{qualified_name}', source_name=SOURCE)
        # a realistic catalog body: a header line, then the description
        lib.add_document(
            content_type='catalog', title='pickExecutor',
            content=('scala_method pkg.merge.Planner.pickExecutor in pkg.merge [scala] '
                     'pkg/merge.py:12-30 :: def pickExecutor(clauses)\n'
                     '\nDescription: Chooses an executor from the clause set.'),
            source_files=['pkg/merge.py'],
            doc_id='doc-pkg.merge.Planner.pickExecutor', source_name=SOURCE)
        lib.add_document(
            content_type='explanation', title='Merge Module',
            content='full module prose', source_files=['pkg/merge.py'],
            doc_id='doc-module', source_name=SOURCE)
        with lib._conn_provider.acquire() as conn:
            for idx, (heading, body) in enumerate((
                ('Overview', 'The merge module coordinates two executors.'),
                ('How It Works', 'It picks an executor from the clause set.'),
            )):
                conn.execute(
                    'INSERT INTO sections (document_id, idx, heading, description, content) '
                    'VALUES (?,?,?,?,?)', ('doc-module', idx, heading, heading, body))
            conn.commit()
        yield lib


CHAIN = [
    _hop('pkg.merge.Classic.writeAllChanges', file='pkg/merge.py', line=285,
         evidence='Writes every change to the target.', title='writeAllChanges'),
    _hop('pkg.merge.InsertOnly.writeOnlyInserts', file='pkg/merge.py', line=53,
         evidence='Appends rows matching nothing.', title='writeOnlyInserts'),
    # the same definition reached again — one menu line, not two
    _hop('pkg.merge.Classic.writeAllChanges', file='pkg/merge.py', line=285,
         evidence='Writes every change to the target.', title='writeAllChanges'),
]


class TestTheMenu:
    def test_each_definition_appears_once_however_many_hops_reached_it(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)

        assert menu.text.count('writeAllChanges') == 1
        assert len(menu.symbols) == 2

    def test_a_definition_is_named_with_its_owner_so_a_choice_is_possible(self, library):
        """``writeAllChanges`` alone does not say whose. ``Classic.writeAllChanges`` does."""
        menu = menu_for(library, CHAIN, source=SOURCE)

        assert 'Classic.writeAllChanges' in menu.text

    def test_the_sections_of_the_documents_covering_the_chain_are_offered(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)

        assert 'Merge Module' in menu.text
        assert 'How It Works' in menu.text
        assert len(menu.sections) == 2

    def test_the_menu_says_which_half_carries_an_exact_anchor(self, library):
        """A catalog entry has a file:line; a module section does not. The model is told.

        The coordinate itself is not in the menu, which is a correction of what this test
        first asserted. A ``file:line`` is what an answer *cites*, and it travels with the
        body of whatever was chosen — printing it for every candidate cost 87,000 characters
        of JVM path across 973 lines, on a choice made by name and purpose.
        """
        menu = menu_for(library, CHAIN, source=SOURCE)

        assert 'pkg/merge.py' not in menu.text, 'paths are for citing, not for choosing'
        lowered = menu.text.lower()
        assert 'exact' in lowered and 'background' in lowered
        assert 'file:line' in lowered, 'the model is told the anchors exist'
    def test_the_line_shows_the_description_not_the_catalog_header(self, library):
        """A catalog body opens with kind, qualified name, package, path and signature.

        The menu line already carries the name, and the rest is not what a choice is made
        on. What helps is the sentence after ``Description:`` — measured on the live store,
        printing the first line instead gave lines like
        ``catalog.CatalogManager — scala_object org.apache.spark.sql.connector.catalog…``,
        which says nothing the name had not already said.
        """
        chain = [*CHAIN, _hop('pkg.merge.Planner.pickExecutor', file='pkg/merge.py',
                              line=12, title='pickExecutor',
                              evidence=('scala_method pkg.merge.Planner.pickExecutor in '
                                        'pkg.merge [scala] pkg/merge.py:12-30 :: def '
                                        'pickExecutor(clauses)\n\nDescription: Chooses an '
                                        'executor from the clause set.'))]

        menu = menu_for(library, chain, source=SOURCE)

        assert 'Chooses an executor from the clause set.' in menu.text
        assert 'scala_method' not in menu.text, 'the header is not a description'
    def test_a_body_with_no_description_falls_back_to_the_signature_not_the_path(
            self, library):
        """The header carries the file path, which is the one thing a menu must not spend.

        Some catalog entries have no ``Description:`` line. Falling back to the first line
        put ``[scala] sql/catalyst/src/main/scala/org/apache/…`` into the menu — the end-to-end
        harness caught it on its first run. The signature after ``::`` says what the thing is,
        in a few characters, with no path.
        """
        chain = [_hop('pkg.merge.Bare.thing', file='pkg/merge.py', line=99,
                      title='thing',
                      evidence=('scala_trait pkg.merge.Bare in pkg.merge [scala] '
                                'pkg/merge.py:99-120 :: trait Bare extends Runnable'))]

        menu = menu_for(library, chain, source=SOURCE)

        assert 'trait Bare extends Runnable' in menu.text
        assert 'pkg/merge.py:99-120' not in menu.text, 'no path in a menu line'
    def test_a_suppressed_hop_still_describes_itself_in_the_menu(self, library):
        """Rationing decides what *evidences* an answer. The menu is for choosing.

        ``plumbing`` and ``revisit`` hops carry no ``evidence`` by design, and the menu read
        that field — so measured on the live store 241 of 973 databricks lines and 93 of 266
        ariadne lines were bare names like ``2. ariadne_adapter.__init__``. The model was
        asked to choose between entries that said nothing, while the document sat in the
        store already fetched.
        """
        suppressed = BundleHop(
            citation=StructuralCitation(
                qualified_name='pkg.merge.Classic.writeAllChanges', file='pkg/merge.py',
                line_start=285, source_name=SOURCE, relation='calls', hop=1,
                call_site_file='m.py', call_site_line=8, stop_reason='plumbing',
                line_end=300),
            document_id='doc-pkg.merge.Classic.writeAllChanges', title='writeAllChanges',
            evidence=None)

        menu = menu_for(library, [suppressed], source=SOURCE)

        assert 'Writes every change to the target.' in menu.text, (
            'the document is fetched already; the menu must read it, not the ration')


class TestResolvingWhatCameBack:
    def test_numbers_resolve_to_the_things_they_labelled(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)

        chosen = resolve_selection(menu, 'I need 1 and S2 to answer this.')

        assert chosen.symbols == ['pkg.merge.Classic.writeAllChanges']
        assert chosen.sections == [('doc-module', 1)]
        assert chosen.unknown == ()

    def test_a_number_that_labels_nothing_is_reported_not_guessed(self, library):
        """A model naming line 99 of a 2-line menu is a fact to report, not to interpret."""
        menu = menu_for(library, CHAIN, source=SOURCE)

        chosen = resolve_selection(menu, 'Fetch 99 and S7.')

        assert chosen.symbols == [] and chosen.sections == []
        assert chosen.unknown == ('99', 'S7')

    def test_selecting_nothing_is_a_valid_answer(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)

        chosen = resolve_selection(menu, 'The chain alone answers it.')

        assert chosen.symbols == [] and chosen.sections == [] and chosen.unknown == ()
class TestFetchingWhatWasChosen:
    """Only the chosen bodies are read, and only those travel."""

    def test_only_the_chosen_definitions_and_sections_are_fetched(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)
        chosen = resolve_selection(menu, 'Give me 2 and S2.')

        fetched = fetch_selected(library, chosen, CHAIN)

        assert list(fetched.definitions) == ['pkg.merge.InsertOnly.writeOnlyInserts']
        assert 'Appends rows matching nothing.' in fetched.definitions[
            'pkg.merge.InsertOnly.writeOnlyInserts']
        assert fetched.sections == [
            ('Merge Module', 'How It Works', 'It picks an executor from the clause set.')]

    def test_choosing_nothing_fetches_nothing(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)

        fetched = fetch_selected(library, resolve_selection(menu, 'no thanks'), CHAIN)

        assert fetched.definitions == {} and fetched.sections == []


class TestWhatTheSecondCallCarries:
    """The chain restricted to what was chosen: coordinates, bodies, and what was left."""
    def test_the_structural_spine_survives_while_selected_detail_is_added(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)
        chosen = resolve_selection(menu, '2')
        fetched = fetch_selected(library, chosen, CHAIN)

        text = render_selected(CHAIN, chosen, fetched)

        classic = text.index('Classic.writeAllChanges')
        inserts = text.index('InsertOnly.writeOnlyInserts')
        assert classic < inserts, 'the complete structural spine retains execution order'
        assert 'pkg/merge.py:285' in text, 'an unselected hop remains compiler evidence'
        assert 'pkg/merge.py:53' in text, 'the selected hop remains compiler evidence'
        assert 'Appends rows matching nothing.' in text
        assert 'Writes every change to the target.' not in text, (
            'selection controls optional detail, not structural evidence')
        for reply in ('no selection', '99'):
            alternate = resolve_selection(menu, reply)
            alternate_text = render_selected(
                CHAIN, alternate, fetch_selected(library, alternate, CHAIN))
            assert 'pkg/merge.py:285' in alternate_text
            assert 'pkg/merge.py:53' in alternate_text
        all_chosen = resolve_selection(menu, '1, 2')
        all_text = render_selected(
            CHAIN, all_chosen, fetch_selected(library, all_chosen, CHAIN))
        assert 'further definition' not in all_text
        assert all_text.count('Writes every change to the target.') == 1
        source_hop = BundleHop(
            citation=CHAIN[0].citation,
            document_id=CHAIN[0].document_id,
            source_excerpts=(SourceExcerpt(
                source_name=SOURCE, file='pkg/merge.py', line_start=285,
                line_end=286, kind='definition', content='if matched:\n    write()',
                sha256='abc'),))
        source_text = render_selected(
            [source_hop], resolve_selection(menu, 'no selection'),
            fetch_selected(library, resolve_selection(menu, 'no selection'), [source_hop]))
        assert 'Source definition [pkg/merge.py:285-286]' in source_text
        assert 'if matched:' in source_text and 'write()' in source_text

    def test_the_hops_not_shown_are_counted_rather_than_dropped_in_silence(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)
        chosen = resolve_selection(menu, '2')
        fetched = fetch_selected(library, chosen, CHAIN)

        text = render_selected(CHAIN, chosen, fetched)

        assert '1 further definition' in text, (
            'the chain had two; one was chosen, and the other is accounted for')

    def test_a_chosen_section_is_marked_as_background(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)
        chosen = resolve_selection(menu, '2 S2')
        fetched = fetch_selected(library, chosen, CHAIN)

        text = render_selected(CHAIN, chosen, fetched)

        assert 'Merge Module' in text and 'How It Works' in text
        assert 'background' in text.lower(), 'prose with no anchor is labelled'
def test_selected_rendering_obeys_the_same_hard_budget_as_the_full_spine():
    chosen = Selection(symbols=("pkg.Alpha.run",))
    fetched = Fetched(definitions={"pkg.Alpha.run": "detail" * 100})

    text = render_selected(CHAIN, chosen, fetched, max_chars=120)

    assert len(text) <= 120
    assert "omitted to fit the context" in text
def test_compiler_selected_hops_are_labeled_mandatory_during_formulation():
    def marked(relation, name):
        base = _hop(name, file="pkg/merge.py", line=12)
        return base.__class__(citation=base.citation.__class__(
            qualified_name=base.citation.qualified_name,
            file=base.citation.file, line_start=base.citation.line_start,
            source_name=base.citation.source_name, relation=relation,
            hop=base.citation.hop, call_site_file=base.citation.call_site_file,
            call_site_line=base.citation.call_site_line,
            line_end=base.citation.line_end), document_id=base.document_id)
    text = render_selected([
        marked("localized", "pkg.DeltaFileFormatWriter.write"),
        marked("shared_reference", "pkg.StreamExecution.runStream"),
    ], Selection(), Fetched())
    assert "QUESTION-LOCALIZED — mandatory" in text
    assert "IDENTITY BRIDGE — mandatory" in text
def test_route_cards_select_and_hydrate_only_connected_routes(library):
    from library.chain_menu import route_menu_for, resolve_route_selection, render_selected_routes
    root = _hop("pkg.StreamExecution.run", file="spark.py", line=1)
    root = root.__class__(citation=root.citation.__class__(
        **{**root.citation.__dict__, "relation": "localized"}),
        document_id=root.document_id, evidence=root.evidence)
    middle = _hop("pkg.DeltaSink.commit", file="delta.py", line=2)
    middle = middle.__class__(citation=middle.citation.__class__(
        **{**middle.citation.__dict__, "parent_qualified_name": "pkg.StreamExecution.run"}),
        document_id=middle.document_id, evidence=middle.evidence)
    target = _hop("pkg.SetTransaction.apply", file="delta.py", line=3)
    target = target.__class__(citation=target.citation.__class__(
        **{**target.citation.__dict__, "parent_qualified_name": "pkg.DeltaSink.commit",
           "stop_reason": "depth"}), document_id=target.document_id,
        evidence=target.evidence)
    noise = _hop("pkg.Unrelated.run", file="noise.py", line=4)
    chain = [root, middle, target, noise]
    menu = route_menu_for(library, chain, source=SOURCE)
    route = next(label for label, names in menu.routes.items()
                 if "pkg.SetTransaction.apply" in names)
    selection = resolve_route_selection(menu, route)
    text = render_selected_routes(chain, selection, Fetched())
    assert "StreamExecution.run" in text
    assert "DeltaSink.commit" in text
    assert "SetTransaction.apply" in text
    assert "Unrelated.run" not in text
    assert len(menu.text) < 1000
def test_selected_evidence_projection_excludes_unselected_routes():
    from library.chain_answer import AnswerEvidence, locations_for
    from library.chain_menu import Selection, project_selected_evidence

    selected = _hop("pkg.Selected.run", file="selected.py", line=10)
    noise = _hop("pkg.Noise.run", file="noise.py", line=20)
    evidence = AnswerEvidence(
        bundle_citations=[selected.citation, noise.citation],
        locations=locations_for((selected, noise)), hops=(selected, noise),
        unresolved_paths=("unpositioned.md",), truncation_reason="global truncation")

    projected = project_selected_evidence(
        evidence, Selection(symbols=["pkg.Selected.run"]))

    assert [hop.citation.qualified_name for hop in projected.hops] == ["pkg.Selected.run"]
    assert [citation.qualified_name for citation in projected.bundle_citations] == ["pkg.Selected.run"]
    assert not any("noise.py" in location for location in projected.locations)
    assert projected.truncation_reason == ""
    assert projected.unresolved_paths == ()
def test_route_selection_does_not_inject_mandatory_symbols_from_other_routes(library):
    from library.chain_menu import route_menu_for, resolve_route_selection

    first = _hop("pkg.First.run", file="first.py", line=1, evidence="Runs first path.")
    first = first.__class__(citation=first.citation.__class__(
        **{**first.citation.__dict__, "relation": "localized", "stop_reason": "leaf"}),
        document_id=first.document_id, evidence=first.evidence)
    second = _hop("pkg.Second.run", file="second.py", line=2, evidence="Runs second path.")
    second = second.__class__(citation=second.citation.__class__(
        **{**second.citation.__dict__, "relation": "localized", "stop_reason": "leaf"}),
        document_id=second.document_id, evidence=second.evidence)
    menu = route_menu_for(library, [first, second], source=SOURCE)
    selected_label = next(label for label, route in menu.routes.items()
                          if "pkg.First.run" in route)

    selection = resolve_route_selection(menu, selected_label)

    assert selection.symbols == ["pkg.First.run"]
    assert "Runs first path" in menu.text
def test_route_selection_records_exact_resolved_labels(library):
    from library.chain_menu import route_menu_for, resolve_route_selection
    first = _hop("pkg.First.run", file="first.py", line=1, evidence="First route.")
    second = _hop("pkg.Second.run", file="second.py", line=2, evidence="Second route.")
    menu = route_menu_for(library, [first, second], source=SOURCE)

    selection = resolve_route_selection(menu, "R2, R999, S999")

    assert selection.route_ids == ("R2",)
    assert selection.section_ids == ()
    assert selection.unknown == ("R999", "S999")
def test_route_cards_preserve_two_occurrences_of_one_symbol_under_distinct_parents(library):
    from library.chain_menu import route_menu_for, resolve_route_selection, project_selected_evidence
    from library.chain_answer import AnswerEvidence, locations_for

    def occurrence(parent, parent_line, child_line):
        root = _hop(parent, file="parents.py", line=parent_line)
        child = _hop("pkg.Shared.commit", file="shared.py", line=50)
        child = child.__class__(citation=child.citation.__class__(
            **{**child.citation.__dict__, "parent_qualified_name": parent,
               "call_site_line": child_line, "stop_reason": "leaf"}),
            document_id=child.document_id, evidence=child.evidence)
        return root, child
    first_root, first_child = occurrence("pkg.First.run", 1, 11)
    second_root, second_child = occurrence("pkg.Second.run", 2, 22)
    hops = [first_root, first_child, second_root, second_child]
    menu = route_menu_for(library, hops, source=SOURCE)

    matching = [label for label, route in menu.routes.items()
                if route[-1] == "pkg.Shared.commit"]
    assert len(matching) == 2
    selection = resolve_route_selection(menu, matching[0])
    evidence = AnswerEvidence(
        hops=tuple(hops), bundle_citations=[hop.citation for hop in hops],
        locations=locations_for(hops))
    projected = project_selected_evidence(evidence, selection)

    selected_shared = [hop for hop in projected.hops
                       if hop.citation.qualified_name == "pkg.Shared.commit"]
    assert len(selected_shared) == 1
def test_section_only_selection_projects_no_code_evidence(library):
    from library.chain_answer import AnswerEvidence, locations_for
    from library.chain_menu import Selection, project_selected_evidence
    hop = _hop("pkg.Code.run", file="code.py", line=4)
    evidence = AnswerEvidence(
        hops=(hop,), bundle_citations=[hop.citation], locations=locations_for([hop]))

    projected = project_selected_evidence(
        evidence, Selection(sections=[("doc", 0)], section_ids=("S1",)))

    assert projected.hops == ()
    assert projected.bundle_citations == []
    assert projected.locations == frozenset()
def test_section_candidates_are_scoped_to_the_requested_source(library):
    hop = _hop("pkg.Code.run", file="pkg/merge.py", line=4)
    library.add_document(
        content_type="explanation", title="Foreign Module", content="foreign",
        source_files=["pkg/merge.py"], doc_id="foreign-doc", source_name="foreign")
    with library._conn_provider.acquire() as conn:
        conn.execute(
            "INSERT INTO sections (document_id, idx, heading, description, content) "
            "VALUES (?,?,?,?,?)", ("foreign-doc", 0, "Foreign Heading", "", "foreign"))
        conn.commit()

    menu = menu_for(library, [hop], source=SOURCE)

    assert "Foreign Module" not in menu.text
    assert "Foreign Heading" not in menu.text
def test_source_hydration_curates_only_selected_occurrences(library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import hydrate_selected_hops, route_menu_for, resolve_route_selection
    first = _hop("pkg.First.run", file="first.py", line=1)
    second = _hop("pkg.Second.run", file="second.py", line=2)
    menu = route_menu_for(library, [first, second], source=SOURCE)
    selected = resolve_route_selection(menu, "R1")
    seen = []
    def fake_curate(lib, citations, **kwargs):
        seen.extend(citations)
        return ChainBundle(hops=[first], source_gaps=())
    monkeypatch.setattr("library.chain_bundle.curate_bundle", fake_curate)

    hydrated, gaps = hydrate_selected_hops(
        library, [first, second], selected, source=SOURCE, source_root="/source")

    assert [citation.qualified_name for citation in seen] == ["pkg.First.run"]
    assert hydrated == (first,)
    assert gaps == ()
def test_parent_occurrence_is_resolved_by_the_recorded_call_site(library):
    from library.chain_menu import route_menu_for
    first = _hop("pkg.Parent.run", file="parent.py", line=10)
    first = first.__class__(citation=first.citation.__class__(
        **{**first.citation.__dict__, "line_end": 30}),
        document_id=first.document_id, evidence=first.evidence)
    second = _hop("pkg.Parent.run", file="parent.py", line=100)
    second = second.__class__(citation=second.citation.__class__(
        **{**second.citation.__dict__, "line_end": 130}),
        document_id=second.document_id, evidence=second.evidence)
    child = _hop("pkg.Child.commit", file="child.py", line=200)
    child = child.__class__(citation=child.citation.__class__(
        **{**child.citation.__dict__, "parent_qualified_name": "pkg.Parent.run",
           "call_site_file": "parent.py", "call_site_line": 20,
           "stop_reason": "leaf"}), document_id=child.document_id,
        evidence=child.evidence)

    menu = route_menu_for(library, [first, second, child], source=SOURCE)
    route = next(label for label, names in menu.routes.items()
                 if names[-1] == "pkg.Child.commit")

    assert menu.route_occurrences[route][0][2] == 10
def test_route_path_is_not_silently_cut_at_sixteen_occurrences(library):
    from library.chain_menu import route_menu_for
    hops = []
    previous = ""
    for number in range(20):
        hop = _hop(f"pkg.Step{number}.run", file="flow.py", line=number + 1)
        changes = {**hop.citation.__dict__, "parent_qualified_name": previous,
                   "stop_reason": "leaf" if number == 19 else "descended"}
        hop = hop.__class__(citation=hop.citation.__class__(**changes),
                            document_id=hop.document_id, evidence=hop.evidence)
        hops.append(hop)
        previous = hop.citation.qualified_name

    menu = route_menu_for(library, hops, source=SOURCE)
    longest = max(menu.routes.values(), key=len)

    assert len(longest) == 20
def test_route_card_names_sections_covering_that_route(library):
    from library.chain_menu import route_menu_for
    route = [_hop("pkg.Merge.run", file="pkg/merge.py", line=4)]

    menu = route_menu_for(library, route, source=SOURCE)

    route_id = next(iter(menu.routes))
    assert menu.route_sections[route_id]
    assert all(label.startswith("S") for label in menu.route_sections[route_id])
    assert "sections " in menu.text
def test_identical_semantic_routes_share_one_card_and_keep_all_occurrences(library):
    from library.chain_menu import route_menu_for, resolve_route_selection

    first_root = _hop("pkg.Root.run", file="root.py", line=1)
    first_child = _hop("pkg.Child.apply", file="child.py", line=10)
    first_child = first_child.__class__(citation=first_child.citation.__class__(
        **{**first_child.citation.__dict__,
           "parent_qualified_name": "pkg.Root.run", "call_site_file": "root.py",
           "call_site_line": 3, "stop_reason": "leaf"}),
        document_id=first_child.document_id, evidence=first_child.evidence)
    second_root = _hop("pkg.Root.run", file="root.py", line=100)
    second_child = _hop("pkg.Child.apply", file="child.py", line=200)
    second_child = second_child.__class__(citation=second_child.citation.__class__(
        **{**second_child.citation.__dict__,
           "parent_qualified_name": "pkg.Root.run", "call_site_file": "root.py",
           "call_site_line": 103, "stop_reason": "leaf"}),
        document_id=second_child.document_id, evidence=second_child.evidence)

    menu = route_menu_for(
        library, [first_root, first_child, second_root, second_child], source=SOURCE)

    matching = [label for label, names in menu.routes.items()
                if names == ("pkg.Root.run", "pkg.Child.apply")]
    assert len(matching) == 1
    selection = resolve_route_selection(menu, matching[0])
    assert len(selection.occurrence_keys) == 4
def test_selection_coverage_keeps_distinct_question_relevant_route_roots(library):
    from library.chain_menu import (
        complete_route_selection, route_menu_for, resolve_route_selection)

    rewrite = _hop(
        "spark.analysis.RewriteMergeIntoTable.apply", file="spark/rewrite.py", line=4)
    rewrite = rewrite.__class__(citation=rewrite.citation.__class__(
        **{**rewrite.citation.__dict__, "stop_reason": "leaf"}),
        document_id=rewrite.document_id, evidence=rewrite.evidence)
    preprocess = _hop(
        "delta.analysis.PreprocessTableMerge.apply", file="delta/preprocess.py", line=8)
    preprocess = preprocess.__class__(citation=preprocess.citation.__class__(
        **{**preprocess.citation.__dict__, "stop_reason": "leaf"}),
        document_id=preprocess.document_id, evidence=preprocess.evidence)
    unrelated = _hop("util.Clock.tick", file="util/clock.py", line=12)
    unrelated = unrelated.__class__(citation=unrelated.citation.__class__(
        **{**unrelated.citation.__dict__, "stop_reason": "leaf"}),
        document_id=unrelated.document_id, evidence=unrelated.evidence)
    menu = route_menu_for(library, [rewrite, preprocess, unrelated], source=SOURCE)
    rewrite_id = next(label for label, route in menu.routes.items()
                      if "RewriteMergeIntoTable" in route[0])
    initial = resolve_route_selection(menu, rewrite_id)

    completed = complete_route_selection(
        menu, initial,
        "Why does a Delta merge table use analysis instead of Spark's normal rewrite?")

    assert any("PreprocessTableMerge" in symbol for symbol in completed.symbols)
    assert not any("Clock" in symbol for symbol in completed.symbols)
    assert rewrite_id in completed.route_ids
def test_selection_coverage_is_bounded_when_many_roots_match(library):
    from library.chain_menu import (
        complete_route_selection, route_menu_for, resolve_route_selection)
    hops = []
    for number in range(20):
        hop = _hop(f"pkg.MergeTableBranch{number}.apply", file=f"f{number}.py", line=1)
        hops.append(hop.__class__(citation=hop.citation.__class__(
            **{**hop.citation.__dict__, "stop_reason": "leaf"}),
            document_id=hop.document_id, evidence=hop.evidence))
    menu = route_menu_for(library, hops, source=SOURCE)
    initial = resolve_route_selection(menu, "R1")

    completed = complete_route_selection(
        menu, initial, "How does merge table processing work?", max_branches=4)

    assert len(completed.route_ids) <= 5
def test_mandatory_route_completion_cannot_be_overruled_by_selector(library):
    from library.chain_menu import (
        retain_mandatory_routes, route_menu_for, resolve_route_selection)

    optional = _hop("pkg.Optional.run", file="optional.py", line=1)
    optional = optional.__class__(citation=optional.citation.__class__(
        **{**optional.citation.__dict__, "stop_reason": "leaf"}),
        document_id=optional.document_id, evidence=optional.evidence)
    mandatory = _hop("pkg.Stream.run", file="stream.py", line=10)
    mandatory = mandatory.__class__(citation=mandatory.citation.__class__(
        **{**mandatory.citation.__dict__, "relation": "shared_reference",
           "stop_reason": "reference_bridge"}),
        document_id=mandatory.document_id, evidence=mandatory.evidence)
    menu = route_menu_for(library, [optional, mandatory], source=SOURCE)
    optional_id = next(label for label, route in menu.routes.items()
                       if route == ("pkg.Optional.run",))
    mandatory_id = next(label for label, route in menu.routes.items()
                        if route == ("pkg.Stream.run",))

    completed = retain_mandatory_routes(
        menu, resolve_route_selection(menu, optional_id))

    assert completed.route_ids == (optional_id, mandatory_id)
    assert "pkg.Stream.run" in completed.symbols
    assert menu.route_occurrences[mandatory_id][0] in completed.occurrence_keys
def test_mandatory_route_completion_is_a_bounded_minimal_cover():
    from library.chain_menu import RouteMenu, Selection, retain_mandatory_routes

    menu = RouteMenu(
        routes={
            "R1": ("pkg.Selected",),
            "R2": ("pkg.Shared", "pkg.A"),
            "R3": ("pkg.Shared", "pkg.B"),
            "R4": ("pkg.Other",),
        },
        mandatory_symbols=("pkg.Shared", "pkg.Other"),
        route_occurrences={
            "R1": (("selected",),),
            "R2": (("shared-a",),),
            "R3": (("shared-b",),),
            "R4": (("other",),),
        },
    )
    initial = Selection(symbols=["pkg.Selected"], route_ids=("R1",),
                        occurrence_keys=(("selected",),))

    completed = retain_mandatory_routes(menu, initial, max_routes=2)

    assert completed.route_ids == ("R1", "R4", "R2")
    assert "R3" not in completed.route_ids
    assert set(menu.mandatory_symbols).issubset(completed.symbols)
def test_embedding_scores_choose_route_local_section_when_titles_do_not_overlap(library):
    import numpy as np
    from library.chain_menu import (
        RouteMenu, Selection, route_section_embedding_scores, select_route_sections)

    with library._conn_provider.acquire() as conn:
        conn.execute("UPDATE sections SET embedding = ? WHERE document_id = ? AND idx = 0",
                     (np.asarray([1.0, 0.0], dtype=np.float32).tobytes(), "doc-module"))
        conn.execute("UPDATE sections SET embedding = ? WHERE document_id = ? AND idx = 1",
                     (np.asarray([0.0, 1.0], dtype=np.float32).tobytes(), "doc-module"))
        conn.commit()
    menu = RouteMenu(
        routes={"R1": ("service.persist",)},
        sections={"S1": ("doc-module", 0), "S2": ("doc-module", 1)},
        route_sections={"R1": ("S1", "S2")},
        section_titles={"S1": "Overview", "S2": "Internals"})
    selection = Selection(route_ids=("R1",))

    scores = route_section_embedding_scores(
        library, menu, selection, np.asarray([0.0, 1.0], dtype=np.float32))
    selected = select_route_sections(
        menu, selection, "How is the request persisted?", section_scores=scores,
        max_sections=1)

    assert scores["S2"] == pytest.approx(1.0)
    assert selected.section_ids == ("S2",)
def test_route_menu_resolves_spool_prefixed_document_source(library):
    from library.chain_menu import route_menu_for
    library.add_document(
        content_type="explanation", title="Spool Merge Guide",
        content="merge guide", source_files=["spool/merge.py"],
        doc_id="spool-guide", source_name="spool:src1", _allow_reserved_source = True)
    with library._conn_provider.acquire() as conn:
        conn.execute(
            "INSERT INTO sections (document_id, idx, heading, description, content) "
            "VALUES (?,?,?,?,?)",
            ("spool-guide", 0, "Join classification", "", "Explains join ordering."))
        conn.commit()
    terminal = _hop("pkg.Merge.run", file="spool/merge.py", line=1)
    terminal = terminal.__class__(citation=terminal.citation.__class__(
        **{**terminal.citation.__dict__, "stop_reason": "leaf"}),
        document_id=terminal.document_id, evidence=terminal.evidence)

    menu = route_menu_for(library, [terminal], source=SOURCE)

    assert menu.sections
    assert "Spool Merge Guide Join classification" in menu.section_titles.values()
def test_all_route_selection_reports_the_route_ids_it_expands():
    from library.chain_menu import RouteMenu, all_route_selection
    menu = RouteMenu(routes={"R1": ("pkg.A",), "R2": ("pkg.B",)})

    selection = all_route_selection(menu)

    assert selection.route_ids == ("R1", "R2")
def test_graph_report_attaches_seed_origins_to_components():
    from library.chain_menu import evidence_graph_for, evidence_graph_report
    root = _hop("pkg.Entry.run", file="entry.py", line=1)
    from dataclasses import replace
    child = _hop("pkg.Store.commit", file="store.py", line=10)
    child = replace(child, citation=replace(
        child.citation, parent_qualified_name="pkg.Entry.run",
        call_site_file="entry.py", call_site_line=2))
    graph = evidence_graph_for([root, child])

    report = evidence_graph_report(graph, seed_provenance=(
        {"symbol": "pkg.Entry.run", "origins": ["clew"]},
        {"symbol": "pkg.Store.commit", "origins": ["catalog"]},))

    assert report["components"][0]["seed_origins"] == {
        "catalog": ["pkg.Store.commit"], "clew": ["pkg.Entry.run"]}
def test_source_hydration_enables_definition_bodies_only_after_selection(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops
    hop = _hop("pkg.Selected.run", file="selected.py", line=2)
    captured = {}

    def fake_curate(lib, citations, **kwargs):
        captured.update(kwargs)
        return ChainBundle(hops=[hop])

    monkeypatch.setattr("library.chain_bundle.curate_bundle", fake_curate)

    hydrate_selected_hops(
        library, [hop], Selection(symbols=["pkg.Selected.run"]),
        source=SOURCE, source_root="/source")

    assert captured["materialize_definition_bodies"] is True
def test_source_hydration_passes_semantic_body_scope_to_curation(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops

    hop = _hop("pkg.Selected.run", file="selected.py", line=2)
    captured = {}

    def fake_curate(lib, citations, **kwargs):
        captured.update(kwargs)
        return ChainBundle(hops=[hop])

    monkeypatch.setattr("library.chain_bundle.curate_bundle", fake_curate)

    hydrate_selected_hops(
        library, [hop], Selection(symbols=["pkg.Selected.run"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Selected.run",),
        definition_body_query="C1: persist the result")

    assert captured["definition_body_query"] == "C1: persist the result"
def test_definition_body_menu_is_compact_and_occurrence_scoped(library):
    from library.chain_menu import (
        Selection, _occurrence_key, definition_body_menu)
    first = _hop("pkg.Entry.run", file="entry.py", line=2)
    second = _hop("pkg.Work.apply", file="work.py", line=20)
    noise = _hop("pkg.Noise.run", file="noise.py", line=40)
    selection = Selection(
        symbols=["pkg.Entry.run", "pkg.Work.apply"],
        occurrence_keys=(_occurrence_key(first), _occurrence_key(second)))

    menu = definition_body_menu([first, second, noise], selection)

    assert menu.symbols == {"B1": "pkg.Entry.run", "B2": "pkg.Work.apply"}
    assert "Entry.run" in menu.text
    assert "Work.apply" in menu.text
    assert "Noise.run" not in menu.text
    assert "entry.py" not in menu.text
    assert "7-line definition" in menu.text
def test_definition_body_selection_requires_intermediate_route_bodies(library):
    from library.chain_menu import (
        DefinitionBodySelection, Selection, _occurrence_key,
        complete_definition_body_selection, definition_body_menu)

    root = _hop("pkg.Flow.start", file="flow.py", line=1)
    middle = _hop("pkg.Flow.transform", file="flow.py", line=10)
    terminal = _hop("pkg.Flow.finish", file="flow.py", line=20)
    middle = middle.__class__(
        citation=middle.citation.__class__(**{
            **middle.citation.__dict__,
            "parent_qualified_name": "pkg.Flow.start",
            "call_site_file": "flow.py", "call_site_line": 3}),
        document_id=middle.document_id, evidence=middle.evidence)
    terminal = terminal.__class__(
        citation=terminal.citation.__class__(**{
            **terminal.citation.__dict__,
            "parent_qualified_name": "pkg.Flow.transform",
            "call_site_file": "flow.py", "call_site_line": 12}),
        document_id=terminal.document_id, evidence=terminal.evidence)
    hops = [root, middle, terminal]
    route = Selection(
        symbols=[hop.citation.qualified_name for hop in hops],
        occurrence_keys=tuple(_occurrence_key(hop) for hop in hops))

    menu = definition_body_menu(hops, route)
    completed = complete_definition_body_selection(
        menu, DefinitionBodySelection(symbols=("pkg.Flow.start",)))

    assert menu.required_symbols == ("pkg.Flow.transform",)
    assert completed.symbols == ("pkg.Flow.start", "pkg.Flow.transform")


def test_definition_body_selection_resolves_ids_and_has_safe_all_fallback(library):
    from library.chain_menu import (
        DefinitionBodyMenu, all_definition_body_selection,
        resolve_definition_body_selection)
    menu = DefinitionBodyMenu(
        text="cards", symbols={"B1": "pkg.Entry.run", "B2": "pkg.Work.apply"})

    selected = resolve_definition_body_selection(menu, "B2 B1 B2 B99")
    fallback = all_definition_body_selection(menu)

    assert selected.symbols == ("pkg.Work.apply", "pkg.Entry.run")
    assert selected.unknown == ("B99",)
    assert fallback.symbols == ("pkg.Entry.run", "pkg.Work.apply")
def test_definition_body_selection_is_deterministic_after_route_selection():
    from library.chain_menu import (
        DefinitionBodyMenu, definition_body_selection_requires_llm)
    narrow = DefinitionBodyMenu(symbols={f"B{i}": f"pkg.S{i}" for i in range(1, 9)})
    broad = DefinitionBodyMenu(symbols={f"B{i}": f"pkg.S{i}" for i in range(1, 80)})

    assert definition_body_selection_requires_llm(narrow, "llm") is False
    assert definition_body_selection_requires_llm(broad, "llm") is True
def test_body_hydration_adds_direct_compiler_dependencies(library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import hydrate_selected_hops, route_menu_for, resolve_route_selection
    from library.structural_assembly import StructuralCitation
    root = _hop("pkg.First.run", file="first.py", line=1)
    menu = route_menu_for(library, [root], source=SOURCE)
    selected = resolve_route_selection(menu, "R1")
    dependency = StructuralCitation(
        qualified_name="pkg.Dependency.save", file="dependency.py", line_start=4,
        line_end=8, source_name=SOURCE, relation="calls", hop=1,
        call_site_file="first.py", call_site_line=3,
        stop_reason="selected_route_fanout",
        parent_qualified_name="pkg.First.run")
    seen = []

    def fake_fanout(conn, roots, **kwargs):
        assert roots == ("pkg.First.run",)
        assert kwargs["depth"] == 1
        assert "max_total" not in kwargs
        assert kwargs["recursive_per_root"] == 0
        assert kwargs["max_recursive_total"] == 0
        return (dependency,)

    def fake_curate(lib, citations, **kwargs):
        seen.extend(citations)
        return ChainBundle(hops=[root], source_gaps=())

    monkeypatch.setattr("library.structural_assembly.qualified_call_fanout", fake_fanout)
    monkeypatch.setattr("library.chain_bundle.curate_bundle", fake_curate)

    hydrate_selected_hops(
        library, [root], selected, source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.First.run",))

    assert [citation.qualified_name for citation in seen] == [
        "pkg.First.run", "pkg.Dependency.save"]
def test_hydration_preserves_unspecified_body_policy(library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import hydrate_selected_hops, route_menu_for, resolve_route_selection
    root = _hop("pkg.First.run", file="first.py", line=1)
    menu = route_menu_for(library, [root], source=SOURCE)
    selected = resolve_route_selection(menu, "R1")
    seen = []

    def fake_curate(lib, citations, **kwargs):
        seen.append(kwargs["definition_body_symbols"])
        return ChainBundle(hops=[root], source_gaps=())

    monkeypatch.setattr("library.chain_bundle.curate_bundle", fake_curate)
    hydrate_selected_hops(
        library, [root], selected, source=SOURCE, source_root="/source")

    assert seen == [None]
def test_route_scope_uses_compact_definition_summaries_before_pruning():
    from library.chain_menu import RouteMenu, scope_route_menu
    menu = RouteMenu(
        routes={
            "R1": ("pkg.Command.run", "pkg.Output.finish"),
            "R2": ("pkg.Rows.scan",),
        },
        route_summaries={
            "R1": "Emits resulting rows after classification.",
            "R2": "Inspects a schema.",
        },
    )

    scoped = scope_route_menu(
        menu, "Which path emits resulting rows?", max_families=1)

    assert tuple(scoped.routes) == ("R1",)
    assert scoped.route_summaries == {
        "R1": "Emits resulting rows after classification."}
    assert "Emits resulting rows" in scoped.text
def test_component_expansion_preserves_route_summaries_and_section_titles():
    from library.chain_menu import RouteMenu, routes_for_modules
    menu = RouteMenu(
        routes={"R1": ("pkg.Command.run", "pkg.Output.finish")},
        sections={"S1": ("doc", 0)},
        route_sections={"R1": ("S1",)},
        section_titles={"S1": "Writing emitted rows"},
        route_summaries={"R1": "Emits resulting rows after classification."},
    )

    expanded = routes_for_modules(menu, ("R1",))

    assert expanded.route_summaries == menu.route_summaries
    assert "Emits resulting rows" in expanded.text
    assert "Writing emitted rows" in expanded.text
def test_obligation_route_selection_keeps_a_small_complete_set():
    from library.chain_menu import RouteMenu, resolve_obligation_route_selection
    menu = RouteMenu(routes={
        f"R{index}": (f"pkg.Flow.step{index}",) for index in range(1, 9)
    })

    selected = resolve_obligation_route_selection(
        menu, "C1: R1 R2 R3 R4\nC2: R5 R6 R7", max_per_obligation=2)

    assert selected.route_ids == ("R1", "R2", "R5", "R6")
    assert selected.symbols == [
        "pkg.Flow.step1", "pkg.Flow.step2", "pkg.Flow.step5", "pkg.Flow.step6"]


def test_obligation_route_selection_falls_back_to_plain_route_reply():
    from library.chain_menu import RouteMenu, resolve_obligation_route_selection
    menu = RouteMenu(routes={"R1": ("pkg.One",), "R2": ("pkg.Two",)})

    selected = resolve_obligation_route_selection(menu, "R2")

    assert selected.route_ids == ("R2",)
def test_route_menu_keeps_semantic_symbols_on_each_route_card(library):
    from library.chain_menu import route_menu_for

    root = _hop(
        "org.example.pipeline.VeryLongOwner.execute",
        file="pipeline.py", line=1)
    first = _hop(
        "org.example.pipeline.FirstTerminal.persist",
        file="pipeline.py", line=20)
    second = _hop(
        "org.example.pipeline.SecondTerminal.emit",
        file="pipeline.py", line=40)
    first = first.__class__(
        citation=first.citation.__class__(**{
            **first.citation.__dict__,
            "parent_qualified_name":
                "org.example.pipeline.VeryLongOwner.execute",
            "call_site_file": "pipeline.py", "call_site_line": 3}),
        document_id=first.document_id, evidence=first.evidence)
    second = second.__class__(
        citation=second.citation.__class__(**{
            **second.citation.__dict__,
            "parent_qualified_name":
                "org.example.pipeline.VeryLongOwner.execute",
            "call_site_file": "pipeline.py", "call_site_line": 4}),
        document_id=second.document_id, evidence=second.evidence)

    menu = route_menu_for(library, [root, first, second], source=SOURCE)

    assert len(menu.routes) == 2
    assert menu.text.count("org.example.pipeline.VeryLongOwner.execute") == 2
    assert "R1. org.example.pipeline.VeryLongOwner.execute -> org.example.pipeline.FirstTerminal.persist" in menu.text
    assert "R2. org.example.pipeline.VeryLongOwner.execute -> org.example.pipeline.SecondTerminal.emit" in menu.text
def test_component_menu_uses_distinct_graph_ids_and_obligation_mapping():
    from library.chain_menu import (
        EvidenceGraph, EvidenceGraphNode, RouteMenu, component_menu_for,
        resolve_component_selection)
    graph = EvidenceGraph(nodes=(
        EvidenceGraphNode("N1", "pkg.Delta.run", (), "delta.py", 1),
        EvidenceGraphNode("N2", "pkg.Plain.run", (), "plain.py", 2),
    ), edges=(), roots=("N1", "N2"), terminals=("N1", "N2"))
    menu = RouteMenu(routes={
        "R1": ("pkg.Delta.run",), "R2": ("pkg.Plain.run",)})

    components = component_menu_for(graph, menu)
    selected = resolve_component_selection(components, "C1: G2")

    assert tuple(components.components) == ("G1", "G2")
    assert selected == ("R2",)
    assert "G1." in components.text and "G2." in components.text
def test_component_menu_maps_routes_by_exact_occurrence_not_shared_symbol():
    from library.chain_menu import (
        EvidenceGraph, EvidenceGraphNode, RouteMenu, component_menu_for,
        resolve_component_selection)
    first_occurrence = (
        "pkg.Shared.run", "first.py", 1, 5, "", "first.py", 1,
        "localized", 1, "leaf")
    second_occurrence = (
        "pkg.Shared.run", "second.py", 10, 15, "", "second.py", 10,
        "localized", 1, "leaf")
    graph = EvidenceGraph(nodes=(
        EvidenceGraphNode(
            "N1", "pkg.Shared.run", first_occurrence, "first.py", 1),
        EvidenceGraphNode(
            "N2", "pkg.Shared.run", second_occurrence, "second.py", 10),
    ), edges=(), roots=("N1", "N2"), terminals=("N1", "N2"))
    menu = RouteMenu(
        routes={
            "R1": ("pkg.Shared.run",),
            "R2": ("pkg.Shared.run",),
        },
        route_occurrences={
            "R1": (first_occurrence,),
            "R2": (second_occurrence,),
        })

    components = component_menu_for(graph, menu)

    assert resolve_component_selection(components, "G1") == ("R1",)
    assert resolve_component_selection(components, "G2") == ("R2",)
def test_graph_connector_closure_starts_from_selected_occurrences_only():
    from library.chain_menu import (
        EvidenceGraph, EvidenceGraphEdge, EvidenceGraphNode,
        selection_for_graph_symbols)
    chosen_parent = (
        "pkg.ChosenParent", "chosen.py", 1, 5, "", "chosen.py", 1,
        "localized", 1, "leaf")
    chosen_target = (
        "pkg.Target.run", "chosen.py", 10, 15, "pkg.ChosenParent",
        "chosen.py", 3, "calls", 2, "leaf")
    noise_parent = (
        "pkg.NoiseParent", "noise.py", 1, 5, "", "noise.py", 1,
        "localized", 1, "leaf")
    noise_target = (
        "pkg.Target.run", "noise.py", 10, 15, "pkg.NoiseParent",
        "noise.py", 3, "calls", 2, "leaf")
    graph = EvidenceGraph(
        nodes=(
            EvidenceGraphNode("N1", "pkg.ChosenParent", chosen_parent,
                              "chosen.py", 1),
            EvidenceGraphNode("N2", "pkg.Target.run", chosen_target,
                              "chosen.py", 10),
            EvidenceGraphNode("N3", "pkg.NoiseParent", noise_parent,
                              "noise.py", 1),
            EvidenceGraphNode("N4", "pkg.Target.run", noise_target,
                              "noise.py", 10),
        ),
        edges=(
            EvidenceGraphEdge("N1", "N2", "calls", "chosen.py", 3),
            EvidenceGraphEdge("N3", "N4", "calls", "noise.py", 3),
        ), roots=("N1", "N3"), terminals=("N2", "N4"))

    selection = selection_for_graph_symbols(
        graph, ("pkg.Target.run",), occurrence_keys=(chosen_target,))

    assert selection.symbols == ["pkg.ChosenParent", "pkg.Target.run"]
    assert selection.occurrence_keys == (chosen_parent, chosen_target)
def test_component_menu_keeps_semantic_boundary_names_on_each_card():
    from library.chain_menu import (
        EvidenceGraph, EvidenceGraphNode, RouteMenu, component_menu_for)
    first_occurrence = (
        "pkg.Shared.run", "first.py", 1, 5, "", "first.py", 1,
        "localized", 1, "leaf")
    second_occurrence = (
        "pkg.Shared.run", "second.py", 10, 15, "", "second.py", 10,
        "localized", 1, "leaf")
    graph = EvidenceGraph(nodes=(
        EvidenceGraphNode("N1", "pkg.Shared.run", first_occurrence,
                          "first.py", 1),
        EvidenceGraphNode("N2", "pkg.Shared.run", second_occurrence,
                          "second.py", 10),
    ), edges=(), roots=("N1", "N2"), terminals=("N1", "N2"))
    menu = RouteMenu(
        routes={"R1": ("pkg.Shared.run",), "R2": ("pkg.Shared.run",)},
        route_occurrences={
            "R1": (first_occurrence,), "R2": (second_occurrence,)})

    components = component_menu_for(graph, menu)
    assert components.text.count("Shared") == 4
    assert 'G1. entry Shared.run; terminal Shared.run' in components.text
    assert 'G1. entry Shared.run; terminal Shared.run' in components.text
    assert 'G2. entry Shared.run; terminal Shared.run' in components.text
def test_graph_connector_closure_adds_one_connected_copy_for_a_localized_seed():
    from library.chain_menu import (
        EvidenceGraph, EvidenceGraphEdge, EvidenceGraphNode,
        selection_for_graph_symbols)
    localized = (
        "pkg.Target.run", "target.py", 10, 15, "", "target.py", 10,
        "localized", 0, "question_symbol")
    chosen_parent = (
        "pkg.ChosenParent", "chosen.py", 1, 5, "", "chosen.py", 1,
        "localized", 1, "leaf")
    chosen_target = (
        "pkg.Target.run", "target.py", 10, 15, "pkg.ChosenParent",
        "chosen.py", 3, "calls", 2, "leaf")
    noise_parent = (
        "pkg.NoiseParent", "noise.py", 1, 5, "", "noise.py", 1,
        "localized", 1, "leaf")
    noise_target = (
        "pkg.Target.run", "target.py", 10, 15, "pkg.NoiseParent",
        "noise.py", 30, "calls", 2, "leaf")
    graph = EvidenceGraph(
        nodes=(
            EvidenceGraphNode("N0", "pkg.Target.run", localized,
                              "target.py", 10),
            EvidenceGraphNode("N1", "pkg.ChosenParent", chosen_parent,
                              "chosen.py", 1),
            EvidenceGraphNode("N2", "pkg.Target.run", chosen_target,
                              "target.py", 10),
            EvidenceGraphNode("N3", "pkg.NoiseParent", noise_parent,
                              "noise.py", 1),
            EvidenceGraphNode("N4", "pkg.Target.run", noise_target,
                              "target.py", 10),
        ),
        edges=(
            EvidenceGraphEdge("N1", "N2", "calls", "chosen.py", 3),
            EvidenceGraphEdge("N3", "N4", "calls", "noise.py", 30),
        ), roots=("N0", "N1", "N3"), terminals=("N0", "N2", "N4"))

    selection = selection_for_graph_symbols(
        graph, ("pkg.Target.run",), occurrence_keys=(localized,))

    assert selection.symbols == ["pkg.Target.run", "pkg.ChosenParent"]
    assert selection.occurrence_keys == (
        localized, chosen_parent, chosen_target)
def test_comparison_completion_keeps_a_bounded_sibling_compiler_branch():
    from library.chain_menu import (
        RouteMenu, Selection, complete_route_selection)

    first_occurrence = ("first",)
    second_occurrence = ("second",)
    menu = RouteMenu(
        routes={
            "R1": ("pkg.Dispatch.apply", "pkg.Dispatch.firstMode"),
            "R2": (
                "pkg.Dispatch.apply", "pkg.Dispatch.secondMode",
                "pkg.Engine.execute"),
        },
        route_occurrences={
            "R1": (first_occurrence,),
            "R2": (second_occurrence,),
        })
    initial = Selection(
        symbols=list(menu.routes["R1"]), route_ids=("R1",),
        occurrence_keys=(first_occurrence,))

    compared = complete_route_selection(
        menu, initial, "Compare the first mode versus the second mode.")
    ordinary = complete_route_selection(
        menu, initial, "How does the first mode execute?")

    assert compared.route_ids == ("R1", "R2")
    assert "pkg.Dispatch.secondMode" in compared.symbols
    assert second_occurrence in compared.occurrence_keys
    assert ordinary.route_ids == ("R1",)
def test_hydrated_body_dependency_completion_preserves_direct_compiler_edges():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)

    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    dependency = _hop("pkg.Output.emit", file="output.py", line=20)
    dependency = dependency.__class__(
        citation=dependency.citation.__class__(**{
            **dependency.citation.__dict__,
            "parent_qualified_name": "pkg.Flow.run",
            "call_site_file": "flow.py",
            "call_site_line": 8,
            "relation": "calls",
            "stop_reason": "selected_route_fanout"}),
        document_id=dependency.document_id,
        evidence=dependency.evidence)
    unrelated = _hop("pkg.Noise.emit", file="noise.py", line=30)
    unrelated = unrelated.__class__(
        citation=unrelated.citation.__class__(**{
            **unrelated.citation.__dict__,
            "parent_qualified_name": "pkg.Other.run",
            "call_site_file": "other.py",
            "call_site_line": 5,
            "relation": "calls",
            "stop_reason": "selected_route_fanout"}),
        document_id=unrelated.document_id,
        evidence=unrelated.evidence)
    initial = Selection(
        symbols=["pkg.Flow.run"],
        route_ids=("R1",),
        occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, dependency, unrelated), ("pkg.Flow.run",))

    assert completed.symbols == ["pkg.Flow.run", "pkg.Output.emit"]
    assert completed.route_ids == ("R1",)
    assert completed.occurrence_keys == (
        _occurrence_key(root), _occurrence_key(dependency))
def _fanout_hop(qualified_name, *, parent, file, line, call_file, call_line,
                relation="calls"):
    """A hop hydration discovered as a direct compiler dependency of a selected body."""
    plain = _hop(qualified_name, file=file, line=line)
    return plain.__class__(
        citation=plain.citation.__class__(**{
            **plain.citation.__dict__,
            "parent_qualified_name": parent,
            "call_site_file": call_file,
            "call_site_line": call_line,
            "relation": relation,
            "stop_reason": "selected_route_fanout"}),
        document_id=plain.document_id,
        evidence=plain.evidence)
def test_body_dependency_completion_dedupes_occurrences_without_spending_the_bound():
    """A repeated occurrence of one dependency must not evict a distinct second one.

    The per-body bound counts distinct retained occurrences: an identical duplicate
    collapses to a single selection and spends no budget, while a second call site
    of the same dependency is separate evidence — retained as its own occurrence
    under the one shared symbol. Route, section, and occurrence identity of the
    original selection stay intact.
    """
    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    first = _fanout_hop("pkg.Output.emit", parent="pkg.Flow.run",
                        file="output.py", line=20, call_file="flow.py", call_line=8)
    first_second_site = _fanout_hop(
        "pkg.Output.emit", parent="pkg.Flow.run",
        file="output.py", line=20, call_file="flow.py", call_line=11)
    second = _fanout_hop("pkg.Journal.append", parent="pkg.Flow.run",
                         file="journal.py", line=40, call_file="flow.py", call_line=9)
    third = _fanout_hop("pkg.Metrics.record", parent="pkg.Flow.run",
                        file="metrics.py", line=60, call_file="flow.py", call_line=10)
    initial = Selection(
        symbols=["pkg.Flow.run"], route_ids=("R1",), section_ids=("S1",),
        sections=[("doc-module", 1)], occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, first, first, first_second_site, second, third),
        ("pkg.Flow.run",), max_per_body=3)

    assert completed.symbols == [
        "pkg.Flow.run", "pkg.Output.emit", "pkg.Journal.append"]
    assert completed.occurrence_keys == (
        _occurrence_key(root), _occurrence_key(first),
        _occurrence_key(first_second_site), _occurrence_key(second))
    assert completed.route_ids == ("R1",)
    assert completed.section_ids == ("S1",)
    assert completed.sections == [("doc-module", 1)]


def test_body_dependency_completion_is_inert_without_selected_bodies():
    """The fallback path hydrates with no body symbols; completion must be a no-op."""
    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    dependency = _fanout_hop("pkg.Output.emit", parent="pkg.Flow.run",
                             file="output.py", line=20, call_file="flow.py",
                             call_line=8)
    initial = Selection(symbols=["pkg.Flow.run"], route_ids=("R1",),
                        occurrence_keys=(_occurrence_key(root),))

    for body_symbols in ((), None):
        assert complete_selection_with_body_dependencies(
            initial, (root, dependency), body_symbols) == initial


def test_completed_body_dependency_survives_projection_to_formulation_evidence():
    """Retention is real only if projection keeps the dependency it was given.

    The original bug: hydration discovered the direct call, then occurrence
    filtering in the projection threw it away because the selection never named
    it. The completed selection must carry the dependency through
    ``project_selected_evidence`` — definition and call site — while an
    unrelated hydrated fan-out stays excluded.
    """
    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    dependency = _fanout_hop("pkg.Output.emit", parent="pkg.Flow.run",
                             file="output.py", line=20, call_file="flow.py",
                             call_line=8)
    unrelated = _fanout_hop("pkg.Noise.emit", parent="pkg.Other.run",
                            file="noise.py", line=30, call_file="other.py",
                            call_line=5)
    hydrated = (root, dependency, unrelated)
    evidence = AnswerEvidence(
        bundle_citations=[root.citation], locations=locations_for((root,)),
        hops=(root,))
    selection = complete_selection_with_body_dependencies(
        Selection(symbols=["pkg.Flow.run"], route_ids=("R1",),
                  occurrence_keys=(_occurrence_key(root),)),
        hydrated, ("pkg.Flow.run",))

    projected = project_selected_evidence(
        evidence, selection, hydrated_hops=hydrated)

    names = [hop.citation.qualified_name for hop in projected.hops]
    assert names == ["pkg.Flow.run", "pkg.Output.emit"]
    assert [citation.qualified_name
            for citation in projected.bundle_citations] == names
    assert "output.py:20" in projected.locations
    assert not any("noise.py" in location for location in projected.locations)
def test_body_dependency_completion_retains_proven_owner_without_siblings():
    root = _hop("pkg.Flow.run", file="flow.py", line=10)
    owner_plain = _hop("pkg.Flow", file="flow.py", line=1)
    owner = owner_plain.__class__(
        citation=owner_plain.citation.__class__(**{
            **owner_plain.citation.__dict__,
            "relation": "localized", "stop_reason": "selected_owner"}),
        document_id=owner_plain.document_id, evidence=owner_plain.evidence)
    member_plain = _hop("pkg.Flow.run", file="flow.py", line=10)
    member = member_plain.__class__(
        citation=member_plain.citation.__class__(**{
            **member_plain.citation.__dict__,
            "parent_qualified_name": "pkg.Flow",
            "call_site_file": "flow.py", "call_site_line": 10,
            "relation": "contains", "stop_reason": "selected_owner_member"}),
        document_id=member_plain.document_id, evidence=member_plain.evidence)
    sibling = _hop("pkg.Flow.noise", file="flow.py", line=20)
    initial = Selection(
        symbols=["pkg.Flow.run"], occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, owner, member, sibling), ("pkg.Flow.run",))

    assert completed.symbols == ["pkg.Flow.run", "pkg.Flow"]
    assert completed.occurrence_keys == (
        _occurrence_key(root), _occurrence_key(owner), _occurrence_key(member))
def test_reference_paths_to_bridges_exclude_nonjoining_candidates():
    from library.chain_menu import reference_paths_to_bridges
    from library.structural_assembly import StructuralCitation

    def citation(name, parent, reason="selected_reference",
                 relation="references", line=1):
        return StructuralCitation(
            qualified_name=name, file=f"{name}.py",
            line_start=line, line_end=line, source_name="src1",
            relation=relation, hop=1, call_site_file=f"{parent}.py",
            call_site_line=line, stop_reason=reason,
            parent_qualified_name=parent)

    local = citation("pkg.Flow.localId", "pkg.Flow.run", line=2)
    key = citation("pkg.Keys.STABLE_ID", "pkg.Flow.localId", line=3)
    noise = citation("pkg.Noise.metric", "pkg.Flow.run", line=4)
    bridge = citation(
        "pkg.Engine.publish", "pkg.Keys.STABLE_ID",
        reason="reference_bridge", relation="shared_reference", line=5)

    retained = reference_paths_to_bridges(
        (local, noise, key), (bridge,))

    assert retained == (local, key)

def test_body_dependency_completion_retains_reverse_reference_owner_chain():
    from library.chain_menu import Selection, _occurrence_key
    from library.structural_assembly import StructuralCitation

    root = _hop("pkg.Flow.run", file="flow.py", line=20)
    reference = _fanout_hop(
        "pkg.Plan", parent="pkg.Flow.run", relation="references",
        file="plan.py", line=1, call_file="flow.py", call_line=21)
    reference = reference.__class__(
        citation=reference.citation.__class__(**{
            **reference.citation.__dict__,
            "stop_reason": "selected_reference"}),
        document_id=reference.document_id, evidence=reference.evidence)

    def reverse(name, parent, file, line):
        plain = _hop(name, file=file, line=line)
        return plain.__class__(
            citation=StructuralCitation(
                qualified_name=name, file=file, line_start=line, line_end=line,
                source_name="src1", relation="referenced_by", hop=0,
                call_site_file=file, call_site_line=line,
                stop_reason="selected_reference_caller",
                parent_qualified_name=parent),
            document_id=plain.document_id, evidence=plain.evidence)

    consumer = reverse("pkg.Rule.prepare", "pkg.Plan", "rule.py", 10)
    owner_plain = _hop("pkg.Rule", file="rule.py", line=1)
    owner = owner_plain.__class__(
        citation=StructuralCitation(
            qualified_name="pkg.Rule", file="rule.py",
            line_start=1, line_end=40, source_name="src1",
            relation="localized", hop=0, call_site_file="rule.py",
            call_site_line=1, stop_reason="selected_owner"),
        document_id=owner_plain.document_id, evidence=owner_plain.evidence)
    member_plain = _hop("pkg.Rule.prepare", file="rule.py", line=10)
    member = member_plain.__class__(
        citation=StructuralCitation(
            qualified_name="pkg.Rule.prepare", file="rule.py",
            line_start=10, line_end=20, source_name="src1",
            relation="contains", hop=1, call_site_file="rule.py",
            call_site_line=10, stop_reason="selected_owner_member",
            parent_qualified_name="pkg.Rule"),
        document_id=member_plain.document_id, evidence=member_plain.evidence)
    registrar = reverse(
        "pkg.Extension.install", "pkg.Rule", "extension.py", 10)
    noise = reverse("pkg.Noise.install", "pkg.Unselected", "noise.py", 10)
    initial = Selection(
        symbols=["pkg.Flow.run"], occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial,
        (root, reference, consumer, owner, member, registrar, noise),
        ("pkg.Flow.run",))

    assert completed.symbols == [
        "pkg.Flow.run", "pkg.Rule", "pkg.Plan",
        "pkg.Rule.prepare", "pkg.Extension.install"]
    assert set(completed.occurrence_keys) == {
        _occurrence_key(root), _occurrence_key(reference),
        _occurrence_key(consumer), _occurrence_key(owner),
        _occurrence_key(member), _occurrence_key(registrar)}

def test_hydration_materializes_reverse_reference_consumer_bodies(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops
    from library.structural_assembly import StructuralCitation

    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    reference = StructuralCitation(
        qualified_name="pkg.Plan", file="plan.py", line_start=1,
        line_end=10, source_name=SOURCE, relation="references", hop=1,
        call_site_file="flow.py", call_site_line=3,
        stop_reason="selected_reference", parent_qualified_name="pkg.Flow.run")
    reverse = StructuralCitation(
        qualified_name="pkg.Rule.prepare", file="rule.py",
        line_start=10, line_end=30, source_name=SOURCE,
        relation="referenced_by", hop=0,
        call_site_file="rule.py", call_site_line=11,
        stop_reason="selected_reference_caller",
        parent_qualified_name="pkg.Plan")
    captured = {}

    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: (reference,))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reverse_reference_fanout",
        lambda _conn, roots, **kwargs: (
            (reverse,) if roots == ("pkg.Plan",) else ()))

    def fake_curate(lib, citations, **kwargs):
        captured.update(kwargs)
        return ChainBundle(hops=[root], source_gaps=())

    monkeypatch.setattr("library.chain_bundle.curate_bundle", fake_curate)

    hydrate_selected_hops(
        library, [root], Selection(symbols=["pkg.Flow.run"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Flow.run",),
        reference_query="why the plan uses a rule")

    assert captured["definition_body_symbols"] == (
        "pkg.Flow.run", "pkg.Rule.prepare")
def test_hydration_closure_is_narrow_and_nonrecursive():
    import inspect
    from library.chain_menu import hydrate_selected_hops

    source = "".join(inspect.getsource(hydrate_selected_hops).split())
    assert "qualified_caller_fanout" not in source
    assert "reference_bridges" not in source
    assert "bridge_dependencies" not in source
    assert "depth=1" in source
    assert "recursive_per_root=0" in source
    assert "max_recursive_total=0" in source


def test_body_dependency_completion_stops_at_selected_body_boundary():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)

    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    direct = _fanout_hop(
        "pkg.Predicate.matches", parent="pkg.Flow.run",
        file="predicate.py", line=20, call_file="flow.py", call_line=8)
    recursive = _fanout_hop(
        "pkg.Bitmap.contains", parent="pkg.Predicate.matches",
        file="bitmap.py", line=40, call_file="predicate.py", call_line=22)
    initial = Selection(
        symbols=["pkg.Flow.run"], route_ids=("R1",),
        occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, recursive, direct), ("pkg.Flow.run",))

    assert completed.symbols == ["pkg.Flow.run", "pkg.Predicate.matches"]
    assert completed.occurrence_keys == (
        _occurrence_key(root), _occurrence_key(direct))
def test_hydration_follows_selected_interface_implementation_one_helper_layer(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops

    root = _hop("pkg.Contract.execute", file="contract.py", line=2)
    implementation = _fanout_hop(
        "pkg.Real.execute", parent="pkg.Contract.execute",
        file="real.py", line=20, call_file="contract.py", call_line=2,
        relation="implements")
    helper = _fanout_hop(
        "pkg.Real.matches", parent="pkg.Real.execute",
        file="real.py", line=40, call_file="real.py", call_line=24)
    calls = []
    captured = {}
    with library._conn_provider.acquire() as conn:
        conn.execute(
            "INSERT INTO scip_symbols "
            "(canonical_id,source_name,language,file,line_start,line_end,kind,"
            "display_name,qualified_name,parent_qualified_name) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("abstract-contract", SOURCE, "python", "contract.py", 2, 2,
             "AbstractMethod", "execute", "pkg.Contract.execute", "pkg.Contract"))
        conn.commit()

    def fake_call_fanout(_conn, roots, **kwargs):
        roots = tuple(roots)
        calls.append(roots)
        if roots == ("pkg.Contract.execute",):
            return (implementation.citation,)
        if roots == ("pkg.Real.execute",):
            return (helper.citation,)
        return ()

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        fake_call_fanout)
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda lib, citations, **kwargs: (
            captured.update(citations=tuple(citations), kwargs=kwargs)
            or ChainBundle(hops=(root, implementation, helper), source_gaps=())))

    hydrate_selected_hops(
        library, (root,), Selection(symbols=["pkg.Contract.execute"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Contract.execute",))

    assert calls == [
        ("pkg.Contract.execute",), ("pkg.Real.execute",)]
    assert captured["kwargs"]["definition_body_symbols"] == (
        "pkg.Contract.execute", "pkg.Real.execute", "pkg.Real.matches")
    assert captured["citations"] == (
        root.citation, implementation.citation, helper.citation)


def test_body_dependency_completion_retains_implementation_helper_but_not_call_recursion():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)

    root = _hop("pkg.Contract.execute", file="contract.py", line=2)
    implementation = _fanout_hop(
        "pkg.Real.execute", parent="pkg.Contract.execute",
        file="real.py", line=20, call_file="contract.py", call_line=2,
        relation="implements")
    helper = _fanout_hop(
        "pkg.Real.matches", parent="pkg.Real.execute",
        file="real.py", line=40, call_file="real.py", call_line=24)
    recursion = _fanout_hop(
        "pkg.Bitmap.contains", parent="pkg.Real.matches",
        file="bitmap.py", line=50, call_file="real.py", call_line=42)
    initial = Selection(
        symbols=["pkg.Contract.execute"],
        occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, implementation, helper, recursion),
        ("pkg.Contract.execute",))

    assert completed.symbols == [
        "pkg.Contract.execute", "pkg.Real.execute", "pkg.Real.matches"]
    assert completed.occurrence_keys == (
        _occurrence_key(root), _occurrence_key(implementation),
        _occurrence_key(helper))


def test_hydration_uses_selected_route_reference_as_reverse_lookup_root(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, _occurrence_key, hydrate_selected_hops
    from library.structural_assembly import StructuralCitation

    reference = StructuralCitation(
        qualified_name="pkg.Keys.STABLE_ID", file="sink.py",
        line_start=6, line_end=6, source_name=SOURCE,
        relation="references", hop=1, call_site_file="sink.py",
        call_site_line=12, stop_reason="selected_route",
        parent_qualified_name="pkg.Sink.id")
    route_hop = _hop("pkg.Keys.STABLE_ID", file="sink.py", line=6)
    route_hop = route_hop.__class__(
        citation=reference, document_id=route_hop.document_id,
        evidence=route_hop.evidence)
    consumer = StructuralCitation(
        qualified_name="pkg.Stream.publish", file="stream.py",
        line_start=20, line_end=40, source_name=SOURCE,
        relation="referenced_by", hop=0, call_site_file="stream.py",
        call_site_line=24, stop_reason="selected_reference_caller",
        parent_qualified_name="pkg.Keys.STABLE_ID")
    roots_seen = []
    captured = {}

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reverse_reference_fanout",
        lambda _conn, roots, **kwargs: (
            roots_seen.append(tuple(roots)) or (consumer,)))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda lib, citations, **kwargs: (
            captured.update(kwargs) or
            ChainBundle(hops=(route_hop,), source_gaps=())))

    hydrate_selected_hops(
        library, (route_hop,),
        Selection(
            symbols=["pkg.Keys.STABLE_ID"],
            occurrence_keys=(_occurrence_key(route_hop),)),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Sink.id",))

    assert roots_seen == [("pkg.Keys.STABLE_ID",)]
    assert captured["definition_body_symbols"] == (
        "pkg.Sink.id", "pkg.Stream.publish")
def test_hydration_bounds_body_reference_consumers_separately(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops

    root = _hop("pkg.Sink.id", file="sink.py", line=2)
    captured = {}

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reverse_reference_fanout",
        lambda *args, **kwargs: captured.update(kwargs) or ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda *args, **kwargs: ChainBundle(hops=(root,), source_gaps=()))

    hydrate_selected_hops(
        library, (root,), Selection(symbols=["pkg.Sink.id"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Sink.id",),
        reference_query="stable id")
    # No reference root means the reverse lookup is skipped.
    assert "per_root" not in captured

    from library.structural_assembly import StructuralCitation
    reference = StructuralCitation(
        qualified_name="pkg.Keys.ID", file="sink.py",
        line_start=3, line_end=3, source_name=SOURCE,
        relation="references", hop=1, call_site_file="sink.py",
        call_site_line=4, stop_reason="selected_reference",
        parent_qualified_name="pkg.Sink.id")
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: (reference,))

    hydrate_selected_hops(
        library, (root,), Selection(symbols=["pkg.Sink.id"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Sink.id",),
        reference_query="stable id")

    assert captured["per_root"] == 1
    assert captured["max_total"] == 8
    assert captured["lift_members"] is True
    assert captured["reserve_registrars"] is True
def test_hydration_reserves_reverse_reference_budget_for_selected_routes(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, _occurrence_key, hydrate_selected_hops
    from library.structural_assembly import StructuralCitation

    route_reference = StructuralCitation(
        qualified_name="pkg.Keys.ROUTE_ID", file="sink.py",
        line_start=3, line_end=3, source_name=SOURCE,
        relation="references", hop=1, call_site_file="sink.py",
        call_site_line=4, stop_reason="selected_route",
        parent_qualified_name="pkg.Sink.id")
    route_hop = _hop("pkg.Keys.ROUTE_ID", file="sink.py", line=3)
    route_hop = route_hop.__class__(
        citation=route_reference, document_id=route_hop.document_id,
        evidence=route_hop.evidence)
    body_reference = StructuralCitation(
        qualified_name="pkg.Types.BODY", file="body.py",
        line_start=5, line_end=5, source_name=SOURCE,
        relation="references", hop=1, call_site_file="body.py",
        call_site_line=6, stop_reason="selected_reference",
        parent_qualified_name="pkg.Sink.id")
    calls = []

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: (body_reference,))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reverse_reference_fanout",
        lambda _conn, roots, **kwargs: (
            calls.append((tuple(roots), kwargs["max_total"])) or ()))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda *args, **kwargs: ChainBundle(hops=(route_hop,), source_gaps=()))

    hydrate_selected_hops(
        library, (route_hop,),
        Selection(
            symbols=["pkg.Keys.ROUTE_ID"],
            occurrence_keys=(_occurrence_key(route_hop),)),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Sink.id",),
        reference_query="route id")

    assert calls == [
        (("pkg.Keys.ROUTE_ID",), 2),
        (("pkg.Types.BODY",), 8)]
def test_body_completion_retains_one_same_owner_reference_from_reverse_consumer():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)
    from library.structural_assembly import StructuralCitation

    key = _hop("pkg.Keys.ID", file="keys.py", line=2)
    consumer_citation = StructuralCitation(
        qualified_name="pkg.Stream.publish", file="stream.py",
        line_start=20, line_end=40, source_name=SOURCE,
        relation="referenced_by", hop=0, call_site_file="stream.py",
        call_site_line=24, stop_reason="selected_reference_caller",
        parent_qualified_name="pkg.Keys.ID")
    consumer = _hop("pkg.Stream.publish", file="stream.py", line=20)
    consumer = consumer.__class__(
        citation=consumer_citation, document_id=consumer.document_id,
        evidence=consumer.evidence)
    local_reference = StructuralCitation(
        qualified_name="pkg.Stream.id", file="stream.py",
        line_start=4, line_end=12, source_name=SOURCE,
        relation="references", hop=1, call_site_file="stream.py",
        call_site_line=25, stop_reason="selected_reference",
        parent_qualified_name="pkg.Stream.publish")
    local = _hop("pkg.Stream.id", file="stream.py", line=4)
    local = local.__class__(
        citation=local_reference, document_id=local.document_id,
        evidence=local.evidence)
    initial = Selection(
        symbols=["pkg.Keys.ID"],
        occurrence_keys=(_occurrence_key(key),))

    completed = complete_selection_with_body_dependencies(
        initial, (key, consumer, local), ("pkg.Sink.id",))

    assert completed.symbols == [
        "pkg.Keys.ID", "pkg.Stream.publish", "pkg.Stream.id"]
    assert set(completed.occurrence_keys) == {
        _occurrence_key(key), _occurrence_key(consumer),
        _occurrence_key(local)}
def test_hydration_keeps_decision_sibling_as_transition_not_full_body(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops
    from library.structural_assembly import StructuralCitation

    root = _hop("pkg.Decision.chosen", file="decision.py", line=20)
    sibling_citation = StructuralCitation(
        qualified_name="pkg.Decision.alternate", file="decision.py",
        line_start=30, line_end=40, source_name=SOURCE,
        relation="calls", hop=1, call_site_file="decision.py",
        call_site_line=13, stop_reason="selected_branch_sibling",
        parent_qualified_name="pkg.Decision.choose")
    captured = {}

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_same_owner_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.selected_route_branch_fanout",
        lambda _conn, matches, **kwargs: (sibling_citation,))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda lib, citations, **kwargs: (
            captured.update(citations=tuple(citations), kwargs=kwargs)
            or ChainBundle(hops=(root,), source_gaps=())))

    hydrate_selected_hops(
        library, (root,), Selection(symbols=["pkg.Decision.chosen"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Decision.chosen",))

    assert captured["kwargs"]["definition_body_symbols"] == ("pkg.Decision.chosen",)
    assert sibling_citation in captured["citations"]
def test_body_completion_retains_bounded_decision_sibling_without_children():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)
    from library.structural_assembly import StructuralCitation

    root = _hop("pkg.Decision.chosen", file="decision.py", line=20)
    sibling_citation = StructuralCitation(
        qualified_name="pkg.Decision.alternate", file="decision.py",
        line_start=30, line_end=40, source_name=SOURCE,
        relation="calls", hop=1, call_site_file="decision.py",
        call_site_line=13, stop_reason="selected_branch_sibling",
        parent_qualified_name="pkg.Decision.choose")
    sibling = _hop("pkg.Decision.alternate", file="decision.py", line=30)
    sibling = sibling.__class__(
        citation=sibling_citation, document_id=sibling.document_id,
        evidence=sibling.evidence)
    child = _fanout_hop(
        "pkg.Decision.classify", parent="pkg.Decision.alternate",
        file="decision.py", line=50, call_file="decision.py", call_line=34)
    initial = Selection(
        symbols=["pkg.Decision.chosen"],
        occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, sibling, child), ("pkg.Decision.chosen",))

    assert completed.symbols == ["pkg.Decision.chosen"]
    assert set(completed.occurrence_keys) == {
        _occurrence_key(root), _occurrence_key(sibling)}
def test_hydration_does_not_follow_implements_edge_from_concrete_body(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, hydrate_selected_hops

    root = _hop("pkg.Real.execute", file="real.py", line=20)
    base = _fanout_hop(
        "pkg.Contract.execute", parent="pkg.Real.execute",
        file="contract.py", line=2, call_file="real.py", call_line=20,
        relation="implements")
    calls = []
    captured = {}
    with library._conn_provider.acquire() as conn:
        conn.execute(
            "INSERT INTO scip_symbols "
            "(canonical_id,source_name,language,file,line_start,line_end,kind,"
            "display_name,qualified_name,parent_qualified_name) "
            "VALUES (?,?,?,?,?,?,?,?,?,?)",
            ("concrete-real", SOURCE, "python", "real.py", 20, 30,
             "Method", "execute", "pkg.Real.execute", "pkg.Real"))
        conn.commit()

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        lambda _conn, roots, **kwargs: (
            calls.append(tuple(roots)) or (base.citation,)))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.selected_route_branch_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda lib, citations, **kwargs: (
            captured.update(kwargs) or ChainBundle(hops=(root,), source_gaps=())))

    hydrate_selected_hops(
        library, (root,), Selection(symbols=["pkg.Real.execute"]),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Real.execute",))

    assert calls == [("pkg.Real.execute",)]
    assert captured["definition_body_symbols"] == ("pkg.Real.execute",)
def test_body_completion_retains_reverse_consumer_of_referenced_member_owner():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)
    from library.structural_assembly import StructuralCitation

    root = _hop("pkg.Flow.run", file="flow.py", line=20)
    reference = _fanout_hop(
        "pkg.Plan.target", parent="pkg.Flow.run", relation="references",
        file="plan.py", line=4, call_file="flow.py", call_line=21)
    reference = reference.__class__(
        citation=reference.citation.__class__(**{
            **reference.citation.__dict__,
            "stop_reason": "selected_reference"}),
        document_id=reference.document_id, evidence=reference.evidence)
    owner_plain = _hop("pkg.Plan", file="plan.py", line=1)
    owner = owner_plain.__class__(
        citation=StructuralCitation(
            qualified_name="pkg.Plan", file="plan.py",
            line_start=1, line_end=20, source_name="src1",
            relation="localized", hop=0, call_site_file="plan.py",
            call_site_line=1, stop_reason="selected_owner"),
        document_id=owner_plain.document_id, evidence=owner_plain.evidence)
    member_plain = _hop("pkg.Plan.target", file="plan.py", line=4)
    member = member_plain.__class__(
        citation=StructuralCitation(
            qualified_name="pkg.Plan.target", file="plan.py",
            line_start=4, line_end=4, source_name="src1",
            relation="contains", hop=1, call_site_file="plan.py",
            call_site_line=4, stop_reason="selected_owner_member",
            parent_qualified_name="pkg.Plan"),
        document_id=member_plain.document_id, evidence=member_plain.evidence)
    consumer_plain = _hop("pkg.Prepare.apply", file="prepare.py", line=10)
    consumer = consumer_plain.__class__(
        citation=StructuralCitation(
            qualified_name="pkg.Prepare.apply", file="prepare.py",
            line_start=10, line_end=20, source_name="src1",
            relation="referenced_by", hop=0, call_site_file="prepare.py",
            call_site_line=12, stop_reason="selected_reference_caller",
            parent_qualified_name="pkg.Plan"),
        document_id=consumer_plain.document_id, evidence=consumer_plain.evidence)
    initial = Selection(
        symbols=["pkg.Flow.run"], occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, reference, owner, member, consumer),
        ("pkg.Flow.run",))

    assert "pkg.Prepare.apply" in completed.symbols
    assert _occurrence_key(consumer) in completed.occurrence_keys
def test_hydration_excludes_selected_reference_from_same_owner_forward_lookup(
        library, monkeypatch):
    from library.chain_bundle import ChainBundle
    from library.chain_menu import Selection, _occurrence_key, hydrate_selected_hops
    from library.structural_assembly import StructuralCitation

    reference = StructuralCitation(
        qualified_name="pkg.Stream.ROUTE_ID", file="stream.py",
        line_start=4, line_end=4, source_name=SOURCE,
        relation="references", hop=1, call_site_file="sink.py",
        call_site_line=12, stop_reason="selected_route",
        parent_qualified_name="pkg.Sink.id")
    route = _hop("pkg.Stream.ROUTE_ID", file="stream.py", line=4)
    route = route.__class__(citation=reference, document_id=route.document_id,
                            evidence=route.evidence)
    consumer = StructuralCitation(
        qualified_name="pkg.Stream.publish", file="stream.py",
        line_start=20, line_end=40, source_name=SOURCE,
        relation="referenced_by", hop=0, call_site_file="stream.py",
        call_site_line=24, stop_reason="selected_reference_caller",
        parent_qualified_name="pkg.Stream.ROUTE_ID")
    captured = {}

    monkeypatch.setattr(
        "library.structural_assembly.qualified_call_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reference_fanout",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_reverse_reference_fanout",
        lambda *args, **kwargs: (consumer,))
    monkeypatch.setattr(
        "library.structural_assembly.qualified_same_owner_reference_fanout",
        lambda *args, **kwargs: captured.update(kwargs) or ())
    monkeypatch.setattr(
        "library.structural_assembly.qualified_owner_closure",
        lambda *args, **kwargs: ())
    monkeypatch.setattr(
        "library.chain_bundle.curate_bundle",
        lambda *args, **kwargs: ChainBundle(hops=(route,), source_gaps=()))

    hydrate_selected_hops(
        library, (route,),
        Selection(symbols=["pkg.Stream.ROUTE_ID"],
                  occurrence_keys=(_occurrence_key(route),)),
        source=SOURCE, source_root="/source",
        definition_body_symbols=("pkg.Sink.id",),
        reference_query="identifier survives")

    assert captured["excluded_targets"] == ("pkg.Stream.ROUTE_ID",)
def test_body_completion_retains_bounded_same_owner_reference_chain():
    from library.chain_menu import (
        Selection, _occurrence_key,
        complete_selection_with_body_dependencies)
    from library.structural_assembly import StructuralCitation

    root = _hop("pkg.Keys.ID", file="keys.py", line=2)
    def related(name, parent, line, relation):
        plain = _hop(name, file="stream.py", line=line)
        return plain.__class__(
            citation=StructuralCitation(
                qualified_name=name, file="stream.py",
                line_start=line, line_end=line + 4, source_name=SOURCE,
                relation=relation, hop=1, call_site_file="stream.py",
                call_site_line=line, stop_reason=(
                    "selected_reference_caller" if relation == "referenced_by"
                    else "selected_reference"),
                parent_qualified_name=parent),
            document_id=plain.document_id, evidence=plain.evidence)
    consumer = related(
        "pkg.Stream.publish", "pkg.Keys.ID", 10, "referenced_by")
    identifier = related(
        "pkg.Stream.id", "pkg.Stream.publish", 20, "references")
    metadata = related(
        "pkg.Stream.metadata", "pkg.Stream.id", 30, "references")
    initial = Selection(
        symbols=["pkg.Keys.ID"], occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, consumer, identifier, metadata), ("pkg.Sink.id",))

    assert completed.symbols == [
        "pkg.Keys.ID", "pkg.Stream.publish",
        "pkg.Stream.id", "pkg.Stream.metadata"]
    assert _occurrence_key(metadata) in completed.occurrence_keys
def test_body_dependency_completion_defaults_to_four_direct_dependencies():
    from library.chain_menu import Selection, _occurrence_key, complete_selection_with_body_dependencies
    root = _hop("pkg.Flow.run", file="flow.py", line=2)
    dependencies = tuple(
        _fanout_hop(f"pkg.Step{index}.run", parent="pkg.Flow.run",
                    file=f"step{index}.py", line=20 + index,
                    call_file="flow.py", call_line=8 + index)
        for index in range(6))
    initial = Selection(symbols=["pkg.Flow.run"],
                        occurrence_keys=(_occurrence_key(root),))

    completed = complete_selection_with_body_dependencies(
        initial, (root, *dependencies), ("pkg.Flow.run",))

    assert completed.symbols == [
        "pkg.Flow.run", "pkg.Step0.run", "pkg.Step1.run",
        "pkg.Step2.run", "pkg.Step3.run"]
    assert completed.occurrence_keys == (
        _occurrence_key(root),
        *(_occurrence_key(hop) for hop in dependencies[:4]))
def test_definition_body_selection_requires_every_selected_same_name_extent():
    from library.chain_menu import (
        DefinitionBodySelection, Selection, _occurrence_key,
        complete_definition_body_selection, definition_body_menu)

    first = _hop("pkg.Plan", file="plan.py", line=10)
    second = _hop("pkg.Plan", file="plan.py", line=30)
    route = Selection(
        symbols=["pkg.Plan"],
        occurrence_keys=(_occurrence_key(first), _occurrence_key(second)))

    menu = definition_body_menu((first, second), route)
    completed = complete_definition_body_selection(
        menu, DefinitionBodySelection())

    assert menu.required_symbols == ("pkg.Plan",)
    assert completed.symbols == ("pkg.Plan",)
def test_render_selected_does_not_describe_contains_as_called():
    from library.chain_menu import Fetched, Selection, render_selected

    plain = _hop("pkg.Owner.member", file="owner.py", line=8)
    contained = plain.__class__(
        citation=StructuralCitation(
            qualified_name="pkg.Owner.member", file="owner.py",
            line_start=8, line_end=12, source_name="src1",
            relation="contains", hop=1, call_site_file="owner.py",
            call_site_line=8, stop_reason="selected_owner_member",
            parent_qualified_name="pkg.Owner"),
        document_id=plain.document_id, evidence=plain.evidence)
    selection = Selection(
        symbols=["pkg.Owner.member"],
        occurrence_keys=(_occurrence_key(contained),))

    rendered = render_selected((contained,), selection, Fetched())

    assert "contained at owner.py:8" in rendered
    assert "called at owner.py:8" not in rendered
def test_route_parenting_treats_zero_line_end_as_single_line_extent(library):
    from library.chain_menu import route_menu_for

    exact = _hop("pkg.Parent.run", file="parent.py", line=10)
    exact = exact.__class__(citation=exact.citation.__class__(
        **{**exact.citation.__dict__, "line_end": 0}),
        document_id=exact.document_id, evidence=exact.evidence)
    later = _hop("pkg.Parent.run", file="parent.py", line=100)
    child = _hop("pkg.Child.commit", file="child.py", line=200)
    child = child.__class__(citation=child.citation.__class__(
        **{**child.citation.__dict__,
           "parent_qualified_name": "pkg.Parent.run",
           "call_site_file": "parent.py", "call_site_line": 10,
           "stop_reason": "leaf"}), document_id=child.document_id,
        evidence=child.evidence)

    menu = route_menu_for(library, [exact, later, child], source=SOURCE)
    route = next(label for label, names in menu.routes.items()
                 if names[-1] == "pkg.Child.commit")

    assert menu.route_occurrences[route][0][2] == 10
def test_module_menu_preserves_ownership_relation_wording():
    from library.chain_menu import (
        EvidenceGraph, EvidenceGraphEdge, EvidenceGraphNode,
        RouteMenu, module_menu_for,
    )

    graph = EvidenceGraph(
        nodes=(
            EvidenceGraphNode(
                id="G1", symbol="pkg.Owner", occurrence=(),
                file="owner.py", line=1),
            EvidenceGraphNode(
                id="G2", symbol="pkg.Owner.member", occurrence=(),
                file="owner.py", line=8),
        ),
        edges=(EvidenceGraphEdge(
            source="G1", target="G2", relation="contains",
            file="owner.py", line=8),),
    )
    menu = RouteMenu(routes={"R1": ("pkg.Owner", "pkg.Owner.member")})

    rendered = module_menu_for(menu, graph=graph).text

    assert "-contains->" in rendered
    assert "-calls->" not in rendered
