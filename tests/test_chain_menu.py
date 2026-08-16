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
from library.chain_bundle import BundleHop
from library.chain_menu import (
    fetch_selected,
    menu_for,
    render_selected,
    resolve_selection,
)
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

    def test_the_chosen_hops_travel_with_their_coordinates_and_bodies(self, library):
        menu = menu_for(library, CHAIN, source=SOURCE)
        chosen = resolve_selection(menu, '2')
        fetched = fetch_selected(library, chosen, CHAIN)

        text = render_selected(CHAIN, chosen, fetched)

        assert 'pkg/merge.py:53' in text, 'the coordinate is what the answer cites'
        assert 'Appends rows matching nothing.' in text
        assert 'writeAllChanges' not in text, 'what was not chosen does not travel'

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
