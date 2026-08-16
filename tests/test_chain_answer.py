"""Stage 4's input and stage 5's payload: what synthesis receives, and what comes back.

``index -> fetch document -> curate bundle -> formulate with LLM -> return response``. The
first three stages exist; this is the bridge into the fourth. Its whole job is that the
**evidence is the spine and prose is commentary** — the inversion the north star asks for.

Three properties are load-bearing:

* the spine is in **execution order**, because that is what makes a chain explicable — the
  live ``runMerge`` trace reads as the MERGE algorithm only because its hops are ordered by
  call-site line, and any reordering destroys it;
* every hop contributes ``file:line`` whether or not it has prose, because coordinates are
  what make an answer checkable;
* an answer may not claim a location the bundle does not contain — checked deterministically,
  no judge, because a plausible fabrication is exactly what this session kept producing.

Synthetic fixtures only: source ``src1``.
"""
from __future__ import annotations

import sqlite3

import pytest

from docgen.catalog_writer import _element_doc_id
from library import Library
from library.chain_answer import evidence_for, render_spine, unsupported_locations
from library.scip import init_scip_schema

SOURCE = 'src1'
RUN = 'scip-python python src1 0.1 `m`/run().'
HELPER = 'scip-python python src1 0.1 `m`/helper().'
DEEP = 'scip-python python src1 0.1 `m`/deep().'


def _symbol(conn, cid, *, file, qn, line_start, line_end, parent=''):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (cid, SOURCE, 'python', file, line_start, line_end, '', '', qn, parent))


def _edge(conn, caller, callee, *, line, file='m.py'):
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        "edge_type, file, line, confidence) VALUES (?,?,'call',?,?,'exact')",
        (caller, callee, file, line))


@pytest.fixture
def library(tmp_path):
    """A store with a chain (run -> helper -> deep) and prose for two of the hops."""
    lib = Library(tmp_path / 'l.db')
    with lib._conn_provider.acquire() as conn:
        init_scip_schema(conn)
        _symbol(conn, RUN, file='m.py', qn='m.run', line_start=5, line_end=20)
        _symbol(conn, HELPER, file='h.py', qn='m.helper', line_start=3, line_end=9)
        _symbol(conn, DEEP, file='d.py', qn='m.deep', line_start=40, line_end=44)
        _edge(conn, RUN, HELPER, line=8)
        _edge(conn, HELPER, DEEP, line=5, file='h.py')
        conn.commit()
    for qn, title, body in (
        ('m.run', 'run', 'Runs the operation end to end.'),
        ('m.helper', 'helper', 'Prepares the batch before writing.'),
    ):
        lib.add_document(content_type='catalog', title=title, content=body,
                         source_files=['m.py'], doc_id=_element_doc_id(SOURCE, qn),
                         source_name=SOURCE)
    yield lib
    lib.close()


DOCS = [{'source_files': ['m.py'], 'content': 'about the run operation'}]


class TestEvidence:
    def test_retrieved_documents_become_a_chain_with_coordinates(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        reached = [c.qualified_name for c in evidence.bundle_citations]
        assert 'm.helper' in reached and 'm.deep' in reached
        assert all(c.file and c.line_start for c in evidence.bundle_citations)

    def test_a_question_that_localizes_nowhere_yields_no_chain_and_says_so(self, library):
        evidence = evidence_for(library, [{'source_files': ['absent.py']}],
                                source=SOURCE, depth=3)

        assert evidence.bundle_citations == []
        assert evidence.unresolved_paths == ('absent.py',)
        assert evidence.spine == ''

    def test_documents_with_no_files_are_not_an_error(self, library):
        evidence = evidence_for(library, [{}, {'source_files': None}],
                                source=SOURCE, depth=3)

        assert evidence.bundle_citations == []


class TestSpine:
    def test_the_spine_is_in_execution_order_and_nests_by_depth(self, library):
        """`deep` is reached *through* `helper`, so it follows it and sits deeper.

        Asserted as position and indentation, not as a sort — an earlier version of this
        test sorted the lines by their own index, which can never fail.
        """
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)
        hops = [ln for ln in evidence.spine.splitlines() if '  [' in ln]

        assert 'm.helper' in hops[0]
        assert 'm.deep' in hops[1]
        indent = [len(ln) - len(ln.lstrip()) for ln in hops]
        assert indent[1] > indent[0], 'the nested hop must be indented under its caller'

    def test_every_hop_carries_its_coordinates_in_the_spine(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert 'h.py:3' in evidence.spine
        assert 'd.py:40' in evidence.spine

    def test_the_call_site_is_shown_because_it_is_what_proves_the_edge(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert 'm.py:8' in evidence.spine

    def test_prose_appears_only_for_hops_that_earned_it(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert 'Prepares the batch before writing.' in evidence.spine

    def test_an_empty_bundle_renders_nothing_rather_than_a_header(self):
        from library.chain_bundle import ChainBundle

        assert render_spine(ChainBundle()) == ''


class TestTheAnswerMayNotExceedTheBundle:
    """Deterministic, no judge — a plausible fabrication is the failure mode."""

    def test_a_location_absent_from_the_bundle_is_reported(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)
        answer = 'It runs through h.py:3 and then invented.py:99.'

        assert unsupported_locations(answer, evidence) == ('invented.py:99',)

    def test_an_answer_citing_only_bundle_locations_is_clean(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)
        answer = 'The batch is prepared at h.py:3 before d.py:40 writes it.'

        assert unsupported_locations(answer, evidence) == ()

    def test_a_right_file_at_a_wrong_line_is_still_unsupported(self, library):
        """The eval's gate gets this right and so must this: the line is the claim."""
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert unsupported_locations('see h.py:999', evidence) == ('h.py:999',)

    def test_an_answer_with_no_locations_at_all_is_not_reported_as_unsupported(
            self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert unsupported_locations('It prepares a batch.', evidence) == ()


class TestTheSpineIsBounded:
    """An LLM context window is a real constraint — unlike a graph walk, where a count
    cap was wrong. Measured on the rebuilt databricks store: 8 retrieved documents give
    279 hops and **18,285 tokens** of spine at depth 3, 22,892 at depth 4, on top of the
    ~17k-token document context. Unbounded, this dominates the prompt.

    The cut is a **prefix**, never a ranking: execution order is the one thing that makes
    a chain explicable, so the beginning survives intact and the tail is reported as
    omitted rather than silently dropped.
    """

    def test_the_spine_respects_a_character_budget_and_says_what_it_cut(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3,
                                max_spine_chars=60)

        assert 'm.deep' not in evidence.spine, 'the budget must actually bind'
        assert 'omitted' in evidence.spine.lower()
        assert evidence.truncation_reason

    def test_the_surviving_hops_are_the_first_ones_not_the_shortest(self, library):
        full = evidence_for(library, DOCS, source=SOURCE, depth=3)
        clipped = evidence_for(library, DOCS, source=SOURCE, depth=3,
                               max_spine_chars=60)

        first_full = [ln for ln in full.spine.splitlines() if '  [' in ln][0]
        first_clipped = [ln for ln in clipped.spine.splitlines() if '  [' in ln][0]
        assert first_clipped == first_full, 'the chain must keep its beginning'

    def test_a_revisited_body_renders_compactly_because_it_is_a_duplicate(self, library):
        """67 of 279 live hops were `revisit`. The call site is new evidence; the body was
        already shown, so repeating its definition line is pure cost.

        Uses a realistically long qualified name, because that is where the saving is: a
        Scala hop like `...merge.ClassicMergeExecutor.writeAllChanges` is 78 characters,
        and dropping it plus the definition coordinates is what pays. On a short synthetic
        name the compact form is actually longer, which an earlier version of this test
        asserted against and failed on.
        """
        from library.chain_bundle import BundleHop, ChainBundle
        from library.structural_assembly import StructuralCitation

        def hop(reason):
            return BundleHop(citation=StructuralCitation(
                qualified_name=('org.apache.spark.sql.delta.commands.merge'
                                '.ClassicMergeExecutor.writeAllChanges'),
                file='delta/spark/src/main/scala/.../ClassicMergeExecutor.scala',
                line_start=285,
                source_name=SOURCE, relation='calls', hop=1,
                call_site_file='m.py', call_site_line=8, stop_reason=reason))

        shown = render_spine(ChainBundle(hops=[hop('descended')]))
        again = render_spine(ChainBundle(hops=[hop('revisit')]))

        assert len(again) < len(shown)
        assert 'already shown' in again
        assert 'm.py:8' in again, 'the call site is still evidence and must stay'

    def test_the_full_spine_is_unchanged_when_it_fits(self, library):
        generous = evidence_for(library, DOCS, source=SOURCE, depth=3,
                                max_spine_chars=100_000)
        default = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert generous.spine == default.spine
        assert 'omitted' not in generous.spine.lower()
