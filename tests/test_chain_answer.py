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
from docgen.pricing import PROMPT_OVERHEAD_TOKENS
from library import Library, chain_answer
from library.chain_answer import (
    AnswerEvidence,
    ANSWER_MAX_TOKENS,
    evidence_for,
    expand_bare_lines,
    locations_for,
    render_spine,
    spine_budget_chars,
    unsupported_locations,
PROMPT_CHARS_PER_TOKEN, resolve_location)
from library.chain_bundle import BundleHop, ChainBundle; from library.scip import init_scip_schema
from library.structural_assembly import StructuralCitation

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
    def test_downstream_localization_recovers_the_connected_caller_path(self, library):
        evidence = evidence_for(
            library, [{'source_files': ['h.py']}], source=SOURCE, depth=3)

        names = [citation.qualified_name for citation in evidence.bundle_citations]
        assert names[:3] == ['m.run', 'm.helper', 'm.deep']
        assert evidence.bundle_citations[0].relation == 'called_by'
        assert evidence.bundle_citations[0].call_site_file == 'm.py'
        assert evidence.bundle_citations[0].call_site_line == 8
    def test_high_fan_in_caller_frontier_is_disclosed(self, library):
        with library._conn_provider.acquire() as conn:
            for number in range(15):
                caller = f'scip-python python src1 0.1 `wide{number}`/run().'
                _symbol(conn, caller, file=f'wide{number}.py', qn=f'wide{number}.run',
                        line_start=1, line_end=3)
                _edge(conn, caller, HELPER, line=2, file=f'wide{number}.py')
            conn.commit()

        evidence = evidence_for(
            library, [{'source_files': ['h.py']}], source=SOURCE, depth=3)

        assert evidence.caller_frontiers == (HELPER,)


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
    def test_the_document_reaches_the_spine_because_it_is_the_payload(self, library):
        """The spine carries each hop's description and its coordinates. Both, for different
        reasons.

        This reverses what this test asserted earlier today. The reasoning then was "code
        first, everything else suspect", applied to mean no generated text may reach the
        model at all — so the spine carried quoted source instead. That conflated distrust
        of *authority* with the question of *payload*. A docstring or a human-authored guide
        can contradict the code, and search-retrieved prose concatenated as background is
        worse; a per-hop catalog entry is neither, being fetched by deterministic id for the
        symbol the walk reached.

        What settles it is which artefact can be wrong. A generated description can; a SCIP
        coordinate cannot. So the description is what the model reads, and ``file:line``
        travels beside it so every resulting claim can be checked against the code.
        """
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert 'Prepares the batch before writing.' in evidence.spine
        assert 'h.py:3' in evidence.spine, 'the coordinate travels with the description'

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
    """The prompt has one real limit: the context window of the model that receives it.

    Not a number picked to leave room for something else. The bound this replaced —
    20,000 characters — was set when the prompt also carried ~17k tokens of retrieved
    documentation and the chain had to fit beside it. Nothing but the chain travels now,
    so that premise is void, and the budget is derived from the window less the answer
    reserved inside it.

    The cut is still a **prefix**, never a ranking: execution order is the one thing that
    makes a chain explicable, so the beginning survives intact and the tail is reported as
    omitted rather than silently dropped. Measured at production width the spine is 279
    hops and ~73k characters, which every window in the table holds many times over — so
    truncation now means the window, not a preference.
    """
    def test_the_spine_respects_a_character_budget_and_says_what_it_cut(self, library):
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3,
                                max_spine_chars=60)

        assert "m.helper" not in evidence.spine, "ancillary work yields to the mandatory route"
        assert "m.deep" in evidence.spine
        assert "omitted" in evidence.spine.lower()
        assert evidence.truncation_reason

    def test_the_surviving_hops_preserve_the_mandatory_route(self, library):
        clipped = evidence_for(library, DOCS, source=SOURCE, depth=3,
                               max_spine_chars=60)

        assert "m.deep" in clipped.spine
        assert "m.helper" not in clipped.spine

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
    def test_the_budget_is_the_context_window_not_a_number_someone_picked(self):
        """The bound is derived from the model that will receive the prompt.

        A declared limit cannot be checked against anything; a derived one can. The
        window is the constraint the API actually enforces, and the answer reserved
        inside it plus the instructions that frame it are the only other claimants.
        """
        opus = spine_budget_chars('claude-opus-5')
        haiku = spine_budget_chars('claude-haiku-4-5')

        assert opus == int((1_000_000 - ANSWER_MAX_TOKENS - PROMPT_OVERHEAD_TOKENS)
                           * PROMPT_CHARS_PER_TOKEN)
        assert haiku < opus, 'a smaller window must yield a smaller budget'
        assert haiku > 200_000, (
            'even the smallest window in the table dwarfs the 20,000 this replaced')

    def test_an_unknown_window_cuts_nothing_rather_than_guessing(self, library,
                                                                monkeypatch):
        """A model the table does not know is reported, never given a plausible limit.

        The same contract ``LLM_PRICING`` keeps by returning ``rates=None``, for the same
        reason: substituting a believable figure for a known one is what a derived bound
        exists to end. An over-long prompt then fails at the API — the boundary that owns
        the rule — instead of being quietly trimmed here.
        """
        assert chain_answer.spine_budget_chars('nobody-registered-this-model') is None

        monkeypatch.setattr(chain_answer, 'spine_budget_chars', lambda *a, **k: None)
        evidence = evidence_for(library, DOCS, source=SOURCE, depth=3)

        assert 'm.deep' in evidence.spine, 'the last hop survives an unknown window'
        assert not evidence.truncation_reason
    def test_a_description_is_shown_once_however_many_hops_reach_it(self):
        """The same symbol reached from three places is one description and three call sites.

        Measured at production width: 1,788 hops carry a document but only 883 documents are
        distinct, so each description was travelling about twice. A type referenced by four
        bodies is four hops — correctly, the call sites are four separate pieces of evidence —
        but its description does not change between them.

        This is the rule ``revisit`` already applies to a body, applied to a description.
        """
        from library.chain_bundle import BundleHop, ChainBundle
        from library.structural_assembly import StructuralCitation

        def hop(call_line):
            return BundleHop(
                citation=StructuralCitation(
                    qualified_name='pkg.data.Dataset', file='pkg/data.py', line_start=3,
                    source_name=SOURCE, relation='references', hop=1,
                    call_site_file='m.py', call_site_line=call_line,
                    stop_reason='reference', line_end=40),
                evidence='A dataset of rows, with a schema.')

        spine = render_spine(ChainBundle(hops=[hop(8), hop(19), hop(31)]))

        assert spine.count('A dataset of rows, with a schema.') == 1
        for line in (8, 19, 31):
            assert f'm.py:{line}' in spine, 'every call site is still evidence'
class TestWhatTheCallerGetsBack:
    """Stage five returns coordinates. It must not return the whole chain as a list.

    Measured at production width: the payload was 2,645 citation entries — 939,389
    characters, ~235,000 tokens — for one question, covering 973 distinct symbols. In an MCP
    client that lands in the caller's context window, so a single ask spent a quarter of it
    restating a chain the answer had already used four coordinates from.
    """

    def _evidence(self, answer_locations=()):
        hops = [
            StructuralCitation(qualified_name='m.run', file='m.py', line_start=5,
                               source_name=SOURCE, relation='calls', hop=1,
                               call_site_file='c.py', call_site_line=1, line_end=20),
            StructuralCitation(qualified_name='m.helper', file='h.py', line_start=3,
                               source_name=SOURCE, relation='calls', hop=2,
                               call_site_file='m.py', call_site_line=8, line_end=9),
            # the same definition reached again from a second call site
            StructuralCitation(qualified_name='m.helper', file='h.py', line_start=3,
                               source_name=SOURCE, relation='calls', hop=2,
                               call_site_file='m.py', call_site_line=12,
                               stop_reason='revisit', line_end=9),
        ]
        return AnswerEvidence(
            bundle_citations=hops,
            locations=frozenset(f'{h.file}:{h.line_start}' for h in hops))

    def test_the_payload_does_not_repeat_a_definition_it_already_named(self):
        """A definition reached twice is one coordinate, not two."""
        payload = self._evidence().citations()

        assert len(payload) == 2, f'expected the two distinct definitions, got {payload}'

    def test_the_payload_narrows_to_what_the_answer_actually_cited(self):
        """The answer used one location; returning 973 does not help the caller."""
        evidence = self._evidence()

        cited = evidence.cited_by('The work happens in h.py:3 after validation.')

        assert [c['file'] for c in cited] == ['h.py']
        assert cited[0]['line'] == 3

    def test_the_shape_travels_even_when_the_answer_cites_nothing(self):
        """Shape is what replaces enumeration — hop and symbol counts, and any forks."""
        summary = self._evidence().summary()

        assert summary['hops'] == 3
        assert summary['symbols'] == 2
        assert summary['files'] == 2
    def _evidence_with_call_sites(self):
        hops = [
            StructuralCitation(
                qualified_name='m.InsertOnly.writeOnlyInserts',
                file='spark/src/main/scala/delta/merge/InsertOnlyMergeExecutor.scala',
                line_start=53, source_name=SOURCE, relation='calls', hop=1,
                call_site_file='spark/src/main/scala/delta/MergeIntoCommand.scala',
                call_site_line=130, line_end=90),
        ]
        return AnswerEvidence(bundle_citations=hops, locations=locations_for(hops))

    def test_a_call_site_the_spine_showed_is_not_a_fabrication(self):
        """The prompt shows "called at X:130" and calls it the proof of the edge.

        A live run cited five call sites and every one was reported as invented, because the
        admissible set held definition coordinates only. The guard was rejecting exactly what
        the prompt had told the model to rely on.
        """
        evidence = self._evidence_with_call_sites()

        assert unsupported_locations(
            'The write happens at MergeIntoCommand.scala:130.', evidence) == ()

    def test_a_basename_resolves_to_the_one_location_it_can_mean(self):
        """An answer writes `InsertOnlyMergeExecutor.scala:53`, not the whole JVM path.

        Measured on a live run: 11 of 11 cited coordinates were real and every one was
        reported unsupported, because the set holds
        `spark/src/main/scala/delta/merge/InsertOnlyMergeExecutor.scala:53` and the answer
        wrote the file name. A basename that matches exactly one known location is that
        location; the index says so.
        """
        evidence = self._evidence_with_call_sites()

        assert unsupported_locations(
            'See InsertOnlyMergeExecutor.scala:53.', evidence) == ()

    def test_an_ambiguous_basename_is_reported_rather_than_assumed(self):
        """Two files of the same name at the same line: the citation cannot be pinned."""
        hops = [
            StructuralCitation(qualified_name='a.Thing.run', file='one/Thing.scala',
                               line_start=7, source_name=SOURCE, relation='calls', hop=1,
                               call_site_file='one/Caller.scala', call_site_line=3,
                               line_end=9),
            StructuralCitation(qualified_name='b.Thing.run', file='two/Thing.scala',
                               line_start=7, source_name=SOURCE, relation='calls', hop=1,
                               call_site_file='two/Caller.scala', call_site_line=4,
                               line_end=9),
        ]
        evidence = AnswerEvidence(bundle_citations=hops, locations=locations_for(hops))

        assert unsupported_locations('It is in Thing.scala:7.', evidence) == (
            'Thing.scala:7',)

    def test_an_invented_location_is_still_caught(self):
        """The guard must keep working: this is what it exists for."""
        evidence = self._evidence_with_call_sites()

        assert unsupported_locations(
            'Handled in NoSuchFile.scala:999.', evidence) == ('NoSuchFile.scala:999',)

    def test_the_payload_finds_the_hop_a_shortened_citation_refers_to(self):
        """`cited_by` must match the way the guard matches, or the payload comes back empty.

        On the live run it returned 0 citations for an answer that cited eleven real
        coordinates — the caller got nothing to navigate with.
        """
        evidence = self._evidence_with_call_sites()

        cited = evidence.cited_by('Written by InsertOnlyMergeExecutor.scala:53, '
                                 'called at MergeIntoCommand.scala:130.')
        assert {entry["qualified_name"] for entry in cited} == {"m.InsertOnly.writeOnlyInserts"}
        assert {(entry["file"], entry["line"]) for entry in cited} == {
            ("spark/src/main/scala/delta/merge/InsertOnlyMergeExecutor.scala", 53),
            ("spark/src/main/scala/delta/MergeIntoCommand.scala", 130),
        }
class TestBareLineReferences:
    """`(and again at :166)` names a line without a file. Expand it, don't ask the model to.

    Two reasons, and the second matters more. A reader gets a coordinate they can open
    without scanning back for the antecedent. And ``_LOCATION`` requires ``file.ext:line``,
    so a bare ``:166`` is invisible to the guard: it is neither validated nor resolved into a
    citation. Expanding it after the fact costs no prompt tokens and puts every claim the
    answer makes in front of the same check.
    """

    def test_a_bare_line_inherits_the_file_named_before_it(self):
        answer = ('invoked at MergeIntoCommand.scala:130 (and again at :166), then '
                  'ClassicMergeExecutor.scala:285 (and :155).')

        expanded = expand_bare_lines(answer)

        assert 'MergeIntoCommand.scala:166' in expanded
        assert 'ClassicMergeExecutor.scala:155' in expanded

    def test_a_bare_line_with_nothing_before_it_is_left_alone(self):
        """No antecedent means no claim to make explicit."""
        assert expand_bare_lines('see :166 for details') == 'see :166 for details'

    def test_ordinary_prose_and_times_are_not_coordinates(self):
        for text in ('Note: 166 rows', 'at 12:30 the job ran', 'ratio 3:4'):
            assert expand_bare_lines(text) == text

    def test_an_expanded_reference_is_judged_by_the_guard_like_any_other(self):
        """Expansion makes the model's implicit claim explicit — including a wrong one.

        Expanding only when it resolves would hide a bad claim from the check, which is
        backwards: the point is that every coordinate is verified, not that the text looks
        tidy.
        """
        hops = [StructuralCitation(
            qualified_name='m.run', file='delta/MergeIntoCommand.scala', line_start=130,
            source_name=SOURCE, relation='calls', hop=1,
            call_site_file='delta/Caller.scala', call_site_line=8, line_end=140)]
        evidence = AnswerEvidence(bundle_citations=hops, locations=locations_for(hops))

        answer = expand_bare_lines('runs at MergeIntoCommand.scala:130 (and again at :999)')

        assert unsupported_locations(answer, evidence) == ('MergeIntoCommand.scala:999',)
class TestAHopSaysWhatSCIPRecorded:
    """A type reference is not a call, and the prompt must not claim it was.

    The walk already records the distinction — ``relation='references'`` for a ``type_ref``
    edge, ``'calls'`` for a ``call`` — and the renderer discarded it, labelling every hop
    ``called at``. That matters because of how framework components are wired: measured on
    the live store, the only inbound edges an analyzer rule has from outside itself are
    ``type_ref``, emitted where the extension registers it. Rendering that as "called at"
    tells synthesis the rule was invoked at the registration site, which the index never
    claimed — a false mechanism on a real coordinate, which is the one fabrication the
    location guard cannot catch.
    """

    def _hop(self, relation, stop_reason, qualified_name, call_line):
        return BundleHop(
            citation=StructuralCitation(
                qualified_name=qualified_name, file='pkg/rule.py', line_start=3,
                source_name=SOURCE, relation=relation, hop=1,
                call_site_file='ext.py', call_site_line=call_line,
                stop_reason=stop_reason, line_end=40),
            evidence=f'What {qualified_name} does.')

    def test_each_hop_is_labelled_the_way_scip_recorded_it(self):
        spine = render_spine(ChainBundle(hops=[
            self._hop('calls', 'leaf', 'pkg.runner.run', 11),
            self._hop('references', 'reference', 'pkg.Rule', 12),
        ]))

        assert 'called at ext.py:11' in spine, 'a call edge is still a call'
        assert 'referenced at ext.py:12' in spine, (
            'a type_ref edge must be presented as a reference')
        assert spine.count('called at') == 1, (
            'the reference hop must not be described as a call')
def test_prompt_preflight_is_a_hard_character_ceiling():
    from library.chain_answer import bounded_prompt, resolve_location

    bounded = bounded_prompt("x" * 100, max_chars=40)

    assert len(bounded) <= 40
    assert bounded.endswith("[prompt truncated]")
    assert bounded_prompt("short", max_chars=40) == "short"
def test_answer_reserve_can_finish_a_multi_repository_chain():
    assert ANSWER_MAX_TOKENS >= 4096
def test_an_accepted_clew_route_replaces_broad_document_seed_expansion(library):
    from library.clews import Clew, ClewMatch
    with library._conn_provider.acquire() as conn:
        for number in range(20):
            symbol = f"scip-python python src1 0.1 `noise{number}`/run()."
            _symbol(conn, symbol, file="m.py", qn=f"noise{number}.run",
                    line_start=100 + number, line_end=100 + number)
        conn.commit()
    match = ClewMatch(clew=Clew(
        id="route", source_name=SOURCE, entry_symbol="m.helper",
        route=["m.helper", "m.deep"], files=["h.py"], strategy="test"),
        similarity=0.9)

    evidence = evidence_for(library, DOCS, source=SOURCE, depth=3,
                            clew_matches=[match])
    names = {citation.qualified_name for citation in evidence.bundle_citations}

    # Bounded compiler-verified entry discovery may recover the upstream
    # caller of a route endpoint; broad document-seed expansion stays off.
    assert {"m.helper", "m.deep"} <= names
    assert names <= {"m.run", "m.helper", "m.deep"}
    assert not any(name.startswith("noise") for name in names)
    assert len(evidence.bundle_citations) < 10
def test_recovered_roots_and_original_seeds_share_one_graph_walk(library, monkeypatch):
    import library.structural_assembly as assembly
    original = assembly.chain_from_seeds
    nonempty_calls = []
    def recording(conn, symbols, **kwargs):
        if symbols:
            nonempty_calls.append(tuple(symbols))
        return original(conn, symbols, **kwargs)
    monkeypatch.setattr(assembly, "chain_from_seeds", recording)

    evidence_for(library, [{"source_files": ["h.py"]}],
                 source=SOURCE, depth=3)

    assert len(nonempty_calls) == 1
    assert RUN in nonempty_calls[0] and HELPER not in nonempty_calls[0]
def test_cited_by_preserves_an_exact_supported_line_inside_a_definition():
    hop = StructuralCitation(
        qualified_name="DeltaSink.PendingTxn.commit",
        file="delta/DeltaSink.scala", line_start=78, line_end=96,
        source_name=SOURCE, relation="calls", hop=1,
        call_site_file="delta/Caller.scala", call_site_line=10)
    evidence = AnswerEvidence(bundle_citations=[hop], locations=locations_for([hop]))

    cited = evidence.cited_by("SetTransaction is constructed at DeltaSink.scala:87.")

    assert cited and cited[0]["line"] == 87
    assert cited[0]["file"] == "delta/DeltaSink.scala"
def test_cited_by_expands_every_supported_line_in_a_cited_range():
    hop = StructuralCitation(
        qualified_name="DeltaSink.PendingTxn.commit",
        file="delta/DeltaSink.scala", line_start=78, line_end=103,
        source_name=SOURCE, relation="calls", hop=1,
        call_site_file="delta/Caller.scala", call_site_line=10)
    evidence = AnswerEvidence(bundle_citations=[hop], locations=locations_for([hop]))

    cited = evidence.cited_by("See DeltaSink.scala:78-103.")
    assert [(entry["line"], entry["line_end"]) for entry in cited] == [(78, 103)]
def test_an_ellipsis_path_resolves_only_by_a_unique_basename():
    locations = frozenset({
        "spark/src/delta/DeltaFileFormatWriter.scala:123",
        "spark/src/other/Other.scala:123",
    })
    assert resolve_location("spark/.../DeltaFileFormatWriter.scala:123", locations) == \
        "spark/src/delta/DeltaFileFormatWriter.scala:123"
    ambiguous = locations | {"other/DeltaFileFormatWriter.scala:123"}
    assert resolve_location("spark/.../DeltaFileFormatWriter.scala:123", ambiguous) is None
def test_summary_exposes_exact_structural_blockers():
    from library.structural_assembly import FanOut
    evidence = AnswerEvidence(
        unresolved_paths=("docs/missing.md",),
        caller_frontiers=("pkg.Dispatch.run",),
        source_gaps=("src: outside root: docs/missing.md",),
        fan_outs=(FanOut("pkg.Interface.run", "api.py", 9, 25,
                         (("prod", 20), ("test", 5)), 5),),
        truncation_reason="chain truncated")

    summary = evidence.summary()

    assert summary["source_gaps"] == ["src: outside root: docs/missing.md"]
    assert summary["unresolved_paths"] == ["docs/missing.md"]
    assert summary["caller_frontiers"] == ["pkg.Dispatch.run"]
    assert summary["forks"][0]["by_package"] == [["prod", 20], ["test", 5]]
    assert summary["forks"][0]["test_implementations"] == 5
def test_question_route_decides_which_forks_are_executable_blockers():
    from library.chain_answer import mandatory_fan_outs
    from library.structural_assembly import FanOut
    ancillary = FanOut("org.spark.SupportsWrite.newWriteBuilder", "SupportsWrite.scala", 10, 25)
    routed = FanOut("org.delta.DeltaSink.addBatch", "DeltaSink.scala", 20, 12)

    assert mandatory_fan_outs("How does DeltaSink add a batch?", (routed,), (ancillary, routed)) == (routed,)
    assert mandatory_fan_outs("How does DeltaSink add a batch?", (), (ancillary,)) == ()
def test_spine_budget_reserves_space_for_later_mandatory_route_hops():
    from library.chain_bundle import BundleHop, ChainBundle
    from library.structural_assembly import StructuralCitation
    def item(name, relation, payload):
        return BundleHop(StructuralCitation(name, f"{name}.py", 1, "src", relation, 1,
                                            "caller.py", 2, line_end=1), evidence=payload)
    bundle = ChainBundle(hops=[
        item("noise", "calls", "x" * 400),
        item("target", "localized", "mandatory evidence"),
    ])

    spine = render_spine(bundle, max_chars=260)

    assert "target" in spine
    assert "mandatory evidence" in spine
    assert "x" * 100 not in spine
    assert "omitted" in spine
def test_external_cost_ceiling_can_tighten_the_model_window(monkeypatch):
    from library.chain_answer import bounded_prompt
    monkeypatch.setenv("ARIADNE_MAX_PROMPT_CHARS", "40")
    bounded = bounded_prompt("x" * 100, max_chars=90)
    assert len(bounded) <= 40
def test_document_fallback_seeds_are_ranked_by_the_question_before_traversal(
        library, monkeypatch):
    target = "scip-python python src1 0.1 `TargetCommit`/write()."
    with library._conn_provider.acquire() as conn:
        _symbol(conn, target, file="m.py", qn="pkg.TargetCommit.write",
                line_start=70, line_end=74)
        for number in range(30):
            noise = f"scip-python python src1 0.1 `Noise{number}`/unrelated()."
            _symbol(conn, noise, file="m.py", qn=f"pkg.Noise{number}.unrelated",
                    line_start=100 + number, line_end=100 + number)
        conn.commit()

    import library.structural_assembly as assembly
    original = assembly.chain_from_seeds
    calls = []
    def recording(conn, symbols, **kwargs):
        calls.append(tuple(symbols))
        return original(conn, symbols, **kwargs)
    monkeypatch.setattr(assembly, "chain_from_seeds", recording)

    evidence_for(
        library, DOCS, source=SOURCE, depth=1,
        question="How does TargetCommit write the commit?")

    assert any(target in call for call in calls)
    assert not any("`Noise" in seed for call in calls for seed in call)
def test_entity_candidates_do_not_expand_ordinary_document_files(
        library, monkeypatch):
    from library.clews import Clew, ClewMatch
    target = "scip-python python src1 0.1 `StableQueryId`/read()."
    with library._conn_provider.acquire() as conn:
        _symbol(conn, target, file="m.py", qn="spark.StableQueryId.read",
                line_start=70, line_end=74)
        conn.commit()
    match = ClewMatch(clew=Clew(
        id="delta-route", source_name=SOURCE, entry_symbol="m.helper",
        route=["m.helper", "m.deep"], files=["h.py"], strategy="test"),
        similarity=0.9)
    import library.structural_assembly as assembly
    original = assembly.chain_from_seeds
    calls = []
    def recording(conn, symbols, **kwargs):
        calls.append(tuple(symbols))
        return original(conn, symbols, **kwargs)
    monkeypatch.setattr(assembly, "chain_from_seeds", recording)

    evidence_for(
        library, DOCS, source=SOURCE, depth=1, clew_matches=[match],
        question="Which StableQueryId survives the restart?")

    flattened = {seed for call in calls for seed in call}
    assert target not in flattened, "entity candidates remain menus, not traversal seeds"
    assert not any("`Noise" in seed for seed in flattened)
def test_catalog_positioning_selects_only_the_best_described_code_files(library):
    from library.chain_answer import catalog_positioning_documents

    library.add_document(
        content_type="catalog", title="ClassicMergeExecutor",
        content=("Performs joins between source and target rows, decides matched insert "
                 "update delete actions, and writes resulting output rows."),
        source_files=["delta/ClassicMergeExecutor.scala"],
        doc_id="classic", source_name="src1")
    library.add_document(
        content_type="catalog", title="MergeMetrics",
        content="Records merge counters and timing metrics.",
        source_files=["delta/MergeMetrics.scala"],
        doc_id="metrics", source_name="src1")
    library.add_document(
        content_type="catalog", title="UnrelatedClock",
        content="Tracks wall clock time.", source_files=["util/Clock.scala"],
        doc_id="clock", source_name="src1")

    found = catalog_positioning_documents(
        library,
        "For MERGE, when is the join run relative to deciding insert update delete fate, "
        "and how are resulting rows emitted?",
        sources=("src1",), limit=2)

    assert [document.id for document in found] == ["classic"]
def test_catalog_positioning_is_source_scoped_and_hard_capped(library):
    from library.chain_answer import catalog_positioning_documents

    for number in range(5):
        library.add_document(
            content_type="catalog", title=f"MergeWriter{number}",
            content="Merge writer joins rows and writes output records.",
            source_files=[f"src/MergeWriter{number}.scala"],
            doc_id=f"local-{number}", source_name="src1")
    library.add_document(
        content_type="catalog", title="ForeignMergeWriter",
        content="Merge writer joins rows and writes output records.",
        source_files=["foreign/MergeWriter.scala"],
        doc_id="foreign", source_name="foreign")

    found = catalog_positioning_documents(
        library, "How does the merge writer join rows and write output records?",
        sources=("src1",), limit=2)

    assert len(found) == 2
    assert all(document.source_name == "src1" for document in found)
def test_selected_obligation_targets_are_connected_by_bounded_scip_path(library):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import connect_obligation_targets
    with library._conn_provider.acquire() as conn:
        match = ClewMatch(clew=Clew(id="route", source_name=SOURCE,
            entry_symbol="m.run", route=["m.run"], files=["m.py"]),
            similarity=1.0, obligations=(1,),
            target_symbols=((1, "m.run"), (1, "m.deep")))

        citations = connect_obligation_targets(conn, [match], source=SOURCE)

    assert [citation.qualified_name for citation in citations] == [
        "m.run", "m.helper", "m.deep"]


def test_selected_clew_target_symbol_is_materialized_as_evidence(library):
    from library.clews import Clew, ClewMatch
    target = "scip-python python src1 0.1 `Reconciliation`#complete()."
    with library._conn_provider.acquire() as conn:
        _symbol(conn, target, file="target.py",
                qn="billing.Reconciliation.complete", line_start=70, line_end=75)
        conn.commit()
    library.add_document(
        content_type="catalog", title="reconciliation complete",
        content="Completes reconciliation.", source_files=["target.py"],
        doc_id=_element_doc_id(SOURCE, "billing.Reconciliation.complete"),
        source_name=SOURCE)
    match = ClewMatch(clew=Clew(
        id="route", source_name=SOURCE, entry_symbol="m.helper",
        route=["m.helper"], files=["h.py"], strategy="test"),
        similarity=0.9, obligations=(1,),
        target_symbols=((1, "billing.Reconciliation.complete"),))

    evidence = evidence_for(
        library, DOCS, source=SOURCE, clew_matches=[match],
        question="How does reconciliation complete?", defer_source=True)

    targets = [hop.citation for hop in evidence.hops
               if hop.citation.qualified_name == "billing.Reconciliation.complete"]
    assert targets
    assert targets[0].relation == "localized"


def test_targeted_positioning_stays_mandatory_when_a_clew_is_authoritative(library):
    from library.clews import Clew, ClewMatch

    target = "scip-python python src1 0.1 `TargetMerge`/run()."
    with library._conn_provider.acquire() as conn:
        _symbol(conn, target, file="target.py", qn="delta.TargetMerge.run",
                line_start=50, line_end=60)
        _edge(conn, target, HELPER, line=55, file="target.py")
        conn.commit()
    targeted = [{"source_files": ["target.py"],
                 "content": "Target merge run orchestration."}]
    match = ClewMatch(clew=Clew(
        id="route", source_name=SOURCE, entry_symbol="m.helper",
        route=["m.helper", "m.deep"], files=["h.py"], strategy="test"),
        similarity=0.9)

    evidence = evidence_for(
        library, DOCS, source=SOURCE, depth=2, clew_matches=[match],
        question="How does TargetMerge run?",
        positioning_documents=targeted)

    targeted_hops = [hop.citation for hop in evidence.hops
                     if hop.citation.qualified_name == "delta.TargetMerge.run"]
    assert targeted_hops
    assert targeted_hops[0].relation == "localized"
def test_targeted_positioning_rejects_nonproduction_files(library):
    from library.chain_answer import catalog_positioning_documents

    library.add_document(
        content_type="explanation", title="MergeExecutorBenchmark",
        content="Merge executor joins rows, updates deletes inserts, and writes output.",
        source_files=["delta/benchmarks/MergeExecutorBenchmark.scala"],
        doc_id="benchmark", source_name="src1")
    library.add_document(
        content_type="explanation", title="MergeExecutor",
        content="Merge executor joins rows, updates deletes inserts, and writes output.",
        source_files=["delta/main/MergeExecutor.scala"],
        doc_id="production", source_name="src1")

    found = catalog_positioning_documents(
        library, "How does MERGE join rows and write update delete insert output?",
        sources=("src1",), limit=2)

    assert [document.id for document in found] == ["production"]
def test_catalog_positioning_uses_embedding_before_fetching_selected_documents(library):
    import numpy as np
    from library.chain_answer import catalog_positioning_documents

    library.add_document(
        content_type="catalog", title="GenericDeltaStreamingWrite",
        content="Mentions every surface term but describes unrelated metrics.",
        source_files=["delta/Metrics.scala"], doc_id="semantic-decoy",
        source_name="src1", embedding=np.array([0.0, 1.0], dtype=np.float32))
    library.add_document(
        content_type="catalog", title="DeltaSink.addBatch",
        content="Checks the transaction log for the query identifier and batch version before writing.",
        source_files=["delta/DeltaSink.scala"], doc_id="semantic-target",
        source_name="src1", embedding=np.array([1.0, 0.0], dtype=np.float32))
    library.add_document(
        content_type="catalog", title="GeneratedStreamingBatch",
        content="Generated wire model.", source_files=["target/Generated.java"],
        doc_id="semantic-generated", source_name="src1",
        embedding=np.array([1.0, 0.0], dtype=np.float32))

    found = catalog_positioning_documents(
        library, "How does replay remain exactly once after restart?",
        sources=["src1"], limit=1,
        query_embedding=np.array([1.0, 0.0], dtype=np.float32),
        matrix_provider=lambda: None)

    assert [document.id for document in found] == ["semantic-target"]
def test_selected_catalog_heading_resolves_only_its_exact_scip_symbol(library, monkeypatch):
    target = "exact-catalog-target"
    neighbor = "same-file-neighbor"
    with library._conn_provider.acquire() as conn:
        _symbol(conn, target, file="delta/Merge.scala", qn="delta.Merge.apply",
                line_start=10, line_end=20)
        _symbol(conn, neighbor, file="delta/Merge.scala", qn="delta.Merge.metrics",
                line_start=30, line_end=40)
        conn.commit()
    positioning = [{
        "source_files": ["delta/Merge.scala"],
        "metadata": {"qualified_name": "delta.Merge.apply"},
        "content": "Selected compact catalog description."}]
    import library.structural_assembly as assembly
    original = assembly.chain_from_seeds
    calls = []
    def recording(conn, symbols, **kwargs):
        calls.append(tuple(symbols))
        return original(conn, symbols, **kwargs)
    monkeypatch.setattr(assembly, "chain_from_seeds", recording)

    evidence = evidence_for(library, [], source=SOURCE, depth=1,
                           question="Why is merge diverted during analysis?",
                           positioning_documents=positioning)

    assert {item["symbol"]: item["origins"] for item in evidence.seed_provenance}[
        "delta.Merge.apply"] == ["catalog"]
    flattened = {seed for call in calls for seed in call}
    assert target in flattened
    assert neighbor not in flattened
def test_catalog_positioning_covers_compound_question_roles_before_embedding_ties(library):
    import numpy as np
    from library.chain_answer import catalog_positioning_documents
    vector = np.array([1.0, 0.0], dtype=np.float32)
    for doc_id, title, file, embedding in (
        ("conflict", "ConflictChecker.checkForDeletedFilesAgainstCurrentTxnReadFiles",
         "conflict/ConflictChecker.scala", np.array([1.0, 0.0], dtype=np.float32)),
        ("scan", "ScanWithDeletionVectors.createRowIndexFilterNode",
         "delta/ScanWithDeletionVectors.scala", np.array([0.98, 0.2], dtype=np.float32)),
        ("reader", "ParquetFileFormat.buildPhysicalReader",
         "spark/ParquetFileFormat.scala", np.array([0.97, 0.24], dtype=np.float32))):
        library.add_document(
            content_type="catalog", title=title,
            content="Equally similar compact catalog description.",
            source_files=[file], doc_id=doc_id, source_name="src1",
            embedding=embedding)

    found = catalog_positioning_documents(
        library,
        "How do the deletion vectors query-plan filter and physical Parquet reader agree?",
        sources=["src1"], limit=2, query_embedding=vector,
        matrix_provider=lambda: None)

    assert {document.id for document in found} == {"scan", "reader"}
def test_catalog_role_ranking_contains_no_domain_vocabulary_aliases():
    import inspect
    from library.chain_answer import catalog_positioning_documents

    source = inspect.getsource(catalog_positioning_documents)

    assert '"identifier": "id"' not in source
    assert '"streaming": "stream"' not in source
    assert '"transaction": "txn"' not in source
    assert '"deleted": "deletion"' not in source
def test_authoritative_clew_still_expands_a_semantic_document_seed(library):
    from library.clews import Clew, ClewMatch
    match = ClewMatch(clew=Clew(
        id="route", source_name=SOURCE, entry_symbol="m.helper",
        route=["m.helper", "m.deep"], files=["h.py"], strategy="test"),
        similarity=0.9)

    evidence = evidence_for(
        library, DOCS, source=SOURCE, depth=2, clew_matches=[match],
        question="How does m.run reach the terminal?", positioning_documents=DOCS)

    names = {citation.qualified_name for citation in evidence.bundle_citations}
    assert {"m.run", "m.helper", "m.deep"}.issubset(names)
def test_shared_reference_target_is_prioritized_with_its_sibling_owners(library):
    import inspect

    source = inspect.getsource(evidence_for)
    target = source.index("target_citations =")
    bridge = source.index("*bridge_expansion.citations", target)
    curate = source.index("curate_bundle(", bridge)

    assert target < bridge < curate
    assert "citation.qualified_name in bridge_targets" in source[target:bridge]
def test_catalog_positioning_fuses_a_lexical_owner_outside_the_semantic_cut(library):
    import numpy as np
    from library.chain_answer import catalog_positioning_documents

    vector = np.array([1.0, 0.0], dtype=np.float32)
    for number in range(60):
        library.add_document(
            content_type="catalog", title=f"GenericTransactionWorker{number}",
            content="Generic transaction worker description.",
            source_files=[f"generic/Worker{number}.scala"], doc_id=f"generic-{number}",
            source_name="src1", embedding=vector)
    library.add_document(
        content_type="catalog", title="StableQueryIdReplayOwner.recordRestartState",
        content="Records the stable query identifier before replay after restart.",
        source_files=["replay/StableQueryIdReplayOwner.scala"], doc_id="lexical-owner",
        source_name="src1", embedding=np.array([0.0, 1.0], dtype=np.float32))

    found = catalog_positioning_documents(
        library,
        "How does StableQueryId protect the replay transaction after a writer restart?",
        sources=["src1"], limit=2, query_embedding=vector,
        matrix_provider=lambda: None)

    assert "lexical-owner" in [document.id for document in found]
def test_render_spine_does_not_describe_contains_as_called():
    from library.chain_answer import render_spine
    from library.chain_bundle import BundleHop, ChainBundle
    from library.structural_assembly import StructuralCitation

    hop = BundleHop(citation=StructuralCitation(
        qualified_name="pkg.Owner.member", file="owner.py", line_start=8,
        source_name=SOURCE, relation="contains", hop=1,
        call_site_file="owner.py", call_site_line=8,
        parent_qualified_name="pkg.Owner", line_end=12))

    spine = render_spine(ChainBundle(hops=[hop]))

    assert "contained at owner.py:8" in spine
    assert "called at owner.py:8" not in spine
def test_render_spine_revisit_preserves_non_call_relation():
    hop = BundleHop(citation=StructuralCitation(
        qualified_name="pkg.Owner.member", file="owner.py", line_start=8,
        line_end=8, source_name=SOURCE, relation="contains", hop=1,
        call_site_file="owner.py", call_site_line=8,
        stop_reason="revisit", parent_qualified_name="pkg.Owner"))

    spine = render_spine(ChainBundle(hops=[hop]))

    assert "contained at owner.py:8" in spine
    assert "called again" not in spine


def test_locations_admit_hash_verified_excerpt_coordinates():
    from library.source_materialization import SourceExcerpt
    hop = BundleHop(
        citation=StructuralCitation(
            qualified_name='m.run', file='m.py', line_start=20, line_end=25,
            source_name=SOURCE, relation='calls', hop=1,
            call_site_file='', call_site_line=0),
        source_excerpts=(SourceExcerpt(
            source_name=SOURCE, file='m.py', line_start=17, line_end=19,
            kind='doc_header', content='# docs', sha256='x'),))

    locations = locations_for([hop])

    assert {'m.py:17', 'm.py:18', 'm.py:19'} <= set(locations)
