"""Deterministic question facets: exact spans, never paraphrases.

Facets are the planner-independent question representation: every facet
carries the exact question-text span it came from, the identifiers found
inside it, and generically ranked roles. They seed retrieval, form the
provisional obligation set, and anchor completeness — so extraction must
be deterministic and must never invent text the question does not
contain.
"""
from __future__ import annotations

from library.question_facets import QuestionFacet, extract_question_facets


def facet_texts(question, kind=None):
    facets = extract_question_facets(question)
    return [facet.exact_text for facet in facets
            if kind is None or facet.kind == kind]


class TestIdentifierExtraction:
    def test_backticked_dotted_camel_and_snake_identifiers_are_facets(self):
        question = ("How does `MergeIntoCommand.run` call "
                    "resolveReferences after preprocess_table is built?")

        facets = extract_question_facets(question)
        identifiers = {
            identifier for facet in facets
            for identifier in facet.identifiers}

        assert "MergeIntoCommand.run" in identifiers
        assert "resolveReferences" in identifiers
        assert "preprocess_table" in identifiers

    def test_spans_are_exact_offsets_into_the_question(self):
        question = "Where does RewriteMerge apply?"

        facets = extract_question_facets(question)
        rewrite = next(facet for facet in facets
                       if "RewriteMerge" in facet.identifiers)

        assert question[rewrite.start:rewrite.end] == rewrite.exact_text
        assert "RewriteMerge" in rewrite.exact_text

    def test_plain_prose_words_are_not_identifier_facets(self):
        facets = extract_question_facets(
            "Why is the table updated after the merge?")

        assert not [facet for facet in facets if facet.kind == "identifier"]


class TestComparisonAndOrdering:
    def test_comparison_sides_become_independent_facets(self):
        question = ("What is the difference between DeltaWriter and "
                    "ParquetWriter when rows are committed?")

        sides = facet_texts(question, kind="comparison-side")

        assert len(sides) == 2
        assert any("DeltaWriter" in side for side in sides)
        assert any("ParquetWriter" in side for side in sides)

    def test_versus_comparison_is_detected(self):
        question = "How does StreamingSink versus BatchSink handle commit?"

        sides = facet_texts(question, kind="comparison-side")

        assert len(sides) == 2

    def test_ordering_language_marks_a_sequence_role(self):
        facets = extract_question_facets(
            "What happens after validateSchema before writeFiles runs?")

        roles = {role for facet in facets for role in facet.roles}
        assert "sequence" in roles


class TestRolesAndDeterminism:
    def test_generic_verbs_rank_roles_without_domain_vocabulary(self):
        facets = extract_question_facets(
            "Where is CommitWriter registered and how does it decide to "
            "write the final rows?")

        roles = {role for facet in facets for role in facet.roles}
        assert "entry" in roles
        assert "decision" in roles
        assert "terminal" in roles

    def test_extraction_is_deterministic_with_stable_ids(self):
        question = ("How does `MergeIntoCommand.run` compare to "
                    "RewriteMerge after preprocess_table?")

        first = extract_question_facets(question)
        second = extract_question_facets(question)

        assert first == second
        assert [facet.id for facet in first] == [
            f"F{index}" for index in range(1, len(first) + 1)]

    def test_facets_never_paraphrase(self):
        question = "How does DeltaSink.addBatch delegate to TransactionLog?"

        for facet in extract_question_facets(question):
            assert facet.exact_text in question
            for identifier in facet.identifiers:
                assert identifier in question
