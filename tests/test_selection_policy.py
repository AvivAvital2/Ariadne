"""Tests for library.selection_policy — typed monotonic selection (B3).

Selection is monotonic promotion, not destructive intersection: a model
reply promotes candidates; obligation and connector bindings are
authoritative; every removal carries a reason; truncated or malformed
replies fail open instead of hard-deleting unnamed evidence.
"""
from __future__ import annotations

import dataclasses

import pytest

from library.selection_policy import (
    CapEvent,
    CompletionSignal,
    OccurrenceRef,
    SelectionDecision,
    classify_reply_tokens,
    resolve_selection,
    signal_from_usage,
    trailing_menu_token,
)


def make_occurrence(
    canonical_id: str,
    file: str = "src/pkg/owner.py",
    line_start: int = 10,
    line_end: int = 20,
    source_name: str = "src1",
    site: str = "",
) -> OccurrenceRef:
    return OccurrenceRef(
        source_name=source_name,
        canonical_id=canonical_id,
        file=file,
        line_start=line_start,
        line_end=line_end,
        site=site,
    )


OK_SIGNAL = CompletionSignal(
    finish_reason="stop", output_tokens=40, max_tokens=400)


class TestMonotonicPromotion:
    def test_model_selection_promotes_without_erasing_required(self):
        preferred = make_occurrence("pkg.Owner.chosen")
        obligation = make_occurrence("pkg.Owner.required_by_obligation")
        connector = make_occurrence("pkg.Bridge.connects")
        extra = make_occurrence("pkg.Other.unrelated")
        candidates = [preferred, obligation, connector, extra]

        decision = resolve_selection(
            candidates=candidates,
            model_preferred=[preferred],
            obligation_required={"O1": (obligation,)},
            connector_required=[connector],
            completion=OK_SIGNAL,
        )

        assert preferred in decision.retained
        assert obligation in decision.retained
        assert connector in decision.retained
        assert decision.completion_status == "ok"

    def test_partition_is_exact_and_every_discard_has_a_reason(self):
        retained_occurrence = make_occurrence("pkg.Owner.kept")
        reserve_occurrence = make_occurrence("pkg.Owner.reserved")
        dropped_occurrence = make_occurrence("pkg.Owner.dropped")
        candidates = [
            retained_occurrence, reserve_occurrence, dropped_occurrence]

        decision = resolve_selection(
            candidates=candidates,
            model_preferred=[retained_occurrence],
            obligation_required={},
            completion=OK_SIGNAL,
            reserve_limit=1,
        )

        assert decision.retained == (retained_occurrence,)
        assert decision.reserve == (reserve_occurrence,)
        assert dropped_occurrence in decision.discarded
        assert decision.discarded[dropped_occurrence].startswith("over-cap:")
        all_accounted = (
            set(decision.retained)
            | set(decision.reserve)
            | set(decision.discarded))
        assert all_accounted == set(candidates)

    def test_unselected_required_evidence_survives_with_binding_reason(self):
        named = make_occurrence("pkg.Owner.named_in_reply")
        unnamed_required = make_occurrence("pkg.Registrar.never_named")
        candidates = [named, unnamed_required]

        decision = resolve_selection(
            candidates=candidates,
            model_preferred=[named],
            obligation_required={"O-entry": (unnamed_required,)},
            completion=OK_SIGNAL,
        )

        assert unnamed_required in decision.retained
        assert unnamed_required not in decision.discarded
        assert decision.obligation_required == {
            "O-entry": (unnamed_required,)}


class TestTruncationDetection:
    def test_finish_reason_length_marks_truncated(self):
        signal = CompletionSignal(
            finish_reason="length", output_tokens=90, max_tokens=128)
        assert signal.truncated

    def test_output_at_cap_marks_truncated_even_when_status_ok(self):
        # The recorded failure shape: usage shows output == max, the
        # provider reply status still says ok, and the text stops mid-word.
        signal = CompletionSignal(
            finish_reason="stop", output_tokens=128, max_tokens=128)
        assert signal.truncated

    def test_truncated_reply_fails_open_despite_one_parseable_id(self):
        parsed = make_occurrence("pkg.Owner.first")
        unnamed = make_occurrence("pkg.Owner.cut_off_before_named")
        candidates = [parsed, unnamed]
        signal = CompletionSignal(
            finish_reason="stop", output_tokens=128, max_tokens=128)

        decision = resolve_selection(
            candidates=candidates,
            model_preferred=[parsed],
            obligation_required={},
            completion=signal,
            select_all_on_incomplete=True,
        )

        assert decision.completion_status == "truncated"
        assert set(decision.retained) == set(candidates)
        assert decision.discarded == {}

    def test_truncated_without_select_all_retains_required_and_reserve(self):
        parsed = make_occurrence("pkg.Owner.first")
        required = make_occurrence("pkg.Owner.required")
        remainder = make_occurrence("pkg.Owner.remainder")
        signal = CompletionSignal(
            finish_reason="length", output_tokens=128, max_tokens=128)

        decision = resolve_selection(
            candidates=[parsed, required, remainder],
            model_preferred=[parsed],
            obligation_required={"O1": (required,)},
            completion=signal,
            reserve_limit=4,
        )

        assert decision.completion_status == "truncated"
        assert required in decision.retained
        assert remainder in decision.reserve

    def test_empty_reply_is_incomplete_not_silently_empty(self):
        required = make_occurrence("pkg.Owner.required")
        other = make_occurrence("pkg.Owner.other")

        decision = resolve_selection(
            candidates=[required, other],
            model_preferred=[],
            obligation_required={"O1": (required,)},
            completion=OK_SIGNAL,
        )

        assert decision.completion_status == "empty"
        assert required in decision.retained
        assert other in decision.reserve

    def test_malformed_schema_is_incomplete(self):
        occurrence = make_occurrence("pkg.Owner.member")
        signal = CompletionSignal(
            finish_reason="stop", output_tokens=10, max_tokens=400,
            schema_valid=False)

        decision = resolve_selection(
            candidates=[occurrence],
            model_preferred=[occurrence],
            obligation_required={},
            completion=signal,
        )

        assert decision.completion_status == "malformed"


class TestReplyTokenClassification:
    def test_partial_trailing_token_is_detected_not_silently_dropped(self):
        known, unknown, partial = classify_reply_tokens(
            ["B1", "B2", "B"], valid_ids=["B1", "B2", "B3", "B10"])
        assert known == ("B1", "B2")
        assert unknown == ()
        assert partial == "B"

    def test_unknown_ids_are_recorded(self):
        known, unknown, partial = classify_reply_tokens(
            ["B1", "B99"], valid_ids=["B1", "B2"])
        assert known == ("B1",)
        assert unknown == ("B99",)
        assert partial is None

    def test_partial_trailing_token_marks_signal_incomplete(self):
        occurrence = make_occurrence("pkg.Owner.member")
        signal = CompletionSignal(
            finish_reason="stop", output_tokens=20, max_tokens=400,
            partial_trailing_token="B")

        decision = resolve_selection(
            candidates=[occurrence],
            model_preferred=[occurrence],
            obligation_required={},
            completion=signal,
        )

        assert decision.completion_status == "malformed"


class TestSignalFromUsage:
    def test_recorded_usage_row_builds_a_truncation_signal(self):
        signal = signal_from_usage(
            {"stop_reason": "end_turn", "output_tokens": 128,
             "input_tokens": 900},
            max_tokens=128)
        assert signal.truncated

    def test_provider_stop_reason_alone_marks_truncation(self):
        signal = signal_from_usage(
            {"stop_reason": "max_tokens", "output_tokens": 90},
            max_tokens=128)
        assert signal.truncated

    def test_normal_usage_is_not_truncated(self):
        signal = signal_from_usage(
            {"stop_reason": "end_turn", "output_tokens": 13}, max_tokens=128)
        assert not signal.truncated

    def test_row_request_cap_is_used_when_not_supplied(self):
        signal = signal_from_usage({"output_tokens": 64, "max_tokens": 64})
        assert signal.truncated

    def test_missing_usage_row_yields_an_unknowable_signal(self):
        signal = signal_from_usage(None, max_tokens=128)
        assert not signal.truncated
        assert signal.output_tokens is None


class TestTrailingMenuToken:
    def test_reply_cut_mid_label_is_detected(self):
        assert trailing_menu_token(
            "B7, B8, B", ["B1", "B7", "B8", "B12"]) == "B"

    def test_complete_reply_has_no_partial_token(self):
        assert trailing_menu_token("B7, B12", ["B7", "B12"]) is None

    def test_unknown_trailing_token_is_not_partial(self):
        assert trailing_menu_token("pick Z9", ["B1", "B2"]) is None

    def test_valid_label_that_prefixes_another_is_not_partial(self):
        # 'B1' is itself selectable; usage metadata, not reply text,
        # decides whether such an ambiguous reply was actually cut.
        assert trailing_menu_token("B7, B1", ["B1", "B7", "B12"]) is None

    def test_empty_reply_has_no_partial_token(self):
        assert trailing_menu_token("", ["B1"]) is None


class TestBudgetsAndCaps:
    def test_duplicate_occurrences_do_not_consume_budgets_twice(self):
        kept = make_occurrence("pkg.Owner.kept")
        duplicate = make_occurrence("pkg.Owner.kept")
        reserved = make_occurrence("pkg.Owner.reserved")

        decision = resolve_selection(
            candidates=[kept, duplicate, reserved],
            model_preferred=[kept],
            obligation_required={},
            completion=OK_SIGNAL,
            reserve_limit=1,
        )

        assert decision.retained.count(kept) == 1
        assert decision.reserve == (reserved,)

    def test_soft_cap_overflow_keeps_required_and_reports(self):
        required = [
            make_occurrence(f"pkg.Owner.required_{index}")
            for index in range(3)]

        decision = resolve_selection(
            candidates=required,
            model_preferred=[],
            obligation_required={"O1": tuple(required)},
            completion=OK_SIGNAL,
            retained_soft_cap=2,
        )

        assert set(required) <= set(decision.retained)
        assert decision.cap_events
        event = decision.cap_events[0]
        assert event.cap == "retained_soft_cap"
        assert event.limit == 2
        assert event.retained == 3

    def test_reserve_overflow_discard_reason_names_the_cap(self):
        candidates = [
            make_occurrence(f"pkg.Owner.member_{index}") for index in range(4)]

        decision = resolve_selection(
            candidates=candidates,
            model_preferred=[candidates[0]],
            obligation_required={},
            completion=OK_SIGNAL,
            reserve_limit=1,
        )

        assert decision.reserve == (candidates[1],)
        assert decision.discarded[candidates[2]] == "over-cap:reserve"
        assert decision.discarded[candidates[3]] == "over-cap:reserve"


class TestDeterminism:
    def test_same_inputs_produce_equal_decisions(self):
        candidates = [
            make_occurrence(f"pkg.Owner.member_{index}") for index in range(5)]
        kwargs = dict(
            candidates=candidates,
            model_preferred=[candidates[2]],
            obligation_required={"O1": (candidates[4],)},
            completion=OK_SIGNAL,
            reserve_limit=2,
        )

        assert resolve_selection(**kwargs) == resolve_selection(**kwargs)

    def test_decision_is_frozen(self):
        occurrence = make_occurrence("pkg.Owner.member")
        decision = resolve_selection(
            candidates=[occurrence],
            model_preferred=[occurrence],
            obligation_required={},
            completion=OK_SIGNAL,
        )
        with pytest.raises(dataclasses.FrozenInstanceError):
            decision.retained = ()

    def test_retained_preserves_candidate_menu_order(self):
        first = make_occurrence("pkg.Owner.first")
        second = make_occurrence("pkg.Owner.second")
        third = make_occurrence("pkg.Owner.third")

        decision = resolve_selection(
            candidates=[first, second, third],
            model_preferred=[third, first],
            obligation_required={"O1": (second,)},
            completion=OK_SIGNAL,
        )

        assert decision.retained == (first, second, third)
