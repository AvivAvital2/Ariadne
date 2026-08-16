"""Monotonic proof retention: required bodies are immune to optional noise.

Optional retrieval growth changed capped route retention, which changed
required-body derivation, which silently dropped a witness-bearing
overload body. The DefinitionBodyPlan fixes the invariant: required
bodies are derived deterministically at occurrence-extent identity
(overloads sharing one qualified name stay distinct), admitted before
any optional cap, and can only ever be missing as an explicit gap.
"""
from __future__ import annotations

from library.body_plan import (
    BodyRef,
    BodyRequirement,
    DefinitionBodyPlan,
    derive_definition_body_plan,
)
from library.chain_bundle import BundleHop
from library.structural_assembly import StructuralCitation

SOURCE = "src1"


def make_hop(
    qualified_name: str,
    *,
    file: str,
    line_start: int,
    line_end: int,
    relation: str = "calls",
    hop: int = 1,
    parent: str = "",
    call_site_file: str = "",
    call_site_line: int = 0,
    stop_reason: str = "descended",
) -> BundleHop:
    return BundleHop(citation=StructuralCitation(
        qualified_name=qualified_name,
        file=file,
        line_start=line_start,
        source_name=SOURCE,
        relation=relation,
        hop=hop,
        call_site_file=call_site_file,
        call_site_line=call_site_line,
        stop_reason=stop_reason,
        line_end=line_end,
        parent_qualified_name=parent,
    ))


def overload_fixture():
    """Two apply overloads; the transition site sits inside the second."""
    short_apply = make_hop(
        "pkg.Preprocess.apply", file="pkg/preprocess.scala",
        line_start=20, line_end=40)
    wide_apply = make_hop(
        "pkg.Preprocess.apply", file="pkg/preprocess.scala",
        line_start=50, line_end=242)
    generated = make_hop(
        "pkg.Generated.expression", file="pkg/generated.scala",
        line_start=5, line_end=30, parent="pkg.Preprocess.apply",
        call_site_file="pkg/preprocess.scala", call_site_line=150, hop=2)
    return short_apply, wide_apply, generated


def noise_hops(count: int):
    return [
        make_hop(f"pkg.Noise{index}.go", file=f"pkg/noise{index}.scala",
                 line_start=2, line_end=9,
                 stop_reason="obligation_continuation")
        for index in range(count)]


class TestRequiredDerivation:
    def test_transition_site_selects_the_narrowest_containing_overload(
            self):
        short_apply, wide_apply, generated = overload_fixture()

        plan = derive_definition_body_plan(
            hops=[short_apply, wide_apply, generated],
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=())

        required_extents = {
            (ref.file, ref.line_start, ref.line_end)
            for requirement in plan.required
            for ref in (requirement.body_ref,)}
        assert ("pkg/preprocess.scala", 50, 242) in required_extents
        assert ("pkg/preprocess.scala", 20, 40) not in required_extents

    def test_causal_callee_definition_is_required(self):
        short_apply, wide_apply, generated = overload_fixture()

        plan = derive_definition_body_plan(
            hops=[short_apply, wide_apply, generated],
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=())

        required_names = {
            requirement.body_ref.qualified_name
            for requirement in plan.required}
        assert "pkg.Generated.expression" in required_names

    def test_obligation_bound_occurrence_is_required(self):
        target = make_hop(
            "pkg.Registrar.install", file="pkg/registrar.scala",
            line_start=4, line_end=19, relation="localized")

        plan = derive_definition_body_plan(
            hops=[target],
            retained_symbols=("pkg.Registrar.install",),
            bindings=((1, "pkg.Registrar.install"),))

        assert any(
            requirement.body_ref.qualified_name == "pkg.Registrar.install"
            and "obligation" in requirement.reason
            for requirement in plan.required)

    def test_continuation_only_hops_stay_optional(self):
        continuation = make_hop(
            "pkg.Extra.branch", file="pkg/extra.scala",
            line_start=3, line_end=9,
            stop_reason="obligation_continuation")

        plan = derive_definition_body_plan(
            hops=[continuation],
            retained_symbols=("pkg.Extra.branch",),
            bindings=())

        required_names = {
            requirement.body_ref.qualified_name
            for requirement in plan.required}
        assert "pkg.Extra.branch" not in required_names
        optional_names = {ref.qualified_name for ref in plan.optional}
        assert "pkg.Extra.branch" in optional_names

    def test_continuation_inside_a_retained_transition_is_required(self):
        continuation_caller = make_hop(
            "pkg.Cont.caller", file="pkg/cont.scala",
            line_start=10, line_end=60,
            stop_reason="obligation_continuation")
        callee = make_hop(
            "pkg.Cont.callee", file="pkg/leaf.scala",
            line_start=2, line_end=9, parent="pkg.Cont.caller",
            call_site_file="pkg/cont.scala", call_site_line=30, hop=2)

        plan = derive_definition_body_plan(
            hops=[continuation_caller, callee],
            retained_symbols=("pkg.Cont.caller", "pkg.Cont.callee"),
            bindings=())

        required_names = {
            requirement.body_ref.qualified_name
            for requirement in plan.required}
        assert "pkg.Cont.caller" in required_names


class TestMonotonicity:
    def test_a_hundred_unrelated_candidates_cannot_evict_required_proof(
            self):
        short_apply, wide_apply, generated = overload_fixture()
        base = [short_apply, wide_apply, generated]

        quiet = derive_definition_body_plan(
            hops=base,
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=())
        noisy = derive_definition_body_plan(
            hops=base + noise_hops(100),
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=())

        assert quiet.required == noisy.required
        selected_extents = {
            (ref.file, ref.line_start, ref.line_end)
            for ref in noisy.selected}
        for requirement in quiet.required:
            ref = requirement.body_ref
            assert (ref.file, ref.line_start, ref.line_end) in (
                selected_extents)

    def test_reordered_candidates_leave_the_plan_byte_identical(self):
        short_apply, wide_apply, generated = overload_fixture()
        forward = [short_apply, wide_apply, generated] + noise_hops(10)

        first = derive_definition_body_plan(
            hops=forward,
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=())
        second = derive_definition_body_plan(
            hops=list(reversed(forward)),
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=())

        assert first == second

    def test_optional_candidates_never_consume_required_capacity(self):
        short_apply, wide_apply, generated = overload_fixture()

        plan = derive_definition_body_plan(
            hops=[short_apply, wide_apply, generated] + noise_hops(50),
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=(),
            optional_cap=2)

        required_extents = {
            (requirement.body_ref.file, requirement.body_ref.line_start)
            for requirement in plan.required}
        selected_extents = {
            (ref.file, ref.line_start) for ref in plan.selected}
        assert required_extents <= selected_extents
        optional_selected = selected_extents - required_extents
        assert len(optional_selected) <= 2
        assert plan.discarded
        assert all(reason for _ref, reason in plan.discarded)

    def test_required_overflow_is_an_explicit_gap_never_silent_loss(self):
        short_apply, wide_apply, generated = overload_fixture()

        plan = derive_definition_body_plan(
            hops=[short_apply, wide_apply, generated],
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=(),
            required_cap=1)

        assert plan.cap_events
        assert plan.gaps
        selected_count = len(plan.selected)
        required_count = len({
            (r.body_ref.file, r.body_ref.line_start)
            for r in plan.required})
        assert selected_count + len(plan.gaps) >= required_count

    def test_overloads_sharing_a_qualified_name_stay_distinct(self):
        short_apply, wide_apply, generated = overload_fixture()

        plan = derive_definition_body_plan(
            hops=[short_apply, wide_apply, generated],
            retained_symbols=("pkg.Preprocess.apply",
                              "pkg.Generated.expression"),
            bindings=((1, "pkg.Preprocess.apply"),))

        apply_refs = {
            (ref.file, ref.line_start, ref.line_end)
            for ref in (*[r.body_ref for r in plan.required],
                        *plan.optional)
            if ref.qualified_name == "pkg.Preprocess.apply"}
        assert ("pkg/preprocess.scala", 20, 40) in apply_refs
        assert ("pkg/preprocess.scala", 50, 242) in apply_refs


class TestCallerRecovery:
    def test_caller_dropped_by_route_caps_is_still_required(self):
        # The q8 shape: the callee stays retained, the caller name fell
        # out of capped route retention, the transition site sits inside
        # the caller's extent. The caller body is mandatory regardless.
        wide_apply = make_hop(
            "pkg.Preprocess.apply", file="pkg/preprocess.scala",
            line_start=50, line_end=242)
        resolved = make_hop(
            "pkg.Preprocess.resolveColumns", file="pkg/preprocess.scala",
            line_start=406, line_end=467, parent="pkg.Preprocess.apply",
            call_site_file="pkg/preprocess.scala", call_site_line=170,
            hop=2)

        plan = derive_definition_body_plan(
            hops=[wide_apply, resolved],
            retained_symbols=("pkg.Preprocess.resolveColumns",),
            bindings=())

        required = {
            (r.body_ref.qualified_name, r.body_ref.line_start,
             r.body_ref.line_end)
            for r in plan.required}
        assert ("pkg.Preprocess.apply", 50, 242) in required
