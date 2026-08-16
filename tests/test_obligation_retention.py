"""Obligation retention nets: model selection promotes, never erases.

Route cards are a compact selection surface, not an authority boundary.
An obligation-bound target survives component scoping, route selection,
and definition-body selection even when the model reply never names it —
and token overlap with the question may rank alternatives but never
gates retention. A truncated or partially parseable reply fails open
instead of hard-deleting unnamed evidence.
"""
from __future__ import annotations

from library.chain_bundle import BundleHop
from library.chain_menu import (
    ComponentMenu,
    DefinitionBodyMenu,
    DefinitionBodySelection,
    RouteMenu,
    Selection,
    _occurrence_key,
    complete_definition_body_selection,
    evidence_graph_for,
    guarded_component_scope,
    guarded_definition_body_selection,
    guarded_route_selection,
    obligation_definition_body_symbols,
    retain_obligation_routes,
    retain_obligation_target_occurrences,
)
from library.selection_policy import CompletionSignal
from library.structural_assembly import StructuralCitation

OK_SIGNAL = CompletionSignal(
    finish_reason="end_turn", output_tokens=20, max_tokens=256)
CUT_SIGNAL = CompletionSignal(
    finish_reason="end_turn", output_tokens=256, max_tokens=256)


def make_hop(
    qualified_name: str,
    *,
    file: str = "src/pkg/pipeline.py",
    line_start: int = 10,
    line_end: int = 20,
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
        source_name="src1",
        relation=relation,
        hop=hop,
        call_site_file=call_site_file,
        call_site_line=call_site_line,
        stop_reason=stop_reason,
        line_end=line_end,
        parent_qualified_name=parent,
    ))


def make_route_menu() -> RouteMenu:
    routes = {
        "R1": ("pkg.Pipeline.run", "pkg.Step.apply"),
        "R2": ("pkg.Registrar.register", "pkg.Rule.apply"),
        "R3": ("pkg.Rule.apply",),
    }
    route_occurrences = {
        "R1": (("pkg.Pipeline.run", "src/pkg/pipeline.py", 10),
               ("pkg.Step.apply", "src/pkg/step.py", 5)),
        "R2": (("pkg.Registrar.register", "src/pkg/registrar.py", 3),
               ("pkg.Rule.apply", "src/pkg/rule.py", 8)),
        "R3": (("pkg.Rule.apply", "src/pkg/rule.py", 8),),
    }
    return RouteMenu(
        text="",
        routes=routes,
        sections={},
        mandatory_symbols=(),
        route_occurrences=route_occurrences,
        route_sections={},
        section_titles={},
        route_summaries={label: "" for label in routes},
    )


class TestRetainObligationRoutes:
    def test_uncovered_obligation_target_gets_its_connected_route(self):
        menu = make_route_menu()
        selection = Selection(route_ids=("R1",))

        retained = retain_obligation_routes(
            menu, selection, ((1, "pkg.Registrar.register"),),
            question="how does the pipeline run steps?",
            obligations="C1: prove where rules are registered")

        assert retained.route_ids == ("R1", "R2")
        assert "pkg.Registrar.register" in retained.symbols
        assert ("pkg.Registrar.register", "src/pkg/registrar.py", 3) in (
            retained.occurrence_keys)

    def test_token_overlap_never_gates_retention(self):
        # The obligation target shares no vocabulary with the question or
        # the plan; binding is authority, overlap is only ranking.
        menu = make_route_menu()
        selection = Selection(route_ids=("R1",))

        retained = retain_obligation_routes(
            menu, selection, ((1, "pkg.Registrar.register"),),
            question="why is the merge preprocessed twice?",
            obligations="C1: unrelated wording entirely")

        assert "R2" in retained.route_ids

    def test_singleton_route_cards_cannot_widen_the_surface(self):
        menu = make_route_menu()
        selection = Selection(route_ids=("R1",))

        retained = retain_obligation_routes(
            menu, selection, ((1, "pkg.OnlyOnSingleton.node"),),
            question="anything", obligations="")

        assert retained.route_ids == ("R1",)

    def test_covered_targets_add_nothing(self):
        menu = make_route_menu()
        selection = Selection(
            route_ids=("R1", "R2"),
            symbols=["pkg.Registrar.register"])

        retained = retain_obligation_routes(
            menu, selection, ((1, "pkg.Registrar.register"),),
            question="anything", obligations="")

        assert retained.route_ids == ("R1", "R2")

    def test_addition_is_bounded_per_obligation(self):
        routes = {
            "R1": ("pkg.Pipeline.run", "pkg.Step.apply"),
            "R2": ("pkg.Registrar.register", "pkg.Rule.apply"),
            "R3": ("pkg.Registrar.register", "pkg.Catalog.load"),
        }
        menu = RouteMenu(
            text="", routes=routes, sections={}, mandatory_symbols=(),
            route_occurrences={label: () for label in routes},
            route_sections={}, section_titles={},
            route_summaries={label: "" for label in routes})
        selection = Selection(route_ids=("R1",))

        retained = retain_obligation_routes(
            menu, selection,
            ((1, "pkg.Registrar.register"), (1, "pkg.Catalog.load")),
            question="anything", obligations="", max_per_obligation=1)

        added = set(retained.route_ids) - {"R1"}
        assert len(added) == 1

    def test_rank_prefers_more_targets_then_shorter_then_lower_label(self):
        routes = {
            "R1": ("pkg.Pipeline.run", "pkg.Step.apply"),
            "R4": ("pkg.Registrar.register", "pkg.Rule.apply",
                   "pkg.Catalog.load"),
            "R5": ("pkg.Registrar.register", "pkg.Rule.apply"),
        }
        menu = RouteMenu(
            text="", routes=routes, sections={}, mandatory_symbols=(),
            route_occurrences={label: () for label in routes},
            route_sections={}, section_titles={},
            route_summaries={label: "" for label in routes})
        selection = Selection(route_ids=("R1",))

        retained = retain_obligation_routes(
            menu, selection, ((1, "pkg.Registrar.register"),),
            question="zz nothing shared", obligations="")

        assert "R5" in retained.route_ids
        assert "R4" not in retained.route_ids


class TestRetainObligationTargetOccurrences:
    def test_disconnected_singleton_target_keeps_its_exact_occurrence(self):
        run = make_hop("pkg.Pipeline.run", line_start=10, line_end=30)
        step = make_hop(
            "pkg.Step.apply", file="src/pkg/step.py", line_start=5,
            line_end=15, parent="pkg.Pipeline.run",
            call_site_file="src/pkg/pipeline.py", call_site_line=12, hop=2)
        registrar = make_hop(
            "pkg.Registrar.register", file="src/pkg/registrar.py",
            line_start=3, line_end=9, relation="localized", hop=1)
        hops = [run, step, registrar]
        graph = evidence_graph_for(hops)
        selection = Selection(
            symbols=["pkg.Pipeline.run", "pkg.Step.apply"],
            occurrence_keys=(_occurrence_key(run), _occurrence_key(step)))

        retained = retain_obligation_target_occurrences(
            graph, selection, ((1, "pkg.Registrar.register"),))

        assert "pkg.Registrar.register" in retained.symbols
        assert _occurrence_key(registrar) in retained.occurrence_keys

    def test_no_bindings_is_a_strict_no_op(self):
        run = make_hop("pkg.Pipeline.run")
        graph = evidence_graph_for([run])
        selection = Selection(
            symbols=["pkg.Pipeline.run"],
            occurrence_keys=(_occurrence_key(run),))

        retained = retain_obligation_target_occurrences(graph, selection, ())

        assert retained is selection

    def test_bindings_are_bounded_per_obligation(self):
        hops = [
            make_hop(f"pkg.Owner.member_{index}", line_start=index * 10 + 1,
                     line_end=index * 10 + 5)
            for index in range(12)]
        graph = evidence_graph_for(hops)
        selection = Selection()
        bindings = tuple(
            (1, f"pkg.Owner.member_{index}") for index in range(12))

        retained = retain_obligation_target_occurrences(
            graph, selection, bindings, max_per_obligation=8)

        assert len(retained.symbols) == 8


class TestDefinitionBodyNets:
    def test_obligation_bodies_join_required_bodies_in_the_selection(self):
        menu = DefinitionBodyMenu(
            text="",
            symbols={"B1": "pkg.A.f", "B2": "pkg.B.g", "B3": "pkg.C.h"},
            required_symbols=("pkg.A.f",))
        selection = DefinitionBodySelection(symbols=("pkg.B.g",))

        completed = complete_definition_body_selection(
            menu, selection, required_symbols=("pkg.C.h",))

        assert completed.symbols == ("pkg.B.g", "pkg.A.f", "pkg.C.h")

    def test_names_outside_the_menu_cannot_be_injected(self):
        menu = DefinitionBodyMenu(
            text="", symbols={"B1": "pkg.A.f"},
            required_symbols=("pkg.Ghost.h",))
        selection = DefinitionBodySelection(
            symbols=("pkg.A.f",), unknown=("B9",))

        completed = complete_definition_body_selection(
            menu, selection, required_symbols=("pkg.Absent.k",))

        assert completed.symbols == ("pkg.A.f",)
        assert completed.unknown == ("B9",)

    def test_obligation_body_symbols_are_menu_scoped_and_bounded(self):
        menu = DefinitionBodyMenu(
            text="",
            symbols={
                f"B{index}": f"pkg.Owner.member_{index}"
                for index in range(1, 9)},
            required_symbols=())
        bindings = tuple(
            (1, f"pkg.Owner.member_{index}") for index in range(1, 7)
        ) + ((2, "pkg.Owner.member_7"), (2, "pkg.Ghost.member"))

        symbols = obligation_definition_body_symbols(
            menu, bindings, max_per_obligation=4)

        assert symbols == (
            "pkg.Owner.member_1", "pkg.Owner.member_2",
            "pkg.Owner.member_3", "pkg.Owner.member_4",
            "pkg.Owner.member_7")


class TestGuardedDefinitionBodySelection:
    def make_body_menu(self) -> DefinitionBodyMenu:
        return DefinitionBodyMenu(
            text="",
            symbols={"B1": "pkg.A.f", "B2": "pkg.B.g", "B3": "pkg.C.h"},
            required_symbols=("pkg.A.f",))

    def test_complete_reply_keeps_model_choice_plus_required(self):
        selection = guarded_definition_body_selection(
            self.make_body_menu(), "B2", completion=OK_SIGNAL,
            required_symbols=("pkg.C.h",))

        assert selection.symbols == ("pkg.B.g", "pkg.A.f", "pkg.C.h")

    def test_truncated_reply_fails_open_to_every_card(self):
        selection = guarded_definition_body_selection(
            self.make_body_menu(), "B2", completion=CUT_SIGNAL)

        assert set(selection.symbols) == {"pkg.A.f", "pkg.B.g", "pkg.C.h"}

    def test_reply_cut_mid_label_fails_open(self):
        selection = guarded_definition_body_selection(
            self.make_body_menu(), "B2, B", completion=OK_SIGNAL)

        assert set(selection.symbols) == {"pkg.A.f", "pkg.B.g", "pkg.C.h"}

    def test_empty_reply_still_selects_everything(self):
        selection = guarded_definition_body_selection(
            self.make_body_menu(), "", completion=None)

        assert set(selection.symbols) == {"pkg.A.f", "pkg.B.g", "pkg.C.h"}


class TestGuardedRouteSelection:
    def test_truncated_route_reply_fails_open_to_all_scoped_routes(self):
        selection = guarded_route_selection(
            make_route_menu(), "C1: R1", (), question="q",
            completion=CUT_SIGNAL)

        assert set(selection.route_ids) == {"R1", "R2", "R3"}

    def test_complete_reply_keeps_model_routes_plus_bound_routes(self):
        selection = guarded_route_selection(
            make_route_menu(), "C1: R1", ((1, "pkg.Registrar.register"),),
            question="unrelated wording", completion=OK_SIGNAL)

        assert set(selection.route_ids) == {"R1", "R2"}

    def test_reply_cut_mid_label_fails_open(self):
        selection = guarded_route_selection(
            make_route_menu(), "C1: R1\nC2: R", (), question="q",
            completion=OK_SIGNAL)

        assert set(selection.route_ids) == {"R1", "R2", "R3"}


class TestGuardedComponentScope:
    def test_bound_route_survives_component_scoping(self):
        components = ComponentMenu(
            text="", components={"G1": ("n1",), "G2": ("n2",)},
            routes={"G1": ("R1",), "G2": ("R2",)})

        scoped, decision = guarded_component_scope(
            make_route_menu(), components, "C1: G1",
            ((1, "pkg.Registrar.register"),),
            question="unrelated", completion=OK_SIGNAL)

        assert set(scoped.routes) == {"R1", "R2"}
        assert decision["outcome"] == "component-scope-applied"
        assert decision["dropped_route_ids"] == ("R3",)
        assert decision["retained_route_ids"] == ("R1", "R2")

    def test_truncated_component_reply_skips_scoping(self):
        components = ComponentMenu(
            text="", components={"G1": ("n1",)}, routes={"G1": ("R1",)})
        menu = make_route_menu()

        scoped, decision = guarded_component_scope(
            menu, components, "C1: G1", (), question="q",
            completion=CUT_SIGNAL)

        assert scoped is menu
        assert decision["outcome"] == "component-scope-skipped:incomplete"

    def test_component_reply_cut_mid_label_skips_scoping(self):
        components = ComponentMenu(
            text="", components={"G1": ("n1",), "G2": ("n2",)},
            routes={"G1": ("R1",), "G2": ("R2",)})
        menu = make_route_menu()

        scoped, decision = guarded_component_scope(
            menu, components, "C1: G", (), question="q",
            completion=OK_SIGNAL)

        assert scoped is menu
        assert decision["outcome"] == "component-scope-skipped:incomplete"

    def test_empty_component_reply_leaves_the_menu_untouched(self):
        components = ComponentMenu(text="", components={}, routes={})
        menu = make_route_menu()

        scoped, decision = guarded_component_scope(
            menu, components, "", (), question="q", completion=None)

        assert scoped is menu
        assert decision["outcome"] == "component-scope-skipped:empty"


class TestReplyOrderIndependence:
    def test_reordered_route_replies_resolve_identically(self):
        forward = guarded_route_selection(
            make_route_menu(), "C1: R1 R2", (), question="q",
            completion=OK_SIGNAL)
        backward = guarded_route_selection(
            make_route_menu(), "C1: R2 R1", (), question="q",
            completion=OK_SIGNAL)

        assert set(forward.route_ids) == set(backward.route_ids)
        assert set(forward.symbols) == set(backward.symbols)

    def test_duplicated_route_ids_do_not_double_retain(self):
        selection = guarded_route_selection(
            make_route_menu(), "C1: R1 R1 R1", (), question="q",
            completion=OK_SIGNAL)

        assert list(selection.route_ids).count("R1") == 1
