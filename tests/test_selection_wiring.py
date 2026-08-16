"""The ask pipeline threads provider usage into selection resolution.

Behavioral guarantees live in tests/test_obligation_retention.py against
the pure chain_menu functions; these pins assert the service actually
routes each selection phase through the guarded resolvers with the
recorded usage signal, following the source-order convention of
tests/test_ask_synthesis_config.py.
"""
from __future__ import annotations

from pathlib import Path

SOURCE = (
    Path(__file__).resolve().parents[1]
    / "ariadne_mcp" / "service_analysis.py").read_text()


def test_component_scope_is_guarded_with_usage_signal():
    resolve = SOURCE.index("resolve_component_selection(components, reply)")
    guarded = SOURCE.index("guarded_component_scope(")
    assert resolve < guarded
    assert "usage_sink=_component_usage" in SOURCE


def test_component_decision_is_recorded():
    assert '_post_walk["component_decision"]' in SOURCE


def test_exact_route_selection_is_guarded():
    assert "selection = guarded_route_selection(" in SOURCE
    assert "usage_sink=_route_usage" in SOURCE
    assert ("selection = resolve_obligation_route_selection("
            "menu, route_reply)") not in SOURCE


def test_body_selection_is_guarded_and_reply_recorded():
    assert "guarded_definition_body_selection(" in SOURCE
    assert "usage_sink=_body_usage" in SOURCE
    assert '_post_walk["body_reply"]' in SOURCE


def test_obligation_targets_survive_scoping_and_graph_closure():
    assert "retain_obligation_target_occurrences(" in SOURCE
    assert SOURCE.count("required_symbols=") >= 2
    assert "_required_body_symbols = obligation_definition_body_symbols(" in (
        SOURCE)


def test_obligation_bindings_are_built_from_selected_clews():
    assert "_obligation_route_targets = tuple(dict.fromkeys(" in SOURCE
def test_proof_appendix_is_immune_to_repair_deletion():
    filter_call = SOURCE.index("answer, ledger = filter_supported(")
    append = SOURCE.index("answer += _proof_appendix")
    assert filter_call < append
    assert "ledger = validate_claims(answer, _formulation_evidence)" in (
        SOURCE)


def test_formulation_completeness_is_obligation_aware():
    assert "assess_obligation_coverage(" in SOURCE
    assert "required_symbols=_required_route_symbols" in SOURCE
def test_required_bodies_derive_from_the_monotonic_plan():
    # The plan is computed and recorded as diagnostics; bare qualified
    # names from it must never broaden materialization.
    assert "derive_definition_body_plan(" in SOURCE
    assert '_post_walk["body_plan"]' in SOURCE
    assert ("selected_body_symbols = list(_body_selection.symbols)"
            in SOURCE)
