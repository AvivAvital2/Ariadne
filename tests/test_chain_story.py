"""Evidence-IR and route-local section behavior."""
import pytest

from library.chain_bundle import BundleHop
from library.chain_menu import Fetched, RouteMenu, Selection, select_route_sections, _occurrence_key, complete_selection_with_body_dependencies
from library.chain_story import (
    build_story_ir, expand_story_placeholders, render_story_evidence, render_formulation_spine)
from library.source_materialization import SourceExcerpt
from library.structural_assembly import StructuralCitation


def hop(name, *, parent="", line=1, call_line=10):
    return BundleHop(
        citation=StructuralCitation(
            qualified_name=name, file="app.py", line_start=line, line_end=line + 2,
            source_name="repo", relation="calls", hop=1,
            parent_qualified_name=parent, call_site_file="app.py",
            call_site_line=call_line),
        document_id=f"doc-{name}",
        source_excerpts=(SourceExcerpt(
            source_name="repo", file="app.py", line_start=line, line_end=line,
            kind="definition", content=f"def {name.rsplit('.', 1)[-1]}(): pass",
            sha256="abc"),))


def test_selected_routes_choose_only_question_relevant_local_sections():
    menu = RouteMenu(
        routes={"R1": ("api.create", "service.persist")},
        sections={"S1": ("doc", 0), "S2": ("doc", 1), "S3": ("doc", 2)},
        route_sections={"R1": ("S1", "S2", "S3")})

    selected = select_route_sections(
        menu, Selection(route_ids=("R1",)),
        "How does create persist the request?", max_sections=2,
        section_titles={"S1": "Authentication overview",
                        "S2": "Persisting create requests",
                        "S3": "Monitoring dashboards"})

    assert selected.section_ids == ("S2",)
    assert selected.sections == [("doc", 1)]


def test_story_ir_contains_only_selected_occurrences_and_supported_edges():
    first = hop("api.create", line=2)
    second = hop("service.persist", parent="api.create", line=20, call_line=5)
    ignored = hop("metrics.emit", line=40)
    selection = Selection(
        symbols=["api.create", "service.persist"], route_ids=("R1",),
        occurrence_keys=((first.citation.qualified_name, first.citation.file,
                          first.citation.line_start, first.citation.line_end,
                          first.citation.parent_qualified_name,
                          first.citation.call_site_file, first.citation.call_site_line,
                          first.citation.relation, first.citation.hop,
                          first.citation.stop_reason),
                         (second.citation.qualified_name, second.citation.file,
                          second.citation.line_start, second.citation.line_end,
                          second.citation.parent_qualified_name,
                          second.citation.call_site_file, second.citation.call_site_line,
                          second.citation.relation, second.citation.hop,
                          second.citation.stop_reason)))
    fetched = Fetched(
        definitions={"api.create": "Receives create requests.",
                     "service.persist": "Persists the object."},
        sections=[("Create flow", "Persistence", "Writes the converted object.")])

    story = build_story_ir([first, second, ignored], selection, fetched)
    rendered = render_story_evidence(story)

    assert tuple(node.symbol for node in story.nodes) == ("api.create", "service.persist")
    assert len(story.edges) == 1
    assert "metrics.emit" not in rendered
    assert "{{N1}}" in rendered and "{{E1}}" in rendered
    assert "Persistence" in rendered


def test_placeholder_expansion_is_deterministic_and_rejects_unknown_ids():
    first = hop("api.create", line=2)
    second = hop("service.persist", parent="api.create", line=20, call_line=5)
    selection = Selection(symbols=["api.create", "service.persist"])
    story = build_story_ir([first, second], selection, Fetched())

    expanded = expand_story_placeholders("{{N1}} {{E1}} {{N2}}.", story)

    assert "api.create (app.py:2)" in expanded
    assert "calls at app.py:5" in expanded
    assert "service.persist (app.py:20)" in expanded
    with pytest.raises(ValueError, match="unknown evidence placeholder"):
        expand_story_placeholders("{{N99}}", story)
def test_route_scope_keeps_relevant_families_and_only_their_sections():
    from library.chain_menu import scope_route_menu

    routes = {f"R{i}": (f"noise.Metric{i}.emit",) for i in range(1, 40)}
    routes["R40"] = ("delta.ClassicMergeExecutor.run", "delta.MergeRows.join")
    menu = RouteMenu(
        routes=routes,
        sections={"S1": ("noise-doc", 0), "S2": ("merge-doc", 0)},
        route_sections={
            **{f"R{i}": ("S1",) for i in range(1, 40)},
            "R40": ("S2",)},
        section_titles={"S1": "Metrics dashboard", "S2": "Merge cardinality join"},
        route_occurrences={label: ((label,),) for label in routes})

    scoped = scope_route_menu(
        menu, "Where does ClassicMergeExecutor enforce merge cardinality in the join?")

    assert tuple(scoped.routes) == ("R40",)
    assert tuple(scoped.sections) == ("S2",)
    assert "ClassicMergeExecutor" in scoped.text
    assert "Metrics dashboard" not in scoped.text
def test_story_ir_uses_compact_catalog_description_not_full_document():
    selected = hop("service.persist", line=20)
    selection = Selection(symbols=["service.persist"])
    fetched = Fetched(definitions={
        "service.persist": (
            "scala_method service.persist in service [scala] app.py:20-40\n"
            "Description: Persists the converted request.\n"
            "Long generated implementation discussion that is not a selected section.")})

    rendered = render_story_evidence(build_story_ir([selected], selection, fetched))

    assert "Persists the converted request." in rendered
    assert "Long generated implementation discussion" not in rendered
def test_route_scope_reserves_every_exact_question_symbol_before_ranked_fill():
    from library.chain_menu import scope_route_menu

    routes = {
        **{f"R{i}": (f"noise.MergePlanner{i}.merge",) for i in range(1, 40)},
        "R40": ("delta.ClassicMergeExecutor.writeAllChanges",),
        "R41": ("spark.MergeRowsExec.processPartition",),
    }
    menu = RouteMenu(
        routes=routes,
        route_occurrences={label: ((label,),) for label in routes})

    scoped = scope_route_menu(
        menu,
        "Compare ClassicMergeExecutor row classification with MergeRowsExec.",
        max_families=1)

    retained = " ".join(name for route in scoped.routes.values() for name in route)
    assert "ClassicMergeExecutor" in retained
    assert "MergeRowsExec" in retained


def test_bare_story_ids_expand_as_well_as_braced_ids():
    first = hop("api.create", line=2)
    second = hop("service.persist", parent="api.create", line=20, call_line=5)
    story = build_story_ir(
        [first, second], Selection(symbols=["api.create", "service.persist"]), Fetched())

    expanded = expand_story_placeholders("N1 E1 N2; {{N1}}.", story)

    assert "N1" not in expanded and "E1" not in expanded and "N2" not in expanded
    assert expanded.count("api.create (app.py:2)") == 2
def test_route_scope_preserves_meaningful_components_of_dotted_question_ids():
    from library.chain_menu import scope_route_menu
    menu = RouteMenu(routes={
        "R1": ("delta.DeltaSink.addBatchWithStatusImpl", "delta.DataFrame"),
        "R2": ("delta.DeltaSink.addBatchWithStatusImpl",
               "delta.OptimisticTransactionImpl.txnVersion"),
    })

    scoped = scope_route_menu(
        menu, "compare txn.txnVersion(queryId) against batchId", max_families=1)

    assert "R2" in scoped.routes
def test_route_scope_uses_route_local_section_embedding_before_selector():
    from library.chain_menu import scope_route_menu
    menu = RouteMenu(routes={
        "R1": ("noise.Clock.tick",),
        "R2": ("delta.OptimisticTransactionImpl.txnVersion",),
    })

    scoped = scope_route_menu(
        menu, "How does the sink know the replay was already committed?",
        max_families=1, route_scores={"R1": 0.05, "R2": 0.91})

    assert tuple(scoped.routes) == ("R2",)
def test_route_scope_limits_one_root_from_monopolizing_family_menu():
    from library.chain_menu import scope_route_menu
    routes = {
        **{f"R{i}": ("spark.RewriteMerge.apply", f"spark.Noise{i}.run")
           for i in range(1, 10)},
        "R10": ("delta.ClassicMergeExecutor.writeAllChanges",),
    }
    menu = RouteMenu(routes=routes)
    scores = {label: 0.9 - index / 100 for index, label in enumerate(routes)}

    scoped = scope_route_menu(
        menu, "How do two merge implementations differ?", max_families=5,
        route_scores=scores, max_per_root=2)

    assert "R10" in scoped.routes
    rewrite_routes = [route for route in scoped.routes.values()
                      if route[0] == "spark.RewriteMerge.apply"]
    assert len(rewrite_routes) <= 4


def test_embedding_section_choice_replaces_unbounded_selector_sections():
    menu = RouteMenu(
        routes={"R1": ("service.persist",)},
        sections={f"S{i}": ("doc", i) for i in range(10)},
        route_sections={"R1": tuple(f"S{i}" for i in range(10))},
        section_titles={f"S{i}": f"Section {i}" for i in range(10)})
    initial = Selection(
        route_ids=("R1",), sections=[("doc", i) for i in range(10)],
        section_ids=tuple(f"S{i}" for i in range(10)))

    selected = select_route_sections(
        menu, initial, "persist", max_sections=1, section_scores={"S2": 0.9})

    assert selected.section_ids == ("S2",)
    assert selected.sections == [("doc", 2)]


def test_non_strict_placeholder_expansion_marks_unknown_edge_without_aborting():
    first = hop("api.create", line=2)
    story = build_story_ir([first], Selection(symbols=["api.create"]), Fetched())

    expanded = expand_story_placeholders("N1 then E15", story, strict=False)

    assert "api.create (app.py:2)" in expanded
    assert "[unsupported evidence E15]" in expanded
def test_route_scope_reserves_terminal_owner_even_when_entry_owner_is_shared():
    from library.chain_menu import scope_route_menu
    menu = RouteMenu(routes={
        "R1": ("delta.MergeCommand.run", "delta.NoiseOne.emit"),
        "R2": ("delta.MergeCommand.run", "delta.NoiseTwo.emit"),
        "R3": ("delta.MergeCommand.run", "delta.ClassicMergeExecutor.writeAllChanges"),
    })

    scoped = scope_route_menu(
        menu, "How is merge output produced?", max_families=3, max_per_root=2,
        route_scores={"R1": 0.99, "R2": 0.98, "R3": 0.97})

    assert "R3" in scoped.routes
def test_module_menu_keeps_every_route_reachable_by_entry_or_terminal_owner():
    from library.chain_menu import module_menu_for
    route_menu = RouteMenu(
        routes={
            "R1": ("delta.MergeCommand.run", "delta.ClassicMergeExecutor.writeAllChanges"),
            "R2": ("spark.MergeRowsExec.processPartition", "spark.InternalRow"),
        },
        route_sections={"R1": ("S1",), "R2": ("S2",)},
        section_titles={"S1": "Classic merge execution",
                        "S2": "Merge rows instruction evaluation"})

    modules = module_menu_for(route_menu)

    reachable = {route_id for route_ids in modules.modules.values()
                 for route_id in route_ids}
    assert reachable == {"R1", "R2"}
    assert "ClassicMergeExecutor" in modules.text
    assert "MergeRowsExec" in modules.text
    assert "Classic merge execution" in modules.text


def test_module_selection_expands_only_selected_modules_back_to_routes():
    from library.chain_menu import (
        module_menu_for, resolve_module_selection, routes_for_modules)
    route_menu = RouteMenu(routes={
        "R1": ("delta.DeltaSink.addBatch", "delta.OptimisticTransaction.txnVersion"),
        "R2": ("metrics.Registry.emit",),
    })
    modules = module_menu_for(route_menu)
    target = next(label for label, owner in modules.owners.items()
                  if owner == "OptimisticTransaction")

    route_ids = resolve_module_selection(modules, target, route_menu)
    expanded = routes_for_modules(route_menu, route_ids)

    assert tuple(expanded.routes) == ("R1",)
    assert "R2" not in expanded.text
def test_module_menu_exposes_middle_owners_that_are_required_clews():
    from library.chain_menu import RouteMenu, module_menu_for, resolve_module_selection
    route_menu = RouteMenu(routes={
        "R1": (
            "spark.SQLExecution.run",
            "delta.PreprocessTableMerge.apply",
            "delta.MergeIntoCommand.run"),
        "R2": (
            "spark.StreamExecution.run",
            "delta.SetTransaction.apply",
            "delta.OptimisticTransaction.commit"),
    })

    modules = module_menu_for(route_menu)

    preprocess = next(label for label, owner in modules.owners.items()
                      if owner == "PreprocessTableMerge")
    transaction = next(label for label, owner in modules.owners.items()
                       if owner == "SetTransaction")
    assert resolve_module_selection(modules, preprocess, route_menu) == ("R1",)
    assert resolve_module_selection(modules, transaction, route_menu) == ("R2",)
    assert "PreprocessTableMerge" in modules.text
    assert "SetTransaction" in modules.text
def test_selected_module_owner_anchors_its_entry_method_during_route_scope():
    from library.chain_menu import RouteMenu, scope_route_menu
    menu = RouteMenu(routes={
        "R1": ("delta.PreprocessTableMerge.alignUpdateActions",),
        "R2": ("delta.PreprocessTableMerge.apply", "delta.DeltaMergeInto"),
        "R3": ("spark.RewriteMergeIntoTable.apply",),
    })

    scoped = scope_route_menu(
        menu, "Why is merge analysis diverted?", max_families=1,
        required_owners=("PreprocessTableMerge",))

    assert "R2" in scoped.routes
def test_module_reply_reports_explicitly_selected_owner_names():
    from library.chain_menu import ModuleMenu, resolve_module_owners
    modules = ModuleMenu(
        modules={"M1": ("R1",), "M2": ("R2",)},
        owners={"M1": "DeltaSink", "M2": "SetTransaction"})

    assert resolve_module_owners(modules, "M2, M999") == ("SetTransaction",)
def test_exact_selection_retains_entry_and_semantic_routes_for_selected_owner():
    from library.chain_menu import RouteMenu, Selection, retain_owner_routes
    menu = RouteMenu(
        routes={
            "R1": ("delta.OptimisticTransaction.snapshot",),
            "R2": ("delta.OptimisticTransaction.txnVersion",),
            "R3": ("delta.OptimisticTransaction.commit",),
            "R4": ("delta.PreprocessTableMerge.alignUpdateActions",),
            "R5": ("delta.PreprocessTableMerge.apply", "delta.DeltaMergeInto"),
        },
        route_occurrences={label: ((label,),) for label in ("R1", "R2", "R3", "R4", "R5")})
    initial = Selection(symbols=["delta.OptimisticTransaction.snapshot"],
                        route_ids=("R1",), occurrence_keys=(("R1",),))

    retained = retain_owner_routes(
        menu, initial, ("OptimisticTransaction", "PreprocessTableMerge"),
        "detect an already committed batch and divert merge analysis",
        route_scores={"R2": 0.95})

    assert "R2" in retained.route_ids
    assert "delta.OptimisticTransaction.txnVersion" in retained.symbols
    assert "R5" in retained.route_ids
    assert "delta.PreprocessTableMerge.apply" in retained.symbols
def test_selection_owner_closure_includes_incidental_route_owners():
    from library.chain_menu import Selection, selection_owners
    selection = Selection(symbols=[
        "delta.PreprocessTableMerge.alignUpdateActions",
        "delta.OptimisticTransaction.snapshot",
    ])

    assert selection_owners(selection, ("DeltaSink",)) == (
        "DeltaSink", "PreprocessTableMerge", "OptimisticTransaction")
def test_method_cards_do_not_collapse_distinct_members_of_one_owner():
    from library.chain_menu import RouteMenu, module_menu_for, resolve_module_symbols
    menu = RouteMenu(routes={
        "R1": ("delta.PreprocessTableMerge.alignUpdateActions",),
        "R2": ("delta.PreprocessTableMerge.apply", "delta.DeltaMergeInto"),
        "R3": ("delta.OptimisticTransaction.snapshot",),
        "R4": ("delta.OptimisticTransaction.txnVersion",),
    })

    cards = module_menu_for(menu)
    apply_card = next(label for label, symbol in cards.symbols.items()
                      if symbol == "delta.PreprocessTableMerge.apply")
    version_card = next(label for label, symbol in cards.symbols.items()
                        if symbol == "delta.OptimisticTransaction.txnVersion")

    assert cards.owners[apply_card] == "PreprocessTableMerge"
    assert resolve_module_symbols(cards, f"{apply_card}, {version_card}") == (
        "delta.PreprocessTableMerge.apply",
        "delta.OptimisticTransaction.txnVersion")
    assert "PreprocessTableMerge.apply" in cards.text
    assert "OptimisticTransaction.txnVersion" in cards.text
def test_selected_method_cards_survive_scope_and_exact_route_selection():
    from library.chain_menu import (
        RouteMenu, Selection, retain_symbol_routes, scope_route_menu)
    menu = RouteMenu(
        routes={
            "R1": ("delta.PreprocessTableMerge.alignUpdateActions",),
            "R2": ("delta.PreprocessTableMerge.apply", "delta.DeltaMergeInto"),
            "R3": ("delta.OptimisticTransaction.snapshot",),
            "R4": ("delta.OptimisticTransaction.txnVersion",),
        },
        route_occurrences={label: ((label,),) for label in ("R1", "R2", "R3", "R4")})
    required = ("delta.PreprocessTableMerge.apply",
                "delta.OptimisticTransaction.txnVersion")

    scoped = scope_route_menu(
        menu, "divert analysis and detect a replay", max_families=1,
        required_symbols=required)
    retained = retain_symbol_routes(
        scoped, Selection(route_ids=()), required)

    assert set(retained.route_ids) == {"R2", "R4"}
    assert set(required).issubset(retained.symbols)
def test_evidence_graph_preserves_typed_occurrence_edges_and_branches():
    from library.chain_menu import evidence_graph_for
    root = hop("delta.DeltaSink.addBatchWithStatusImpl", line=10)
    check = hop("delta.OptimisticTransaction.txnVersion", line=20)
    check = check.__class__(citation=check.citation.__class__(
        **{**check.citation.__dict__,
           "parent_qualified_name": "delta.DeltaSink.addBatchWithStatusImpl",
           "call_site_file": 'app.py', "call_site_line": 12,
           "relation": "calls", "stop_reason": "leaf"}),
        document_id=check.document_id, evidence=check.evidence)
    action = hop("delta.actions.SetTransaction", line=30)
    action = action.__class__(citation=action.citation.__class__(
        **{**action.citation.__dict__,
           "parent_qualified_name": "delta.DeltaSink.addBatchWithStatusImpl",
           "call_site_file": 'app.py', "call_site_line": 15,
           "relation": "references", "stop_reason": "reference"}),
        document_id=action.document_id, evidence=action.evidence)

    graph = evidence_graph_for([root, check, action])

    assert len(graph.nodes) == 3
    assert {(edge.relation, edge.file, edge.line) for edge in graph.edges} == {
        ("calls", 'app.py', 12), ("references", 'app.py', 15)}
    assert len(graph.roots) == 1
    assert len(graph.terminals) == 2


def test_graph_seed_selection_keeps_connectors_but_not_unselected_siblings():
    from library.chain_menu import evidence_graph_for, selection_for_graph_symbols
    root = hop("delta.DeltaSink.addBatchWithStatusImpl", line=10)
    middle = hop("delta.PendingTxn.commit", line=40)
    middle = middle.__class__(citation=middle.citation.__class__(
        **{**middle.citation.__dict__,
           "parent_qualified_name": "delta.DeltaSink.addBatchWithStatusImpl",
           "call_site_file": 'app.py', "call_site_line": 12}),
        document_id=middle.document_id, evidence=middle.evidence)
    target = hop("delta.actions.SetTransaction", line=30)
    target = target.__class__(citation=target.citation.__class__(
        **{**target.citation.__dict__, "parent_qualified_name": "delta.PendingTxn.commit",
           "call_site_file": 'app.py', "call_site_line": 45,
           "stop_reason": "reference"}),
        document_id=target.document_id, evidence=target.evidence)
    sibling = hop("delta.Unrelated.metric", line=2)
    sibling = sibling.__class__(citation=sibling.citation.__class__(
        **{**sibling.citation.__dict__,
           "parent_qualified_name": "delta.DeltaSink.addBatchWithStatusImpl",
           "call_site_file": 'app.py', "call_site_line": 18,
           "stop_reason": "leaf"}),
        document_id=sibling.document_id, evidence=sibling.evidence)
    graph = evidence_graph_for([root, middle, target, sibling])

    selection = selection_for_graph_symbols(graph, ("delta.actions.SetTransaction",))

    assert selection.symbols == [
        "delta.DeltaSink.addBatchWithStatusImpl",
        "delta.PendingTxn.commit", "delta.actions.SetTransaction"]
    assert "delta.Unrelated.metric" not in selection.symbols
    assert len(selection.occurrence_keys) == 3


def test_method_card_graph_context_names_typed_neighbors():
    from library.chain_menu import RouteMenu, evidence_graph_for, module_menu_for
    root = hop("delta.DeltaSink.addBatchWithStatusImpl", line=10)
    target = hop("delta.OptimisticTransaction.txnVersion", line=20)
    target = target.__class__(citation=target.citation.__class__(
        **{**target.citation.__dict__,
           "parent_qualified_name": "delta.DeltaSink.addBatchWithStatusImpl",
           "call_site_file": 'app.py', "call_site_line": 12,
           "stop_reason": "leaf"}),
        document_id=target.document_id, evidence=target.evidence)
    graph = evidence_graph_for([root, target])
    menu = RouteMenu(routes={"R1": (
        "delta.DeltaSink.addBatchWithStatusImpl",
        "delta.OptimisticTransaction.txnVersion")})

    cards = module_menu_for(menu, graph=graph)

    assert "DeltaSink.addBatchWithStatusImpl -calls-> OptimisticTransaction.txnVersion" in cards.text
def test_component_menu_keeps_disconnected_cross_repository_flows_separate():
    from library.chain_menu import RouteMenu, component_menu_for, evidence_graph_for
    spark = hop("spark.StreamExecution.runStream", line=1)
    query = hop("spark.StreamExecution.QUERY_ID_KEY", parent="spark.StreamExecution.runStream",
                line=5, call_line=2)
    delta = hop("delta.DeltaSink.addBatchWithStatusImpl", line=20)
    txn = hop("delta.OptimisticTransaction.txnVersion",
              parent="delta.DeltaSink.addBatchWithStatusImpl", line=30, call_line=22)
    graph = evidence_graph_for([spark, query, delta, txn])
    routes = RouteMenu(routes={
        "R1": (spark.citation.qualified_name, query.citation.qualified_name),
        "R2": (delta.citation.qualified_name, txn.citation.qualified_name),
    })

    cards = component_menu_for(graph, routes)

    assert len(cards.components) == 2
    assert "StreamExecution.runStream" in cards.text
    assert "OptimisticTransaction.txnVersion" in cards.text
    assert set().union(*(set(ids) for ids in cards.routes.values())) == {"R1", "R2"}
def test_component_selection_expands_only_selected_component_routes():
    from library.chain_menu import ComponentMenu, resolve_component_selection
    cards = ComponentMenu(
        components={"C1": ("G1", "G2"), "C2": ("G3",)},
        routes={"C1": ("R1", "R2"), "C2": ("R3",)})

    assert resolve_component_selection(cards, "C2, C999") == ("R3",)
def test_graph_report_accounts_for_every_node_edge_and_component():
    from library.chain_menu import evidence_graph_for, evidence_graph_report
    root = hop("api.start", line=1)
    child = hop("service.finish", parent="api.start", line=10, call_line=2)

    report = evidence_graph_report(evidence_graph_for([root, child]))

    assert report["node_count"] == 2
    assert report["edge_count"] == 1
    assert report["component_count"] == 1
    assert report["components"][0]["nodes"][1]["symbol"] == "service.finish"
    assert report["components"][0]["edges"][0]["source"] == "api.start"
    assert report["components"][0]["edges"][0]["target"] == "service.finish"
def test_story_ir_renders_a_definition_body_once_across_repeated_occurrences():
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=10, line_end=20,
        kind="definition_body", content="large implementation", sha256="body")
    first = hop("service.persist", line=10, call_line=3)
    second = hop("service.persist", line=10, call_line=7)
    first = BundleHop(
        citation=first.citation, document_id=first.document_id,
        source_excerpts=(*first.source_excerpts, body))
    second = BundleHop(
        citation=StructuralCitation(**{
            **second.citation.__dict__, "call_site_line": 7}),
        document_id=second.document_id,
        source_excerpts=(*second.source_excerpts, body))
    selection = Selection(
        symbols=["service.persist"],
        occurrence_keys=tuple(
            (item.citation.qualified_name, item.citation.file,
             item.citation.line_start, item.citation.line_end,
             item.citation.parent_qualified_name, item.citation.call_site_file,
             item.citation.call_site_line, item.citation.relation,
             item.citation.hop, item.citation.stop_reason)
            for item in (first, second)))

    story = build_story_ir([first, second], selection, Fetched())

    bodies = [
        excerpt for node in story.nodes for excerpt in node.excerpts
        if excerpt.kind == "definition_body"]
    assert len(story.nodes) == 2
    assert len(bodies) == 1
def test_story_ir_renders_a_definition_slice_once_across_repeated_occurrences():
    source_slice = SourceExcerpt(
        source_name="repo", file="app.py", line_start=12, line_end=15,
        kind="definition_slice", content="relevant statements", sha256="slice")
    first = hop("service.persist", line=10, call_line=3)
    second = hop("service.persist", line=10, call_line=7)
    first = BundleHop(
        citation=first.citation, document_id=first.document_id,
        source_excerpts=(*first.source_excerpts, source_slice))
    second = BundleHop(
        citation=StructuralCitation(**{
            **second.citation.__dict__, "call_site_line": 7}),
        document_id=second.document_id,
        source_excerpts=(*second.source_excerpts, source_slice))
    selection = Selection(
        symbols=["service.persist"],
        occurrence_keys=tuple(
            (item.citation.qualified_name, item.citation.file,
             item.citation.line_start, item.citation.line_end,
             item.citation.parent_qualified_name, item.citation.call_site_file,
             item.citation.call_site_line, item.citation.relation,
             item.citation.hop, item.citation.stop_reason)
            for item in (first, second)))

    story = build_story_ir([first, second], selection, Fetched())

    slices = [
        excerpt for node in story.nodes for excerpt in node.excerpts
        if excerpt.kind == "definition_slice"]
    assert len(slices) == 1
def test_story_ir_omits_source_excerpts_already_contained_in_selected_body():
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=1, line_end=20,
        kind="definition_body",
        content="def start():\n    persist()\n\ndef persist():\n    return done",
        sha256="body")
    root = hop("api.start", line=1, call_line=1)
    child = hop("service.persist", parent="api.start", line=10, call_line=2)
    root = BundleHop(
        citation=root.citation, document_id=root.document_id,
        source_excerpts=(
            *root.source_excerpts,
            SourceExcerpt(source_name="repo", file="app.py", line_start=2,
                          line_end=2, kind="body_edge", content="persist()",
                          sha256="body"),
            body))
    child = BundleHop(
        citation=child.citation, document_id=child.document_id,
        source_excerpts=(
            *child.source_excerpts,
            SourceExcerpt(source_name="repo", file="app.py", line_start=2,
                          line_end=2, kind="call_site", content="persist()",
                          sha256="body"),
            SourceExcerpt(source_name="repo", file="other.py", line_start=7,
                          line_end=7, kind="definition", content="external",
                          sha256="other")))
    selection = Selection(symbols=["api.start", "service.persist"])

    story = build_story_ir([root, child], selection, Fetched())
    excerpts = [excerpt for node in story.nodes for excerpt in node.excerpts]

    assert [excerpt.kind for excerpt in excerpts].count("definition_body") == 1
    assert [(excerpt.file, excerpt.kind) for excerpt in excerpts
            if excerpt.kind != "definition_body"] == [("other.py", "definition")]
def test_completed_body_dependency_becomes_story_node_and_edge():
    """A retained direct body dependency must reach StoryIR as a node and an edge.

    The occurrence key the completion records in chain_menu must be the same
    identity chain_story filters on, or the dependency silently disappears
    between the two modules.
    """
    root = hop("app.Flow.run", line=1, call_line=10)
    plain = hop("app.Output.emit", parent="app.Flow.run", line=20, call_line=3)
    dependency = BundleHop(
        citation=plain.citation.__class__(**{
            **plain.citation.__dict__,
            "stop_reason": "selected_route_fanout"}),
        document_id=plain.document_id,
        source_excerpts=plain.source_excerpts)
    selection = complete_selection_with_body_dependencies(
        Selection(symbols=["app.Flow.run"],
                  occurrence_keys=(_occurrence_key(root),)),
        (root, dependency), ("app.Flow.run",))

    story = build_story_ir((root, dependency), selection, Fetched())

    by_symbol = {node.symbol: node for node in story.nodes}
    assert "app.Output.emit" in by_symbol
    assert any(
        edge.source == by_symbol["app.Flow.run"].id
        and edge.target == by_symbol["app.Output.emit"].id
        and edge.relation == "calls" and edge.line == 3
        for edge in story.edges)
def test_story_ir_carries_source_chunks_and_expands_them_compactly():
    """Selected bodies become an X ledger the narration references, never retypes.

    Compact rendering suppresses the raw body dump and offers the chunks;
    expansion re-attaches the exact line with its coordinate; the default
    rendering is unchanged so the paid path is not silently switched.
    """
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=20, line_end=23,
        kind="definition_body", content="\n".join((
            "def emit(rows: Seq[Row]): Unit = {",
            '  log.info("noise line")',
            "  sink.write(rows)",
            "}",
        )), sha256="sha-emit")
    plain = hop("app.Flow.run", line=1, call_line=10)
    root = BundleHop(
        citation=plain.citation, document_id=plain.document_id,
        source_excerpts=(body,))
    dependency = hop("app.Sink.write", parent="app.Flow.run", line=40,
                     call_line=22)
    selection = Selection(
        symbols=["app.Flow.run", "app.Sink.write"],
        occurrence_keys=(_occurrence_key(root), _occurrence_key(dependency)))

    story = build_story_ir((root, dependency), selection, Fetched())

    assert [chunk.id for chunk in story.chunks] == ["X1", "X2"]
    assert [(chunk.line_start, chunk.line_end) for chunk in story.chunks] == [
        (20, 20), (22, 22)]

    compact = render_story_evidence(story, compact_source=True)
    assert "{{X2}}" in compact
    assert "noise line" not in compact
    assert "{{N1}}" in compact

    default = render_story_evidence(story)
    assert "noise line" in default

    expanded = expand_story_placeholders("Proof: {{X2}} then {{N1}}.", story)
    assert "app.py:22 `  sink.write(rows)`" in expanded
    assert "app.Flow.run (app.py:1)" in expanded
    with pytest.raises(ValueError, match="X9"):
        expand_story_placeholders("see {{X9}}", story)
def test_story_ir_turns_uncovered_transition_sites_into_chunks():
    """A dependency's call site outside every selected body still gets an X id."""
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=20, line_end=23,
        kind="definition_body", content="\n".join((
            "def emit(rows: Seq[Row]): Unit = {",
            "  journal.mark()",
            "  sink.write(rows)",
            "}",
        )), sha256="sha-emit")
    plain = hop("app.Flow.run", line=1, call_line=10)
    root = BundleHop(
        citation=plain.citation, document_id=plain.document_id,
        source_excerpts=(body,))
    bridge_plain = hop("repo.Bridge.send", parent="app.Flow.run", line=60,
                       call_line=7)
    bridge = BundleHop(
        citation=bridge_plain.citation.__class__(**{
            **bridge_plain.citation.__dict__,
            "call_site_file": "bridge.py"}),
        document_id=bridge_plain.document_id,
        source_excerpts=(SourceExcerpt(
            source_name="repo", file="bridge.py", line_start=7, line_end=7,
            kind="call_site", content="    Bridge.send(payload)",
            sha256="sha-bridge"),))
    selection = Selection(
        symbols=["app.Flow.run", "repo.Bridge.send"],
        occurrence_keys=(_occurrence_key(root), _occurrence_key(bridge)))

    story = build_story_ir((root, bridge), selection, Fetched())

    by_file = {chunk.file: chunk for chunk in story.chunks}
    assert "bridge.py" in by_file
    assert by_file["bridge.py"].lines == ((7, "    Bridge.send(payload)"),)
    assert by_file["bridge.py"].reason == "call_site"
    expanded = expand_story_placeholders(
        "Handoff: {{" + by_file["bridge.py"].id + "}}", story)
    assert "bridge.py:7 `    Bridge.send(payload)`" in expanded
def test_compact_render_keeps_dependency_nodes_to_one_line():
    """A retained body dependency is provable from its node line alone.

    Its call site is already a ledger chunk and its transition an edge, so the
    compact prompt suppresses its description and excerpts — while the default
    rendering keeps them and route nodes keep their descriptions in both.
    """
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=20, line_end=23,
        kind="definition_body", content="\n".join((
            "def emit(rows: Seq[Row]): Unit = {",
            "  journal.mark()",
            "  sink.write(rows)",
            "}",
        )), sha256="sha-emit")
    plain = hop("app.Flow.run", line=1, call_line=10)
    root = BundleHop(
        citation=plain.citation, document_id=plain.document_id,
        source_excerpts=(body,))
    fanout_plain = hop("repo.Bridge.send", parent="app.Flow.run", line=60,
                       call_line=7)
    fanout = BundleHop(
        citation=fanout_plain.citation.__class__(**{
            **fanout_plain.citation.__dict__,
            "call_site_file": "bridge.py",
            "stop_reason": "selected_route_fanout"}),
        document_id=fanout_plain.document_id,
        source_excerpts=(SourceExcerpt(
            source_name="repo", file="bridge.py", line_start=7, line_end=7,
            kind="call_site", content="    Bridge.send(payload)",
            sha256="sha-bridge"),))
    selection = Selection(
        symbols=["app.Flow.run", "repo.Bridge.send"],
        occurrence_keys=(_occurrence_key(root), _occurrence_key(fanout)))
    fetched = Fetched(definitions={
        "app.Flow.run": "Description: Drives the flow.",
        "repo.Bridge.send": "Description: Ships the payload."})

    story = build_story_ir((root, fanout), selection, fetched)
    compact = render_story_evidence(story, compact_source=True)
    default = render_story_evidence(story)

    assert "repo.Bridge.send [app.py:60]" in compact
    assert "Drives the flow." in compact
    assert "Ships the payload." not in compact
    assert compact.count("Bridge.send(payload)") == 1
    assert "Ships the payload." in default
    assert "source call_site" in default
    assert "source call_site" not in compact
def test_formulation_spine_prefers_the_compact_ledger():
    """The live path renders the ledger once chunks exist; bare stories keep
    the classic rendering so nothing regresses when no body was selected."""
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=20, line_end=23,
        kind="definition_body", content="\n".join((
            "def emit(rows: Seq[Row]): Unit = {",
            '  log.info("noise line")',
            "  sink.write(rows)",
            "}",
        )), sha256="sha-emit")
    plain = hop("app.Flow.run", line=1, call_line=10)
    root = BundleHop(
        citation=plain.citation, document_id=plain.document_id,
        source_excerpts=(body,))
    dependency = hop("app.Sink.write", parent="app.Flow.run", line=40,
                     call_line=22)
    selection = Selection(
        symbols=["app.Flow.run", "app.Sink.write"],
        occurrence_keys=(_occurrence_key(root), _occurrence_key(dependency)))
    story = build_story_ir((root, dependency), selection, Fetched())

    spine = render_formulation_spine(story)

    assert "SOURCE CHUNKS" in spine
    assert "noise line" not in spine
    assert spine == render_story_evidence(story, compact_source=True)

    bare = build_story_ir((dependency,), Selection(
        symbols=["app.Sink.write"],
        occurrence_keys=(_occurrence_key(dependency),)), Fetched())
    assert not bare.chunks
    assert render_formulation_spine(bare) == render_story_evidence(bare)
def test_doc_header_excerpts_become_chunks_and_leave_compact_nodes():
    """A definition's header comment is exact, hash-bound proof — it gets an
    X id like any transition site and stops double-rendering on the node."""
    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=20, line_end=22,
        kind="definition_body", content="\n".join((
            "def emit(rows): Unit = {",
            "  sink.write(rows)",
            "}",
        )), sha256="sha-emit")
    header = SourceExcerpt(
        source_name="repo", file="app.py", line_start=18, line_end=19,
        kind="doc_header", content="\n".join((
            "/**",
            " * Copies the write loop from upstream.",
            " */",
        )), sha256="sha-emit")
    plain = hop("app.Flow.run", line=1, call_line=10)
    root = BundleHop(
        citation=plain.citation, document_id=plain.document_id,
        source_excerpts=(body, header))
    selection = Selection(
        symbols=["app.Flow.run"], occurrence_keys=(_occurrence_key(root),))

    story = build_story_ir((root,), selection, Fetched())

    by_span = {(chunk.line_start, chunk.line_end): chunk
               for chunk in story.chunks}
    assert (18, 19) in by_span
    assert by_span[(18, 19)].reason == "doc_header"
    compact = render_story_evidence(story, compact_source=True)
    assert compact.count("Copies the write loop") == 1
    assert "source doc_header" not in compact
def test_unreferenced_story_proof_appends_compiler_transitions_and_exact_chunks():
    from library.chain_story import render_unreferenced_story_evidence

    body = SourceExcerpt(
        source_name="repo", file="app.py", line_start=20, line_end=23,
        kind="definition_body", content="\n".join((
            "def emit(rows):",
            "    audit(rows)",
            "    sink.write(rows)",
            "    return rows",
        )), sha256="sha-emit")
    plain = hop("app.Flow.run", line=20, call_line=10)
    root = BundleHop(
        citation=plain.citation, document_id=plain.document_id,
        source_excerpts=(body,))
    dependency = hop(
        "app.Sink.write", parent="app.Flow.run", line=40, call_line=22)
    selection = Selection(
        symbols=["app.Flow.run", "app.Sink.write"],
        occurrence_keys=(_occurrence_key(root), _occurrence_key(dependency)))
    story = build_story_ir((root, dependency), selection, Fetched())
    draft = "The entry is {{N1}}."

    appendix = render_unreferenced_story_evidence(draft, story)

    assert "app.Flow.run (app.py:20)" not in appendix
    assert "app.Sink.write (app.py:40)" not in appendix
    assert "app.Flow.run calls app.Sink.write at app.py:22" in appendix
    assert "app.py:20 `def emit(rows):`" in appendix
    assert "app.py:22 `    sink.write(rows)`" in appendix
    assert appendix == render_unreferenced_story_evidence(draft, story)
def test_compact_render_keeps_reverse_reference_closure_to_one_line():
    body = SourceExcerpt(
        source_name="repo", file="rule.py", line_start=10, line_end=13,
        kind="definition_body", content="\n".join((
            "def prepare(plan):",
            "    register(plan)",
            "    return plan",
            "",
        )), sha256="sha-rule")
    plain = hop("pkg.Rule.prepare", parent="pkg.Plan", line=10, call_line=11)
    closure = BundleHop(
        citation=plain.citation.__class__(**{
            **plain.citation.__dict__,
            "stop_reason": "selected_reference_caller",
            "relation": "referenced_by"}),
        document_id=plain.document_id, source_excerpts=(body,))
    selection = Selection(
        symbols=["pkg.Rule.prepare"],
        occurrence_keys=(_occurrence_key(closure),))
    story = build_story_ir(
        (closure,), selection, Fetched(definitions={
            "pkg.Rule.prepare": "Description: Generated duplicate prose."}))

    compact = render_story_evidence(story, compact_source=True)
    default = render_story_evidence(story)

    assert "pkg.Rule.prepare [app.py:10]" in compact
    assert "Generated duplicate prose." not in compact
    assert "Generated duplicate prose." in default
    assert compact.count("register(plan)") == 1
def test_story_renderers_preserve_contains_and_reverse_call_relations():
    from library.chain_story import (
        StoryEdge, StoryIR, StoryNode, expand_story_placeholders,
        render_unreferenced_story_evidence, render_story_evidence)

    nodes = (
        StoryNode("N1", "pkg.Owner", "owner.py", 1, "owner.py", 1, "localized"),
        StoryNode("N2", "pkg.Owner.member", "owner.py", 8, "owner.py", 8, "contains"),
        StoryNode("N3", "pkg.Target.execute", "target.py", 2, "caller.py", 7, "localized"),
        StoryNode("N4", "pkg.Caller.run", "caller.py", 3, "caller.py", 7, "called_by"),
    )
    story = StoryIR(nodes=nodes, edges=(
        StoryEdge("E1", "N1", "N2", "contains", "owner.py", 8),
        StoryEdge("E2", "N3", "N4", "called_by", "caller.py", 7),
    ))

    assert expand_story_placeholders("{{E1}} / {{E2}}", story) == (
        "contains at owner.py:8 / is called by at caller.py:7")
    appendix = render_unreferenced_story_evidence("", story)
    assert "pkg.Owner contains pkg.Owner.member at owner.py:8" in appendix
    assert "pkg.Target.execute is called by pkg.Caller.run at caller.py:7" in appendix
    assert "pkg.Owner calls pkg.Owner.member" not in appendix
    rendered = render_story_evidence(story)
    assert "contained at owner.py:8" in rendered
    assert "called at owner.py:8" not in rendered
