"""Route-family cards: complete causal alternatives, never symbol lists.

Candidates are gathered per obligation under a fixed budget — exact
question identifiers, semantic seeds, bounded reverse entry discovery,
bounded forward execution, owner bridges — and clustered into
source-aware families. Overloads, companions, and module shadows never
merge by qualified name; one namespace cannot consume the menu; every
obligation keeps a reserve; and the selector prompt carries no source
bodies. Synthetic fixtures only.
"""
from __future__ import annotations

import sqlite3

import pytest

from library.question_facets import extract_question_facets
from library.route_families import (
    RouteFamilyCard,
    build_route_family_cards,
    render_family_selector_prompt,
    resolve_family_selection,
)
from library.scip import init_scip_schema

SOURCE = "src1"


def _symbol(conn, cid, *, file, qn, line_start, line_end, parent=""):
    conn.execute(
        "INSERT INTO scip_symbols (canonical_id, source_name, language, "
        "file, line_start, line_end, kind, display_name, qualified_name, "
        "parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (cid, SOURCE, "scala", file, line_start, line_end, "method",
         qn.rsplit(".", 1)[-1], qn, parent))


def _edge(conn, caller, callee, *, line, file, edge_type="call"):
    conn.execute(
        "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id,"
        " edge_type, file, line, confidence) VALUES (?,?,?,?,?,'exact')",
        (caller, callee, edge_type, file, line))


def cid(qn: str) -> str:
    return f"scip-x x src1 0.1 `{qn}`."


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    init_scip_schema(connection)
    # A small causal world: writer path, a registrar, a comparison twin.
    for qn, file, start, end, parent in (
            ("pkg.Writer.write", "core/writer.scala", 10, 60, "pkg.Writer"),
            ("pkg.Writer", "core/writer.scala", 5, 80, ""),
            ("pkg.Sink.flush", "core/sink.scala", 3, 20, "pkg.Sink"),
            ("pkg.Registrar.install", "core/registrar.scala", 4, 18,
             "pkg.Registrar"),
            ("alt.Writer.write", "alt/writer.scala", 12, 55, "alt.Writer"),
            ("pkg.Deep.helper", "core/deep.scala", 2, 9, "pkg.Deep"),
    ):
        _symbol(connection, cid(qn), file=file, qn=qn,
                line_start=start, line_end=end, parent=parent)
    _edge(connection, cid("pkg.Writer.write"), cid("pkg.Sink.flush"),
          line=30, file="core/writer.scala")
    _edge(connection, cid("pkg.Sink.flush"), cid("pkg.Deep.helper"),
          line=8, file="core/sink.scala")
    _edge(connection, cid("pkg.Registrar.install"), cid("pkg.Writer"),
          line=9, file="core/registrar.scala", edge_type="type_ref")
    _edge(connection, cid("alt.Writer.write"), cid("pkg.Sink.flush"),
          line=20, file="alt/writer.scala")
    connection.commit()
    yield connection
    connection.close()


def obligations_for(question: str, bindings) -> list:
    return [{"id": f"O{index}", "text": text, "symbols": list(symbols)}
            for index, (text, symbols) in enumerate(bindings, start=1)]


class TestFamilyGeneration:
    def test_semantic_seed_outside_clew_routes_reaches_a_card(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How does pkg.Writer.write flush rows?",
            obligations=obligations_for("q", [
                ("prove the write path", ["pkg.Writer.write"])]),
            semantic_seed_ids=[cid("pkg.Deep.helper")])

        names = {name for card in cards.cards
                 for name in card.node_qualified_names}
        assert "pkg.Deep.helper" in names

    def test_upstream_registrar_reaches_a_reverse_family(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="Where is the writer registered?",
            obligations=obligations_for("q", [
                ("prove registration", ["pkg.Writer.write"])]))

        registrar_cards = [
            card for card in cards.cards
            if "pkg.Registrar.install" in card.node_qualified_names]
        assert registrar_cards
        assert "referenced_by" in registrar_cards[0].relation_sequence or (
            "shared_reference" in registrar_cards[0].relation_sequence)

    def test_forward_execution_dependency_reaches_a_family(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How does write reach the helper?",
            obligations=obligations_for("q", [
                ("prove execution", ["pkg.Writer.write"])]))

        names = {name for card in cards.cards
                 for name in card.node_qualified_names}
        assert "pkg.Deep.helper" in names

    def test_comparison_sides_get_separate_cards(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question=("What is the difference between pkg.Writer.write "
                      "and alt.Writer.write?"),
            obligations=obligations_for("q", [
                ("side one", ["pkg.Writer.write"]),
                ("side two", ["alt.Writer.write"])]))

        modules = {card.module for card in cards.cards}
        assert "core" in modules and "alt" in modules
        by_obligation = {}
        for card in cards.cards:
            by_obligation.setdefault(card.obligation_id, []).append(card)
        assert len(by_obligation) == 2

    def test_every_obligation_keeps_reserved_candidates(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("one", ["pkg.Writer.write"]),
                ("two", ["alt.Writer.write"])]),
            per_obligation_budget=1)

        assert set(cards.reserve_by_obligation) == {"O1", "O2"}

    def test_one_namespace_cannot_consume_the_menu(self, conn):
        for index in range(80):
            qn = f"pkg.Bulk{index}.go"
            _symbol(conn, cid(qn), file=f"core/bulk{index}.scala",
                    qn=qn, line_start=2, line_end=8,
                    parent=f"pkg.Bulk{index}")
            _edge(conn, cid("pkg.Writer.write"), cid(qn),
                  line=31 + index, file="core/writer.scala")
        conn.commit()

        cards = build_route_family_cards(
            conn, source=SOURCE,
            question=("difference between pkg.Writer.write and "
                      "alt.Writer.write"),
            obligations=obligations_for("q", [
                ("side one", ["pkg.Writer.write"]),
                ("side two", ["alt.Writer.write"])]))

        assert len(cards.cards) <= 64
        alt_cards = [card for card in cards.cards
                     if card.module == "alt"]
        assert alt_cards, "the alt side survived the bulk namespace"

    def test_overloads_and_shadows_stay_distinct(self, conn):
        _symbol(conn, cid("pkg.Writer.write+1"),
                file="shadow/writer.scala", qn="pkg.Writer.write",
                line_start=100, line_end=140, parent="pkg.Writer")
        conn.commit()

        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("prove", ["pkg.Writer.write"])]))

        write_nodes = {
            node for card in cards.cards
            for node in card.node_identities
            if node[0] == "pkg.Writer.write"}
        files = {node[2] for node in write_nodes}
        assert {"core/writer.scala", "shadow/writer.scala"} <= files


class TestSelectorSurface:
    def test_prompt_is_bounded_and_body_free(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("prove", ["pkg.Writer.write"])]))

        prompt = render_family_selector_prompt(
            cards, question="How does write flush?")

        assert len(prompt) // 4 <= 8000
        assert "def " not in prompt and "{" not in prompt.replace(
            "{}", "")
        assert "F1." in prompt

    def test_reply_selects_families_and_marks_unresolved(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("one", ["pkg.Writer.write"]),
                ("two", ["alt.Writer.write"])]))

        selection = resolve_family_selection(
            "O1: F1\nO2:", cards)

        assert selection.selected_by_obligation["O1"]
        assert "O2" in selection.unresolved_obligations
        reserve = cards.reserve_by_obligation.get("O2", ())
        assert selection.retained_by_obligation["O2"] == tuple(
            reserve)

    def test_truncated_reply_keeps_bounded_reserve_not_the_pool(
            self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("one", ["pkg.Writer.write"])]))

        selection = resolve_family_selection(
            "O1: F", cards, truncated=True)

        assert "O1" in selection.unresolved_obligations
        retained = selection.retained_by_obligation["O1"]
        assert 0 < len(retained) <= 3
        assert len(retained) < max(len(cards.cards), 4)


class TestDeterministicExpansion:
    def test_selected_families_expand_to_exact_bodies_and_edge_sites(
            self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="How does write flush?",
            obligations=obligations_for("q", [
                ("prove execution", ["pkg.Writer.write"])]))
        from library.route_families import expand_selected_families
        target = next(
            card for card in cards.cards
            if card.node_qualified_names[:2] == (
                "pkg.Writer.write", "pkg.Sink.flush"))
        selection = resolve_family_selection(
            f"O1: {target.card_id}", cards)

        expansion = expand_selected_families(
            conn, cards, selection, source=SOURCE)

        assert ("core/writer.scala", 30) in expansion.edge_sites
        extents = set(expansion.required_body_extents)
        assert ("pkg.Writer.write", "core/writer.scala", 10, 60) in extents
        names = {c.qualified_name for c in expansion.citations}
        assert {"pkg.Writer.write", "pkg.Sink.flush"} <= names

    def test_unselected_overload_never_rides_in_by_name(self, conn):
        _symbol(conn, cid("pkg.Writer.write+1"),
                file="shadow/writer.scala", qn="pkg.Writer.write",
                line_start=100, line_end=140, parent="pkg.Writer")
        conn.commit()
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("prove", ["pkg.Writer.write"])]))
        from library.route_families import expand_selected_families
        core_identity = next(
            card for card in cards.cards
            if card.node_identities[0][2] == "core/writer.scala"
            and len(card.node_qualified_names) == 1)
        selection = resolve_family_selection(
            f"O1: {core_identity.card_id}", cards)
        # Restrict retention to the explicitly selected identity card.
        selection = type(selection)(
            selected_by_obligation=selection.selected_by_obligation,
            retained_by_obligation={
                "O1": (core_identity.card_id,)},
            unresolved_obligations=(),
            unknown_ids=())

        expansion = expand_selected_families(
            conn, cards, selection, source=SOURCE)

        extents = set(expansion.required_body_extents)
        assert ("pkg.Writer.write", "core/writer.scala", 10, 60) in extents
        assert not any(extent[1] == "shadow/writer.scala"
                       for extent in extents)


class TestSelectorGrammarPins:
    def test_duplicate_obligation_lines_are_rejected_to_reserve(
            self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("prove", ["pkg.Writer.write"])]))
        first = cards.cards[0].card_id

        selection = resolve_family_selection(
            f"O1: {first}\nO1: {first}", cards)

        assert "O1" in selection.unresolved_obligations
        assert any(entry.startswith("duplicate-line:")
                   for entry in selection.unknown_ids)

    def test_family_of_another_obligation_is_rejected(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("one", ["pkg.Writer.write"]),
                ("two", ["alt.Writer.write"])]))
        foreign = next(card.card_id for card in cards.cards
                       if card.obligation_id == "O2")

        selection = resolve_family_selection(f"O1: {foreign}", cards)

        assert "O1" in selection.unresolved_obligations
        assert any(entry.startswith("wrong-obligation:")
                   for entry in selection.unknown_ids)

    def test_incidental_numbers_never_select(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("prove", ["pkg.Writer.write"])]))

        selection = resolve_family_selection(
            "I would pick F1 because of line 42.", cards)

        assert selection.selected_by_obligation == {}
        assert "O1" in selection.unresolved_obligations


class TestExpansionBounds:
    def test_overflow_becomes_a_gap_never_select_all(self, conn):
        from library.route_families import expand_selected_families
        cards = build_route_family_cards(
            conn, source=SOURCE, question="q",
            obligations=obligations_for("q", [
                ("prove", ["pkg.Writer.write"])]))
        retained = tuple(card.card_id for card in cards.cards)
        selection = resolve_family_selection("O1:", cards)
        selection = type(selection)(
            selected_by_obligation={},
            retained_by_obligation={"O1": retained},
            unresolved_obligations=("O1",),
            unknown_ids=())

        expansion = expand_selected_families(
            conn, cards, selection, source=SOURCE,
            total_body_extents=2)

        assert len(expansion.required_body_extents) == 2
        assert any(gap.startswith("total-bodies-overflow:")
                   for gap in expansion.gaps)


class TestExactExtentMaterialization:
    def test_required_extents_materialize_without_name_restoration(
            self, tmp_path):
        from docgen.catalog_writer import _element_doc_id
        from library import Library
        from library.chain_bundle import curate_bundle
        from library.route_families import _identity_citation

        root = tmp_path / "corpus"
        (root / "core").mkdir(parents=True)
        (root / "shadow").mkdir(parents=True)
        core_lines = ["// doc for the writer", "object Writer:"] + [
            f"  line {index}" for index in range(3, 70)]
        (root / "core" / "writer.scala").write_text(
            "\n".join(core_lines) + "\n")
        (root / "shadow" / "writer.scala").write_text(
            "\n".join(f"shadow {index}" for index in range(1, 160))
            + "\n")

        library = Library(tmp_path / "l.db")
        try:
            with library._conn_provider.acquire() as connection:
                init_scip_schema(connection)
                _symbol(connection, cid("pkg.Writer.write"),
                        file="core/writer.scala", qn="pkg.Writer.write",
                        line_start=10, line_end=60, parent="pkg.Writer")
                _symbol(connection, cid("pkg.Writer.write+1"),
                        file="shadow/writer.scala",
                        qn="pkg.Writer.write",
                        line_start=100, line_end=140,
                        parent="pkg.Writer")
                connection.commit()
            library.add_document(
                content_type="catalog", title="write",
                content="Writes.", source_files=["core/writer.scala"],
                doc_id=_element_doc_id(SOURCE, "pkg.Writer.write"),
                source_name=SOURCE)

            citation = _identity_citation(
                "pkg.Writer.write", "core/writer.scala", 10, 60, SOURCE)
            bundle = curate_bundle(
                library, [citation], source=SOURCE,
                source_root=str(root), materialize_source=True,
                required_body_extents=[
                    ("pkg.Writer.write", "core/writer.scala", 10, 60)])

            excerpts = [
                excerpt for hop in bundle.hops
                for excerpt in hop.source_excerpts]
            body_extents = {
                (excerpt.file, excerpt.line_start, excerpt.line_end)
                for excerpt in excerpts
                if excerpt.kind == "definition_body"}
            assert ("core/writer.scala", 10, 60) in body_extents
            assert not any(file == "shadow/writer.scala"
                           for file, _s, _e in body_extents)
        finally:
            library.close()


class TestClewMatchSeeds:
    def test_clew_route_endpoint_reaches_a_card(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How is data flushed downstream?",
            obligations=obligations_for("q", [
                ("prove the flush path", [])]),
            clew_matches=(synthetic_clew_match("pkg.Deep.helper"),))

        names = {name for card in cards.cards
                 for name in card.node_qualified_names}
        assert "pkg.Deep.helper" in names


class TestSeedRelevanceOrdering:
    def test_question_relevant_clew_outranks_direction(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How is the registrar install wired?",
            obligations=obligations_for("q", [
                ("prove the wiring", [])]),
            clew_matches=(
                synthetic_clew_match("pkg.Registrar.install"),
                synthetic_clew_match("pkg.Deep.helper")))

        assert "pkg.Registrar.install" in (
            cards.cards[0].node_qualified_names)


class TestObligationScopedRanking:
    def test_obligation_text_focuses_its_own_family_ranking(self, conn):
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How is output produced?",
            obligations=obligations_for("q", [
                ("prove the flush helper path", []),
                ("prove the registrar install wiring", [])]),
            clew_matches=(
                synthetic_clew_match("pkg.Deep.helper"),
                synthetic_clew_match("pkg.Registrar.install")))

        second_obligation_cards = [
            card for card in cards.cards if card.obligation_id == "O2"]
        assert second_obligation_cards
        assert "pkg.Registrar.install" in (
            second_obligation_cards[0].node_qualified_names)


def synthetic_clew_match(*route):
    """A pool entry shaped like a real ClewMatch, synthetic values only."""
    clew = type("Clew", (), {
        "id": "-".join(route), "route": route, "steps": (),
        "question": "", "entry_symbol": route[0], "files": (),
        "strategy": "synthetic"})()
    return type("Match", (), {
        "clew": clew, "target_symbols": (), "similarity": 0.0,
        "structure_score": 0, "origin_rank": None})()


class TestOriginBreadth:
    def test_later_seed_families_survive_a_top_heavy_budget(self, conn):
        for qn, file, start, end in (
                ("zone1.A", "z1/a.scala", 5, 30),
                ("zone1.B", "z1/b.scala", 3, 12),
                ("zone1.B2", "z1/b2.scala", 3, 12),
                ("zone1.C", "z1/c.scala", 3, 12),
                ("zone2.X", "z2/x.scala", 4, 20),
                ("zone2.Y", "z2/y.scala", 2, 10),
        ):
            _symbol(conn, cid(qn), file=file, qn=qn,
                    line_start=start, line_end=end)
        _edge(conn, cid("zone1.A"), cid("zone1.B"), line=7,
              file="z1/a.scala")
        _edge(conn, cid("zone1.A"), cid("zone1.C"), line=8,
              file="z1/a.scala")
        _edge(conn, cid("zone1.B"), cid("zone1.B2"), line=5,
              file="z1/b.scala")
        _edge(conn, cid("zone2.X"), cid("zone2.Y"), line=6,
              file="z2/x.scala")
        conn.commit()

        # The first origin owns three families; a plain sorted cut of
        # three would keep them all and the second origin would never
        # surface. The per-origin cap keeps two and admits the next
        # origin.
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How do the zones interact?",
            obligations=obligations_for("q", [
                ("prove the zone flow", [])]),
            clew_matches=(
                synthetic_clew_match("zone1.A", "zone1.B", "zone1.C"),
                synthetic_clew_match("zone2.X")),
            per_obligation_budget=3)

        names = {name for card in cards.cards
                 for name in card.node_qualified_names}
        assert "zone2.X" in names


class TestRouteForking:
    def test_sibling_call_paths_each_form_a_route(self, conn):
        for qn, file in (("zone1.A", "z1/a.scala"),
                         ("zone1.B", "z1/b.scala"),
                         ("zone1.C", "z1/c.scala")):
            _symbol(conn, cid(qn), file=file, qn=qn,
                    line_start=3, line_end=20)
        _edge(conn, cid("zone1.A"), cid("zone1.B"), line=7,
              file="z1/a.scala")
        _edge(conn, cid("zone1.A"), cid("zone1.C"), line=8,
              file="z1/a.scala")
        conn.commit()

        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How do the zone branches work?",
            obligations=obligations_for("q", [
                ("prove both branches", [])]),
            clew_matches=(synthetic_clew_match("zone1.A"),))

        names = {name for card in cards.cards
                 for name in card.node_qualified_names}
        assert "zone1.B" in names
        assert "zone1.C" in names


class TestPreferenceRetention:
    def test_obligation_named_callee_survives_the_per_seed_cap(
            self, conn):
        _symbol(conn, cid("zone1.Hub"), file="z1/hub.scala",
                qn="zone1.Hub", line_start=2, line_end=40)
        for index in range(1, 8):
            _symbol(conn, cid(f"zone1.N{index}"),
                    file=f"z1/n{index}.scala", qn=f"zone1.N{index}",
                    line_start=2, line_end=9)
            _edge(conn, cid("zone1.Hub"), cid(f"zone1.N{index}"),
                  line=index + 2, file="z1/hub.scala")
        _symbol(conn, cid("zone1.WriteTarget"),
                file="z1/write_target.scala", qn="zone1.WriteTarget",
                line_start=2, line_end=9)
        _edge(conn, cid("zone1.Hub"), cid("zone1.WriteTarget"),
              line=20, file="z1/hub.scala")
        conn.commit()

        # Eight callees, cap six by file order: the last-by-line callee
        # only survives because the obligation names it.
        cards = build_route_family_cards(
            conn, source=SOURCE,
            question="How does the hub work?",
            obligations=obligations_for("q", [
                ("prove the write target hookup", [])]),
            clew_matches=(synthetic_clew_match("zone1.Hub"),))

        names = {name for card in cards.cards
                 for name in card.node_qualified_names}
        assert "zone1.WriteTarget" in names
