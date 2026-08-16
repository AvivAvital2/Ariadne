"""The compact production path: three model-call slots, no cascade.

question -> obligation plan -> deterministic family generation -> one
route-family selector -> exact expansion -> materialization ->
formulation. The recorded provider phases are exactly
scip-obligation-plan, scip-route-family-select, completion; every
legacy cascade selector phase is structurally unreachable from the
compact path; budgets refuse — they never truncate; and an unresolved
compact selection fails cheaply, reporting the unresolved obligations,
instead of silently falling back to the legacy path. Synthetic
fixtures only.
"""
from __future__ import annotations

import asyncio
import inspect
import re
import json

import numpy as np

from library.clews import init_clews_schema

import pytest

import ariadne_mcp.service_analysis as service_analysis
from ariadne_mcp.service_analysis import AnalysisMixin
from docgen.catalog_writer import _element_doc_id
from library import Library
from library.scip import init_scip_schema

SOURCE = "src1"
FORBIDDEN_PHASES = (
    "scip-family-select", "scip-route-select", "scip-symbol-select",
    "scip-component-select", "scip-exact-route-select",
    "scip-body-select")


def cid(qn: str) -> str:
    return f"scip-x x src1 0.1 `{qn}`."


def _symbol(conn, canonical_id, *, file, qn, line_start, line_end,
            parent=""):
    conn.execute(
        "INSERT INTO scip_symbols (canonical_id, source_name, language, "
        "file, line_start, line_end, kind, display_name, qualified_name, "
        "parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (canonical_id, SOURCE, "scala", file, line_start, line_end,
         "method", qn.rsplit(".", 1)[-1], qn, parent))


def _edge(conn, caller, callee, *, line, file, edge_type="call"):
    conn.execute(
        "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id,"
        " edge_type, file, line, confidence) VALUES (?,?,?,?,?,'exact')",
        (caller, callee, edge_type, file, line))


def _first_family_reply(prompt: str) -> str:
    """Pick the first listed family for every obligation on the menu."""
    chosen: dict = {}
    current = None
    for line in prompt.splitlines():
        header = re.fullmatch(r"(O\d+):", line.strip())
        if header:
            current = header.group(1)
            continue
        card = re.match(r"\s+(F\d+)\.", line)
        if card and current and current not in chosen:
            chosen[current] = card.group(1)
    return "\n".join(f"{obligation}: {card}"
                     for obligation, card in chosen.items())


class RecordingChat:
    """Scripted provider: records every phase, feeds usage sinks."""

    def __init__(self, replies, *, selector_exhausts_budget=False):
        self.replies = replies
        self.calls = []
        self.selector_exhausts_budget = selector_exhausts_budget

    async def __call__(self, *, messages, **kwargs):
        phase = str(kwargs.get("phase", "completion"))
        self.calls.append(phase)
        sink = kwargs.get("usage_sink")
        max_tokens = kwargs.get("max_tokens")
        if sink is not None:
            output = (max_tokens if self.selector_exhausts_budget
                      else 8)
            sink.append({"finish_reason": "stop",
                         "output_tokens": output,
                         "max_tokens": max_tokens})
        reply = self.replies.get(phase, "")
        if callable(reply):
            return reply(messages[-1]["content"])
        return reply


class _Config:
    def __init__(self, root):
        self._config = {"ask_pipeline": "compact"}
        self._root = root

    def get_all_source_paths(self):
        return {SOURCE: str(self._root)}


class _Service(AnalysisMixin):
    def __init__(self, library, root):
        self.library = library
        self.config = _Config(root)


@pytest.fixture
def service(tmp_path):
    root = tmp_path / "corpus"
    for file, count in (("core/writer.scala", 80),
                        ("core/sink.scala", 25),
                        ("core/deep.scala", 12),
                        ("core/registrar.scala", 20),
                        ("alt/writer.scala", 60)):
        path = root / file
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(
            f"line {index}" for index in range(1, count + 1)) + "\n")

    library = Library(tmp_path / "l.db")
    with library._conn_provider.acquire() as connection:
        init_scip_schema(connection)
        for qn, file, start, end, parent in (
                ("pkg.Writer.write", "core/writer.scala", 10, 60,
                 "pkg.Writer"),
                ("pkg.Writer", "core/writer.scala", 5, 80, ""),
                ("pkg.Sink.flush", "core/sink.scala", 3, 20, "pkg.Sink"),
                ("pkg.Registrar.install", "core/registrar.scala", 4, 18,
                 "pkg.Registrar"),
                ("alt.Writer.write", "alt/writer.scala", 12, 55,
                 "alt.Writer"),
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
    library.add_document(
        content_type="catalog", title="write",
        content="Writes rows to the sink.",
        source_files=["core/writer.scala"],
        doc_id=_element_doc_id(SOURCE, "pkg.Writer.write"),
        source_name=SOURCE)
    yield _Service(library, root)
    library.close()


def _run(service, chat, question="How does the writer flush rows?"):
    diagnostics: dict = {}
    response = asyncio.run(service._ask_compact(
        question, source=SOURCE, notes=(), diagnostics=diagnostics,
        ask_chat=chat, trace=lambda *args, **kwargs: None,
        phase_timings={}))
    return response, diagnostics


class TestCompactPhases:
    def test_exactly_three_calls_with_pinned_phases(self, service):
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": _first_family_reply,
            "completion":
                "The write path calls flush at core/writer.scala:30.",
        })

        response, diagnostics = _run(service, chat)

        assert chat.calls == ["scip-obligation-plan",
                              "scip-route-family-select", "completion"]
        assert not set(FORBIDDEN_PHASES).intersection(chat.calls)
        assert response.llm_calls == 3
        assert "core/writer.scala:30" in response.answer
        compact = diagnostics["compact"]
        assert compact["pipeline"] == "compact"
        assert compact["status"] == "complete"
        assert compact["cards_total"] <= 64
        assert compact["selector_prompt_tokens"] <= 8000
        assert response.selected_symbols

    def test_compact_source_never_names_a_cascade_selector(self):
        compact_source = inspect.getsource(AnalysisMixin._ask_compact)
        assert not any(phase in compact_source
                       for phase in FORBIDDEN_PHASES)

    def test_gate_precedes_every_cascade_selector_in_ask(self):
        module_source = inspect.getsource(service_analysis)
        gate = module_source.index("ask_pipeline")
        assert all(
            gate < module_source.index(f'"{phase}"')
            for phase in FORBIDDEN_PHASES)


class TestCheapFailure:
    def test_unresolvable_obligation_fails_before_the_selector(
            self, service):
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that ghost.Missing.thing runs",
        })

        response, diagnostics = _run(
            service, chat, question="Does the ghost module run cleanup?")

        assert chat.calls == ["scip-obligation-plan"]
        assert response.answer.startswith(
            "Compact selection could not assemble")
        assert "O1" in response.answer
        assert response.confidence == "low"
        assert diagnostics["compact"]["status"] == "no-cards"

    def test_unparseable_plan_fails_without_any_further_call(
            self, service):
        chat = RecordingChat({
            "scip-obligation-plan": "I could not plan anything.",
        })

        response, diagnostics = _run(service, chat)

        assert chat.calls == ["scip-obligation-plan"]
        assert diagnostics["compact"]["status"] == "no-obligations"
        assert response.confidence == "low"


class TestBudgetRefusals:
    def test_selector_budget_refuses_before_the_selector_call(
            self, service, monkeypatch):
        monkeypatch.setattr(
            "library.route_families.render_family_selector_prompt",
            lambda cards, *, question: "menu " * 10000)
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
        })

        response, diagnostics = _run(service, chat)

        assert chat.calls == ["scip-obligation-plan"]
        assert diagnostics["compact"]["status"] == (
            "selector-budget-exceeded")
        assert response.confidence == "low"

    def test_formulation_budget_refuses_never_truncates(
            self, service, monkeypatch):
        monkeypatch.setattr(
            "library.chain_story.render_formulation_spine",
            lambda story_ir: "spine " * 25000)
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": _first_family_reply,
        })

        response, diagnostics = _run(service, chat)

        assert chat.calls == ["scip-obligation-plan",
                              "scip-route-family-select"]
        assert diagnostics["compact"]["status"] == (
            "formulation-budget-exceeded")
        assert "20k budget" in response.answer


class TestTruncationSignal:
    def test_truncated_selector_keeps_reserve_and_reports_unresolved(
            self, service):
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": _first_family_reply,
            "completion": "The write path flushes.",
        }, selector_exhausts_budget=True)

        response, diagnostics = _run(service, chat)

        assert chat.calls == ["scip-obligation-plan",
                              "scip-route-family-select", "completion"]
        assert diagnostics["compact"]["unresolved_obligations"] == ["O1"]
        assert response.confidence == "low"
        assert "unresolved obligation O1" in response.confidence_reasons


class TestDiagnosticCardRecord:
    def test_card_identities_are_recorded_before_the_selector_fires(
            self, service):
        seen_at_selector: dict = {}
        diagnostics: dict = {}

        def selector_reply(prompt):
            seen_at_selector["cards"] = [
                dict(card)
                for card in diagnostics["compact"].get("cards", ())]
            return _first_family_reply(prompt)

        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": selector_reply,
            "completion": "The write path flushes.",
        })
        asyncio.run(service._ask_compact(
            "How does the writer flush rows?", source=SOURCE, notes=(),
            diagnostics=diagnostics, ask_chat=chat,
            trace=lambda *args, **kwargs: None, phase_timings={}))

        assert seen_at_selector["cards"], "cards not recorded pre-selector"
        nodes = [tuple(node) for card in seen_at_selector["cards"]
                 for node in card["nodes"]]
        assert ("pkg.Writer.write", "core/writer.scala", 10, 60) in nodes
        assert diagnostics["compact"]["reserve_by_obligation"]["O1"]


class FakeEmbedding:
    def __init__(self, vector):
        self.vector = vector
        self.calls = 0

    async def embed(self, text):
        self.calls += 1
        return self.vector


class TestSemanticClewSeeds:
    def test_embedded_clew_route_reaches_the_menu(self, service):
        vector = np.ones(8, dtype=np.float32)
        with service.library._conn_provider.acquire() as connection:
            init_clews_schema(connection)
            connection.execute(
                "INSERT INTO clews (id, source_name, entry_symbol, "
                "steps, route, files, strategy, question, embedding) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                ("c1", SOURCE, "pkg.Deep.helper", json.dumps([]),
                 json.dumps(["pkg.Deep.helper"]),
                 json.dumps(["core/deep.scala"]), "test",
                 "How does the writer flush rows?", vector.tobytes()))
            connection.commit()
        service.embedding_service = FakeEmbedding(vector)
        chat = RecordingChat({
            "scip-obligation-plan": "C1: prove the flush path",
            "scip-route-family-select": _first_family_reply,
            "completion": "The helper flushes.",
        })

        response, diagnostics = _run(service, chat)

        compact = diagnostics["compact"]
        assert compact["embedding_calls"] == 1
        assert compact["clew_matches"] >= 1
        nodes = [tuple(node) for card in compact["cards"]
                 for node in card["nodes"]]
        assert any(node[0] == "pkg.Deep.helper" for node in nodes)
        assert service.embedding_service.calls == 1

    def test_no_embedding_service_keeps_the_lexical_path(self, service):
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": _first_family_reply,
            "completion": "The write path flushes.",
        })

        response, diagnostics = _run(service, chat)

        assert diagnostics["compact"]["embedding_calls"] == 0
        assert diagnostics["compact"]["status"] == "complete"


class TestBoundaryTrace:
    def test_trace_records_every_boundary_with_reasons(self, service):
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": _first_family_reply,
            "completion": "The write path flushes.",
        })
        diagnostics: dict = {}
        asyncio.run(service._ask_compact(
            "How does the writer flush rows?", source=SOURCE, notes=(),
            diagnostics=diagnostics, ask_chat=chat,
            trace=lambda *args, **kwargs: None, phase_timings={},
            deep_trace=True))

        trace = diagnostics["compact"]["trace"]
        assert "O1" in trace
        assert "shortlist" in trace["O1"]
        assert "lexical_order" in trace["O1"]
        assert "expansion_reasons" in trace["O1"]
        assert isinstance(
            trace.get("required_body_extents"), list)
        assert isinstance(trace.get("chunk_extents"), list)

    def test_family_drops_name_their_cap(self, service):
        with service.library._conn_provider.acquire() as connection:
            for index in range(1, 9):
                _symbol(connection, cid(f"crowd.M{index}"),
                        file=f"crowd/m{index}.scala",
                        qn=f"crowd.M{index}", line_start=2, line_end=9)
                _edge(connection, cid("pkg.Writer.write"),
                      cid(f"crowd.M{index}"), line=30 + index,
                      file="core/writer.scala")
            connection.commit()
        chat = RecordingChat({
            "scip-obligation-plan":
                "C1: prove that pkg.Writer.write calls pkg.Sink.flush",
            "scip-route-family-select": _first_family_reply,
            "completion": "The write path flushes.",
        })
        diagnostics: dict = {}
        asyncio.run(service._ask_compact(
            "How does the writer flush rows?", source=SOURCE, notes=(),
            diagnostics=diagnostics, ask_chat=chat,
            trace=lambda *args, **kwargs: None, phase_timings={},
            deep_trace=True))

        dropped = diagnostics["compact"]["trace"]["O1"].get(
            "dropped", [])
        assert all(
            set(entry) == {"entry", "terminal", "reason"}
            for entry in dropped)
