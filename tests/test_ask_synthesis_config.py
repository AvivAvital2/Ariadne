"""``ask`` LLM-synthesis config gate (``ask_synthesis``) — TDD.

Contract for the ``ask_synthesis`` flag (default ON, disable-able):

- ``Config.ask_synthesis`` defaults to ``True`` and honors
  ``ask_synthesis: false`` in ``ariadne.yaml``.
- With synthesis ON and an API key present, ``ask`` calls
  ``chat_complete`` with a ``messages`` list and the synthesized text
  flows into ``AskResponse.answer``.
- With synthesis OFF, ``ask`` returns retrieval-only docs and does NOT
  call ``chat_complete`` — no autonomous LLM call / spend, even when a
  key is present (the config gate wins over key presence).

RED today for two reasons: ``Config.ask_synthesis`` doesn't exist yet,
and the ``ask`` call site passes ``system_prompt=/user_prompt=`` to a
``chat_complete`` that now takes ``messages=[...]`` (so a
signature-faithful stub raises ``TypeError``, which the bare ``except``
masks as "unavailable").

Fixtures are synthetic and source/name-agnostic.
"""
from __future__ import annotations
import inspect

import hashlib
from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest

from ariadne_mcp.service_analysis import AnalysisMixin
from ariadne_mcp.service_search import SearchMixin
from config import Config, get_config
from library import Library
from scope_resolution import make_scoped_library
from tests._scoped_config_fixture import install_test_config

_DIM = 3072  # matches schema.EMBEDDING_DIM


def _unit_vec(seed: int, dim: int = _DIM) -> np.ndarray:
    v = np.zeros(dim, dtype=np.float32)
    v[seed % dim] = 1.0
    return v


def _write_cfg(tmp_path: Path, body: str) -> Path:
    p = tmp_path / 'ariadne.yaml'
    p.write_text(body, encoding='utf-8')
    return p


# --------------------------------------------------------------------------
# Config flag — default on, disable-able via ariadne.yaml
# --------------------------------------------------------------------------


def test_ask_synthesis_defaults_true(tmp_path):
    assert Config(_write_cfg(tmp_path, 'sources: {}\n')).ask_synthesis is True


def test_ask_synthesis_respects_false(tmp_path):
    assert Config(_write_cfg(tmp_path, 'ask_synthesis: false\n')).ask_synthesis is False


# --------------------------------------------------------------------------
# ask() behavior — gated on the flag
# --------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cfg(monkeypatch, tmp_path):
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path):
    lib = Library(tmp_path / 'ask.db')
    orig = lib.add_document

    def add(*a, **k):
        k.setdefault('source_name', 'test')
        return orig(*a, **k)

    lib.add_document = add
    yield lib
    lib.close()


class _StableEmbedder:
    async def embed(self, text):
        h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        return _unit_vec(h)


class _Svc(SearchMixin, AnalysisMixin):
    @staticmethod
    def _cache_key(*a, **k):
        return hash((a, tuple(sorted(k.items()))))

    def get_branch(self):
        return None

    def _resolve_scope(self, source):
        return make_scoped_library(self.config, self.library, source or 'test')


@pytest.fixture
def service(library):
    svc = _Svc()
    svc.library = library
    svc.config = get_config()
    svc._query_cache = {}
    svc.embedding_service = _StableEmbedder()
    return svc


def _seed(library):
    library.add_document(
        content_type='explanation',
        title='Widget Guide',
        content='The widget subsystem batches records before flush.',
        source_files=[],
        embedding=_unit_vec(1),
        metadata={},
    )
def _seed_chain(library):
    """A retrievable document over a file SCIP knows, so ``ask`` actually has a chain.

    ``_seed`` alone cannot exercise the chain path: its document names no source file, so
    nothing localizes into the graph and the walk has no seed.
    """
    from docgen.catalog_writer import _element_doc_id
    from library.scip import init_scip_schema

    with library._conn_provider.acquire() as conn:
        init_scip_schema(conn)
        for canonical_id, qualified_name, line in (
            ('sym-run', 'w.Widget.run', 5),
            ('sym-flush', 'w.Widget.flush', 20),
        ):
            conn.execute(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
                'line_start, line_end, kind, display_name, qualified_name, '
                'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
                (canonical_id, 'test', 'python', 'w.py', line, line + 8, '', '',
                 qualified_name, 'w.Widget'))
        conn.execute(
            'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, edge_type, '
            "file, line, confidence) VALUES ('sym-run','sym-flush','call','w.py',7,'exact')")
        conn.commit()
    for qualified_name, description in (
        ('w.Widget.run', 'Runs the widget end to end.'),
        ('w.Widget.flush', 'Flushes the batch to storage.'),
    ):
        library.add_document(
            content_type='catalog', title=qualified_name.rsplit('.', 1)[-1],
            content=f'python_method {qualified_name} in w [python] w.py'
                    f'\n\nDescription: {description}',
            source_files=['w.py'], doc_id=_element_doc_id('test', qualified_name),
            embedding=_unit_vec(3), metadata={})
    library.add_document(
        content_type='explanation', title='Widget Guide',
        content='The widget subsystem batches records before flush.',
        source_files=['w.py'], embedding=_unit_vec(1), metadata={})


@pytest.mark.asyncio
async def test_ask_disabled_returns_docs_without_calling_llm(
    service, library, monkeypatch,
):
    """Synthesis OFF ⇒ retrieval-only answer and no ``chat_complete``
    call — even with a key present, so the config gate wins over the
    key, not the other way around."""
    _seed(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    service.config._config['ask_synthesis'] = False
    fake = AsyncMock(return_value='SHOULD-NOT-APPEAR')
    monkeypatch.setattr('llm.chat_complete', fake)

    result = await service.ask(question='How does the widget subsystem work?')

    fake.assert_not_called()
    assert 'SHOULD-NOT-APPEAR' not in result.answer
    assert 'Widget Guide' in result.answer  # retrieval content still surfaced


@pytest.mark.asyncio
async def test_ask_enabled_synthesizes_via_messages(
    service, library, monkeypatch,
):
    """Synthesis ON + key present ⇒ ``chat_complete`` invoked with a
    ``messages`` list and its text reaches ``AskResponse.answer``."""
    _seed(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    service.config._config['ask_synthesis'] = True
    captured: dict = {}

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0, **kwargs):
        captured['messages'] = messages
        return 'SYNTHESIZED-ANSWER'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?')

    assert 'SYNTHESIZED-ANSWER' in result.answer
    roles = {m['role'] for m in captured['messages']}
    assert {'system', 'user'} <= roles


@pytest.mark.asyncio
async def test_ask_enabled_without_key_returns_docs(service, library, monkeypatch):
    """Synthesis ON but no provider key ⇒ retrieval-only, no LLM call.
    Enabling the flag alone never triggers an autonomous call without a
    key — the 'safe when unconfigured' path."""
    _seed(library)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    service.config._config['ask_synthesis'] = True
    fake = AsyncMock(return_value='SHOULD-NOT-APPEAR')
    monkeypatch.setattr('llm.chat_complete', fake)

    result = await service.ask(question='How does the widget subsystem work?')

    fake.assert_not_called()
    assert 'SHOULD-NOT-APPEAR' not in result.answer
    assert 'Widget Guide' in result.answer


@pytest.mark.asyncio
async def test_ask_synthesis_error_degrades_to_docs(service, library, monkeypatch):
    """A genuine completion failure degrades to retrieval-only rather than
    surfacing the error to the caller."""
    _seed(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    service.config._config['ask_synthesis'] = True

    async def boom(messages, *, model=None, max_tokens=2048, timeout=60.0):
        raise RuntimeError('provider down')

    monkeypatch.setattr('llm.chat_complete', boom)

    result = await service.ask(question='How does the widget subsystem work?')

    assert 'Widget Guide' in result.answer      # retrieval content still returned
    assert 'unavailable' in result.answer        # graceful-degrade marker


@pytest.mark.asyncio
async def test_ask_synthesizes_on_an_anthropic_only_install(
    service, library, monkeypatch,
):
    """The provider gate must be the configured provider, not a hardcoded one.

    ``ask`` checked ``os.environ['OPENAI_API_KEY']`` before synthesizing, but
    ``llm.chat_complete`` routes by ``provider:`` and reads ANTHROPIC_API_KEY
    for Anthropic models. This repo is configured ``provider: anthropic`` /
    ``model: claude-opus-4-8``, so an install holding only an Anthropic key
    silently fell back to dumping retrieval context — synthesis skipped, with
    a perfectly valid key present.

    The redundant pre-check is also the wrong place to fail: ``chat_complete``
    already raises a specific "ANTHROPIC_API_KEY is required" error, which the
    caller's exception path reports.
    """
    _seed(library)
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    monkeypatch.setenv('ANTHROPIC_API_KEY', 'test-key')
    service.config._config['provider'] = 'anthropic'
    service.config._config['model'] = 'claude-opus-4-8'
    fake = AsyncMock(return_value='SYNTHESIZED-BY-ANTHROPIC')
    monkeypatch.setattr('llm.chat_complete', fake)

    result = await service.ask(question='How does the widget subsystem work?')

    fake.assert_called_once()
    assert 'SYNTHESIZED-BY-ANTHROPIC' in result.answer
@pytest.mark.asyncio
async def test_ask_offers_a_menu_then_prunes_unaccounted_claims(
    service, library, monkeypatch,
):
    """One bounded repair removes claims outside the compiler evidence."""
    _seed_chain(library)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    service.config._config["ask_synthesis"] = True
    calls: list = []

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0, **kwargs):
        calls.append(messages)
        if len(calls) <= 2:
            return "R1"
        if len(calls) == 3:
            return "Supported at w.py:20.\nInvented at invented.py:999."
        return "Supported at w.py:20."

    monkeypatch.setattr("llm.chat_complete", fake_cc)

    result = await service.ask(question="How does the widget subsystem work?",
                               source="test")
    assert result.answer.startswith("Supported at w.py:20.")
    assert "w.Widget.run calls w.Widget.flush" in result.answer
    assert "invented.py" not in result.answer
    assert len(calls) == 4, 'module menu, route menu, formulation, then one repair'
    menu_prompt = "\n".join(m["content"] for m in calls[0] if m["role"] == "user")
    answer_prompt = "\n".join(m["content"] for m in calls[2] if m["role"] == "user")
    answer_system = "\n".join(m["content"] for m in calls[2] if m["role"] == "system")
    assert 'SCIP CONNECTED COMPONENTS' in menu_prompt
    assert 'SCIP CONNECTED COMPONENTS' not in answer_prompt
    assert "w.py:20" in answer_prompt and "w.py:7" in answer_prompt
    assert "file:line" in answer_system
    assert result.citations
    assert result.chain_citations
    assert result.unsupported_locations == []
    assert result.evidence_gaps == []
    assert result.claims[0] == {"text": "Supported at w.py:20.", "locations": ["w.py:20"], "supported": True}
    assert any("w.Widget.run calls w.Widget.flush" in claim["text"] for claim in result.claims)
    assert result.confidence == "low"
    assert any(reason.startswith("source materialized for 0/") for reason in result.confidence_reasons)
    assert result.chain_summary["hops"] >= 1
    assert result.chain_complete is False
    assert result.completeness_reasons
    assert result.formulation_complete is True
    assert result.scope_complete is True, result.scope_reasons
    assert result.scope_reasons == []
    assert result.formulation_reasons == []
    assert result.chain_summary["transitions"]["unaccounted"] == 0


@pytest.mark.asyncio
async def test_ask_falls_back_to_the_whole_chain_when_nothing_is_chosen(
    service, library, monkeypatch,
):
    """A pick that resolves to nothing must cost tokens, never evidence.

    The selection is additive: if the model names nothing the menu recognises, the answer
    call carries the chain as it did before the menu existed, rather than an empty prompt.
    """
    _seed_chain(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    service.config._config['ask_synthesis'] = True
    calls: list = []

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0, **kwargs):
        prompt = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
        calls.append(prompt)
        return 'nothing looks relevant' if len(calls) == 1 else 'ANSWER at w.py:20'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?',
                               source='test')

    assert 'ANSWER at w.py:20' in result.answer
    assert len(calls) == 3
    assert 'Call chain:' in calls[2], 'the full chain travels when nothing was chosen'
@pytest.mark.asyncio
async def test_ask_expands_a_bare_line_reference_in_the_answer(
    service, library, monkeypatch,
):
    """The caller receives `MergeIntoCommand.scala:166`, not `:166`.

    Done after the fact rather than by prompting: repeating the file name costs tokens on
    every answer and relies on the model complying. It also matters for correctness — a bare
    line reference does not match the location pattern, so before this it was neither checked
    nor returned as a citation.
    """
    _seed_chain(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    service.config._config['ask_synthesis'] = True
    calls: list = []

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            return "R1"
        return 'Runs at w.py:20 (and again at :7).'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?',
                               source='test')

    assert 'w.py:20' in result.answer, result.answer
    assert '(and again at :20)' not in result.answer
@pytest.mark.asyncio
async def test_ask_without_localization_returns_an_honest_empty_response(
    service, monkeypatch,
):
    fake = AsyncMock(return_value='MUST NOT RUN')
    monkeypatch.setattr('llm.chat_complete', fake)

    result = await service.ask(question='Where is the absent mechanism?', source='test')

    fake.assert_not_called()
    assert result.confidence == 'low'
    assert 'No relevant documentation found' in result.answer
    assert result.chain_citations == []
    assert result.unsupported_locations == []
@pytest.mark.asyncio
async def test_menu_failure_cannot_remove_the_compiler_spine(
    service, library, monkeypatch,
):
    _seed_chain(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    calls: list = []

    async def fake_cc(messages, **kwargs):
        calls.append(messages)
        if len(calls) == 1:
            raise RuntimeError('selector unavailable')
        prompt = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
        assert 'w.py:7' in prompt and 'flush' in prompt
        return 'ANSWER at w.py:20'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?',
                               source='test')

    assert 'ANSWER at w.py:20' in result.answer
    assert result.chain_citations
    assert result.route_selection_status == 'error-fallback-hydrated'
    assert result.hydrated_symbols
    assert result.unsupported_locations == []
@pytest.mark.asyncio
async def test_ask_drops_unproved_claims_when_the_single_repair_still_fails(
    service, library, monkeypatch,
):
    _seed_chain(library)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls: list = []

    async def fake_cc(messages, **kwargs):
        calls.append(messages)
        if len(calls) <= 2:
            return "R1"
        return "Invented at invented.py:999."

    monkeypatch.setattr("llm.chat_complete", fake_cc)

    result = await service.ask(question="How does the widget subsystem work?",
                               source="test")

    assert len(calls) == 4
    assert "invented.py" not in result.answer
    assert result.claims
    assert all(claim["supported"] for claim in result.claims)
    assert any("w.Widget.run" in claim["text"] for claim in result.claims)
    assert result.unsupported_locations == []
    assert result.evidence_gaps == ["unsupported location: invented.py:999"]
    assert "w.Widget.run calls w.Widget.flush" in result.answer
@pytest.mark.asyncio
async def test_chain_synthesis_failure_cannot_retain_chain_confidence(
    service, library, monkeypatch,
):
    _seed_chain(library)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = 0

    async def fake_cc(messages, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return "R1"
        raise RuntimeError("prompt rejected")

    monkeypatch.setattr("llm.chat_complete", fake_cc)
    result = await service.ask(question="How does the widget subsystem work?",
                               source="test")

    assert result.confidence == "low"
    assert result.confidence_reasons == ["synthesis failed: RuntimeError"]
    assert result.citations == []
    assert result.chain_citations
def test_transition_ledger_wiring_is_not_duplicated():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    assert source.count("transition_supported = []") == 1
    assert source.count(
        "_transition_ledger = transition_claims(_formulation_evidence)") == 1
def test_ask_applies_route_coverage_after_the_selector():
    import inspect
    source = inspect.getsource(AnalysisMixin.ask)
    assert source.index('resolve_component_selection(components, reply)') < source.index(
        "complete_route_selection(menu, selection, question)")
def test_ask_keeps_document_positioning_bounded_after_broad_pool_regression():
    import inspect

    source = inspect.getsource(AnalysisMixin.ask)
    assert "limit=8" in source
    assert "top_docs = _balanced_ask_docs" in source
    assert "evidence_for, self.library, positioning_docs" in source
    assert "clew_route_menu" in source
    assert "_clew_matches[:2]" in source
def test_ask_selected_routes_retain_semantic_positioning_for_catalog_holes():
    import inspect

    source = inspect.getsource(AnalysisMixin.ask)
    selected = source.index("_clew_matches = (_clew_matches[:2] if _route_selection_failed")
    evidence = source.index("evidence_for, self.library, positioning_docs", selected)
    between = source[selected:evidence]
    assert "positioning_docs = []" not in between
def test_story_placeholders_expand_before_initial_claim_validation():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    formulation = source.index("answer = await _ask_chat")
    expansion = source.index("answer = expand_story_placeholders", formulation)
    validation = source.index("ledger = validate_claims", formulation)

    assert expansion < validation
def test_route_local_sections_receive_question_embedding_scores():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    score = source.index("route_section_embedding_scores,")
    modules = source.index('resolve_component_selection(', score)
    scope = source.index("menu = scope_route_menu(", modules)
    select = source.index("select_route_sections(", scope)

    assert score < scope < select
    assert "section_scores=_section_scores" in source[select:select + 300]
def test_scope_diagnostics_cross_the_ask_response_boundary():
    model_source = Path("ariadne_mcp/models.py").read_text()
    service_source = Path("ariadne_mcp/service_analysis.py").read_text().replace(" ", "")
    arm_source = Path("evaluation/spool-clean-room/ariadne_arm.py").read_text().replace("\'", '\"')

    for field in ("route_scope_total", "route_scope_retained", "section_candidates"):
        assert field in model_source
        assert f"{field}={field}" in service_source
        assert f'"{field}"' in arm_source
def test_ask_selects_modules_before_expanding_and_scoping_routes():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    build = source.index('component_menu_for(graph, menu)')
    choose = source.index('resolve_component_selection(', build)
    expand = source.index("routes_for_modules(", choose)
    scope = source.index("scope_route_menu(", expand)

    assert build < choose < expand < scope
def test_ask_selects_exact_routes_after_module_expansion_and_scope():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    modules = source.index('resolve_component_selection(')
    scope = source.index("scope_route_menu(", modules)
    route_prompt = source.index('_menu_prompt(question, menu.text, _coverage_plan)', scope)
    route_select = source.index('resolve_obligation_route_selection(menu, route_reply)', route_prompt)
    hydrate = source.index("hydrate_selected_hops(", route_select)

    assert modules < scope < route_prompt < route_select < hydrate
def test_ask_retains_selected_owner_routes_after_exact_route_selection():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    exact = source.index('resolve_obligation_route_selection(menu, route_reply)')
    retain = source.index('merge_selections(', exact)
    complete = source.index("complete_route_selection(", retain)

    assert exact < retain < complete
def test_ask_anchors_and_retains_exact_method_cards():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    resolve = source.index('resolve_component_selection(')
    scope = source.index('scope_route_menu(', resolve)
    exact = source.index('resolve_obligation_route_selection(menu, route_reply)', scope)
    retain = source.index('merge_selections(', exact)

    assert resolve < scope < exact < retain
def test_ask_selects_connected_graph_closure_before_hydration():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    build = source.index("evidence_graph_for(hops)")
    menu = source.index('component_menu_for(graph, menu)', build)
    seeds = source.index('selection_for_graph_symbols(graph, selection.symbols, occurrence_keys = selection.occurrence_keys)', menu)
    merge = source.index("merge_selections(selection, graph_selection)", seeds)
    hydrate = source.index("hydrate_selected_hops(", merge)

    assert build < menu < seeds < merge < hydrate
def test_ask_selects_components_before_routes_then_closes_the_graph():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    graph = source.index("evidence_graph_for(hops)")
    components = source.index("component_menu_for(graph, menu)", graph)
    choose_components = source.index("resolve_component_selection(", components)
    choose_routes = source.index('resolve_obligation_route_selection(menu, route_reply)', choose_components)
    closure = source.index('selection_for_graph_symbols(graph, selection.symbols, occurrence_keys = selection.occurrence_keys)', choose_routes)

    assert graph < components < choose_components < choose_routes < closure
def test_ask_builds_graph_diagnostics_before_synthesis_guard():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    report = source.index("evidence_graph_report(evidence_graph_for(_evidence.hops), _evidence.seed_provenance)")
    guard = source.index("if not self.config.ask_synthesis:", report)
    response = source.index('graph_diagnostics = _graph_diagnostics', guard)

    assert report < guard < response
def test_graph_inspector_bootstraps_repository_import_path():
    source = Path("evaluation/chain-benchmark/inspect_graph.py").read_text()
    root = source.index("ROOT = HERE.parent.parent")
    path = source.index("sys.path.insert(0, str(ROOT))", root)
    service = source.index("from ariadne_mcp.service import AriadneService", path)

    assert root < path < service
def test_obligations_are_frozen_before_the_family_catalog_is_shown():
    source = inspect.getsource(AnalysisMixin.ask)
    obligation = source.index('phase="scip-obligation-plan"')
    family_menu = source.index('_family_menu.text', obligation)
    family_select = source.index('phase="scip-family-select"', family_menu)

    assert obligation < family_menu < family_select
    obligation_call = source[source.rfind('_obligation_reply =', 0, obligation):obligation]
    assert '_family_menu.text' not in obligation_call
    assert 'Do not name symbols not present in the question' in obligation_call
    family_call = source[source.rfind('_family_reply =', 0, family_select):family_select]
    assert 'FIXED OBLIGATIONS' in family_call
    assert '_coverage_plan' in family_call


def test_family_catalog_is_semantically_ranked_and_scope_isolated():
    source = inspect.getsource(AnalysisMixin.ask)
    call = source[source.index('_family_menu = clew_family_menu'):]

    assert 'question=question, limit=200' in call[:300]
def test_obligation_protocol_rejects_premises_and_duplicate_causal_claims():
    source = inspect.getsource(AnalysisMixin.ask)

    assert "distinct source-code chain" in source
    assert "merely mentioned entities, premises, or target types" in source
    assert "Combine restatements of the same cause, action, or effect" in source
    assert "Prefer 3-6 complete obligations" in source


def test_route_protocol_requires_id_only_lines_and_reserves_full_mapping_output():
    source = inspect.getsource(AnalysisMixin.ask)
    route = source.index('phase="scip-route-select"')
    call = source[source.rfind('_route_reply =', 0, route):route]

    assert "no prose" in call
    assert "max_tokens=768" in call
@pytest.mark.asyncio
async def test_deterministic_selection_uses_one_formulation_call_and_reports_phases(
    service, library, monkeypatch,
):
    """The warm SCIP path spends one remote call: grounded formulation only."""
    _seed_chain(library)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ARIADNE_BENCHMARK_NO_REPAIR", "1")
    service.config._config["ask_synthesis"] = True
    service.config._config["ask_selector_mode"] = "deterministic"
    calls = []

    async def fake_cc(messages, **kwargs):
        calls.append(messages)
        return "The batch is flushed at w.py:20."

    monkeypatch.setattr("llm.chat_complete", fake_cc)

    result = await service.ask(
        question="How does the widget subsystem work?", source="test")

    assert len(calls) == 1
    prompt = "\n".join(
        message["content"] for message in calls[0] if message["role"] == "user")
    assert "Call chain:" in prompt
    assert "w.py:7" in prompt and "w.py:20" in prompt, (result.selected_symbols, result.selected_body_symbols, result.hydrated_symbols, result.selected_route_ids)
    assert result.route_selection_status == "deterministic"
    assert result.selected_route_ids
    assert result.llm_calls == 1
    assert {"search", "chain_assembly", "evidence_selection", "formulation", "total"} <= set(
        result.phase_timings)
    assert all(value >= 0 for value in result.phase_timings.values())
    assert result.phase_timings["total"] >= result.phase_timings["formulation"]
    post_walk = result.graph_diagnostics["post_walk_selection"]
    assert post_walk["component_menu_chars"] > 0
    assert post_walk["exact_route_menu_chars"] > 0
    assert post_walk["formulation_prompt_chars"] > 0
@pytest.mark.asyncio
async def test_fast_path_telemetry_accounts_for_positioning_recall_and_walk(
        service, library, monkeypatch):
    _seed_chain(library)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ARIADNE_BENCHMARK_NO_REPAIR", "1")
    service.config._config["ask_selector_mode"] = "deterministic"

    async def fake_cc(messages, **kwargs):
        return "The batch is flushed at w.py:20."

    monkeypatch.setattr("llm.chat_complete", fake_cc)

    result = await service.ask(
        question="How does the widget subsystem work?", source="test")

    subphases = {"catalog_positioning", "clew_selection", "evidence_walk"}
    assert subphases <= set(result.phase_timings)
    assert result.phase_timings["chain_assembly"] >= sum(
        result.phase_timings[name] for name in subphases)
    assert result.graph_diagnostics
@pytest.mark.asyncio
async def test_benchmark_trace_reports_real_ask_phase_boundaries(
        service, library, monkeypatch, capsys):
    _seed_chain(library)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("ARIADNE_BENCHMARK_NO_REPAIR", "1")
    service.config._config["ask_synthesis"] = True
    service.config._config["ask_selector_mode"] = "deterministic"

    async def fake_cc(messages, **kwargs):
        return "The batch is flushed at w.py:20."

    monkeypatch.setattr("llm.chat_complete", fake_cc)

    await service.ask(
        question="How does the widget subsystem work?", source="test",
        trace_id=6)

    trace = capsys.readouterr().err
    for phase in (
            "ask", "search", "catalog_positioning", "clew_selection",
            "evidence_walk", "evidence_selection", "formulation"):
        assert f"[Q6] {phase} start" in trace
    for phase in (
            "search", "catalog_positioning", "clew_selection",
            "evidence_walk", "evidence_selection", "formulation"):
        assert f"[Q6] {phase} end" in trace
    assert trace.count("[Q6] catalog_positioning start") == 1
def test_definition_body_cards_are_selected_before_source_hydration():
    import inspect
    from ariadne_mcp.service_analysis import AnalysisMixin
    source = inspect.getsource(AnalysisMixin.ask)

    cards = source.index("definition_body_menu(")
    selection = source.index('phase="scip-body-select"', cards)
    hydration = source.index("hydrate_selected_hops(", selection)

    assert cards < selection < hydration
    assert "definition_body_symbols=tuple(selected_body_symbols)" in source[hydration:]
def test_ask_preserves_llm_phase_for_usage_accounting():
    import inspect
    from ariadne_mcp.service_analysis import AnalysisMixin
    source = inspect.getsource(AnalysisMixin.ask)

    assert 'kwargs.get("phase", None)' in source
    assert 'kwargs.pop("phase", None)' not in source
def test_llm_route_selection_sees_routes_before_deterministic_scope_pruning():
    source = Path("ariadne_mcp/service_analysis.py").read_text()
    expand = source.index("routes_for_modules(")
    scope = source.index("menu = scope_route_menu(", expand)
    guard = source.rfind('if _selector_mode == "deterministic":', expand, scope)
    prompt = source.index('_menu_prompt(question, menu.text, _coverage_plan)', scope)

    assert expand < guard < scope < prompt
def test_exact_route_prompt_uses_fixed_obligations_and_minimal_routes():
    from ariadne_mcp.service_analysis import _menu_prompt

    prompt = _menu_prompt("How does it flow?", "R1. route", "C1: entry to sink")

    assert "C1: entry to sink" in prompt
    assert "smallest complete set" in prompt
    assert "missing a relevant route is worse" not in prompt
    assert "C<n>: R<id>" in prompt


def test_ask_preserves_obligations_for_exact_route_selection():
    import inspect
    source = inspect.getsource(AnalysisMixin.ask)

    exact_prompt = source.index("_menu_prompt(question, menu.text, _coverage_plan)")
    evidence = source.index("hydrate_selected_hops(", exact_prompt)
    assert "_coverage_plan = \"\"" not in source[source.index("_selected_clews = []"):exact_prompt]
    assert "resolve_obligation_route_selection" in source[exact_prompt:evidence]


def test_ask_does_not_restore_every_unselected_mandatory_alternative():
    import inspect
    source = inspect.getsource(AnalysisMixin.ask)

    exact = source.index("resolve_obligation_route_selection")
    hydrate = source.index("hydrate_selected_hops(", exact)
    assert "retain_mandatory_routes(menu, selection)" not in source[exact:hydrate]


def test_selection_calls_have_distinct_usage_phases():
    import inspect
    source = inspect.getsource(AnalysisMixin.ask)

    assert 'phase="scip-component-select"' in source
    assert 'phase="scip-exact-route-select"' in source
def test_component_prompt_maps_fixed_obligations_to_graph_components():
    from ariadne_mcp.service_analysis import _component_prompt

    prompt = _component_prompt(
        "Compare both paths", "G1. first\nG2. second",
        "C1: first path\nC2: second path")

    assert "FIXED OBLIGATIONS" in prompt
    assert "C1: first path" in prompt
    assert "C<n>: G<id>" in prompt
    assert "G-prefixed IDs" in prompt


def test_ask_passes_fixed_obligations_to_component_selection():
    import inspect
    source = inspect.getsource(AnalysisMixin.ask)

    assert "_component_prompt(question, components.text, _coverage_plan)" in source
def test_post_walk_selection_diagnostics_capture_each_scope_boundary():
    import inspect
    source = inspect.getsource(AnalysisMixin.ask)

    assert '"component_plan"' in source
    assert '"component_routes"' in source
    assert '"expanded_routes"' in source
    assert '"exact_route_plan"' in source
    assert '"selected_routes"' in source
    assert '"body_candidates"' in source
    assert '"selected_bodies"' in source
def test_ask_completes_compiler_transition_bodies_after_model_selection():
    import inspect
    from ariadne_mcp.service_analysis import AnalysisMixin

    source = inspect.getsource(AnalysisMixin.ask)
    selection = source.index("resolve_definition_body_selection(")
    completion = source.index(
        "complete_definition_body_selection(", selection)
    hydration = source.index("hydrate_selected_hops(", completion)

    assert selection < completion < hydration
def test_chain_formulation_requires_exact_source_syntax_not_paraphrased_code():
    from ariadne_mcp.service_analysis import _chain_prompt

    prompt = _chain_prompt(
        "How is the value emitted?",
        "source definition_body [flow.py:4-8]: value = emit(record)")

    assert "short verbatim source excerpts" in prompt
    assert "Do not simplify or rewrite quoted source syntax" in prompt
def test_catalog_positioning_reuses_the_service_embedding_matrix():
    source = inspect.getsource(AnalysisMixin.ask)
    call = source[source.index("_catalog_docs = catalog_positioning_documents("):]
    call = call[:call.index(")\n", call.index("_catalog_docs")) + 2]

    assert "matrix_provider=self._get_embedding_matrix" in "".join(call.split())
def test_ask_preserves_unreferenced_story_proof_without_sending_it_through_repair():
    source = Path("ariadne_mcp/service_analysis.py").read_text()

    assert source.count(
        "answer = expand_story_placeholders(answer, story_ir, strict = False)"
    ) == 2
    assert source.count(
        "_proof_appendix = render_unreferenced_story_evidence(answer, story_ir)"
    ) == 2
    assert source.count("answer += _proof_appendix") == 1
    first_appendix = source.index(
        "_proof_appendix = render_unreferenced_story_evidence(answer, story_ir)")
    first_expansion = source.index(
        "answer = expand_story_placeholders(answer, story_ir, strict = False)")
    first_validation = source.index(
        "ledger = validate_claims(answer + _proof_appendix, _formulation_evidence)")
    repair = source.index("repair_prompt(answer, prompt, ledger)")
    final_attach = source.index("answer += _proof_appendix")
    assert first_appendix < first_expansion < first_validation < repair < final_attach
def test_live_hydration_uses_the_literal_question_for_reference_ranking():
    source = Path("ariadne_mcp/service_analysis.py").read_text().replace(" ", "")
    assert "reference_query=question" in source
def test_ask_reports_indexed_declarations_from_selected_exact_source():
    import inspect
    from ariadne_mcp.service_analysis import AnalysisMixin

    source = inspect.getsource(AnalysisMixin.ask)
    assert source.count("indexed_symbols_covered_by_source(") == 2
    first_story = source.index("story_ir = build_story_ir(")
    first_proven = source.index("indexed_symbols_covered_by_source(", first_story)
    first_hydrated = source.index("hydrated_symbols =", first_proven)
    second_story = source.index("story_ir = build_story_ir(", first_story + 1)
    second_proven = source.index("indexed_symbols_covered_by_source(", second_story)
    second_hydrated = source.index("hydrated_symbols =", second_proven)
    assert first_story < first_proven < first_hydrated
    assert second_story < second_proven < second_hydrated
