"""``ask`` resolves the question against the clew index and hands the route to the walk.

The unit contract is asserted next door: :mod:`tests.test_clews_store` shows a clew alone can
seed a walk with no documents. What that cannot show is whether ``ask`` actually *looks* —
the lookup lives inside a broad ``try`` in the answer path, so a wiring mistake would be
swallowed and the arm would quietly fall back to document seeding while appearing to work.
That is exactly the failure this session hit twice: a path that produced plausible output while
the mechanism under it was disconnected.

Two properties, and the second is the one that costs money if it breaks:

* a matched clew's route reaches ``evidence_for``;
* **no embedding call is made when the index holds no embedded clew** — the gate is a one-row
  indexed lookup first, because otherwise every ``ask`` in every store built so far pays a
  provider call to search an empty table.

The embedder is deterministic (a one-hot vector keyed on the text's md5), so a clew can be
stored carrying the exact vector the question will produce and the match is arithmetic rather
than approximate.

Synthetic fixtures only: source ``test``.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

from ariadne_mcp.service_analysis import AnalysisMixin
from ariadne_mcp.service_search import SearchMixin
from config import get_config
from library import Library
from library.chain_answer import AnswerEvidence
from library.clews import add_clew, init_clews_schema
from scope_resolution import make_scoped_library
from tests._scoped_config_fixture import install_test_config

_DIM = 3072  # matches schema.EMBEDDING_DIM
QUESTION = 'how does `m.run` flush records?'
ROUTE = ['m.run', 'm.helper']


def _unit_vec(seed: int, dim: int = _DIM) -> np.ndarray:
    vector = np.zeros(dim, dtype=np.float32)
    vector[seed % dim] = 1.0
    return vector


def _vector_for(text: str) -> np.ndarray:
    return _unit_vec(int(hashlib.md5(text.encode()).hexdigest()[:8], 16))


class _StableEmbedder:
    """Deterministic, and it counts calls so the gate can be asserted rather than assumed."""

    def __init__(self) -> None:
        self.calls: list = []

    async def embed(self, text):
        self.calls.append(text)
        return _vector_for(text)


class _Svc(SearchMixin, AnalysisMixin):
    @staticmethod
    def _cache_key(*a, **k):
        return hash((a, tuple(sorted(k.items()))))

    def get_branch(self):
        return None

    def _resolve_scope(self, source):
        return make_scoped_library(self.config, self.library, source or 'test')


@pytest.fixture(autouse=True)
def _cfg(monkeypatch, tmp_path):
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path):
    lib = Library(tmp_path / 'clewask.db')
    lib.add_document(
        content_type='explanation', title='Widget Guide',
        content='The widget subsystem batches records before flush.',
        source_files=[], embedding=_unit_vec(1), metadata={}, source_name='test')
    return lib


@pytest.fixture
def service(library):
    svc = _Svc()
    svc.library = library
    svc.config = get_config()
    svc._query_cache = {}
    svc.embedding_service = _StableEmbedder()
    return svc
def _store_clew(library, *, embedded: bool) -> None:
    with library._conn_provider.acquire() as conn:
        init_clews_schema(conn)
        add_clew(conn, source_name='test', entry_symbol=ROUTE[0],
                 steps=[name.split('.')[-1] for name in ROUTE], route=ROUTE,
                 files=['m.py'], strategy='theme-walk',
                 embedding=_vector_for(QUESTION) if embedded else None)
        if embedded:
            add_clew(conn, source_name='test', entry_symbol='other.run',
                     steps=['run', 'unrelated'], route=['other.run', 'other.unrelated'],
                     files=['other.py'], strategy='theme-walk',
                     embedding=_vector_for(QUESTION))
        conn.commit()


def _capture(monkeypatch) -> dict:
    """Stand in for the walk and record what the wiring passed it."""
    seen: dict = {}

    def fake_evidence_for(library, documents, **kwargs):
        seen['documents'] = list(documents)
        seen.update(kwargs)
        return AnswerEvidence()

    monkeypatch.setattr('library.chain_answer.evidence_for', fake_evidence_for)
    return seen
@pytest.mark.asyncio
async def test_a_matched_clew_route_reaches_the_walk(service, library, monkeypatch):
    _store_clew(library, embedded=True)
    seen = _capture(monkeypatch)
    monkeypatch.setattr('llm.chat_complete', lambda *a, **k: _answer())

    await service.ask(QUESTION, source='test')

    matches = seen.get('clew_matches')
    assert len(matches) == 1
    assert matches[0].clew.route == ROUTE
    assert matches[0].similarity == pytest.approx(1.0)
    assert 'clew_symbols' not in seen, 'route identity must not be flattened before localization'


@pytest.mark.asyncio
async def test_no_embedding_is_spent_when_no_clew_is_embedded(service, library,
                                                              monkeypatch):
    """A stored-but-unembedded clew must not trigger a provider call.

    Generation is local and embedding is the paid step, so a pack legitimately holds routes
    with no vector. Searching that index would cost one embedding per question and return
    nothing.
    """
    _store_clew(library, embedded=False)
    _capture(monkeypatch)
    monkeypatch.setattr('llm.chat_complete', lambda *a, **k: _answer())
    before = len(service.embedding_service.calls)

    await service.ask(QUESTION, source='test')

    spent = [text for text in service.embedding_service.calls[before:]
             if text == QUESTION]
    assert len(spent) <= 1, (
        'the question is embedded once for retrieval; the clew gate must not embed it again')


async def _answer() -> str:
    return 'K1'
@pytest.mark.asyncio
async def test_the_user_question_reaches_compiler_localization(service, monkeypatch):
    seen = _capture(monkeypatch)
    monkeypatch.setattr("llm.chat_complete", lambda *a, **k: _answer())

    await service.ask(QUESTION, source="test")

    assert seen.get("question") == QUESTION
@pytest.mark.asyncio
async def test_ask_retrieves_a_recall_pool_before_filtering_clews(
        service, library, monkeypatch):
    _store_clew(library, embedded=True)
    _capture(monkeypatch)
    monkeypatch.setattr('llm.chat_complete', lambda *a, **k: _answer())
    seen = {}

    from library import clews
    original = clews.nearest_clew_matches

    def recording_nearest(*args, **kwargs):
        seen['top_k'] = kwargs['top_k']
        return original(*args, **kwargs)

    monkeypatch.setattr(clews, 'nearest_clew_matches', recording_nearest)

    await service.ask(QUESTION, source='test')

    assert seen['top_k'] == 5000
@pytest.mark.asyncio
async def test_selected_clew_does_not_discard_semantic_positioning_documents(
        service, library, monkeypatch):
    _store_clew(library, embedded=True)
    seen = _capture(monkeypatch)
    monkeypatch.setattr("llm.chat_complete", lambda *a, **k: _answer())

    await service.ask(QUESTION, source="test")

    assert [document.title for document in seen["documents"]] == ["Widget Guide"]
@pytest.mark.asyncio
async def test_catalog_positioning_is_added_only_to_structural_documents(
        service, library, monkeypatch):
    from ariadne_mcp.models import DocumentResult, SearchResponse
    from schema import Document

    with library._conn_provider.acquire() as conn:
        guide_id = conn.execute(
            "SELECT id FROM documents WHERE title = ?", ("Widget Guide",)).fetchone()[0]
    guide = library.get_documents_batch([guide_id])[0]
    catalog = Document(
        id="catalog-only", content_type="catalog",
        title="LedgerWriter.recordStableIdentifier",
        content="STRUCTURAL-ONLY-CATALOG-CONTENT",
        source_files=["ledger.py"], metadata={"qualified_name": "LedgerWriter.recordStableIdentifier"},
        source_name="test")
    assembled = []
    seen = _capture(monkeypatch)

    async def fake_search(**_kwargs):
        return SearchResponse(documents=[DocumentResult(
            id=guide.id, title=guide.title, content_type=guide.content_type,
            content=guide.content, source_files=guide.source_files,
            metadata=guide.metadata, source_name=guide.source_name, score=1.0)], event_id=1)

    def capture_context(documents, *args, **kwargs):
        assembled.extend(documents)
        return "retrieved prose only"

    monkeypatch.setattr(service, "search", fake_search)
    monkeypatch.setattr("library.chain_answer.catalog_positioning_documents",
                        lambda *_args, **_kwargs: [catalog])
    monkeypatch.setattr("ariadne_mcp.service_analysis._assemble_ask_context", capture_context)
    monkeypatch.setattr("llm.chat_complete", lambda *a, **k: _answer())

    await service.ask(QUESTION, source="test")

    assert [document.title for document in assembled] == ["Widget Guide"]
    assert catalog in seen["documents"]
@pytest.mark.asyncio
async def test_deterministic_mode_skips_all_clew_selector_model_calls(
        service, library, monkeypatch):
    from library.clews import add_clew, init_clews_schema

    with library._conn_provider.acquire() as conn:
        init_clews_schema(conn)
        for index in range(16):
            add_clew(
                conn, source_name="test",
                entry_symbol=f"pkg.Owner{index}.run",
                steps=["run", f"step{index}"],
                route=[f"pkg.Owner{index}.run", f"pkg.Owner{index}.step{index}"],
                files=[f"owner{index}.py"], strategy="theme-walk",
                question=f"process widget stage {index}",
                embedding=_vector_for(QUESTION))
        conn.commit()
    seen = _capture(monkeypatch)
    service.config._config["ask_selector_mode"] = "deterministic"
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    calls = []

    async def fake_cc(messages, **kwargs):
        calls.append(messages)
        return "Grounded response."

    monkeypatch.setattr("llm.chat_complete", fake_cc)

    result = await service.ask(QUESTION, source="test")

    assert len(calls) == 1
    assert 1 <= len(seen["clew_matches"]) <= 8
    assert result.llm_calls == 1
