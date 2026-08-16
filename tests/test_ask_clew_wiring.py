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
QUESTION = 'how does the widget flush records?'
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
        conn.commit()


def _capture(monkeypatch) -> dict:
    """Stand in for the walk and record what the wiring passed it."""
    seen: dict = {}

    def fake_evidence_for(library, documents, **kwargs):
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

    assert seen.get('clew_symbols') == ROUTE, (
        'the route the clew index matched must be what positions the walk')


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
    return 'answered'
