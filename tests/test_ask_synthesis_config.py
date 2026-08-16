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

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0):
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
