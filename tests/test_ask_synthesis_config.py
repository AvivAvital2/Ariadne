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
@pytest.mark.asyncio
async def test_ask_offers_a_menu_then_answers_from_what_was_chosen(
    service, library, monkeypatch,
):
    """Two calls: the chain is offered, the model picks, only those bodies travel.

    Measured at production width, sending the whole bundle in one call is 240,945 tokens
    and $1.20 a question, 68% of it coordinates for hops the answer never mentions. The
    menu of the same chain is $0.19, and the second call carries the handful chosen.

    This also puts the question into a path that had never seen it: the walk expands from
    wherever retrieval landed, with no notion of what was asked.
    """
    _seed_chain(library)
    monkeypatch.setenv('OPENAI_API_KEY', 'test-key')
    service.config._config['ask_synthesis'] = True
    calls: list = []

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0):
        prompt = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
        calls.append(prompt)
        return '1' if len(calls) == 1 else 'ANSWER'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?',
                               source='test')

    assert 'ANSWER' in result.answer
    assert len(calls) == 2, f'expected a menu call then an answer call, got {len(calls)}'
    assert 'DEFINITIONS' in calls[0], 'the first call offers the chain'
    assert 'DEFINITIONS' not in calls[1], 'the second call carries bodies, not the menu'


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

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0):
        prompt = '\n'.join(m['content'] for m in messages if m['role'] == 'user')
        calls.append(prompt)
        return 'nothing looks relevant' if len(calls) == 1 else 'ANSWER'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?',
                               source='test')

    assert 'ANSWER' in result.answer
    assert len(calls) == 2
    assert 'Call chain:' in calls[1], 'the full chain travels when nothing was chosen'
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

    async def fake_cc(messages, *, model=None, max_tokens=2048, timeout=60.0):
        calls.append(messages)
        if len(calls) == 1:
            return '1'
        return 'Runs at w.py:5 (and again at :20).'

    monkeypatch.setattr('llm.chat_complete', fake_cc)

    result = await service.ask(question='How does the widget subsystem work?',
                               source='test')

    assert 'w.py:20' in result.answer, result.answer
    assert '(and again at :20)' not in result.answer
