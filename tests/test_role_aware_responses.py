"""Role-aware response tests — TDD red phase.

These tests pin the contract for the optional role-aware response
layer described in ``designs/role-aware-responses.md``. Today they
FAIL behaviorally (not on ImportError) under the stubs landed in:
- ``mcp_service_search.SearchMixin.search`` — ``role`` kwarg added, ignored
- ``mcp_service_analysis.AnalysisMixin.ask`` — ``role`` kwarg added, ignored
- ``docgen/role_adapter.py`` — ``adapt_for_audience`` returns ``''``
- ``schema.ContentType`` — added ``'audience_response'`` so tests can
  seed rows without rejection

The green-phase implementation will:
1. Filter ``content_type='audience_response'`` rows from
   ``search()`` retrieval when ``role='developer'`` (default).
2. In ``ask()`` with ``role='product_manager'``, check for a cached
   ``audience_response`` row before calling the adapter; on cache
   miss, call ``adapt_for_audience`` with dev docs as context, then
   persist the result.

Cache invalidation (cascade-delete on parent regen) is covered in a
separate test file once the green phase lands and the cascade hook
location is known.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import numpy as np
import pytest


_DIM = 3072  # text-embedding-3-large; matches schema.EMBEDDING_DIM


def _unit_vec(seed: int, dim: int = _DIM) -> 'np.ndarray':
    """Stable axis-aligned unit vector keyed on ``seed``. Different
    seeds → orthogonal vectors → near-zero cosine; same seed → 1.0."""
    v = np.zeros(dim, dtype=np.float32)
    v[seed % dim] = 1.0
    return v


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    """Configure the ``'test'`` source so the chokepoint admits the
    fixture docs. The contract under test is role-aware response
    handling; source naming is environmental."""
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


@pytest.fixture
def library(tmp_path: Path):
    from library import Library
    lib = Library(tmp_path / 'role_aware.db')
    # The role-aware contract is orthogonal to source scoping. Wrap
    # add_document so the test docs go through with source_name='test'
    # without making every call site thread it — mirrors the production
    # cmd_add/cmd_finding behavior of auto-resolving the source.
    original_add = lib.add_document

    def add_with_source(*args, **kwargs):
        kwargs.setdefault('source_name', 'test')
        return original_add(*args, **kwargs)

    lib.add_document = add_with_source
    yield lib
    lib.close()


@pytest.fixture
def service(library):
    """Minimal AnalysisMixin + SearchMixin service stitched together
    over the library fixture. Mirrors the test_themes_e2e pattern."""
    from ariadne_mcp.service_analysis import AnalysisMixin
    from ariadne_mcp.service_search import SearchMixin

    class _StableEmbedder:
        """Returns a stable unit vector for any input — embeddings
        match exactly when the seed matches. Eliminates similarity
        randomness from the contract tests."""

        async def embed(self, text):
            # Hash text to a seed deterministically so different
            # queries get different vectors.
            import hashlib
            h = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
            return _unit_vec(h)

    class _Svc(SearchMixin, AnalysisMixin):
        @staticmethod
        def _cache_key(*args, **kwargs):
            return hash((args, tuple(sorted(kwargs.items()))))

        def get_branch(self):
            return None

        def _resolve_scope(self, source):
            from scope_resolution import make_scoped_library
            return make_scoped_library(
                self.config, self.library, source or 'test',
            )

    from config import get_config
    svc = _Svc()
    svc.library = library
    svc.config = get_config()
    svc._query_cache = {}
    svc.embedding_service = _StableEmbedder()
    return svc


# ---------------------------------------------------------------------------
# Signature smoke — passes under the stub by virtue of the kwarg being
# in place. Pairs with the behavioral tests below: signature alone
# isn't a contract; behavior is.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_accepts_role_kwarg(library, service):
    """Calling ``search`` with ``role='developer'`` doesn't raise.
    Smoke check: ensures the signature change landed. Seed a doc so
    result_count > 0 — sidesteps a pre-existing
    ``usage_events.outcome`` NOT NULL constraint that fires on
    zero-result searches; unrelated to the role-aware contract."""
    library.add_document(
        content_type='explanation', title='Stub', content='x',
        source_files=[], embedding=_unit_vec(1), metadata={},
    )
    result = await service.search(query='stub', role='developer')
    assert result is not None


@pytest.mark.asyncio
async def test_ask_accepts_role_kwarg(library, service):
    """Same shape for ``ask`` — kwarg accepted."""
    library.add_document(
        content_type='explanation', title='Stub', content='x',
        source_files=[], embedding=_unit_vec(1), metadata={},
    )
    result = await service.ask(question='What does it do?', role='developer')
    assert result is not None


# ---------------------------------------------------------------------------
# Contract bite 1 — developer role (default) excludes audience_response rows
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_role_developer_excludes_audience_response_rows(
    library, service,
):
    """Audience-adapted responses must not surface for developer
    queries — they're the dev baseline's shallower sibling. Today
    the search stub doesn't filter; an audience_response row leaks
    into the result set."""
    seed = 12345
    library.add_document(
        content_type='explanation',
        title='Auth Module',
        content='Developer-level explanation of auth.',
        source_files=['auth.py'],
        embedding=_unit_vec(seed),
        metadata={},
    )
    library.add_document(
        content_type='audience_response',
        title='product_manager response: how does auth work?',
        content='Product-targeted summary of auth.',
        source_files=['auth.py'],
        embedding=_unit_vec(seed),  # identical embedding → both candidates
        metadata={
            'audience': 'product_manager',
            'derived_from': [],
            'question': 'how does auth work?',
        },
    )

    result = await service.search(query='auth', role='developer', limit=10)
    titles = {d.title for d in result.documents}

    assert 'Auth Module' in titles, (
        'developer query should see the dev-level explanation'
    )
    assert 'product_manager response: how does auth work?' not in titles, (
        'developer query must NOT see audience_response rows; '
        f'got titles={titles}'
    )


# ---------------------------------------------------------------------------
# Contract bite 2 — PM role on cache miss invokes adapter with dev docs
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_role_pm_cache_miss_invokes_adapter(
    library, service, monkeypatch,
):
    """First PM question for a topic: no cached audience_response
    row exists, so ``adapt_for_audience`` is called. The dev-level
    doc content must be passed in as ``dev_docs_context`` so the
    adapter has the source-of-truth to translate from. Today the
    stub doesn't call the adapter at all."""
    seed = 54321
    library.add_document(
        content_type='explanation',
        title='Auth Module',
        content='Technical: JWT tokens validated via HMAC-SHA256.',
        source_files=['auth.py'],
        embedding=_unit_vec(seed),
        metadata={},
    )

    spy = AsyncMock(return_value='PM-friendly auth summary.')
    monkeypatch.setattr(
        'docgen.role_adapter.adapt_for_audience', spy,
    )

    await service.ask(
        question='How does authentication work?',
        role='product_manager',
    )

    spy.assert_awaited_once()
    call = spy.await_args
    # Collapse positional + kwargs into one bag so the assertion
    # doesn't depend on whether the caller passes by position or kwarg.
    bound = {**call.kwargs}
    for i, val in enumerate(call.args):
        bound[f'_pos_{i}'] = val

    role_val = call.kwargs.get('role')
    if role_val is None and call.args:
        role_val = call.args[0]
    assert role_val == 'product_manager', (
        f'adapter called without correct role; args={call.args} kwargs={call.kwargs}'
    )

    # ``dev_docs_context`` must contain the dev doc's content so the
    # adapter has the technical source to translate from.
    full_call = ' '.join(str(v) for v in bound.values())
    assert 'JWT tokens validated via HMAC-SHA256' in full_call, (
        'adapter must receive the dev-level doc content as context; '
        f'call args/kwargs did not include it. args={call.args} '
        f'kwargs={call.kwargs}'
    )


# ---------------------------------------------------------------------------
# Contract bite 3 — PM role on cache miss persists a new audience_response row
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_role_pm_cache_miss_persists_audience_response_row(
    library, service, monkeypatch,
):
    """The PM-adapted response must be persisted as a new
    ``content_type='audience_response'`` row so subsequent identical
    questions are cache hits (zero LLM cost). Today the stub
    doesn't persist anything."""
    seed = 99001
    library.add_document(
        content_type='explanation',
        title='Token Service',
        content='Developer: TokenService.issue_token() flow.',
        source_files=['token_service.py'],
        embedding=_unit_vec(seed),
        metadata={},
    )

    spy = AsyncMock(return_value='Tokens are issued when X. PMs care because Y.')
    monkeypatch.setattr(
        'docgen.role_adapter.adapt_for_audience', spy,
    )

    await service.ask(
        question='How do we issue tokens?',
        role='product_manager',
    )

    audience_rows = library.list_documents(content_type='audience_response')
    assert len(audience_rows) == 1, (
        f'expected exactly one persisted audience_response row; '
        f'got {len(audience_rows)}: {[d.title for d in audience_rows]}'
    )

    row = audience_rows[0]
    assert row.content == 'Tokens are issued when X. PMs care because Y.', (
        f'persisted content does not match adapter output; got: {row.content!r}'
    )
    assert row.metadata.get('audience') == 'product_manager', (
        f"metadata.audience missing or wrong; got: {row.metadata}"
    )
    assert 'token_service.py' in row.source_files, (
        f'source_files should carry the parent dev doc files; got: '
        f'{row.source_files}'
    )


# ---------------------------------------------------------------------------
# Contract bite 4 — PM role with cache hit returns cached content without
# re-invoking the adapter (zero LLM cost on repeat questions)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_role_pm_cache_hit_skips_adapter(
    library, service, monkeypatch,
):
    """A repeat PM question whose audience_response already exists
    must not re-invoke the adapter — that's the cache's value-add.
    Pre-seed a matching audience_response, patch the adapter as a
    spy, ask the same question, assert: cached content returned and
    adapter NOT called."""
    seed = 31415

    # Parent dev doc the cached response derives from.
    dev_doc = library.add_document(
        content_type='explanation',
        title='Token Service',
        content='Developer-level token service explanation.',
        source_files=['token_service.py'],
        embedding=_unit_vec(seed),
        metadata={},
    )

    # Pre-seeded cached audience response.
    library.add_document(
        content_type='audience_response',
        title='product_manager response: How does the token service work?',
        content='Cached PM-friendly content about tokens.',
        source_files=['token_service.py'],
        embedding=_unit_vec(seed),
        metadata={
            'audience': 'product_manager',
            'derived_from': [dev_doc.id],
            'question': 'How does the token service work?',
        },
    )

    spy = AsyncMock(return_value='UNUSED — adapter should not be called on cache hit')
    monkeypatch.setattr(
        'docgen.role_adapter.adapt_for_audience', spy,
    )

    response = await service.ask(
        question='How does the token service work?',
        role='product_manager',
    )

    spy.assert_not_awaited()
    assert response.answer == 'Cached PM-friendly content about tokens.', (
        f'expected cached content; got: {response.answer!r}'
    )


# ---------------------------------------------------------------------------
# Contract bite 5 — wiring: source/role must reach the service from the MCP
# tools, and ask must thread source into its OWN retrieval. This is the gap
# that made an explicit source-named question fail with "No source context".
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_threads_source_into_search(library, service, monkeypatch):
    """ask must pass ``source`` through to its internal search so a
    decomposed source actually scopes retrieval — previously ask ignored
    source and the scoped search fell back / failed closed."""
    captured: dict = {}
    real_search = service.search

    async def spy(*args, **kwargs):
        captured['source'] = kwargs.get('source')
        return await real_search(*args, **kwargs)

    monkeypatch.setattr(service, 'search', spy)
    library.add_document(
        content_type='explanation', title='X', content='x',
        source_files=[], embedding=_unit_vec(1), metadata={},
    )
    await service.ask(question='what does it do', role='developer', source='test')
    assert captured['source'] == 'test'


@pytest.mark.asyncio
async def test_mcp_ask_tool_threads_source_and_role(monkeypatch):
    """The ariadne_ask MCP tool must expose ``source`` + ``role`` and forward
    them, so a PM-scoped question reaches the scoped, role-aware path."""
    from ariadne_mcp import server_knowledge

    captured: dict = {}

    class _FakeSvc:
        async def ask(self, question, branch=None, role='developer', source=None):
            captured.update(question=question, role=role, source=source)
            return 'ASK_OK'

    monkeypatch.setattr('ariadne_mcp.service.AriadneService.get', lambda: _FakeSvc())
    out = await server_knowledge.ariadne_ask(
        question='how does it work', source='demo', role='product_manager',
    )
    assert out == 'ASK_OK'
    assert captured == {
        'question': 'how does it work', 'role': 'product_manager', 'source': 'demo',
    }


@pytest.mark.asyncio
async def test_mcp_search_tool_threads_role(monkeypatch):
    """ariadne_search must expose ``role`` and forward it (it already had
    ``source``) so PM-scoped retrieval is reachable through the tool."""
    from ariadne_mcp import server

    captured: dict = {}

    class _FakeSvc:
        def get_branch(self):
            return None

        async def search(self, *args, **kwargs):
            captured['role'] = kwargs.get('role')
            captured['source'] = kwargs.get('source')
            return 'SEARCH_OK'

    monkeypatch.setattr('ariadne_mcp.service.AriadneService.get', lambda: _FakeSvc())
    out = await server.ariadne_search(
        query='x', source='demo', role='product_manager',
    )
    assert out == 'SEARCH_OK'
    assert captured == {'role': 'product_manager', 'source': 'demo'}


# ---------------------------------------------------------------------------
# Contract bite 6 — graceful degradation: an LLM (adapter) or DB (persist)
# failure on the PM path must NOT break the answer.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ask_role_pm_adapter_failure_falls_back_to_dev(
    library, service, monkeypatch,
):
    """If the role adapter (an LLM call) fails, ask degrades gracefully —
    returns the developer-level docs, doesn't raise, and persists nothing."""
    library.add_document(
        content_type='explanation', title='Auth',
        content='Dev detail: HMAC validation.', source_files=['auth.py'],
        embedding=_unit_vec(7), metadata={},
    )
    monkeypatch.setattr(
        'docgen.role_adapter.adapt_for_audience',
        AsyncMock(side_effect=RuntimeError('LLM down')),
    )
    resp = await service.ask(
        question='how does auth work', role='product_manager', source='test',
    )
    assert 'Dev detail: HMAC validation.' in resp.answer
    assert library.list_documents(content_type='audience_response') == []


@pytest.mark.asyncio
async def test_ask_role_pm_persist_failure_still_returns(
    library, service, monkeypatch,
):
    """A cache-persist failure must not break the response — the adapted
    answer is still returned to the caller."""
    library.add_document(
        content_type='explanation', title='Tokens',
        content='Dev token flow.', source_files=['t.py'],
        embedding=_unit_vec(8), metadata={},
    )
    monkeypatch.setattr(
        'docgen.role_adapter.adapt_for_audience',
        AsyncMock(return_value='PM answer.'),
    )

    def _boom(*a, **k):
        raise RuntimeError('db full')

    monkeypatch.setattr(
        'ariadne_mcp.service_analysis._persist_audience_response', _boom,
    )
    resp = await service.ask(
        question='token flow', role='product_manager', source='test',
    )
    assert resp.answer == 'PM answer.'


# ---------------------------------------------------------------------------
# Contract bite 7 — the adapter itself: must call chat_complete with its real
# `messages` signature (not the non-existent system_prompt=/user_prompt= kwargs
# that made the PM path fall back to dev docs), carrying the role prompt + docs.
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adapt_for_audience_calls_chat_complete_with_messages(monkeypatch):
    from docgen import role_adapter

    captured: dict = {}

    async def fake_chat_complete(
        messages, *, model=None, max_tokens=2048, timeout=60.0,
    ):
        captured['messages'] = messages
        captured['max_tokens'] = max_tokens
        return 'PM-ADAPTED'

    monkeypatch.setattr('llm.chat_complete', fake_chat_complete)

    out = await role_adapter.adapt_for_audience(
        role='product_manager',
        dev_docs_context='Dev: HMAC validation in auth.py.',
        query='how does auth work?',
    )

    assert out == 'PM-ADAPTED'
    roles = [m['role'] for m in captured['messages']]
    assert 'system' in roles and 'user' in roles
    system = next(m['content'] for m in captured['messages'] if m['role'] == 'system')
    user = next(m['content'] for m in captured['messages'] if m['role'] == 'user')
    assert 'product manager' in system.lower()
    assert 'HMAC validation' in user and 'how does auth work?' in user


@pytest.mark.asyncio
async def test_adapt_for_audience_rejects_unknown_role():
    """Defensive guard: an unsupported role raises ValueError rather than
    silently producing a wrong-audience answer."""
    from docgen import role_adapter

    with pytest.raises(ValueError, match='No system prompt for role'):
        await role_adapter.adapt_for_audience(
            role='wizard', dev_docs_context='x', query='y',
        )
