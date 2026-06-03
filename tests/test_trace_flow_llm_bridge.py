"""Contract for the production LLM bridge for trace_flow.

The walker (``docgen/trace_flow.py``) accepts an injected sync
``llm_bridge: (cursor_id, conn) -> (next_id | None, rationale)``.

This file tests the production builder ``build_llm_bridge``: it assembles
the prompt from cursor metadata + nearby docs, calls the LLM, parses the
JSON response, resolves the suggested symbol, and degrades gracefully on
every failure mode. The bridge core is provider-agnostic — the LLM call is
an injected ``complete(prompt) -> text`` callable, so these tests never make
a network call.

A separate suite (``TestMakeCompletion``) covers ``make_completion`` — the
provider-aware sync HTTP call that production wires in: Anthropic's
``/messages`` vs OpenAI's ``/chat/completions``, each parsed from its own
response shape. This is what lets trace_flow's LLM fallback run on either
provider (previously it hardcoded the Anthropic SDK).
"""
from __future__ import annotations

import sqlite3

import pytest


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _insert_symbol(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    qualified_name: str | None = None,
    source_name: str = 'myapp',
    language: str = 'python',
    file: str = 'app.py',
    line_start: int = 1,
    line_end: int = 10,
) -> None:
    qn = qualified_name or canonical_id
    conn.execute(
        'INSERT INTO scip_symbols VALUES '
        '(?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
        (canonical_id, source_name, language, file,
         line_start, line_end, 'function',
         qn.rsplit('.', 1)[-1], qn, None),
    )


def _completer(response_text: str = 'null', *, raise_on_call: bool = False):
    """A fake ``complete(prompt) -> text`` that records the prompts it sees
    (on ``.prompts``) and returns canned text — or raises, to exercise the
    bridge's graceful-degradation path."""
    prompts: list[str] = []

    def complete(prompt: str) -> str:
        if raise_on_call:
            raise RuntimeError('simulated API error')
        prompts.append(prompt)
        return response_text

    complete.prompts = prompts  # type: ignore[attr-defined]
    return complete


def _no_docs(conn, qualified_name):
    """Default doc_search stub — no related documents found."""
    return []


# ---------------------------------------------------------------------------
# Resolved suggestion path
# ---------------------------------------------------------------------------


class TestResolvedSuggestion:
    def test_valid_json_with_resolvable_symbol_returns_canonical(
        self, conn: sqlite3.Connection,
    ) -> None:
        """LLM returns valid JSON with a symbol_query that resolves
        uniquely → bridge returns (canonical_id, reasoning)."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        _insert_symbol(conn, canonical_id='myapp.target')
        conn.commit()

        complete = _completer(
            '{"symbol_query": "myapp.target", '
            '"reasoning": "Doc says cursor delegates to target"}',
        )
        bridge = build_llm_bridge(complete=complete, doc_search=_no_docs)
        next_sym, rationale = bridge('myapp.cursor', conn)
        assert next_sym == 'myapp.target'
        assert rationale == 'Doc says cursor delegates to target'
        # Sanity: we actually called the LLM (catches a short-circuit-to-None).
        assert len(complete.prompts) == 1

    def test_suffix_query_resolves_via_qualified_name_match(
        self, conn: sqlite3.Connection,
    ) -> None:
        """LLM returns a short identifier matching the suffix of a unique
        qualified_name → resolves."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        _insert_symbol(conn, canonical_id='myapp.deep.handler')
        conn.commit()

        complete = _completer('{"symbol_query": "deep.handler", "reasoning": "r"}')
        bridge = build_llm_bridge(complete=complete, doc_search=_no_docs)
        next_sym, _ = bridge('myapp.cursor', conn)
        assert next_sym == 'myapp.deep.handler'


# ---------------------------------------------------------------------------
# Null / decline path
# ---------------------------------------------------------------------------


class TestDeclinePath:
    def test_explicit_null_response_returns_none(
        self, conn: sqlite3.Connection,
    ) -> None:
        """LLM responds with the literal ``null`` → bridge returns (None, '')."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        conn.commit()

        bridge = build_llm_bridge(complete=_completer('null'), doc_search=_no_docs)
        next_sym, rationale = bridge('myapp.cursor', conn)
        assert next_sym is None
        assert rationale == ''

    def test_malformed_json_returns_none(
        self, conn: sqlite3.Connection,
    ) -> None:
        """LLM returns garbage that doesn't parse → graceful decline."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        conn.commit()

        bridge = build_llm_bridge(
            complete=_completer('I think the answer is myapp.target'),
            doc_search=_no_docs,
        )
        next_sym, _ = bridge('myapp.cursor', conn)
        assert next_sym is None

    def test_empty_symbol_query_returns_none(
        self, conn: sqlite3.Connection,
    ) -> None:
        """JSON parses but symbol_query is empty/missing → decline."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        conn.commit()

        bridge = build_llm_bridge(
            complete=_completer('{"symbol_query": "", "reasoning": "r"}'),
            doc_search=_no_docs,
        )
        next_sym, _ = bridge('myapp.cursor', conn)
        assert next_sym is None


# ---------------------------------------------------------------------------
# Symbol resolution failures
# ---------------------------------------------------------------------------


class TestSymbolResolution:
    def test_unresolvable_symbol_returns_none(
        self, conn: sqlite3.Connection,
    ) -> None:
        """LLM suggests a symbol absent from scip_symbols → decline (better
        than a fabricated canonical_id)."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        conn.commit()

        bridge = build_llm_bridge(
            complete=_completer(
                '{"symbol_query": "ghost.never.existed", "reasoning": "r"}',
            ),
            doc_search=_no_docs,
        )
        next_sym, _ = bridge('myapp.cursor', conn)
        assert next_sym is None

    def test_ambiguous_symbol_returns_none(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Two symbols share the suggested suffix → ambiguous, can't pick."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        _insert_symbol(conn, canonical_id='myapp.a.handler')
        _insert_symbol(conn, canonical_id='myapp.b.handler')
        conn.commit()

        bridge = build_llm_bridge(
            complete=_completer('{"symbol_query": "handler", "reasoning": "r"}'),
            doc_search=_no_docs,
        )
        next_sym, _ = bridge('myapp.cursor', conn)
        assert next_sym is None


# ---------------------------------------------------------------------------
# Pre-call validation and graceful degradation
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_unknown_cursor_skips_llm_call(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Cursor not in scip_symbols → (None, '') WITHOUT calling the LLM."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        complete = _completer('{"symbol_query": "ghost", "reasoning": "r"}')
        bridge = build_llm_bridge(complete=complete, doc_search=_no_docs)
        next_sym, _ = bridge('NONEXISTENT', conn)
        assert next_sym is None
        assert complete.prompts == []

    def test_completion_exception_returns_none(
        self, conn: sqlite3.Connection,
    ) -> None:
        """The LLM call raises → bridge returns (None, '') gracefully.
        Trace_flow callers must never see an exception from a bridge —
        bridge failure means "no answer", not crash."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        conn.commit()

        bridge = build_llm_bridge(
            complete=_completer(raise_on_call=True), doc_search=_no_docs,
        )
        next_sym, _ = bridge('myapp.cursor', conn)
        assert next_sym is None

    def test_prompt_includes_cursor_qualified_name(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Prompt construction surfaces the cursor's qualified_name."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.distinctive_name_xyz')
        conn.commit()

        complete = _completer('null')
        bridge = build_llm_bridge(complete=complete, doc_search=_no_docs)
        bridge('myapp.distinctive_name_xyz', conn)
        assert len(complete.prompts) == 1
        assert 'distinctive_name_xyz' in complete.prompts[0]

    def test_prompt_includes_doc_context(
        self, conn: sqlite3.Connection,
    ) -> None:
        """doc_search results must appear in the prompt."""
        from docgen.trace_flow_llm_bridge import build_llm_bridge

        _insert_symbol(conn, canonical_id='myapp.cursor')
        conn.commit()

        def _docs(_conn, qn):
            return [{'title': 'How cursor works', 'content': 'UNIQUE_DOC_TOKEN_QWERTY'}]

        complete = _completer('null')
        bridge = build_llm_bridge(complete=complete, doc_search=_docs)
        bridge('myapp.cursor', conn)
        assert 'UNIQUE_DOC_TOKEN_QWERTY' in complete.prompts[0]


# ---------------------------------------------------------------------------
# make_completion — provider-aware sync HTTP
# ---------------------------------------------------------------------------


class _FakeSyncResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class _FakeSyncClient:
    """Captures (method, path, json, headers); replays a canned response."""

    def __init__(self, response: dict):
        self._response = response
        self.calls: list[tuple] = []

    def post(self, path, *, json=None, headers=None):
        self.calls.append(('POST', path, json, headers))
        return _FakeSyncResponse(self._response)


class TestMakeCompletion:
    def test_anthropic_posts_to_messages_and_parses_content(self) -> None:
        from docgen.trace_flow_llm_bridge import make_completion

        client = _FakeSyncClient({'content': [{'type': 'text', 'text': 'ANT'}]})
        complete = make_completion(
            provider='anthropic', model='claude-opus-4-8',
            api_key='ant-key', client=client,
        )
        assert complete('hello') == 'ANT'
        _method, path, body, headers = client.calls[0]
        assert path == '/messages'
        assert body['model'] == 'claude-opus-4-8'
        assert body['messages'] == [{'role': 'user', 'content': 'hello'}]
        assert body['max_tokens']  # mandatory for Anthropic
        assert headers.get('x-api-key') == 'ant-key'

    def test_openai_posts_to_chat_completions_and_parses_choices(self) -> None:
        from docgen.trace_flow_llm_bridge import make_completion

        client = _FakeSyncClient(
            {'choices': [{'message': {'content': 'OAI'}}]},
        )
        complete = make_completion(
            provider='openai', model='gpt-5.5',
            api_key='oai-key', client=client,
        )
        assert complete('hello') == 'OAI'
        _method, path, body, headers = client.calls[0]
        assert path == '/chat/completions'
        assert body['model'] == 'gpt-5.5'
        assert body['messages'] == [{'role': 'user', 'content': 'hello'}]
        # gpt-5.x uses max_completion_tokens, never temperature.
        assert 'max_completion_tokens' in body
        assert 'temperature' not in body
        assert headers.get('Authorization') == 'Bearer oai-key'


class TestProductionWiring:
    def test_build_llm_bridge_without_complete_resolves_from_config(
        self, monkeypatch,
    ) -> None:
        """With no injected ``complete``, build_llm_bridge resolves
        provider/model + the matching API key from config and builds
        make_completion — the production path used by ``trace-flow
        --llm-bridge`` and the MCP caller (previously untested)."""
        from types import SimpleNamespace

        import docgen.trace_flow_llm_bridge as bridge_mod

        captured: dict = {}

        def fake_make_completion(*, provider, model, api_key, **kw):
            captured.update(provider=provider, model=model, api_key=api_key)
            return lambda prompt: 'x'

        monkeypatch.setattr(bridge_mod, 'make_completion', fake_make_completion)
        monkeypatch.setattr(
            'config.get_config',
            lambda: SimpleNamespace(
                model='claude-opus-4-8', provider='anthropic',
            ),
        )
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'ant-key')

        # No complete= → routes through _completion_from_config.
        bridge = bridge_mod.build_llm_bridge()
        assert callable(bridge)
        assert captured == {
            'provider': 'anthropic',
            'model': 'claude-opus-4-8',
            'api_key': 'ant-key',
        }
