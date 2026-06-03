"""Production LLM bridge for trace_flow (Phase 9.b).

When the static graph (SCIP + HTTP + process tiers) is empty for a
cursor, the walker calls an injected ``llm_bridge`` to suggest the
next plausible hop. This module ships the production builder
``build_llm_bridge`` — a sync callable backed by the configured
provider (Anthropic or OpenAI) that:

1. Looks up cursor metadata (qualified_name, source, language, file)
   from ``scip_symbols``. Unknown cursor → decline immediately
   without calling the LLM.
2. Pulls related documentation context (the ``doc_search`` hook —
   defaults to a keyword match on the ``documents`` table; callers
   that want semantic search wrap a Library search instead).
3. Builds a prompt asking the LLM for the most likely next callee.
4. Parses the strict-JSON response: ``{"symbol_query": "...",
   "reasoning": "..."}`` or ``null``.
5. Resolves the suggested ``symbol_query`` to a unique
   ``canonical_id`` via exact / suffix / display-name match.
6. Returns ``(canonical_id, reasoning)`` on success, ``(None, '')``
   on any failure mode.

Failure modes that all degrade to ``(None, '')`` (never raise):
unknown cursor, API exception, malformed JSON, missing/empty
``symbol_query``, unresolvable symbol_query, ambiguous match.

The walker calls the returned bridge synchronously; the LLM call
uses sync httpx (no provider SDK), since the walker queries SQLite
synchronously and the bridge fires once per graph-exhausted cursor
(rare).

``complete`` and ``doc_search`` are the test-injection points.
Production passes None for both: the completion is built from config
(provider + model) via ``make_completion``, and the default doc
search runs a LIKE query against ``documents.content``.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from sqlite3 import Connection


_DEFAULT_MAX_TOKENS = 200
_DOC_CONTENT_CAP = 500  # truncate per-doc to keep prompts bounded
_DOC_SEARCH_LIMIT = 3


_PROMPT_TEMPLATE = (
    "You're tracing the flow of a function across a polyglot codebase.\n"
    '\n'
    'Cursor: {qualified_name} (source={source}, language={language})\n'
    'File: {file}:{line_start}\n'
    '\n'
    'The static cross-language graph (SCIP edges, HTTP boundaries, '
    'subprocess invocations) has no further edges from this cursor — '
    "that's why you're being asked.\n"
    '\n'
    'Documentation context for this symbol and related code:\n'
    '\n'
    '{doc_context}\n'
    '\n'
    'Suggest the single most likely next callee — the function or '
    'method this cursor most plausibly invokes. Return strict JSON:\n'
    '\n'
    '  {{"symbol_query": "<qualified_name or unique substring>", '
    '"reasoning": "<one sentence>"}}\n'
    '\n'
    'OR if no plausible bridge exists:\n'
    '\n'
    '  null\n'
    '\n'
    'Output only the JSON. No preamble, no markdown fences.'
)


def _lookup_full_symbol_info(
    conn: 'Connection', canonical_id: str,
) -> dict | None:
    cur = conn.execute(
        'SELECT qualified_name, source_name, language, file, line_start '
        'FROM scip_symbols WHERE canonical_id = ?',
        (canonical_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        'qualified_name': row[0],
        'source_name': row[1],
        'language': row[2],
        'file': row[3],
        'line_start': row[4],
    }


def _default_doc_search(
    conn: 'Connection', qualified_name: str,
) -> list[dict]:
    """Keyword-match query against ``documents.content``. Cheap
    fallback for when the bridge isn't wrapped with semantic search.
    Tolerates a missing ``documents`` table (returns empty list)."""
    try:
        cur = conn.execute(
            'SELECT title, content FROM documents '
            'WHERE content LIKE ? '
            'ORDER BY rowid DESC LIMIT ?',
            (f'%{qualified_name}%', _DOC_SEARCH_LIMIT),
        )
        return [
            {'title': t, 'content': c}
            for t, c in cur.fetchall()
        ]
    except Exception:
        return []


def _format_doc_context(docs: list[dict]) -> str:
    if not docs:
        return '(no related documentation found)'
    parts: list[str] = []
    for d in docs[:_DOC_SEARCH_LIMIT]:
        title = d.get('title') or '(untitled)'
        content = d.get('content') or ''
        if len(content) > _DOC_CONTENT_CAP:
            content = content[:_DOC_CONTENT_CAP] + '...'
        parts.append(f'## {title}\n{content}')
    return '\n\n'.join(parts)


def _resolve_symbol_query(
    conn: 'Connection', query: str,
) -> str | None:
    """Resolve a free-form ``symbol_query`` to a unique
    ``canonical_id``. Tries exact qualified_name match, then
    ``%.{query}`` suffix, then bare display_name. Returns None when
    zero or multiple matches — we never guess between candidates."""
    # Exact qualified_name
    cur = conn.execute(
        'SELECT canonical_id FROM scip_symbols WHERE qualified_name = ?',
        (query,),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    if len(rows) > 1:
        return None
    # Suffix match (".query" at end of qualified_name)
    cur = conn.execute(
        'SELECT canonical_id FROM scip_symbols WHERE qualified_name LIKE ?',
        (f'%.{query}',),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    if len(rows) > 1:
        return None
    # Bare display_name match (last identifier)
    last = query.rsplit('.', 1)[-1]
    cur = conn.execute(
        'SELECT canonical_id FROM scip_symbols WHERE display_name = ?',
        (last,),
    )
    rows = cur.fetchall()
    if len(rows) == 1:
        return rows[0][0]
    return None


def _parse_response(text: str) -> tuple[str | None, str]:
    """Parse the LLM's JSON output. Returns (symbol_query, reasoning)
    or (None, '') for null / malformed / missing fields."""
    text = text.strip()
    if not text or text == 'null':
        return (None, '')
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return (None, '')
    if parsed is None or not isinstance(parsed, dict):
        return (None, '')
    sq = parsed.get('symbol_query', '')
    rationale = parsed.get('reasoning', '')
    if not isinstance(sq, str) or not sq.strip():
        return (None, '')
    if not isinstance(rationale, str):
        rationale = ''
    return (sq.strip(), rationale.strip())


def build_llm_bridge(
    *,
    complete: Callable[[str], str | None] | None = None,
    doc_search: Callable[
        ['Connection', str], list[dict],
    ] | None = None,
):
    """Construct a sync LLM bridge for trace_flow.

    Returns a callable matching ``trace_flow.LlmBridgeFn``:
    ``(cursor_id, conn) -> (next_id | None, rationale)``.

    ``complete`` is the LLM call — ``(prompt) -> text | None``. When omitted,
    one is built from config via :func:`make_completion`, honoring the
    configured provider (Anthropic or OpenAI) and model. Tests inject a stub.
    ``doc_search`` defaults to a LIKE query against ``documents.content``.
    """
    if complete is None:
        complete = _completion_from_config()
    if doc_search is None:
        doc_search = _default_doc_search

    def bridge(cursor: str, conn: 'Connection') -> tuple[str | None, str]:
        info = _lookup_full_symbol_info(conn, cursor)
        if info is None:
            return (None, '')

        docs = doc_search(conn, info['qualified_name']) or []
        prompt = _PROMPT_TEMPLATE.format(
            qualified_name=info['qualified_name'],
            source=info['source_name'] or '?',
            language=info['language'] or '?',
            file=info['file'] or '?',
            line_start=info['line_start'] if info['line_start'] is not None
            else 0,
            doc_context=_format_doc_context(docs),
        )

        try:
            text = complete(prompt)
        except Exception:
            return (None, '')
        if not text:
            return (None, '')

        symbol_query, rationale = _parse_response(text)
        if symbol_query is None:
            return (None, '')

        canonical = _resolve_symbol_query(conn, symbol_query)
        if canonical is None:
            return (None, '')

        return (canonical, rationale)

    return bridge


def make_completion(
    *,
    provider: str,
    model: str,
    api_key: str,
    base_url: str | None = None,
    max_tokens: int = _DEFAULT_MAX_TOKENS,
    client=None,
) -> Callable[[str], str | None]:
    """Build a sync ``complete(prompt) -> text`` over the provider's chat API.

    Mirrors ``AnthropicProvider.call`` / ``OpenAIProvider.call`` (the async
    unary path) for the sync trace-flow walker — keep the endpoint/header/parse
    facts in sync if a provider's API changes.

    Anthropic posts to ``/messages`` (text at ``content[0].text``); OpenAI to
    ``/chat/completions`` (text at ``choices[0].message.content``). Uses sync
    httpx — no provider SDK required, matching the rest of the codebase. The
    trace_flow fallback fires rarely (once per graph-exhausted cursor), so a
    per-call client is fine. ``client`` (a sync httpx.Client-like with
    ``.post(path, json=, headers=)``) is the test-injection point.
    """
    if provider == 'anthropic':
        base_url = base_url or 'https://api.anthropic.com/v1'
        path = '/messages'
        headers = {
            'x-api-key': api_key,
            'anthropic-version': '2023-06-01',
            'content-type': 'application/json',
        }
    else:
        base_url = base_url or 'https://api.openai.com/v1'
        path = '/chat/completions'
        headers = {
            'Authorization': f'Bearer {api_key}',
            'content-type': 'application/json',
        }

    def _body(prompt: str) -> dict:
        body: dict = {
            'model': model,
            'messages': [{'role': 'user', 'content': prompt}],
        }
        # Anthropic requires max_tokens; OpenAI's gpt-5.x renamed it (shared
        # rule via token_limit_field). temperature is sent by neither.
        if provider == 'openai':
            from docgen.llm.openai import token_limit_field
            body[token_limit_field(model)] = max_tokens
        else:
            body['max_tokens'] = max_tokens
        return body

    def _parse(data: dict) -> str | None:
        if provider == 'anthropic':
            return data['content'][0]['text']
        return data['choices'][0]['message']['content']

    def complete(prompt: str) -> str | None:
        import contextlib
        import httpx

        # Injected client (tests) or a fresh per-call one (production — the
        # bridge fires rarely, so no pooling needed). One post/parse path.
        http_ctx = (
            contextlib.nullcontext(client) if client is not None
            else httpx.Client(base_url=base_url, timeout=httpx.Timeout(30.0))
        )
        with http_ctx as http:
            response = http.post(path, json=_body(prompt), headers=headers)
            response.raise_for_status()
            return _parse(response.json())

    return complete


def _completion_from_config() -> Callable[[str], str | None]:
    """Resolve provider/model/api_key from config and build the production
    completion. Used when ``build_llm_bridge`` is called without an injected
    ``complete`` — i.e. by the CLI (`trace-flow --llm-bridge`) and MCP callers.
    """
    import os

    from cli.generate import resolve_provider
    from config import get_config

    cfg = get_config()
    model = cfg.model
    provider = resolve_provider(
        cli_provider=None,
        cfg_provider=getattr(cfg, 'provider', None),
        model=model,
    )
    api_key = (
        os.environ.get('ANTHROPIC_API_KEY', '') if provider == 'anthropic'
        else os.environ.get('OPENAI_API_KEY', '')
    )
    return make_completion(provider=provider, model=model, api_key=api_key)


__all__ = ['build_llm_bridge', 'make_completion']
