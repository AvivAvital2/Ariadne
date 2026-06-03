"""LLM-based audience adaptation for role-aware responses.

The optional layer at ``ariadne_ask`` query time: when a non-default
role is requested (``product_manager`` today; the only supported
non-default value), the dev-level docs are passed to an LLM with an
audience-named system prompt that translates the technical material
into role-appropriate prose. Result is persisted as a new
``documents`` row with ``content_type='audience_response'`` so the
next matching question is a cache hit, not a new LLM call.

This module currently ships a stub. The implementation lands in the
green phase (TDD plan: tests under
``tests/test_role_aware_responses.py`` exercise this contract
end-to-end).
"""
from __future__ import annotations


_SYSTEM_PROMPTS: dict[str, str] = {
    'product_manager': (
        'Translate the following technical documentation for a '
        'product manager. No code, no implementation details, no '
        'type signatures. Focus on user-facing behavior, business '
        'consequences, and the constraints stakeholders need to '
        'understand. Be concise.'
    ),
}


async def adapt_for_audience(
    role: str,
    dev_docs_context: str,
    query: str,
) -> str:
    """Generate an audience-targeted response from developer-level docs.

    Calls ``llm.chat_complete`` with a role-named system prompt and
    the developer-level docs as context. Returns the adapted markdown.

    Raises ``ValueError`` when ``role`` isn't in ``_SYSTEM_PROMPTS``.
    The caller (``ask``) is expected to only invoke this with
    supported non-default roles; the validation here is a defensive
    second line.
    """
    if role not in _SYSTEM_PROMPTS:
        msg = (
            f'No system prompt for role {role!r}. Supported roles: '
            f'{sorted(_SYSTEM_PROMPTS)}'
        )
        raise ValueError(msg)

    from llm import chat_complete

    user_prompt = (
        f'Question: {query}\n\n'
        f'Developer-level documentation to translate:\n'
        f'{dev_docs_context}'
    )
    return await chat_complete(
        system_prompt=_SYSTEM_PROMPTS[role],
        user_prompt=user_prompt,
        max_tokens=1024,
    )


__all__ = ['adapt_for_audience']
