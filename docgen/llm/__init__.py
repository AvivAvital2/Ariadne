"""LLM provider abstraction.

Lets DocGenerator call OpenAI or Anthropic through a common interface.
Use ``make_llm_provider`` to get a concrete provider; pass it the
``call(system_prompt, user_prompt, ...)`` method to invoke the model.
"""
from docgen.llm.base import LLMProvider
from docgen.llm.factory import make_llm_provider

__all__ = ['LLMProvider', 'make_llm_provider']
