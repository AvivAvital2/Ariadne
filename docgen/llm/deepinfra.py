"""Hosted Qwen using DeepInfra's OpenAI-compatible live API."""

from docgen.llm.local import parse_chat_response
from docgen.llm.openai import OpenAIProvider


class DeepInfraProvider(OpenAIProvider):
    def _request_payload(self, system_prompt: str, user_prompt: str, max_tokens: int) -> dict:
        return {
            **super()._request_payload(system_prompt, user_prompt, max_tokens),
            'reasoning_effort': 'none',
            'stream': False,
        }

    def _response_text(self, data: dict) -> str:
        return parse_chat_response(data)[0]
