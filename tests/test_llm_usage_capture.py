from __future__ import annotations

from types import SimpleNamespace

import pytest

import llm


class FakeProvider:
    def __init__(self, usage):
        self.last_usage = usage
        self.closed = False

    async def call(self, system_prompt, user_prompt, *, max_tokens):
        return "answer"

    async def aclose(self):
        self.closed = True


@pytest.mark.asyncio
@pytest.mark.parametrize("provider_name,key_name", [
    ("anthropic", "ANTHROPIC_API_KEY"),
    ("openai", "OPENAI_API_KEY"),
])
async def test_chat_complete_captures_exactly_one_usage_row(
        monkeypatch, provider_name, key_name):
    import config
    import cli.generate
    import docgen.llm.factory

    usage = {"input_tokens": 11, "output_tokens": 7}
    provider = FakeProvider(usage)
    monkeypatch.setenv(key_name, "test-key")
    monkeypatch.setattr(
        config, "get_config",
        lambda: SimpleNamespace(
            model="claude-opus-4-8" if provider_name == "anthropic" else "gpt-5",
            configured_provider=provider_name))
    monkeypatch.setattr(
        cli.generate, "resolve_provider", lambda **kwargs: provider_name)
    monkeypatch.setattr(
        docgen.llm.factory, "make_llm_provider", lambda **kwargs: provider)

    with llm.capture_completion_usage() as rows:
        result = await llm.chat_complete(
            [{"role": "user", "content": "question"}], phase="route-select")

    assert result == "answer"
    assert rows == [{
        "phase": "route-select",
        "model": "claude-opus-4-8" if provider_name == "anthropic" else "gpt-5",
        "input_tokens": 11,
        "output_tokens": 7,
    }]
    assert provider.closed is True


def test_provider_key_check_uses_the_configured_provider(monkeypatch):
    import config
    import cli.generate

    monkeypatch.setattr(
        config, "get_config",
        lambda: SimpleNamespace(
            model="claude-opus-4-8", configured_provider="anthropic"))
    monkeypatch.setattr(
        cli.generate, "resolve_provider", lambda **kwargs: "anthropic")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "present")

    assert llm.provider_key_env() == "ANTHROPIC_API_KEY"
    assert llm.has_provider_key() is True


def test_usage_capture_contexts_are_isolated():
    with llm.capture_completion_usage() as outer:
        outer.append({"phase": "outer"})
        with llm.capture_completion_usage() as inner:
            inner.append({"phase": "inner"})
        outer.append({"phase": "outer-again"})

    assert outer == [{"phase": "outer"}, {"phase": "outer-again"}]
    assert inner == [{"phase": "inner"}]
class FakeResponse:
    status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return {
            "choices": [{"message": {"content": "answer"}}],
            "usage": {"prompt_tokens": 13, "completion_tokens": 5},
        }


class FakeClient:
    async def post(self, url, json):
        return FakeResponse()


@pytest.mark.asyncio
async def test_openai_provider_exposes_response_usage(monkeypatch):
    from docgen.llm.openai import OpenAIProvider

    monkeypatch.setattr(OpenAIProvider, "_get_client", lambda self: FakeClient())
    provider = OpenAIProvider(model="gpt-5", api_key="test")

    assert await provider.call("system", "user") == "answer"
    assert provider.last_usage == {
        "input_tokens": 13,
        "output_tokens": 5,
    }
@pytest.mark.asyncio
async def test_anthropic_provider_exposes_response_usage(monkeypatch):
    from docgen.llm.anthropic import AnthropicProvider

    class Response:
        status_code = 200

        def raise_for_status(self):
            return None

        def json(self):
            return {
                "content": [{"type": "text", "text": "answer"}],
                "usage": {"input_tokens": 17, "output_tokens": 3},
            }

    class Client:
        async def post(self, url, json):
            return Response()

    monkeypatch.setattr(AnthropicProvider, "_get_client", lambda self: Client())
    provider = AnthropicProvider(model="claude-opus-4-8", api_key="test")

    assert await provider.call("system", "user") == "answer"
    assert provider.last_usage == {"input_tokens": 17, "output_tokens": 3}
@pytest.mark.asyncio
async def test_completion_trace_records_exact_request_response_and_usage(monkeypatch):
    import config
    import cli.generate
    import docgen.llm.factory

    provider = FakeProvider({"input_tokens": 11, "output_tokens": 7})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        config, "get_config",
        lambda: SimpleNamespace(
            model="claude-opus-4-8", configured_provider="anthropic"))
    monkeypatch.setattr(
        cli.generate, "resolve_provider", lambda **kwargs: "anthropic")
    monkeypatch.setattr(
        docgen.llm.factory, "make_llm_provider", lambda **kwargs: provider)
    messages = [
        {"role": "system", "content": "choose exact routes"},
        {"role": "user", "content": "C1: R2"},
    ]

    with llm.capture_completion_trace() as rows:
        result = await llm.chat_complete(
            messages, max_tokens=37, timeout=19.0, phase="route-select")

    assert result == "answer"
    assert rows == [{
        "phase": "route-select",
        "model": "claude-opus-4-8",
        "max_tokens": 37,
        "timeout": 19.0,
        "messages": messages,
        "response": "answer",
        "usage": {"input_tokens": 11, "output_tokens": 7},
        "status": "ok",
    }]


@pytest.mark.asyncio
async def test_completion_trace_records_provider_failure_without_hiding_it(
        monkeypatch):
    import config
    import cli.generate
    import docgen.llm.factory

    class FailingProvider(FakeProvider):
        async def call(self, system_prompt, user_prompt, *, max_tokens):
            raise RuntimeError("synthetic provider failure")

    provider = FailingProvider(None)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setattr(
        config, "get_config",
        lambda: SimpleNamespace(
            model="claude-opus-4-8", configured_provider="anthropic"))
    monkeypatch.setattr(
        cli.generate, "resolve_provider", lambda **kwargs: "anthropic")
    monkeypatch.setattr(
        docgen.llm.factory, "make_llm_provider", lambda **kwargs: provider)

    with llm.capture_completion_trace() as rows:
        with pytest.raises(RuntimeError, match="synthetic provider failure"):
            await llm.chat_complete(
                [{"role": "user", "content": "question"}],
                phase="formulation")

    assert rows[0]["status"] == "error"
    assert rows[0]["error_type"] == "RuntimeError"
    assert rows[0]["error"] == "synthetic provider failure"
    assert rows[0]["response"] == ""
    assert provider.closed is True


def test_disabled_completion_trace_collects_nothing():
    with llm.capture_completion_trace(enabled=False) as rows:
        assert rows == []
