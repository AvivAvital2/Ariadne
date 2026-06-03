"""Generate-side calibration capture: the generator tags each doc's
LLM call with (phase=generate, doc_type, language) so real per-type
output tokens land in the store. The context is set by the caller
(around ``_call_llm``), keeping ``_call_llm``'s signature unchanged so
existing mocks of it still work.
"""
from __future__ import annotations

import types

import pytest


@pytest.mark.asyncio
async def test_generate_captures_usage_per_doc_type(tmp_path, monkeypatch):
    from docgen.calibration import (
        CalibrationStore, emit_usage, set_usage_observer,
    )
    from docgen.generator import DocGenerator

    class _FakeProvider:
        async def call(self, system, user, **kw):
            # Mirror the real provider reporting usage mid-call.
            emit_usage(
                model='claude-opus-4-8', input_tokens=900, output_tokens=1300,
            )
            return 'doc body'

    gen = DocGenerator()
    gen._provider = _FakeProvider()
    # Skip prompt/template machinery — we're testing the usage tagging.
    monkeypatch.setattr(
        DocGenerator, '_build_prompt_for_bundle_doc',
        lambda self, bundle, source_code, doc_type: ('sys', 'user'),
    )
    bundle = types.SimpleNamespace(language='python')

    store = CalibrationStore(tmp_path / 'cal.db')
    with set_usage_observer(store.record):
        out = await gen._generate_doc_from_bundle(bundle, 'src', 'explanation')
    assert out == 'doc body'

    cal = store.mean_tokens(
        phase='generate', model='claude-opus-4-8',
        doc_type='explanation', language='python',
    )
    assert cal is not None, 'generate usage must be captured per (type, lang)'
    assert cal.mean_input == 900.0 and cal.mean_output == 1300.0
    # A different doc_type bucket is isolated.
    assert store.mean_tokens(
        phase='generate', model='claude-opus-4-8', doc_type='diagram',
    ) is None
