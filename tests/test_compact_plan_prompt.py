"""Compact planning asks for evidence requirements without invented answers."""

from types import SimpleNamespace

import pytest

from ariadne_mcp.service_analysis import _ask_compact


class PlanCapturedError(Exception):
    pass


@pytest.mark.parametrize('question', [
    'How does pkg.Writer.write call pkg.Sink.flush?',
    'Compare read() and write() in the dependency library without assuming they call each other.',
])
async def test_compact_plan_is_question_scoped_before_any_retrieval(question):
    captured = {}

    async def capture_first_call(**request):
        captured.update(request)
        raise PlanCapturedError

    # A receiver without library/config proves this guard stops before retrieval.
    with pytest.raises(PlanCapturedError):
        await _ask_compact(SimpleNamespace(), question, source='requested-source', notes=(),
                           diagnostics={}, ask_chat=capture_first_call,
                           trace=lambda *_args, **_kwargs: None, phase_timings={})
    assert captured['phase'] == 'scip-obligation-plan'
    assert captured['max_tokens'] == 384
    system, user = [message['content'] for message in captured['messages']]
    assert 'C<n>: <requirement>' in system
    assert user.endswith('Question: ' + question)
    assert 'Do not answer the question or propose how the code works.' in user
    assert 'Do not introduce code identifiers absent from the question.' in user
    assert 'Do not invent intermediate calls, routes, directness, or implementation behavior.' in user
    assert 'Use one obligation for a single focused question when sufficient.' in user
    assert 'Five is a ceiling, not a target.' in user
