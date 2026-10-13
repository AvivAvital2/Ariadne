"""Exercise the blind runner with controlled HTTP, including negative oracles."""

import hashlib
import json
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from tests.live_model_smoke import Budget
from tests.live_qwen_blind import execute, rank_matches


def response(request):
    payload = json.loads(request.content)
    if request.url.path.endswith('/embeddings'):
        data = []
        for i, text in enumerate(payload['input']):
            seed = int.from_bytes(hashlib.sha256(text.encode()).digest()[:8])
            vector = np.random.default_rng(seed).normal(size=4096).tolist()
            data.append({'index': i, 'embedding': vector})
        return httpx.Response(200, json={'data': data[::-1], 'usage': {'prompt_tokens': 20}})
    marker = hashlib.sha256(request.content).hexdigest()[:8]
    body = (
        f'# Fixture {marker}\n\n## Overview\n'
        + f'Synthetic documentation {marker} preserves Unicode café and שלום. ' * 20
        + '\n\n## Question\nHow is this used?\n\n## Answer\nUse the supplied function.\n'
        + '\n\n## Details\nThe caller supplies the input.\n'
    )
    if 'technical diagram specialist' in json.dumps(payload):
        body = f'```mermaid\ngraph LR\n A[Input {marker}] --> B[Output]\n```'
    return httpx.Response(
        200,
        json={
            'choices': [{'message': {'content': body}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 20, 'completion_tokens': 100},
        },
    )


@pytest.mark.asyncio
async def test_frozen_blind_workflows_use_real_storage_and_http_contracts(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'test-blind-key')
    report = await execute(
        tmp_path / 'run',
        Budget(1),
        transport_factory=lambda: httpx.MockTransport(response),
        pacing=0,
    )
    assert report['status'] == 'passed', report
    assert len(report['checks']) == 41
    assert report['transport_mode'] == 'simulated'
    assert len(report['model_observations']) == 6
    assert all(c['passed'] for c in report['checks'])
    assert (tmp_path / 'run/protocol.json').exists()


def test_numeric_oracle_detects_bad_scores_and_wrong_order():
    expected = {'a': np.array([1.0, 0.0]), 'b': np.array([0.0, 1.0])}
    query = np.array([1.0, 0.0])
    a = SimpleNamespace(document=SimpleNamespace(id='a'), chunk=None, score=1.0)
    b = SimpleNamespace(document=SimpleNamespace(id='b'), chunk=None, score=0.0)
    assert rank_matches(query, [a, b], expected)
    assert not rank_matches(query, [b, a], expected)
    b.score = 0.5
    assert not rank_matches(query, [a, b], expected)


@pytest.mark.asyncio
async def test_blind_budget_exhaustion_stops_before_sending(tmp_path, monkeypatch):
    monkeypatch.setenv('DEEPINFRA_API_KEY', 'test-blind-key')
    calls = []

    def record(request):
        calls.append(request)
        return response(request)

    report = await execute(
        tmp_path / 'run',
        Budget(0.000001),
        transport_factory=lambda: httpx.MockTransport(record),
        pacing=0,
    )
    assert report['status'] == 'failed'
    assert not calls
