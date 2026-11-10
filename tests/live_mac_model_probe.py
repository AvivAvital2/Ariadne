"""Opt-in hosted 9B quality diagnostic using the previous Qwen request recorder.

Replays frozen production generation prompts and checks hosted embedding vectors.
This is not a local Q4, M5, library, retrieval benchmark or UI qualification.
Ordinary pytest never sends paid calls.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from dataclasses import replace
from pathlib import Path

import httpx
import numpy as np
from dotenv import dotenv_values

from docgen.llm.deepinfra import DeepInfraProvider
from docgen.llm.local import parse_chat_response
from docgen.validator import ContentValidator
from embedding import EmbeddingConfig, EmbeddingService, doc_embedding_text
from library.prepared import validate_vector
from tests.live_model_smoke import PROFILES, ROOT, Budget, RecordedTransport, Run, save_json
from tests.source_verification_cases import CASES_BY_NAME, observe_case

REMOTE_PROFILE = replace(PROFILES['qwen'], model='Qwen/Qwen3.5-9B',
                         input_rate=0.10, reserve_input_rate=0.10, output_rate=0.15)
ARCHIVE = ROOT / 'evaluation/mac-local-models-2026-10-22/openai/requests'
FROZEN_CASES = {'cache': '001', 'guarded_cache': '003', 'token_expiry': '005'}
REPETITIONS = 2


def frozen_requests() -> dict[str, dict]:
    provider = DeepInfraProvider(model=REMOTE_PROFILE.model, api_key='')
    requests = {}
    for name, number in FROZEN_CASES.items():
        original = json.loads((ARCHIVE / f'{number}.json').read_text())
        messages = original['messages']
        if ([m['role'] for m in messages] != ['system', 'user']
                or CASES_BY_NAME[name].source.strip() not in messages[1]['content']):
            raise ValueError(f'Frozen generation prompt does not match {name}')
        requests[name] = {
            **provider._request_payload(messages[0]['content'], messages[1]['content'], 8192),
            'temperature': 0.7, 'top_p': 0.8, 'top_k': 20, 'min_p': 0.0,
            'presence_penalty': 1.5, 'repetition_penalty': 1.0,
            'chat_template_kwargs': {'enable_thinking': False},
        }
    return requests


async def check_embeddings(run: Run, api_key: str, transport_factory) -> None:
    service = EmbeddingService(EmbeddingConfig(
        provider='deepinfra', model=REMOTE_PROFILE.embedding_model, api_key=api_key))
    service._client = httpx.AsyncClient(
        base_url=REMOTE_PROFILE.embedding_url, timeout=180, trust_env=False,
        headers={'Authorization': f'Bearer {api_key}'},
        transport=RecordedTransport(run, 'embeddings', transport_factory()),
    )
    try:
        entries = run.report['documents']
        documents = await service.embed_documents([
            doc_embedding_text(entry['case'], (run.output / 'documents' / entry['file']).read_text())
            for entry in entries
        ])
        queries = [await service.embed_query(CASES_BY_NAME[name].question) for name in FROZEN_CASES]
        for vector in [*documents, *queries]:
            validate_vector(vector)
        np.savez_compressed(run.output / 'vectors.npz', documents=np.asarray(documents),
                            queries=np.asarray(queries))
        run.report['embeddings'] = {
            'model': REMOTE_PROFILE.embedding_model, 'hosted_precision': 'undisclosed',
            'document_vectors': len(documents), 'query_vectors': len(queries),
            'native_dimensions': 4096, 'stored_dimensions': 3072,
            'recipe': 'Production DeepInfra document/query framing, prefix reduction and normalization',
            'ranking_observations': {
                name: [{'document': entries[i]['file'], 'score': float(scores[i])}
                       for i in np.argsort(-scores)]
                for name, query in zip(FROZEN_CASES, queries, strict=True)
                for scores in [np.asarray(documents) @ query]
            },
            'limit': 'Six documents and three queries are a vector smoke check, not a retrieval benchmark.',
        }
    finally:
        await service.close()


def prepare_run(output: Path, budget: Budget, transport_factory, pacing: float) -> tuple[Run, dict]:
    output.mkdir(parents=True, exist_ok=False)
    requests = frozen_requests()
    run = Run(REMOTE_PROFILE, output, budget, pacing=pacing)
    run.report.update(
        rates_verified_on='2026-10-22', scope='Hosted 9B generation and 8B embedding diagnostic',
        transport_mode='live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated',
        quality_verdict='unreviewed', hosted_generation_precision='bfloat16',
        local_q4_qualified=False, hardware_qualified=False,
    )
    source_paths = (Path(__file__), ROOT / 'tests/live_model_smoke.py',
                    ROOT / 'docgen/llm/deepinfra.py', ROOT / 'docgen/llm/local.py',
                    ROOT / 'embedding.py', ROOT / 'model_config.py',
                    ROOT / 'tests/source_verification_cases.py')
    protocol = {
        'cases': {name: {'source_sha256': hashlib.sha256(CASES_BY_NAME[name].source.encode()).hexdigest(),
                         'oracle': observe_case(CASES_BY_NAME[name]), 'request': payload}
                  for name, payload in requests.items()},
        'repetitions': REPETITIONS, 'concurrency': 1,
        'budget_usd': budget.limit, 'max_attempts': budget.max_requests,
        'hosted_generation_precision': 'bfloat16', 'local_q4_qualified': False,
        'sampling_limit': 'Original local min_p was not captured; this run explicitly uses zero.',
        'pricing_source': 'https://deepinfra.com/Qwen/Qwen3.5-9B/api',
        'embedding_pricing_source': 'https://deepinfra.com/Qwen/Qwen3-Embedding-8B/api',
        'source_hashes': {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_paths},
    }
    save_json(output / 'protocol.json', protocol)
    for name in requests:
        (output / 'sources' / f'{name}.py').write_text(CASES_BY_NAME[name].source)
    run.save()
    return run, requests


def response_content(body: dict) -> str:
    if body.get('model') != REMOTE_PROFILE.model:
        raise ValueError('Response does not identify the requested 9B model')
    return parse_chat_response(body)[0]


async def execute(output: Path, budget: Budget, *, api_key: str,
                  transport_factory=httpx.AsyncHTTPTransport, pacing: float = 15) -> dict:
    run, requests = prepare_run(output, budget, transport_factory, pacing)
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            base_url=REMOTE_PROFILE.base_url, timeout=180, trust_env=False,
            headers={'Authorization': f'Bearer {api_key}'},
            transport=RecordedTransport(run, 'chat', transport_factory()),
        ) as client:
            for name, payload in requests.items():
                for repetition in range(1, REPETITIONS + 1):
                    response = await client.post('/chat/completions', json=payload)
                    response.raise_for_status()
                    content = response_content(response.json())
                    filename = f'{name}-r{repetition}.md'
                    (output / 'documents' / filename).write_text(content)
                    validation = ContentValidator().validate(content, 'qa', name)
                    run.report['documents'].append({
                        'case': name, 'repetition': repetition, 'file': filename,
                        'structurally_valid': validation.is_valid,
                        'issues': [{'level': issue.level, 'code': issue.code} for issue in validation.issues],
                    })
                    run.save()
        await check_embeddings(run, api_key, transport_factory)
        run.report['status'] = 'completed_pending_review'
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        raise
    except Exception as exc:  # noqa: BLE001 — retain failed paid-run evidence without claiming qualification.
        run.report.update(status='failed', error=f'{type(exc).__name__}: {exc}'.replace(api_key, '<redacted>'))
    finally:
        run.report['seconds'] = time.monotonic() - started
        run.save()
    return run.report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, default=1.0)
    args = parser.parse_args()
    key = os.environ.get('DEEPINFRA_API_KEY') or dotenv_values(ROOT / '.env').get('DEEPINFRA_API_KEY')
    if not key:
        parser.error('DEEPINFRA_API_KEY is missing')
    report = asyncio.run(execute(args.output.resolve(), Budget(args.budget_usd, max_requests=30), api_key=key))
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=2))
    return 0 if report['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
