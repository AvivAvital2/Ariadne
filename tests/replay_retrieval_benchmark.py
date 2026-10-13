"""Recheck retrieval changes against saved provider vectors, without network access.

python -m tests.replay_retrieval_benchmark --source ORIGINAL_RUN --output NEW_DIRECTORY
All 50 questions are replayed. Original quality scores and evidence are unchanged.
"""

import argparse
import asyncio
import gzip
import hashlib
import json
import os
import shutil
from pathlib import Path

import httpx

from tests.live_model_smoke import PROFILES, ROOT, Budget, save_json
from tests.live_retrieval_benchmark import execute
from tests.retrieval_cases import FIXTURE


def response_key(payload):
    return json.dumps(payload, sort_keys=True)


def replay_transport(source):
    responses, used = {}, set()
    for path in sorted((source / 'requests').glob('*.json')):
        payload = json.loads(path.read_text())
        response_path = source / 'responses' / path.name
        response = (
            response_path.read_bytes()
            if response_path.exists()
            else gzip.decompress(response_path.with_suffix('.json.gz').read_bytes())
        )
        if 'data' in json.loads(response):
            assert 'input' in payload and 'messages' not in payload
            responses[response_key(payload)] = response

    def respond(request):
        assert request.method == 'POST' and request.url.path.endswith('/embeddings')
        key = response_key(json.loads(request.content))
        assert key in responses, 'No recorded response for this exact request'
        used.add(key)
        return httpx.Response(
            200, content=responses[key], headers={'content-type': 'application/json'}
        )

    return lambda: httpx.MockTransport(respond), responses.keys(), used


async def replay(source, output):
    corpus_bytes = (source / 'corpus.json').read_bytes()
    assert corpus_bytes == FIXTURE.read_bytes(), 'Replay must use the unchanged corpus and labels'
    output.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(source / 'corpus.json', output / 'corpus.json')
    for path in (
        Path(__file__),
        ROOT / 'tests/live_retrieval_benchmark.py',
        ROOT / 'ariadne_mcp/service_search.py',
    ):
        shutil.copyfile(path, output / (path.name + '.txt'))
    original = json.loads((source / 'summary.json').read_text())
    summary = {
        'source': str(source),
        'corpus_sha256': hashlib.sha256(corpus_bytes).hexdigest(),
        'provider_spending_usd': 0,
        'method': 'Post-fix replay of every frozen question using exact successful recorded embedding responses. No HTTP network transport, new model output, changed labels or threshold tuning. Not an independent held-out experiment.',
        'profiles': {},
    }
    for name in ('qwen', 'openai'):
        profile = PROFILES[name]
        for key in {profile.key_env, profile.embedding_key}:
            os.environ[key] = 'recorded-replay-only'
        factory, expected, used = replay_transport(source / name)
        report = await execute(
            profile,
            output / name,
            Budget(0.5),
            json.loads(corpus_bytes),
            transport_factory=factory,
            pacing=0,
        )
        assert report['status'] == 'passed', report.get('error') or report['checks']
        assert set(expected) == used, 'Every recorded successful request must be used'
        summary['profiles'][name] = {
            'status': report['status'],
            'implementation_checks': len(report['checks']),
            'before': original['profiles'][name]['metrics'],
            'after': report['metrics'],
            'transport_mode': 'recorded',
            'provider_spending_usd': 0,
            'exact_recorded_requests': len(used),
        }
        save_json(output / 'summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(replay(args.source.resolve(), args.output.resolve()))


if __name__ == '__main__':
    main()
