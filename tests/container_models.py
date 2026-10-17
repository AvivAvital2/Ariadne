"""Qwen protocol fixtures and clew checks for the existing Docker browser journey.

Mounted only by compose.qwen.yaml; no model weights, keys or paid calls.
"""
from __future__ import annotations

import asyncio
import json
import sys
import threading
from pathlib import Path
from urllib.request import urlopen

import yaml

from model_config import CHAT_MODEL, EMBEDDING_MODEL, QUERY_INSTRUCTION
from tests.local_model_server import Deployment, local_config


def serve():
    config = local_config('http://qwen-fixture:8100/chat/v1', 'http://qwen-fixture:8101/embed/v1')
    config['sources'] = {}
    Path('/data/local-template.yaml').write_text(yaml.safe_dump(config))
    chat = Deployment(CHAT_MODEL, '/chat', host='0.0.0.0', port=8100)
    embed = Deployment(EMBEDDING_MODEL, '/embed', host='0.0.0.0', port=8101)

    def reply():
        messages = json.dumps(chat.requests[-1][1]['messages'])
        if 'C<n>: K<id>' in messages:
            content = 'C1: K1'
        elif 'C<n>: S<id>' in messages:
            content = 'C1: S1 S2 S3'
        elif 'IDs only' in messages or 'IDs, not prose' in messages:
            content = 'M1 R1 B1'
        elif 'record_receipt' in messages:
            content = 'payments.accept_payment calls authorize_payment, which calls record_receipt.'
        else:
            content = '# Retry policy\n\n## Overview\n\n' + (
                'The retry helper applies exponential backoff with bounded attempts. '
                'retry_delay computes the delay, and should_retry checks the attempt limit.\n\n'
            ) * 16
        chat.chat_response = {
            'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 32, 'completion_tokens': 200},
        }

    chat.barrier = reply
    try:
        threading.Event().wait()
    finally:
        chat.close()
        embed.close()


async def seed_clews():
    from config import get_config
    from inference import InferenceContext
    from library.clew_embeddings import embed_clews
    from tests.clew_workflow import add_guide, build_routes, seed_graph

    cfg = get_config()
    context = InferenceContext.from_config(cfg)
    try:
        with context.activate(), context.write_session():
            with context.open_library(operation='write') as library:
                source = cfg.default_source
                with library._conn_provider.acquire() as conn:
                    languages = {row[0] for row in conn.execute(
                        'SELECT DISTINCT language FROM scip_symbols WHERE source_name=?', (source,))}
                    assert {'python', 'go'} <= languages, languages
                seed_graph(library, cfg.resolve_source(source), source)
                assert build_routes(library, source) == 0
                assert await embed_clews(library, source_name=source) >= 2
                await add_guide(library, source)
    finally:
        await context.aclose()


def check_requests(preview=False):
    with urlopen('http://qwen-fixture:8100/__requests') as response:
        chat = json.load(response)
    with urlopen('http://qwen-fixture:8101/__requests') as response:
        embed = json.load(response)
    if preview:
        from config import get_config

        cfg = get_config()
        assert cfg.provider == 'local'
        assert cfg.default_source == 'retryutils'
        assert not Path(cfg.db_path).exists()
        assert not Path(cfg.staleness_db_path).exists()
        assert all(payload is None for _, payload, _ in chat + embed)
        assert yaml.safe_load(Path('/data/local-template.yaml').read_text())['sources'] == {}
        return
    assert any(route.endswith('/tokenize') for route, _, _ in chat)
    assert any(route.endswith('/tokenize') for route, _, _ in embed)
    inputs = [text for route, payload, _ in embed if route.endswith('/embeddings')
              for text in payload['input']]
    assert any(text.startswith(f'Instruct: {QUERY_INSTRUCTION}\nQuery:') for text in inputs)
    assert any(not text.startswith('Instruct:') for text in inputs)
    assert all('Authorization' not in headers for _, _, headers in chat + embed)
    assert any("return {'receipt': amount}" in json.dumps(payload)
               for route, payload, _ in chat if route.endswith('/chat/completions'))
    import sqlite3
    from config import get_config
    from library.model_binding import inspect_binding

    cfg = get_config()
    assert inspect_binding(Path(cfg.db_path)) is not None
    with sqlite3.connect(cfg.db_path) as conn:
        assert conn.execute('SELECT state FROM spool_runs').fetchall() == [('completed',)]
    print(json.dumps({'chat_requests': len(chat), 'embedding_requests': len(embed),
                      'clew_source_evidence': True}))


if __name__ == '__main__':
    if sys.argv[1] == 'serve':
        serve()
    elif sys.argv[1] == 'seed-clews':
        asyncio.run(seed_clews())
    elif sys.argv[1] == 'check':
        check_requests()
    elif sys.argv[1] == 'check-preview':
        check_requests(preview=True)
    else:
        raise SystemExit('Expected serve, seed-clews or check')
