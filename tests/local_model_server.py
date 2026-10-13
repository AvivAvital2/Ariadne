"""Deterministic HTTP deployment; production request construction stays in use."""

from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx

from model_config import CHAT_MODEL, EMBEDDING_MODEL

DOCUMENT = (
    '# Core arithmetic\n\n## Overview\n\n'
    + (
        'The module implements arithmetic operations on two integers. The function accepts two '
        'arguments and returns their calculated result. Callers supply integers directly.\n\n'
    )
    * 16
    + '## Usage\n\n```python\nfrom core import add\nresult = add(2, 3)\n```\n\n## Behavior\n\nThe result is an integer.\n'
)


class Deployment:
    def __init__(self, model: str, prefix: str):
        self.model = model
        self.prefix = prefix
        self.requests = []
        self.failures = []
        self.chat_response = None
        self.embedding_response = None
        self.token_count = 32
        self.barrier = None
        deployment = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def do_GET(self):
                deployment.requests.append((self.path, None, dict(self.headers)))
                if self.path != prefix + '/v1/models':
                    self.reply(400, {'error': 'unexpected route'})
                    return
                self.reply(200, {'data': [{'id': model}]})

            def reply(self, status, body):
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Retry-After', '0')
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                deployment.requests.append((self.path, payload, dict(self.headers)))
                if payload.get('model') != model:
                    self.reply(400, {'error': 'incorrect model'})
                    return
                if deployment.failures:
                    self.reply(deployment.failures.pop(0), {'error': 'injected failure'})
                    return
                if self.path == prefix + '/tokenize':
                    n = deployment.token_count
                    if 'messages' in payload:
                        assert payload['chat_template_kwargs'] == {'enable_thinking': False}
                    else:
                        assert payload['add_special_tokens'] is True
                    self.reply(200, {'count': n, 'max_model_len': 32768, 'tokens': list(range(n))})
                elif self.path == prefix + '/v1/chat/completions' and model == CHAT_MODEL:
                    assert payload['stream'] is False
                    assert payload['chat_template_kwargs'] == {'enable_thinking': False}
                    assert 'extra_body' not in payload
                    if deployment.barrier:
                        deployment.barrier()
                    self.reply(
                        200,
                        deployment.chat_response
                        or {
                            'choices': [
                                {'message': {'content': DOCUMENT}, 'finish_reason': 'stop'}
                            ],
                            'usage': {'prompt_tokens': 32, 'completion_tokens': 200},
                        },
                    )
                elif self.path == prefix + '/v1/embeddings' and model == EMBEDDING_MODEL:
                    if deployment.barrier:
                        deployment.barrier()
                    assert 'dimensions' not in payload
                    assert payload['encoding_format'] == 'float'
                    rows = [
                        {'index': i, 'embedding': [float(j + 1) for j in range(4096)]}
                        for i, _ in enumerate(payload['input'])
                    ]
                    self.reply(200, deployment.embedding_response or {'data': rows[::-1]})
                else:
                    self.reply(400, {'error': 'unexpected inference route'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}{prefix}/v1'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


@contextmanager
def deployments():
    chat = Deployment(CHAT_MODEL, '/chat')
    embeddings = Deployment(EMBEDDING_MODEL, '/embed')
    try:
        yield chat, embeddings
    finally:
        chat.close()
        embeddings.close()


def local_config(chat_url='http://127.0.0.1:8000/v1', embed_url='http://127.0.0.1:8001/v1'):
    return {
        'defaults': {'db_path': 'state/local.db', 'staleness_db_path': 'state/local_staleness.db'},
        'llm': {
            'provider': 'local',
            'model': CHAT_MODEL,
            'model_revision': 'a' * 40,
            'weight_format': 'bf16',
            'base_url': chat_url,
            'api_key_env': None,
        },
        'embeddings': {
            'provider': 'local',
            'model': EMBEDDING_MODEL,
            'model_revision': 'b' * 40,
            'weight_format': 'bf16',
            'base_url': embed_url,
            'api_key_env': None,
        },
    }


def trap_inference(monkeypatch, endpoints):
    """Fail on unexpected egress before it reaches a network transport."""
    ports = {httpx.URL(endpoint).port for endpoint in endpoints}
    original = httpx.AsyncClient.send
    original_sync = httpx.Client.send

    def check(request):
        assert request.url.host == '127.0.0.1' and request.url.port in ports, (
            f'Forbidden inference host: {request.url}'
        )
        assert '/batches' not in request.url.path and '/files' not in request.url.path

    async def send(client, request, **kwargs):
        check(request)
        return await original(client, request, **kwargs)

    def send_sync(client, request, **kwargs):
        check(request)
        return original_sync(client, request, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, 'send', send)
    monkeypatch.setattr(httpx.Client, 'send', send_sync)
