"""Independent llama.cpp HTTP fixture; no production request builders."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class LlamaServer:
    def __init__(self, model, *, embeddings=False, thinking=False):
        self.model = model
        self.embeddings = embeddings
        self.thinking = thinking
        self.requests = []
        self.tokens = [101, 102, 103, 104]
        self.context = 32768
        self.overrides = {}
        self.failures = {}
        self.rendered = '<start>system\nuser<assistant><think>\n\n</think>\n\n'
        fixture = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args):
                pass

            def reply(self, status, body):
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(body).encode())

            def do_GET(self):
                fixture.requests.append((self.path, None))
                if fixture.failures.get(self.path):
                    self.reply(fixture.failures[self.path].pop(0), {'error': 'temporary failure'})
                elif self.path in fixture.overrides:
                    self.reply(200, fixture.overrides[self.path])
                elif self.path == '/native/v1/models':
                    self.reply(200, {'data': [{'id': fixture.model}]})
                elif self.path == '/native/props':
                    self.reply(200, {'default_generation_settings': {'n_ctx': fixture.context}})
                else:
                    self.reply(404, {'error': 'unsupported route'})

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                fixture.requests.append((self.path, payload))
                if fixture.failures.get(self.path):
                    self.reply(fixture.failures[self.path].pop(0), {'error': 'temporary failure'})
                elif self.path in fixture.overrides:
                    self.reply(200, fixture.overrides[self.path])
                elif self.path == '/native/apply-template':
                    if (payload.get('chat_template_kwargs') != {'enable_thinking': fixture.thinking}
                            or payload.get('add_generation_prompt') is not True):
                        self.reply(400, {'error': 'incorrect template options'})
                    else:
                        self.reply(200, {'prompt': fixture.rendered})
                elif self.path == '/native/tokenize':
                    if (not isinstance(payload.get('content'), str)
                            or payload.get('add_special') is not True
                            or payload.get('parse_special') is not True
                            or payload.get('with_pieces') is not False
                            or 'messages' in payload or 'prompt' in payload):
                        self.reply(400, {'error': 'not a llama.cpp tokenization request'})
                    else:
                        self.reply(200, {'tokens': fixture.tokens})
                elif self.path == '/native/v1/chat/completions':
                    if (payload.get('model') != fixture.model or fixture.embeddings
                            or payload.get('chat_template_kwargs') != {'enable_thinking': fixture.thinking}):
                        self.reply(400, {'error': 'incorrect chat request'})
                    else:
                        self.reply(200, {'choices': [{'message': {'content': 'Documented arithmetic.'},
                                                     'finish_reason': 'stop'}],
                                         'usage': {'prompt_tokens': len(fixture.tokens),
                                                   'completion_tokens': 5}})
                elif self.path == '/native/v1/embeddings' and fixture.embeddings:
                    if payload.get('model') != fixture.model or 'dimensions' in payload:
                        self.reply(400, {'error': 'incorrect embedding request'})
                    else:
                        rows = [{'index': i, 'embedding': [float(j + i + 1) for j in range(4096)]}
                                for i, _ in enumerate(payload['input'])]
                        self.reply(200, {'data': rows[::-1]})
                else:
                    self.reply(404, {'error': 'unsupported route'})

        self.server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base_url = f'http://127.0.0.1:{self.server.server_port}/native/v1'

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
