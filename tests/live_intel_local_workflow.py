"""Opt-in native Intel smoke under a profile saved by the real web API.

This is backend generation under a UI-saved project, NOT the UI Generate or
durable onboarding pipeline. No server is launched and no weight is downloaded.
Run with ``python -m tests.live_intel_local_workflow --output <new-scratch-dir>``.
Raw native responses can include model reasoning; they stay in the local archive.
Structural success is deliberately separate from independent factual review.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml
from aiohttp import ClientSession, ClientTimeout, web

import config as config_module
from ariadne_mcp.service import AriadneService
from config import Config
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.llm.local import LocalInferenceError, LocalTransport, parse_chat_response
from docgen.validator import ContentValidator
from inference import InferenceContext
from local_models import MAC_ACCURACY_ARTIFACTS, mac_accuracy_profile
from model_config import identity_hash, resolve_models
from tests.source_verification_cases import CASES_BY_NAME, observe_case
from web.external_models import build_profile, register_routes
from web.projects import ProjectStore
from writer import ChunkConfig, LibraryWriter

ROOT = Path(__file__).resolve().parents[1]
CASE = CASES_BY_NAME['token_expiry']
SEARCH_QUESTION = 'How does consume_token consume a token when its deadline is reached?'
ASK_QUESTION = CASE.question
PHASE_CAPS = {'connection_and_saved_project': 2048, 'generate_qa': 8192, 'ask': 4096}
SCOPE = ('Backend generation under a project saved through POST /api/external-models; '
         'not full UI Generate/onboarding, compiler indexing, or hardware qualification')
CONTROLS = {
    'shared': ('Shared token policy',
               '# Shared token policy\n\n## Scope control\n\n'
               'This authored fixture documents the shared source policy used by app. '
               'Its EXPIRY_POLICY constant says that token consumption removes a token '
               'before checking its deadline. This is an authored dependency control.\n\n'
               '## Boundary\n\nThe shared policy applies to app; unrelated implements '
               'a separate retention policy and is outside this dependency scope.\n'),
    'unrelated': ('Unrelated retention policy',
                  '# Unrelated retention policy\n\n## Scope control\n\n'
                  'This authored fixture belongs to unrelated, which app does not depend on. '
                  'retain_expired checks the deadline before removing a token, so an expired '
                  'token remains stored when ValueError is raised.\n\n'
                  '## Boundary\n\nUNRELATED_RETENTION_SENTINEL identifies this out-of-scope '
                  'control. This behavior must not be attributed to app.consume_token.\n'),
}
PROVENANCE = (
    'tests/live_intel_local_workflow.py', 'tests/source_verification_cases.py',
    'web/external_models.py', 'web/projects.py', 'local_models.py', 'model_config.py',
    'docgen/generator.py', 'docgen/llm/local.py', 'docgen/validator.py', 'embedding.py',
    'writer.py', 'inference.py', 'llm.py', 'ariadne_mcp/service.py',
    'ariadne_mcp/service_search.py', 'ariadne_mcp/service_analysis.py',
)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + '\n')


def vector_record(vector) -> dict:
    if (not isinstance(vector, np.ndarray) or vector.shape != (3072,)
            or vector.dtype != np.float32 or not np.isfinite(vector).all()
            or not np.isclose(np.linalg.norm(vector.astype(np.float64)), 1)):
        raise ValueError('Expected a finite normalized 3072-dimensional float32 vector')
    return {'dimensions': 3072, 'dtype': str(vector.dtype),
            'sha256': digest(vector.tobytes()),
            'norm': float(np.linalg.norm(vector.astype(np.float64)))}


def snapshot(library) -> dict:
    records = {}
    for doc in library.list_documents():
        records[doc.id] = {
            'content': doc.content, 'title': doc.title, 'content_type': doc.content_type,
            'source_name': doc.source_name, 'source_files': list(doc.source_files),
            'metadata': doc.metadata, 'updated_at': str(doc.updated_at),
            'vector': vector_record(doc.embedding),
            'chunks': [{'content': item.content, 'vector': vector_record(item.embedding)}
                       for item in library.get_chunks(doc.id)],
            'sections': [{'heading': item.heading, 'content': item.content,
                          'vector': vector_record(item.embedding)}
                         for item in library.get_sections(doc.id)],
        }
    return records


def assert_persisted(before: dict, after: dict) -> None:
    if not before or before != after:
        raise ValueError('Persistence mismatch: content, provenance, children or vector bytes changed')


def fact_review(text: str) -> dict:
    return {'verdict': 'requires_independent_review', 'automated_correctness_claim': False,
            'expected_facts': CASE.correct, 'output': text,
            'review_instruction': 'Compare every material claim against frozen source and oracle; '
                                  'normal completion, keywords and schema validity are not correctness.',
            'limitation': 'Ask can use the authored dependency facts. A correct Ask answer '
                          'does not establish that the independently generated QA is factual.'}


def completed_chat(run, row) -> bool:
    if row.get('http_status') != 200 or row.get('status') != 'received':
        return False
    try:
        content, _usage = parse_chat_response(json.loads(
            (run.output / row['response_file']).read_bytes()))
        return bool(content.strip())
    except (KeyError, OSError, ValueError, TypeError, LocalInferenceError):
        return False


class Run:
    def __init__(self, output: Path, frozen: dict[str, str]):
        self.output, self.frozen = output, frozen
        self.phase_name = 'prepared'
        self.report = {'scope': SCOPE, 'status': 'prepared', 'structural_verdict': 'not_run',
                       'factual_verdict': 'requires_independent_review',
                       'hardware_qualified': False, 'phases': [], 'requests': [], 'checks': []}

    def save(self):
        save_json(self.output / 'report.json', self.report)

    def verify(self):
        for name, expected in self.frozen.items():
            if digest(Path(name).read_bytes()) != expected:
                raise ValueError(f'Frozen input changed: {name}')

    def freeze_saved(self, *paths):
        self.verify()
        for path in paths:
            name = str(Path(path).resolve())
            self.frozen[name] = digest(Path(name).read_bytes())
            self.report.setdefault('saved_input_hashes', {})[name] = self.frozen[name]
        self.save()

    def require(self, name, condition, **details):
        self.report['checks'].append({'name': name, 'passed': bool(condition), **details})
        self.save()
        if not condition:
            raise ValueError(f'Workflow check failed: {name}')

    @contextmanager
    def phase(self, name):
        previous, self.phase_name = self.phase_name, name
        started = time.monotonic()
        entry = {'name': name, 'status': 'running', 'started_unix': time.time()}
        self.report['phases'].append(entry)
        self.save()
        print(f'{name}: starting', flush=True)
        try:
            self.verify()
            yield
            entry['status'] = 'completed'
        except BaseException:
            entry['status'] = 'failed'
            raise
        finally:
            entry['seconds'] = time.monotonic() - started
            self.phase_name = previous
            self.save()
            print(f'{name}: {entry["status"]} ({entry["seconds"]:.1f}s)', flush=True)

    @contextmanager
    def record_native_requests(self):
        """Observe production HTTP clients without replacing native model behavior."""
        original = LocalTransport.client
        run = self

        def client(transport):
            instance = original(transport)
            if not getattr(instance, '_intel_workflow_recorded', False):
                instance.event_hooks['request'].append(run.request_started)
                instance.event_hooks['response'].append(run.response_received)
                instance._intel_workflow_recorded = True
            return instance

        with patch.object(LocalTransport, 'client', client):
            yield

    async def request_started(self, request):
        self.verify()
        number = len(self.report['requests']) + 1
        if number > 200:
            raise ValueError('Diagnostic request ceiling reached')
        payload = json.loads(request.content) if request.content else None
        if request.url.path.endswith('/chat/completions'):
            ceiling = PHASE_CAPS.get(self.phase_name)
            if (ceiling is None or type(payload.get('max_tokens')) is not int
                    or not 0 < payload['max_tokens'] <= ceiling):
                raise ValueError('Chat request exceeds its frozen phase ceiling or has an unexpected phase')
            settings = mac_accuracy_profile()['llm']['request_options']
            if (payload['model'] != 'Qwen/Qwen3.8-27B'
                    or any(payload.get(key) != value for key, value in settings.items())):
                raise ValueError('Chat request changed the selected native thinking recipe')
        row = {'number': number, 'phase': self.phase_name, 'method': request.method,
               'url': str(request.url), 'status': 'started', 'started_unix': time.time(),
               'timeout': request.extensions.get('timeout'),
               'request_file': f'requests/{number:03}.json'}
        request.extensions['intel_workflow_row'] = row
        request.extensions['intel_workflow_started'] = time.monotonic()
        save_json(self.output / row['request_file'], payload)
        self.report['requests'].append(row)
        self.save()

    async def response_received(self, response):
        row = response.request.extensions['intel_workflow_row']
        raw = await response.aread()
        name = f'responses/{row["number"]:03}.bin'
        (self.output / name).write_bytes(raw)
        row.update(status='received', http_status=response.status_code, response_file=name,
                   response_sha256=digest(raw), response_bytes=len(raw),
                   seconds=time.monotonic() - response.request.extensions['intel_workflow_started'])
        try:
            body = json.loads(raw)
            row['usage'] = body.get('usage') if isinstance(body, dict) else None
        except ValueError:
            row['malformed_json'] = True
        self.save()


def prepare(output: Path, *, chat_url: str, embedding_url: str, timeout: int,
            config_path: Path | None = None) -> tuple[Run, Config]:
    output = output.resolve()
    config_path = (config_path or output / 'root' / 'ariadne.yaml').resolve()
    if not config_path.is_relative_to(output):
        raise ValueError('Scratch config must be inside the new output directory')
    body = {'chat_url': chat_url, 'embedding_url': embedding_url,
            'request_timeout_seconds': timeout}
    profile = build_profile(body)
    observations = observe_case(CASE)
    output.mkdir(parents=True, exist_ok=False)
    for name in ('requests', 'responses', 'documents', 'provenance'):
        (output / name).mkdir()
    fixtures = {
        'app/tokens.py': CASE.source,
        'shared/policy.py': 'EXPIRY_POLICY = "remove token before checking deadline"\n',
        'unrelated/tokens.py': 'def retain_expired(tokens, token, now):\n'
                               '    user, deadline = tokens[token]\n'
                               '    if now >= deadline:\n'
                               '        raise ValueError("expired")\n'
                               '    tokens.pop(token)\n    return user\n',
    }
    frozen = {}
    for relative, content in fixtures.items():
        path = output / 'sources' / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        frozen[str(path)] = digest(path.read_bytes())
    data = {**profile, 'sources': {name: {'path': str(output / 'sources' / name)}
                                  for name in ('app', 'shared', 'unrelated')},
            'default_source': 'app', 'themes_enabled': False, 'ask_synthesis': True}
    data['sources']['app']['depends_on'] = ['shared']
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(data, sort_keys=False))
    frozen[str(config_path)] = digest(config_path.read_bytes())
    source_hashes = {}
    for name in PROVENANCE:
        path = ROOT / name
        target = output / 'provenance' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        source_hashes[name] = digest(target.read_bytes())
        frozen[str(path)] = frozen[str(target)] = source_hashes[name]
    protocol = {'scope': SCOPE, 'request_body': body, 'profile': profile,
                'profile_hash': resolve_models(profile).profile_hash,
                'deployment_hash': identity_hash(profile), 'source_hashes': source_hashes,
                'expected_artifacts': {
                    role: {'repository': artifact.repository, 'revision': artifact.revision,
                           'filename': artifact.filename, 'sha256': artifact.sha256,
                           'size_bytes': artifact.size}
                    for role, artifact in MAC_ACCURACY_ARTIFACTS.items()},
                'artifact_attestation': 'Endpoint aliases do not attest loaded weights; '
                                        'operator must record verified files and server commands separately.',
                'fixture_hashes': {name: digest(content.encode()) for name, content in fixtures.items()},
                'facts': {'expected': CASE.correct, 'observations': observations},
                'controls': CONTROLS, 'control_chunk_size': 128,
                'search_question': SEARCH_QUESTION, 'ask_question': ASK_QUESTION,
                'phase_caps': PHASE_CAPS,
                'maximum_http_requests': 200, 'retry_policy': 'One transport attempt per operation',
                'response_archive': 'Exact native response bytes; no request headers or credentials',
                'factual_verdict': 'requires_independent_review'}
    save_json(output / 'protocol.json', protocol)
    frozen[str(output / 'protocol.json')] = digest((output / 'protocol.json').read_bytes())
    run = Run(output, frozen)
    run.report['protocol_sha256'] = frozen[str(output / 'protocol.json')]
    run.save()
    return run, Config(config_path)


async def save_through_api(run: Run, cfg: Config, body: dict, timeout: int):
    store = ProjectStore(cfg)
    app = web.Application()
    app['projects'] = store
    register_routes(app)
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, '127.0.0.1', 0)
        await site.start()
        address = runner.addresses[0]
        async with ClientSession(timeout=ClientTimeout(total=timeout + 30)) as client:
            async with client.post(f'http://127.0.0.1:{address[1]}/api/external-models',
                                   json=body) as response:
                result = await response.json()
                save_json(run.output / 'connection-response.json', result)
                run.require('real_api_saved_profile', response.status == 201,
                            http_status=response.status)
    finally:
        await runner.cleanup()
        await store.close()
    restarted = ProjectStore(Config(cfg.config_path))
    try:
        key = result['profile_id']
        saved_profile = restarted.external_profiles()[key]
        run.require('saved_profile_exact', saved_profile == build_profile(body))
        project = restarted.prepare(key, 'app', themes_enabled=False)
        project_cfg = Config(restarted.path(project['id']))
        run.require('saved_project_model_settings', project_cfg.model_settings() == resolve_models(saved_profile))
        run.require('saved_dependencies', project_cfg.sources['app']['depends_on'] == ['shared']
                    and project_cfg.scope_closure('app') == frozenset({'app', 'shared'}))
        run.freeze_saved(project_cfg.config_path, restarted.external_profiles_path)
        save_json(run.output / 'project.json', {**project, 'config_path': str(project_cfg.config_path),
                                              'root_config_path': str(cfg.config_path)})
        return project_cfg
    finally:
        await restarted.close()


async def workflow(run: Run, cfg: Config):
    context = InferenceContext.from_config(cfg)
    before, identity = {}, None
    try:
        with context.single_attempt(), context.write_session(), context.open_library(operation='write') as library:
            identity = context.binding.manifest
            with run.phase('generate_qa'):
                async with DocGenerator(GeneratorConfig(model=context.settings.chat.model,
                                                       provider='local', inference_context=context)) as generator:
                    docs = await generator.generate_for_file(cfg.get_source_path('app') / 'tokens.py', ('qa',))
                run.require('one_generated_qa', len(docs) == 1)
                doc = docs[0]
                (run.output / 'documents' / 'token-expiry-qa.md').write_text(doc.content)
                save_json(run.output / 'generation-factual-review.json', fact_review(doc.content))
                validation = ContentValidator().validate(doc.content, doc.doc_type, doc.title)
                run.require('generated_structure', validation.is_valid,
                            issues=[{'level': item.level, 'code': item.code} for item in validation.issues])
                run.require('generated_source_snapshot', doc.source_snapshot == digest(CASE.source.encode()))
            with run.phase('persist_vectors'):
                await LibraryWriter(library).add_document(
                    doc.doc_type, doc.title, doc.content, list(doc.source_files),
                    metadata=doc.metadata, source_name='app', doc_id='generated-token-expiry')
                controls_writer = LibraryWriter(library, chunk_config=ChunkConfig(chunk_size=128, chunk_overlap=16))
                for name, (title, content) in CONTROLS.items():
                    (run.output / 'documents' / f'authored-{name}.md').write_text(content)
                    await controls_writer.add_document('explanation', title, content,
                                                      [str(next(cfg.get_source_path(name).glob('*.py')))],
                                                      source_name=name, doc_id=f'authored-{name}',
                                                      metadata={'fixture_kind': 'authored_scope_control'})
                before = snapshot(library)
                run.require('three_persisted_documents', len(before) == 3)
                run.require('authored_control_child_vectors', all(
                    before[f'authored-{name}']['chunks'] and before[f'authored-{name}']['sections']
                    for name in CONTROLS))
                save_json(run.output / 'before-reopen.json', before)
                save_json(run.output / 'binding.json', identity)
    finally:
        await context.aclose()
    reopened = InferenceContext.from_config(Config(cfg.config_path))
    service = AriadneService()
    service._config, service._inference_context = cfg, reopened
    try:
        with run.phase('reopen'):
            with reopened.open_library(operation='inspect') as library:
                after = snapshot(library)
                assert_persisted(before, after)
                run.require('binding_reopened', reopened.binding.manifest == identity)
                save_json(run.output / 'after-reopen.json', after)
        with reopened.single_attempt():
            with run.phase('search'):
                found = await service.search(SEARCH_QUESTION, source='app', sections_only=True, limit=8)
                save_json(run.output / 'search.json', found.model_dump(mode='json'))
                run.require('search_dependency_scope', {item.source_name for item in found.documents} == {'app', 'shared'})
                run.require('search_generated_evidence', any(item.id == 'generated-token-expiry' for item in found.documents))
            with run.phase('ask'):
                answer = await service.ask(ASK_QUESTION, source='app')
                save_json(run.output / 'answer.json', answer.model_dump(mode='json'))
                save_json(run.output / 'answer-factual-review.json', fact_review(answer.answer))
                rows = [row for row in run.report['requests']
                        if row['phase'] == 'ask' and row['url'].endswith('/chat/completions')]
                calls = [json.loads((run.output / row['request_file']).read_text()) for row in rows]
                run.require('ask_real_synthesis', bool(rows) and all(completed_chat(run, row) for row in rows)
                            and bool(answer.answer.strip()) and answer.llm_calls > 0
                            and not any(reason.startswith('synthesis failed:')
                                        for reason in answer.confidence_reasons),
                            llm_calls=answer.llm_calls, confidence_reasons=answer.confidence_reasons)
                prompts = json.dumps([item['messages'] for item in calls])
                run.require('ask_dependency_context', 'Shared token policy' in prompts
                            and 'UNRELATED_RETENTION_SENTINEL' not in prompts)
                run.require('ask_sources', bool(answer.sources) and 'Unrelated retention policy' not in answer.sources)
    finally:
        await service.close()
        await reopened.aclose()


async def execute(output: Path, *, chat_url='http://127.0.0.1:18080/v1',
                  embedding_url='http://127.0.0.1:18081/v1', timeout=14400,
                  config_path: Path | None = None):
    run, root_cfg = prepare(output, chat_url=chat_url, embedding_url=embedding_url,
                            timeout=timeout, config_path=config_path)
    previous = config_module._global_config
    started = time.monotonic()
    try:
        with run.record_native_requests():
            with run.phase('connection_and_saved_project'):
                cfg = await save_through_api(run, root_cfg,
                                            json.loads((run.output / 'protocol.json').read_text())['request_body'],
                                            timeout)
            config_module._global_config = cfg
            await workflow(run, cfg)
        run.report.update(status='passed_structure_pending_factual_review', structural_verdict='passed')
    except asyncio.CancelledError:
        run.report.update(status='cancelled', structural_verdict='incomplete')
        raise
    except Exception as error:  # noqa: BLE001 - archive failed live diagnostics without retry.
        run.report.update(status='failed', structural_verdict='failed',
                          error_type=type(error).__name__, error=str(error))
    finally:
        config_module._global_config = previous
        for row in run.report['requests']:
            if row['status'] == 'started':
                row.update(status='no_response_received',
                           seconds=time.time() - row['started_unix'])
        run.report['seconds'] = time.monotonic() - started
        run.report['inference_requests'] = sum(row['url'].endswith(('/chat/completions', '/embeddings'))
                                              for row in run.report['requests'])
        run.save()
    return run.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, help='New scratch root config, inside --output')
    parser.add_argument('--chat-url', default='http://127.0.0.1:18080/v1')
    parser.add_argument('--embedding-url', default='http://127.0.0.1:18081/v1')
    parser.add_argument('--timeout', type=int, default=14400)
    args = parser.parse_args()
    report = asyncio.run(execute(args.output, chat_url=args.chat_url, embedding_url=args.embedding_url,
                                 timeout=args.timeout, config_path=args.config))
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'inference_requests')}))
    return 0 if report['structural_verdict'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
