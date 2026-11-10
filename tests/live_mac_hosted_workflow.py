"""Opt-in hosted orchestration check using the selected 27B thinking recipe.

The library retains its hosted DeepInfra identity. A harness-only provider maps
the local request recipe onto the hosted API; no native artifact or M5 is tested.
All model output requires separate source-based quality review.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
import time
from pathlib import Path

import httpx
import numpy as np
from attrs import evolve
from dotenv import dotenv_values

import config as config_module
from ariadne_mcp.service import AriadneService
from docgen.catalog_describer import describe_source_elements
from docgen.catalog_writer import sync_file_catalog
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.llm.local import build_chat_request
from docgen.validator import ContentValidator
from inference import InferenceContext
from library.prepared import validate_vector
from llm import capture_completion_trace
from local_models import mac_accuracy_profile
from model_config import resolve_models
from tests.live_mac_accuracy_probe import CONTEXT_WINDOW, PROFILE, FinalOnlyTransport
from tests.live_mac_call_probe import validated_content
from tests.live_model_smoke import (
    ROOT,
    Budget,
    RecordedTransport,
    Run,
    configuration,
    save_json,
    snapshot,
)
from tests.model_smoke_cases import CASES
from writer import LibraryWriter

MAX_ATTEMPTS = 60
TIMEOUT = 600
SOURCE = CASES[0][2]
QUESTION = 'How does mean handle an empty list and calculate a nonempty list average?'
DOC_TYPES = ('explanation', 'qa')
SOURCE_FILES = (
    'tests/live_mac_hosted_workflow.py', 'tests/test_live_mac_hosted_workflow.py',
    'tests/live_mac_accuracy_probe.py', 'tests/live_mac_call_probe.py',
    'tests/live_model_smoke.py', 'tests/model_smoke_cases.py',
    'docgen/generator.py', 'docgen/prompts.py', 'docgen/catalog_writer.py',
    'docgen/catalog_describer.py', 'docgen/llm/local.py', 'docgen/validator.py',
    'embedding.py', 'writer.py', 'inference.py', 'model_config.py', 'local_models.py',
    'llm.py', 'ariadne_mcp/service.py', 'ariadne_mcp/service_analysis.py',
    'library/prepared.py',
)


class WorkflowAbort(BaseException):
    """Stop the diagnostic without production best-effort catches or SDK retries.

    This is a harness control signal, not a production exception or provider change.
    It deliberately bypasses broad Exception handlers in optional hosted operations.
    """


class BoundedTransport(httpx.AsyncBaseTransport):
    def __init__(self, inner, verify=None):
        self.inner, self.verify = inner, verify

    async def aclose(self):
        await self.inner.aclose()

    async def handle_async_request(self, request):
        try:
            if self.verify is not None:
                self.verify()
            response = await self.inner.handle_async_request(request)
        except asyncio.CancelledError:
            raise
        except Exception as error:
            raise WorkflowAbort(type(error).__name__) from error
        if response.status_code != 200:
            status = response.status_code
            await response.aclose()
            raise WorkflowAbort(f'HTTP {status}')
        return response


def hosted_request(settings, system_prompt, user_prompt, max_tokens):
    payload = build_chat_request(settings, system_prompt, user_prompt, max_tokens)
    payload.pop('reasoning_format')
    payload['repetition_penalty'] = payload.pop('repeat_penalty')
    envelope = sum(len(message['content'].encode()) for message in payload['messages']) + 1024
    if envelope + max_tokens + 64 > CONTEXT_WINDOW:
        raise ValueError('Conservative context admission exceeded')
    return payload


class HostedThinkingProvider:
    """Experimental request mapping; configuration and storage remain hosted."""

    def __init__(self, run, api_key, transport_factory, verify=None):
        self.run = run
        self.model, self.base_url = PROFILE.model, PROFILE.base_url
        self.api_key = api_key
        self.settings = resolve_models(mac_accuracy_profile()).chat
        self.last_usage = {}
        self.client = httpx.AsyncClient(
            base_url=self.base_url, headers={'Authorization': f'Bearer {api_key}'},
            timeout=TIMEOUT, trust_env=False, follow_redirects=False,
            transport=BoundedTransport(RecordedTransport(
                run, 'chat', FinalOnlyTransport(run, transport_factory())), verify),
        )

    async def call(self, system_prompt, user_prompt, *, max_tokens=4096,
                   cache_system_prompt=True, timeout=None):
        del cache_system_prompt  # This hosted OpenAI-compatible adapter has no cache hint.
        effective_timeout = TIMEOUT if timeout is None else timeout
        if (type(effective_timeout) not in (int, float)
                or not math.isfinite(effective_timeout) or not 0 < effective_timeout <= TIMEOUT):
            raise WorkflowAbort('Invalid bounded timeout')
        record = None
        first_attempt = len(self.run.report['requests'])
        try:
            payload = hosted_request(self.settings, system_prompt, user_prompt, max_tokens)
            record = {'max_tokens': max_tokens, 'effective_timeout_seconds': effective_timeout,
                      'http_attempts': [], 'status': 'running'}
            self.run.report.setdefault('adapter_calls', []).append(record)
            self.run.save()
            response = await self.client.post('/chat/completions', json=payload,
                                              timeout=effective_timeout)
            content, usage = validated_content(response.json(), max_tokens)
            self.last_usage = {**usage, 'output_tokens': usage['completion_tokens'],
                               'input_tokens': usage['prompt_tokens'], 'stop_reason': 'end_turn'}
            record['status'] = 'completed'
            return content.replace(self.api_key, '<redacted>')
        except asyncio.CancelledError:
            if record is not None:
                record['status'] = 'cancelled'
            raise
        except Exception as error:
            raise WorkflowAbort(type(error).__name__) from error
        finally:
            if record is not None:
                record['http_attempts'] = list(range(first_attempt + 1, len(self.run.report['requests']) + 1))
                if record['status'] == 'running':
                    record['status'] = 'failed'
                self.run.save()

    async def aclose(self):
        await self.client.aclose()


def require(run, name, passed, **evidence):
    run.check(name, passed, **evidence)
    if not passed:
        raise WorkflowAbort(name)


def check_vectors(library, doc):
    parts = [doc, *library.get_chunks(doc.id), *library.get_sections(doc.id)]
    for part in parts:
        validate_vector(part.embedding)
        if (part.embedding.shape != (3072,) or part.embedding.dtype != np.float32
                or not np.isclose(np.linalg.norm(part.embedding.astype(np.float64)), 1)):
            raise WorkflowAbort('Invalid stored 3072-dimensional normalized vector')
    return len(parts)


def prepare(output, budget, transport_factory, pacing):
    output.mkdir(parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, MAX_ATTEMPTS)
    run = Run(PROFILE, output, budget, pacing=pacing)
    cfg = configuration(PROFILE, output)
    (output / 'sources' / 'average.py').write_text(SOURCE)
    hashes = {}
    for name in SOURCE_FILES:
        content = (ROOT / name).read_bytes()
        target = output / 'provenance' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    protocol = {
        'scope': 'Hosted application orchestration; factual answer quality requires separate review',
        'library_provider': 'deepinfra', 'chat_model': PROFILE.model,
        'embedding_model': PROFILE.embedding_model, 'stored_dimensions': 3072,
        'native_embedding_dimensions': 4096, 'embedding_recipe': 'Production DeepInfra framing and prefix/L2 reduction',
        'hosted_artifact_identity': 'Provider deployment; immutable precision and weights are not pinned',
        'experimental_chat_adapter': 'Production local request builder with selected thinking options; '
                                     'repeat_penalty maps to repetition_penalty, reasoning_format omitted',
        'chat_options': resolve_models(mac_accuracy_profile()).chat.request_options,
        'identity_limit': 'Hosted library/config identity; no claim of exact native wire or runtime parity',
        'source_hashes': hashes, 'fixture_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(),
        'fixture_origin': 'Previously used synthetic mean fixture, not a held-out quality benchmark',
        'document_types': list(DOC_TYPES), 'question': QUESTION,
        'budget_usd': budget.limit, 'maximum_http_attempts': budget.max_requests,
        'timeout_seconds': TIMEOUT, 'context_window': CONTEXT_WINDOW,
        'timeout_reporting': 'The shared hosted completion trace can retain its historical 60s '
                             'default without passing it to this experimental provider. '
                             'report.adapter_calls records the actual adapter max_tokens and '
                             'effective HTTP timeout (600s by default, or the explicit smaller value) '
                             'before dispatch, then identifies the HTTP attempts actually made.',
        'max_total_output_tokens': 8192, 'concurrency': 1,
        'context_admission': 'UTF-8 byte envelope plus 1024 overhead; not native tokenizer verification',
        'retry_policy': 'Only RecordedTransport HTTP429 retries, at most two per call; terminal errors '
                        'abort before optional production catches or embedding retry loops',
        'response_archive': 'Transformed final-only chat response, original body hash/size and numeric usage; '
                            'private reasoning omitted',
        'quality_verdict': 'unreviewed', 'local_q4_qualified': False, 'hardware_qualified': False,
        'limitations': ['One Python source; no SCIP selectors or dependency graph',
                       'No browser, native server, durable local spool, M5 memory or timing qualification',
                       'Normal completion and structural checks do not establish factual quality'],
    }
    save_json(output / 'protocol.json', protocol)
    protocol_hash = hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest()

    def verify():
        if hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest() != protocol_hash:
            raise ValueError('Frozen protocol changed')
        if hashlib.sha256((output / 'sources' / 'average.py').read_bytes()).hexdigest() != protocol['fixture_sha256']:
            raise ValueError('Frozen source fixture changed')
        for name, digest in hashes.items():
            if hashlib.sha256((output / 'provenance' / name).read_bytes()).hexdigest() != digest:
                raise ValueError('Frozen provenance changed')

    run.report.update(
        scope=protocol['scope'], quality_verdict='unreviewed', protocol_sha256=protocol_hash,
        rates_verified_on='2026-10-22', local_q4_qualified=False, hardware_qualified=False,
        transport_mode='live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated',
    )
    run.save()
    return run, cfg, verify


async def workflow(run, cfg, api_key, transport_factory, verify):
    context = InferenceContext.from_config(cfg)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    saved = {}
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            context._chat = HostedThinkingProvider(run, api_key, transport_factory, verify)
            context.embeddings._client = httpx.AsyncClient(
                base_url=PROFILE.embedding_url, headers={'Authorization': f'Bearer {api_key}'},
                timeout=TIMEOUT, trust_env=False, follow_redirects=False,
                transport=BoundedTransport(RecordedTransport(run, 'embeddings', transport_factory()), verify),
            )
            identity = context.binding.manifest
            require(run, 'hosted_identity', not context.local
                    and context.settings.chat.provider == context.settings.embeddings.provider == 'deepinfra',
                    identity=identity)
            source = run.output / 'sources' / 'average.py'
            writer = LibraryWriter(library)
            synced = await sync_file_catalog(library, writer, 'synthetic', source.parent, source)
            elements = [doc for doc in library.list_documents('catalog') if doc.metadata.get('kind') == 'element']
            require(run, 'catalog_sync', synced.added == len(elements) == 1)
            for doc in elements:
                check_vectors(library, doc)
            described = await describe_source_elements(library, writer, 'synthetic', concurrency=1)
            require(run, 'catalog_describe', described['described'] == 1 and described['failed'] == 0,
                    result=described)
            catalog = library.get_document(elements[0].id)
            save_json(run.output / 'catalog-description.json', {
                'id': catalog.id, 'metadata': catalog.metadata, 'content': catalog.content,
            })
            before, calls = snapshot(library, catalog.id), len(run.report['requests'])
            repeated_sync = await sync_file_catalog(library, writer, 'synthetic', source.parent, source)
            repeated_describe = await describe_source_elements(library, writer, 'synthetic', concurrency=1)
            require(run, 'catalog_idempotence', repeated_sync.unchanged == 1
                    and repeated_describe['described'] == 0 and snapshot(library, catalog.id) == before
                    and len(run.report['requests']) == calls)
            generator_config = GeneratorConfig(model=PROFILE.model, provider='deepinfra', inference_context=context)
            async with DocGenerator(generator_config) as generator:
                for kind in DOC_TYPES:
                    generated = await generator.generate_for_file(source, (kind,))
                    require(run, f'generation/{kind}', len(generated) == 1)
                    item = generated[0]
                    (run.output / 'documents' / f'average-{kind}.md').write_text(item.content)
                    validation = ContentValidator().validate(item.content, item.doc_type, item.title)
                    require(run, f'structure/{kind}', validation.is_valid,
                            issues=[{'level': issue.level, 'code': issue.code} for issue in validation.issues])
                    doc = await writer.add_document(item.doc_type, item.title, item.content,
                                                    list(item.source_files), metadata=item.metadata,
                                                    doc_id=f'average--{kind}', source_name='synthetic')
                    parts = check_vectors(library, doc)
                    require(run, f'storage/{kind}', doc.content == item.content
                            and item.source_snapshot == hashlib.sha256(source.read_bytes()).hexdigest(),
                            vector_parts=parts, stored_dimensions=3072)
                    run.report['documents'].append({'id': doc.id, 'doc_type': kind,
                                                    'source_snapshot': item.source_snapshot})
                    run.save()
            doc = library.get_document('average--explanation')
            before, calls = snapshot(library, doc.id), len(run.report['requests'])
            await writer.add_document(doc.content_type, doc.title, doc.content, doc.source_files,
                                      metadata=doc.metadata, doc_id=doc.id, source_name=doc.source_name)
            require(run, 'document_idempotence', snapshot(library, doc.id) == before
                    and len(run.report['requests']) == calls)
            search = await service.search(query=QUESTION, source='synthetic', sections_only=True, limit=3)
            save_json(run.output / 'search.json', search.model_dump(mode='json'))
            require(run, 'search_returns_persisted_evidence', bool(search.documents) and all(
                library.get_document(row.id) is not None and (row.sections or row.content)
                for row in search.documents), ids=[row.id for row in search.documents])
            with capture_completion_trace() as trace:
                answer = await service.ask(QUESTION, source='synthetic')
            save_json(run.output / 'ask-trace.json', trace)
            save_json(run.output / 'answer.json', answer.model_dump(mode='json'))
            require(run, 'ask_returns_answer_and_sources', bool(answer.answer.strip() and answer.sources),
                    sources=answer.sources, quality_verdict='unreviewed')
            prepared = await writer.prepare_document(evolve(
                doc, content=doc.content + '\n\n## Rollback probe\nThis update must not persist.\n'))
            before = snapshot(library, doc.id)
            with library._conn_provider.acquire() as connection:
                connection.execute("CREATE TRIGGER smoke_fail BEFORE INSERT ON sections BEGIN SELECT RAISE(ABORT, 'smoke rollback'); END")
            failed = False
            try:
                library.apply_prepared_document(prepared)
            except sqlite3.IntegrityError:
                failed = True
            finally:
                with library._conn_provider.acquire() as connection:
                    connection.execute('DROP TRIGGER smoke_fail')
            require(run, 'atomic_rollback', failed and snapshot(library, doc.id) == before)
            saved = {row.id: snapshot(library, row.id) for row in library.list_documents()
                     if row.embedding is not None}
    finally:
        await service.close()
        await context.aclose()
    reopened = InferenceContext.from_config(cfg)
    try:
        with reopened.open_library(operation='inspect') as library:
            require(run, 'reopen_preserves_content_and_vectors', bool(saved)
                    and reopened.binding.manifest == identity
                    and all(snapshot(library, doc_id) == value for doc_id, value in saved.items()),
                    documents_checked=len(saved))
    finally:
        await reopened.aclose()


async def execute(output, budget, *, api_key, transport_factory=httpx.AsyncHTTPTransport, pacing=15):
    if budget.limit > 1 or not api_key:
        raise ValueError('At most $1 and a provider credential are required')
    run, cfg, verify = prepare(output, budget, transport_factory, pacing)
    old_config, old_key = config_module._global_config, os.environ.get(PROFILE.key_env)
    config_module._global_config = cfg
    os.environ[PROFILE.key_env] = api_key
    started = time.monotonic()
    try:
        await workflow(run, cfg, api_key, transport_factory, verify)
        run.report['status'] = 'passed_orchestration_pending_quality_review'
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        raise
    except (WorkflowAbort, Exception) as error:  # noqa: BLE001 - preserve every diagnostic failure without retry.
        run.report.update(status='failed', error_type=type(error).__name__,
                          error=str(error).replace(api_key, '<redacted>'))
    finally:
        config_module._global_config = old_config
        if old_key is None:
            os.environ.pop(PROFILE.key_env, None)
        else:
            os.environ[PROFILE.key_env] = old_key
        run.report['seconds'] = time.monotonic() - started
        run.save()
    return run.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, default=1)
    args = parser.parse_args()
    key = os.environ.get(PROFILE.key_env) or dotenv_values(ROOT / '.env').get(PROFILE.key_env)
    if not key:
        parser.error('DEEPINFRA_API_KEY is missing')
    report = asyncio.run(execute(args.output.resolve(), Budget(args.budget_usd, MAX_ATTEMPTS), api_key=key))
    print(json.dumps({name: report[name] for name in ('status', 'seconds', 'accounted_upper_bound_usd')}))
    return 0 if report['status'] == 'passed_orchestration_pending_quality_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
