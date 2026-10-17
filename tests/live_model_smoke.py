"""Opt-in hosted workflow verification; run with --profile qwen, claude or openai.

All profiles execute the same application scenarios. Only HTTP adapters differ.
Ordinary pytest never makes paid requests. See docs/live-model-verification.md.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
import traceback
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx
from attrs import evolve
from dotenv import dotenv_values

from ariadne_mcp.service import AriadneService
from config import Config
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.llm.local import parse_chat_response
from docgen.validator import ContentValidator
from inference import InferenceContext
from library.prepared import validate_vector
from model_config import CHAT_MODEL, EMBEDDING_MODEL
from tests.model_smoke_cases import CASES
from writer import LibraryWriter

ROOT = Path(__file__).resolve().parents[1]
DOC_TYPES = ('explanation', 'architecture', 'qa', 'gotcha', 'diagram')


@dataclass(frozen=True)
class Profile:
    name: str
    provider: str
    model: str
    base_url: str
    key_env: str
    input_rate: float
    output_rate: float
    # Reserve the highest input rate, including possible prompt cache writes.
    reserve_input_rate: float

    @property
    def embedding_url(self) -> str:
        return self.base_url if self.name == 'qwen' else 'https://api.openai.com/v1'

    @property
    def embedding_model(self) -> str:
        return EMBEDDING_MODEL if self.name == 'qwen' else 'text-embedding-3-large'

    @property
    def embedding_key(self) -> str:
        return self.key_env if self.name == 'qwen' else 'OPENAI_API_KEY'

    @property
    def embedding_rate(self) -> float:
        return 0.01 if self.name == 'qwen' else 0.13


# Standard, short-context rates verified 2026-10-11. Sources are in the guide.
# Qwen uses undiscounted rates for reservations; returned provider cost is saved.
PROFILES = {
    'qwen': Profile(
        'qwen',
        'deepinfra',
        CHAT_MODEL,
        'https://api.deepinfra.com/v1/openai',
        'DEEPINFRA_API_KEY',
        0.20,
        2.50,
        0.20,
    ),
    'claude': Profile(
        'claude',
        'anthropic',
        'claude-opus-5-5',
        'https://api.anthropic.com/v1',
        'ANTHROPIC_API_KEY',
        4.0,
        20.0,
        5.0,
    ),
    'openai': Profile(
        'openai',
        'openai',
        'gpt-6-astra',
        'https://api.openai.com/v1',
        'OPENAI_API_KEY',
        10.0,
        50.0,
        12.50,
    ),
}


def save_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


class Budget:
    """Shared across selected profiles; unknown/failed usage retains its reservation."""

    def __init__(self, limit: float, max_requests: int = 120) -> None:
        if not math.isfinite(limit) or limit <= 0:
            raise ValueError('Budget must be a positive finite dollar amount')
        self.limit = limit
        self.committed = 0.0
        self.requests = 0
        self.max_requests = max_requests

    def reserve(self, cost: float) -> None:
        if self.requests >= self.max_requests or self.committed + cost > self.limit:
            raise RuntimeError('Live request or spending limit reached; no request sent')
        self.committed += cost
        self.requests += 1


class Run:
    def __init__(
        self, profile: Profile, output: Path, budget: Budget, *, pacing: float = 15
    ) -> None:
        self.profile, self.output, self.budget = profile, output, budget
        self.pacing = pacing if profile.name == 'qwen' else 0
        self.last_embedding = 0.0
        self.report = {
            'profile': profile.name,
            'chat_model': profile.model,
            'embedding_model': profile.embedding_model,
            'rates_verified_on': '2026-10-11',
            'rates_per_million_tokens': {
                'input': profile.input_rate,
                'input_reservation': profile.reserve_input_rate,
                'output': profile.output_rate,
                'embeddings': profile.embedding_rate,
            },
            'started_at': datetime.now(UTC).isoformat(),
            'status': 'running',
            'scope': 'Hosted application workflows with synthetic source',
            'not_qualified': [
                'local vLLM transport and /tokenize',
                'local durable spool',
                'GPU deployment and pinned artifacts',
                'full factual correctness',
            ],
            'requests': [],
            'checks': [],
            'documents': [],
        }
        for directory in ('sources', 'documents', 'requests', 'responses'):
            (output / directory).mkdir(parents=True, exist_ok=True)

    def save(self) -> None:
        self.report['accounted_upper_bound_usd'] = sum(
            r.get('accounted_usd', r['reserved_usd']) for r in self.report['requests']
        )
        save_json(self.output / 'report.json', self.report)

    def check(self, name: str, passed: bool, **evidence: object) -> None:
        self.report['checks'].append({'name': name, 'passed': bool(passed), **evidence})
        self.save()


class RecordedTransport(httpx.AsyncBaseTransport):
    """Observe production clients, bound spend, and adapt hosted Qwen options only."""

    def __init__(self, run: Run, kind: str, inner: httpx.AsyncBaseTransport) -> None:
        self.run, self.kind, self.inner = run, kind, inner

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        run, profile = self.run, self.run.profile
        embedding = self.kind == 'embeddings'
        url = (
            profile.embedding_url + '/embeddings'
            if embedding
            else (
                profile.base_url
                + ('/messages' if profile.provider == 'anthropic' else '/chat/completions')
            )
        )
        if request.method != 'POST' or str(request.url) != url:
            raise ValueError(f'Unexpected live route: {request.method} {request.url}')
        payload = json.loads(request.content)
        model = profile.embedding_model if embedding else profile.model
        if payload.get('model') != model:
            raise ValueError('Unexpected live model')
        raw = json.dumps(payload).encode()
        cap = 0 if embedding else payload.get('max_completion_tokens', payload.get('max_tokens'))
        if (
            not isinstance(cap, int)
            or (not embedding and not 0 < cap <= 8192)
            or len(raw) > 150_000
        ):
            raise ValueError('Live input/output request limit exceeded')
        input_rate = profile.embedding_rate if embedding else profile.reserve_input_rate
        overhead = 1024 * (len(payload['input']) if embedding else 1)
        reserve = ((len(raw) + overhead) * input_rate + cap * profile.output_rate) / 1_000_000
        headers = dict(request.headers)
        headers.pop('content-length', None)
        wire = httpx.Request(
            'POST', url, headers=headers, json=payload, extensions=request.extensions
        )
        secrets = [v for k, v in headers.items() if k in ('authorization', 'x-api-key')]
        secrets += [s.removeprefix('Bearer ') for s in secrets]

        for retry in range(3):
            run.budget.reserve(reserve)
            number = len(run.report['requests']) + 1
            record = {
                'kind': self.kind,
                'model': model,
                'reserved_usd': reserve,
                'attempt': retry + 1,
            }
            run.report['requests'].append(record)
            save_json(run.output / 'requests' / f'{number:03}.json', payload)
            run.save()
            if embedding:
                await asyncio.sleep(max(0, run.pacing - (time.monotonic() - run.last_embedding)))
                run.last_embedding = time.monotonic()
            print(f'{profile.name}: request {number} {self.kind}', flush=True)
            start = time.monotonic()
            response = await self.inner.handle_async_request(wire)
            await response.aread()
            record.update(status=response.status_code, seconds=round(time.monotonic() - start, 3))
            body = response.text
            for secret in secrets:
                if secret:
                    body = body.replace(secret, '<redacted>')
            (run.output / 'responses' / f'{number:03}.json').write_text(body)
            run.save()
            if response.status_code == 429 and profile.name == 'qwen' and retry < 2:
                try:
                    delay = max(run.pacing, float(response.headers.get('retry-after', '15')))
                except ValueError:
                    delay = 15
                if math.isfinite(delay) and 0 <= delay <= 45:
                    await response.aclose()
                    await asyncio.sleep(delay)
                    continue
            if response.status_code == 200:
                data = response.json()
                usage = data.get('usage') or {}
                record['usage'] = usage
                in_key, out_key = (
                    ('input_tokens', 'output_tokens')
                    if profile.provider == 'anthropic' and not embedding
                    else ('prompt_tokens', 'completion_tokens')
                )
                input_tokens = usage.get(in_key)
                output_tokens = 0 if embedding else usage.get(out_key)
                cache_tokens = sum(
                    usage.get(k, 0)
                    for k in ('cache_creation_input_tokens', 'cache_read_input_tokens')
                )
                if all(
                    isinstance(n, int) and n >= 0
                    for n in (input_tokens, output_tokens, cache_tokens)
                ):
                    accounted = (
                        (input_tokens + cache_tokens) * input_rate
                        + output_tokens * profile.output_rate
                    ) / 1_000_000
                    # Never free a reservation unless complete usage is available.
                    run.budget.committed += accounted - reserve
                    record['accounted_usd'] = accounted
                if not embedding:
                    if profile.provider == 'anthropic':
                        if data.get('stop_reason') != 'end_turn':
                            raise ValueError('Incomplete Claude response')
                    else:
                        parse_chat_response(data)
                        message = data['choices'][0]['message']
                        if profile.name == 'qwen' and (
                            message.get('reasoning_content') or message.get('reasoning')
                        ):
                            raise ValueError('Qwen returned reasoning despite non-thinking request')
            run.save()
            return response
        raise AssertionError('Unreachable retry state')


def configuration(profile: Profile, output: Path) -> Config:
    data = {
        'default_source': 'synthetic',
        'sources': {'synthetic': str(output / 'sources')},
        'themes_enabled': False,
        'defaults': {
            'db_path': str(output / 'library.db'),
            'staleness_db_path': str(output / 'staleness.db'),
        },
        'llm': {
            'provider': profile.provider,
            'model': profile.model,
            'base_url': profile.base_url,
            'api_key_env': profile.key_env,
        },
        'embeddings': {
            'provider': 'deepinfra' if profile.name == 'qwen' else 'openai',
            'model': profile.embedding_model,
            'base_url': profile.embedding_url,
            'api_key_env': profile.embedding_key,
            'dimensions': 3072,
        },
        'execution': {'batch_mode': 'never', 'max_attempts': 3},
    }
    path = output / 'config.json'
    save_json(path, data)
    return Config(path)


async def attach_clients(context: InferenceContext, run: Run, transport_factory) -> None:
    # Instantiate through the real factories; retain native auth and wire protocols.
    chat = context.chat
    previous = chat._get_client()
    headers = dict(previous.headers)
    await previous.aclose()
    chat._client = httpx.AsyncClient(
        base_url=run.profile.base_url,
        headers=headers,
        timeout=180,
        trust_env=False,
        follow_redirects=False,
        transport=RecordedTransport(run, 'chat', transport_factory()),
    )
    embeddings = context.embeddings
    previous = await embeddings._get_client()
    headers = dict(previous.headers)
    await previous.aclose()
    embeddings._client = httpx.AsyncClient(
        base_url=run.profile.embedding_url,
        headers=headers,
        timeout=120,
        trust_env=False,
        follow_redirects=False,
        transport=RecordedTransport(run, 'embeddings', transport_factory()),
    )


def snapshot(library, doc_id: str) -> tuple:
    doc = library.get_document(doc_id)
    return (
        doc.content,
        doc.updated_at,
        doc.embedding.tobytes(),
        [(c.content, c.embedding.tobytes()) for c in library.get_chunks(doc_id)],
        [(s.content, s.embedding.tobytes()) for s in library.get_sections(doc_id)],
    )


async def workflows(run: Run, transport_factory=httpx.AsyncHTTPTransport) -> None:
    output, profile = run.output, run.profile
    cfg = configuration(profile, output)
    for case_id, _, source, _ in CASES:
        (output / 'sources' / f'{case_id}.py').write_text(source)
    context = InferenceContext.from_config(cfg)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    try:
        with context.write_session():
            library = context.open_library(operation='write')
            await attach_clients(context, run, transport_factory)
            writer = LibraryWriter(library)
            generator_config = GeneratorConfig(
                model=profile.model, provider=profile.provider, inference_context=context
            )
            async with DocGenerator(generator_config) as generator:
                for case_id, _, _, _ in CASES:
                    # Every profile sees the same seven tasks, covering all five types.
                    types = DOC_TYPES if case_id == 'cache' else ('explanation',)
                    for doc_type in types:
                        generated = await generator.generate_for_file(
                            output / 'sources' / f'{case_id}.py', (doc_type,)
                        )
                        if len(generated) != 1:
                            raise ValueError(
                                f'Generation did not produce {case_id}/{doc_type}; inspect saved responses'
                            )
                        item = generated[0]
                        doc_id = f'{case_id}--{doc_type}'
                        (output / 'documents' / f'{doc_id}.md').write_text(item.content)
                        validation = ContentValidator().validate(
                            item.content, item.doc_type, item.title
                        )
                        run.check(
                            f'validation/{doc_id}',
                            validation.is_valid,
                            issues=[
                                {'code': issue.code, 'level': issue.level, 'message': issue.message}
                                for issue in validation.issues
                            ],
                        )
                        doc = await writer.add_document(
                            item.doc_type,
                            item.title,
                            item.content,
                            list(item.source_files),
                            metadata=item.metadata,
                            doc_id=doc_id,
                            source_name='synthetic',
                        )
                        chunks, sections = library.get_chunks(doc_id), library.get_sections(doc_id)
                        for part in (doc, *chunks, *sections):
                            if part.embedding is None:
                                raise ValueError('Missing persisted embedding')
                            validate_vector(part.embedding)
                        run.check(
                            f'storage/{doc_id}',
                            doc.content == item.content,
                            chunks=len(chunks),
                            sections=len(sections),
                            content_matches=doc.content == item.content,
                        )
                        run.report['documents'].append(
                            {
                                'id': doc_id,
                                'title': doc.title,
                                'source_snapshot': item.source_snapshot,
                            }
                        )
                        run.save()

                doc = library.get_document('average--explanation')
                before, requests = snapshot(library, doc.id), len(run.report['requests'])
                await writer.add_document(
                    doc.content_type,
                    doc.title,
                    doc.content,
                    doc.source_files,
                    metadata=doc.metadata,
                    doc_id=doc.id,
                    source_name=doc.source_name,
                )
                run.check(
                    'idempotent_replay',
                    snapshot(library, doc.id) == before and len(run.report['requests']) == requests,
                )

                for case_id, _, _, question in CASES:
                    vector = await context.embeddings.embed_query(question)
                    documents = library.search(vector, k=3)
                    chunks = library.search_chunks(vector, k=3)
                    ranks = [r.document.id for r in documents]
                    chunk_ranks = [r.document.id for r in chunks]
                    run.check(
                        f'retrieval/{case_id}',
                        bool(ranks and chunk_ranks)
                        and ranks[0].startswith(case_id + '--')
                        and chunk_ranks[0].startswith(case_id + '--'),
                        query=question,
                        document_ranking=ranks,
                        chunk_ranking=chunk_ranks,
                    )
                    selected = await service.search(
                        query=question, source='synthetic', sections_only=True, limit=3
                    )
                    save_json(output / f'sections-{case_id}.json', selected.model_dump(mode='json'))
                    top = selected.documents[0] if selected.documents else None
                    stored_sections = library.get_sections(top.id) if top else []
                    selection_valid = bool(top) and (
                        bool(top.sections)
                        if stored_sections
                        else top.content == library.get_document(top.id).content
                    )
                    run.check(
                        f'mcp_sections/{case_id}',
                        selection_valid and top.id.startswith(case_id + '--'),
                        selected_id=top.id if top else None,
                        stored_sections=len(stored_sections),
                        returned_sections=len(top.sections or []) if top else 0,
                        full_document_fallback=bool(top) and not stored_sections,
                    )

                question = 'What exception does cache_put raise when capacity is negative?'
                answer = await service.ask(question, source='synthetic')
                save_json(output / 'answer.json', answer.model_dump(mode='json'))
                (output / 'answer.md').write_text(answer.answer)
                run.check(
                    'mcp_ask_grounded_fact',
                    'keyerror' in answer.answer.lower()
                    and any('cache' in title.lower() for title in answer.sources),
                    question=question,
                    expected_exception='KeyError',
                    sources=answer.sources,
                )

                # A source edit traverses the same generator and replaces the stored bundle.
                path = output / 'sources' / 'average.py'
                updated_source = path.read_text().replace(
                    'raise ValueError("empty input")', 'return 0.0'
                )
                (output / 'average-before.py').write_text(path.read_text())
                path.write_text(updated_source)
                updated = await generator.generate_for_file(path, ('explanation',))
                if len(updated) != 1:
                    raise ValueError('Source edit failed to generate a replacement')
                item = updated[0]
                (output / 'documents' / 'average-updated.md').write_text(item.content)
                replacement = await writer.add_document(
                    item.doc_type,
                    item.title,
                    item.content,
                    list(item.source_files),
                    metadata=item.metadata,
                    doc_id=doc.id,
                    source_name='synthetic',
                )
                run.check(
                    'source_edit_replacement',
                    replacement.content != doc.content
                    and replacement.embedding.tobytes() != doc.embedding.tobytes()
                    and item.source_snapshot == hashlib.sha256(path.read_bytes()).hexdigest(),
                )

                prepared = await writer.prepare_document(
                    evolve(
                        replacement,
                        content=replacement.content
                        + '\n\n## Rollback probe\nThis update must not persist.',
                    )
                )
                before = snapshot(library, replacement.id)
                with library._conn_provider.acquire() as conn:
                    conn.execute(
                        "CREATE TRIGGER smoke_fail BEFORE INSERT ON sections BEGIN SELECT RAISE(ABORT, 'smoke rollback'); END"
                    )
                failed = False
                try:
                    library.apply_prepared_document(prepared)
                except sqlite3.IntegrityError:
                    failed = True
                finally:
                    with library._conn_provider.acquire() as conn:
                        conn.execute('DROP TRIGGER smoke_fail')
                run.check('atomic_rollback', failed and snapshot(library, replacement.id) == before)
    finally:
        await service.close()
        await context.aclose()


async def clew_workflows(run: Run, transport_factory=httpx.AsyncHTTPTransport) -> None:
    """Build and embed actual routes, then assert their use in MCP answer evidence."""
    import config
    from library.clew_embeddings import embed_clews
    from library.clews import nearest_clew_matches
    from llm import capture_completion_trace
    from tests.clew_workflow import (
        QUESTION,
        ROUTE,
        add_guide,
        build_routes,
        script,
        seed_graph,
        seed_theme,
    )

    cfg = configuration(run.profile, run.output)
    context = InferenceContext.from_config(cfg)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    previous_config = config._global_config
    config._global_config = cfg
    try:
        with context.write_session():
            library = context.open_library(operation='write')
            await attach_clients(context, run, transport_factory)
            seed_graph(library, run.output / 'sources', 'synthetic')
            build_routes(library, 'synthetic')
            count = await embed_clews(library, source_name='synthetic')
            run.check('clew_embeddings_stored', count == 2, count=count)
            before = len(run.report['requests'])
            repeated = await embed_clews(library, source_name='synthetic')
            run.check('clew_embedding_resume', repeated == 0 and len(run.report['requests']) == before)
            seed_theme(library, 'synthetic')
            await script('question_clews')['question_routes'](library, 'synthetic', True, None)
            with library._conn_provider.acquire() as conn:
                question, stored = conn.execute(
                    'SELECT question, embedding FROM clews WHERE entry_symbol=?', (ROUTE[0],),
                ).fetchone()
            refreshed = await embed_clews(library, source_name='synthetic')
            run.check('clew_question_and_reembedding', bool(question) and stored is None
                      and refreshed == 1, question=question, refreshed=refreshed)
            await add_guide(library, 'synthetic')
        vector = await context.embeddings.embed_query(QUESTION)
        with library._conn_provider.acquire() as conn:
            matches = nearest_clew_matches(conn, vector, source_name='synthetic',
                                           top_k=2, min_similarity=-1.0)
        run.check('clew_semantic_retrieval', bool(matches) and matches[0].clew.route == ROUTE,
                  routes=[match.clew.route for match in matches],
                  scores=[match.similarity for match in matches])
        with capture_completion_trace() as trace:
            answer = await service.ask(QUESTION, source='synthetic')
        save_json(run.output / 'clew-answer.json', answer.model_dump(mode='json'))
        save_json(run.output / 'clew-completions.json', trace)
        selected = answer.graph_diagnostics.get('clew_selection', {})
        run.check('clew_selected_for_walk', selected.get('selected', 0) >= 1
                  and not selected.get('error'), diagnostics=selected)
        run.check('clew_route_in_answer_evidence',
                  any(set(ROUTE).issubset(route) for route in answer.route_candidates.values()),
                  route_candidates=answer.route_candidates)
        prompts = '\n'.join(json.dumps(row['messages']) for row in trace)
        run.check('clew_source_reaches_model', all(symbol in prompts for symbol in ROUTE)
                  and "return {'receipt': amount}" in prompts,
                  phases=[row['phase'] for row in trace])
        run.check('clew_answer_completed', bool(answer.answer.strip()) and bool(trace)
                  and all(row['status'] == 'ok' for row in trace)
                  and 'synthesis unavailable' not in answer.answer,
                  models=sorted({row['model'] for row in trace}))
    finally:
        await service.close()
        await context.aclose()
        config._global_config = previous_config


async def execute(
    profile: Profile,
    output: Path,
    budget: Budget,
    *,
    transport_factory=httpx.AsyncHTTPTransport,
    pacing: float = 15,
    scenario: str = 'workflows',
) -> dict:
    run = Run(profile, output, budget, pacing=pacing)
    run.report['transport_mode'] = (
        'live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated'
    )
    run.save()
    try:
        run.report['scenario'] = scenario
        workflow = {'workflows': workflows, 'clews': clew_workflows}[scenario]
        await workflow(run, transport_factory)
        run.report['status'] = (
            'passed' if all(c['passed'] for c in run.report['checks']) else 'failed'
        )
    except Exception as exc:  # noqa: BLE001 - keep evidence for failed live workflows.
        error = f'{type(exc).__name__}: {exc}'
        trace = traceback.format_exc()
        for name in (profile.key_env, profile.embedding_key):
            key = os.environ.get(name)
            if key:
                error = error.replace(key, '<redacted>')
                trace = trace.replace(key, '<redacted>')
        (output / 'error.txt').write_text(trace)
        run.report.update(status='failed', error=error[:1000])
    finally:
        run.report['finished_at'] = datetime.now(UTC).isoformat()
        run.save()
    return run.report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=(*PROFILES, 'all'), default='qwen')
    parser.add_argument('--scenario', choices=('workflows', 'clews'), default='workflows')
    parser.add_argument(
        '--budget-usd', type=float, default=1.0, help='Total cap across all selected profiles'
    )
    parser.add_argument('--output', type=Path, help='New evidence directory (must not exist)')
    args = parser.parse_args(argv)
    budget = Budget(args.budget_usd)
    profiles = list(PROFILES.values()) if args.profile == 'all' else [PROFILES[args.profile]]
    saved = dotenv_values(ROOT / '.env')
    required = {key for profile in profiles for key in (profile.key_env, profile.embedding_key)}
    for key in required:
        value = os.environ.get(key) or saved.get(key)
        if not value:
            parser.error(f'{key} is missing; no requests made')
        os.environ[key] = value
    output = (
        args.output.resolve()
        if args.output
        else Path(tempfile.mkdtemp(prefix='ariadne-model-live-'))
    )
    if args.output:
        output.mkdir(parents=True, exist_ok=False)
    print(f'Evidence directory: {output}', flush=True)

    async def run_all():
        reports = []
        for profile in profiles:
            report = await execute(profile, output / profile.name, budget, scenario=args.scenario)
            reports.append(report)
            save_json(
                output / 'summary.json',
                {
                    'budget_usd': budget.limit,
                    'accounted_upper_bound_usd': budget.committed,
                    'profiles': reports,
                },
            )
            print(
                f'{profile.name}: {report["status"]}; accounted ${report["accounted_upper_bound_usd"]:.6f}',
                flush=True,
            )
        return reports

    reports = asyncio.run(run_all())
    return 0 if all(r['status'] == 'passed' for r in reports) else 1


if __name__ == '__main__':
    raise SystemExit(main())
