"""Paid, frozen-input Qwen implementation checks; no factual-quality gate.

python -m tests.live_qwen_blind --output NEW_DIRECTORY --budget-usd 1
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sqlite3
import time
import traceback
from itertools import pairwise
from pathlib import Path

import httpx
import numpy as np
from attrs import evolve
from dotenv import dotenv_values

from ariadne_mcp.service import AriadneService
from config import Config
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.validator import ContentValidator
from inference import InferenceContext
from library.model_binding import ModelBindingMismatch
from tests.blind_model_cases import CASES
from tests.live_model_smoke import (
    PROFILES,
    ROOT,
    Budget,
    Run,
    attach_clients,
    configuration,
    save_json,
    snapshot,
)
from writer import LibraryWriter


def rank_matches(query, results, expected):
    """Independent numeric oracle; no expected semantic answer or model judgment."""
    scores = {
        key: float(np.dot(query.astype(np.float64), vector) / np.linalg.norm(query))
        for key, vector in expected.items()
    }
    actual = [(r.chunk.id if r.chunk else r.document.id, r.score) for r in results]
    return (
        len(actual) == len(scores)
        and {key for key, _ in actual} == set(scores)
        and all(abs(score - scores[key]) < 1e-5 for key, score in actual)
        and all(left[1] + 1e-5 >= right[1] for left, right in pairwise(actual))
    )


def check_vectors(run, library, *, native_dimensions=4096):
    """Match persisted vectors to raw provider responses for the exact input text."""
    expected = {}
    for path in sorted((run.output / 'requests').glob('*.json')):
        request = json.loads(path.read_text())
        if 'input' not in request:
            continue
        response = json.loads((run.output / 'responses' / path.name).read_text())
        for item in response.get('data', []):
            raw = np.asarray(item['embedding'], dtype=np.float64)
            if raw.shape != (native_dimensions,):
                raise ValueError('Provider vector has unexpected dimensions')
            prefix = raw[:3072]
            reduced = (prefix / np.sqrt(np.dot(prefix, prefix))).astype(np.float32)
            expected.setdefault(request['input'][item['index']], []).append(reduced)
    count, matches = 0, True
    for doc in library.list_documents():
        parts = [(f'{doc.title}\n\n{doc.content[:2000]}', doc.embedding)]
        parts += [(c.content, c.embedding) for c in library.get_chunks(doc.id)]
        parts += [(s.heading or 'Introduction', s.embedding) for s in library.get_sections(doc.id)]
        for text, vector in parts:
            count += 1
            matches &= (
                vector is not None
                and vector.shape == (3072,)
                and np.isfinite(vector).all()
                and abs(float(np.linalg.norm(vector)) - 1) < 1e-5
                and any(
                    np.allclose(vector, v, rtol=1e-6, atol=1e-7) for v in expected.get(text, [])
                )
            )
    run.check('vectors_match_exact_inputs_and_native_reduction', bool(matches), vectors=count)


async def workflows(run, transport_factory):
    output = run.output
    configuration(run.profile, output)
    data = json.loads((output / 'config.json').read_text())
    data['sources'] = {}
    for name, _, source, _ in CASES:
        directory = output / 'sources' / name
        directory.mkdir()
        (directory / 'module.py').write_text(source)
        data['sources'][name] = str(directory)
    data['default_source'] = CASES[0][0]
    save_json(output / 'config.json', data)
    cfg = Config(output / 'config.json')
    context = InferenceContext.from_config(cfg)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    saved = {}
    try:
        await attach_clients(context, run, transport_factory)
        with context.write_session():
            library = context.open_library(operation='write')
            writer = LibraryWriter(library)
            async with DocGenerator(
                GeneratorConfig(
                    provider=run.profile.provider, model=run.profile.model, inference_context=context
                )
            ) as generator:
                for name, doc_type, source, _ in CASES:
                    path = output / 'sources' / name / 'module.py'
                    items = await generator.generate_for_file(path, (doc_type,))
                    if len(items) != 1:
                        raise ValueError(
                            f'No usable model response for {name}; inspect raw evidence'
                        )
                    item = items[0]
                    (output / 'documents' / f'{name}.md').write_text(item.content)
                    validation = ContentValidator().validate(item.content, doc_type, item.title)
                    run.report.setdefault('model_observations', []).append(
                        {
                            'case': name,
                            'structurally_valid': validation.is_valid,
                            'issues': [
                                {'code': i.code, 'level': i.level} for i in validation.issues
                            ],
                        }
                    )
                    doc = await writer.add_document(
                        doc_type,
                        item.title,
                        item.content,
                        list(item.source_files),
                        metadata=item.metadata,
                        doc_id=name,
                        source_name=name,
                    )
                    run.check(
                        f'roundtrip/{name}',
                        doc.content == item.content
                        and doc.source_name == name
                        and doc.source_files == list(item.source_files)
                        and item.source_snapshot == hashlib.sha256(source.encode()).hexdigest(),
                    )
                    before, requests = snapshot(library, name), len(run.report['requests'])
                    await writer.add_document(
                        doc_type,
                        doc.title,
                        doc.content,
                        doc.source_files,
                        metadata=doc.metadata,
                        doc_id=name,
                        source_name=name,
                    )
                    run.check(
                        f'replay/{name}',
                        snapshot(library, name) == before
                        and len(run.report['requests']) == requests,
                    )

            # Numeric ranking runs before any MCP usage events can weight results.
            for name, _, _, question in CASES:
                vector = await context.embeddings.embed_query(question)
                docs = library.list_documents()
                chunks = [c for d in docs for c in library.get_chunks(d.id)]
                run.check(
                    f'document_ranking/{name}',
                    rank_matches(
                        vector,
                        library.search(vector, k=len(docs)),
                        {d.id: d.embedding for d in docs},
                    ),
                )
                run.check(
                    f'chunk_ranking/{name}',
                    rank_matches(
                        vector,
                        library.search_chunks(vector, k=max(1, len(chunks))),
                        {c.id: c.embedding for c in chunks},
                    ),
                )

            for name, _, _, question in CASES:
                selected = await service.search(
                    query=question,
                    source=name,
                    sections_only=True,
                    limit=6,
                )
                save_json(output / f'selection-{name}.json', selected.model_dump(mode='json'))
                valid = bool(selected.documents) and all(d.id == name for d in selected.documents)
                for doc in selected.documents:
                    sections = library.get_sections(doc.id)
                    valid &= (
                        bool(doc.sections)
                        if sections
                        else doc.content == library.get_document(doc.id).content
                    )
                run.check(f'scoped_mcp_sections/{name}', bool(valid))
                answer = await service.ask(question, source=name)
                save_json(output / f'answer-{name}.json', answer.model_dump(mode='json'))
                (output / f'answer-{name}.md').write_text(answer.answer)
                run.check(
                    f'scoped_mcp_answer/{name}',
                    bool(answer.answer.strip())
                    and bool(answer.sources)
                    and set(answer.sources) == {library.get_document(name).title},
                )

            # Deterministic replacement forces a content change independent of model prose.
            doc = library.get_document(CASES[0][0])
            replacement = await writer.replace_document(
                doc.id,
                content=doc.content + '\n\n## Persistence probe\nUnicode: שלום / café / ✓.\n',
            )
            run.check(
                'replacement',
                replacement.content != doc.content
                and replacement.embedding is not None
                and library.get_document(doc.id).content == replacement.content,
            )
            prepared = await writer.prepare_document(
                evolve(
                    replacement,
                    content=replacement.content + '\n\n## Rejected update\nMust roll back.\n',
                )
            )
            before = snapshot(library, doc.id)
            with library._conn_provider.acquire() as conn:
                conn.execute(
                    "CREATE TRIGGER blind_fail BEFORE INSERT ON sections BEGIN SELECT RAISE(ABORT, 'blind rollback'); END"
                )
            failed = False
            try:
                library.apply_prepared_document(prepared)
            except sqlite3.IntegrityError:
                failed = True
            finally:
                with library._conn_provider.acquire() as conn:
                    conn.execute('DROP TRIGGER blind_fail')
            run.check('rollback', failed and snapshot(library, doc.id) == before)
            check_vectors(run, library)
            saved = {d.id: snapshot(library, d.id) for d in library.list_documents()}
    finally:
        await service.close()
        await context.aclose()

    reopened = InferenceContext.from_config(cfg)
    try:
        library = reopened.open_library()
        run.check(
            'reopen_preserves_content_and_vectors',
            {d.id: snapshot(library, d.id) for d in library.list_documents()} == saved,
        )
    finally:
        await reopened.aclose()
    data['llm'] = {'provider': 'openai', 'model': 'gpt-6-astra'}
    data['embeddings'] = {'provider': 'openai'}
    save_json(output / 'incompatible-config.json', data)
    incompatible = None
    rejected = False
    try:
        incompatible = InferenceContext.from_config(Config(output / 'incompatible-config.json'))
        incompatible.open_library()
    except ModelBindingMismatch:
        rejected = True
    finally:
        if incompatible is not None:
            await incompatible.aclose()
    run.check('incompatible_pair_rejected', rejected)


async def execute(output, budget, *, transport_factory=httpx.AsyncHTTPTransport, pacing=15):
    output.mkdir(parents=True, exist_ok=False)
    run = Run(PROFILES['qwen'], output, budget, pacing=pacing)
    run.report['scope'] = (
        'Fresh frozen cases; implementation invariants only; no model factuality gate'
    )
    run.report['transport_mode'] = (
        'live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated'
    )
    save_json(
        output / 'protocol.json',
        {
            'cases': CASES,
            'cases_sha256': hashlib.sha256(
                (ROOT / 'tests/blind_model_cases.py').read_bytes()
            ).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),  # noqa: ASYNC240
            'budget_usd': budget.limit,
            'blind_definition': 'Inputs and predicates frozen before seeing outputs. Model receives source only for generation; no expected outputs. No prompt tuning or selective reruns.',
            'checks': [
                'roundtrip',
                'replay without requests',
                'numeric document/chunk ranking',
                'source isolation',
                'section or full-document delivery',
                'answer transport and source metadata',
                'replacement',
                'atomic rollback',
                'exact embedding input/vector correspondence',
                'reopen',
                'incompatible model pair rejection',
            ],
        },
    )
    started = time.monotonic()
    try:
        await workflows(run, transport_factory)
        run.report['status'] = (
            'passed' if all(c['passed'] for c in run.report['checks']) else 'failed'
        )
    except Exception as exc:  # noqa: BLE001 - preserve failures without discarding evidence
        error, trace = f'{type(exc).__name__}: {exc}', traceback.format_exc()
        key = os.environ.get('DEEPINFRA_API_KEY')
        if key:
            error, trace = error.replace(key, '<redacted>'), trace.replace(key, '<redacted>')
        (output / 'error.txt').write_text(trace)
        run.report.update(status='failed', error=error)
    finally:
        run.report['seconds'] = round(time.monotonic() - started, 3)
        run.save()
    return run.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, default=1)
    args = parser.parse_args()
    key = os.environ.get('DEEPINFRA_API_KEY') or dotenv_values(ROOT / '.env').get(
        'DEEPINFRA_API_KEY'
    )
    if not key:
        parser.error('DEEPINFRA_API_KEY is missing')
    os.environ['DEEPINFRA_API_KEY'] = key
    report = asyncio.run(execute(args.output.resolve(), Budget(args.budget_usd)))
    print(json.dumps({k: v for k, v in report.items() if k != 'requests'}, indent=2))
    return 0 if report['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
