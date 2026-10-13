"""Compare hosted Qwen/OpenAI embeddings on a frozen repository retrieval corpus.

python -m tests.live_retrieval_benchmark --output NEW_DIRECTORY --budget-usd 0.50
Makes embedding requests only. Quality metrics are observations, not assertions
about model correctness. Nonzero exit means the experiment or an implementation
check failed. No prompts, thresholds or relevance labels are tuned during a run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import httpx
import numpy as np
from dotenv import dotenv_values

from ariadne_mcp.service import AriadneService
from inference import InferenceContext
from model_config import query_input
from tests.live_model_smoke import (
    PROFILES,
    ROOT,
    Budget,
    Run,
    attach_clients,
    configuration,
    save_json,
)
from tests.live_qwen_blind import check_vectors, rank_matches
from tests.retrieval_cases import FIXTURE
from writer import LibraryWriter


def metrics(rankings, questions, level):
    rows = [(row, q[level]) for row, q in zip(rankings, questions, strict=True) if q[level]]
    recalls, hits, reciprocal = [], [], []
    for row, relevant in rows:
        retrieved = row[level][:5]
        matched = set(retrieved) & set(relevant)
        recalls.append(len(matched) / len(set(relevant)))
        hits.append(bool(matched))
        reciprocal.append(next((1 / i for i, key in enumerate(retrieved, 1) if key in relevant), 0))
    return {
        'eligible_questions': len(rows),
        'recall_at_5': float(np.mean(recalls)) if rows else None,
        'hit_at_5': float(np.mean(hits)) if rows else None,
        'mrr_at_5': float(np.mean(reciprocal)) if rows else None,
    }


class FixedQueries:
    """Reuse the same paid query vectors in each retrieval path; never synthesize vectors."""

    def __init__(self, original, questions, vectors):
        self.original = original
        self.vectors = dict(zip(questions, vectors, strict=True))

    async def embed_query(self, text):
        return self.vectors[text].copy()

    async def close(self):
        await self.original.close()


async def query_vectors(embeddings, profile, questions):
    # Both production query methods delegate to document embedding after this
    # exact formatting. Batching avoids 50 separately paced hosted Qwen calls.
    inputs = [query_input(q) if profile.name == 'qwen' else q for q in questions]
    result = []
    for start in range(0, len(inputs), 25):
        result.extend(await embeddings.embed_documents(inputs[start : start + 25]))
    return result


async def workflows(run, corpus, transport_factory):
    cfg = configuration(run.profile, run.output)
    context = InferenceContext.from_config(cfg)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    try:
        await attach_clients(context, run, transport_factory)
        with context.write_session():
            library = context.open_library(operation='write')
            writer = LibraryWriter(library)
            for row in corpus['documents']:
                doc = await writer.add_document(
                    'explanation',
                    row['title'],
                    row['content'],
                    [row['source_file']],
                    doc_id=row['id'],
                    source_name='synthetic',
                )
                assert doc.content == row['content']
                assert [c.content for c in library.get_chunks(doc.id)] == row['chunks']
                assert [(s.heading, s.content) for s in library.get_sections(doc.id)] == [
                    (heading or 'Introduction', text) for heading, text in row['sections']
                ]
            run.check('frozen_corpus_roundtrip', True, documents=len(corpus['documents']))
            check_vectors(
                run, library, native_dimensions=4096 if run.profile.name == 'qwen' else 3072
            )

        questions = corpus['questions']
        texts = [q['question'] for q in questions]
        vectors = await query_vectors(context.embeddings, run.profile, texts)
        context._embeddings = FixedQueries(context.embeddings, texts, vectors)
        docs = library.list_documents()
        chunks = [c for d in docs for c in library.get_chunks(d.id)]
        sections = [s for d in docs for s in library.get_sections(d.id)]
        section_matrix = np.stack([s.embedding for s in sections]).astype(np.float64)
        rankings = []
        # Finish every library search before MCP writes usage events. Unanswered
        # events must not down-weight later questions in this fixed-corpus study.
        for q, vector in zip(questions, vectors, strict=True):
            documents = library.search(vector, k=len(docs))
            matches = library.search_chunks(vector, k=len(chunks))
            run.check(
                'numeric_ranking/' + q['id'],
                rank_matches(vector, documents, {d.id: d.embedding for d in docs})
                and rank_matches(vector, matches, {c.id: c.embedding for c in chunks}),
            )
            scores = section_matrix @ vector.astype(np.float64)
            order = np.argsort(-scores, kind='stable')[:5]
            rankings.append(
                {
                    'id': q['id'],
                    'question': q['question'],
                    'documents': [r.document.id for r in documents[:5]],
                    'document_scores': [r.score for r in documents[:5]],
                    'chunks': [f'{r.document.id}:{r.chunk.chunk_index}' for r in matches[:5]],
                    'chunk_scores': [r.score for r in matches[:5]],
                    'sections': [f'{sections[i].document_id}:{sections[i].index}' for i in order],
                    'section_scores': [float(scores[i]) for i in order],
                }
            )
        run.report['metrics'] = {
            level: metrics(rankings, questions, level)
            for level in ('documents', 'chunks', 'sections')
        }
        # Sections above are a global heading-vector diagnostic. The following
        # separately measures the actual MCP document-then-section delivery path.
        for row, q in zip(rankings, questions, strict=True):
            response = await service.search(
                query=q['question'], source='synthetic', limit=5, sections_only=True
            )
            row['mcp_documents'] = [d.id for d in response.documents]
            row['mcp_evidence_delivered'] = any(
                d.id in q['documents'] and q['evidence'] in d.content for d in response.documents
            )
            save_json(run.output / f'selection-{q["id"]}.json', response.model_dump(mode='json'))
        run.report['metrics']['mcp_evidence_delivery'] = {
            'questions': len(questions),
            'delivered': sum(r['mcp_evidence_delivered'] for r in rankings),
            'rate': sum(r['mcp_evidence_delivered'] for r in rankings) / len(questions),
        }
        save_json(run.output / 'rankings.json', rankings)
        run.check(
            'embedding_requests_only',
            all(r['kind'] == 'embeddings' for r in run.report['requests']),
        )
    finally:
        await service.close()
        await context.aclose()


async def execute(
    profile, output, budget, corpus, *, transport_factory=httpx.AsyncHTTPTransport, pacing=15
):
    run = Run(profile, output, budget, pacing=pacing)
    run.report.update(
        scope='Frozen repository excerpts; embedding retrieval only',
        corpus_sha256=hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
        transport_mode='live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated',
    )
    start = time.monotonic()
    try:
        await workflows(run, corpus, transport_factory)
        run.report['status'] = (
            'passed' if all(c['passed'] for c in run.report['checks']) else 'failed'
        )
    except Exception as exc:  # noqa: BLE001 - preserve failed experiment evidence and return nonzero.
        run.report.update(status='failed', error=str(exc))
        # Response/transport exceptions may echo credentials; redact before saving.
        trace = traceback.format_exc()
        for key in (profile.key_env, profile.embedding_key):
            secret = os.environ.get(key)
            if secret:
                trace = trace.replace(secret, '<redacted>')
                run.report['error'] = run.report['error'].replace(secret, '<redacted>')
        (output / 'error.txt').write_text(trace)
    finally:
        run.report['seconds'] = round(time.monotonic() - start, 3)
        run.save()
    return run.report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('all', 'qwen', 'openai'), default='all')
    parser.add_argument('--budget-usd', type=float, default=0.5)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    profiles = [
        PROFILES[name]
        for name in (('qwen', 'openai') if args.profile == 'all' else (args.profile,))
    ]
    budget = Budget(args.budget_usd)
    values = dotenv_values(ROOT / '.env')
    for profile in profiles:
        for name in {profile.key_env, profile.embedding_key}:
            if not os.environ.get(name) and values.get(name):
                os.environ[name] = values[name]
            if not os.environ.get(name):
                parser.error(f'Missing credential variable: {name}')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    corpus = json.loads(FIXTURE.read_text())
    shutil.copyfile(FIXTURE, output / 'corpus.json')
    dependencies = {}
    for filename in (
        'live_retrieval_benchmark.py',
        'retrieval_cases.py',
        'live_model_smoke.py',
        'live_qwen_blind.py',
    ):
        path = ROOT / 'tests' / filename
        shutil.copyfile(path, output / (filename + '.txt'))
        dependencies[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
    save_json(
        output / 'protocol.json',
        {
            'corpus_sha256': hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),
            'runner_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'dependencies': dependencies,
            'profiles': [p.name for p in profiles],
            'budget_usd': budget.limit,
            'questions': len(corpus['questions']),
            'documents': len(corpus['documents']),
            'quality_policy': 'Report every question and failure; no reranking, prompt changes, threshold tuning or selective reruns.',
            'judgments': corpus['judgment_method'],
            'metrics': 'Mean recall, hit rate and reciprocal rank at five known-relevant documents/chunks; global section-heading ranking is diagnostic. MCP evidence delivery uses its unchanged threshold and full-content fallback.',
            'short_documents': 'Four questions have no chunk targets because production does not chunk short documents; chunk denominator is 46.',
            'baseline': 'Separate OpenAI library with the same corpus. Report Qwen minus OpenAI; the proposed tolerance is five percentage points. No comparison of vector coordinates.',
            'queries': 'Batch the exact production query input recipe; reuse each resulting vector for all retrieval paths.',
        },
    )
    print(f'Evidence directory: {output}', flush=True)

    async def run():
        summary = {'profiles': {}, 'budget_usd': budget.limit}
        for profile in profiles:
            summary['profiles'][profile.name] = await execute(
                profile, output / profile.name, budget, corpus
            )
            summary['accounted_upper_bound_usd'] = budget.committed
            save_json(output / 'summary.json', summary)
        if all(
            summary['profiles'].get(name, {}).get('status') == 'passed'
            for name in ('qwen', 'openai')
        ):
            comparisons = {}
            for level in ('documents', 'chunks', 'sections'):
                qwen = summary['profiles']['qwen']['metrics'][level]['recall_at_5']
                baseline = summary['profiles']['openai']['metrics'][level]['recall_at_5']
                comparisons[level] = {
                    'qwen': qwen,
                    'openai': baseline,
                    'difference_percentage_points': (qwen - baseline) * 100,
                    'within_proposed_five_point_tolerance': qwen + 0.05 >= baseline,
                }
            summary['comparison'] = comparisons
        summary['implementation_checks_passed'] = all(
            r['status'] == 'passed' for r in summary['profiles'].values()
        )
        save_json(output / 'summary.json', summary)
        return 0 if summary['implementation_checks_passed'] else 1

    return asyncio.run(run())


if __name__ == '__main__':
    raise SystemExit(main())
