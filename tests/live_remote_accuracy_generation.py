"""Bounded, opt-in source-grounded hosted generation evaluation.

Baseline repository cases and a separate QA task-scope experiment preserve
source truth outside model requests. Execution success never grants accuracy.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from docgen.generator import DocGenerator
from docgen.validator import ContentValidator
from tests.live_mac_accuracy_probe import (
    CONTEXT_WINDOW,
    MAX_OUTPUT,
    MODEL,
    PROFILE,
    TIMEOUT,
    FinalOnlyTransport,
    request_for,
    validate_final,
    validate_request,
)
from tests.live_model_smoke import ROOT, Budget, RecordedTransport, Run, save_json

MAX_ATTEMPTS = 48
QA_SCOPE_TASK = '''Generate source-proportional Q&A documentation: use as many questions as needed
to cover every established public operation, normal result, boundary condition,
exception and state change, without a fixed minimum question count.

Format each Q&A as:
## Question
[Question text]

## Answer
[Answer text with code examples if helpful]

Cover basic usage and tasks actually supported by the supplied source. Include
configuration, customization, integration or best-practice topics only when the
supplied material establishes them. Do not add topics merely to fill a quota.
Do not omit established behavior to shorten the document.

Output in Markdown format.'''


def corpus_cases():
    from tests.remote_accuracy_cases import frozen_cases

    return frozen_cases()


def regression_cases():
    from tests.live_mac_generation_experiment import NEW_SOURCES
    from tests.model_smoke_cases import CASES
    from tests.source_verification_cases import CASES_BY_NAME

    cache = CASES_BY_NAME['cache']
    sources = {
        'mean': (CASES[0][2], {
            'empty': 'Empty input raises ValueError.',
            'formula': 'Nonempty input returns sum(values)/len(values).',
            'return_type': 'Return type follows the arithmetic; complex and Decimal inputs need not produce float.',
            'rounding': 'There is no explicit rounding step.',
        }),
        'cache': (cache.source, {
            'zero': 'Zero capacity leaves an empty cache and returns None.',
            'negative_empty': 'Negative capacity empties the cache.',
            'negative_error': 'Negative capacity then raises KeyError on another empty popitem.',
            'termination': 'The exception terminates execution; no infinite loop.',
            'read_recency': 'Only writes refresh recency; a plain read does not.',
        }),
        'resource_view': (NEW_SOURCES['resource_view'], {
            'context': 'open_view is a context manager.',
            'owned': 'A string source calls factory and yields its returned handle.',
            'borrowed': 'Other sources are yielded directly without calling factory.',
            'open_inside': 'The handle is not closed before the body runs.',
            'normal_close': 'Factory-created handles are closed on normal exit.',
            'error_close': 'Factory-created handles are closed when the body raises.',
            'borrowed_cleanup': 'Caller-supplied handles are not closed by this manager.',
            'factory_error': 'A factory exception prevents entry and propagates.',
            'close_error': 'A close exception propagates.',
            'close_masks': 'A close exception may replace an exception raised by the body.',
        }),
    }
    return {name: {'case_id': name, 'split': 'known_regression', 'source': source,
                   'doc_types': ['qa'], 'required_facts': [{'id': key, 'text': text} for key, text in facts.items()],
                   'source_segments': [], 'observations': {},
                   'origin': 'Previously reviewed synthetic regression; not a holdout.'}
            for name, (source, facts) in sources.items()}


def scoped_qa_request(baseline):
    candidate = copy.deepcopy(baseline)
    original = candidate['messages'][1]['content']
    if original.count('\n---\n\n') != 1:
        raise ValueError('Expected unique production evidence/task boundary')
    evidence, _ = original.split('\n---\n\n')
    candidate['messages'][1]['content'] = evidence + '\n---\n\n' + QA_SCOPE_TASK
    validate_request(candidate)
    return candidate


async def prepare(output: Path, budget: Budget, *, mode='baseline', split='development', repeats=2):
    if (mode not in ('baseline', 'qa_scope') or split not in ('development', 'confirmation')
            or type(repeats) is not int or not 1 <= repeats <= 3
            or (mode == 'qa_scope' and split != 'development') or budget.limit > 1):
        raise ValueError('Invalid bounded experiment schedule or budget')
    specs = regression_cases() if mode == 'qa_scope' else {
        key: case for key, case in corpus_cases().items() if case['split'] == split}
    if not specs or len(specs) > 6:
        raise ValueError('Expected one to six frozen cases')
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, MAX_ATTEMPTS)
    run = Run(PROFILE, output, budget, pacing=0)
    generator, cases, review = DocGenerator(), {}, {}
    for name, spec in specs.items():
        if not name.replace('_', '').isalnum():
            raise ValueError('Unsafe case identity')
        source = output / 'sources' / f'{name}.py'
        source.write_text(spec['source'])
        requests = {}
        kinds = ('qa',) if mode == 'qa_scope' else tuple(spec['doc_types'])
        for bundle in await generator.build_prompts_for_module(generator.analyzer.analyze_file(source), kinds):
            baseline = request_for(bundle.system_prompt, bundle.user_prompt)
            requests[bundle.doc_type] = {'baseline': baseline}
            if mode == 'qa_scope':
                requests[bundle.doc_type]['candidate'] = scoped_qa_request(baseline)
        if set(requests) != set(kinds):
            raise ValueError('Production generator omitted a scheduled document type')
        cases[name] = {'split': spec['split'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'source_segments': spec.get('source_segments', []), 'requests': requests}
        review[name] = {key: value for key, value in spec.items() if key != 'source'}
    save_json(output / 'review-oracles.json', review)
    schedule = []
    pair = 0
    for repeat in range(1, repeats + 1):
        for name, case in cases.items():
            kinds = list(case['requests'])
            if repeat % 2 == 0:
                kinds.reverse()
            for kind in kinds:
                arms = ('baseline', 'candidate') if mode == 'qa_scope' else ('baseline',)
                if pair % 2:
                    arms = tuple(reversed(arms))
                pair += 1
                schedule.extend({'case': name, 'doc_type': kind, 'arm': arm, 'repeat': repeat,
                                 'file': f'{name}-{kind}-{arm}-r{repeat}.md'} for arm in arms)
    if len(schedule) > MAX_ATTEMPTS:
        raise ValueError('Schedule exceeds maximum request count')
    paths = (
        'tests/live_remote_accuracy_generation.py', 'tests/test_live_remote_accuracy_generation.py',
        'tests/remote_accuracy_cases.py', 'tests/test_remote_accuracy_cases.py',
        'tests/live_mac_accuracy_probe.py', 'tests/live_model_smoke.py',
        'tests/live_mac_generation_experiment.py', 'tests/model_smoke_cases.py',
        'tests/source_verification_cases.py', 'docgen/generator.py', 'docgen/prompts.py',
        'docgen/_legacy_analyzer.py', 'docgen/_legacy_metadata.py', 'docgen/validator.py',
        'docgen/llm/local.py', 'model_config.py',
    )
    hashes = {}
    for name in paths:
        source = ROOT / name
        if not source.exists():
            raise ValueError('Missing evaluation provenance')
        target = output / 'provenance' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    protocol = {
        'scope': 'Hosted source-grounded generation evaluation; separate factual review required',
        'mode': mode, 'split': split, 'model': MODEL, 'cases': cases, 'schedule': schedule,
        'repetitions': repeats, 'concurrency': 1, 'budget_usd': budget.limit,
        'maximum_http_attempts': budget.max_requests, 'timeout_seconds': TIMEOUT,
        'context_window': CONTEXT_WINDOW, 'max_total_output_tokens': MAX_OUTPUT,
        'context_admission': 'UTF-8 byte envelope plus1024 overhead, not native tokenization',
        'changed_factor': 'QA task topic/count scope jointly; system/evidence/format/settings unchanged'
                          if mode == 'qa_scope' else 'None: production explanation/QA prompts',
        'source_hashes': hashes,
        'review_oracles_sha256': hashlib.sha256((output / 'review-oracles.json').read_bytes()).hexdigest(),
        'quality_gates': {
            'automatic_factual_pass': False,
            'requires': ['Every scheduled output completed and independently reviewed against source',
                         'Zero material false claims or unsupported implementation claims',
                         'Every required fact consistently covered; omissions and abstentions are misses',
                         'No loss of correct behavior within paired case/type/repetition'],
            'candidate_promotion': 'Requires separate untouched confirmation; aggregate gains cannot mask lost facts',
            'statistical_scope': 'Finite purposive sample; no universal accuracy estimate',
        },
        'retry_policy': 'Only bounded HTTP429 retries; no content or timeout retries; all attempts retained',
        'response_archive': 'Final-only content/usage and original response hash; private reasoning omitted',
        'native_mapping': 'Native repeat_penalty maps to repetition_penalty; reasoning_format omitted',
        'pricing_checked_utc': '2026-10-23', 'local_q4_qualified': False, 'm5_qualified': False,
        'limitations': ['Hosted deployment precision/weights are not immutable pinned GGUFs',
                        'Source extracts are selected cases, not complete repository qualification'],
    }
    save_json(output / 'protocol.json', protocol)
    run.report.update(scope=protocol['scope'], quality_verdict='unreviewed', trials=[],
                      expected_documents=len(schedule), embedding_model=None,
                      protocol_sha256=hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
                      rates_verified_on='2026-10-23', local_q4_qualified=False, m5_qualified=False)
    run.save()
    return run, protocol


def verify_frozen(run, protocol):
    paths = {
        'protocol.json': run.report['protocol_sha256'],
        'review-oracles.json': protocol['review_oracles_sha256'],
        **{f'sources/{name}.py': case['source_sha256'] for name, case in protocol['cases'].items()},
        **{f'provenance/{name}': digest for name, digest in protocol['source_hashes'].items()},
    }
    for name, expected in paths.items():
        if hashlib.sha256((run.output / name).read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen evidence changed; no further requests allowed')


async def execute(output, budget, *, api_key, mode='baseline', split='development', repeats=2,
                  transport_factory=httpx.AsyncHTTPTransport):
    if not api_key:
        raise ValueError('Provider credential required')
    run, protocol = await prepare(output, budget, mode=mode, split=split, repeats=repeats)
    run.report['transport_mode'] = 'live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated'
    started, active = time.monotonic(), None
    try:
        async with httpx.AsyncClient(
            base_url=PROFILE.base_url, timeout=TIMEOUT, trust_env=False, follow_redirects=False,
            headers={'Authorization': f'Bearer {api_key}'},
            transport=RecordedTransport(run, 'chat', FinalOnlyTransport(run, transport_factory())),
        ) as client:
            for row in protocol['schedule']:
                active = None
                verify_frozen(run, protocol)
                active = {**row, 'status': 'running', 'first_http_attempt': len(run.report['requests']) + 1}
                run.report['trials'].append(active)
                run.save()
                payload = protocol['cases'][row['case']]['requests'][row['doc_type']][row['arm']]
                validate_request(payload)
                response = await client.post('/chat/completions', json=payload)
                response.raise_for_status()
                content, usage = validate_final(response.json())
                content = content.replace(api_key, '<redacted>')
                (output / 'documents' / row['file']).write_text(content)
                validation = ContentValidator().validate(content, row['doc_type'], row['case'])
                run.report['documents'].append({**row, 'sha256': hashlib.sha256(content.encode()).hexdigest(),
                                                **usage, 'structurally_valid': validation.is_valid,
                                                'issues': [{'level': i.level, 'code': i.code} for i in validation.issues]})
                active.update(status='completed', last_http_attempt=len(run.report['requests']))
                run.save()
        run.report['status'] = 'completed_pending_review'
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        if active is not None:
            active['status'] = 'cancelled'
        raise
    except Exception as error:  # noqa: BLE001 - preserve bounded terminal failure without secret content.
        run.report.update(status='failed', error_type=type(error).__name__)
        if active is not None:
            active.update(status='failed', last_http_attempt=len(run.report['requests']))
    finally:
        run.report['seconds'] = time.monotonic() - started
        run.save()
    return run.report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--mode', choices=('baseline', 'qa_scope'), default='baseline')
    parser.add_argument('--split', choices=('development', 'confirmation'), default='development')
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--budget-usd', type=float, default=.75)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    kwargs = {'mode': args.mode, 'split': args.split, 'repeats': args.repeats}
    budget = Budget(args.budget_usd, MAX_ATTEMPTS)
    if args.prepare_only:
        run, _ = asyncio.run(prepare(args.output.resolve(), budget, **kwargs))
        print(json.dumps({'prepared': str(run.output), 'documents': run.report['expected_documents']}))
        return 0
    key = os.environ.get('DEEPINFRA_API_KEY') or dotenv_values(ROOT / '.env').get('DEEPINFRA_API_KEY')
    if not key:
        parser.error('DEEPINFRA_API_KEY missing')
    report = asyncio.run(execute(args.output.resolve(), budget, api_key=key, **kwargs))
    print(json.dumps({key: report[key] for key in ('status', 'seconds', 'accounted_upper_bound_usd')}))
    return 0 if report['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
