"""Opt-in paired 27B thinking prompt experiment: 24 calls, at most $1.

The production baseline is unchanged. A generic documentation recipe is tested
on three known regressions and three fresh synthetic cases. Exact prompts,
sources and separate review oracles freeze before calls. No automatic factual
pass, production prompt change, local Q4 or hardware qualification is implied.
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
    API_REFERENCE,
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
from tests.source_verification_cases import CASES_BY_NAME, observe_case

MAX_ATTEMPTS = 32
DOC_TYPES = ('explanation', 'qa')
KNOWN_CASES = ('cache', 'guarded_cache', 'token_expiry')
CANDIDATE_SYSTEM = '''You write accurate developer documentation from supplied source code.
Document only behavior established by that source and any supplied evidence.
Cover normal results, boundary conditions, exceptions and state changes in execution order.
Check every example against the same source and keep examples consistent with the prose.
Do not invent APIs, fixes, callers or integration details. Distinguish source behavior from
an assumption about inputs or an external component. State when integration evidence is absent.
Keep the document concise without omitting behavior; do not pad it with generic engineering advice.'''
CANDIDATE_TASKS = {
    'explanation': '''Write a Markdown explanation with clear headings covering:
1. Overview and the public operations present in the supplied code.
2. How each operation works: inputs, return values, execution order and changes to caller-owned state.
3. Boundary conditions, exceptions and state remaining after an exception, where established by the source.
4. Usage: include a small worked example only if every claimed result follows from the source;
   otherwise explain that behavior in prose. Examples must agree with the rest of the document.
5. Integration: describe only connections evidenced by the supplied material; explicitly identify missing evidence.
Explain the supplied implementation, without proposing replacement implementations or unsupported advice.
Return only the final documentation.''',
    'qa': '''Write 5-10 concise Q&A pairs in Markdown, using ## Question and ## Answer headings.
Cover the supplied public operations, normal inputs and results, execution order, state changes,
boundary conditions, exceptions and state remaining after failures. Keep complete coverage of
behavior established by the source instead of padding questions with generic engineering advice.
Include a small example only if every claimed result follows from the supplied implementation;
examples must agree with every other answer. Describe integration only when evidence is supplied,
and explicitly identify missing integration evidence. Do not propose replacement implementations.
Return only the final documentation.''',
}

NEW_SOURCES = {
    'partial_settings': '''def apply_changes(settings, changes, allowed):
    for name, value in changes:
        if name not in allowed:
            raise ValueError(name)
        if value is None:
            if name in settings:
                del settings[name]
        else:
            settings[name] = value
    return settings
''',
    'generator_groups': '''def grouped(items, size):
    if size < 1:
        raise ValueError("size must be positive")
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield tuple(batch)
            batch.clear()
    if batch:
        yield tuple(batch)
''',
    'resource_view': '''from contextlib import contextmanager

@contextmanager
def open_view(source, factory):
    owned = isinstance(source, str)
    handle = factory(source) if owned else source
    try:
        yield handle
    finally:
        if owned:
            handle.close()
''',
}
NEW_REQUIRED = {
    'partial_settings': (
        'Changes are processed in input order and mutate the supplied settings object. '
        'Allowed non-None values assign or replace entries; None removes an existing allowed entry '
        'and does nothing when that entry is absent. An unallowed name raises ValueError before '
        'that change is applied, retains earlier changes and skips later ones; there is no rollback. '
        'Repeated names take effect in sequence. Successful calls, including empty changes, return '
        'the same settings object.'),
    'generator_groups': (
        'grouped returns a generator; calling it alone does not consume items or execute its size check. '
        'Advancing with size below one raises ValueError before consuming input. For positive integer '
        'sizes it consumes input lazily and yields tuples of that size, followed by one shorter tuple '
        'if items remain. Empty input yields nothing. It preserves input order, and yielding an early '
        'group does not consume the rest of the iterator.'),
    'resource_view': (
        'open_view is a context manager. Entering with a string calls factory and yields its returned '
        'handle; any other source is yielded directly without calling factory. The handle is not closed '
        'before the body runs. A factory-created handle is closed on normal exit and when the body raises. '
        'Caller-supplied handles are not closed by this manager. A factory exception prevents entry and '
        'propagates. A close exception propagates and may replace an exception raised by the body.'),
}


def _checked_function(name: str, function: str):
    namespace = {}
    # Only these checked-in source literals are executed, never model output.
    exec(compile(NEW_SOURCES[name], name + '.py', 'exec'), namespace)  # noqa: S102
    return namespace[function]


def _observe_settings() -> list[dict]:
    function = _checked_function('partial_settings', 'apply_changes')
    observations = []
    probes = (
        ('replace_and_remove', {'a': 1}, [('b', 2), ('a', None)], {'a', 'b'}),
        ('later_invalid_key', {'a': 1}, [('a', 2), ('bad', 8), ('a', 3)], {'a'}),
        ('repeated_name', {}, [('a', 1), ('a', 3)], {'a'}),
        ('absent_removal', {}, [('a', None)], {'a'}),
        ('empty_changes', {'a': 1}, [], {'a'}),
    )
    for label, settings, changes, allowed in probes:
        result = {'scenario': label}
        try:
            returned = function(settings, changes, allowed)
            result['returned_same_object'] = returned is settings
        except ValueError as error:
            result['exception'] = type(error).__name__
        observations.append({**result, 'state': settings})
    return observations


def _observe_groups() -> list[dict]:
    function = _checked_function('generator_groups', 'grouped')

    class Tracked:
        def __init__(self):
            self.items, self.position = [1, 2, 3], 0

        def __iter__(self):
            return self

        def __next__(self):
            if self.position == len(self.items):
                raise StopIteration
            item = self.items[self.position]
            self.position += 1
            return item

    items = Tracked()
    generated = function(items, 2)
    first = {'scenario': 'lazy_groups', 'consumed_at_creation': items.position}
    first.update(first=list(next(generated)), consumed_after_first=items.position)
    first.update(rest=[list(group) for group in generated], consumed_at_end=items.position)
    items = Tracked()
    invalid = function(items, 0)
    second = {'scenario': 'invalid_size', 'creation_raised': False}
    try:
        next(invalid)
    except ValueError as error:
        second['exception_on_first_next'] = type(error).__name__
    second['consumed'] = items.position
    return [first, second, *(
        {'scenario': label, 'groups': [list(group) for group in function(values, size)]}
        for label, values, size in (
            ('empty_input', [], 2), ('unit_groups', [1, 2], 1), ('short_input', [1, 2], 3),
        )
    )]


def _observe_resources() -> list[dict]:
    function = _checked_function('resource_view', 'open_view')

    class Handle:
        def __init__(self, *, fails=False):
            self.close_calls, self.fails = 0, fails

        def close(self):
            self.close_calls += 1
            if self.fails:
                raise RuntimeError('close failed')

    calls, handle = [], Handle()

    def factory(name):
        calls.append(name)
        return handle

    def body_error():
        raise ValueError('body failed')

    with function('data', factory) as actual:
        first = {'scenario': 'owned_normal', 'factory_calls': list(calls),
                 'same_handle': actual is handle, 'closed_inside': bool(handle.close_calls)}
    first['close_calls'] = handle.close_calls
    calls, handle = [], Handle()
    with function(handle, factory) as actual:
        second = {'scenario': 'borrowed_normal', 'factory_calls': list(calls), 'same_handle': actual is handle}
    second['close_calls'] = handle.close_calls
    handle = Handle()
    third = {'scenario': 'owned_body_error'}
    try:
        with function('data', factory):
            body_error()
    except ValueError as error:
        third['exception'] = type(error).__name__
    third['close_calls'] = handle.close_calls

    def broken_factory(_name):
        raise LookupError('open failed')

    fourth = {'scenario': 'factory_error', 'body_entered': False}
    try:
        with function('data', broken_factory):
            fourth['body_entered'] = True
    except LookupError as error:
        fourth['exception'] = type(error).__name__
    handle = Handle(fails=True)
    fifth = {'scenario': 'close_masks_body_error'}
    try:
        with function('data', factory):
            body_error()
    except RuntimeError as error:
        fifth['exception'] = type(error).__name__
    fifth['close_calls'] = handle.close_calls
    handle = Handle()
    sixth = {'scenario': 'borrowed_body_error'}
    try:
        with function(handle, factory):
            body_error()
    except ValueError as error:
        sixth['exception'] = type(error).__name__
    sixth['close_calls'] = handle.close_calls
    handle = Handle(fails=True)
    seventh = {'scenario': 'owned_close_error'}
    try:
        with function('data', factory):
            pass
    except RuntimeError as error:
        seventh['exception'] = type(error).__name__
    seventh['close_calls'] = handle.close_calls
    return [first, second, third, fourth, fifth, sixth, seventh]


def frozen_cases() -> dict:
    cases = {
        name: {'source': CASES_BY_NAME[name].source, 'split': 'known_regression',
               'required_behavior': CASES_BY_NAME[name].correct, 'oracle': observe_case(CASES_BY_NAME[name])}
        for name in KNOWN_CASES
    }
    for name, observe in (
        ('partial_settings', _observe_settings), ('generator_groups', _observe_groups),
        ('resource_view', _observe_resources),
    ):
        cases[name] = {'source': NEW_SOURCES[name], 'split': 'fresh_synthetic',
                       'required_behavior': NEW_REQUIRED[name], 'oracle': observe()}
    return cases


def candidate_request(baseline: dict, doc_type: str) -> dict:
    if doc_type not in CANDIDATE_TASKS:
        raise ValueError('Unsupported experiment document type')
    candidate = copy.deepcopy(baseline)
    original = candidate['messages'][1]['content']
    if original.count('\n---\n\n') != 1:
        raise ValueError('Production prompt has no unique task boundary')
    evidence, _task = original.split('\n---\n\n')
    candidate['messages'][0]['content'] = CANDIDATE_SYSTEM
    candidate['messages'][1]['content'] = evidence + '\n---\n\n' + CANDIDATE_TASKS[doc_type]
    validate_request(candidate)
    return candidate


async def prepare_run(output: Path, budget: Budget) -> tuple[Run, dict]:
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, MAX_ATTEMPTS)
    run = Run(PROFILE, output, budget, pacing=0)
    generator, cases, review = DocGenerator(), {}, {}
    for name, case in frozen_cases().items():
        source = output / 'sources' / f'{name}.py'
        source.write_text(case['source'])
        requests = {}
        for bundle in await generator.build_prompts_for_module(generator.analyzer.analyze_file(source), DOC_TYPES):
            baseline = request_for(bundle.system_prompt, bundle.user_prompt)
            requests[bundle.doc_type] = {'baseline': baseline, 'candidate': candidate_request(baseline, bundle.doc_type)}
        cases[name] = {'split': case['split'], 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                       'requests': requests}
        review[name] = {key: case[key] for key in ('split', 'required_behavior', 'oracle')}
    save_json(output / 'review-oracles.json', review)
    schedule = []
    for name in cases:
        for kind in DOC_TYPES:
            arms = ('baseline', 'candidate') if len(schedule) // 2 % 2 == 0 else ('candidate', 'baseline')
            schedule.extend({'case': name, 'doc_type': kind, 'arm': arm,
                             'file': f'{name}-{kind}-{arm}.md'} for arm in arms)
    source_paths = (
        Path(__file__), ROOT / 'tests/test_live_mac_generation_experiment.py',
        ROOT / 'tests/live_mac_accuracy_probe.py', ROOT / 'tests/live_model_smoke.py',
        ROOT / 'tests/source_verification_cases.py', ROOT / 'tests/model_smoke_cases.py',
        ROOT / 'docgen/generator.py', ROOT / 'docgen/prompts.py',
        ROOT / 'docgen/_legacy_analyzer.py', ROOT / 'docgen/_legacy_metadata.py',
        ROOT / 'docgen/llm/local.py', ROOT / 'docgen/validator.py', ROOT / 'model_config.py',
    )
    hashes = {}
    for path in source_paths:
        relative, content = path.relative_to(ROOT), path.read_bytes()
        destination = output / 'provenance' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    protocol = {
        'scope': 'Paired hosted 27B thinking documentation prompt-recipe experiment; generation only',
        'model': MODEL, 'cases': cases, 'schedule': schedule, 'repetitions': 1, 'concurrency': 1,
        'budget_usd': budget.limit, 'maximum_http_attempts': budget.max_requests, 'timeout_seconds': TIMEOUT,
        'context_window': CONTEXT_WINDOW, 'max_total_output_tokens': MAX_OUTPUT,
        'context_admission': 'Conservative UTF-8 byte envelope plus 1024 overhead; not exact tokenizer preflight',
        'changed_factor': 'Documentation prompt recipe only; system/task directives replaced, source/metadata retained',
        'candidate_system': CANDIDATE_SYSTEM, 'candidate_tasks': CANDIDATE_TASKS,
        'baseline': 'Unchanged production DocGenerator.build_prompts_for_module explanation and QA requests',
        'order_policy': 'Adjacent paired arms, alternating initial arm; one sample per case/type/arm',
        'holdout_limit': 'Fresh to this experiment, not claimed training-independent or unseen by all prior studies; '
                         'three known regressions plus three newly authored behavior families. Broader confirmation remains separate.',
        'source_hashes': hashes,
        'review_oracles_sha256': hashlib.sha256((output / 'review-oracles.json').read_bytes()).hexdigest(),
        'review_policy': {
            'method': 'Independent source-based review of every full document; no automatic factual pass',
            'decision_subject': 'Candidate arm; baseline supplies paired comparisons and previously correct facts',
            'dimensions': ['material false claims or contradictions', 'required behavior coverage',
                           'examples and advice', 'unsupported integration', 'structural validity'],
            'coverage_rule': 'Omissions and abstentions are coverage misses.',
            'non_regression_unit': 'Each required fact within the same case/document-type pair; aggregate gains cannot conceal a lost fact',
            'promising_requires_all': [
                'Fewer documents containing material false claims or contradictions than baseline',
                'No loss of correctly documented required behavior',
                'No reduction in required behavior coverage',
            ],
            'further_validation_requires_all': [
                'Every scheduled document completed and independently reviewed',
                'Zero material false claims or contradictions',
                'All required behavior consistently covered correctly',
                'No loss of previously correct behavior',
            ],
            'production_qualification': False,
            'production_change_requires': 'Technical review of evidence and untouched confirmation; user already authorized local fixes',
        },
        'retry_policy': 'Only HTTP 429 may retry twice within 32 total attempts; no content or timeout retries',
        'response_archive_policy': 'Final-only transformed responses; private reasoning omitted, original hash/size retained',
        'hosted_mapping': {'reference': API_REFERENCE,
                           'limits': 'Provider deployment not pinned to local GGUF; no exact native/hosted wire parity',
                           'repeat_penalty': 'Native repeat_penalty maps to hosted repetition_penalty',
                           'reasoning_format': 'Native deepseek option omitted; only hosted separated final content is archived'},
        'pricing_source': 'https://deepinfra.com/Qwen/Qwen3.8-27B/api',
        'local_artifact_qualified': False, 'hardware_qualified': False,
    }
    save_json(output / 'protocol.json', protocol)
    run.report.update(scope=protocol['scope'], quality_verdict='unreviewed', trials=[],
                      expected_documents=24, embedding_model=None,
                      protocol_sha256=hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
                      rates_verified_on='2026-10-22', local_artifact_qualified=False, hardware_qualified=False)
    run.save()
    return run, protocol


def _verify_frozen(run: Run, output: Path, protocol: dict) -> None:
    for path, expected in (
        (output / 'protocol.json', run.report['protocol_sha256']),
        (output / 'review-oracles.json', protocol['review_oracles_sha256']),
        *((output / 'sources' / f'{name}.py', case['source_sha256']) for name, case in protocol['cases'].items()),
        *((output / 'provenance' / name, digest) for name, digest in protocol['source_hashes'].items()),
    ):
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Frozen experiment evidence changed; no further requests are allowed')


async def execute(output: Path, budget: Budget, *, api_key: str,
                  transport_factory=httpx.AsyncHTTPTransport) -> dict:
    if budget.limit > 1:
        raise ValueError('This experiment permits at most $1 in request accounting')
    if not api_key:
        raise ValueError('DeepInfra credential is missing')
    run, protocol = await prepare_run(output, budget)
    run.report['transport_mode'] = 'live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated'
    started, active = time.monotonic(), None
    try:
        async with httpx.AsyncClient(
            base_url=PROFILE.base_url, timeout=TIMEOUT, trust_env=False, follow_redirects=False,
            headers={'Authorization': f'Bearer {api_key}'},
            transport=RecordedTransport(run, 'chat', FinalOnlyTransport(run, transport_factory())),
        ) as client:
            for scheduled in protocol['schedule']:
                active = None
                _verify_frozen(run, output, protocol)
                active = {**scheduled, 'status': 'running', 'first_http_attempt': len(run.report['requests']) + 1}
                run.report['trials'].append(active)
                run.save()
                payload = protocol['cases'][scheduled['case']]['requests'][scheduled['doc_type']][scheduled['arm']]
                validate_request(payload)
                response = await client.post('/chat/completions', json=payload)
                response.raise_for_status()
                content, counts = validate_final(response.json())
                content = content.replace(api_key, '<redacted>')
                (output / 'documents' / scheduled['file']).write_text(content)
                validation = ContentValidator().validate(content, scheduled['doc_type'], scheduled['case'])
                run.report['documents'].append({
                    **scheduled, 'sha256': hashlib.sha256(content.encode()).hexdigest(), **counts,
                    'reasoning_tokens': run.report['requests'][-1]['response_archive']['reasoning_tokens'],
                    'structurally_valid': validation.is_valid,
                    'issues': [{'level': item.level, 'code': item.code} for item in validation.issues],
                })
                active.update(status='completed', last_http_attempt=len(run.report['requests']))
                run.save()
        run.report['status'] = 'completed_pending_review'
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        if active is not None:
            active['status'] = 'cancelled'
        raise
    except Exception as error:  # noqa: BLE001 - preserve terminal evidence without private response text.
        run.report.update(status='failed', error_type=type(error).__name__)
        if active is not None:
            active.update(status='failed', last_http_attempt=len(run.report['requests']))
    finally:
        run.report['seconds'] = time.monotonic() - started
        run.save()
    return run.report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, default=1)
    args = parser.parse_args()
    key = os.environ.get('DEEPINFRA_API_KEY') or dotenv_values(ROOT / '.env').get('DEEPINFRA_API_KEY')
    if not key:
        parser.error('DEEPINFRA_API_KEY is missing')
    report = asyncio.run(execute(args.output.resolve(), Budget(args.budget_usd, MAX_ATTEMPTS), api_key=key))
    print(json.dumps({key: value for key, value in report.items() if key not in ('requests', 'trials')}, indent=2))
    return 0 if report['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
