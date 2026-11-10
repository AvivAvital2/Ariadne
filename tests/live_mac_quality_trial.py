"""Opt-in paired 9B diagnostics; no production prompt or sampling is changed.

Source-scope schedules 24 chat requests; presence-zero schedules six.
Both use the existing DeepInfra recorder and freeze their protocols first.
Outputs always require separate factual review. This is neither a general
accuracy benchmark nor local Q4, embedding, application or M5 qualification.
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

from docgen.validator import ContentValidator
from tests.live_mac_model_probe import ARCHIVE, REMOTE_PROFILE, frozen_requests, response_content
from tests.live_model_smoke import ROOT, Budget, RecordedTransport, Run, save_json
from tests.source_verification_cases import CASES_BY_NAME, observe_case

CASE_NAMES = ('cache', 'guarded_cache', 'token_expiry', 'job_claim', 'index_lookup', 'range_pop')
SCOPE_INSTRUCTION = (
    'Keep this documentation strictly about behavior justified by the supplied source. '
    'Do not fill answers with generic engineering advice, speculative features, or invented APIs. '
    'Any example must use operations supported by that source. '
    'If a material claim is not established, omit it or explicitly identify it as uncertain.'
)
MAX_ATTEMPTS = 32
VARIANT_LIMITS = {'source-scope': (180, MAX_ATTEMPTS), 'presence-zero': (600, 10)}


def _substitute_confirmation(template: dict, name: str) -> dict:
    """Change only module, function and source in the exact frozen cache template."""
    payload = copy.deepcopy(template)
    case, original = CASES_BY_NAME[name], CASES_BY_NAME['cache']
    text = payload['messages'][1]['content']
    for before, after in (
        (f'**Module**: `{original.name}`', f'**Module**: `{name}`'),
        (f'**Functions**: {original.function}', f'**Functions**: {case.function}'),
        (f'```python\n{original.source}\n```', f'```python\n{case.source}\n```'),
    ):
        if text.count(before) != 1:
            raise ValueError(f'Frozen QA template no longer permits exact substitution for {name}')
        text = text.replace(before, after, 1)
    payload['messages'][1]['content'] = text
    return payload


def frozen_trial(variant: str = 'source-scope') -> dict:
    if variant not in VARIANT_LIMITS:
        raise ValueError(f'Unknown quality trial variant: {variant}')
    presence_zero = variant == 'presence-zero'
    timeout, max_attempts = VARIANT_LIMITS[variant]
    names = CASE_NAMES[:3] if presence_zero else CASE_NAMES
    repetitions = 1 if presence_zero else 2
    originals = frozen_requests()
    cases = {}
    for name in names:
        case = CASES_BY_NAME[name]
        baseline = (copy.deepcopy(originals[name]) if name in originals
                    else _substitute_confirmation(originals['cache'], name))
        candidate = copy.deepcopy(baseline)
        if presence_zero:
            candidate['presence_penalty'] = 0
        else:
            candidate['messages'][0]['content'] += '\n\n' + SCOPE_INSTRUCTION
        cases[name] = {
            'split': 'known_regression' if name in originals else 'nine_b_confirmation',
            'source_sha256': hashlib.sha256(case.source.encode()).hexdigest(),
            'oracle': observe_case(case),
            'review_target': {'question': case.question, 'required_behavior': case.correct},
            'request_origin': ('Exact previously frozen production QA request' if name in originals
                               else 'Frozen cache QA request with only module, function and source substituted'),
            'requests': {'baseline': baseline, 'candidate': candidate},
        }
    schedule = []
    for repeat in range(1, repetitions + 1):
        for index, name in enumerate(names):
            arms = ('baseline', 'candidate') if (index + repeat) % 2 else ('candidate', 'baseline')
            for arm in arms:
                schedule.append({'case': name, 'repeat': repeat, 'arm': arm,
                                 'file': f'{name}-{arm}-r{repeat}.md'})
    protocol = {
        'variant': variant,
        'timeout_seconds': timeout,
        'maximum_http_attempts': max_attempts,
        'scope': ('Hosted BF16 9B paired QA presence-penalty diagnostic; generation only' if presence_zero
                  else 'Hosted BF16 9B paired QA prompt-scope experiment; generation only'),
        'generation_model': REMOTE_PROFILE.model,
        'hosted_generation_precision': 'provider-reported bfloat16; immutable weights undisclosed',
        'quality_verdict': 'unreviewed',
        'local_q4_qualified': False,
        'hardware_qualified': False,
        'confirmation_limit': ('Three previously examined regression cases; no independent holdout.' if presence_zero
                               else 'The three confirmation cases were used in earlier 27B studies; '
                                    'not independent holdout and not novel behavior families.'),
        'single_changed_factor': ('Change presence_penalty from 1.5 to 0 only' if presence_zero
                                  else 'Append SCOPE_INSTRUCTION to the system message only'),
        'scope_instruction': None if presence_zero else SCOPE_INSTRUCTION,
        'repetitions': repetitions,
        'concurrency': 1,
        'cases': cases,
        'schedule': schedule,
        'order_policy': ('Adjacent paired arms; alternate initial arm across cases' if presence_zero
                         else 'Adjacent paired arms; reverse each case order on its second repeat'),
        'retry_policy': ('Only the existing recorder may retry HTTP 429, at most twice. '
                         'No retry for content, truncation or malformed responses; no selective reruns.'),
        'review_policy': {
            'method': 'Separate source-based review of every complete document; no automatic factual pass',
            'dimensions': ['material false claims and contradictions', 'critical target coverage',
                           'unsupported claims and uncertainty', 'structural validity'],
            'coverage_rule': 'Omitting a target fact or abstaining is a coverage miss, not a correct answer',
            'comparison': ('Compare both arms per case; retain every result and failure' if presence_zero
                           else 'Compare both arms and both repeats per case; retain every result and failure'),
            'non_regression': ('Report losses of correctly documented baseline behavior; '
                               'an improvement claim requires preserving known successes and target coverage'),
            'decision_rule': {
                'comparison_requires': (
                    'All scheduled documents completed and independently reviewed against source'),
                'promising_requires_all': [
                    'Fewer documents containing material false claims than baseline',
                    'No loss of previously correct required behavior',
                    'No reduction in target coverage',
                ],
                'promotion_requires_all': [
                    'All required behavior is covered correctly',
                    'Zero material errors',
                    'Successful evaluation on broader representative cases',
                ],
                'this_trial_can_grant_production_qualification': False,
            },
            'limits': 'Any apparent improvement remains a small synthetic pilot, not production qualification',
        },
    }
    if presence_zero:
        protocol['hypothesis'] = (
            'Test whether zero presence penalty improves source-grounded documentation. '
            'This single-repeat diagnostic does not establish causation or general accuracy.')
        protocol['review_policy']['decision_rule']['further_validation_requires_all'] = [
            'All required behavior is covered correctly',
            'Zero material errors',
            'No loss of previously correct required behavior',
        ]
    return protocol


def _prepare(output: Path, budget: Budget, variant: str) -> tuple[Run, dict]:
    protocol = frozen_trial(variant)
    output.mkdir(parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, protocol['maximum_http_attempts'])
    run = Run(REMOTE_PROFILE, output, budget, pacing=0)
    run.report.update(
        scope=protocol['scope'], variant=variant, timeout_seconds=protocol['timeout_seconds'],
        rates_verified_on='2026-10-22',
        quality_verdict='unreviewed', local_q4_qualified=False, hardware_qualified=False,
        trials=[], expected_documents=len(protocol['schedule']),
        not_qualified=['general factual correctness', 'independent held-out validation',
                       'local Q4 artifacts', 'M5 or Metal operation', 'embedding quality',
                       'library, Ask/Search, dependencies, clews, Spools and UI workflows'],
    )
    sources = (
        Path(__file__), ROOT / 'tests/test_mac_quality_trial.py',
        ROOT / 'tests/live_mac_model_probe.py', ROOT / 'tests/live_model_smoke.py',
        ROOT / 'tests/source_verification_cases.py', ROOT / 'tests/model_smoke_cases.py',
        ROOT / 'docgen/llm/deepinfra.py', ROOT / 'docgen/llm/local.py',
        ROOT / 'docgen/validator.py', ROOT / 'model_config.py',
        *(ARCHIVE / f'{number}.json' for number in ('001', '003', '005')),
    )
    hashes = {}
    for source in sources:
        relative = source.relative_to(ROOT)
        snapshot = output / 'provenance' / relative
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        snapshot.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    for name in protocol['cases']:
        (output / 'sources' / f'{name}.py').write_text(CASES_BY_NAME[name].source)
    protocol.update(source_hashes=hashes, budget_usd=budget.limit,
                    maximum_http_attempts=budget.max_requests)
    save_json(output / 'protocol.json', protocol)
    run.report['protocol_sha256'] = hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest()
    run.save()
    return run, protocol


async def execute(output: Path, budget: Budget, *, api_key: str, variant: str = 'source-scope',
                  transport_factory=httpx.AsyncHTTPTransport) -> dict:
    if budget.limit > 1:
        raise ValueError('This bounded trial permits at most $1 in request accounting')
    if not api_key:
        raise ValueError('DeepInfra credential is missing')
    run, protocol = _prepare(output, budget, variant)
    run.report['transport_mode'] = 'live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated'
    started, active = time.monotonic(), None
    try:
        async with httpx.AsyncClient(
            base_url=REMOTE_PROFILE.base_url, timeout=protocol['timeout_seconds'],
            trust_env=False, follow_redirects=False,
            headers={'Authorization': f'Bearer {api_key}'},
            transport=RecordedTransport(run, 'chat', transport_factory()),
        ) as client:
            for scheduled in protocol['schedule']:
                active = {**scheduled, 'status': 'running', 'first_http_attempt': len(run.report['requests']) + 1}
                run.report['trials'].append(active)
                run.save()
                payload = protocol['cases'][scheduled['case']]['requests'][scheduled['arm']]
                response = await client.post('/chat/completions', json=payload)
                response.raise_for_status()
                content = response_content(response.json())
                # The existing recorder redacts credentials from raw provider output too.
                content = content.replace(api_key, '<redacted>')
                (output / 'documents' / scheduled['file']).write_text(content)
                validation = ContentValidator().validate(content, 'qa', scheduled['case'])
                run.report['documents'].append({
                    **scheduled,
                    'sha256': hashlib.sha256(content.encode()).hexdigest(),
                    'structurally_valid': validation.is_valid,
                    'issues': [{'level': i.level, 'code': i.code} for i in validation.issues],
                })
                active.update(status='completed', last_http_attempt=len(run.report['requests']))
                run.save()
        run.report['status'] = 'completed_pending_review'
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        if active is not None:
            active['status'] = 'cancelled'
        raise
    except Exception as exc:  # noqa: BLE001 — preserve all failed trial evidence without declaring quality.
        error = f'{type(exc).__name__}: {exc}'.replace(api_key, '<redacted>')
        run.report.update(status='failed', error=error)
        if active is not None:
            active.update(status='failed', error=error, last_http_attempt=len(run.report['requests']))
    finally:
        run.report['seconds'] = time.monotonic() - started
        run.save()
    return run.report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, default=1)
    parser.add_argument('--variant', choices=VARIANT_LIMITS, default='source-scope')
    args = parser.parse_args()
    key = os.environ.get('DEEPINFRA_API_KEY') or dotenv_values(ROOT / '.env').get('DEEPINFRA_API_KEY')
    if not key:
        parser.error('DEEPINFRA_API_KEY is missing')
    budget = Budget(args.budget_usd, max_requests=VARIANT_LIMITS[args.variant][1])
    if budget.limit > 1:
        parser.error('This trial is limited to $1')
    report = asyncio.run(execute(args.output.resolve(), budget, api_key=key, variant=args.variant))
    print(json.dumps({k: v for k, v in report.items() if k not in ('requests', 'trials')}, indent=2))
    return 0 if report['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
