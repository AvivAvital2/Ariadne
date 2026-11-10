"""Opt-in bounded replay of captured application calls against hosted Qwen27B.

This tests individual prompts and their actual limits, not an end-to-end native
workflow. Expected answers remain in the protocol, never in transmitted messages.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import re
import time
from pathlib import Path

import httpx
from dotenv import dotenv_values

from docgen.llm.local import build_chat_request, parse_chat_response
from local_models import mac_accuracy_profile
from model_config import resolve_models
from tests.live_mac_accuracy_probe import CONTEXT_WINDOW, MODEL, PROFILE, FinalOnlyTransport
from tests.live_model_smoke import ROOT, Budget, RecordedTransport, Run, save_json

MAX_ATTEMPTS = 32


def frozen_cases(manifest: dict) -> list[dict]:
    cases = manifest.get('cases')
    if not isinstance(cases, list) or not 1 <= len(cases) <= 16:
        raise ValueError('Expected one to sixteen captured cases')
    settings = resolve_models(mac_accuracy_profile()).chat
    frozen, names = [], set()
    for case in cases:
        name = case.get('name')
        cap, timeout = case.get('max_tokens'), case.get('timeout_seconds')
        if (not isinstance(name, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name)
                or name in names or type(cap) is not int or not 0 < cap <= 8192
                or type(timeout) not in (int, float) or not math.isfinite(timeout)
                or not 0 < timeout <= 600
                or not isinstance(case.get('system_prompt'), str)
                or not isinstance(case.get('user_prompt'), str) or not case['user_prompt'].strip()):
            raise ValueError('Invalid captured case identity, input or limit')
        names.add(name)
        native = build_chat_request(settings, case['system_prompt'], case['user_prompt'], cap)
        hosted = dict(native)
        hosted.pop('reasoning_format')
        hosted['repetition_penalty'] = hosted.pop('repeat_penalty')
        if (('native_request' in case and case['native_request'] != native)
                or ('hosted_request' in case and case['hosted_request'] != hosted)):
            raise ValueError('Captured payload differs from current production request construction')
        envelope = sum(len(message['content'].encode()) for message in hosted['messages']) + 1024
        if envelope + hosted['max_tokens'] + 64 > CONTEXT_WINDOW:
            raise ValueError('Conservative context admission exceeded')
        frozen.append({**case, 'native_request': native, 'hosted_request': hosted,
                       'input_byte_envelope': envelope})
    return json.loads(json.dumps(frozen))


def validated_content(data: dict, cap: int) -> tuple[str, dict]:
    if data.get('model') != MODEL:
        raise ValueError('Unexpected response model')
    content, usage = parse_chat_response(data)
    if (any(type(n) is not int or n < 0 for n in usage.values())
            or usage['completion_tokens'] > cap
            or usage['prompt_tokens'] + cap + 64 > CONTEXT_WINDOW):
        raise ValueError('Missing or excessive token usage')
    return content, usage


async def execute(manifest: dict, output: Path, budget: Budget, *, api_key: str,
                  transport_factory=httpx.AsyncHTTPTransport) -> dict:
    if budget.limit > 1 or not api_key:
        raise ValueError('At most $1 per diagnostic and a provider credential are required')
    cases = frozen_cases(manifest)
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, MAX_ATTEMPTS)
    run = Run(PROFILE, output, budget, pacing=0)
    hashes = {}
    for name in ('tests/live_mac_call_probe.py', 'tests/test_live_mac_call_probe.py',
                 'tests/live_mac_accuracy_probe.py', 'tests/live_model_smoke.py',
                 'docgen/llm/local.py', 'local_models.py', 'model_config.py'):
        source = ROOT / name
        target = output / 'provenance' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        content = source.read_bytes()
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    protocol = {
        'scope': 'Individual production call replay; upstream fixture responses may be scripted',
        'model': MODEL, 'cases': cases, 'capture_metadata': {k: v for k, v in manifest.items() if k != 'cases'},
        'budget_usd': budget.limit, 'maximum_http_attempts': budget.max_requests,
        'source_hashes': hashes, 'context_window': CONTEXT_WINDOW,
        'mapping': 'Native repeat_penalty maps to repetition_penalty; native reasoning_format omitted',
        'response_archive': 'Final-only transformed response; private reasoning omitted, original body hash retained',
        'context_admission': 'Conservative UTF-8 byte envelope; not actual native tokenization',
        'quality_review': 'Independent review required; normal completion is not correctness',
        'retry_policy': 'Only bounded HTTP429 retries; never retry content failures or timeouts',
        'local_q4_qualified': False, 'hardware_qualified': False,
    }
    save_json(output / 'protocol.json', protocol)
    run.report.update(scope=protocol['scope'], status='running', quality_verdict='unreviewed',
                      rates_verified_on='2026-10-22', cases=[],
                      protocol_sha256=hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
                      transport_mode='live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated')
    run.save()
    started = time.monotonic()
    try:
        async with httpx.AsyncClient(
            base_url=PROFILE.base_url, headers={'Authorization': f'Bearer {api_key}'},
            trust_env=False, follow_redirects=False,
            transport=RecordedTransport(run, 'chat', FinalOnlyTransport(run, transport_factory())),
        ) as client:
            for case in cases:
                row = {'name': case['name'], 'status': 'running',
                       'first_http_attempt': len(run.report['requests']) + 1}
                run.report['cases'].append(row)
                run.save()
                try:
                    response = await client.post('/chat/completions', json=case['hosted_request'],
                                                 timeout=case['timeout_seconds'])
                    response.raise_for_status()
                    content, usage = validated_content(response.json(), case['hosted_request']['max_tokens'])
                    content = content.replace(api_key, '<redacted>')
                    (output / 'documents' / f'{case["name"]}.txt').write_text(content)
                    row.update(status='completed', content=content, usage=usage, finish_reason='stop')
                except asyncio.CancelledError:
                    row['status'] = 'cancelled'
                    raise
                except Exception as exc:  # noqa: BLE001 - retain paid failures; no content retry.
                    row.update(status='failed', error_type=type(exc).__name__)
                    if isinstance(exc, RuntimeError) and str(exc) == 'Live request or spending limit reached; no request sent':
                        run.report['status'] = 'budget_exhausted'
                        break
                finally:
                    row['last_http_attempt'] = len(run.report['requests'])
                    if row['last_http_attempt'] >= row['first_http_attempt']:
                        record = run.report['requests'][-1]
                        row['usage'] = record.get('usage', {})
                        response_path = output / 'responses' / f'{row["last_http_attempt"]:03}.json'
                        if response_path.exists():
                            data = json.loads(response_path.read_text())
                            choices = data.get('choices') if isinstance(data, dict) else None
                            if isinstance(choices, list) and choices and isinstance(choices[0], dict):
                                row['finish_reason'] = choices[0].get('finish_reason')
                    run.save()
        if run.report['status'] != 'budget_exhausted':
            run.report['status'] = ('completed_pending_review' if all(
                row['status'] == 'completed' for row in run.report['cases']) else 'completed_with_failures')
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        raise
    finally:
        run.report['seconds'] = time.monotonic() - started
        run.save()
    return run.report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--budget-usd', type=float, default=.5)
    args = parser.parse_args()
    key = os.environ.get('DEEPINFRA_API_KEY') or dotenv_values(ROOT / '.env').get('DEEPINFRA_API_KEY')
    if not key:
        parser.error('DEEPINFRA_API_KEY is missing')
    result = asyncio.run(execute(json.loads(args.manifest.read_text()), args.output.resolve(),
                                Budget(args.budget_usd, MAX_ATTEMPTS), api_key=key))
    print(json.dumps({key: result[key] for key in ('status', 'seconds', 'accounted_upper_bound_usd')}))
    return 0 if result['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
