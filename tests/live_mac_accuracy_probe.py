"""Opt-in hosted 27B thinking diagnostic: twelve explanation/QA generations.

Uses production prompt construction and the existing bounded DeepInfra recorder.
Only final responses and numeric usage are archived; private reasoning is omitted.
This is not local-artifact, M5, embedding, Ask or production qualification.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import replace
from pathlib import Path

import httpx
from dotenv import dotenv_values

from docgen.generator import DocGenerator
from docgen.llm.local import parse_chat_response
from docgen.validator import ContentValidator
from tests.live_model_smoke import PROFILES, ROOT, Budget, RecordedTransport, Run, save_json
from tests.source_verification_cases import CASES_BY_NAME, observe_case

MODEL = 'Qwen/Qwen3.8-27B'
PROFILE = replace(PROFILES['qwen'], model=MODEL, input_rate=0.20,
                  reserve_input_rate=0.20, output_rate=2.50)
CASES = ('cache', 'guarded_cache', 'token_expiry')
DOC_TYPES = ('explanation', 'qa')
CONTEXT_WINDOW = 32768
MAX_OUTPUT = 8192
MAX_ATTEMPTS = 18
TIMEOUT = 600
SAMPLING = {
    'reasoning_effort': 'xhigh', 'chat_template_kwargs': {'enable_thinking': True},
    'temperature': 1.0, 'top_p': 0.95, 'top_k': 20, 'min_p': 0.0,
    'presence_penalty': 0.0, 'repetition_penalty': 1.0,
}
API_REFERENCE = 'https://docs.deepinfra.com/api-reference/chat-completions/openai-chat-completions'
NUMERIC_USAGE = (
    'prompt_tokens', 'completion_tokens', 'total_tokens', 'reasoning_tokens', 'estimated_cost',
    'cache_creation_input_tokens', 'cache_read_input_tokens',
)
TOKEN_DETAILS = ('cached_tokens', 'audio_tokens', 'reasoning_tokens',
                 'accepted_prediction_tokens', 'rejected_prediction_tokens')


def request_for(system: str, user: str) -> dict:
    payload = {
        'model': MODEL, 'messages': [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user}],
        'max_tokens': MAX_OUTPUT, 'stream': False, **json.loads(json.dumps(SAMPLING)),
    }
    validate_request(payload)
    return payload


def validate_request(payload: dict) -> int:
    if (payload.get('model') != MODEL or payload.get('max_tokens') != MAX_OUTPUT
            or payload.get('stream') is not False
            or any(payload.get(key) != value for key, value in SAMPLING.items())):
        raise ValueError('Request differs from the frozen model, thinking or sampling contract')
    messages = payload.get('messages')
    if (not isinstance(messages, list) or len(messages) != 2
            or [m.get('role') for m in messages] != ['system', 'user']
            or any(not isinstance(m.get('content'), str) or not m['content'].strip() for m in messages)):
        raise ValueError('Expected nonempty production system/user messages')
    # Deliberately conservative for the tiny frozen inputs; no hosted tokenizer is invoked.
    input_envelope = sum(len(m['content'].encode()) for m in messages) + 1024
    if input_envelope + MAX_OUTPUT + 64 > CONTEXT_WINDOW:
        raise ValueError('Request exceeds the fixed 32K context admission envelope')
    return input_envelope


def _usage(data: object) -> dict:
    if not isinstance(data, dict):
        return {}

    def numeric(value):
        return type(value) in (int, float) and math.isfinite(value) and value >= 0

    safe = {key: data[key] for key in NUMERIC_USAGE if key in data and numeric(data[key])}
    for key in ('prompt_tokens_details', 'completion_tokens_details'):
        details = data.get(key)
        if isinstance(details, dict):
            safe[key] = {name: details[name] for name in TOKEN_DETAILS
                         if name in details and numeric(details[name])}
    return safe


def final_only_response(raw: bytes, status: int) -> tuple[dict, dict]:
    """Keep final content/usage, never unknown fields or private reasoning text."""
    archive = {'format': 'final-only-provider-response-v1',
               'original_sha256': hashlib.sha256(raw).hexdigest(), 'original_bytes': len(raw),
               'reasoning_present': False, 'reasoning_tokens': None}
    try:
        data = json.loads(raw)
    except (ValueError, UnicodeError):
        archive['invalid'] = 'malformed-json'
        return {'diagnostic': 'Provider body omitted: malformed JSON'}, archive
    if not isinstance(data, dict) or status != 200:
        archive['invalid'] = 'non-success-or-non-object'
        return {'diagnostic': 'Provider body omitted', 'http_status': status}, archive
    safe = {key: data[key] for key in ('id', 'object', 'created', 'model', 'system_fingerprint')
            if key in data and type(data[key]) in (str, int)}
    safe['usage'] = _usage(data.get('usage'))
    details = safe['usage'].get('completion_tokens_details', {})
    archive['reasoning_tokens'] = details.get('reasoning_tokens', safe['usage'].get('reasoning_tokens'))
    safe['choices'] = []
    choices = data.get('choices')
    if not isinstance(choices, list):
        archive['invalid'] = 'malformed-choices'
        return safe, archive
    for choice in choices:
        if not isinstance(choice, dict) or not isinstance(choice.get('message'), dict):
            safe['choices'].append({})
            continue
        message = choice['message']
        archive['reasoning_present'] |= any(bool(message.get(key))
                                            for key in ('reasoning', 'reasoning_content', 'reasoning_details'))
        content = message.get('content')
        if (not isinstance(content, str) or '<think>' in content or '</think>' in content
                or message.get('tool_calls') or message.get('function_call')
                or message.get('role') != 'assistant'):
            content = None
            archive['invalid'] = 'non-final-content'
        finish = choice.get('finish_reason')
        safe['choices'].append({
            'index': choice.get('index') if type(choice.get('index')) is int else None,
            'finish_reason': finish if finish in ('stop', 'length', 'tool_calls', 'content_filter') else 'unsupported',
            'message': {'role': 'assistant', 'content': content},
        })
    return safe, archive


class FinalOnlyTransport(httpx.AsyncBaseTransport):
    """Transform before RecordedTransport can archive or inspect provider output."""

    def __init__(self, run: Run, inner: httpx.AsyncBaseTransport):
        self.run, self.inner = run, inner

    async def aclose(self) -> None:
        await self.inner.aclose()

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        try:
            raw = await response.aread()
            data, archive = final_only_response(raw, response.status_code)
            self.run.report['requests'][-1]['response_archive'] = archive
            headers = {'retry-after': response.headers['retry-after']} if 'retry-after' in response.headers else {}
            return httpx.Response(response.status_code, json=data, headers=headers, request=request)
        finally:
            await response.aclose()


def validate_final(data: dict) -> tuple[str, dict]:
    if data.get('model') != MODEL:
        raise ValueError('Response does not identify the requested 27B model')
    content, counts = parse_chat_response(data)
    if any(type(counts[key]) is not int or counts[key] < 0 for key in ('prompt_tokens', 'completion_tokens')):
        raise ValueError('Complete token usage is required to verify the fixed output/context limits')
    if (counts['completion_tokens'] > MAX_OUTPUT
            or counts['prompt_tokens'] + MAX_OUTPUT + 64 > CONTEXT_WINDOW):
        raise ValueError('Returned usage exceeds the fixed output/context limits')
    return content, counts


async def prepare_run(output: Path, budget: Budget) -> tuple[Run, dict]:
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, MAX_ATTEMPTS)
    run = Run(PROFILE, output, budget, pacing=0)
    generator, cases = DocGenerator(), {}
    for name in CASES:
        case = CASES_BY_NAME[name]
        path = output / 'sources' / f'{name}.py'
        path.write_text(case.source)
        metadata = generator.analyzer.analyze_file(path)
        bundles = await generator.build_prompts_for_module(metadata, DOC_TYPES)
        requests = {bundle.doc_type: request_for(bundle.system_prompt, bundle.user_prompt) for bundle in bundles}
        cases[name] = {
            'source_sha256': hashlib.sha256(case.source.encode()).hexdigest(),
            'oracle': observe_case(case), 'required_behavior': case.correct,
            'review_question': case.question, 'requests': requests,
            'input_admission_envelopes': {kind: validate_request(payload) for kind, payload in requests.items()},
        }
    schedule = [
        {'case': name, 'doc_type': kind, 'repeat': repeat, 'file': f'{name}-{kind}-r{repeat}.md'}
        for repeat in (1, 2) for name in CASES
        for kind in (DOC_TYPES if repeat == 1 else tuple(reversed(DOC_TYPES)))
    ]
    source_paths = (
        Path(__file__), ROOT / 'tests/test_live_mac_accuracy_probe.py',
        ROOT / 'tests/live_model_smoke.py', ROOT / 'tests/source_verification_cases.py',
        ROOT / 'tests/model_smoke_cases.py', ROOT / 'docgen/generator.py',
        ROOT / 'docgen/prompts.py', ROOT / 'docgen/_legacy_analyzer.py',
        ROOT / 'docgen/_legacy_metadata.py', ROOT / 'docgen/llm/local.py',
        ROOT / 'docgen/validator.py', ROOT / 'model_config.py',
    )
    hashes = {}
    for source in source_paths:
        relative, content = source.relative_to(ROOT), source.read_bytes()
        destination = output / 'provenance' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        hashes[str(relative)] = hashlib.sha256(content).hexdigest()
    protocol = {
        'scope': 'Hosted 27B thinking generation diagnostic; no application or hardware qualification',
        'model': MODEL, 'hosted_artifact_identity': 'Provider deployment; immutable weights/precision not pinned',
        'cases': cases, 'schedule': schedule, 'repetitions': 2, 'concurrency': 1,
        'source_hashes': hashes, 'budget_usd': budget.limit, 'maximum_http_attempts': budget.max_requests,
        'timeout_seconds': TIMEOUT, 'context_window': CONTEXT_WINDOW, 'max_total_output_tokens': MAX_OUTPUT,
        'context_admission': 'Conservative UTF-8 byte envelope plus 1024 overhead; not exact tokenizer preflight',
        'output_limit': '8192 total completion tokens including thinking; truncation fails, no automatic increase',
        'prompt_origin': 'Unmodified production DocGenerator.build_prompts_for_module, explanation and QA',
        'case_limit': 'Three previously studied synthetic regression cases; not independent holdout',
        'local_artifact_qualified': False, 'hardware_qualified': False,
        'hosted_mapping': {
            'repeat_penalty': 'Native repeat_penalty=1 maps to hosted repetition_penalty=1',
            'reasoning_format': 'Native deepseek response-format option omitted from hosted request',
            'thinking': 'Explicit xhigh effort and enable_thinking=true requested',
            'limit': 'Same semantic intent, not exact wire parity or proof of identical sampling/runtime behavior',
            'reference': API_REFERENCE, 'verified_on': '2026-10-22',
        },
        'thinking_evidence': 'Record reasoning-field presence and provider token counts; absent counts remain unknown',
        'response_archive_policy': (
            'Transformed final-only responses, numeric usage and safe metadata; original body SHA-256/size retained. '
            'No private reasoning, tool arguments, unknown fields or malformed body text is archived.'),
        'retry_policy': 'Only HTTP 429 may retry twice within the attempt budget; no content or timeout retries',
        'pricing_source': 'https://deepinfra.com/Qwen/Qwen3.8-27B/api',
        'review_policy': {
            'method': 'Separate source review of every complete document; omissions are coverage misses',
            'required': [
                'All twelve documents completed and independently reviewed against source',
                'All required source behavior covered correctly',
                'Zero material false claims or contradictions',
                'No loss of previously correct required behavior',
            ],
            'scope': 'Eligibility for broader evaluation only; no automatic factual pass',
            'production_qualification': False,
        },
    }
    save_json(output / 'protocol.json', protocol)
    run.report.update(
        scope=protocol['scope'], quality_verdict='unreviewed', embedding_model=None,
        rates_verified_on='2026-10-22', trials=[], expected_documents=len(schedule),
        protocol_sha256=hashlib.sha256((output / 'protocol.json').read_bytes()).hexdigest(),
        local_artifact_qualified=False, hardware_qualified=False,
        not_qualified=['local quantized artifacts', 'M5/Metal memory and latency', 'embedding quality',
                       'Ask/Search and UI workflows', 'general factual correctness'],
    )
    run.save()
    return run, protocol


async def execute(output: Path, budget: Budget, *, api_key: str,
                  transport_factory=httpx.AsyncHTTPTransport) -> dict:
    if budget.limit > 1:
        raise ValueError('This diagnostic permits at most $1 in request accounting')
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
                active = {**scheduled, 'status': 'running', 'first_http_attempt': len(run.report['requests']) + 1}
                run.report['trials'].append(active)
                run.save()
                payload = protocol['cases'][scheduled['case']]['requests'][scheduled['doc_type']]
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
    except Exception as exc:  # noqa: BLE001 — terminal evidence, never retry or discard a paid failure.
        # Provider exception messages may contain response text; retain only the safe type.
        run.report.update(status='failed', error_type=type(exc).__name__)
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
    print(json.dumps({k: v for k, v in report.items() if k not in ('requests', 'trials')}, indent=2))
    return 0 if report['status'] == 'completed_pending_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
