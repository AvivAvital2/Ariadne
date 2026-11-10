"""Offline capture of actual production prompts and limits for isolated call replay.

No provider is called. Existing synthetic workflow fixtures supply library/SCIP
state and scripted prior completions; this is not an end-to-end quality run.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sqlite3
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from docgen.llm.local import build_chat_request
from docgen.local_pipeline import plan_catalog_descriptions, plan_themes
from llm import chat_complete as shared_complete
from local_models import mac_accuracy_profile
from model_config import resolve_models
from tests import test_ask_synthesis_config as cascade_fixture
from tests import test_compact_path as compact_fixture
from tests._scoped_config_fixture import install_test_config

ROOT = Path(__file__).resolve().parents[1]


class Capture:
    def __init__(self):
        self.models = resolve_models(mac_accuracy_profile())
        self.settings = self.models.chat
        self.cases = []

    def add(self, name, messages, max_tokens, *, timeout=None, expected=None):
        system = '\n\n'.join(m['content'] for m in messages if m['role'] == 'system')
        user = '\n\n'.join(m['content'] for m in messages if m['role'] == 'user')
        native = build_chat_request(self.settings, system, user, max_tokens)
        hosted = json.loads(json.dumps(native))
        hosted['model'] = 'Qwen/Qwen3.8-27B'
        hosted['repetition_penalty'] = hosted.pop('repeat_penalty')
        hosted.pop('reasoning_format')
        self.cases.append({
            'name': name, 'system_prompt': system, 'user_prompt': user,
            'max_tokens': max_tokens,
            'timeout_seconds': self.settings.timeout_seconds if timeout is None else timeout,
            'native_request': native, 'hosted_request': hosted,
            'expected': expected or {},
        })

    async def complete(self, name, messages, *, reply, expected, phase, max_tokens,
                       timeout=None, usage_sink=None):
        recorder = self

        class Provider:
            def __init__(self):
                self.last_usage = {'prompt_tokens': 17, 'completion_tokens': 8}

            async def call(self, system, user, *, max_tokens, timeout):
                recorder.add(name, [{'role': 'system', 'content': system},
                                    {'role': 'user', 'content': user}], max_tokens,
                             timeout=timeout, expected=expected)
                return reply

        context = SimpleNamespace(
            local=True, settings=self.models, chat=Provider(),
            validate_producer=lambda **_kwargs: None)
        return await shared_complete(
            messages=messages, phase=phase, max_tokens=max_tokens, timeout=timeout,
            usage_sink=usage_sink, context=context)


async def _compact(recorder, root):
    fixture = compact_fixture.service.__wrapped__(root)
    service = next(fixture)
    try:
        replies = {
            'scip-obligation-plan': 'C1: prove that pkg.Writer.write calls pkg.Sink.flush',
            'scip-route-family-select': compact_fixture._first_family_reply,
            'completion': 'The write path calls flush at core/writer.scala:30.',
        }
        names = {'scip-obligation-plan': 'compact-plan',
                 'scip-route-family-select': 'compact-family', 'completion': 'compact-answer'}
        expected = {
            'compact-plan': {'format': 'C<n>: <requirement> lines',
                             'target': 'Plan the named Writer.write to Sink.flush call chain.'},
            'compact-family': {'format': 'O1: F<id>',
                               'target': 'Select a listed family supporting Writer.write to Sink.flush.'},
            'compact-answer': {'target': 'Explain only the supplied compiler chain with supplied citations.',
                               'limit': 'Synthetic source consists of fixture line placeholders.'},
        }

        async def complete(*, messages, **kwargs):
            phase = kwargs.get('phase', 'completion')
            name = names[phase]
            reply = replies[phase]
            reply = reply(messages[-1]['content']) if callable(reply) else reply
            return await recorder.complete(
                name, messages, reply=reply, expected=expected[name],
                phase=phase, max_tokens=kwargs['max_tokens'], usage_sink=kwargs.get('usage_sink'))

        await service._ask_compact(
            'How does pkg.Writer.write call pkg.Sink.flush?', source=compact_fixture.SOURCE,
            notes=(), diagnostics={}, ask_chat=complete, trace=lambda *_args, **_kwargs: None,
            phase_timings={})
    finally:
        fixture.close()


async def _cascade(recorder, root):
    with pytest.MonkeyPatch.context() as patches:
        install_test_config(patches, root, 'test')
        fixture = cascade_fixture.library.__wrapped__(root)
        library = next(fixture)
        try:
            service = cascade_fixture.service.__wrapped__(library)
            cascade_fixture._seed_chain(library)
            patches.setenv('OPENAI_API_KEY', 'offline-fixture-only')
            patches.setattr('library.chain_menu.definition_body_selection_requires_llm', lambda *_a: True)
            names = {'scip-component-select': 'cascade-component',
                     'scip-exact-route-select': 'cascade-route', 'scip-body-select': 'cascade-body',
                     'completion': 'cascade-answer'}
            replies = {'scip-component-select': '1', 'scip-exact-route-select': 'R1',
                       'scip-body-select': 'B1', 'completion': 'The widget flushes at w.py:20.'}

            async def complete(messages, *, phase='completion', max_tokens=2048,
                               usage_sink=None, timeout=None, **_kwargs):
                return await recorder.complete(
                    names[phase], messages, reply=replies[phase], phase=phase,
                    max_tokens=max_tokens, timeout=timeout, usage_sink=usage_sink,
                    expected={'scripted_fixture_reply': replies[phase],
                              'target': 'Select listed evidence for the widget flush chain; '
                                        'final prose must use only supplied evidence.'})

            patches.setattr('llm.chat_complete', complete)
            await service.ask('How does the widget subsystem work?', source='test')
        finally:
            fixture.close()


def _catalog_and_theme(recorder):
    docs = {
        'widget-run': SimpleNamespace(
            id='widget-run', title='Widget.run', content='Runs the widget.',
            source_files=['w.py'], source_name='test', updated_at='frozen-fixture',
            metadata={'kind': 'element', 'subtype': 'method', 'qualified_name': 'w.Widget.run',
                      'parent_qualified_name': 'w.Widget', 'signature': 'run(self, rows)',
                      'file': 'w.py', 'location': {'line_start': 5, 'line_end': 13}}),
        'widget-flush': SimpleNamespace(
            id='widget-flush', title='Widget.flush', content='Flushes the batch to storage.',
            source_files=['w.py'], source_name='test', updated_at='frozen-fixture',
            metadata={'description': 'Flushes the batch to storage.'}),
        'theme-doc': SimpleNamespace(id='theme-doc', title='Widget batching', metadata={},
                                     source_files=[], source_name=None, updated_at='frozen-fixture'),
    }
    library = SimpleNamespace(
        list_documents=lambda _kind: [docs['widget-run']],
        get_dirty_themes=lambda: ['widget-theme'],
        get_theme=lambda _key: SimpleNamespace(resolution=1.0, doc_id='theme-doc'),
        get_theme_members=lambda _key: [('widget-run', 1.0), ('widget-flush', 1.0)],
        get_document=docs.get,
    )
    catalog, _, _ = plan_catalog_descriptions(library, source_name='test')
    for name, spec in [('catalog', catalog[0]), ('theme', plan_themes(library)[0])]:
        request = spec.request
        recorder.add(name, [{'role': 'system', 'content': request.system_prompt},
                            {'role': 'user', 'content': request.user_prompt}], request.max_tokens,
                     expected={'target': 'Describe only the supplied metadata/member summaries.',
                               'limit': 'Catalog metadata has no implementation body; do not infer unsupported details.'})


def _trace(recorder):
    from docgen.trace_flow_llm_bridge import build_llm_bridge
    from library.scip import init_scip_schema
    from tests.test_trace_flow_llm_bridge import _insert_symbol

    class Chat:
        def call_sync(self, system, user, *, max_tokens, timeout):
            recorder.add('trace', [{'role': 'system', 'content': system},
                                   {'role': 'user', 'content': user}], max_tokens, timeout=timeout,
                         expected={'format': 'JSON object with symbol_query and reasoning, or null',
                                   'target': 'w.Widget.flush is the documented next callee.'})
            return '{"symbol_query": "w.Widget.flush", "reasoning": "The provided documentation names it."}'

    context = SimpleNamespace(local=True, chat=Chat(), validate=lambda **_kwargs: None)
    with sqlite3.connect(':memory:') as conn, pytest.MonkeyPatch.context() as patches:
        init_scip_schema(conn)
        _insert_symbol(conn, canonical_id='w.Widget.run', file='w.py', line_start=5)
        _insert_symbol(conn, canonical_id='w.Widget.flush', file='w.py', line_start=20)
        patches.setattr('inference.current_context', lambda: context)
        bridge = build_llm_bridge(doc_search=lambda _connection, _name: [{
            'title': 'Widget flow', 'content': 'w.Widget.run calls w.Widget.flush to flush the batch.'}])
        result = bridge('w.Widget.run', conn)
        if result[0] != 'w.Widget.flush':
            raise RuntimeError('Trace fixture did not reach the production completion path')


async def capture(output: Path) -> dict:
    await asyncio.to_thread(output.mkdir, parents=True, exist_ok=False)
    recorder = Capture()
    with tempfile.TemporaryDirectory(prefix='ariadne-short-call-capture-') as directory:
        root = Path(directory)
        for subdir in ('compact', 'cascade'):
            (root / subdir).mkdir()
        await _compact(recorder, root / 'compact')
        await _cascade(recorder, root / 'cascade')
        _catalog_and_theme(recorder)
        _trace(recorder)
    paths = (
        'tests/capture_mac_short_calls.py', 'tests/test_mac_short_call_capture.py',
        'tests/test_compact_path.py', 'tests/test_ask_synthesis_config.py',
        'tests/test_trace_flow_llm_bridge.py', 'tests/_scoped_config_fixture.py',
        'ariadne_mcp/service_analysis.py', 'library/chain_answer.py', 'library/chain_menu.py',
        'library/route_families.py', 'library/chain_story.py', 'docgen/local_pipeline.py',
        'docgen/themes.py', 'docgen/prompts.py', 'docgen/catalog_describer.py',
        'docgen/trace_flow_llm_bridge.py', 'docgen/llm/local.py', 'llm.py', 'local_models.py', 'model_config.py',
    )
    hashes = {}
    for relative in paths:
        data = (ROOT / relative).read_bytes()
        destination = output / 'provenance' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(data)
        hashes[relative] = hashlib.sha256(data).hexdigest()
    protocol = {
        'scope': 'Offline production request capture for individual hosted short-call diagnostics',
        'cases': recorder.cases, 'source_hashes': hashes, 'network_calls': 0,
        'end_to_end_quality_qualified': False,
        'downstream_input_limit': 'Ask downstream prompts use scripted earlier fixture replies, '
                                  'not a live end-to-end model run. Synthetic SCIP fixtures do not '
                                  'qualify representative code comprehension or depends_on behavior.',
        'timeout_origin': 'Selected local profile600s unless caller explicitly overrides; trace uses30s.',
        'hosted_mapping': 'Exact production messages/caps; native repeat_penalty maps to repetition_penalty; '
                          'native reasoning_format is omitted. Hosted weights/precision are not localQ4.',
    }
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    protocol = asyncio.run(capture(args.output))
    print(f'Captured {len(protocol["cases"])} production requests without network calls: {args.output / "protocol.json"}')


if __name__ == '__main__':
    main()
