"""Bounded remote accuracy diagnostic over real repository excerpts and dependencies.

Run explicitly with ``python -m tests.live_remote_accuracy_workflow --output NEW_DIR``.
Production generation, storage, retrieval and Ask consume actual upstream model replies.
The source-backed graph is seeded: this does not qualify an indexer or native Q4/M5.
"""

from __future__ import annotations

import argparse
import ast
import asyncio
import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import numpy as np
from dotenv import dotenv_values

import config as config_module
import llm
from ariadne_mcp.service import AriadneService
from config import Config
from docgen.catalog_describer import describe_source_elements
from docgen.catalog_writer import sync_file_catalog
from docgen.generator import DocGenerator, GeneratorConfig
from docgen.validator import ContentValidator
from inference import InferenceContext
from library.clew_embeddings import embed_clews
from library.clews import add_clew
from local_models import mac_accuracy_profile
from model_config import resolve_models
from tests.live_mac_accuracy_probe import PROFILE
from tests.live_mac_hosted_workflow import (
    BoundedTransport,
    HostedThinkingProvider,
    WorkflowAbort,
    check_vectors,
    require,
)
from tests.live_model_smoke import (
    ROOT,
    Budget,
    RecordedTransport,
    Run,
    configuration,
    save_json,
    snapshot,
)
from writer import LibraryWriter

MAX_ATTEMPTS = 80
ORACLE_SENTINEL = 'REVIEW_ONLY_REMOTE_ACCURACY_ORACLE'
QUESTIONS = (
    {'id': 'consumer-chain', 'source': 'consumer', 'pipeline': 'cascade',
     'question': 'How does writer.LibraryWriter.update_document_embedding use '
                 'embedding.doc_embedding_text when the writer has no bound inference context?'},
    {'id': 'shared-bound', 'source': 'shared', 'pipeline': 'compact',
     'question': 'How does embedding.doc_embedding_text use '
                 'embedding._DOC_EMBED_CONTENT_CHARS to bound content and preserve the title?'},
    {'id': 'missing-network-policy', 'source': 'consumer', 'pipeline': 'cascade',
     'question': 'Which embedding model and network retry policy does '
                 'writer.LibraryWriter.update_document_embedding use? Are they established '
                 'by the supplied implementation?'},
)
SOURCE_FILES = (
    'tests/live_remote_accuracy_workflow.py', 'tests/test_live_remote_accuracy_workflow.py',
    'tests/live_mac_hosted_workflow.py', 'tests/live_mac_accuracy_probe.py',
    'tests/live_mac_call_probe.py', 'tests/live_model_smoke.py',
    'docgen/generator.py', 'docgen/prompts.py', 'docgen/catalog_writer.py',
    'docgen/catalog_describer.py', 'docgen/llm/local.py', 'docgen/validator.py',
    'embedding.py', 'writer.py', 'inference.py', 'model_config.py', 'local_models.py', 'llm.py',
    'config.py', 'scope_resolution.py', 'ariadne_mcp/service.py',
    'ariadne_mcp/service_analysis.py', 'ariadne_mcp/service_search.py',
    'library/clews.py', 'library/clew_embeddings.py', 'library/chain_answer.py',
    'library/structural_assembly.py', 'library/chain_story.py', 'library/prepared.py',
    'library/__init__.py', 'library/search.py', 'docgen/_legacy_analyzer.py',
    'docgen/_legacy_metadata.py',
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def repository_fixtures():
    """Keep exact source text and line coordinates; never execute extracted code."""
    result = []
    for origin, source in (('writer.py', 'consumer'), ('embedding.py', 'shared')):
        lines = (ROOT / origin).read_text().splitlines(keepends=True)
        tree = ast.parse(''.join(lines))
        if source == 'consumer':
            owner = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                         and node.name == 'LibraryWriter')
            function = next(node for node in owner.body if isinstance(node, ast.AsyncFunctionDef)
                            and node.name == 'update_document_embedding')
            kept = [(owner.lineno, owner.lineno), (function.lineno, function.end_lineno)]
            kept += [(node.lineno, node.end_lineno) for node in tree.body
                     if isinstance(node, ast.ImportFrom) and node.module in ('__future__', 'embedding')]
            symbols = [{'qualified_name': 'writer.LibraryWriter.update_document_embedding',
                        'kind': 'method', 'parent': 'writer.LibraryWriter',
                        'start': function.lineno, 'end': function.end_lineno}]
            call = next(node for node in ast.walk(function) if isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name) and node.func.id == 'doc_embedding_text')
            edge = {'caller': symbols[0]['qualified_name'], 'callee': 'embedding.doc_embedding_text',
                    'line': call.lineno, 'type': 'call'}
        else:
            function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                            and node.name == 'doc_embedding_text')
            constant = next(node for node in tree.body if isinstance(node, ast.Assign)
                            and any(isinstance(target, ast.Name) and target.id == '_DOC_EMBED_CONTENT_CHARS'
                                    for target in node.targets))
            kept = [(constant.lineno, constant.end_lineno), (function.lineno, function.end_lineno)]
            symbols = [
                {'qualified_name': 'embedding.doc_embedding_text', 'kind': 'function',
                 'parent': '', 'start': function.lineno, 'end': function.end_lineno},
                {'qualified_name': 'embedding._DOC_EMBED_CONTENT_CHARS', 'kind': 'constant',
                 'parent': '', 'start': constant.lineno, 'end': constant.end_lineno},
            ]
            reference = next(node for node in ast.walk(function) if isinstance(node, ast.Name)
                             and node.id == '_DOC_EMBED_CONTENT_CHARS')
            edge = {'caller': symbols[0]['qualified_name'], 'callee': symbols[1]['qualified_name'],
                    'line': reference.lineno, 'type': 'type_ref'}
        final_line = max(end for _, end in kept)
        content = ''.join(line if any(start <= number <= end for start, end in kept) else '\n'
                          for number, line in enumerate(lines[:final_line], 1))
        ast.parse(content)
        result.append({'origin': origin, 'origin_sha256': digest(ROOT / origin), 'source': source,
                       'path': f'sources/{source}/{origin}', 'ranges': sorted(kept),
                       'content': content, 'symbols': symbols, 'edge': edge})
    result.append({
        'source': 'unrelated', 'path': 'sources/unrelated/embedding.py',
        'origin': 'Synthetic out-of-scope negative control, not repository code',
        'content': 'def doc_embedding_text(title, content):\n    return title + content[:8]\n',
    })
    return result


@contextmanager
def selector_mapping(run):
    """Apply the production local selector allowance without relabelling hosted storage."""
    original = llm.chat_complete
    selected = SimpleNamespace(local=True, settings=resolve_models(mac_accuracy_profile()))

    async def mapped(*args, **kwargs):
        phase = kwargs.get('phase', 'completion')
        requested = kwargs.get('max_tokens', 2048)
        effective = llm._selector_max_tokens(selected, phase, requested)
        if effective != requested:
            run.report.setdefault('selector_cap_mapping', []).append({
                'phase': phase, 'requested_max_tokens': requested, 'effective_max_tokens': effective,
            })
            run.save()
            kwargs['max_tokens'] = effective
        return await original(*args, **kwargs)

    with patch.object(llm, 'chat_complete', mapped):
        yield


def prepare(output, budget, transport_factory, pacing):
    output.mkdir(parents=True, exist_ok=False)
    budget.max_requests = min(budget.max_requests, MAX_ATTEMPTS)
    run = Run(PROFILE, output, budget, pacing=pacing)
    configuration(PROFILE, output)
    data = json.loads((output / 'config.json').read_text())
    data.update(default_source='consumer', ask_pipeline='cascade', ask_selector_mode='llm', sources={
        'consumer': {'path': str(output / 'sources/consumer'), 'depends_on': ['shared']},
        'shared': {'path': str(output / 'sources/shared')},
        'unrelated': {'path': str(output / 'sources/unrelated')},
    })
    save_json(output / 'config.json', data)
    cfg = Config(output / 'config.json')
    fixtures = repository_fixtures()
    frozen = ['config.json', 'oracle.json', 'fixture-manifest.json']
    for item in fixtures:
        path = output / item['path']
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(item['content'])
        frozen.append(item['path'])
    save_json(output / 'fixture-manifest.json', [
        {key: value for key, value in item.items() if key != 'content'} for item in fixtures])
    save_json(output / 'oracle.json', {
        'review_only_marker': ORACLE_SENTINEL,
        'not_transmitted': True,
        'expected_scope': {'consumer': ['consumer', 'shared'], 'shared': ['consumer', 'shared'],
                           'unrelated': ['unrelated']},
        'required_facts': {
            'consumer-chain': ['Unbound branch passes doc.title/doc.content to doc_embedding_text, '
                               'awaits service.embed, then updates the document embedding.',
                               'The helper preserves the title, adds two newline characters and '
                               'keeps the first 2000 characters of (content or empty string).'],
            'shared-bound': ['The constant is 2000 characters, not tokens or total text length.',
                             'Title is not truncated; two newline characters separate title/content.'],
            'missing-network-policy': ['Selected embedding model and retry policy are absent from '
                                       'these extracts; do not infer them from function names.'],
        },
        'generated_document_required_facts': {
            'consumer': ['Missing document returns None before embedding.',
                         'A bound inference context uses nested_write, prepare_document and apply_prepared_document.',
                         'The unbound branch constructs helper text, awaits embed and updates the document.'],
            'shared': ['The constant bounds content to 2000 characters; title is not truncated.',
                       'Two newline characters separate title and content.',
                       'False-valued content becomes empty before slicing.'],
        },
        'coverage_policy': 'Appropriate withholding of a required fact because its evidence was not '
                           'delivered remains a coverage miss, separately from factual correctness. '
                           'Unasked branch details are not required Ask facts.',
        'review_gates': ['Every substantive generated/answer claim supported by supplied source.',
                         'Required question facts covered consistently; honest withholding is '
                         'recorded separately and does not pass completeness.',
                         'Citations resolve to delivered source extents and support each claim.',
                         'No out-of-scope negative-control content reaches Search or Ask.'],
        'quality_note': 'Finite diagnostic, not a general accuracy percentage; no automatic prose grader.',
    })
    source_hashes = {}
    for name in SOURCE_FILES:
        target = output / 'provenance' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        source_hashes[name] = digest(target)
        frozen.append('provenance/' + name)
    protocol = {
        'scope': 'Real repository excerpt generation -> 3072 embeddings -> dependency-scoped Search '
                 'and source-backed compiler-evidence-path Ask, with live upstream model replies',
        'questions': list(QUESTIONS), 'document_types': ['explanation', 'qa'],
        'library_provider': 'deepinfra', 'chat_model': PROFILE.model,
        'embedding_model': PROFILE.embedding_model, 'native_embedding_dimensions': 4096,
        'stored_dimensions': 3072, 'embedding_recipe': 'Production query framing and prefix/L2 reduction',
        'graph_origin': 'AST-checked source-backed seeded SCIP-table records; not an indexer qualification',
        'constant_edge': 'The seeded type_ref denotes a module-constant Name read, following the '
                         'existing structural_assembly convention for field/type references; it '
                         'does not assert a compiler emitted this relationship.',
        'fixture_origin': 'Exact selected repository lines with omitted lines blanked; partial '
                          'extracts are not independently executable complete modules',
        'control_origin': 'One frozen synthetic source/explanation is an out-of-scope negative control; '
                          'it is not a scripted model reply or generated-document accuracy case',
        'selector_mapping': 'Scoped harness wrapper calls production _selector_max_tokens using '
                            'selected local settings solely for caps; actual context and library '
                            'remain hosted. Real chat_complete and downstream responses are used.',
        'chat_adapter': 'HostedThinkingProvider maps native selected thinking options onto hosted '
                        'Qwen; hosted weights/precision are not pinned to the selected local artifact.',
        'chat_options': resolve_models(mac_accuracy_profile()).chat.request_options,
        'timeout_reporting': 'Actual adapter_calls record 600-second HTTP defaults; shared hosted '
                             'completion trace can still display historical 60-second defaults.',
        'maximum_http_attempts': budget.max_requests, 'budget_usd': budget.limit, 'concurrency': 1,
        'retry_policy': 'Only bounded RecordedTransport HTTP429 retries; terminal failures abort '
                        'before optional application catches and SDK retries.',
        'quality_verdict': 'unreviewed', 'source_hashes': source_hashes,
        'frozen_artifacts': {name: digest(output / name) for name in frozen},
        'limitations': ['Seeded graph does not test a real indexer.',
                        'Graph source gates and Search dependency closures are observed separately; '
                        'cross-source compiler completeness is not assumed.',
                        'No native Q4, GPU, browser, durable spool, M5 memory/latency qualification.',
                        'Structural completion does not establish factual quality.'],
    }
    save_json(output / 'protocol.json', protocol)
    protocol_hash = digest(output / 'protocol.json')

    def verify():
        if digest(output / 'protocol.json') != protocol_hash:
            raise ValueError('Frozen protocol changed')
        for name, expected in protocol['frozen_artifacts'].items():
            if digest(output / name) != expected:
                raise ValueError(f'Frozen artifact changed: {name}')
        for name, expected in source_hashes.items():
            if digest(ROOT / name) != expected:
                raise ValueError(f'Frozen production source changed: {name}')

    run.report.update(scope=protocol['scope'], quality_verdict='unreviewed', answers=[],
                      protocol_sha256=protocol_hash, rates_verified_on='2026-10-22',
                      local_q4_qualified=False, hardware_qualified=False,
                      transport_mode='live' if transport_factory is httpx.AsyncHTTPTransport else 'simulated')
    run.save()
    return run, cfg, fixtures, verify


def check_ask_scope(run, name, trace, answer):
    value = answer.model_dump(mode='json')
    delivered = json.dumps({
        'messages': [row.get('messages', []) for row in trace],
        'citations': value.get('citations', []), 'chain_files': value.get('chain_files', []),
    })
    require(run, 'ask_scope/' + name, not any(marker in delivered for marker in (
        'UNRELATED_CONTROL', 'unrelated-control', str(run.output / 'sources/unrelated'))))


def seed_graph(library, output, fixtures):
    symbols = {symbol['qualified_name']: (item, symbol) for item in fixtures[:2]
               for symbol in item['symbols']}
    ids = {name: f'scip-python python accuracy 1 `{name}`.' for name in symbols}
    with library._conn_provider.acquire() as conn:
        for name, (item, symbol) in symbols.items():
            conn.execute(
                'INSERT INTO scip_symbols(canonical_id,source_name,language,file,line_start,line_end,'
                'kind,display_name,qualified_name,parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
                (ids[name], item['source'], 'python', str(output / item['path']), symbol['start'],
                 symbol['end'], symbol['kind'], name.rsplit('.', 1)[-1], name, symbol['parent']))
        for item in fixtures[:2]:
            edge = item['edge']
            conn.execute(
                'INSERT INTO scip_edges(caller_canonical_id,callee_canonical_id,edge_type,file,line,'
                'confidence) VALUES (?,?,?,?,?,?)',
                (ids[edge['caller']], ids[edge['callee']], edge['type'],
                 str(output / item['path']), edge['line'], 'exact'))
            route = [edge['caller'], edge['callee']]
            add_clew(conn, source_name=item['source'], entry_symbol=route[0], route=route,
                     steps=[name.rsplit('.', 1)[-1] for name in route],
                     files=[str(output / symbols[name][0]['path']) for name in route],
                     strategy='ast-checked-seeded-fixture')


async def workflow(run, cfg, fixtures, api_key, transport_factory, verify):
    context = InferenceContext.from_config(cfg)
    service = AriadneService()
    service._config, service._inference_context = cfg, context
    saved = {}
    try:
        with context.write_session(), context.open_library(operation='write') as library:
            context._chat = HostedThinkingProvider(run, api_key, transport_factory, verify)
            context.embeddings._client = httpx.AsyncClient(
                base_url=PROFILE.embedding_url, headers={'Authorization': f'Bearer {api_key}'},
                timeout=600, trust_env=False, follow_redirects=False,
                transport=BoundedTransport(RecordedTransport(run, 'embeddings', transport_factory()), verify))
            identity = context.binding.manifest
            require(run, 'hosted_identity', not context.local and
                    context.settings.chat.provider == context.settings.embeddings.provider == 'deepinfra')
            seed_graph(library, run.output, fixtures)
            require(run, 'source_backed_graph', True, symbols=3, edges=2,
                    origin='Seeded records with exact AST extents and reference/call locations')
            writer = LibraryWriter(library)
            for item in fixtures[:2]:
                source = run.output / item['path']
                synced = await sync_file_catalog(library, writer, item['source'], source.parent, source)
                require(run, 'catalog_sync/' + item['source'], synced.added >= 1)
                described = await describe_source_elements(library, writer, item['source'], concurrency=1)
                require(run, 'catalog_describe/' + item['source'], described['failed'] == 0,
                        result=described)
            save_json(run.output / 'catalog.json', [
                {'id': doc.id, 'source_name': doc.source_name, 'metadata': doc.metadata, 'content': doc.content}
                for doc in library.list_documents('catalog')])
            generator_cfg = GeneratorConfig(model=PROFILE.model, provider='deepinfra', inference_context=context)
            async with DocGenerator(generator_cfg) as generator:
                for item in fixtures[:2]:
                    source = run.output / item['path']
                    for kind in ('explanation', 'qa'):
                        generated = await generator.generate_for_file(source, (kind,))
                        require(run, f'generation/{item["source"]}/{kind}', len(generated) == 1)
                        content = generated[0]
                        name = f'{item["source"]}-{kind}'
                        (run.output / 'documents' / (name + '.md')).write_text(content.content)
                        validation = ContentValidator().validate(content.content, kind, content.title)
                        require(run, 'structure/' + name, validation.is_valid,
                                issues=[issue.code for issue in validation.issues])
                        doc = await writer.add_document(
                            kind, content.title, content.content, list(content.source_files),
                            metadata=content.metadata, doc_id=name, source_name=item['source'])
                        require(run, 'storage/' + name, doc.content == content.content and
                                content.source_snapshot == digest(source), vector_parts=check_vectors(library, doc))
                        run.report['documents'].append({'id': doc.id, 'source_name': item['source'],
                                                        'source_snapshot': content.source_snapshot})
                        run.save()
            control = fixtures[-1]
            await writer.add_document(
                'explanation', 'UNRELATED_CONTROL embedding helper',
                '# UNRELATED_CONTROL embedding helper\n\nThis unrelated source appends only the first '
                'eight content characters directly after the title. It has no separator.\n',
                [str(run.output / control['path'])], doc_id='unrelated-control', source_name='unrelated')
            for source in ('consumer', 'shared'):
                require(run, 'clew_embedding/' + source, await embed_clews(library, source_name=source) == 1)
            with library._conn_provider.acquire() as conn:
                vectors = [np.frombuffer(row[0], dtype=np.float32) for row in conn.execute('SELECT embedding FROM clews')]
            require(run, 'clew_vectors', len(vectors) == 2 and all(
                vector.shape == (3072,) and np.isclose(np.linalg.norm(vector.astype(np.float64)), 1)
                for vector in vectors))
            with selector_mapping(run):
                for case in QUESTIONS:
                    cfg._config['ask_pipeline'] = case['pipeline']
                    search = await service.search(query=case['question'], source=case['source'],
                                                  sections_only=True, limit=8)
                    save_json(run.output / f'search-{case["id"]}.json', search.model_dump(mode='json'))
                    delivered = sorted({doc.source_name for doc in search.documents})
                    require(run, 'search_scope/' + case['id'], bool(search.documents) and
                            set(delivered).issubset(cfg.scope_closure(case['source'])) and
                            'unrelated' not in delivered, delivered_sources=delivered,
                            expected_scope=sorted(cfg.scope_closure(case['source'])))
                    with llm.capture_completion_trace() as trace:
                        try:
                            answer = await service.ask(case['question'], source=case['source'])
                        finally:
                            save_json(run.output / f'ask-trace-{case["id"]}.json', trace)
                    save_json(run.output / f'answer-{case["id"]}.json', answer.model_dump(mode='json'))
                    check_ask_scope(run, case['id'], trace, answer)
                    expected = (['shared-explanation', 'shared-qa'] if case['source'] == 'shared'
                                else ['consumer-explanation', 'consumer-qa', 'shared-explanation', 'shared-qa'])
                    if case['id'] == 'missing-network-policy':
                        expected = ['consumer-explanation', 'consumer-qa']
                    ranks = {doc.id: rank for rank, doc in enumerate(search.documents, 1)}
                    save_json(run.output / f'evidence-{case["id"]}.json', {
                        'question': case, 'search_ids': [doc.id for doc in search.documents],
                        'expected_relevant_documents': expected,
                        'relevant_document_ranks': {name: ranks.get(name) for name in expected},
                        'delivered_sources': delivered, 'phases': [row['phase'] for row in trace],
                        'citations': answer.model_dump(mode='json').get('citations', []),
                        'chain_files': answer.model_dump(mode='json').get('chain_files', []),
                        'graph_diagnostics': answer.model_dump(mode='json').get('graph_diagnostics', {}),
                        'quality_verdict': 'unreviewed',
                    })
                    require(run, 'answer_recorded/' + case['id'], bool(answer.answer.strip()))
                    run.report['answers'].append({'id': case['id'], 'quality_verdict': 'unreviewed'})
                    run.save()
            saved = {doc.id: snapshot(library, doc.id) for doc in library.list_documents()
                     if doc.embedding is not None}
    finally:
        await service.close()
        await context.aclose()
    reopened = InferenceContext.from_config(cfg)
    try:
        with reopened.open_library(operation='inspect') as library:
            require(run, 'reopen_preserves_content_and_vectors', bool(saved) and
                    reopened.binding.manifest == identity and all(
                        snapshot(library, doc_id) == before for doc_id, before in saved.items()))
    finally:
        await reopened.aclose()


async def execute(output, budget, *, api_key, transport_factory=httpx.AsyncHTTPTransport, pacing=15):
    if budget.limit > 1 or not api_key:
        raise ValueError('At most $1 and a provider credential are required')
    run, cfg, fixtures, verify = prepare(output, budget, transport_factory, pacing)
    previous, key = config_module._global_config, os.environ.get(PROFILE.key_env)
    config_module._global_config = cfg
    os.environ[PROFILE.key_env] = api_key
    started = time.monotonic()
    try:
        await workflow(run, cfg, fixtures, api_key, transport_factory, verify)
        run.report['status'] = 'completed_pending_quality_review'
    except asyncio.CancelledError:
        run.report['status'] = 'cancelled'
        raise
    except (WorkflowAbort, Exception) as error:  # noqa: BLE001 - preserve diagnostic failures, no retry.
        run.report.update(status='failed', error_type=type(error).__name__,
                          error=str(error).replace(api_key, '<redacted>'))
    finally:
        config_module._global_config = previous
        if key is None:
            os.environ.pop(PROFILE.key_env, None)
        else:
            os.environ[PROFILE.key_env] = key
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
    return 0 if report['status'] == 'completed_pending_quality_review' else 1


if __name__ == '__main__':
    raise SystemExit(main())
