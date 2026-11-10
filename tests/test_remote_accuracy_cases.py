"""Independent behavior oracles for real-source remote accuracy fixtures.

Only trusted repository functions execute here. Model-produced code never does.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import yaml

from tests.remote_accuracy_cases import frozen_cases


def _config(tmp_path, data):
    from config import Config

    path = tmp_path / 'ariadne.yaml'
    path.write_text(yaml.safe_dump(data))
    return Config(path), path


def test_corpus_is_reproducible_serializable_and_source_verifiable():
    cases = frozen_cases()
    assert tuple(cases) == (
        'config_mutation', 'dependency_scope', 'embedding_delegation',
        'catalog_metadata', 'chunk_boundaries', 'endpoint_routes',
    )
    assert json.loads(json.dumps(cases)) == cases
    assert [c['split'] for c in cases.values()] == ['development'] * 4 + ['confirmation'] * 2
    root = Path(__file__).resolve().parents[1]
    for case_id, case in cases.items():
        assert case['case_id'] == case_id
        assert case['doc_types'] == ['explanation', 'qa']
        assert len(case['source'].encode()) < 12_000
        ast.parse(case['source'])
        assert case['source_sha256'] == hashlib.sha256(case['source'].encode()).hexdigest()
        assert case['scaffolding']
        assert len(case['required_facts']) >= 4
        assert len({f['id'] for f in case['required_facts']}) == len(case['required_facts'])
        segment_ids = {s['id'] for s in case['source_segments']}
        for segment in case['source_segments']:
            original = (root / segment['path']).read_bytes()
            assert hashlib.sha256(original).hexdigest() == segment['file_sha256']
            selected = ''.join(original.decode().splitlines(keepends=True)[
                segment['start_line'] - 1:segment['end_line']
            ])
            assert selected == segment['text']
            assert hashlib.sha256(selected.encode()).hexdigest() == segment['sha256']
        for question in case['questions']:
            assert set(question['relevant_segment_ids']) <= segment_ids
        assert 'observations' not in case['source']
    cases['config_mutation']['observations'].clear()
    cases['dependency_scope']['required_facts'].clear()
    assert frozen_cases()['config_mutation']['observations']
    assert frozen_cases()['dependency_scope']['required_facts']


def test_config_mutation_oracle_uses_actual_production_functions(tmp_path):
    from config import Config

    cfg, path = _config(tmp_path, {
        'default_source': 'shared',
        'sources': {
            'shared': {'path': str(tmp_path / 'shared')},
            'product': {'path': str(tmp_path / 'product'), 'depends_on': ['shared']},
        },
    })
    before = path.read_bytes()
    absent = cfg.remove_source('absent')
    unchanged = path.read_bytes() == before
    removed = cfg.remove_source('shared')
    removed_result = {
        'returned': removed,
        'sources': sorted(cfg.sources),
        'default_source': cfg.default_source,
        'consumer_dependencies': cfg.get_source_dependencies('product'),
    }
    original = path.read_bytes()

    def abort(data):
        data['ignored'] = True
        return False

    abort_result = cfg._mutate_config(abort)
    abort_unchanged = path.read_bytes() == original

    def zero(data):
        data['custom_marker'] = 'committed'
        return 0

    zero_result = cfg._mutate_config(zero)
    zero_persisted = yaml.safe_load(path.read_text())['custom_marker']

    def raises(_data):
        raise RuntimeError('oracle callback failure')

    with pytest.raises(RuntimeError, match='oracle callback failure'):
        cfg._mutate_config(raises)
    unbound = Config.__new__(Config)
    unbound._config_path = None
    no_path = unbound._mutate_config(lambda _data: pytest.fail('callback must not run'))
    path.write_text('sources: [unterminated')
    invalid_yaml = cfg._mutate_config(lambda _data: pytest.fail('callback must not run'))
    observed = {
        'absent': {'returned': absent, 'bytes_unchanged': unchanged},
        'removed': removed_result,
        'abort': {'returned': abort_result, 'bytes_unchanged': abort_unchanged},
        'zero_return': {'returned': zero_result, 'persisted': zero_persisted},
        'callback_error': 'RuntimeError', 'no_path': no_path, 'invalid_yaml': invalid_yaml,
    }
    assert observed == frozen_cases()['config_mutation']['observations']


def test_dependency_scope_oracle_covers_direction_and_errors(tmp_path):
    cfg, _ = _config(tmp_path, {'sources': {
        'base': {'path': str(tmp_path / 'base')},
        'left': {'path': str(tmp_path / 'left'), 'depends_on': ['base']},
        'right': {'path': str(tmp_path / 'right'), 'depends_on': ['base']},
        'top': {'path': str(tmp_path / 'top'), 'depends_on': ['left', 'right']},
        'isolated': {'path': str(tmp_path / 'isolated')},
    }})
    observed = {name: sorted(cfg.scope_closure(name)) for name in cfg.sources}
    with pytest.raises(KeyError) as exc:
        cfg.scope_closure('unknown')
    assert all(name in str(exc.value) for name in cfg.sources)
    cycle_dir = tmp_path / 'cycle'
    cycle_dir.mkdir()
    cyclic, _ = _config(cycle_dir, {'sources': {
        'alpha': {'path': str(cycle_dir / 'a'), 'depends_on': ['beta']},
        'beta': {'path': str(cycle_dir / 'b'), 'depends_on': ['alpha']},
    }})
    with pytest.raises(ValueError, match='alpha -> beta -> alpha'):
        cyclic.scope_closure('alpha')
    assert {'closures': observed, 'unknown': 'KeyError', 'cycle': 'ValueError'} == (
        frozen_cases()['dependency_scope']['observations']
    )


@pytest.mark.asyncio
async def test_embedding_delegation_oracle_uses_real_writer_and_helper():
    from writer import LibraryWriter

    events = []
    doc = SimpleNamespace(title='A title', content='x' * 2001)
    service = SimpleNamespace(embed=AsyncMock(return_value='vector-sentinel'))
    library = SimpleNamespace(
        get_document=lambda _id: doc,
        update_document=lambda doc_id, **kw: events.append([doc_id, kw]) or 'updated-sentinel',
    )
    writer = SimpleNamespace(
        library=library, inference_context=None,
        _get_embedding_service=AsyncMock(return_value=service),
    )
    result = await LibraryWriter.update_document_embedding(writer, 'doc-id')
    supplied = service.embed.call_args.args[0]
    legacy = {
        'returned': result, 'input': supplied,
        'update_arguments': events[:], 'embedding_calls': service.embed.await_count,
    }
    library.get_document = lambda _id: None
    missing = await LibraryWriter.update_document_embedding(writer, 'missing')
    missing_extra_calls = service.embed.await_count - legacy['embedding_calls']
    events.clear()

    @contextmanager
    def nested_write():
        events.append('enter')
        try:
            yield
        finally:
            events.append('exit')

    async def prepare(value):
        assert value is doc
        events.append('prepare')
        return 'prepared-sentinel'

    def apply(value):
        events.append(['apply', value])
        return 'bound-sentinel'

    library.get_document = lambda _id: doc
    library.apply_prepared_document = apply
    writer.prepare_document = prepare
    writer.inference_context = SimpleNamespace(binding=object(), nested_write=nested_write)
    bound_result = await LibraryWriter.update_document_embedding(writer, 'doc-id')
    observed = {
        'legacy': legacy,
        'missing': {'returned': missing, 'additional_embedding_calls': missing_extra_calls},
        'bound': {'returned': bound_result, 'events': events, 'legacy_embedding_calls':
                  service.embed.await_count - legacy['embedding_calls']},
    }
    assert observed == frozen_cases()['embedding_delegation']['observations']


@pytest.mark.asyncio
async def test_catalog_metadata_oracle_observes_actual_prompt_and_call(monkeypatch):
    from docgen import catalog_describer

    metadata = {
        'subtype': 'method', 'qualified_name': 'sample.Widget.run',
        'parent_qualified_name': 'sample.Widget', 'signature': 'run(self, rows)',
        'file': 'sample.py', 'location': {'line_start': 3, 'line_end': 8},
        'body': 'UNSUPPLIED_BODY_SENTINEL', 'description': 'OLD_DESCRIPTION_SENTINEL',
    }
    original = copy.deepcopy(metadata)
    prompt = catalog_describer.build_describe_prompt(metadata)
    complete = AsyncMock(return_value='  Returned description. \n')
    monkeypatch.setattr(catalog_describer, 'chat_complete', complete)
    result = await catalog_describer.describe_element(metadata, model='chosen')
    assert complete.call_args.args == ([{'role': 'user', 'content': prompt}],)
    assert complete.call_args.kwargs == {'model': 'chosen'}
    defaults = catalog_describer.build_describe_prompt({})
    observed = {
        'fields_present': all(field in prompt for field in (
            'Kind: method', 'Name: sample.Widget.run', 'Parent: sample.Widget',
            'Signature: run(self, rows)', 'Location: sample.py:3-8',
        )),
        'body_absent': 'UNSUPPLIED_BODY_SENTINEL' not in prompt,
        'old_description_absent': 'OLD_DESCRIPTION_SENTINEL' not in prompt,
        'metadata_unchanged': metadata == original,
        'default_parent': 'Parent: (module-level)' in defaults,
        'default_location': 'Location: :0-0' in defaults,
        'returned': result,
        'model_argument': complete.call_args.kwargs['model'],
    }
    assert observed == frozen_cases()['catalog_metadata']['observations']


def test_reserved_chunk_oracle():
    from writer import ChunkConfig, _split_long_text, chunk_text

    observed = {
        'empty': chunk_text(''),
        'short': chunk_text('abc', ChunkConfig(chunk_size=4)),
        'fixed_overlap': _split_long_text('abcdefgh', 4, 2),
        'clamped_overlap': _split_long_text('abcdef', 4, 20),
        'nonpositive': _split_long_text('abcdef', 0, 2),
        'paragraphs': chunk_text('aaa\n\nbbb', ChunkConfig(chunk_size=5, chunk_overlap=2)),
    }
    assert observed == frozen_cases()['chunk_boundaries']['observations']


def test_reserved_endpoint_oracle():
    from model_config import ModelConfigurationError, normalize_url, tokenize_url

    invalid = ['ftp://host/v1', 'https://user:pass@host/v1',
               'https://host/v1?x=1', 'https://host/v1#fragment', '/v1']
    for url in invalid:
        with pytest.raises(ModelConfigurationError):
            normalize_url(url)
    with pytest.raises(ModelConfigurationError):
        tokenize_url('https://host/prefix/v10')
    observed = {
        'normalized': normalize_url('HTTPS://EXAMPLE.COM:443/prefix/v1///'),
        'tokenized': tokenize_url('https://host/prefix/v1///'),
        'root': tokenize_url('http://localhost:8080/v1'),
        'rejected_count': len(invalid),
        'invalid_error': 'ModelConfigurationError',
        'wrong_route_error': 'ModelConfigurationError',
    }
    assert observed == frozen_cases()['endpoint_routes']['observations']
