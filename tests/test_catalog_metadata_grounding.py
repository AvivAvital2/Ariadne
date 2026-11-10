"""Catalog requests must reflect their metadata-only evidence boundary.

These offline checks validate requests and persistence, not model accuracy.
"""

import argparse
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from docgen import catalog_describer
from docgen.local_pipeline import plan_catalog_descriptions
from docgen.spool import SpoolStore
from inference import InferenceContext
from library import Library
from local_models import mac_accuracy_profile
from model_config import resolve_models

METADATA = {
    'kind': 'element',
    'source_name': 'test',
    'subtype': 'method',
    'qualified_name': 'w.Widget.run',
    'parent_qualified_name': 'w.Widget',
    'signature': 'run(self, rows)',
    'file': 'w.py',
    'location': {'line_start': 5, 'line_end': 13},
}


def _add(library):
    return library.add_document(
        'catalog', 'Widget.run', 'Unverified old descriptive text.',
        source_name='test', source_files=['w.py'], metadata=METADATA,
    )


def _assert_boundary(prompt):
    lowered = prompt.lower()
    assert 'metadata only' in lowered
    assert 'implementation body is not provided' in lowered
    assert 'names alone' in lowered
    assert 'do not infer' in lowered
    assert 'unknown' in lowered
    for field in ('Kind: method', 'Name: w.Widget.run', 'Parent: w.Widget',
                  'Signature: run(self, rows)', 'Location: w.py:5-13'):
        assert field in prompt


@pytest.mark.asyncio
async def test_actual_live_request_explains_missing_body_and_retains_metadata(monkeypatch):
    complete = AsyncMock(return_value='  Declares run(self, rows); its body is unavailable.  ')
    monkeypatch.setattr(catalog_describer, 'chat_complete', complete)
    metadata = {**METADATA, 'description': 'OLD_DESCRIPTION_SENTINEL',
                'content': 'UNSUPPLIED_BODY_SENTINEL'}
    result = await catalog_describer.describe_element(metadata, model='chosen-model')
    messages = complete.call_args.args[0]
    assert messages == [{'role': 'user', 'content':
                         catalog_describer.build_describe_prompt(metadata)}]
    assert complete.call_args.kwargs == {'model': 'chosen-model'}
    prompt = messages[0]['content']
    _assert_boundary(prompt)
    assert 'OLD_DESCRIPTION_SENTINEL' not in prompt
    assert 'UNSUPPLIED_BODY_SENTINEL' not in prompt
    assert result == 'Declares run(self, rows); its body is unavailable.'


@pytest.mark.asyncio
async def test_live_hosted_batch_and_local_plan_use_identical_evidence(tmp_path, monkeypatch):
    complete = AsyncMock(return_value='Description')
    monkeypatch.setattr(catalog_describer, 'chat_complete', complete)
    strategy = SimpleNamespace(
        submit_batch=AsyncMock(return_value=SimpleNamespace(batch_id='catalog-test')),
        poll_batch=AsyncMock(), fetch_batch_results=AsyncMock(return_value={}),
    )
    with Library(tmp_path / 'library.db') as library:
        doc = _add(library)
        await catalog_describer.describe_element(doc.metadata)
        specs, candidates, pending = plan_catalog_descriptions(library, source_name='test')
        await catalog_describer.describe_source_elements_batched(
            library, None, 'test', strategy=strategy,
            staleness_db_path=tmp_path / 'staleness.db',
        )
    actual = complete.call_args.args[0][0]['content']
    submitted = strategy.submit_batch.call_args.args[0]
    assert len(submitted) == len(specs) == candidates == pending == 1
    assert submitted[0].user_prompt == specs[0].request.user_prompt == actual
    assert submitted[0].system_prompt == specs[0].request.system_prompt == ''
    assert catalog_describer.build_describe_prompt(doc.metadata) == actual
    _assert_boundary(actual)


@pytest.mark.asyncio
async def test_dry_run_counts_same_prompt_as_live_without_model_call(tmp_path, monkeypatch):
    from cli.dry_run import cmd_dry_run
    from tests._scoped_config_fixture import install_test_config

    install_test_config(monkeypatch, tmp_path, 'test')
    database = tmp_path / 'library.db'
    with Library(database) as library:
        doc = _add(library)
    monkeypatch.setattr('cli.index.cmd_discover', lambda *_a, **_kw: 0)
    monkeypatch.setattr('cli.index.cmd_index', lambda *_a, **_kw: 0)
    monkeypatch.setattr('cli.dry_run.cmd_catalog_sync', AsyncMock(return_value=0))
    monkeypatch.setattr('cli.dry_run.get_library', lambda *_a: Library(database))
    counted = []

    def count(prompt, model):
        counted.append((prompt, model))
        return 123

    monkeypatch.setattr('docgen.token_count.count_text_tokens', count)
    complete = AsyncMock(side_effect=AssertionError('A dry-run must not call a model'))
    monkeypatch.setattr(catalog_describer, 'chat_complete', complete)
    assert await cmd_dry_run(argparse.Namespace(
        source='test', db=str(database), model='claude-opus-4-7', verbose=True,
    )) == 0
    assert counted == [(catalog_describer.build_describe_prompt(doc.metadata), 'claude-opus-4-7')]
    _assert_boundary(counted[0][0])
    complete.assert_not_called()


@pytest.mark.asyncio
async def test_saved_local_request_survives_prompt_update_and_reopen(tmp_path, monkeypatch):
    context = InferenceContext(
        resolve_models(mac_accuracy_profile()), tmp_path / 'library.db', tmp_path / 'side.db',
    )
    original_template = catalog_describer._PROMPT
    with context.write_session(), context.open_library(operation='write') as library:
        _add(library)
        # Simulate a job saved by the previous application version.
        monkeypatch.setattr(catalog_describer, '_PROMPT', 'Previous request: {qualified_name}')
        specs, _, _ = plan_catalog_descriptions(library, source_name='test')
        store = SpoolStore(context, library)
        run_id = store.create(specs, operation='catalog-describe', scope={'source_name': 'test'})
        before = store.jobs(run_id)[0]
    await context.aclose()
    monkeypatch.setattr(catalog_describer, '_PROMPT', original_template)
    reopened = InferenceContext(context.settings, context.db_path, context.staleness_db_path)
    try:
        with reopened.open_library(operation='inspect') as library:
            saved = SpoolStore(reopened, library).jobs(run_id)[0]
            assert saved['request_hash'] == before['request_hash']
            assert saved['payload_json'] == before['payload_json']
            request = json.loads(saved['payload_json'])['request']
            assert request['messages'][-1]['content'] == 'Previous request: w.Widget.run'
            fresh, _, _ = plan_catalog_descriptions(library, source_name='test')
            _assert_boundary(fresh[0].request.user_prompt)
    finally:
        await reopened.aclose()


@pytest.mark.parametrize('metadata', [{}, {'location': None}, {
    'subtype': 'function', 'qualified_name': 'café.函数', 'signature': 'f(x: int) -> str',
    'parent_qualified_name': None,
}])
def test_sparse_and_unicode_metadata_remain_renderable(metadata):
    prompt = catalog_describer.build_describe_prompt(metadata)
    assert 'implementation body is not provided' in prompt.lower()
    assert 'Parent: (module-level)' in prompt
    assert 'Location: :0-0' in prompt
    assert metadata.get('qualified_name', '') in prompt
