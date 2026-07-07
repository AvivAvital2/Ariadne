"""OpenAI Batch API over /v1/embeddings (designs/import-embedding-cost-controls.md).

Same submit → poll → fetch → cancel lifecycle as the chat strategy (~50%
off, 24h window), but each JSONL line embeds a GROUP of texts — so even a
very large rebuild stays far below the per-job request cap — and fetch
returns unit-normalized float32 vectors per custom_id (None for failed
groups), matching the live ``embed_batch`` contract.

New surface is reached via the module attribute (``oaib.…``) so the red
phase fails at call time instead of erroring at collection.
"""
from __future__ import annotations

import json

import numpy as np

import docgen.llm.openai_batch as oaib
from docgen.llm.batch import BatchStatus, BatchSubmission
from embedding import EmbeddingConfig, EmbeddingService
from library import Library
from writer import ChunkConfig, LibraryWriter


class _FakeResponse:
    def __init__(self, payload, *, text: str | None = None):
        self._payload = payload
        self.text = text if text is not None else json.dumps(payload)

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return self._payload


class _FakeOpenAIClient:
    """Fakes the OpenAI Batch API surface — files upload/content and batches
    create/retrieve/cancel — dispatching by (method, url)."""

    def __init__(self, *, polls: list[dict], file_contents: dict[str, str]):
        self.calls: list[tuple[str, str]] = []
        self.headers: dict[str, str] = {}
        self._polls = list(polls)
        self._file_contents = dict(file_contents)
        self.uploads: list[dict] = []
        self.created: list[dict] = []
        self.cancelled: list[str] = []

    async def post(self, url: str, *, json=None, data=None, files=None, **kw):
        self.calls.append(('POST', url))
        if url == '/files':
            self.uploads.append({'data': data, 'files': files})
            return _FakeResponse({'id': 'file_in_1'})
        if url == '/batches':
            self.created.append(json)
            return _FakeResponse({'id': 'batch_emb_1', 'status': 'validating'})
        if url.endswith('/cancel'):
            self.cancelled.append(url)
            return _FakeResponse({'id': 'batch_emb_1', 'status': 'cancelling'})
        raise AssertionError(f'unexpected POST {url}')

    async def get(self, url: str, **kw):
        self.calls.append(('GET', url))
        if url.startswith('/batches/'):
            poll = self._polls.pop(0) if len(self._polls) > 1 else self._polls[0]
            return _FakeResponse(poll)
        if url.startswith('/files/') and url.endswith('/content'):
            file_id = url.split('/')[2]
            return _FakeResponse({}, text=self._file_contents.get(file_id, ''))
        raise AssertionError(f'unexpected GET {url}')

    async def aclose(self) -> None:
        return None


def _service_with(client: _FakeOpenAIClient) -> EmbeddingService:
    service = EmbeddingService(EmbeddingConfig(
        api_key='test-key', model='emb-test-model', dimensions=4))
    service._client = client
    return service


class TestOpenAIEmbeddingsBatchStrategy:
    async def test_lifecycle_submits_groups_and_returns_normalized_vectors(self):
        in_progress = {
            'id': 'batch_emb_1', 'status': 'in_progress',
            'request_counts': {'total': 2, 'completed': 0, 'failed': 0},
        }
        completed = {
            'id': 'batch_emb_1', 'status': 'completed',
            'request_counts': {'total': 2, 'completed': 1, 'failed': 1},
            'output_file_id': 'file_out_1', 'error_file_id': 'file_err_1',
        }
        out_jsonl = json.dumps({
            'custom_id': 'grp:0',
            'response': {'status_code': 200, 'body': {'data': [
                {'index': 1, 'embedding': [0.0, 3.0, 0.0, 4.0]},
                {'index': 0, 'embedding': [1.0, 0.0, 0.0, 0.0]},
            ]}},
            'error': None,
        })
        err_jsonl = json.dumps({
            'custom_id': 'grp:1',
            'response': {'status_code': 500, 'body': {}},
            'error': {'message': 'boom'},
        })
        client = _FakeOpenAIClient(
            polls=[in_progress, completed],
            file_contents={'file_out_1': out_jsonl, 'file_err_1': err_jsonl},
        )
        strategy = oaib.OpenAIEmbeddingsBatchStrategy(_service_with(client))

        submission = await strategy.submit_batch([
            oaib.EmbeddingBatchRequest(custom_id='grp:0', texts=['alpha', 'beta']),
            oaib.EmbeddingBatchRequest(custom_id='grp:1', texts=['gamma']),
        ])
        assert submission.batch_id == 'batch_emb_1'
        assert client.created[0]['endpoint'] == '/v1/embeddings'
        assert client.created[0]['input_file_id'] == 'file_in_1'
        raw = client.uploads[0]['files']['file'][1]
        lines = [json.loads(l) for l in raw.decode().splitlines() if l.strip()]
        assert [row['custom_id'] for row in lines] == ['grp:0', 'grp:1']
        first = lines[0]
        assert first['method'] == 'POST'
        assert first['url'] == '/v1/embeddings'
        assert first['body'] == {
            'model': 'emb-test-model', 'input': ['alpha', 'beta'], 'dimensions': 4,
        }

        seen: list[tuple[int, int, int]] = []
        status = await strategy.poll_batch(
            'batch_emb_1', poll_interval=0,
            on_progress=lambda p, s, e: seen.append((p, s, e)),
        )
        assert status.processing_status == 'ended'
        assert seen == [(2, 0, 0), (0, 1, 1)]

        results = await strategy.fetch_batch_results('batch_emb_1')
        assert set(results) == {'grp:0', 'grp:1'}
        assert results['grp:1'] is None
        vectors = results['grp:0']
        assert len(vectors) == 2  # ordered by index, not file order
        assert vectors[0].dtype == np.float32
        assert np.allclose(vectors[0], [1.0, 0.0, 0.0, 0.0])
        assert np.allclose(vectors[1], [0.0, 0.6, 0.0, 0.8])  # unit-normalized

    async def test_cancel_posts_to_cancel_endpoint(self):
        client = _FakeOpenAIClient(polls=[{'status': 'cancelling'}], file_contents={})
        strategy = oaib.OpenAIEmbeddingsBatchStrategy(_service_with(client))
        await strategy.cancel_batch('batch_emb_1')
        assert client.cancelled == ['/batches/batch_emb_1/cancel']


class TestExtractGroupVectors:
    def test_non_200_row_is_none(self):
        row = {'response': {'status_code': 500, 'body': {}}}
        assert oaib._extract_group_vectors(row) is None

    def test_empty_data_is_none(self):
        row = {'response': {'status_code': 200, 'body': {'data': []}}}
        assert oaib._extract_group_vectors(row) is None

    def test_zero_vector_survives_unnormalized(self):
        row = {'response': {'status_code': 200, 'body': {'data': [
            {'index': 0, 'embedding': [0.0, 0.0]},
        ]}}}
        vectors = oaib._extract_group_vectors(row)
        assert np.allclose(vectors[0], [0.0, 0.0])

    def test_dimensionless_config_omits_the_field(self):
        service = EmbeddingService(EmbeddingConfig(
            api_key='test-key', model='emb-test-model', dimensions=None))
        strategy = oaib.OpenAIEmbeddingsBatchStrategy(service)
        line = strategy._build_batch_line(
            oaib.EmbeddingBatchRequest(custom_id='grp:0', texts=['alpha']))
        assert 'dimensions' not in line['body']


class TestFetchEdges:
    async def test_missing_result_files_and_anonymous_rows(self):
        completed = {
            'id': 'batch_emb_1', 'status': 'completed',
            'request_counts': {'total': 0, 'completed': 0, 'failed': 0},
        }
        client = _FakeOpenAIClient(polls=[completed], file_contents={})
        strategy = oaib.OpenAIEmbeddingsBatchStrategy(_service_with(client))
        # no output_file_id / error_file_id on the batch -> empty results
        assert await strategy.fetch_batch_results('batch_emb_1') == {}

        anon = json.dumps({'response': {'status_code': 200, 'body': {'data': [
            {'index': 0, 'embedding': [1.0]},
        ]}}})  # row without custom_id: skipped, not crashed on
        completed_with_files = dict(completed,
                                    output_file_id='f_out', error_file_id='f_err')
        client = _FakeOpenAIClient(
            polls=[completed_with_files],
            file_contents={'f_out': anon, 'f_err': anon},
        )
        strategy = oaib.OpenAIEmbeddingsBatchStrategy(_service_with(client))
        assert await strategy.fetch_batch_results('batch_emb_1') == {}


class _FakeStrategy:
    """Records submitted groups; replays canned vectors keyed by custom_id."""

    def __init__(self, override_results=None):
        self.requests = []
        self.polled = []
        self._override = override_results

    async def submit_batch(self, requests):
        self.requests = list(requests)
        return BatchSubmission(batch_id='batch_fake_1')

    async def poll_batch(self, batch_id, *, poll_interval=30.0, on_progress=None):
        self.polled.append(batch_id)
        if on_progress is not None:
            on_progress(0, len(self.requests), 0)
        return BatchStatus(batch_id=batch_id, processing_status='ended',
                           processing=0, succeeded=len(self.requests), errored=0)

    async def fetch_batch_results(self, batch_id):
        if self._override is not None:
            return self._override(self.requests)
        return {
            r.custom_id: [np.full(8, 0.5, dtype=np.float32) for _ in r.texts]
            for r in self.requests
        }


class TestWriterBatchRebuild:
    async def test_batch_rebuild_writes_doc_and_chunk_vectors(self, tmp_path):
        lib = Library(tmp_path / 'lib.db')
        short = lib.add_document(content_type='explanation', title='Short Doc',
                                 content='tiny body')
        long_content = ' '.join(f'word{i}' for i in range(80))
        long = lib.add_document(content_type='explanation', title='Long Doc',
                                content=long_content)
        writer = LibraryWriter(lib, chunk_config=ChunkConfig(
            chunk_size=40, chunk_overlap=5))
        strategy = _FakeStrategy()
        seen: list[tuple[int, int]] = []
        submitted: list[str] = []

        count = await writer.rebuild_all_embeddings_batch(
            strategy, only_missing=True,
            on_submit=submitted.append,
            on_progress=lambda done, total: seen.append((done, total)),
        )

        assert count == 2
        assert lib.get_document(short.id).embedding is not None
        assert lib.get_document(long.id).embedding is not None
        chunks = lib.get_chunks(long.id)
        assert chunks and all(c.embedding is not None for c in chunks)
        assert lib.get_chunks(short.id) == []
        # one grouped doc request, plus one chunk group for the long doc
        ids = [r.custom_id for r in strategy.requests]
        assert ids[0] == 'doc:0'
        assert f'chunks:{long.id}' in ids
        assert {t.split('\n')[0] for t in strategy.requests[0].texts} == {'Short Doc', 'Long Doc'}
        assert strategy.polled == ['batch_fake_1']
        assert submitted == ['batch_fake_1']
        n_groups = len(strategy.requests)
        assert seen == [(n_groups, n_groups)]

    async def test_failed_group_leaves_docs_null_for_a_live_rerun(self, tmp_path):
        lib = Library(tmp_path / 'lib.db')
        doc = lib.add_document(content_type='explanation', title='Doc A',
                               content='body')
        writer = LibraryWriter(lib)
        strategy = _FakeStrategy(
            override_results=lambda reqs: {r.custom_id: None for r in reqs})

        count = await writer.rebuild_all_embeddings_batch(strategy, only_missing=True)

        assert count == 0
        assert lib.get_document(doc.id).embedding is None

    async def test_empty_target_set_submits_nothing(self, tmp_path):
        lib = Library(tmp_path / 'lib.db')
        writer = LibraryWriter(lib)
        strategy = _FakeStrategy()

        assert await writer.rebuild_all_embeddings_batch(strategy, only_missing=True) == 0
        assert strategy.requests == []
        assert strategy.polled == []

    async def test_failed_chunk_group_keeps_doc_vector_and_existing_chunks(self, tmp_path):
        lib = Library(tmp_path / 'lib.db')
        long_content = ' '.join(f'word{i}' for i in range(80))
        doc = lib.add_document(content_type='explanation', title='Long Doc',
                               content=long_content)
        writer = LibraryWriter(lib, chunk_config=ChunkConfig(
            chunk_size=40, chunk_overlap=5))

        def only_docs_succeed(reqs):
            return {
                r.custom_id: (
                    [np.full(8, 0.5, dtype=np.float32) for _ in r.texts]
                    if r.custom_id.startswith('doc:') else None
                )
                for r in reqs
            }

        strategy = _FakeStrategy(override_results=only_docs_succeed)
        count = await writer.rebuild_all_embeddings_batch(strategy, only_missing=True)

        assert count == 1
        assert lib.get_document(doc.id).embedding is not None
        assert lib.get_chunks(doc.id) == []  # failed chunk group: nothing written

    async def test_oversized_doc_with_no_chunkable_text_gets_no_chunk_group(
            self, tmp_path, monkeypatch):
        import writer as writer_module
        monkeypatch.setattr(writer_module, 'chunk_text', lambda content, config: [])
        lib = Library(tmp_path / 'lib.db')
        long_content = ' '.join(f'word{i}' for i in range(80))
        lib.add_document(content_type='explanation', title='Long Doc',
                         content=long_content)
        writer = LibraryWriter(lib, chunk_config=ChunkConfig(
            chunk_size=40, chunk_overlap=5))
        strategy = _FakeStrategy()

        count = await writer.rebuild_all_embeddings_batch(strategy, only_missing=True)

        assert count == 1
        assert [r.custom_id for r in strategy.requests] == ['doc:0']
