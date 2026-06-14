"""Evolutionary-TDD walk for batched catalog-describe.

The existing ``describe_source_elements`` makes one LLM call per
element. Anthropic's Message Batches API offers ~50% off and processes
up to 100,000 requests asynchronously — perfect fit for the
many-small-prompts shape of catalog-describe.

This file grows one demand at a time. Each cycle adds a behavioral
demand to ``TestDescribeBatched``; the test file *is* the spec for
the batched code path.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest


class _FakeProvider:
    """Stand-in for ``AnthropicProvider`` with deterministic responses."""

    def __init__(self, batch_id: str = 'batch_test_001'):
        self.submitted_requests: list = []
        self.batch_id = batch_id
        self.poll_calls = 0
        self.fetch_called = False
        self._results: dict[str, str | None] = {}

    async def submit_batch(self, requests):
        from docgen.llm.anthropic import BatchSubmission
        self.submitted_requests = list(requests)
        return BatchSubmission(batch_id=self.batch_id)

    async def poll_batch(self, batch_id, **kw):
        from docgen.llm.anthropic import BatchStatus
        self.poll_calls += 1
        n = len(self._results)
        return BatchStatus(
            batch_id=batch_id,
            processing_status='ended',
            processing=0,
            succeeded=sum(1 for v in self._results.values() if v),
            errored=sum(1 for v in self._results.values() if not v),
        )

    async def fetch_batch_results(self, batch_id):
        self.fetch_called = True
        return self._results

    def set_results(self, results: dict[str, str | None]) -> None:
        self._results = results


@pytest.fixture(autouse=True)
def _mock_embedding(monkeypatch):
    """Stub the embedding service so writer.update_document_embedding
    doesn't reach for OPENAI_API_KEY during these tests."""
    import numpy as np

    async def fake_embed(self, text):
        return np.ones(3072, dtype=np.float32) / 100

    async def fake_embed_batch(self, texts):
        return [np.ones(3072, dtype=np.float32) / 100 for _ in texts]

    async def fake_get_client(self):
        return None

    async def fake_close(self):
        return None

    monkeypatch.setattr('embedding.EmbeddingService.embed', fake_embed)
    monkeypatch.setattr(
        'embedding.EmbeddingService.embed_batch', fake_embed_batch,
    )
    monkeypatch.setattr(
        'embedding.EmbeddingService._get_client', fake_get_client,
    )
    monkeypatch.setattr('embedding.EmbeddingService.close', fake_close)


class TestDescribeBatched:
    # ---- T1 ----------------------------------------------------------
    # Smallest demand: with one element to describe, the function
    # calls ``provider.submit_batch`` exactly once with exactly one
    # BatchRequest whose ``custom_id`` is the element's doc_id.
    @pytest.mark.asyncio
    async def test_t1_single_element_submits_one_batch_request(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter

        with Library(tmp_path / 'library.db') as library:
            doc = library.add_document(
                content_type='catalog',
                title='product.module.foo',
                content='def foo(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element',
                    'source_name': 'product',
                    'qualified_name': 'product.module.foo',
                    'subtype': 'function',
                },
            )

            provider = _FakeProvider()
            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                )

            assert len(provider.submitted_requests) == 1, (
                f'expected exactly 1 batch request, got '
                f'{len(provider.submitted_requests)}'
            )
            req = provider.submitted_requests[0]
            assert req.custom_id == doc.id, (
                f'batch request custom_id should be the element doc.id '
                f'so results can be re-attached; got {req.custom_id!r}'
            )

    # ---- T2 ----------------------------------------------------------
    # Multiple elements → ONE batch with N requests. Each request's
    # ``custom_id`` is unique (so results can be correctly routed
    # back) and equals its element's doc_id.
    @pytest.mark.asyncio
    async def test_t2_multiple_elements_in_one_batch(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter

        with Library(tmp_path / 'library.db') as library:
            docs = [
                library.add_document(
                    content_type='catalog',
                    title=f'product.foo_{i}',
                    content=f'def foo_{i}(): pass',
                    source_name='product',
                    source_files=['product/mod.py'],
                    metadata={
                        'kind': 'element',
                        'source_name': 'product',
                        'qualified_name': f'product.foo_{i}',
                        'subtype': 'function',
                    },
                ) for i in range(5)
            ]

            provider = _FakeProvider()
            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                result = await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                )

            assert len(provider.submitted_requests) == 5
            assert result['submitted'] == 5
            # All custom_ids distinct.
            cids = [r.custom_id for r in provider.submitted_requests]
            assert len(set(cids)) == 5
            # Each cid matches one of the seeded doc ids.
            doc_ids = {d.id for d in docs}
            assert set(cids) == doc_ids

    # ---- T3 ----------------------------------------------------------
    # After a completed batch's results are fetched, each element's
    # metadata.description is set to the returned text. ``described``
    # in the return summary equals the number of successful rows.
    @pytest.mark.asyncio
    async def test_t3_applies_results_to_docs(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter

        with Library(tmp_path / 'library.db') as library:
            doc_a = library.add_document(
                content_type='catalog', title='product.a',
                content='def a(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.a',
                    'subtype': 'function',
                },
            )
            doc_b = library.add_document(
                content_type='catalog', title='product.b',
                content='def b(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.b',
                    'subtype': 'function',
                },
            )

            provider = _FakeProvider()
            # Pre-seed results: A gets a real description, B is the
            # second test in this cycle.
            provider.set_results({
                doc_a.id: 'A is a no-op function.',
                doc_b.id: 'B is also a no-op function.',
            })

            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                result = await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                )

            # Both docs got their descriptions written.
            updated_a = library.get_document(doc_a.id)
            updated_b = library.get_document(doc_b.id)
            assert updated_a.metadata.get('description') == (
                'A is a no-op function.'
            )
            assert updated_b.metadata.get('description') == (
                'B is also a no-op function.'
            )
            assert result['described'] == 2
            assert result.get('failed', 0) == 0
            # Provider was polled at least once and fetched once.
            assert provider.poll_calls >= 1
            assert provider.fetch_called

    # ---- T4 ----------------------------------------------------------
    # Partial failures don't abort the whole apply step. The N
    # successful rows are written; the M failures are counted in
    # ``failed`` and ``described`` reflects the successes.
    @pytest.mark.asyncio
    async def test_t4_partial_failures_dont_abort_apply(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter

        with Library(tmp_path / 'library.db') as library:
            ok_doc = library.add_document(
                content_type='catalog', title='product.ok',
                content='def ok(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.ok',
                    'subtype': 'function',
                },
            )
            fail_doc = library.add_document(
                content_type='catalog', title='product.failed',
                content='def failed(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.failed',
                    'subtype': 'function',
                },
            )

            provider = _FakeProvider()
            provider.set_results({
                ok_doc.id: 'OK description.',
                fail_doc.id: None,  # Anthropic returned an error for this row
            })

            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                result = await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                )

            # ok_doc got its description.
            updated_ok = library.get_document(ok_doc.id)
            assert updated_ok.metadata.get('description') == (
                'OK description.'
            )
            # fail_doc remains without a description.
            updated_fail = library.get_document(fail_doc.id)
            assert not updated_fail.metadata.get('description')
            # Counts reflect the partial outcome.
            assert result['described'] == 1
            assert result['failed'] == 1

    # ---- T4b ---------------------------------------------------------
    # A provider that returns a SHORT result set — a submitted custom_id
    # absent from the results map entirely — must surface those as
    # failures, not silently drop them. Anthropic returns a row per id;
    # OpenAI's split output/error files can come back short on a
    # wholesale-batch failure ('failed'/'expired').
    @pytest.mark.asyncio
    async def test_t4b_missing_result_rows_counted_as_failed(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter

        with Library(tmp_path / 'library.db') as library:
            ok_doc = library.add_document(
                content_type='catalog', title='product.ok',
                content='def ok(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.ok',
                    'subtype': 'function',
                },
            )
            dropped_doc = library.add_document(
                content_type='catalog', title='product.dropped',
                content='def dropped(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.dropped',
                    'subtype': 'function',
                },
            )

            provider = _FakeProvider()
            # Only ok_doc comes back; dropped_doc's row is missing entirely.
            provider.set_results({ok_doc.id: 'OK description.'})

            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                result = await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                )

            assert result['described'] == 1
            # The dropped element is a failure, not a silent no-op.
            assert result['failed'] == 1
            updated_dropped = library.get_document(dropped_doc.id)
            assert not updated_dropped.metadata.get('description')

    # ---- T6 ----------------------------------------------------------
    # Resume: when a pending batch exists for this source, re-running
    # with --batch should NOT submit again — it should fetch results
    # from the existing batch_id and apply them. Persistence in the
    # ``pending_batches`` table is how the new function discovers the
    # in-flight batch across process invocations.
    @pytest.mark.asyncio
    async def test_t6_resume_uses_pending_batch_without_resubmit(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter
        from docgen.staleness import StalenessTracker

        # Seed one element + a pending batch row claiming a batch
        # already exists for this source. The config_hash format the
        # implementation uses is documented in catalog_describer.
        with Library(tmp_path / 'library.db') as library:
            doc = library.add_document(
                content_type='catalog', title='product.foo',
                content='def foo(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.foo',
                    'subtype': 'function',
                },
            )

            staleness_db = tmp_path / 'staleness.db'
            tracker = StalenessTracker(staleness_db)
            try:
                tracker.record_pending_batch(
                    batch_id='resumed_batch_id',
                    prompts_json='[]',
                    file_to_idxs_json='{}',
                    config_hash=(
                        f'catalog-describe::product::claude-opus-4-7'
                    ),
                )
            finally:
                tracker.close()

            provider = _FakeProvider(batch_id='this_should_not_be_used')
            provider.set_results({doc.id: 'Resumed description.'})

            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                result = await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                    resume=True,
                    staleness_db_path=staleness_db,
                )

            # The function must NOT have submitted a new batch.
            assert provider.submitted_requests == [], (
                'resume should fetch from pending batch, not submit '
                f'a new one; got {len(provider.submitted_requests)} '
                'submissions'
            )
            # It must have fetched results.
            assert provider.fetch_called
            # And applied them.
            updated = library.get_document(doc.id)
            assert updated.metadata.get('description') == (
                'Resumed description.'
            )
            assert result.get('described', 0) == 1
            # The result should reference the resumed batch_id.
            assert result.get('batch_id') == 'resumed_batch_id'

    # ---- T8 ----------------------------------------------------------
    # Auto-resume: when a pending batch exists and resume=False (the
    # default), the function must adopt the pending batch rather than
    # submit a new one. This is the contract that makes onboard's
    # batched runs ctrl-C-safe without an explicit --resume.
    @pytest.mark.asyncio
    async def test_t8_auto_resume_when_pending_exists_and_resume_false(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter
        from docgen.staleness import StalenessTracker

        with Library(tmp_path / 'library.db') as library:
            doc = library.add_document(
                content_type='catalog', title='product.bar',
                content='def bar(): pass',
                source_name='product',
                source_files=['product/mod.py'],
                metadata={
                    'kind': 'element', 'source_name': 'product',
                    'qualified_name': 'product.bar',
                    'subtype': 'function',
                },
            )

            staleness_db = tmp_path / 'staleness.db'
            tracker = StalenessTracker(staleness_db)
            try:
                tracker.record_pending_batch(
                    batch_id='auto_resumed_batch_id',
                    prompts_json='[]',
                    file_to_idxs_json='{}',
                    config_hash=(
                        f'catalog-describe::product::claude-opus-4-7'
                    ),
                )
            finally:
                tracker.close()

            provider = _FakeProvider(batch_id='should_not_be_submitted')
            provider.set_results({doc.id: 'Auto-resumed description.'})

            async with LibraryWriter(library) as writer:
                from docgen.catalog_describer import (
                    describe_source_elements_batched,
                )
                # resume=False here — auto-resume must still kick in.
                result = await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider,
                    model='claude-opus-4-7',
                    resume=False,
                    staleness_db_path=staleness_db,
                )

            assert provider.submitted_requests == [], (
                'auto-resume must adopt pending batch without resubmit; '
                f'got {len(provider.submitted_requests)} submissions'
            )
            assert provider.fetch_called
            updated = library.get_document(doc.id)
            assert updated.metadata.get('description') == (
                'Auto-resumed description.'
            )
            assert result.get('described', 0) == 1
            assert result.get('batch_id') == 'auto_resumed_batch_id'
            assert result.get('resumed') is True, (
                'auto-resume should report resumed=True in the summary'
            )

    # ---- T9 (evolving): staged progress + concurrent apply ----------
    # The batch lifecycle has four stages — submit, processing,
    # download, apply/embed — but only polling was ever surfaced, so a
    # run looked "done" (bar full) while it silently re-embedded tens of
    # thousands of docs. This test grows to pin: (a) the apply/embed
    # stage reports progress; (b) all four stages are announced in
    # order; (c) the re-embed loop runs concurrently.
    @pytest.mark.asyncio
    async def test_t9_staged_progress_and_concurrent_apply(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        from library import Library
        from writer import LibraryWriter
        from docgen.catalog_describer import (
            describe_source_elements_batched,
        )

        with Library(tmp_path / 'library.db') as library:
            docs = [
                library.add_document(
                    content_type='catalog',
                    title=f'product.foo_{i}',
                    content=f'def foo_{i}(): pass',
                    source_name='product',
                    source_files=['product/mod.py'],
                    metadata={
                        'kind': 'element',
                        'source_name': 'product',
                        'qualified_name': f'product.foo_{i}',
                        'subtype': 'function',
                    },
                ) for i in range(4)
            ]
            provider = _FakeProvider()
            provider.set_results({d.id: f'desc {i}' for i, d in enumerate(docs)})

            stages: list[tuple] = []

            # Track how many re-embeds are in flight at once, to prove the
            # apply/embed loop runs concurrently rather than one-at-a-time.
            import asyncio
            inflight = {'cur': 0, 'max': 0}

            async def tracking_embed(doc_id):
                inflight['cur'] += 1
                inflight['max'] = max(inflight['max'], inflight['cur'])
                await asyncio.sleep(0.02)
                inflight['cur'] -= 1

            async with LibraryWriter(library) as writer:
                monkeypatch.setattr(
                    writer, 'update_document_embedding', tracking_embed,
                )
                await describe_source_elements_batched(
                    library, writer, 'product',
                    strategy=provider, model='claude-opus-4-7',
                    concurrency=4,
                    on_stage=lambda stage, completed, total: stages.append(
                        (stage, completed, total),
                    ),
                )

            # (a) The apply/embed stage reports progress reaching all 4.
            apply = [s for s in stages if s[0] == 'apply']
            assert apply, f'apply/embed stage must report progress; got {stages}'
            assert apply[-1][1] == 4, (
                f'apply stage must reach all 4 docs; got {apply}'
            )

            # (b) All four lifecycle stages are announced, in order, so
            # the user sees submit → process → download → apply rather
            # than a single bar that freezes after processing.
            seen_order: list[str] = []
            for stage, *_ in stages:
                if stage not in seen_order:
                    seen_order.append(stage)
            assert seen_order == [
                'submit', 'processing', 'download', 'apply',
            ], f'stages must be announced in lifecycle order; got {seen_order}'

            # (c) The re-embed loop overlaps work (concurrency>1) rather
            # than awaiting each doc sequentially — otherwise 64k docs
            # crawl one at a time.
            assert inflight['max'] >= 2, (
                f'apply/embed must run concurrently; max in-flight was '
                f'{inflight["max"]} (sequential)'
            )

    # ---- T10 (calibration capture) ----------------------------------
    # When a run sets a usage observer, the real token usage the provider
    # reports (emit_usage) flows through the describer's usage_context
    # into the CalibrationStore — so the next dry-run self-calibrates.
    @pytest.mark.asyncio
    async def test_t10_batch_usage_is_captured_to_store(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter
        from docgen.calibration import (
            CalibrationStore, emit_usage, set_usage_observer,
        )
        from docgen.catalog_describer import (
            describe_source_elements_batched,
        )

        class _EmittingProvider(_FakeProvider):
            """Like _FakeProvider, but reports token usage on fetch —
            mirroring the real provider's emit_usage call."""

            async def fetch_batch_results(self, batch_id):
                for _cid in self._results:
                    emit_usage(
                        model='claude-opus-4-8',
                        input_tokens=512, output_tokens=88,
                    )
                return await super().fetch_batch_results(batch_id)

        with Library(tmp_path / 'library.db') as library:
            docs = [
                library.add_document(
                    content_type='catalog', title=f'product.f_{i}',
                    content=f'def f_{i}(): pass', source_name='product',
                    source_files=['product/mod.py'],
                    metadata={'kind': 'element', 'source_name': 'product',
                              'qualified_name': f'product.f_{i}',
                              'subtype': 'function'},
                ) for i in range(2)
            ]
            provider = _EmittingProvider()
            provider.set_results({d.id: f'desc {i}' for i, d in enumerate(docs)})

            store = CalibrationStore(tmp_path / 'cal.db')
            async with LibraryWriter(library) as writer:
                with set_usage_observer(store.record):
                    await describe_source_elements_batched(
                        library, writer, 'product',
                        strategy=provider, model='claude-opus-4-8',
                    )

        cal = store.mean_tokens(phase='describe', model='claude-opus-4-8')
        assert cal is not None, 'usage must be captured into the store'
        assert cal.n == 2
        assert cal.mean_input == 512.0 and cal.mean_output == 88.0

    # ---- T11 (abort → cancel) ---------------------------------------
    # Aborting (Ctrl-C) during the batch poll must CANCEL the in-flight
    # batch at Anthropic — otherwise it keeps processing and charging
    # with no way to stop it but manual intervention. The interrupt must
    # still propagate after the cancel.
    @pytest.mark.asyncio
    async def test_t11_abort_during_poll_cancels_batch(
        self, tmp_path: Path,
    ) -> None:
        from library import Library
        from writer import LibraryWriter
        from docgen.catalog_describer import (
            describe_source_elements_batched,
        )

        class _AbortingProvider(_FakeProvider):
            def __init__(self):
                super().__init__()
                self.cancelled: list = []

            async def poll_batch(self, batch_id, **kw):
                raise KeyboardInterrupt  # user hits Ctrl-C mid-poll

            async def cancel_batch(self, batch_id):
                self.cancelled.append(batch_id)

        with Library(tmp_path / 'library.db') as library:
            library.add_document(
                content_type='catalog', title='product.f',
                content='def f(): pass', source_name='product',
                source_files=['product/mod.py'],
                metadata={'kind': 'element', 'source_name': 'product',
                          'qualified_name': 'product.f',
                          'subtype': 'function'},
            )
            provider = _AbortingProvider()
            async with LibraryWriter(library) as writer:
                with pytest.raises(KeyboardInterrupt):
                    await describe_source_elements_batched(
                        library, writer, 'product',
                        strategy=provider, model='m',
                    )
            assert provider.cancelled == ['batch_test_001'], (
                'aborting mid-poll must cancel the in-flight batch'
            )

    # ---- T5 ----------------------------------------------------------
    # The CLI ``catalog-describe --batch`` routes to the batched path.
    # Verified by: with --batch set, ``describe_source_elements_batched``
    # is called and ``describe_source_elements`` (the live path) is NOT.
    # Without --batch, the inverse holds.
    @pytest.mark.asyncio
    async def test_t5_cli_flag_routes_to_batched(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import argparse
        from library import Library

        # cmd_catalog_describe rejects --batch / --resume when the resolved
        # provider's API key is missing; tests bypass the real strategy but
        # still hit the precondition. The test config resolves to the openai
        # default (gpt-5.5), so set OPENAI_API_KEY.
        monkeypatch.setenv('OPENAI_API_KEY', 'test-key')

        # Track which path was invoked.
        called: dict[str, int] = {'live': 0, 'batched': 0}

        async def fake_live(*a, **kw):
            called['live'] += 1
            return {'described': 0, 'failed': 0, 'total_candidates': 0,
                    'already_had_description': 0}

        async def fake_batched(*a, **kw):
            called['batched'] += 1
            return {'submitted': 0, 'batch_id': None,
                    'described': 0, 'failed': 0,
                    'total_candidates': 0}

        monkeypatch.setattr(
            'docgen.catalog_describer.describe_source_elements', fake_live,
        )
        monkeypatch.setattr(
            'docgen.catalog_describer.describe_source_elements_batched',
            fake_batched,
        )

        # Minimal config + library.
        from tests._scoped_config_fixture import install_test_config
        install_test_config(monkeypatch, tmp_path, 'product')
        source_dir = tmp_path / 'src' / 'product'
        source_dir.mkdir(parents=True)
        from config import Config
        cfg_dir = tmp_path / 'cfg'
        cfg_dir.mkdir()
        (cfg_dir / 'ariadne.yaml').write_text(
            f'sources:\n  product:\n    path: {source_dir}\n',
            encoding='utf-8',
        )
        monkeypatch.setattr(
            'config.get_config',
            lambda: Config(cfg_dir / 'ariadne.yaml'),
        )
        monkeypatch.setattr(
            'cli.catalog.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        from cli.catalog import cmd_catalog_describe

        # Without --batch: live path runs.
        args = argparse.Namespace(
            source='product', force=False, model=None,
            concurrency=4, max_calls=None, db=None,
            dry_run=False, batch=False,
        )
        rc = await cmd_catalog_describe(args)
        assert rc == 0
        assert called == {'live': 1, 'batched': 0}, (
            f'Without --batch the live path should run; got {called}'
        )

        # With --batch: batched path runs.
        args.batch = True
        args.resume = False
        rc = await cmd_catalog_describe(args)
        assert rc == 0
        assert called == {'live': 1, 'batched': 1}, (
            f'With --batch the batched path should run; got {called}'
        )

        # ---- T7: --resume routes to batched path with resume=True ----
        # Capture the kwargs the batched function receives so we can
        # verify resume is propagated.
        captured_resume: list[bool] = []

        async def fake_batched_capturing(*a, **kw):
            captured_resume.append(kw.get('resume', False))
            return {
                'submitted': 0, 'batch_id': 'resume_test_batch',
                'described': 0, 'failed': 0, 'total_candidates': 0,
                'resumed': kw.get('resume', False),
            }
        monkeypatch.setattr(
            'docgen.catalog_describer.describe_source_elements_batched',
            fake_batched_capturing,
        )

        args.batch = False
        args.resume = True
        rc = await cmd_catalog_describe(args)
        assert rc == 0
        assert captured_resume == [True], (
            f'--resume should route to batched path with resume=True; '
            f'got captured_resume={captured_resume}'
        )

    # ---- T12 (provider-aware batch selection) ------------------------
    # The CLI batch path must pick the batch strategy by the *resolved*
    # provider — Anthropic Message Batches for claude-*, OpenAI's Batch API
    # for gpt-* — and check that provider's API key, not always Anthropic's.
    # This is what makes ``catalog-describe --batch`` work with an OpenAI
    # model instead of crashing on the old hardcoded AnthropicProvider.
    @pytest.mark.asyncio
    async def test_t12_cli_batch_selects_provider_strategy(
        self, tmp_path: Path, monkeypatch,
    ) -> None:
        import argparse

        from config import Config
        from library import Library
        from tests._scoped_config_fixture import install_test_config

        install_test_config(monkeypatch, tmp_path, 'product')
        source_dir = tmp_path / 'src' / 'product'
        source_dir.mkdir(parents=True)
        monkeypatch.setattr(
            'cli.catalog.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        captured: dict = {}

        async def fake_batched(*a, **kw):
            captured['strategy'] = kw.get('strategy')
            return {'submitted': 0, 'batch_id': None, 'described': 0,
                    'failed': 0, 'total_candidates': 0}

        monkeypatch.setattr(
            'docgen.catalog_describer.describe_source_elements_batched',
            fake_batched,
        )

        from cli.catalog import cmd_catalog_describe

        # provider is set explicitly: cfg.provider defaults to 'openai', so a
        # claude model with no provider: would be a fail-fast mismatch in
        # resolve_provider rather than resolving to anthropic.
        def _configure(provider: str, model: str) -> None:
            cfg_dir = tmp_path / 'cfg'
            cfg_dir.mkdir(exist_ok=True)
            (cfg_dir / 'ariadne.yaml').write_text(
                f'defaults:\n  provider: {provider}\n  model: {model}\n'
                f'sources:\n  product:\n    path: {source_dir}\n',
                encoding='utf-8',
            )
            # cmd_catalog_describe binds get_config at import (``from config
            # import get_config``), so patch the name in *its* module, not
            # ``config.get_config`` (which would be a no-op here).
            monkeypatch.setattr(
                'cli.catalog.get_config',
                lambda: Config(cfg_dir / 'ariadne.yaml'),
            )

        args = argparse.Namespace(
            source='product', force=False, model=None, concurrency=4,
            max_calls=None, db=None, dry_run=False, batch=True, resume=False,
        )

        # (a) gpt-* → OpenAIBatchStrategy, keyed on OPENAI_API_KEY.
        from docgen.llm.openai_batch import OpenAIBatchStrategy
        _configure('openai', 'gpt-5.5')
        monkeypatch.setenv('OPENAI_API_KEY', 'oai-key')
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        assert await cmd_catalog_describe(args) == 0
        assert isinstance(captured['strategy'],OpenAIBatchStrategy), (
            f'gpt-* batch must use OpenAIBatchStrategy; got '
            f'{type(captured.get("strategy")).__name__}'
        )

        # (b) claude-* → AnthropicBatchStrategy, keyed on ANTHROPIC_API_KEY.
        from docgen.llm.anthropic_batch import AnthropicBatchStrategy
        captured.clear()
        _configure('anthropic', 'claude-opus-4-8')
        monkeypatch.setenv('ANTHROPIC_API_KEY', 'ant-key')
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        assert await cmd_catalog_describe(args) == 0
        assert isinstance(captured['strategy'],AnthropicBatchStrategy), (
            f'claude-* batch must use AnthropicBatchStrategy; got '
            f'{type(captured.get("strategy")).__name__}'
        )

        # (c) Missing the resolved provider's key → clean error, no dispatch.
        captured.clear()
        _configure('openai', 'gpt-5.5')
        monkeypatch.delenv('OPENAI_API_KEY', raising=False)
        monkeypatch.delenv('ANTHROPIC_API_KEY', raising=False)
        assert await cmd_catalog_describe(args) == 1
        assert 'strategy' not in captured
