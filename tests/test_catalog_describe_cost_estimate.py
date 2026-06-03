"""Pins the ``--dry-run`` cost-estimate contract for catalog-describe.

The current ``--max-calls`` flag's help text mentions "useful for
dry-run cost estimates" but doesn't actually compute one — users have
to run a small sample, observe token logs, and extrapolate manually.

This file pins a real dry-run: it counts candidates, multiplies by an
empirical per-call token estimate, looks up the model rate in
``docgen.pricing.LLM_PRICING``, prints a cost preview, and exits
WITHOUT making any LLM calls.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock

import pytest


class TestCatalogDescribeDryRun:
    @pytest.fixture(autouse=True)
    def _config(self, tmp_path: Path, monkeypatch):
        from tests._scoped_config_fixture import install_test_config
        install_test_config(monkeypatch, tmp_path, 'product')

    # ---- demand 1: --dry-run makes zero LLM calls ----------------------
    @pytest.mark.asyncio
    async def test_dry_run_makes_no_llm_calls(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        """The whole point of dry-run: estimate cost WITHOUT paying.
        Any call to chat_complete during dry-run is a contract
        violation. We count invocations rather than relying on an
        exception (the worker swallows exceptions into a failure count
        and still returns rc=0).
        """
        import argparse
        from library import Library

        library = Library(tmp_path / 'library.db')
        try:
            library.add_document(
                content_type='catalog',
                title='product.module.func',
                content='def func(): pass',
                source_name='product',
                source_files=['product/module.py'],
                metadata={
                    'kind': 'element',
                    'source_name': 'product',
                    'qualified_name': 'product.module.func',
                    'subtype': 'function',
                },
            )
        finally:
            library.close()

        call_count = {'n': 0}
        async def counting_chat_complete(*a, **kw):
            call_count['n'] += 1
            return 'fake description'

        monkeypatch.setattr(
            'docgen.catalog_describer.chat_complete', counting_chat_complete,
        )
        monkeypatch.setattr(
            'cli.generation.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        args = argparse.Namespace(
            source='product', force=False, model=None, concurrency=4,
            max_calls=None, db=None, dry_run=True,
        )
        from cli.generation import cmd_catalog_describe
        rc = await cmd_catalog_describe(args)
        assert rc == 0
        assert call_count['n'] == 0, (
            f'dry-run made {call_count["n"]} LLM call(s); expected 0'
        )

    # ---- demand 2: dry-run output names the candidate count and cost ---
    @pytest.mark.asyncio
    async def test_dry_run_prints_candidate_count_and_cost(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        """The dry-run output must include the number of candidates
        AND a dollar figure so the user can decide whether to proceed.
        """
        import argparse
        from library import Library

        library = Library(tmp_path / 'library.db')
        try:
            for i in range(3):
                library.add_document(
                    content_type='catalog',
                    title=f'product.func{i}',
                    content=f'def func{i}(): pass',
                    source_name='product',
                    source_files=['product/mod.py'],
                    metadata={
                        'kind': 'element',
                        'source_name': 'product',
                        'qualified_name': f'product.func{i}',
                        'subtype': 'function',
                    },
                )
        finally:
            library.close()

        monkeypatch.setattr(
            'docgen.catalog_describer.chat_complete',
            AsyncMock(side_effect=AssertionError(
                'chat_complete called during dry-run',
            )),
        )
        monkeypatch.setattr(
            'cli.generation.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        args = argparse.Namespace(
            source='product', force=False,
            # Use a model we know is in LLM_PRICING so the rate lookup
            # produces a concrete number.
            model='claude-opus-4-7',
            concurrency=4, max_calls=None, db=None, dry_run=True,
        )
        from cli.generation import cmd_catalog_describe
        rc = await cmd_catalog_describe(args)
        assert rc == 0

        out = capsys.readouterr().out
        # Candidate count visible.
        assert '3' in out, (
            f'expected candidate count 3 in dry-run output; got: {out!r}'
        )
        # Dollar figure visible (any $ amount).
        assert '$' in out, (
            f'expected a $ cost figure in dry-run output; got: {out!r}'
        )
