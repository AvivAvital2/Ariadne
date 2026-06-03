"""Pins the ``ariadne dry-run`` wrapper command.

Contract:
- Executes ``discover``, ``index``, ``catalog-sync`` (the free phases)
  in order, end-to-end.
- For each LLM-paid phase (``catalog-describe``, ``generate``,
  ``themes build``), runs the dry-run estimator inline and reports
  the per-phase cost.
- Prints a unified total and exits with 0.
- Zero ``chat_complete`` invocations from start to finish.
"""
from __future__ import annotations

from pathlib import Path

import pytest


class TestDryRunPipeline:
    @pytest.fixture(autouse=True)
    def _config(self, tmp_path: Path, monkeypatch):
        from tests._scoped_config_fixture import install_test_config
        install_test_config(monkeypatch, tmp_path, 'product')

    @pytest.mark.asyncio
    async def test_pipeline_runs_free_phases_and_estimates_rest(
        self, tmp_path: Path, monkeypatch, capsys,
    ) -> None:
        """End-to-end: dry-run wrapper executes the free phases, runs
        cost estimators for the paid phases, prints a total, makes
        zero LLM calls.
        """
        import argparse
        from library import Library

        # Mock the three free phases so the test doesn't need real
        # scip binaries / source trees. Each mock returns 0 (success)
        # and records that it was invoked in the expected order.
        invocation_log: list[str] = []

        async def mock_discover(args):
            invocation_log.append('discover')
            return 0

        async def mock_index(args):
            invocation_log.append('index')
            return 0

        async def mock_catalog_sync(args):
            invocation_log.append('catalog-sync')
            # Seed enough catalog elements that the cost estimate is
            # large enough to surface at 2-decimal-dollar precision —
            # otherwise both baseline and batched round to $0.00 and
            # the batched-vs-baseline assertion can't fire. 100 docs
            # at ~200 in + 60 out tokens each gives ~$0.25 baseline,
            # ~$0.13 batched (claude-opus-4-7 rates).
            lib = Library(tmp_path / 'library.db')
            try:
                for i in range(100):
                    lib.add_document(
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
                    )
            finally:
                lib.close()
            return 0

        # cmd_discover / cmd_index / cmd_catalog_sync are mixed
        # async/sync; we wrap each appropriately. The wrapper command
        # under test should call them via the same dispatch the CLI
        # uses, so monkeypatching the module-level functions works.
        import cli.core as cli_core
        import cli.generation as cli_generation

        def sync_to_async(fn):
            async def inner(args):
                return fn(args)
            return inner

        # Mocks that ALSO verify the namespace the dry-run wrapper
        # hands them has every attribute the real cmd_* reads. Earlier
        # mocks just appended to a log; they didn't touch args, so the
        # production cmd_discover's ``args.all`` access escaped the
        # test. Now each mock dereferences the attributes its real
        # counterpart reads — any missing attr fails the test loudly.
        def assert_discover_args(args):
            # Real cmd_discover reads these — see cli_core.py.
            for attr in (
                'source', 'all', 'dry_run', 'review', 'config_only',
            ):
                assert hasattr(args, attr), (
                    f'cmd_discover expects ``args.{attr}`` but the '
                    'dry-run namespace did not provide it'
                )
            invocation_log.append('discover')
            return 0

        def assert_index_args(args, **kwargs):
            for attr in ('source', 'all', 'dry_run', 'kind'):
                assert hasattr(args, attr), (
                    f'cmd_index expects ``args.{attr}`` but the '
                    'dry-run namespace did not provide it'
                )
            invocation_log.append('index')
            return 0

        async def assert_catalog_sync_args(args):
            for attr in (
                'source', 'allow_degraded', 'concurrency', 'force',
            ):
                assert hasattr(args, attr), (
                    f'cmd_catalog_sync expects ``args.{attr}`` but '
                    'the dry-run namespace did not provide it'
                )
            await mock_catalog_sync(args)
            return 0

        monkeypatch.setattr('cli.core.cmd_discover', assert_discover_args)
        monkeypatch.setattr('cli.core.cmd_index', assert_index_args)
        monkeypatch.setattr(
            'cli.generation.cmd_catalog_sync', assert_catalog_sync_args,
        )

        # Hard fail if any LLM call escapes.
        call_count = {'n': 0}

        async def counting_chat_complete(*a, **kw):
            call_count['n'] += 1
            return 'unexpected llm call'

        monkeypatch.setattr(
            'docgen.catalog_describer.chat_complete',
            counting_chat_complete,
        )
        monkeypatch.setattr(
            'cli.generation.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        # Provide a source path on disk so cmd_dry_run's config
        # resolution doesn't bail.
        source_dir = tmp_path / 'src' / 'product'
        source_dir.mkdir(parents=True)
        (source_dir / 'mod.py').write_text('def foo(): pass\n')

        # Rewrite the autouse config to point at the real source dir.
        from tests._scoped_config_fixture import install_test_config
        from config import Config
        cfg_dir = tmp_path / 'cfg-product'
        cfg_dir.mkdir()
        (cfg_dir / 'ariadne.yaml').write_text(
            f'sources:\n  product:\n    path: {source_dir}\n',
            encoding='utf-8',
        )
        monkeypatch.setattr(
            'config.get_config',
            lambda: Config(cfg_dir / 'ariadne.yaml'),
        )

        args = argparse.Namespace(
            source='product',
            db=None,
            model='claude-opus-4-7',
        )
        from cli.generation import cmd_dry_run
        rc = await cmd_dry_run(args)

        assert rc == 0
        assert call_count['n'] == 0, (
            f'dry-run made {call_count["n"]} LLM call(s); expected 0'
        )

        out = capsys.readouterr().out
        # Free phases ran in order.
        assert 'discover' in invocation_log
        assert 'index' in invocation_log
        assert 'catalog-sync' in invocation_log
        # Output names each paid phase + a total.
        assert 'catalog-describe' in out.lower()
        assert 'generate' in out.lower()
        assert 'themes' in out.lower()
        assert 'total' in out.lower()
        # Output contains at least one dollar figure.
        assert '$' in out

        # ---- catalog-describe line shows the WORK SIZE, not just cost -
        # The phase describes catalog *elements* (one LLM call each), and
        # that count differs sharply from the file count — surface it so
        # the user knows what 64k+ "processing" means. 100 elements were
        # seeded above.
        cd_count_line = next(
            ln for ln in out.splitlines()
            if 'catalog-describe' in ln.lower()
        )
        assert 'element' in cd_count_line.lower(), (
            f'catalog-describe line must name the unit "elements"; '
            f'got {cd_count_line!r}'
        )
        assert '100' in cd_count_line, (
            f'catalog-describe line must show the element count (100); '
            f'got {cd_count_line!r}'
        )

        # ---- Cycle T1: generate line shows TWO dollar figures --------
        # Anthropic offers a ~50% Message Batches discount; generate is
        # the only phase that batches today. The dry-run should show
        # both numbers so the user sees the trade-off without re-running.
        # Find the line containing "generate" and count $ figures on it.
        generate_lines = [
            ln for ln in out.splitlines()
            if 'generate' in ln.lower() and '$' in ln
        ]
        assert generate_lines, (
            f'expected a generate line with dollar figures; '
            f'got:\n{out}'
        )
        generate_line = generate_lines[0]
        assert generate_line.count('$') >= 2, (
            f'generate line should show baseline AND batched cost (two '
            f'$ figures); got: {generate_line!r}'
        )

        # ---- Cycle T2: batched figure is strictly less than baseline -
        # Verifies the second figure isn't a copy of the first — we
        # must actually pass ``batch_enabled=True`` to estimate_cost,
        # not just print the baseline twice.
        import re
        amounts = re.findall(r'\$([\d.]+)', generate_line)
        assert len(amounts) >= 2, (
            f'expected two parseable $ amounts on generate line; '
            f'got: {amounts!r} from {generate_line!r}'
        )
        baseline, batched = float(amounts[0]), float(amounts[1])
        assert batched < baseline, (
            f'batched cost ${batched:.2f} should be LESS than baseline '
            f'${baseline:.2f} (Anthropic\'s ~50% Message Batches '
            f'discount); same number suggests batch_enabled=True is '
            f'not actually being passed to estimate_cost'
        )

        # ---- Cycle T3: themes-build on first run is "not estimated" --
        # No clusters exist yet (clustering hasn't run), so themes-build
        # must NOT be shown as $0.00 — that implies it's free, when
        # onboard WILL cluster + summarize. It's shown as not-estimated,
        # with no $ figure, and kept out of the total.
        themes_lines = [
            ln for ln in out.splitlines() if 'themes build' in ln.lower()
        ]
        assert themes_lines, 'expected a themes-build line'
        themes_line = themes_lines[0]
        assert '$' not in themes_line, (
            f'themes-build with 0 clusters must not show a $ figure; '
            f'got {themes_line!r}'
        )
        assert 'not estimated' in themes_line.lower(), (
            f'themes-build with 0 clusters should read "not estimated"; '
            f'got {themes_line!r}'
        )

        # ---- Cycle T3.5: catalog-describe also shows TWO figures -----
        # catalog-describe gained batch support (Anthropic Message
        # Batches API), so the dry-run should mirror generate's
        # display: baseline + batched on the same line, with the
        # batched figure strictly smaller.
        cd_lines = [
            ln for ln in out.splitlines()
            if 'catalog-describe' in ln.lower() and '$' in ln
            and 'total' not in ln.lower()
        ]
        assert cd_lines, 'expected a catalog-describe line with a $'
        cd_line = cd_lines[0]
        assert cd_line.count('$') >= 2, (
            f'catalog-describe line should show baseline AND batched '
            f'cost (two $ figures); got: {cd_line!r}'
        )
        cd_amounts = re.findall(r'\$([\d.]+)', cd_line)
        cd_baseline, cd_batched = (
            float(cd_amounts[0]), float(cd_amounts[1]),
        )
        assert cd_batched < cd_baseline, (
            f'catalog-describe batched ${cd_batched:.2f} should be '
            f'less than baseline ${cd_baseline:.2f}'
        )

        # ---- Cycle T4: total reflects both scenarios -----------------
        # Since generate has two figures and the user can choose either,
        # the total line must present both — otherwise the user picks
        # one mode and has to mentally subtract to figure out the
        # alternative.
        total_lines = [
            ln for ln in out.splitlines()
            if 'total' in ln.lower() and '$' in ln
        ]
        assert total_lines, f'expected a total line with $; got:\n{out}'
        total_line = total_lines[0]
        total_amounts = re.findall(r'\$([\d.]+)', total_line)
        assert len(total_amounts) >= 2, (
            f'total line should show baseline AND batched totals (two $ '
            f'figures); got: {total_line!r}'
        )
        baseline_total = float(total_amounts[0])
        batched_total = float(total_amounts[1])
        assert batched_total < baseline_total, (
            f'batched total ${batched_total:.2f} should be less than '
            f'baseline ${baseline_total:.2f}'
        )
        # Math agreement: total delta = sum of per-phase deltas for
        # every phase that batches. As of cycle T3.5 BOTH generate
        # AND catalog-describe batch; only themes-build is single-mode.
        # So the total savings should equal generate-delta plus
        # catalog-describe-delta (within rounding tolerance).
        delta_total = baseline_total - batched_total
        delta_generate = baseline - batched
        delta_cd = cd_baseline - cd_batched
        assert abs(delta_total - (delta_generate + delta_cd)) < 0.02, (
            f'total delta ${delta_total:.2f} should equal '
            f'generate-delta (${delta_generate:.2f}) + '
            f'catalog-describe-delta (${delta_cd:.2f}); difference '
            f'suggests an off-by-one summing bug'
        )

        # ---- Cycle T4.5: total carries an uncertainty band ----------
        # The total is a char-based heuristic — present it as an estimate
        # (±50% band), not a precise quote.
        assert '±' in out, (
            f'total must show an estimate band (±50%); got:\n{out}'
        )

        # ---- Cycle T5: default mode is more compact than --verbose ---
        # The original default showed indexer adapter details, file
        # change lists, and per-phase progress bars. That belongs
        # behind a --verbose flag; the default should be a tight
        # summary so a user can read the cost at a glance.
        # Capture verbose output for the same fixture and assert it is
        # strictly longer than the default. The non-verbose run is
        # what we already produced above.
        non_verbose_out = out

        # Re-run the same scenario but with verbose=True. Need to
        # reset things the previous run consumed — easiest is to
        # re-add a fresh catalog element (the dry-run is idempotent).
        verbose_args = argparse.Namespace(
            source='product', db=None, model='claude-opus-4-7',
            verbose=True,
        )
        from cli.generation import cmd_dry_run as _cmd_dry_run
        await _cmd_dry_run(verbose_args)
        verbose_out = capsys.readouterr().out
        assert len(verbose_out) > len(non_verbose_out), (
            f'verbose mode should produce strictly more output than '
            f'default. Got {len(verbose_out)} chars verbose vs '
            f'{len(non_verbose_out)} chars default.'
        )

        # ---- Cycle T6: default cost table is uncluttered -------------
        # In default (non-verbose) mode, the cost table drops the
        # ``in=…`` and ``out=…`` token counts — those belong in
        # verbose. Only phase name + cost(s) per line.
        # The Cost-estimate section starts after the "Cost estimate"
        # header; check those lines specifically.
        cost_lines = [
            ln for ln in non_verbose_out.splitlines()
            if any(p in ln for p in (
                'catalog-describe', 'generate', 'themes build',
            )) and '$' in ln
        ]
        for ln in cost_lines:
            assert 'in=' not in ln and 'out=' not in ln, (
                f'default-mode cost line should not show token counts; '
                f'got: {ln!r}. Token detail belongs behind --verbose.'
            )
        # Verbose DOES show token counts on at least one cost line.
        verbose_cost_lines = [
            ln for ln in verbose_out.splitlines()
            if 'catalog-describe' in ln and '$' in ln
        ]
        assert verbose_cost_lines, 'expected verbose cost output'
        assert any(
            'in=' in ln for ln in verbose_cost_lines
        ), (
            'verbose mode should show token counts on cost lines'
        )

    # ---- Cycle T7: live progress indication during slow phases -------
    # The spinner alone doesn't tell the user whether the phase is
    # making progress — just that the process is alive. With a slow
    # fake sub-phase (~1s), the captured stderr output should contain
    # elapsed-time information so the user can see forward motion.
    @pytest.mark.asyncio
    async def test_default_mode_shows_elapsed_time_during_phases(
        self, tmp_path: Path, monkeypatch, capfd,
    ) -> None:
        # ``capfd`` (not ``capsys``) — the progress widget writes to
        # ``sys.__stderr__`` which capsys can't see (capsys hooks
        # sys.stderr but the spinner uses the original fd 2 directly).
        import argparse
        import asyncio
        import time
        from library import Library

        # Slow synchronous sub-phases so the Progress widget has time
        # to render at least one elapsed-time update.
        def slow_discover(args):
            time.sleep(1.2)
            return 0

        def slow_index(args, **kwargs):
            time.sleep(0.1)
            return 0

        async def slow_catalog_sync(args):
            await asyncio.sleep(0.1)
            return 0

        monkeypatch.setattr('cli.core.cmd_discover', slow_discover)
        monkeypatch.setattr('cli.core.cmd_index', slow_index)
        monkeypatch.setattr(
            'cli.generation.cmd_catalog_sync', slow_catalog_sync,
        )
        monkeypatch.setattr(
            'cli.generation.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        from config import Config
        cfg_dir = tmp_path / 'cfg2'
        cfg_dir.mkdir()
        (cfg_dir / 'ariadne.yaml').write_text(
            f'sources:\n  product:\n    path: {tmp_path}\n',
            encoding='utf-8',
        )
        monkeypatch.setattr(
            'config.get_config',
            lambda: Config(cfg_dir / 'ariadne.yaml'),
        )

        args = argparse.Namespace(
            source='product', db=None, model='claude-opus-4-7',
            verbose=False,
        )
        from cli.generation import cmd_dry_run
        await cmd_dry_run(args)

        captured = capfd.readouterr()
        # Rich Progress's TimeElapsedColumn renders as e.g. ``0:00:01``
        # or ``0:00:00`` (HH:MM:SS). The stderr stream should contain
        # at least one such timestamp during the slow phase.
        import re
        time_pattern = re.compile(r'\d:\d{2}:\d{2}')
        all_output = captured.out + captured.err
        assert time_pattern.search(all_output), (
            'expected elapsed-time indicator (HH:MM:SS) in default-'
            'mode output; got stderr={!r} stdout={!r}'.format(
                captured.err[:500], captured.out[:500],
            )
        )

    # ---- Cycle T8: cmd_index and cmd_catalog_sync's chatter is gone --
    # The progress bars are good UX, but the surrounding "Running X
    # adapter / cwd: / output: / Wrote ... / Catalog sync for X:" lines
    # are noise in dry-run's default mode. Real default-mode output
    # should NOT contain those.
    @pytest.mark.asyncio
    async def test_default_mode_suppresses_indexer_chatter(
        self, tmp_path: Path, monkeypatch, capfd,
    ) -> None:
        """Pin the contract: dry-run default mode propagates a quiet
        signal into cmd_index and cmd_catalog_sync. Each command
        respects ``args.quiet`` to skip its boilerplate prints while
        keeping its progress widget."""
        import argparse
        import asyncio
        from library import Library

        # Capture what cmd_index actually receives — does it see quiet?
        seen_index_args: list = []

        def mock_index(args, **kwargs):
            seen_index_args.append(args)
            # Simulate the chatty prints — these MUST be skipped when
            # quiet is set. We use console.print so the test verifies
            # the real suppression discipline.
            from cli.core import console as _c
            if not getattr(args, 'quiet', False):
                _c.print('Running python adapter')
                _c.print('  cwd:    /foo/bar')
            return 0

        seen_cs_args: list = []

        async def mock_catalog_sync(args):
            seen_cs_args.append(args)
            from cli.generation import console as _c
            if not getattr(args, 'quiet', False):
                _c.print(f'Catalog sync for {args.source}:')
                _c.print('  Files scanned: 318 (0 skipped)')
            return 0

        monkeypatch.setattr('cli.core.cmd_discover', lambda a: 0)
        monkeypatch.setattr('cli.core.cmd_index', mock_index)
        monkeypatch.setattr(
            'cli.generation.cmd_catalog_sync', mock_catalog_sync,
        )
        monkeypatch.setattr(
            'cli.generation.get_library',
            lambda *_a, **_kw: Library(tmp_path / 'library.db'),
        )

        from config import Config
        cfg_dir = tmp_path / 'cfg3'
        cfg_dir.mkdir()
        (cfg_dir / 'ariadne.yaml').write_text(
            f'sources:\n  product:\n    path: {tmp_path}\n',
            encoding='utf-8',
        )
        monkeypatch.setattr(
            'config.get_config',
            lambda: Config(cfg_dir / 'ariadne.yaml'),
        )

        args = argparse.Namespace(
            source='product', db=None, model='claude-opus-4-7',
            verbose=False,
        )
        from cli.generation import cmd_dry_run
        await cmd_dry_run(args)

        # The mocks each received a namespace; verify ``quiet=True``
        # was set on it by dry-run.
        assert seen_index_args and getattr(
            seen_index_args[0], 'quiet', False,
        ), (
            'cmd_index args must carry quiet=True in default-mode '
            f'dry-run; got {vars(seen_index_args[0])}'
        )
        assert seen_cs_args and getattr(
            seen_cs_args[0], 'quiet', False,
        ), (
            'cmd_catalog_sync args must carry quiet=True in '
            f'default-mode dry-run; got {vars(seen_cs_args[0])}'
        )

        # And the captured output must NOT contain the chatter lines.
        captured = capfd.readouterr()
        whole = captured.out + captured.err
        assert 'Running python adapter' not in whole, (
            'index adapter line leaked into default-mode output'
        )
        assert 'Catalog sync for' not in whole, (
            'catalog-sync final-report block leaked into default-mode output'
        )
