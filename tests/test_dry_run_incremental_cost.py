"""The dry-run cost preview must reflect the staleness DB.

Re-running `generate`/`onboard` with a populated staleness DB skips
files whose source is unchanged (DocGenOrchestrator.run filters via
`get_stale_files`). The cost preview therefore has to price only the
stale/new subset as the headline "this run" cost — pricing every
discovered file overstates a repeat run, which is the reported bug.
The full from-scratch figure stays available as a secondary note.

This evolves one scenario through: fresh DB (everything stale) →
populated DB (most files skipped).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from cli.generate_cost import _print_cost_estimate
from docgen.staleness import StalenessTracker


def _src_tree(tmp_path: Path) -> Path:
    src = tmp_path / 'src'
    src.mkdir()
    (src / 'alpha.py').write_text('def alpha():\n    return 1\n')
    (src / 'beta.py').write_text('def beta():\n    return 2\n')
    return src


def _estimate(src: Path, sdb: Path) -> int:
    return _print_cost_estimate(
        source_path=src,
        source_name='t',
        target_path=None,
        doc_types=('explanation',),
        model='gpt-5.4',
        exclude_patterns=(),
        exclude_dir_names=(),
        catalog_only=True,
        provider='openai',
        staleness_db_path=sdb,
        base_path=src,
    )


def test_preview_prices_only_stale_files(tmp_path, capsys) -> None:
    src = _src_tree(tmp_path)
    sdb = tmp_path / 'staleness.db'

    # rich word-wraps notes to the (test-time default) console width, which
    # would split multi-word phrases mid-assertion — a display artifact only.
    # Collapse all whitespace so substring checks are width-independent
    # (no need to patch a wide console).
    def _flat() -> str:
        return ' '.join(capsys.readouterr().out.split())

    # Fresh DB: nothing documented yet → both files are stale/new, so
    # the preview is a full run and carries no "already generated" note.
    rc = _estimate(src, sdb)
    assert rc == 0
    out = _flat()
    assert '2 stale/new of 2 files' in out, out
    assert 'Full regeneration' not in out, (
        'nothing skipped yet — no incremental note expected'
    )
    # Floor framing: estimate shown as a minimum with a +50% ceiling, no
    # lower bound below it (the estimate is never an over-estimate).
    assert 'Estimated minimum' in out, out
    assert 'up to' in out.lower(), out
    assert 'range' not in out.lower(), out
    assert '±' not in out, (
        'a ± band implies the cost could be below the estimate; '
        f'it cannot. got:\n{out}'
    )

    # Document alpha.py at its current hash; beta.py stays undocumented.
    with StalenessTracker(sdb) as tracker:
        tracker.record_documentation(
            src / 'alpha.py', ['doc-alpha'], base_path=src,
        )

    # Repeat run: only beta.py is stale. Headline must reflect 1 file,
    # and the full-regeneration figure appears as a secondary note.
    rc = _estimate(src, sdb)
    assert rc == 0
    out = _flat()
    assert '1 stale/new of 2 files' in out, out
    assert 'up-to-date' in out, out
    assert 'Full regeneration' in out, out


@pytest.mark.asyncio
async def test_onboard_preview_prices_only_stale_files(
    tmp_path, capsys, monkeypatch,
) -> None:
    """The onboard/`dry-run` preview (cmd_dry_run) must also price only
    the stale subset for its generate phase — the path the bug report
    hit. Free phases are mocked; the staleness check there is type-aware
    (matches run()), so the up-to-date file gets library docs of every
    default type.
    """
    from rich.console import Console

    from cli.core import get_library
    from tests._scoped_config_fixture import install_test_config

    install_test_config(monkeypatch, tmp_path, 'ds')
    monkeypatch.setattr('cli.dry_run.console', Console(width=200))

    src = _src_tree(tmp_path)  # alpha.py, beta.py under tmp_path

    # Mock the free phases so no real discover/index/catalog-sync runs.
    monkeypatch.setattr('cli.index.cmd_discover', lambda *_a, **_k: 0)
    monkeypatch.setattr('cli.index.cmd_index', lambda *_a, **_k: 0)

    async def _no_catalog_sync(*_a, **_k):
        return 0
    monkeypatch.setattr('cli.dry_run.cmd_catalog_sync', _no_catalog_sync)

    # alpha.py is up-to-date: it has a doc of every default type AND a
    # matching staleness record. beta.py has neither → stale.
    from cli.generate import DEFAULT_GENERATE_DOC_TYPES
    lib = get_library(None)
    doc_ids = []
    for ct in DEFAULT_GENERATE_DOC_TYPES:
        doc = lib.add_document(
            content_type=ct,
            title=f'{ct} alpha',
            content='placeholder content for ' + ct,
            source_files=['alpha.py'],
            source_name='ds',
            metadata={'source_name': 'ds'},
        )
        doc_ids.append(doc.id)

    # The source root cmd_dry_run scans is the configured path (tmp_path);
    # files live under tmp_path/src, so record relative to tmp_path so the
    # staleness key matches the run's lookup ('src/alpha.py').
    from config import get_config
    cfg = get_config()
    with StalenessTracker(cfg.staleness_db_path) as tracker:
        tracker.record_documentation(
            src / 'alpha.py', doc_ids, base_path=tmp_path,
        )

    from cli.dry_run import cmd_dry_run
    args = argparse.Namespace(
        source='ds', model='gpt-5.4', db=None,
        verbose=False, concurrency=None, force=False,
    )
    rc = await cmd_dry_run(args)
    assert rc == 0
    out = capsys.readouterr().out
    # generate phase prices only beta.py; alpha.py is skipped, with the
    # full-regeneration figure offered as the secondary note.
    assert 'already generated' in out, out
    assert '1 of 2 files' in out, out
    assert 'Full regeneration' in out, out
