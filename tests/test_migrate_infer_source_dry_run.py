"""``migrate --infer-source-name`` must honour ``--dry-run``.

The backfill matches every NULL-source document's ``source_files[0]`` against
the configured source paths and rewrites ``documents.source_name`` in place. On
a real store that is a bulk relabel of tens of thousands of rows, and relabels
are not obviously reversible: once a document claims a source it becomes
purgeable by that name and scopeable into that project's closure.

Every other repair on this command previews first (``--fix-locations``,
``--fix-paths``, ``--doc-ids`` all branch on ``args.dry_run``). This one takes
the flag on the parser and ignores it, so the one operation whose blast radius
is widest is the one with no preview.

Synthetic fixtures only: tmp store, tmp source tree.
"""
from __future__ import annotations

import argparse
import textwrap

import config as config_module
import pytest
from config import Config
from library import Library


def _migrate_args(db_path, **overrides):
    base = dict(
        db=str(db_path), check=False, source_files=False, fix_paths=False,
        fix_catalog_language=False, doc_ids=False, infer_source_name=False,
        fix_locations=False, dry_run=False, verbose=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


@pytest.fixture
def store(tmp_path):
    """One configured source and one NULL-source document under its path."""
    src = tmp_path / 'proj'
    src.mkdir()
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text(textwrap.dedent(f'''
        default_source: proj
        sources:
          proj:
            path: {src}
    '''))
    old = config_module._global_config
    config_module._global_config = Config(cfg_file)

    db = tmp_path / 'cli.db'
    lib = Library(db)
    doc_id = lib.add_document(
        title='orphan', content='x', content_type='explanation',
        source_files=[str(src / 'mod.py')],
    ).id
    with lib._conn_provider.acquire() as conn:
        conn.execute('UPDATE documents SET source_name = NULL WHERE id = ?',
                     (doc_id,))
    lib.close()
    try:
        yield db, doc_id
    finally:
        config_module._global_config = old


def _source_name(db, doc_id):
    lib = Library(db)
    try:
        with lib._conn_provider.acquire() as conn:
            return conn.execute(
                'SELECT source_name FROM documents WHERE id = ?', (doc_id,),
            ).fetchone()[0]
    finally:
        lib.close()


def test_dry_run_previews_without_relabelling(store, capsys) -> None:
    from cli.maintenance import cmd_migrate

    db, doc_id = store
    rc = cmd_migrate(_migrate_args(db, infer_source_name=True, dry_run=True))
    out = capsys.readouterr().out

    assert rc == 0
    assert _source_name(db, doc_id) is None, (
        '--dry-run must not write; the document was relabelled anyway'
    )
    assert '1' in out and 'Would' in out, (
        f'a preview must say what it would do; got {out!r}'
    )


def test_without_dry_run_it_relabels(store, capsys) -> None:
    """The flag gates the write and nothing else — the real run still works."""
    from cli.maintenance import cmd_migrate

    db, doc_id = store
    rc = cmd_migrate(_migrate_args(db, infer_source_name=True))
    assert rc == 0
    assert _source_name(db, doc_id) == 'proj'
    assert 'Backfilled' in capsys.readouterr().out
