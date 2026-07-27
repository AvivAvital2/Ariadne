"""Increment 5: reconcile spool↔project theme associations to config.

Synthetic fixtures only (fake spool/project names, tmp-path store). The
reconcile makes the persisted theme partitions match
``spools.<name>.projects``: it deletes dropped associations, clusters newly
opted-in project × spool passes, and reports the dirty count (the paid
summarization the caller gates on cost).
"""
import textwrap

import numpy as np
import pytest

from config import Config
from docgen.cluster import _association_key
from library import Library
from spools import (
    SpoolError,
    build_spool_internal_themes,
    reconcile_spool_themes,
    spool_source_id,
)


def _config(tmp_path, body):
    path = tmp_path / 'ariadne.yaml'
    path.write_text(textwrap.dedent(body))
    return Config(config_path=path)


def _add(lib, doc_id, vec, source):
    v = np.array(vec, dtype=np.float32)
    v /= np.linalg.norm(v)
    lib.add_document(
        content_type='catalog', title=doc_id, content='x', source_files=[],
        embedding=v, metadata={'kind': 'element', 'source_name': source},
        doc_id=doc_id,
    )
    with lib._conn_provider.acquire() as c:
        c.execute('UPDATE documents SET source_name=? WHERE id=?', (source, doc_id))


def test_reconcile_adds_removes_and_is_idempotent(tmp_path):
    lib = Library(tmp_path / 'r.db')
    try:
        sp = spool_source_id('databricks')          # 'spool:databricks'
        v1 = [1.0, 0.05, 0, 0, 0, 0, 0, 0]
        v2 = [0, 0, 1.0, 0.05, 0, 0, 0, 0]
        # proj1 + spool cluster near v1; proj2 + spool near v2 (each a genuine
        # cross-source cluster once its pass runs).
        for i in range(3):
            _add(lib, f'p1_{i}', v1, 'proj1')
            _add(lib, f'sp1_{i}', v1, sp)
            _add(lib, f'p2_{i}', v2, 'proj2')
            _add(lib, f'sp2_{i}', v2, sp)

        cfg = _config(tmp_path, '''
            spools:
              databricks:
                runtime: rt
                projects: [proj1]
        ''')
        assoc1 = _association_key(frozenset({'proj1', sp}))
        assoc2 = _association_key(frozenset({'proj2', sp}))

        # Add proj1: a cross-source theme is created; it is dirty (pending
        # summarization) and reported as the paid work.
        r1 = reconcile_spool_themes(lib, cfg, 'databricks')
        assert r1.added == ('proj1',)
        assert r1.removed == ()
        assert lib.list_themes(coherent_only=False, association=assoc1)
        assert r1.dirty_theme_count >= 1

        # Idempotent: re-running against the same config changes nothing.
        r2 = reconcile_spool_themes(lib, cfg, 'databricks')
        assert r2.added == ()
        assert r2.removed == ()

        # Swap proj1 -> proj2: proj1's partition is deleted (free), proj2's is
        # clustered.
        cfg2 = _config(tmp_path, '''
            spools:
              databricks:
                runtime: rt
                projects: [proj2]
        ''')
        r3 = reconcile_spool_themes(lib, cfg2, 'databricks')
        assert r3.added == ('proj2',)
        assert assoc1 in r3.removed
        assert not lib.list_themes(coherent_only=False, association=assoc1)
        assert lib.list_themes(coherent_only=False, association=assoc2)
    finally:
        lib.close()


def test_reconcile_refreshes_existing_association_on_base_change(tmp_path):
    # CRITICAL-1: re-running the reconcile must re-cluster an ALREADY-enabled
    # association so cross-source themes track base-project changes. They used
    # to freeze at first enable (the add-loop skipped existing associations and
    # never rebuilt their edges).
    lib = Library(tmp_path / 'refresh.db')
    try:
        sp = spool_source_id('databricks')
        v1 = [1.0, 0.05, 0, 0, 0, 0, 0, 0]
        for i in range(2):
            _add(lib, f'p1_{i}', v1, 'proj1')
            _add(lib, f'sp_{i}', v1, sp)
        cfg = _config(tmp_path, '''
            spools:
              databricks:
                runtime: rt
                projects: [proj1]
        ''')
        assoc = _association_key(frozenset({'proj1', sp}))
        reconcile_spool_themes(lib, cfg, 'databricks', min_cluster_size=3)
        t1 = lib.list_themes(coherent_only=False, association=assoc)
        assert len(t1) == 1
        members1 = {e for e, _ in lib.get_theme_members(t1[0].cluster_id)}
        assert 'p1_new' not in members1

        # The base project gains a new element in the same cross-source cluster.
        _add(lib, 'p1_new', v1, 'proj1')
        reconcile_spool_themes(lib, cfg, 'databricks', min_cluster_size=3)
        members2 = set()
        for t in lib.list_themes(coherent_only=False, association=assoc):
            members2 |= {e for e, _ in lib.get_theme_members(t.cluster_id)}
        assert 'p1_new' in members2          # refreshed, not frozen
    finally:
        lib.close()


def test_reconcile_skips_unchanged_association(tmp_path):
    # HIGH-A: re-running the reconcile must NOT rebuild an association whose
    # {project, spool} content is unchanged — skip the costly edge-rebuild +
    # re-cluster, so enabling/refreshing one project doesn't re-index the whole
    # spool corpus for every other already-current project.
    lib = Library(tmp_path / 'skip.db')
    try:
        sp = spool_source_id('databricks')
        v1 = [1.0, 0.05, 0, 0, 0, 0, 0, 0]
        for i in range(3):
            _add(lib, f'p1_{i}', v1, 'proj1')
            _add(lib, f'sp_{i}', v1, sp)
        cfg = _config(tmp_path, '''
            spools:
              databricks:
                runtime: rt
                projects: [proj1]
        ''')
        r1 = reconcile_spool_themes(lib, cfg, 'databricks')
        assert r1.added == ('proj1',)
        assert r1.skipped == ()

        # Re-run, nothing changed -> the association is skipped, not rebuilt.
        r2 = reconcile_spool_themes(lib, cfg, 'databricks')
        assert r2.added == ()
        assert r2.skipped == ('proj1',)

        # A base-content change -> NOT skipped (rebuilt so the theme refreshes).
        _add(lib, 'p1_new', v1, 'proj1')
        r3 = reconcile_spool_themes(lib, cfg, 'databricks')
        assert r3.skipped == ()
    finally:
        lib.close()


def test_reconcile_unknown_spool_raises(tmp_path):
    # Reconciling a spool that isn't enabled in ariadne.yaml is a loud error,
    # not a silent no-op.
    lib = Library(tmp_path / 'u.db')
    try:
        cfg = _config(tmp_path, '''
            spools:
              databricks: true
        ''')
        with pytest.raises(SpoolError):
            reconcile_spool_themes(lib, cfg, 'nonexistent')
    finally:
        lib.close()


def test_build_spool_internal_themes_tags_corpus_and_avoids_base_pass(tmp_path):
    # A spool BUILD themes its OWN corpus. Its single-source (spool-internal)
    # clusters must be KEPT — a cross-source reconcile pass would drop them —
    # and tagged under the spool's reserved association, NEVER the base ''
    # pass (that base-pass tagging is exactly the leak: corpus themes
    # surfacing in every project). The single-source analog of reconcile.
    lib = Library(tmp_path / 'b.db')
    try:
        corpus = 'databricks'
        sp = spool_source_id(corpus)  # 'spool:databricks'
        v = [1.0, 0.05, 0, 0, 0, 0, 0, 0]
        for i in range(3):
            _add(lib, f'c_{i}', v, corpus)

        dirty = build_spool_internal_themes(lib, corpus, {corpus})

        # Retained (single-source kept) and tagged under spool:databricks.
        tagged = lib.list_themes(coherent_only=False, association=sp)
        assert len(tagged) == 1
        assert dirty >= 1
        # NOT in the base '' pass (no leak) and NOT under the raw corpus key.
        assert lib.list_themes(coherent_only=False, association='') == []
        assert lib.list_themes(coherent_only=False, association=corpus) == []
    finally:
        lib.close()
