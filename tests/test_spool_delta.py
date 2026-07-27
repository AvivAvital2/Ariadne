"""Slice 3: delta-create foundation — per-repo reuse plan + the substrate.

``delta_corpus_plan`` decides, per repo, whether a new build can reuse the
prior pack's contribution (pinned sha unchanged) or must rebuild it; the
manifest's ``corpus_shas`` is the recorded substrate it compares against.
Synthetic fixtures only.
"""
import zipfile

import yaml

from library import Library
from spool_pack import build_pack
from spools import SpoolManifest, delta_corpus_plan


def test_delta_corpus_plan_splits_reuse_and_rebuild():
    prior = {'spark': 'sha_a', 'delta': 'sha_b'}
    new = {'spark': 'sha_a', 'delta': 'sha_c', 'sdk': 'sha_d'}
    reuse, rebuild = delta_corpus_plan(prior, new)
    assert reuse == ('spark',)              # unchanged sha -> reuse
    assert rebuild == ('delta', 'sdk')      # changed sha + new repo -> rebuild


def test_delta_corpus_plan_no_prior_rebuilds_all():
    reuse, rebuild = delta_corpus_plan({}, {'a': '1', 'b': '2'})
    assert reuse == ()
    assert rebuild == ('a', 'b')


def test_pack_records_corpus_shas(tmp_path):
    # build_pack records the per-repo shas in the manifest so a later delta
    # build can compare them and skip unchanged repos.
    lib = Library(tmp_path / 'src.db')
    try:
        lib.add_document(
            content_type='catalog', title='x', content='c',
            source_files=['f.py'],
            metadata={'kind': 'element', 'source_name': 'env'}, doc_id='d1',
        )
        with lib._conn_provider.acquire() as c:
            c.execute("UPDATE documents SET source_name='env' WHERE id='d1'")
        out = tmp_path / 'pack.zip'
        shas = {'spark': 'sha_a', 'delta': 'sha_b'}
        manifest = build_pack(
            lib, environment='env', version='1', target_runtime='rt',
            source_root=tmp_path, out_path=str(out), corpus_shas=shas,
        )
        assert manifest.corpus_shas == shas
        # Round-trips through the written manifest.yaml.
        with zipfile.ZipFile(out) as zf:
            data = yaml.safe_load(zf.read('manifest.yaml'))
        assert SpoolManifest.from_dict(data).corpus_shas == shas
    finally:
        lib.close()
