"""Extraction-coverage version: a code-level change to what the SCIP extractors
cover is invisible to content-based staleness, so a source indexed under older
coverage (or never stamped) must be flagged as behind — and the flag must
self-clear once re-indexed. Surfaces regardless of ignore_staleness (D2).
"""
from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

from docgen.extraction_coverage import (
    EXTRACTION_COVERAGE_VERSION,
    coverage_gap,
    coverage_notice,
    stamp_coverage,
)
from library import Library
from spool_pack import build_pack
from spools import SpoolManifest


def _src(tmp_path, manifest: dict | None = None) -> Path:
    root = tmp_path / 'src'
    (root / '.ariadne').mkdir(parents=True)
    if manifest is not None:
        (root / '.ariadne' / 'manifest.json').write_text(
            json.dumps(manifest), encoding='utf-8')
    return root


def test_unstamped_source_is_behind(tmp_path):
    root = _src(tmp_path, {})
    assert coverage_gap(root) == (0, EXTRACTION_COVERAGE_VERSION)


def test_missing_manifest_is_behind(tmp_path):
    root = tmp_path / 'nomanifest'
    root.mkdir()
    assert coverage_gap(root) == (0, EXTRACTION_COVERAGE_VERSION)


def test_older_stamp_is_behind(tmp_path):
    root = _src(tmp_path, {'extraction_coverage_version': EXTRACTION_COVERAGE_VERSION - 1})
    assert coverage_gap(root) == (
        EXTRACTION_COVERAGE_VERSION - 1, EXTRACTION_COVERAGE_VERSION)


def test_stamp_clears_the_gap_and_preserves_keys(tmp_path):
    root = _src(tmp_path, {'merged_scip': 'index.scip', 'foo': 1})
    stamp_coverage(root)
    assert coverage_gap(root) is None
    man = json.loads((root / '.ariadne' / 'manifest.json').read_text())
    assert man['extraction_coverage_version'] == EXTRACTION_COVERAGE_VERSION
    assert man['merged_scip'] == 'index.scip' and man['foo'] == 1  # preserved


def test_stamp_creates_manifest_when_absent(tmp_path):
    root = tmp_path / 'nomanifest'
    root.mkdir()
    stamp_coverage(root)
    assert coverage_gap(root) is None


def test_notice_is_actionable_when_behind_else_none(tmp_path):
    behind = _src(tmp_path, {})
    msg = coverage_notice('mylib', behind)
    assert msg and 'behind' in msg and 'ariadne index --source mylib' in msg
    stamp_coverage(behind)
    assert coverage_notice('mylib', behind) is None


# --- spool packs carry the coverage version so a consumer can flag a stale one ---

def test_spool_manifest_roundtrips_coverage_version():
    m = SpoolManifest(
        environment='x', version='1', target_runtime='r', checksum='sha256:z',
        extraction_coverage_version=EXTRACTION_COVERAGE_VERSION)
    assert m.to_dict()['extraction_coverage_version'] == EXTRACTION_COVERAGE_VERSION
    assert SpoolManifest.from_dict(
        m.to_dict()).extraction_coverage_version == EXTRACTION_COVERAGE_VERSION


def test_old_pack_manifest_defaults_coverage_zero():
    # A pre-tracking pack has no key → 0 → treated as behind (rebuild advisory).
    back = SpoolManifest.from_dict(
        {'environment': 'x', 'version': '1', 'target_runtime': 'r',
         'checksum': 'sha256:z'})
    assert back.extraction_coverage_version == 0


def test_build_pack_stamps_current_coverage(tmp_path):
    root = tmp_path / 'repo'
    root.mkdir()
    with Library(tmp_path / 'b.db') as lib:
        lib.add_document('explanation', 't', 'body', source_files=[],
                         source_name='envx')
        manifest = build_pack(
            lib, environment='envx', version='1.0', target_runtime='r',
            certified_docs=(), source_root=root, out_path=tmp_path / 'p.zip')
    assert manifest.extraction_coverage_version == EXTRACTION_COVERAGE_VERSION
    with zipfile.ZipFile(tmp_path / 'p.zip') as zf:
        man = yaml.safe_load(zf.read('manifest.yaml'))
    assert man['extraction_coverage_version'] == EXTRACTION_COVERAGE_VERSION
