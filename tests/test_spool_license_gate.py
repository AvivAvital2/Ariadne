"""License-admission gate (§18.1 redistribution safety).

A spool ships DERIVED docs + a SCIP index built from its corpus, so the corpus
must be under a license that permits redistributing derived work. BUSL / SSPL /
Elastic / proprietary / unrecognized corpora are refused at build — before the
paid onboard — unless the builder opts into a LOCAL-only pack with
``--allow-nonfree``. Synthetic license fixtures only.
"""
from pathlib import Path

import pytest
import yaml

import spool_acquire
from spool_acquire import AcquireResult, create_spool
from spools import SpoolError, classify_license, detect_corpus_license

_APACHE = 'Apache License\nVersion 2.0, January 2004\nhttp://www.apache.org/\n'
_MPL = 'Mozilla Public License Version 2.0\n\n1. Definitions\n...\n'
_MIT = ('MIT License\n\nPermission is hereby granted, free of charge, to any '
        'person obtaining a copy...\n')
_BUSL = ('Business Source License 1.1\n\nParameters\nLicensor: Example Inc.\n'
         'Additional Use Grant: ...\nChange Date: ...\n')
_SSPL = 'Server Side Public License\nVERSION 1, October 16, 2018\n...\n'


class TestClassifyLicense:
    def test_categories(self) -> None:
        assert classify_license(_APACHE) == ('permissive', 'Apache-2.0')
        assert classify_license(_MIT)[0] == 'permissive'
        assert classify_license(_MPL) == ('weak-copyleft', 'MPL-2.0')
        # non-free upstreams the gate must refuse
        assert classify_license(_BUSL) == ('source-available', 'BUSL')
        assert classify_license(_SSPL)[0] == 'source-available'
        # empty / unrecognized fails CLOSED (not redistribution-safe)
        assert classify_license('') == ('unknown', None)
        assert classify_license('All rights reserved.')[0] == 'unknown'

    def test_nonfree_marker_wins_over_open(self) -> None:
        # A Commons-Clause rider on an otherwise-permissive grant is NOT
        # redistribution-safe — the non-free marker dominates.
        text = _MIT + '\n\nThe Commons Clause restriction applies.\n'
        assert classify_license(text)[0] == 'source-available'


class TestDetectCorpusLicense:
    def test_reads_top_level_license_file(self, tmp_path) -> None:
        (tmp_path / 'LICENSE').write_text(_APACHE)
        assert detect_corpus_license(tmp_path)[0] == 'permissive'

    def test_missing_license_is_unknown(self, tmp_path) -> None:
        assert detect_corpus_license(tmp_path) == ('unknown', None)

    def test_absent_dir_is_unknown(self, tmp_path) -> None:
        # A clone dir that isn't present → fail-closed as unknown (the gate then
        # refuses), never a crash.
        assert detect_corpus_license(tmp_path / 'nope') == ('unknown', None)


def _spoolfile(tmp_path, corpus):
    data = {'name': 'envspool', 'runtime': 'edition-1', 'version': '1.0.0',
            'corpus': corpus}
    path = tmp_path / 'spools.yaml'
    path.write_text(yaml.safe_dump(data))
    return path


def _fake_acquire(licenses_by_repo):
    """A fake ``acquire`` that materializes each corpus clone under dest_dir
    with a fetch marker + its LICENSE text — standing in for the real clone."""
    def _acquire(packfile, *, dest_dir, approve, confirm=input):
        dest = Path(dest_dir)
        for repo, text in licenses_by_repo.items():
            clone = dest / repo
            clone.mkdir(parents=True, exist_ok=True)
            (clone / '.ariadne-corpus-sha').write_text('deadbeef\n')
            (clone / 'LICENSE').write_text(text)
        return AcquireResult(accepted=True, cloned=tuple(licenses_by_repo))
    return _acquire


_NOOP_PHASES = {
    'source_add': lambda *a: None, 'index': lambda *a: None,
    'onboard': lambda name: None, 'theme': lambda *a: None,
    'build': lambda **k: None,
}


class TestLicenseGate:
    def test_refuses_nonfree_corpus(self, tmp_path, monkeypatch) -> None:
        sf = _spoolfile(tmp_path, {'tf': {'url': 'u', 'tag': 't', 'sha': 'dead'}})
        monkeypatch.setattr(spool_acquire, 'acquire', _fake_acquire({'tf': _BUSL}))
        with pytest.raises(SpoolError) as excinfo:
            create_spool(str(sf), dest_dir=tmp_path / 'd',
                         out_path=str(tmp_path / 'p.zip'), approve=True,
                         allow_ungrounded=True, phases=dict(_NOOP_PHASES))
        msg = str(excinfo.value).lower()
        assert 'tf' in msg and ('redistribut' in msg or 'open-source' in msg)

    def test_allows_open_source_corpus(self, tmp_path, monkeypatch) -> None:
        sf = _spoolfile(tmp_path, {'otf': {'url': 'u', 'tag': 't', 'sha': 'dead'}})
        built = []
        phases = dict(_NOOP_PHASES, build=lambda **k: built.append(k))
        monkeypatch.setattr(spool_acquire, 'acquire', _fake_acquire({'otf': _MPL}))
        res = create_spool(str(sf), dest_dir=tmp_path / 'd',
                           out_path=str(tmp_path / 'p.zip'), approve=True,
                           allow_ungrounded=True, phases=phases)
        assert res.accepted and built, 'an open-source corpus should build'

    def test_allow_nonfree_override_builds_local_pack(self, tmp_path, monkeypatch):
        sf = _spoolfile(tmp_path, {'tf': {'url': 'u', 'tag': 't', 'sha': 'dead'}})
        built = []
        phases = dict(_NOOP_PHASES, build=lambda **k: built.append(k))
        monkeypatch.setattr(spool_acquire, 'acquire', _fake_acquire({'tf': _BUSL}))
        res = create_spool(str(sf), dest_dir=tmp_path / 'd',
                           out_path=str(tmp_path / 'p.zip'), approve=True,
                           allow_ungrounded=True, allow_nonfree=True, phases=phases)
        assert res.accepted and built, '--allow-nonfree permits a local build'
