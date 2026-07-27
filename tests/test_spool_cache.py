"""CRIT-11: the result caches must be keyed on enabled-spool state.

A spool toggle (enable / disable / update) must invalidate cached search
and audience-response results — otherwise disabling a bad spool leaves its
content in cached answers for the 30-day TTL, defeating the CRIT-6
disable-remediation. Synthetic fixtures only; no LLM (query=None search;
direct audience-cache calls).
Design: designs/spool-environment-plugin.md §17 · §20 (audit).
"""
import asyncio
import textwrap
from pathlib import Path

from ariadne_mcp.service import AriadneService
from ariadne_mcp.service_analysis import (
    _find_cached_audience_response,
    _persist_audience_response,
)
from config import Config
from library import Library
from spools import spool_scope_fingerprint, spool_source_id


def _write_cfg(root, body):
    (root / 'ariadne.yaml').write_text(textwrap.dedent(body))
    return Config(config_path=root / 'ariadne.yaml')


def _enable_spool(root, name='databricks'):
    d = root / '.ariadne' / 'spools' / name
    d.mkdir(parents=True, exist_ok=True)
    (d / 'manifest.yaml').write_text(
        f'environment: {name}\nversion: 1\ntarget_runtime: r\nchecksum: c1\n')


class TestSearchCacheSpoolKeyed:
    def test_toggle_invalidates_cached_search(self, tmp_path):
        (tmp_path / 'src1').mkdir()
        cfg = _write_cfg(tmp_path, f'''
            sources:
              src1:
                path: {tmp_path / 'src1'}
        ''')
        lib = Library(Path(cfg.db_path))
        lib.add_document('explanation', 'user doc', 'body', source_name='src1')
        lib.add_document('explanation', 'SPOOL DOC', 'body',
                         source_name=spool_source_id('databricks'),
                         _allow_reserved_source=True)  # installed spool doc
        svc = AriadneService()
        svc._library = lib
        svc._config = cfg

        async def run():
            r1 = await svc.search(query=None, branch='main', limit=10,
                                  source='src1')
            assert {d.title for d in r1.documents} == {'user doc'}

            # ENABLE the spool -> the SAME cached query must now surface it
            # (persistent cache included: clear only the in-memory layer).
            _enable_spool(tmp_path)
            svc._config = _write_cfg(tmp_path, f'''
                sources:
                  src1:
                    path: {tmp_path / 'src1'}
                spools:
                  databricks:
                    runtime: r
            ''')
            svc._query_cache.clear()
            r2 = await svc.search(query=None, branch='main', limit=10,
                                  source='src1')
            assert {d.title for d in r2.documents} == {'user doc', 'SPOOL DOC'}

            # DISABLE -> the spool's content must leave cached answers
            # (the remediation path; must not persist from cache).
            svc._config = _write_cfg(tmp_path, f'''
                sources:
                  src1:
                    path: {tmp_path / 'src1'}
            ''')
            svc._query_cache.clear()
            r3 = await svc.search(query=None, branch='main', limit=10,
                                  source='src1')
            assert {d.title for d in r3.documents} == {'user doc'}

        asyncio.run(run())
        lib.close()


class TestPMSearchSpoolGated:
    def test_pm_search_excludes_stale_fp_audience_response(self, tmp_path):
        # CRIT-11 PM-gap — a persisted PM audience_response is in the PM-role
        # SEARCH candidate set; it must be spool_fp-gated there too, or a PM
        # answer shaped by a spool is still returned by search after disable.
        (tmp_path / 'src1').mkdir()
        cfg = _write_cfg(tmp_path, f'''
            sources:
              src1:
                path: {tmp_path / 'src1'}
        ''')
        lib = Library(Path(cfg.db_path))
        parent = lib.add_document('explanation', 'dev baseline', 'body',
                                  source_name='src1')
        # a PM answer synthesized WHILE a spool was enabled (stale fp now
        # that no spool is active), inheriting the real source src1
        _persist_audience_response(
            lib, role='product_manager', question='how scale?',
            content='PM answer laced with SPOOL claims', dev_docs=[parent],
            spool_fp='fp-with-spool',
        )
        svc = AriadneService()
        svc._library = lib
        svc._config = cfg

        async def run():
            # no spool active now (current fp = '') -> the stale-fp PM row
            # must NOT be returned by a PM-role search
            r = await svc.search(query=None, branch='main', limit=10,
                                 role='product_manager', source='src1')
            titles = {d.title for d in r.documents}
            assert not any('PM answer' in t or 'response:' in t for t in titles)
            assert 'dev baseline' in titles

        asyncio.run(run())
        lib.close()


class TestExpandSpoolGated:
    def test_expand_drops_disabled_spool_docs(self, tmp_path):
        # CRIT-12 — expand re-fetches by id; it must re-apply the current
        # spool scope so a disabled (or malicious) spool's content is NOT
        # returned via a stale event_id.
        (tmp_path / 'src1').mkdir()
        cfg = _write_cfg(tmp_path, f'''
            sources:
              src1:
                path: {tmp_path / 'src1'}
            spools:
              evil:
                runtime: r
        ''')
        _enable_spool(tmp_path, 'evil')
        lib = Library(Path(cfg.db_path))
        user = lib.add_document('explanation', 'user doc', 'user body',
                                source_name='src1')
        evil = lib.add_document('explanation', 'EVIL SPOOL DOC',
                                'malicious body', source_name=spool_source_id('evil'),
                                _allow_reserved_source=True)  # installed spool doc
        svc = AriadneService()
        svc._library = lib
        svc._config = cfg

        async def run():
            r = await svc.search(query=None, branch='main', limit=10,
                                 source='src1')
            assert {evil.id, user.id} <= {d.id for d in r.documents}
            event_id = r.event_id

            # while still enabled, expand returns both
            expanded = svc.expand(event_id)
            got = {d['id'] for d in expanded['documents']}
            assert evil.id in got and user.id in got

            # DISABLE the evil spool -> expand must drop its doc
            svc._config = _write_cfg(tmp_path, f'''
                sources:
                  src1:
                    path: {tmp_path / 'src1'}
            ''')
            svc._query_cache.clear()
            expanded = svc.expand(event_id)
            got = {d['id'] for d in expanded['documents']}
            assert user.id in got
            assert evil.id not in got                     # disabled -> gone
            assert evil.id in expanded.get('missing_document_ids', [])

        asyncio.run(run())
        lib.close()


class TestAudienceCacheSpoolKeyed:
    def test_audience_cache_keyed_on_spool_fingerprint(self, tmp_path):
        cfg = _write_cfg(tmp_path, 'sources: {}\n')
        with Library(Path(cfg.db_path)) as lib:
            parent = lib.add_document('explanation', 'dev baseline', 'body',
                                      source_name='src1')
            _persist_audience_response(
                lib, role='product_manager', question='how does X scale?',
                content='PM answer (spool ON)', dev_docs=[parent],
                spool_fp='fp-with-spool',
            )
            # same audience+question, but the spool set changed -> miss
            assert _find_cached_audience_response(
                lib, role='product_manager', question='how does X scale?',
                spool_fp='fp-without-spool',
            ) is None
            # same fingerprint -> hit
            hit = _find_cached_audience_response(
                lib, role='product_manager', question='how does X scale?',
                spool_fp='fp-with-spool',
            )
            assert hit is not None and 'spool ON' in hit.content


class TestFingerprint:
    def test_fingerprint_tracks_enable_disable_update(self, tmp_path):
        cfg_off = _write_cfg(tmp_path, 'sources: {}\n')
        assert spool_scope_fingerprint(cfg_off) == ''      # no spools -> empty

        _enable_spool(tmp_path)
        cfg_on = _write_cfg(tmp_path, 'spools:\n  databricks:\n    runtime: r\n')
        fp_on = spool_scope_fingerprint(cfg_on)
        assert fp_on and fp_on != ''

        # update = manifest checksum changes -> fingerprint changes
        (tmp_path / '.ariadne' / 'spools' / 'databricks' / 'manifest.yaml'
         ).write_text('environment: databricks\nversion: 2\n'
                      'target_runtime: r\nchecksum: c2\n')
        assert spool_scope_fingerprint(cfg_on) != fp_on
