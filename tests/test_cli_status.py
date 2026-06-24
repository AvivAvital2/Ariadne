"""Guardrail/contract tests for the library-reporting commands
(``stats`` / ``status`` / ``usage`` / ``gaps`` / ``vacuum``) in cli/status.py.

These encode the *expected* user-facing behavior promised by each command's
help text, driven black-box against a real Library at a tmp ``--db`` seeded
with synthetic, neutral data (src1/src2, pkg.mod.*). Assertions are on the
text each command prints (whitespace-normalized so they're terminal-width
independent). No pytest ``monkeypatch`` — config is isolated via a fixture
that restores the global singleton; the only mocks are stable *external*
boundaries (the LLM gap analyzer). A failing test here means a command
stopped honoring its contract.
"""
from __future__ import annotations

import argparse
import json
import os
from unittest import mock

import pytest

import config as config_module
from cli.main import create_parser
from cli.status import (
    HANDLERS,
    _attributed_bytes,
    _human_bytes,
    _load_status_cache,
    _save_status_cache,
    _status_cache_path,
    cmd_gaps,
    cmd_stats,
    cmd_status,
    cmd_usage,
    cmd_vacuum,
)
import slack_usage
import testimonials
from config import Config
from gap_analysis import GapRecommendation, GapReport
from library import Library
from schema import _now_iso

# --- fixtures ---------------------------------------------------------------

@pytest.fixture(autouse=True)
def wide_console():
    """Pin a wide terminal so Rich tables don't truncate content under
    capsys (no tty). Rich honors $COLUMNS at render time.
    """
    old = os.environ.get('COLUMNS')
    os.environ['COLUMNS'] = '240'
    try:
        yield
    finally:
        if old is None:
            os.environ.pop('COLUMNS', None)
        else:
            os.environ['COLUMNS'] = old


@pytest.fixture
def config_src1(tmp_path):
    """Isolate get_config() to a config whose default_source is 'src1'
    (cmd_stats reads default_source for the sync-state row). Monkeypatch-free:
    swap the cached singleton + $ARIADNE_CONFIG, restore on teardown.
    """
    cfg_dir = tmp_path / 'cfg'
    cfg_dir.mkdir()
    cfg_file = cfg_dir / 'ariadne.yaml'
    cfg_file.write_text('default_source: src1\nsources:\n  src1:\n    path: /x\n')
    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        yield
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


@pytest.fixture
def config_none(tmp_path):
    """Isolate get_config() to a config with no default_source (covers the
    stats branch that skips the sync-state lookup).
    """
    cfg_file = tmp_path / 'none.yaml'
    cfg_file.write_text('sources: {}\n')
    old_env = os.environ.get('ARIADNE_CONFIG')
    old_singleton = config_module._global_config
    os.environ['ARIADNE_CONFIG'] = str(cfg_file)
    config_module._global_config = Config(cfg_file)
    try:
        yield
    finally:
        config_module._global_config = old_singleton
        if old_env is None:
            os.environ.pop('ARIADNE_CONFIG', None)
        else:
            os.environ['ARIADNE_CONFIG'] = old_env


def _ns(**kw):
    base = {'db': None, 'by_source': False, 'days': 30, 'tool': None,
            'by_document': False, 'top_served': None, 'analyze': False}
    return argparse.Namespace(**{**base, **kw})


def _new_lib(tmp_path, name='lib.db'):
    return Library(tmp_path / name)


def _add_doc(lib, *, source, title, content_type='explanation',
             content='body text', doc_files=None, doc_id=None):
    return lib.add_document(
        content_type=content_type,
        title=title,
        content=content,
        source_files=doc_files or [],
        source_name=source,
        doc_id=doc_id,
    )


def _norm(capsys):
    # Drop Rich's box-drawing column separator so we can assert on the
    # "label value" pairs inside table rows, not just label presence.
    text = capsys.readouterr().out.replace('│', ' ')
    return ' '.join(text.split())


# --- stats ------------------------------------------------------------------

def test_stats_reports_totals_and_sync(config_src1, tmp_path, capsys):
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='Exp', content_type='explanation')
    _add_doc(lib, source='src1', title='E2', content_type='explanation')
    _add_doc(lib, source='src1', title='Find', content_type='finding')
    lib.set_sync_state('src1', 'abc12345deadbeef00')
    db = lib.path
    lib.close()

    rc = cmd_stats(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'Library Statistics' in out
    # the counts must be right, not merely present: 3 docs, 2 explanation, 1 finding
    assert 'Total documents 3' in out
    assert 'explanation 2' in out
    assert 'finding 1' in out
    # a content type with zero docs must NOT get a row
    assert 'architecture' not in out
    # sync row shows the 8-char short hash + the stored timestamp
    assert 'Last sync abc12345' in out


def test_stats_by_source_breakdown(config_src1, tmp_path, capsys):
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='A')
    _add_doc(lib, source='src1', title='A2')
    _add_doc(lib, source='src2', title='B')
    db = lib.path
    lib.close()

    rc = cmd_stats(_ns(db=db, by_source=True))
    out = _norm(capsys)
    assert rc == 0
    assert 'Per-Source Breakdown' in out
    # per-source doc counts must be attributed correctly: src1=2, src2=1
    assert 'src1 2' in out
    assert 'src2 1' in out
    # the DB-total footer row is present (guard branch taken)
    assert 'DB total' in out


def test_stats_by_source_without_total_meta_skips_db_row(config_none, tmp_path, capsys):
    """Defensive `if total_meta:` branch: if stats_by_source ever returns
    without a '_total' entry, the DB-total footer row is omitted (and the
    command still renders the per-source rows without crashing).
    """
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='A')
    db = lib.path
    lib.close()

    fake_stats = {'src1': {'doc_count': 1, 'content_size': 100,
                           'embedding_size': 50, 'chunk_count': 0}}
    with mock.patch('library.Library.stats_by_source', return_value=fake_stats):
        rc = cmd_stats(_ns(db=db, by_source=True))
    out = _norm(capsys)
    assert rc == 0
    assert 'Per-Source Breakdown' in out and 'src1' in out
    assert 'DB total' not in out


def test_stats_no_default_source_skips_sync(config_none, tmp_path, capsys):
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='A')
    db = lib.path
    lib.close()

    rc = cmd_stats(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'Total documents' in out
    assert 'Last sync' not in out


# --- status -----------------------------------------------------------------

def test_status_empty_library(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    db = lib.path
    lib.close()

    rc = cmd_status(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'No documents in library.' in out


def test_status_renders_matrix_and_writes_cache(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='A', content_type='explanation')
    _add_doc(lib, source='src1', title='A2', content_type='explanation')
    _add_doc(lib, source='src2', title='B', content_type='architecture')
    sig1 = lib.source_signature('src1')
    db = lib.path
    lib.close()

    rc = cmd_status(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'Library status' in out
    assert 'src1' in out and 'src2' in out
    assert 'DB file' in out and 'Overhead' in out
    # TOTAL row sums doc_count across sources (2 + 1)
    assert 'TOTAL 3' in out

    # the cache must be persisted with the right per-source signature + stats
    cache = json.loads(_status_cache_path(_StubLib(db)).read_text())
    assert set(cache) == {'src1', 'src2'}
    assert cache['src1']['cache_key'] == sig1
    assert cache['src1']['stats']['doc_count'] == 2
    assert cache['src1']['stats']['by_content_type'] == {'explanation': 2}
    assert cache['src2']['stats']['doc_count'] == 1


def test_status_uses_cache_and_purges_stale(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='A')
    db = lib.path
    lib.close()

    cache_path = db.parent / f'{db.name}.status-cache.json'
    cache_path.write_text(json.dumps({'ghost': {'cache_key': 'x', 'stats': {}}}))

    # first run: ghost (no longer a real source) gets purged, src1 cached
    assert cmd_status(_ns(db=db)) == 0
    capsys.readouterr()
    saved = json.loads(cache_path.read_text())
    assert 'ghost' not in saved
    assert 'src1' in saved

    # Prove the cache is actually READ on a signature-matching run: poison the
    # cached doc_count; a matching run must render the cached (poisoned) value
    # instead of recomputing the true count of 1.
    saved['src1']['stats']['doc_count'] = 99
    cache_path.write_text(json.dumps(saved))
    rc = cmd_status(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'src1 99' in out


# --- usage ------------------------------------------------------------------

def test_usage_reports_calls_and_feedback(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    lib.log_usage('ariadne_search', 'q1', result_count=3)
    miss_id = lib.log_usage('ariadne_search', 'q2', result_count=0)
    lib.mark_miss(miss_id, 'needed auth docs')
    db = lib.path
    lib.close()

    rc = cmd_usage(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    # 2 calls (one hit, one miss) -> hit rate exactly 50%
    assert 'Total calls 2' in out
    assert 'Total hits 1' in out
    assert 'Total misses 1' in out
    assert 'Hit rate 50.0%' in out
    assert 'ariadne_search' in out
    assert 'Recent Feedback' in out and 'needed auth docs' in out


def test_usage_by_document_lists_served(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    doc = _add_doc(lib, source='src1', title='Served Doc')
    other = _add_doc(lib, source='src1', title='Rare Doc')
    for _ in range(3):
        lib.log_usage('ariadne_search', 'q', result_count=1, document_ids=[doc.id])
    lib.log_usage('ariadne_search', 'q', result_count=1, document_ids=[other.id])
    db = lib.path
    lib.close()

    rc = cmd_usage(_ns(db=db, by_document=True))
    out = _norm(capsys)
    assert rc == 0
    assert 'Top Served Documents' in out
    # serve counts must be right: Served Doc=3 (explanation), Rare Doc=1
    assert 'Served Doc explanation 3' in out
    assert 'Rare Doc explanation 1' in out


def test_usage_by_document_empty_message(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    lib.log_usage('ariadne_search', 'q', result_count=0)
    db = lib.path
    lib.close()

    rc = cmd_usage(_ns(db=db, top_served=5))
    out = _norm(capsys)
    assert rc == 0
    assert 'No per-document tracking data yet' in out


def test_usage_empty_library(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    db = lib.path
    lib.close()

    rc = cmd_usage(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    # no events -> zero counts, zero hit rate, no by-tool/feedback sections
    assert 'Total calls 0' in out
    assert 'Hit rate 0.0%' in out
    assert 'Recent Feedback' not in out


def test_usage_by_user_from_slack_store(tmp_path, capsys):
    """`ariadne usage` adds a per-user Slack section from the bridge store:
    questions/hits/misses by resolved name, busiest first. Counts only."""
    lib = _new_lib(tmp_path)
    db = lib.path
    lib.close()

    store = testimonials.local_dir(tmp_path)
    for outcome in ('hit', 'miss'):
        slack_usage.record(store, asked_at=_now_iso(), actor='U_a',
                           name='alice', outcome=outcome, score=5)
    slack_usage.record(store, asked_at=_now_iso(), actor='U_b',
                       name='bob', outcome='hit', score=9)

    rc = cmd_usage(_ns(db=db, dir=str(tmp_path)))
    out = _norm(capsys)
    assert rc == 0
    assert 'By User' in out
    # alice: 2 questions, 1 hit, 1 miss (busiest first); bob: 1 question, 1 hit
    assert 'alice 2 1 1' in out
    assert 'bob 1 1 0' in out
    assert out.index('alice') < out.index('bob')


def test_usage_no_slack_section_when_store_empty(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    db = lib.path
    lib.close()

    rc = cmd_usage(_ns(db=db, dir=str(tmp_path)))
    assert rc == 0
    assert 'By User' not in _norm(capsys)


# --- gaps -------------------------------------------------------------------

def test_gaps_none_recorded(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    lib.log_usage('ariadne_search', 'q', result_count=2)
    db = lib.path
    lib.close()

    rc = cmd_gaps(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'No misses recorded' in out


def _seed_miss(tmp_path):
    lib = _new_lib(tmp_path)
    mid = lib.log_usage('ariadne_search', 'how to configure auth', result_count=0)
    lib.mark_miss(mid, 'no auth configuration docs')
    db = lib.path
    lib.close()
    return db


def test_gaps_lists_top_gaps(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    # two misses with identical feedback (group to one gap, count 2) ...
    for _ in range(2):
        mid = lib.log_usage('ariadne_search', 'how to configure auth', result_count=0)
        lib.mark_miss(mid, 'no auth configuration docs')
    # ... plus one hit, so miss rate is 2/3 rather than trivially 100%
    lib.log_usage('ariadne_search', 'unrelated', result_count=3)
    db = lib.path
    lib.close()

    rc = cmd_gaps(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'Documentation Gaps' in out
    # the gap is grouped by feedback with the correct count
    assert 'no auth configuration docs 2' in out
    # 2 misses across 3 events -> 66.7%
    assert 'Total misses: 2' in out
    assert '66.7%' in out


def test_gaps_analyze_prints_recommendations(tmp_path, capsys):
    db = _seed_miss(tmp_path)

    async def fake_analyze(misses):
        return GapReport(
            total_misses=1,
            analysis_period_days=30,
            recommendations=(GapRecommendation(
                theme='Auth docs',
                miss_count=1,
                description='Users keep asking about auth.',
                recommendation='Write an auth guide.',
                example_queries=('how to configure auth',),
            ),),
            summary='One recurring gap around authentication.',
        )

    with mock.patch('gap_analysis.analyze_gaps', fake_analyze):
        rc = cmd_gaps(_ns(db=db, analyze=True))
    out = _norm(capsys)
    assert rc == 0
    assert 'Running LLM gap analysis' in out
    assert 'One recurring gap around authentication.' in out
    assert 'Auth docs' in out and 'Write an auth guide.' in out


def test_gaps_analyze_handles_llm_failure(tmp_path, capsys):
    db = _seed_miss(tmp_path)

    async def boom(misses):
        raise RuntimeError('model exploded')

    with mock.patch('gap_analysis.analyze_gaps', boom):
        rc = cmd_gaps(_ns(db=db, analyze=True))
    out = _norm(capsys)
    assert rc == 0
    assert 'LLM analysis failed' in out


def test_gaps_analyze_handles_missing_module(tmp_path, capsys):
    db = _seed_miss(tmp_path)
    # Simulate the optional gap_analysis module being absent.
    with mock.patch.dict('sys.modules', {'gap_analysis': None}):
        rc = cmd_gaps(_ns(db=db, analyze=True))
    out = _norm(capsys)
    assert rc == 0
    assert 'LLM analysis unavailable' in out


# --- vacuum -----------------------------------------------------------------

def test_vacuum_optimizes(tmp_path, capsys):
    lib = _new_lib(tmp_path)
    _add_doc(lib, source='src1', title='A')
    db = lib.path
    lib.close()

    rc = cmd_vacuum(_ns(db=db))
    out = _norm(capsys)
    assert rc == 0
    assert 'Database optimized.' in out


# --- pure helpers -----------------------------------------------------------

@pytest.mark.parametrize(('n', 'expected'), [
    (0, '0 B'),
    (512, '512 B'),
    (1024, '1.0 KB'),
    (1536, '1.5 KB'),
    (1024 * 1024, '1.0 MB'),
    (1024 ** 3, '1.0 GB'),
    (1024 ** 4, '1.0 TB'),
    (5 * 1024 ** 5, '5120.0 TB'),
])
def test_human_bytes(n, expected):
    assert _human_bytes(n) == expected


def test_attributed_bytes_sums_all_buckets():
    data = {
        'doc_content': 1, 'doc_embed': 2,
        'chunk_content': 4, 'chunk_embed': 8,
        'section_content': 16, 'section_embed': 32,
    }
    assert _attributed_bytes(data) == 63


class _StubLib:
    def __init__(self, path):
        self.path = path


def test_status_cache_path_sits_beside_db(tmp_path):
    db = tmp_path / 'my.db'
    p = _status_cache_path(_StubLib(db))
    assert p == tmp_path / 'my.db.status-cache.json'


def test_load_status_cache_missing_is_empty(tmp_path):
    assert _load_status_cache(tmp_path / 'nope.json') == {}


def test_load_status_cache_valid(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text(json.dumps({'src1': {'cache_key': 'k'}}))
    assert _load_status_cache(p) == {'src1': {'cache_key': 'k'}}


def test_load_status_cache_non_dict_is_empty(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text(json.dumps([1, 2, 3]))
    assert _load_status_cache(p) == {}


def test_load_status_cache_corrupt_is_empty(tmp_path):
    p = tmp_path / 'c.json'
    p.write_text('{ not json')
    assert _load_status_cache(p) == {}


def test_save_status_cache_roundtrip(tmp_path):
    p = tmp_path / 'c.json'
    _save_status_cache(p, {'src1': {'cache_key': 'k'}})
    assert json.loads(p.read_text()) == {'src1': {'cache_key': 'k'}}


def test_save_status_cache_swallows_oserror(tmp_path):
    # Parent dir doesn't exist -> write raises OSError, must be swallowed.
    p = tmp_path / 'missing_dir' / 'c.json'
    _save_status_cache(p, {'x': 1})
    assert not p.exists()


# --- wiring -----------------------------------------------------------------

def test_commands_registered_and_dispatchable():
    parser = create_parser()
    for argv in (['stats', '--by-source'], ['status'], ['usage', '--days', '7'],
                 ['gaps', '--analyze'], ['vacuum']):
        assert parser.parse_args(argv).command == argv[0]
    assert {'stats', 'status', 'usage', 'gaps', 'vacuum'} <= set(HANDLERS)
