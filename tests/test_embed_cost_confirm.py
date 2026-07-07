"""Guardrail: a full re-embed announces its cost and asks before a big bill.

* ``_rebuild_embeddings`` prints an estimate (doc count + ~$) before embedding.
* A run at/above the confirm threshold prompts; declining skips embedding.
* ``--yes`` / ``assume_yes`` skips the prompt; below threshold never prompts;
  zero docs is a no-op.
* ``ariadne rebuild`` and ``ariadne import`` both expose ``--yes`` and
  thread it — a large import prompts exactly like a rebuild.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest
from rich.text import Text

import cli.core as core
from cli.main import create_parser
from library import Library


@pytest.fixture
def printed(monkeypatch):
    lines: list[str] = []
    monkeypatch.setattr(
        core.console, 'print',
        lambda *a, **k: lines.append(' '.join(str(x) for x in a)),
    )
    return lines


@pytest.fixture
def embed_recorder(monkeypatch):
    rec = {'called': False}

    async def fake_rebuild_all(self, only_missing=False, on_progress=None):
        rec['called'] = True
        rec['only_missing'] = only_missing
        return 0

    monkeypatch.setattr('writer.LibraryWriter.rebuild_all_embeddings', fake_rebuild_all)
    monkeypatch.setattr('library.embedding_matrix.ensure_matrix', lambda library: None)
    return rec


def _seed(tmp_path, n: int) -> Library:
    lib = Library(tmp_path / 't.db')
    for i in range(n):
        lib.add_document(content_type='explanation', title=f'D{i}', content=f'body {i}')
    return lib


async def test_estimate_is_printed(printed, embed_recorder, tmp_path):
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=True)
    finally:
        lib.close()
    blob = '\n'.join(printed)
    assert '3' in blob and '$' in blob, 'estimate with doc count and cost must be printed'
    # A tiny run where fast and slow bounds format identically must collapse
    # to a single duration, not print a degenerate '~0s–0s' range.
    assert '–' not in blob


async def test_declined_large_run_skips_embedding(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'n')
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=False, use_batch=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is False, 'declining must skip embedding'


async def test_zero_docs_skips_embedding(printed, embed_recorder, tmp_path):
    lib = _seed(tmp_path, 0)
    try:
        await core._rebuild_embeddings(lib, only_missing=True, assume_yes=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is False, 'no missing docs → nothing to embed'


async def test_confirmed_large_run_embeds(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    prompts: list[str] = []

    def _input(prompt='', *a, **k):
        prompts.append(prompt)
        return 'y'

    monkeypatch.setattr(core.console, 'input', _input)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=False, use_batch=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is True
    # The y/N hint must survive Rich markup rendering — an unescaped
    # '[y/N]' parses as a markup tag and vanishes from the terminal,
    # leaving the user guessing what to type.
    rendered = Text.from_markup(prompts[0]).plain
    assert '[y/N]' in rendered


async def test_assume_yes_skips_prompt(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)

    def _boom(*a, **k):
        raise AssertionError('must not prompt when assume_yes is set')

    monkeypatch.setattr(core.console, 'input', _boom)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=True)
    finally:
        lib.close()
    assert embed_recorder['called'] is True


async def test_below_threshold_no_prompt(printed, embed_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 100)

    def _boom(*a, **k):
        raise AssertionError('must not prompt below threshold')

    monkeypatch.setattr(core.console, 'input', _boom)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(lib, only_missing=False, assume_yes=False)
    finally:
        lib.close()
    assert embed_recorder['called'] is True


def test_rebuild_parser_has_yes_flag():
    parser = create_parser()
    assert parser.parse_args(['rebuild', '--yes']).yes is True
    assert parser.parse_args(['rebuild']).yes is False


async def test_cmd_rebuild_threads_yes(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False, use_batch=False):
        seen['assume_yes'] = assume_yes

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    await core.cmd_rebuild(SimpleNamespace(db=str(tmp_path / 't.db'), only_missing=False, yes=True))
    assert seen['assume_yes'] is True


def test_import_parser_has_yes_flag():
    parser = create_parser()
    assert parser.parse_args(['import', '--yes']).yes is True
    assert parser.parse_args(['import']).yes is False


def test_cmd_import_threads_yes(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False, use_batch=False):
        seen['assume_yes'] = assume_yes

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    monkeypatch.setattr(core, 'get_config', lambda: SimpleNamespace(default_source=None))
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'n.md').write_text('---\ntype: explanation\ntitle: "N"\n---\n# N\n\nbody\n')
    for flag in (False, True):
        core.cmd_import_(SimpleNamespace(
            input=str(docs), source=None, db=str(tmp_path / f't{flag}.db'),
            skip_embeddings=False, yes=flag,
        batch=False, live=False))
        assert seen['assume_yes'] is flag


@pytest.fixture
def batch_recorder(monkeypatch):
    rec = {'called': False}

    async def fake_batch(self, strategy, only_missing=False, on_progress=None,
                         on_submit=None):
        rec['called'] = True
        rec['only_missing'] = only_missing
        if on_submit is not None:
            on_submit('batch_test_1')
        return 0

    monkeypatch.setattr(
        'writer.LibraryWriter.rebuild_all_embeddings_batch', fake_batch)
    monkeypatch.setattr('library.embedding_matrix.ensure_matrix', lambda library: None)
    return rec


async def test_batch_estimate_halves_cost_and_uses_batch_path(
        printed, embed_recorder, batch_recorder, tmp_path):
    lib = _seed(tmp_path, 4)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=True, use_batch=True)
    finally:
        lib.close()
    blob = '\n'.join(printed)
    assert 'Batch API' in blob, 'estimate must say the batch price applies'
    assert 'batch_test_1' in blob, 'the submitted batch id must be printed'
    live_cost = 4 * core.EMBED_TOKENS_PER_DOC / 1_000_000 * core.EMBED_COST_PER_1M_TOKENS
    assert f'${live_cost * 0.5:.2f}' in blob
    assert batch_recorder['called'] is True
    assert embed_recorder['called'] is False, 'batch mode must not hit the live path'


async def test_batch_prompt_declined_skips_everything(
        printed, embed_recorder, batch_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'n')
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=False, use_batch=True)
    finally:
        lib.close()
    assert batch_recorder['called'] is False
    assert embed_recorder['called'] is False


def test_parsers_expose_batch_flag():
    parser = create_parser()
    assert parser.parse_args(['rebuild', '--batch']).batch is True
    assert parser.parse_args(['rebuild']).batch is False
    assert parser.parse_args(['import', '--batch']).batch is True
    assert parser.parse_args(['import']).batch is False


async def test_cmd_rebuild_threads_batch(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False, use_batch=False):
        seen['use_batch'] = use_batch

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    await core.cmd_rebuild(SimpleNamespace(
        db=str(tmp_path / 't.db'), only_missing=False, yes=True, batch=True, live=False))
    assert seen['use_batch'] is True


def test_cmd_import_threads_batch(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False, use_batch=False):
        seen['use_batch'] = use_batch

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    monkeypatch.setattr(core, 'get_config', lambda: SimpleNamespace(default_source=None))
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'n.md').write_text('---\ntype: explanation\ntitle: "N"\n---\n# N\n\nbody\n')
    core.cmd_import_(SimpleNamespace(
        input=str(docs), source=None, db=str(tmp_path / 't.db'),
        skip_embeddings=False, yes=True, batch=True,
    live=False))
    assert seen['use_batch'] is True


async def test_no_flag_large_run_mode_prompt_batch(
        printed, embed_recorder, batch_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    seen: dict[str, object] = {}

    def fake_mode_prompt(n, live_cost, live_eta, batch_cost):
        seen['live_eta'] = live_eta
        return 'batch'

    monkeypatch.setattr(core, '_prompt_embedding_mode', fake_mode_prompt)
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'y')
    lib = _seed(tmp_path, 30)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=False, use_batch=None)
    finally:
        lib.close()
    assert batch_recorder['called'] is True
    assert embed_recorder['called'] is False
    assert 'Batch API' in '\n'.join(printed)
    # The up-front live ETA must bracket real throughput as a fast–slow
    # range (sustained rate limiting slows runs ~10×); a single-point
    # promise at the ideal rate misled a real 64k-doc run by an hour+.
    assert seen['live_eta'] == '~0s–3s'


async def test_no_flag_large_run_mode_prompt_live(
        printed, embed_recorder, batch_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    monkeypatch.setattr(core, '_prompt_embedding_mode',
                        lambda n, live_cost, live_eta, batch_cost: 'live')
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'y')
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=False, use_batch=None)
    finally:
        lib.close()
    assert embed_recorder['called'] is True
    assert batch_recorder['called'] is False


async def test_no_flag_mode_batch_then_declined_embeds_nothing(
        printed, embed_recorder, batch_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)
    monkeypatch.setattr(core, '_prompt_embedding_mode',
                        lambda n, live_cost, live_eta, batch_cost: 'batch')
    monkeypatch.setattr(core.console, 'input', lambda *a, **k: 'n')
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=False, use_batch=None)
    finally:
        lib.close()
    assert batch_recorder['called'] is False
    assert embed_recorder['called'] is False


async def test_no_flag_small_run_defaults_live_without_mode_prompt(
        printed, embed_recorder, batch_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 100)

    def _boom(*a, **k):
        raise AssertionError('must not mode-prompt below the threshold')

    monkeypatch.setattr(core, '_prompt_embedding_mode', _boom)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=False, use_batch=None)
    finally:
        lib.close()
    assert embed_recorder['called'] is True
    assert batch_recorder['called'] is False


async def test_assume_yes_no_flag_stays_live_and_silent(
        printed, embed_recorder, batch_recorder, monkeypatch, tmp_path):
    monkeypatch.setattr(core, 'EMBED_CONFIRM_THRESHOLD', 1)

    def _boom(*a, **k):
        raise AssertionError('--yes must not mode-prompt')

    monkeypatch.setattr(core, '_prompt_embedding_mode', _boom)
    lib = _seed(tmp_path, 3)
    try:
        await core._rebuild_embeddings(
            lib, only_missing=False, assume_yes=True, use_batch=None)
    finally:
        lib.close()
    assert embed_recorder['called'] is True
    assert batch_recorder['called'] is False


def test_parsers_expose_live_flag_and_exclusivity():
    parser = create_parser()
    assert parser.parse_args(['import', '--live']).live is True
    assert parser.parse_args(['rebuild', '--live']).live is True
    assert parser.parse_args(['import']).live is False
    with pytest.raises(SystemExit):
        parser.parse_args(['import', '--batch', '--live'])
    with pytest.raises(SystemExit):
        parser.parse_args(['rebuild', '--batch', '--live'])


async def test_cmd_rebuild_resolves_three_state_mode(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False, use_batch=False):
        seen['use_batch'] = use_batch

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    base = dict(db=str(tmp_path / 't.db'), only_missing=False, yes=True)
    await core.cmd_rebuild(SimpleNamespace(**base, batch=False, live=True))
    assert seen['use_batch'] is False
    await core.cmd_rebuild(SimpleNamespace(**base, batch=False, live=False))
    assert seen['use_batch'] is None


def test_cmd_import_resolves_three_state_mode(monkeypatch, tmp_path):
    seen: dict[str, object] = {}

    async def fake(library, only_missing=False, assume_yes=False, use_batch=False):
        seen['use_batch'] = use_batch

    monkeypatch.setattr(core, '_rebuild_embeddings', fake)
    monkeypatch.setattr(core, 'get_config', lambda: SimpleNamespace(default_source=None))
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'n.md').write_text('---\ntype: explanation\ntitle: "N"\n---\n# N\n\nbody\n')
    base = dict(input=str(docs), source=None, db=str(tmp_path / 't.db'),
                skip_embeddings=False, yes=True)
    core.cmd_import_(SimpleNamespace(**base, batch=False, live=True))
    assert seen['use_batch'] is False
    core.cmd_import_(SimpleNamespace(**base, batch=False, live=False))
    assert seen['use_batch'] is None


def test_prompt_embedding_mode_delegates_to_onboard_selector(monkeypatch):
    import cli.onboard as onboard
    seen: dict[str, object] = {}

    def fake_selector(options, title):
        seen['options'] = options
        seen['title'] = title
        return 'batch'

    monkeypatch.setattr(onboard, '_prompt_for_batch_mode', fake_selector)

    assert core._prompt_embedding_mode(66000, 10.0, '~11m', 5.0) == 'batch'
    assert seen['title'] == 'Embedding mode (66,000 documents)'
    values = [value for value, _, _ in seen['options']]
    assert values == ['live', 'batch']
    descriptions = ' '.join(desc for _, _, desc in seen['options'])
    assert '$10.00' in descriptions and '$5.00' in descriptions
