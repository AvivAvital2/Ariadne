"""Contract tests for ``ariadne onboard`` (interactive single-run flow).

onboard runs the free phases + cost estimate ONCE (via the dry-run
preview), then decides whether to continue into the paid phases —
without re-running discover/index/catalog-sync. The free phases must
NOT run a second time (the whole point: don't re-index, especially the
slow scip-java compile).

Evolves one test through the demands:
  T1 — no --approve, user declines  → preview only, no paid phases, hint.
  T2 — no --approve, user accepts   → preview, then paid phases (describe,
       generate, themes). onboard does NOT re-invoke the free phases.
  T3 — --approve                    → preview runs, prompt is SKIPPED,
       paid phases run; describe is live (batch=False).
  T4 — --approve --batch            → describe routes to the batched path.
  T5 — preview (free-phase) failure → onboard returns its rc, no paid
       phases, no prompt.
  T6 — paid-phase failure           → stops the pipeline, returns the rc.
  T7 — --concurrency reaches the preview (catalog-sync) and the paid
       phases (describe, generate).
  T8 — omitted --concurrency → per-phase defaults.
"""
from __future__ import annotations

import argparse

import pytest

import cli.onboard as onboard


@pytest.fixture(autouse=True)
def _test_config(monkeypatch, tmp_path):
    from tests._scoped_config_fixture import install_test_config
    install_test_config(monkeypatch, tmp_path, 'test')


class _EnterOnlyStdin:
    """Fake TTY stdin whose first keypress is Enter (confirm)."""

    def fileno(self) -> int:
        return 0

    def read(self, n: int) -> str:
        return '\r'


class _DummyLive:
    """Stand-in for rich.live.Live — the transient picker UI that erases
    itself; anything only rendered through it does NOT persist on screen."""

    def __init__(self, *a, **k):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, *a, **k):
        pass


def test_arrow_select_echo_keeps_every_option_visible(monkeypatch):
    """After the transient picker closes, the scrollback must retain ALL
    options with their descriptions (prices live there), chosen one marked —
    otherwise the user never sees what the un-chosen mode would have cost."""
    lines: list[str] = []
    monkeypatch.setattr(
        onboard.console, 'print',
        lambda *a, **k: lines.append(' '.join(str(x) for x in a)),
    )
    monkeypatch.setattr('rich.live.Live', _DummyLive)
    monkeypatch.setattr('sys.stdin', _EnterOnlyStdin())
    monkeypatch.setattr('termios.tcgetattr', lambda fd: None)
    monkeypatch.setattr('termios.tcsetattr', lambda fd, when, attrs: None)
    monkeypatch.setattr('tty.setcbreak', lambda fd: None)

    value = onboard._arrow_key_select(
        [('live', 'Live', '~$8.00, ~1m–10m — interactive speed'),
         ('batch', 'Batch', '~$4.00 (about half price), finishes within 24h')],
        title='Embedding mode (9,999 documents)',
    )

    assert value == 'live'
    blob = '\n'.join(lines)
    assert '$8.00' in blob and '$4.00' in blob
    assert '▶' in blob


@pytest.mark.asyncio
async def test_onboard_evolves_through_contract(monkeypatch, capsys):
    from cli.onboard import cmd_onboard

    invoked: list[str] = []
    seen_args: dict[str, argparse.Namespace] = {}

    def make_async_stub(name: str, rc: int = 0):
        async def _stub(args, **kwargs):
            invoked.append(name)
            seen_args[name] = args
            return rc
        return _stub

    def make_sync_stub(name: str, rc: int = 0):
        def _stub(args, **kwargs):
            invoked.append(name)
            seen_args[name] = args
            return rc
        return _stub

    # Paid phases — these are what onboard runs directly after approval.
    monkeypatch.setattr(
        'cli.onboard.cmd_catalog_describe',
        make_async_stub('catalog-describe'),
    )
    monkeypatch.setattr(
        'cli.onboard.cmd_generate', make_async_stub('generate'),
    )
    # onboard awaits the themes phase as a coroutine (its pipeline is
    # async). Stub the awaitable core, not the sync `cmd_themes`
    # dispatcher — the async stub also guards that onboard actually
    # awaits it.
    monkeypatch.setattr(
        'cli.onboard.cmd_themes_build', make_async_stub('themes'),
    )

    # Free phases live INSIDE the preview. Stub them so that if onboard
    # ever re-runs them directly, they'd show up in `invoked` (they must
    # not). The preview itself is stubbed below.
    monkeypatch.setattr('cli.index.cmd_discover', make_sync_stub('discover'))
    monkeypatch.setattr('cli.index.cmd_index', make_sync_stub('index'))
    monkeypatch.setattr(
        'cli.catalog.cmd_catalog_sync', make_async_stub('catalog-sync'),
    )

    # The preview (free phases + estimate) is delegated to cmd_dry_run.
    dry_run_calls: list[argparse.Namespace] = []
    dry_run_rc = {'rc': 0}

    async def fake_dry_run(args):
        dry_run_calls.append(args)
        return dry_run_rc['rc']

    monkeypatch.setattr('cli.onboard.cmd_dry_run', fake_dry_run)

    # Proceed decision (interactive prompt) — monkeypatched per demand.
    proceed = {'value': False}
    monkeypatch.setattr(
        'cli.onboard._prompt_proceed', lambda: proceed['value'],
    )

    # By default, the doc-type picker is non-interactive in tests
    # (no TTY) → returns all defaults; individual demands override it.
    base_kw = dict(
        source='test', model=None, db=None, verbose=False,
        approve=False, batch_mode='live', concurrency=None, types=None,
    )

    def _args(**over):
        return argparse.Namespace(**{**base_kw, **over})

    # ---- T1: no --approve, user declines → preview only -------------
    proceed['value'] = False
    rc = await cmd_onboard(_args())
    assert rc == 0
    assert len(dry_run_calls) == 1, 'preview must run exactly once'
    assert invoked == [], f'declining must run no paid phases; got {invoked}'
    out = capsys.readouterr().out
    assert '--approve' in out and 'ariadne onboard' in out and 'test' in out

    # ---- T2: no --approve, user accepts → paid phases after preview --
    dry_run_calls.clear(); invoked.clear(); seen_args.clear()
    proceed['value'] = True
    rc = await cmd_onboard(_args())
    assert rc == 0
    assert len(dry_run_calls) == 1, 'preview still runs once'
    # Only the paid phases run here — the free phases are NOT re-run by
    # onboard (they happened inside the single preview).
    assert invoked == ['catalog-describe', 'generate', 'themes'], (
        f'onboard must not re-run free phases; got {invoked}'
    )
    assert 'discover' not in invoked and 'index' not in invoked

    # ---- T3: --approve skips the prompt, runs paid phases -----------
    dry_run_calls.clear(); invoked.clear(); seen_args.clear()
    monkeypatch.setattr(
        'cli.onboard._prompt_proceed',
        lambda: (_ for _ in ()).throw(
            AssertionError('--approve must NOT prompt'),
        ),
    )
    rc = await cmd_onboard(_args(approve=True))
    assert rc == 0
    assert len(dry_run_calls) == 1
    assert invoked == ['catalog-describe', 'generate', 'themes']
    assert getattr(seen_args['catalog-describe'], 'batch', False) is False

    # ---- T4: --approve --batch → describe batched -------------------
    invoked.clear(); seen_args.clear()
    rc = await cmd_onboard(_args(approve=True, batch_mode='batch'))
    assert rc == 0
    assert getattr(seen_args['catalog-describe'], 'batch', False) is True

    # ---- T5: preview failure → no paid phases, propagate rc ---------
    dry_run_calls.clear(); invoked.clear(); seen_args.clear()
    dry_run_rc['rc'] = 9
    rc = await cmd_onboard(_args(approve=True))
    assert rc == 9, f'preview failure rc must propagate; got {rc}'
    assert invoked == [], 'no paid phases after a failed preview'
    dry_run_rc['rc'] = 0

    # ---- T6: paid-phase failure stops the pipeline ------------------
    invoked.clear(); seen_args.clear()
    monkeypatch.setattr(
        'cli.onboard.cmd_generate', make_async_stub('generate', rc=5),
    )
    rc = await cmd_onboard(_args(approve=True))
    assert rc == 5
    assert invoked == ['catalog-describe', 'generate'], (
        f'themes must not run after generate fails; got {invoked}'
    )
    monkeypatch.setattr(
        'cli.onboard.cmd_generate', make_async_stub('generate'),
    )

    # ---- T7: --concurrency reaches preview + paid phases ------------
    invoked.clear(); seen_args.clear()
    rc = await cmd_onboard(_args(approve=True, concurrency=8))
    assert rc == 0
    assert dry_run_calls[-1].concurrency == 8, (
        '--concurrency must reach the preview (for catalog-sync)'
    )
    assert seen_args['catalog-describe'].concurrency == 8
    assert seen_args['generate'].concurrency == 8

    # ---- T8: omitted --concurrency → per-phase defaults -------------
    invoked.clear(); seen_args.clear()
    rc = await cmd_onboard(_args(approve=True, concurrency=None))
    assert rc == 0
    assert seen_args['catalog-describe'].concurrency == 4
    assert seen_args['generate'].concurrency == 3
    # No --types and non-interactive picker → generate gets all defaults.
    assert seen_args['generate'].types == (
        'explanation,architecture,qa,gotcha,diagram'
    )

    # ---- T9: the interactive doc-type selection reaches generate ----
    invoked.clear(); seen_args.clear()
    monkeypatch.setattr(
        'cli.onboard._select_generate_doc_types',
        lambda defaults, off=frozenset(): ('explanation', 'qa'),
    )
    rc = await cmd_onboard(_args(approve=True))
    assert rc == 0
    assert seen_args['generate'].types == 'explanation,qa', (
        f"generate must use the selected doc types; got "
        f"{seen_args['generate'].types!r}"
    )

    # ---- T10: explicit --types wins and skips the picker ------------
    invoked.clear(); seen_args.clear()
    monkeypatch.setattr(
        'cli.onboard._select_generate_doc_types',
        lambda defaults, off=frozenset(): (_ for _ in ()).throw(
            AssertionError('--types must skip the interactive picker'),
        ),
    )
    rc = await cmd_onboard(_args(approve=True, types='architecture'))
    assert rc == 0
    assert seen_args['generate'].types == 'architecture'

    # ---- T11: on a TTY, onboard always opens the file browser and asks it to
    # offer the staleness modal — no pre-browser y/N prompts ---------------
    dry_run_calls.clear(); invoked.clear(); seen_args.clear()
    monkeypatch.setattr(
        'cli.onboard._select_onboard_dependencies', lambda *a: None)
    monkeypatch.setattr(  # T3 left this throwing; this demand runs without --approve
        'cli.onboard._prompt_proceed', lambda: proceed['value'])
    monkeypatch.setattr('sys.stdin.isatty', lambda: True)
    monkeypatch.setattr('sys.stdout.isatty', lambda: True)
    proceed['value'] = False          # stop after preview; we assert the flags
    rc = await cmd_onboard(_args())
    assert rc == 0
    assert dry_run_calls[-1].interactive is True      # browser always opens
    assert dry_run_calls[-1].offer_staleness is True  # staleness modal offered

    # ---- T12: a themes-phase failure is NON-FATAL — the completed (and, for
    # a spool, paid) generate+embed run must not be discarded because the
    # semantic-clustering augmentation failed at the last step -------------
    invoked.clear(); seen_args.clear()
    monkeypatch.setattr(
        'cli.onboard.cmd_themes_build', make_async_stub('themes', rc=1))
    # T10 left the doc-type picker as a throwing guard; restore a benign one
    # so this demand reaches the paid phases (proceed=True) without tripping it.
    monkeypatch.setattr(
        'cli.onboard._select_generate_doc_types',
        lambda defaults, off=frozenset(): ('explanation',))
    proceed['value'] = True
    rc = await cmd_onboard(_args(approve=True))
    assert rc == 0, 'themes failure must not fail onboard'
    assert 'generate' in invoked and 'themes' in invoked
    out = capsys.readouterr().out
    assert "'Building themes' failed" in out and 'continuing' in out


def test_select_generate_doc_types_off_defaults_are_dropped() -> None:
    """``off_types`` start off: on a non-TTY the picker returns the remaining
    types in order (a spool build defaults architecture/qa/diagram off).
    No ``off_types`` keeps the prior all-on behavior."""
    types = ('explanation', 'architecture', 'qa', 'gotcha', 'diagram')
    assert onboard._select_generate_doc_types(
        types, frozenset({'architecture', 'qa', 'diagram'}),
    ) == ('explanation', 'gotcha')
    assert onboard._select_generate_doc_types(types) == types
