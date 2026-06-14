"""Tier 3 of the dry-run explorer — the headless ExplorerState.

``ExplorerState`` composes a Tier 1 ``ScanNode`` tree with a Tier 2
``{rel_path: NodeCost}`` map and drives navigation, exclusion, a live
total, and rule generation. The termios view + CLI wiring are smoke-tested
elsewhere; this state model carries the coverage.

Exclusion rules follow the verified walk semantics: a file excluded on its
own becomes an ``exclude`` glob (its rel-path); when *all* files under a
directory are excluded the rules collapse to a single ``exclude_dirs`` entry
(the dir basename — the only mechanism that recursively prunes a subtree,
since ``Path.match`` globs don't recurse).

Grown as one evolving ``test_explorer``, then split into the focused tests
below. Fixtures are synthetic: neutral names, hand-built tree + cost map.
"""
from __future__ import annotations

import pytest
from textual.screen import ModalScreen
from textual.widgets import OptionList, Tree

from cli.explorer_themes import ALLOWED_THEME_NAMES, EXPLORER_DEFAULT_THEME
from cli.explorer_ui import (
    _load_explorer_theme,
    _make_explorer_tui_app,
    _save_explorer_theme,
)
from config import Config
from docgen.cost_by_dir import NodeCost
from docgen.explorer_state import ExplorerState, Totals, apply_excludes
from docgen.scan_tree import ScanNode


def _file(name, rel, tokens):
    return ScanNode(name, rel, False, 1, tokens, tokens * 4, ())


def _dir(name, rel, children):
    return ScanNode(
        name, rel, True,
        sum(c.mapped_files for c in children),
        sum(c.content_tokens for c in children),
        sum(c.total_bytes for c in children),
        tuple(children),
    )


def _nc(rel, total, docs):
    return NodeCost(rel_path=rel, by_type=(), ingestion_cost=0.0, total=total, docs=docs)


def _fixture():
    """A small synthetic project:

        .                       $1.16  7 docs
          src/                  $0.15  5 docs
            a.py                $0.10  3 docs
            api/                $0.05  2 docs
              b.py              $0.05  2 docs
          vendor/               $1.00  1 doc   (cost hog)
            big.js              $1.00  1 doc
          notes.md              $0.01  1 doc
    """
    tree = _dir('proj', '.', [
        _dir('src', 'src', [
            _file('a.py', 'src/a.py', 1000),
            _dir('api', 'src/api', [_file('b.py', 'src/api/b.py', 500)]),
        ]),
        _dir('vendor', 'vendor', [_file('big.js', 'vendor/big.js', 100_000)]),
        _file('notes.md', 'notes.md', 50),
    ])
    costs = {
        '.': _nc('.', 1.16, 7),
        'src': _nc('src', 0.15, 5),
        'src/a.py': _nc('src/a.py', 0.10, 3),
        'src/api': _nc('src/api', 0.05, 2),
        'src/api/b.py': _nc('src/api/b.py', 0.05, 2),
        'vendor': _nc('vendor', 1.00, 1),
        'vendor/big.js': _nc('vendor/big.js', 1.00, 1),
        'notes.md': _nc('notes.md', 0.01, 1),
    }
    return tree, costs


def test_root_rows_order():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs, auto_excluded=('.git', 'dist'))

    rows = st.current_rows()
    selectable = [r for r in rows if not r.auto_excluded]
    # directories first then files, each ranked by cost descending
    assert [r.name for r in selectable] == ['vendor', 'src', 'notes.md']
    vendor = selectable[0]
    assert vendor.is_dir and vendor.est_cost == 1.00 and vendor.docs == 1
    assert vendor.tokens == 100_000 and not vendor.excluded
    # policy-pruned names appended as dimmed, excluded, non-selectable rows
    auto = [r for r in rows if r.auto_excluded]
    assert {r.name for r in auto} == {'.git', 'dist'}
    assert all(r.excluded for r in auto)


def test_navigate_in_and_up():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)

    st.enter('src')
    assert st.breadcrumb() == ['proj', 'src']
    assert [r.name for r in st.current_rows()] == ['api', 'a.py']  # dir then file
    st.enter('nope')  # no such directory child → no-op
    assert st.breadcrumb() == ['proj', 'src']
    st.up()
    assert st.breadcrumb() == ['proj']
    st.up()  # already at root → no-op
    assert st.breadcrumb() == ['proj']


def test_toggle_updates_total():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)

    assert st.live_total() == Totals(7, 1.16, 0, 0.0)
    st.toggle('vendor')
    t = st.live_total()
    assert t.excluded_docs == 1 and t.excluded_cost == pytest.approx(1.00)
    assert t.kept_docs == 6 and t.kept_cost == pytest.approx(0.16)
    st.toggle('vendor')  # restore
    assert st.live_total() == Totals(7, 1.16, 0, 0.0)


def test_toggle_at_depth():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)

    st.enter('src')
    st.toggle('src/api')
    api_row = next(r for r in st.current_rows() if r.name == 'api')
    assert api_row.excluded
    assert st.live_total().excluded_cost == pytest.approx(0.05)


def test_excluded_rules_dir_vs_file():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)

    st.toggle('src/api')   # whole directory → exclude_dirs (basename)
    st.toggle('notes.md')  # single file → exclude glob
    assert st.excluded_rules() == {'exclude_dirs': ['api'], 'exclude': ['notes.md']}


def test_excluded_rules_collapses_full_dir():
    tree, costs = _fixture()
    # Excluding every file under src individually collapses to one
    # exclude_dirs entry — never a per-file enumeration.
    st = ExplorerState(tree, costs)
    st.toggle('src/a.py')
    st.toggle('src/api/b.py')
    assert st.excluded_rules() == {'exclude_dirs': ['src'], 'exclude': []}

    # Toggling the whole vendor dir → exclude_dirs only.
    st2 = ExplorerState(tree, costs)
    st2.toggle('vendor')
    assert st2.excluded_rules() == {'exclude_dirs': ['vendor'], 'exclude': []}


def test_auto_excluded_not_selectable():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs, auto_excluded=('.git',))
    st.toggle('.git')  # policy row → no-op
    assert st.excluded_rules() == {'exclude_dirs': [], 'exclude': []}
    assert st.live_total() == Totals(7, 1.16, 0, 0.0)


def test_keep_directory_clears_stale_child_selections():
    # Exclude two files, then the whole dir, then keep the dir → fully clean.
    # (Previously the earlier file selections lingered and re-collapsed to the
    # directory even after it was "kept".)
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)
    st.toggle('src/a.py')
    st.toggle('src/api/b.py')
    st.toggle('src')   # exclude the whole dir — subsumes the two files
    assert st.excluded_rules() == {'exclude_dirs': ['src'], 'exclude': []}
    st.toggle('src')   # keep it again
    assert st.excluded_rules() == {'exclude_dirs': [], 'exclude': []}
    assert st.live_total() == Totals(7, 1.16, 0, 0.0)
    assert not st.is_excluded('src/a.py')
    assert not st.is_excluded('src/api/b.py')


def test_carve_out_keep_inside_excluded_dir():
    # Exclude a whole dir, then KEEP one child: the child is carved out, the
    # rest of the dir stays excluded, and the rules express it.
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)
    st.toggle('src')        # whole src excluded
    assert st.is_excluded('src/a.py') and st.is_excluded('src/api/b.py')
    st.toggle('src/a.py')   # keep this one
    assert not st.is_excluded('src/a.py')     # carved out
    assert st.is_excluded('src/api/b.py')     # rest still excluded
    # src is no longer fully excluded → its excluded subdir collapses, a.py kept
    assert st.excluded_rules() == {'exclude_dirs': ['api'], 'exclude': []}
    t = st.live_total()
    assert t.excluded_cost == pytest.approx(0.05) and t.excluded_docs == 2


def test_carve_out_toggle_off_re_excludes():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)
    st.toggle('src')        # exclude dir
    st.toggle('src/a.py')   # carve out (keep)
    assert not st.is_excluded('src/a.py')
    st.toggle('src/a.py')   # toggle again → re-excluded via the dir
    assert st.is_excluded('src/a.py')
    assert st.excluded_rules() == {'exclude_dirs': ['src'], 'exclude': []}


def test_live_total_skips_files_without_cost():
    # A scanned file with no cost entry (not in this run's generate set)
    # contributes nothing to the excluded total even when toggled.
    tree, costs = _fixture()
    del costs['notes.md']
    st = ExplorerState(tree, costs)
    st.toggle('notes.md')
    assert st.is_excluded('notes.md')
    t = st.live_total()
    assert t.excluded_cost == 0.0 and t.excluded_docs == 0


def test_apply_writes_excludes(tmp_path):
    # A source with pre-existing excludes; the explorer's rules must MERGE in,
    # not clobber (set_source_config replaces, so apply_excludes unions).
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text(
        'sources:\n'
        '  mylib:\n'
        '    path: /some/where\n'
        '    exclude_dirs: [preexisting]\n'
        '    exclude: ["**/keep.json"]\n'
    )
    tree, costs = _fixture()
    state = ExplorerState(tree, costs)
    state.toggle('vendor')      # dir → exclude_dirs
    state.toggle('notes.md')    # file → exclude

    assert apply_excludes(state, Config(config_path=cfg_file), 'mylib') is True

    reread = Config(config_path=cfg_file).get_source_config('mylib')
    assert set(reread.exclude_dirs) == {'preexisting', 'vendor'}
    assert set(reread.exclude) == {'**/keep.json', 'notes.md'}


def test_apply_noop_when_nothing_excluded(tmp_path):
    cfg_file = tmp_path / 'ariadne.yaml'
    cfg_file.write_text('sources:\n  mylib:\n    path: /some/where\n')
    tree, costs = _fixture()
    # Nothing toggled → no write, returns True, config untouched.
    assert apply_excludes(
        ExplorerState(tree, costs), Config(config_path=cfg_file), 'mylib',
    ) is True
    reread = Config(config_path=cfg_file).get_source_config('mylib')
    assert reread.exclude_dirs == ()
    assert reread.exclude == ()


def test_state_accessors():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)
    assert st.root is tree
    assert st.cost_of('vendor').total == 1.00
    assert st.cost_of('does/not/exist') is None
    st.toggle('src')
    assert st.is_excluded('src/a.py')  # ancestor excluded
    assert not st.is_excluded('vendor')


async def test_tui_app_toggle_and_apply():
    # Drive the full-screen Textual explorer headlessly: move the cursor to the
    # first child (children are ranked by cost, so it's the dearest one),
    # toggle it excluded, apply. The shared ExplorerState carries the rule.
    from cli.explorer_ui import _make_explorer_tui_app

    tree, costs = _fixture()
    state = ExplorerState(tree, costs)
    app = _make_explorer_tui_app(state)
    async with app.run_test() as pilot:
        await pilot.press('down')  # root → dearest child (vendor, $1.00)
        await pilot.press('x')     # exclude it
        await pilot.press('a')     # apply & exit
    assert app.applied
    assert state.excluded_rules() == {'exclude_dirs': ['vendor'], 'exclude': []}


async def test_apply_offers_staleness_modal_yes(tmp_path, monkeypatch):
    # When onboarding offers it, Apply pops a modal asking about staleness
    # exemption *after* the user has worked the tree; the apply is gated on the
    # answer, and 'y' records the choice for the caller to persist.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    tree, costs = _fixture()
    app = _make_explorer_tui_app(
        ExplorerState(tree, costs),
        offer_staleness=True, staleness_source='src1', staleness_exempt=False)
    async with app.run_test() as pilot:
        await pilot.press('a')                       # apply → staleness modal pops up
        assert isinstance(app.screen, ModalScreen)   # the pop-up is shown
        assert app.applied is False                  # apply gated on the modal answer
        await pilot.press('y')                       # mark exempt
    assert app.applied is True
    assert app.staleness_exempt is True


async def test_apply_offers_staleness_modal_no(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    tree, costs = _fixture()
    app = _make_explorer_tui_app(
        ExplorerState(tree, costs),
        offer_staleness=True, staleness_source='src1', staleness_exempt=False)
    async with app.run_test() as pilot:
        await pilot.press('a')
        await pilot.press('n')                       # keep staleness checks
    assert app.applied is True
    assert app.staleness_exempt is False


async def test_apply_skips_modal_when_not_offered():
    # Standalone dry-run -i (no offer) → Apply exits straight away, no modal.
    tree, costs = _fixture()
    app = _make_explorer_tui_app(ExplorerState(tree, costs))
    async with app.run_test() as pilot:
        await pilot.press('a')
    assert app.applied is True


async def test_apply_skips_modal_when_already_exempt(tmp_path, monkeypatch):
    # Already exempt → nothing to ask; Apply goes straight through, stays exempt.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))
    tree, costs = _fixture()
    app = _make_explorer_tui_app(
        ExplorerState(tree, costs),
        offer_staleness=True, staleness_source='src1', staleness_exempt=True)
    async with app.run_test() as pilot:
        await pilot.press('a')
    assert app.applied is True
    assert app.staleness_exempt is True


def test_set_costs_keeps_exclusions():
    tree, costs = _fixture()
    st = ExplorerState(tree, costs)
    st.toggle('vendor')                       # exclude before re-pricing
    st.set_costs({
        '.': _nc('.', 5.21, 14),
        'vendor': _nc('vendor', 5.00, 9),
        'vendor/big.js': _nc('vendor/big.js', 5.00, 9),
    })
    assert st.cost_of('vendor').total == 5.00   # new costs applied
    assert st.is_excluded('vendor/big.js')      # exclusion survived the swap


async def test_tui_doc_types_recost_live():
    # Toggling a doc type in the left panel re-prices the whole tree: here the
    # recost scales cost with the number of selected types, so unchecking one
    # drops the total.
    from cli.explorer_ui import _make_explorer_tui_app

    tree, _ = _fixture()
    doc_types = ('explanation', 'architecture', 'qa')

    def recost(selected):
        n = len(selected)
        return {
            '.': _nc('.', 0.30 * n, 3 * n),
            'vendor': _nc('vendor', 0.20 * n, n),
            'vendor/big.js': _nc('vendor/big.js', 0.20 * n, n),
            'src': _nc('src', 0.10 * n, 2 * n),
            'src/a.py': _nc('src/a.py', 0.10 * n, 2 * n),
        }

    state = ExplorerState(tree, recost(doc_types))
    app = _make_explorer_tui_app(
        state, doc_types=doc_types, selected=doc_types, recost=recost)
    async with app.run_test() as pilot:
        from textual.widgets import SelectionList

        sl = app.query_one('#doctypes', SelectionList)
        # each doc type shows its own total cost in the left panel
        assert all(
            '$' in str(sl.get_option_at_index(i).prompt)
            for i in range(sl.option_count)
        )
        assert app.selected_doc_types == doc_types
        before = state.live_total().kept_cost            # 0.30 * 3
        await pilot.press('1')                           # uncheck 'explanation'
        assert app.selected_doc_types == ('architecture', 'qa')
        after = state.live_total().kept_cost             # 0.30 * 2
        assert after < before


async def test_tui_doc_type_checkbox_glyph():
    # Selected types show a check (☑); unselected show nothing (no gray X).
    from textual.widgets import SelectionList

    from cli.explorer_ui import _make_explorer_tui_app

    tree, _ = _fixture()
    dt = ('explanation', 'architecture', 'qa')

    def recost(sel):
        return {'.': _nc('.', float(len(sel)), 3)}

    app = _make_explorer_tui_app(
        ExplorerState(tree, recost(dt)), doc_types=dt,
        selected=('explanation', 'qa'), recost=recost)
    async with app.run_test():
        sl = app.query_one('#doctypes', SelectionList)
        lines = [sl.render_line(i).text for i in range(sl.option_count)]
        assert lines[0].startswith('☑')        # explanation: selected
        assert not lines[1].startswith('☑')    # architecture: unselected → blank
        assert lines[2].startswith('☑')        # qa: selected


async def test_tui_tree_aligns_files_with_dirs():
    # Leaves are padded by 2 (dirs get Tree's 2-cell "▶ " toggle), so the
    # bar/$ columns line up across files and folders.
    from textual.widgets import Tree

    from cli.explorer_ui import _make_explorer_tui_app

    tree, costs = _fixture()
    app = _make_explorer_tui_app(ExplorerState(tree, costs))
    async with app.run_test():
        labels = {}

        def walk(node):
            if node.data is not None:
                labels[node.data.rel_path] = node.label.plain
            for child in node.children:
                walk(child)

        walk(app.query_one(Tree).root)
        assert not labels['vendor'].startswith('  ')   # expandable dir: no pad
        assert labels['notes.md'].startswith('  ')      # leaf file: padded


def test_explorer_theme_persistence(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    assert _load_explorer_theme() == EXPLORER_DEFAULT_THEME  # vibrant default when unset
    _save_explorer_theme('tokyo-night')
    assert _load_explorer_theme() == 'tokyo-night'           # curated theme round-trips
    _save_explorer_theme('nord')                             # built-in, no longer offered
    assert _load_explorer_theme() == EXPLORER_DEFAULT_THEME  # disallowed → falls back


async def test_theme_picker_preview_and_commit(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    tree, costs = _fixture()
    app = _make_explorer_tui_app(ExplorerState(tree, costs))
    async with app.run_test() as pilot:
        start = app.theme                  # the vibrant default (tmp empty)
        await pilot.press('t')             # open the picker
        await pilot.press('down')          # highlight next theme → live preview
        previewed = app.theme
        assert previewed != start          # preview applied while browsing
        await pilot.press('enter')         # commit
        assert app.theme == previewed
        assert _load_explorer_theme() == previewed   # persisted for next time


async def test_theme_picker_cancel_reverts(tmp_path, monkeypatch):
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    tree, costs = _fixture()
    app = _make_explorer_tui_app(ExplorerState(tree, costs))
    async with app.run_test() as pilot:
        start = app.theme
        await pilot.press('t')
        await pilot.press('down')
        await pilot.press('down')          # preview a couple of themes
        assert app.theme != start
        await pilot.press('escape')        # cancel → revert to the entry theme
        assert app.theme == start


async def test_theme_picker_lists_only_curated(tmp_path, monkeypatch):
    # The picker offers the curated, high-contrast set (+ the ansi terminal
    # default) and hides the bland built-ins it used to show.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))

    tree, costs = _fixture()
    app = _make_explorer_tui_app(ExplorerState(tree, costs))
    async with app.run_test() as pilot:
        await pilot.press('t')
        picker = app.screen.query_one(OptionList)
        offered = {
            picker.get_option_at_index(i).id
            for i in range(picker.option_count)
        }
        assert offered <= set(ALLOWED_THEME_NAMES)   # nothing outside the allowed set
        assert 'dracula' in offered                  # curated themes are listed
        assert 'rose-pine' not in offered            # bland built-ins are hidden
        assert 'nord' not in offered


async def test_tree_labels_are_coloured_by_theme(tmp_path, monkeypatch):
    # The render is no longer monochrome: directory names take the theme accent,
    # and the cost bar runs a success→error gradient by a node's share of cost.
    monkeypatch.setenv('XDG_CONFIG_HOME', str(tmp_path))   # unset → vibrant default

    tree, costs = _fixture()
    app = _make_explorer_tui_app(ExplorerState(tree, costs))
    async with app.run_test() as pilot:
        theme = app.current_theme
        styles: dict[str, str] = {}

        def walk(node):
            if node.data is not None and node.data is not app._state.root:
                styles[node.data.rel_path] = ' '.join(
                    str(span.style) for span in node.label.spans).lower()
            for child in node.children:
                walk(child)

        walk(app.query_one(Tree).root)

        # Directory names render in the accent colour, not plain/dim.
        assert theme.primary.lower() in styles['vendor']
        # Gradient: 'vendor' is the cost hog (~86% of total) → error colour;
        # a cheap leaf → success colour.
        assert theme.error.lower() in styles['vendor']
        assert theme.success.lower() in styles['src/a.py']


async def test_tui_left_returns_to_parent_dir():
    # In a subdirectory, ← on a file moves the cursor back to the dir name.
    from textual.widgets import Tree

    from cli.explorer_ui import _make_explorer_tui_app

    tree, costs = _fixture()
    state = ExplorerState(tree, costs)
    app = _make_explorer_tui_app(state)
    async with app.run_test() as pilot:
        widget = app.query_one(Tree)
        await pilot.press('down')   # vendor (dearest)
        await pilot.press('down')   # src
        await pilot.press('right')  # expand src
        await pilot.press('down')   # step onto src/a.py (a file)
        assert widget.cursor_node.data.is_dir is False
        await pilot.press('left')   # ← back to the directory name
        assert widget.cursor_node.data.rel_path == 'src'


def test_run_explorer_tui_is_async():
    # Must be a coroutine: cmd_dry_run runs under asyncio.run, so the explorer
    # is awaited via app.run_async(). A sync run_explorer_tui calling app.run()
    # would nest asyncio.run() and raise at runtime.
    import inspect

    from cli.explorer_ui import run_explorer_tui
    assert inspect.iscoroutinefunction(run_explorer_tui)


async def test_tui_app_cancel_keeps_clean():
    from cli.explorer_ui import _make_explorer_tui_app

    tree, costs = _fixture()
    state = ExplorerState(tree, costs)
    app = _make_explorer_tui_app(state)
    async with app.run_test() as pilot:
        await pilot.press('down')
        await pilot.press('x')     # toggle something...
        await pilot.press('q')     # ...but cancel
    assert app.applied is False
    # State still carries the toggle (run_explorer_tui raises on !applied, so
    # the caller discards it); here we just assert cancel didn't set applied.
