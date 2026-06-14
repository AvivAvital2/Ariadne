"""Interactive dry-run explorer TUI + persisted colour theme.

Extracted from cli/generation.py. The full-screen Textual explorer behind
``dry-run -i`` (a tree of per-directory generate cost with exclude toggles),
plus its persisted theme. Pure UI — invoked by ``cmd_dry_run`` via
:func:`run_explorer_tui`; not a standalone command.
"""
from __future__ import annotations

import os
from pathlib import Path


def _explorer_theme_path() -> Path:
    """User-global file storing the explorer's chosen palette (XDG-aware, so
    the preference is per-user, not committed into a project's ariadne.yaml)."""
    base = os.environ.get('XDG_CONFIG_HOME') or str(Path.home() / '.config')
    return Path(base) / 'ariadne' / 'ui.json'


def _load_explorer_theme(default: str = 'ansi-dark') -> str:
    """The persisted explorer theme, or ``default`` when unset/unknown."""
    import json

    from textual.theme import BUILTIN_THEMES
    try:
        name = json.loads(_explorer_theme_path().read_text()).get('explorer_theme')
    except (OSError, ValueError):
        return default
    return name if name in BUILTIN_THEMES else default


def _save_explorer_theme(name: str) -> None:
    """Persist the explorer theme (best-effort — a pref write never crashes)."""
    import json

    path = _explorer_theme_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({'explorer_theme': name}))
    except OSError:
        pass


def _make_theme_picker(app):
    """A modal theme picker that previews each theme as you move through the
    list (commit on Enter → persist; Escape → revert to the theme on entry).
    Used by both ``t`` and Ctrl+P → Change theme. Module-level so it's
    headless-testable via ``App.run_test()``."""
    from textual.binding import Binding
    from textual.screen import ModalScreen
    from textual.widgets import OptionList
    from textual.widgets.option_list import Option

    names = sorted(app.available_themes)
    original = app.theme

    class _ThemePicker(ModalScreen):
        CSS = """
        _ThemePicker { align: center middle; }
        #themes { width: 44; max-height: 80%; border: round $panel; padding: 0 1; }
        """
        BINDINGS = [Binding('escape', 'cancel', 'Cancel')]

        def compose(self):
            picker = OptionList(*[Option(n, id=n) for n in names], id='themes')
            picker.border_title = 'theme — ↑/↓ preview · enter apply · esc cancel'
            yield picker

        def on_mount(self) -> None:
            picker = self.query_one(OptionList)
            picker.focus()
            if original in names:
                picker.highlighted = names.index(original)

        def on_option_list_option_highlighted(self, event) -> None:
            app.theme = event.option_id          # live preview while browsing

        def on_option_list_option_selected(self, event) -> None:
            app.theme = event.option_id
            _save_explorer_theme(event.option_id)
            app.sub_title = app._theme_subtitle()
            self.dismiss()

        def action_cancel(self) -> None:
            app.theme = original                  # revert the preview
            app.sub_title = app._theme_subtitle()
            self.dismiss()

    return _ThemePicker()


def _make_explorer_tui_app(state, *, doc_types=(), selected=None, recost=None):
    """Build the full-screen Textual explorer backing :func:`run_explorer_tui`.

    The whole source tree, expandable inline (Space/Enter on a dir), each row
    an ncdu-style cost bar + ``$`` + ``%`` of total + doc count, ranked by cost.
    Excluded rows show struck-through; the footer keeps a live KEEP/EXCLUDE
    total. ``x`` toggles exclude, ``a`` applies, ``q`` cancels, ``t`` cycles theme.

    When ``recost`` is given — ``callable(selected_doc_types) -> {rel: NodeCost}``
    — a left-hand checkbox list of ``doc_types`` is shown and toggling a type
    (click, or number keys ``1``-``9``) re-prices the whole tree live, since the
    doc-type set is the other big cost lever besides excluding files.
    ``selected`` is the initially-checked set (default: all of ``doc_types``).

    Separated from ``run_explorer_tui`` so the app can be driven headlessly via
    ``App.run_test()``.
    """
    from rich.segment import Segment
    from rich.style import Style
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.containers import Horizontal
    from textual.strip import Strip
    from textual.widgets import Footer, Header, OptionList, SelectionList, Static, Tree
    from textual.widgets.option_list import OptionDoesNotExist
    from textual.widgets.selection_list import Selection

    class _DocTypeCheckList(SelectionList):
        """A checklist that shows a check (☑) only when an item is selected —
        instead of Textual's always-rendered 'X' (gray when unselected)."""

        def render_line(self, y) -> Strip:
            prompt = OptionList.render_line(self, y)  # the label, no button
            _, scroll_y = self.scroll_offset
            index = scroll_y + y
            try:
                selection = self.get_option_at_index(index)
            except OptionDoesNotExist:
                return prompt
            first = next(iter(prompt), None)
            base = (first.style if first else None) or self.rich_style
            meta = Style(meta={'option': index})  # keep click-to-toggle working
            if selection.value in self._selected:
                check = self.get_component_rich_style('selection-list--button-selected')
                style = Style.from_color(check.color, base.bgcolor) + meta
                return Strip([Segment('☑ ', style=style), *prompt])
            return Strip([Segment('  ', style=base + meta), *prompt])

    doc_types = tuple(doc_types)
    has_doc_types = recost is not None and bool(doc_types)

    # Per-doc-type total cost over the whole source, for the left-panel labels.
    # Each type is independent/additive, so a type's total is just recost for
    # that one type. (Like the tree, these are full-source figures; the footer
    # carries the live kept-vs-excluded split.)
    type_cost: dict = {}
    if has_doc_types:
        for _t in doc_types:
            _root = recost((_t,)).get('.')
            type_cost[_t] = _root.total if _root else 0.0

    def _grand_total():
        # Read live — the doc-type selection re-prices the whole tree, so the
        # bar denominator (root total) can change between renders.
        rc = state.cost_of(state.root.rel_path)
        return rc.total if rc else 0.0

    def _bar(value, width=12):
        # Filled portion in the default foreground, the remainder dimmed — a
        # quiet, monochrome gauge rather than a bright accent bar.
        gt = _grand_total()
        if gt <= 0:
            return f'[dim]{"╌" * width}[/dim]'
        filled = max(0, min(width, round(width * value / gt)))
        return f'{"━" * filled}[dim]{"╌" * (width - filled)}[/dim]'

    def _label(node):
        # Rich markup (Tree labels use Text.from_markup) — attribute-only, no
        # colour; emphasis via weight/dim so the muted theme palette carries it.
        gt = _grand_total()
        cost = state.cost_of(node.rel_path)
        docs = cost.docs if cost else 0
        dollars = cost.total if cost else 0.0
        pct = (dollars / gt * 100) if gt else 0.0
        name = node.name + ('/' if node.is_dir else '')
        # Expandable dirs get a 2-cell "▶ " toggle from Tree; leaves get none,
        # so pad leaves by 2 to keep the bar/$ columns aligned across rows.
        pad = '' if (node.is_dir and node.children) else '  '
        body = (
            f'{pad}{_bar(dollars)}  [b]${dollars:>7.2f}[/b]  '
            f'[dim]{pct:>3.0f}%   {docs:>4} docs[/dim]  {name}'
        )
        if state.is_excluded(node.rel_path):
            return f'[dim strike]{body}[/dim strike]'
        return body

    def _by_cost(node):
        cost = state.cost_of(node.rel_path)
        return (-(cost.total if cost else 0.0), node.name)

    class _ExplorerTuiApp(App):
        CSS = """
        #panes { height: 1fr; }
        #doctypes { width: 30; border-right: solid $panel; padding: 1 1; }
        #tree { width: 1fr; padding: 1 2; }
        #total { height: 1; padding: 0 2; color: $text-muted; }
        """
        # Space/Enter expand the tree (Tree built-ins); x excludes, a applies;
        # 1-9 toggle a doc type (when the doc-type panel is shown).
        BINDINGS = [
            Binding('left,h', 'back', 'Back', show=False),
            Binding('right,l', 'forward', 'Open', show=False),
            Binding('x', 'toggle', 'Exclude / keep'),
            Binding('t', 'pick_theme', 'Theme'),
            *[Binding(str(i + 1), f'toggle_doc({i})', show=False) for i in range(9)],
            Binding('a', 'apply', 'Apply & quit'),
            Binding('q', 'cancel', 'Cancel'),
        ]

        def __init__(self, state) -> None:
            super().__init__()
            self._state = state
            self.applied = False
            self._doc_types = doc_types
            self._selected = set(doc_types if selected is None else selected)
            self._recost = recost
            self._built = False  # NB: 'App._ready' is reserved by Textual

        @property
        def selected_doc_types(self) -> tuple:
            return tuple(t for t in self._doc_types if t in self._selected)

        def compose(self) -> ComposeResult:
            yield Header(show_clock=False)
            if has_doc_types:
                with Horizontal(id='panes'):
                    yield _DocTypeCheckList(
                        *[
                            Selection(
                                f'{t:<13}${type_cost.get(t, 0.0):>8.2f}',
                                t, t in self._selected,
                            )
                            for t in self._doc_types
                        ],
                        id='doctypes',
                    )
                    yield Tree('root', id='tree')
            else:
                yield Tree('root', id='tree')
            yield Static(id='total')
            yield Footer()

        def _theme_subtitle(self) -> str:
            return f'{self._state.root.name}   ·   theme: {self.theme}  (t to change)'

        def on_mount(self) -> None:
            # Restore the user's chosen palette ('ansi-dark' = the terminal's
            # own colours, the default). 't' cycles the curated set + persists.
            self.theme = _load_explorer_theme()
            self.title = 'ariadne · dry-run explorer'
            self.sub_title = self._theme_subtitle()
            if has_doc_types:
                self.query_one('#doctypes', SelectionList).border_title = 'doc types'
            tree = self.query_one(Tree)
            tree.root.data = self._state.root
            tree.root.set_label(f'[b]{self._state.root.name}[/]')
            self._build(tree.root, self._state.root)
            tree.root.expand()
            tree.focus()
            self._update_total()
            self._built = True  # gate live re-cost until the tree exists

        def _build(self, tnode, snode) -> None:
            for child in sorted(snode.children, key=_by_cost):
                if child.is_dir and child.children:
                    self._build(tnode.add(_label(child), data=child), child)
                else:
                    tnode.add_leaf(_label(child), data=child)

        def _relabel(self, tnode) -> None:
            if tnode.data is not None and tnode.data is not self._state.root:
                tnode.set_label(_label(tnode.data))
            for child in tnode.children:
                self._relabel(child)

        def _update_total(self) -> None:
            # Static uses Textual content markup, so theme variables resolve
            # here (unlike the Rich-markup tree labels). Kept monochrome +
            # muted: bold numbers, dimmed labels, no saturated colour.
            t = self._state.live_total()
            tail = (
                f'     [$text-muted]types:[/] {", ".join(self.selected_doc_types) or "(none)"}'
                if has_doc_types
                else '     [$text-muted]generate runs on the subscription · $0 metered[/]'
            )
            self.query_one('#total', Static).update(
                f'[$text-muted]keep[/]  [b]{t.kept_docs}[/b] docs · '
                f'[b]${t.kept_cost:.2f}[/b]'
                f'     [$text-muted]exclude[/]  [b]{t.excluded_docs}[/b] docs · '
                f'[b]${t.excluded_cost:.2f}[/b]'
                f'{tail}'
            )

        def action_back(self) -> None:
            # ← : collapse an expanded directory, else jump the cursor to the
            # parent directory (so left on a file lands on its dir name).
            tree = self.query_one(Tree)
            node = tree.cursor_node
            if node is None:
                return
            if node.allow_expand and node.is_expanded:
                node.collapse()
            else:
                tree.action_cursor_parent()

        def action_forward(self) -> None:
            # → : expand a collapsed directory, or step into the first child of
            # an already-expanded one.
            tree = self.query_one(Tree)
            node = tree.cursor_node
            if node is None or not node.allow_expand:
                return
            if node.is_expanded:
                tree.action_cursor_down()
            else:
                node.expand()

        def action_toggle(self) -> None:
            node = self.query_one(Tree).cursor_node
            if (
                node is not None and node.data is not None
                and node.data.rel_path != self._state.root.rel_path
            ):
                self._state.toggle(node.data.rel_path)
                self._relabel(self.query_one(Tree).root)
                self._update_total()

        def _apply_recost(self) -> None:
            # Re-price every node for the current doc-type selection and refresh.
            if not self._built or self._recost is None:
                return
            self._state.set_costs(self._recost(self.selected_doc_types))
            self._relabel(self.query_one(Tree).root)
            self._update_total()

        def on_selection_list_selected_changed(self, event) -> None:
            self._selected = set(event.control.selected)
            self._apply_recost()

        def action_toggle_doc(self, idx: int) -> None:
            # Number-key shortcut for the left-hand checkbox panel.
            if not has_doc_types or idx >= len(self._doc_types):
                return
            self.query_one('#doctypes', SelectionList).toggle(self._doc_types[idx])

        def search_themes(self) -> None:
            # Override the built-in (Ctrl+P → Change theme) so it previews live.
            self.push_screen(_make_theme_picker(self))

        def action_pick_theme(self) -> None:
            self.search_themes()

        def action_apply(self) -> None:
            self.applied = True
            self.exit()

        def action_cancel(self) -> None:
            self.exit()

    return _ExplorerTuiApp(state)


async def run_explorer_tui(state, *, doc_types=(), selected=None, recost=None):
    """Full-screen Textual explorer for ``ariadne dry-run --interactive``:
    x toggles an exclude, a applies & quits, q cancels (raises
    ``KeyboardInterrupt``); Space/Enter expand dirs; 1-9 toggle a doc type when
    the doc-type panel is shown. Returns the **app** so the caller can read the
    final ``selected_doc_types``. Logic is headless-tested via
    ``_make_explorer_tui_app`` + ``App.run_test()``.

    Async because ``cmd_dry_run`` already runs under ``asyncio.run`` — use
    ``run_async()`` rather than ``app.run()`` (which would nest event loops).
    """
    app = _make_explorer_tui_app(
        state, doc_types=doc_types, selected=selected, recost=recost)
    await app.run_async()
    if not app.applied:
        raise KeyboardInterrupt
    return app
