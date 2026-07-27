"""ncdu-style arrow-key picker for corpus git tags, used by ``spools create``
setup. Textual, mirroring the dry-run explorer (``cli/explorer_ui.py``):
:func:`_make_versions_picker_app` builds the app so it can be driven headlessly
via ``App.run_test()``; :func:`pick_versions` runs it for real.

The picker CASCADES: instead of a fixed list of pages, it takes the repo
``order`` plus a ``candidates_fn(repo, chosen) -> (tags, default, warn)`` and
recomputes each page from the picks made on the earlier pages — so an upstream
choice narrows the groups that follow (steps 6-8 of the create flow).
"""
from __future__ import annotations


def _make_versions_picker_app(order, candidates_fn):
    """Paginated single-select CASCADE over ``order`` (the corpus repos, in
    dependency order). ``candidates_fn(repo, chosen)`` returns that repo's
    ``(tags, default, warn)`` given the versions picked so far, so every page is
    re-narrowed from the previous pick. One page per repo shows a prominent
    ``repo (N of M)`` banner (with a compatibility warning when ``warn``) and the
    repo's compatible tags (cursor on the default). Enter picks and advances
    (exits on the last); ← goes back to revise, DROPPING the later picks that
    depended on it so they re-narrow; q/Esc keeps the current default for the
    rest. The chosen ``{repo: tag}`` lands on ``app.result``. Split out from
    :func:`pick_versions` so tests can drive it with a headless pilot."""
    from textual.app import App, ComposeResult
    from textual.binding import Binding
    from textual.widgets import Footer, Header, OptionList, Static
    from textual.widgets.option_list import Option

    order = list(order)

    class VersionsPicker(App):
        CSS = ('OptionList { height: 1fr; border: round $accent; } '
               '#banner { padding: 1 2; text-style: bold; }')
        BINDINGS = [
            Binding('left', 'back', 'Back'),
            Binding('q,escape', 'keep_rest', 'Keep defaults'),
        ]
        TITLE = 'Pick corpus versions'

        def __init__(self):
            super().__init__()
            self.page = 0
            self.result = {}                 # confirmed picks, repo -> tag
            self._tags = []                  # tags shown on the current page
            self._default = ''               # default for the current page

        def compose(self) -> ComposeResult:
            yield Header()
            yield Static(id='banner')
            yield OptionList(id='tags')
            yield Footer()

        def on_mount(self) -> None:
            self._load_page()

        def _load_page(self) -> None:
            repo = order[self.page]
            # This page (and everything after it) is about to be re-decided, so
            # drop any stale picks from here on — a revised upstream pick must
            # re-narrow the downstream, not keep what it chose before.
            for stale in order[self.page:]:
                self.result.pop(stale, None)
            tags, default, warn = candidates_fn(repo, dict(self.result))
            self._tags, self._default = list(tags), default
            note = ('   ⚠ compatibility unknown — showing the pinned version'
                    ' only; verify' if warn else '')
            self.query_one('#banner', Static).update(
                f'{repo}  —  package {self.page + 1} of {len(order)}{note}'
                f'   (Enter picks · ← back · q keeps {default})')
            option_list = self.query_one('#tags', OptionList)
            option_list.clear_options()
            option_list.add_options(
                [Option(t, id=str(i)) for i, t in enumerate(self._tags)])
            option_list.highlighted = (
                self._tags.index(default) if default in self._tags else 0)
            option_list.focus()

        def on_option_list_option_selected(self, event) -> None:
            self.result[order[self.page]] = self._tags[event.option_index]
            if self.page + 1 < len(order):
                self.page += 1
                self._load_page()
            else:
                self.exit()

        def action_back(self) -> None:
            if self.page > 0:
                self.page -= 1
                self._load_page()

        def action_keep_rest(self) -> None:
            # Keep the default (blessed pin) for every remaining repo, cascading
            # each default forward so the ones after it narrow against it.
            for repo in order[self.page:]:
                _tags, default, _warn = candidates_fn(repo, dict(self.result))
                self.result[repo] = default
            self.exit()

    return VersionsPicker()


def pick_versions(order, candidates_fn) -> dict:
    """Run the cascading picker for real (takes over the terminal); return
    ``{repo: chosen_tag}`` (defaults for anything left unpicked)."""
    app = _make_versions_picker_app(order, candidates_fn)
    app.run()
    return app.result
