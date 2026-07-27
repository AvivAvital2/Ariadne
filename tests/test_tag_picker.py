"""Headless tests for the paginated, CASCADING compatibility-aware version
picker (cli/tag_picker.py), driven with a Textual pilot like the dry-run
explorer. Each page's candidates are recomputed from the picks made on the
earlier pages, so an upstream choice narrows the groups that follow."""
from cli.tag_picker import _make_versions_picker_app
from textual.widgets import Static


def _static(table):
    """A candidates_fn from a static ``{repo: (tags, default, warn)}`` map — no
    cascade (a repo's candidates don't depend on the prior picks)."""
    def candidates(repo, chosen):
        return table[repo]
    return candidates


def _two():
    return ['spark', 'delta'], _static({
        'spark': (['v4.0.0', 'v3.5.1'], 'v4.0.0', False),
        'delta': (['v4.0.0', 'v3.3.1'], 'v4.0.0', False),
    })


async def test_arrow_then_enter_picks_and_advances():
    app = _make_versions_picker_app(*_two())
    async with app.run_test() as pilot:
        await pilot.press('down')
        await pilot.press('enter')
        await pilot.press('enter')
    assert app.result == {'spark': 'v3.5.1', 'delta': 'v4.0.0'}


async def test_starts_on_default():
    app = _make_versions_picker_app(
        ['spark'], _static({'spark': (['v4.0.0', 'v3.5.1'], 'v3.5.1', False)}))
    async with app.run_test() as pilot:
        await pilot.press('enter')
    assert app.result == {'spark': 'v3.5.1'}


async def test_back_revises_previous_page():
    app = _make_versions_picker_app(*_two())
    async with app.run_test() as pilot:
        await pilot.press('enter')
        await pilot.press('left')
        await pilot.press('down')
        await pilot.press('enter')
        await pilot.press('enter')
    assert app.result == {'spark': 'v3.5.1', 'delta': 'v4.0.0'}


async def test_quit_keeps_defaults():
    app = _make_versions_picker_app(*_two())
    async with app.run_test() as pilot:
        await pilot.press('down')
        await pilot.press('q')
    assert app.result == {'spark': 'v4.0.0', 'delta': 'v4.0.0'}


async def test_default_not_a_tag_opens_at_top():
    app = _make_versions_picker_app(
        ['spark'], _static({'spark': (['v4.0.0', 'v3.5.1'], 'main', False)}))
    async with app.run_test() as pilot:
        await pilot.press('enter')
    assert app.result == {'spark': 'v4.0.0'}


async def test_back_at_first_page_is_noop():
    app = _make_versions_picker_app(*_two())
    async with app.run_test() as pilot:
        await pilot.press('left')
        await pilot.press('enter')
        await pilot.press('enter')
    assert app.result == {'spark': 'v4.0.0', 'delta': 'v4.0.0'}


async def test_warn_shows_in_banner():
    app = _make_versions_picker_app(
        ['sdk'], _static({'sdk': (['v0.68.0', 'v0.67.0'], 'main', True)}))
    async with app.run_test() as pilot:
        banner = str(app.query_one('#banner', Static).render())
    assert 'compatibility unknown' in banner and 'sdk' in banner


async def test_pick_cascades_to_next_page():
    """The heart of the feature: page 2's candidates are recomputed from the
    page-1 pick. Choosing spark v4.0.1 unlocks delta's 4.0 line; a different
    spark pick would have offered delta a different (here, incompatible) set."""
    def candidates(repo, chosen):
        if repo == 'spark':
            return ['v4.0.1', 'v3.5.1'], 'v4.0.1', False
        # delta's compatible tags depend on which spark was chosen upstream
        if chosen.get('spark') == 'v4.0.1':
            return ['v4.0.0'], 'v4.0.0', False
        return ['v1.2.0'], 'v1.2.0', True

    app = _make_versions_picker_app(['spark', 'delta'], candidates)
    async with app.run_test() as pilot:
        await pilot.press('enter')      # spark: keep default v4.0.1
        await pilot.press('enter')      # delta: only the 4.0-line tag offered
    assert app.result == {'spark': 'v4.0.1', 'delta': 'v4.0.0'}


async def test_revised_upstream_pick_re_narrows_downstream():
    """Going back and changing spark drops the delta pick and recomputes it
    from the NEW spark — the cascade re-narrows, it doesn't keep a stale pick."""
    def candidates(repo, chosen):
        if repo == 'spark':
            return ['v4.0.1', 'v3.5.1'], 'v4.0.1', False
        return (['v4.0.0'], 'v4.0.0', False) if chosen.get('spark') == 'v4.0.1' \
            else (['v3.3.9'], 'v3.3.9', False)

    app = _make_versions_picker_app(['spark', 'delta'], candidates)
    async with app.run_test() as pilot:
        await pilot.press('enter')      # spark v4.0.1 -> delta offered v4.0.0
        await pilot.press('left')       # back to spark
        await pilot.press('down')       # highlight v3.5.1
        await pilot.press('enter')      # spark v3.5.1 -> delta must re-narrow
        await pilot.press('enter')      # delta: now v3.3.9, not the stale v4.0.0
    assert app.result == {'spark': 'v3.5.1', 'delta': 'v3.3.9'}
