"""Curated explorer colour themes — readability + conversion.

These guard the *reason* the curated set exists: every shipped palette must be
high-contrast (text never blends into the background), and the iTerm2 ->
Textual conversion must map a scheme's colours onto the right Textual roles.

The conversion test uses a synthetic scheme (neutral hex, not a real vendored
palette) so it pins the mapping logic, not any one scheme's values.
"""
from __future__ import annotations

import pytest

from cli.explorer_themes import (
    ALLOWED_THEME_NAMES,
    CURATED_THEMES,
    EXPLORER_DEFAULT_THEME,
    _contrast_ratio,
    _ensure_min_contrast,
    _relative_luminance,
    iterm_to_textual,
)

# A neutral synthetic windowsterminal-format scheme — dark bg, light fg, and
# distinct primaries so each role mapping is unambiguous.
_SYNTHETIC = {
    'name': 'Test Scheme',
    'slug': 'test-scheme',
    'background': '#101014',
    'foreground': '#f4f4f8',
    'red': '#dd3344',
    'green': '#22cc55',
    'yellow': '#ddcc33',
    'blue': '#3366dd',
    'purple': '#9933dd',
    'cyan': '#33cccc',
    'brightBlack': '#444455',
    'selectionBackground': '#2a2a44',
}


def test_curated_set_is_non_empty_and_self_consistent():
    # A real curated set, not silently empty, and the exposed name set + default
    # stay in sync with the actual Theme objects.
    names = {t.name for t in CURATED_THEMES}
    assert len(CURATED_THEMES) >= 10
    assert len(names) == len(CURATED_THEMES)            # no dup names
    assert EXPLORER_DEFAULT_THEME in names
    # Picker shows the curated set plus the terminal-default passthrough.
    assert ALLOWED_THEME_NAMES == names | {'ansi-dark'}


def test_every_curated_theme_is_readable():
    # The whole point: text is clearly distinct from the background. Foreground
    # meets WCAG AA (4.5); the muted/secondary colour stays readable (>= 3.0).
    # This also doubles as the curation gate — a low-contrast scheme can't ship.
    for theme in CURATED_THEMES:
        fg_ratio = _contrast_ratio(theme.foreground, theme.background)
        assert fg_ratio >= 4.5, f'{theme.name}: fg/bg contrast {fg_ratio:.2f}'

        muted = (theme.variables or {}).get('text-muted')
        assert isinstance(muted, str) and muted.startswith('#'), (
            f'{theme.name}: text-muted not a concrete hex ({muted!r})')
        muted_ratio = _contrast_ratio(muted, theme.background)
        assert muted_ratio >= 3.0, f'{theme.name}: muted contrast {muted_ratio:.2f}'


def test_iterm_to_textual_maps_palette_roles():
    theme = iterm_to_textual(_SYNTHETIC)

    assert theme.name == 'test-scheme'
    assert theme.background == '#101014'
    assert theme.foreground == '#f4f4f8'
    assert theme.success == '#22cc55'      # green -> success
    assert theme.error == '#dd3344'        # red   -> error
    assert theme.warning == '#ddcc33'      # yellow -> warning
    assert theme.primary == '#3366dd'      # blue  -> primary/accent
    assert theme.dark is True              # dark background

    # Muted text is derived and guaranteed readable against the background.
    muted = theme.variables['text-muted']
    assert _contrast_ratio(muted, theme.background) >= 4.5


def test_iterm_to_textual_detects_light_background():
    light = {**_SYNTHETIC, 'background': '#f4f4f8', 'foreground': '#101014'}
    assert iterm_to_textual(light).dark is False


def test_ensure_min_contrast_pushes_text_off_background():
    bg = '#202024'
    faint = '#303034'                       # nearly the same as bg
    assert _contrast_ratio(faint, bg) < 4.5  # precondition: unreadable

    fixed = _ensure_min_contrast(faint, bg, 4.5)
    assert _contrast_ratio(fixed, bg) >= 4.5
    # Dark bg -> nudged lighter (toward white), so luminance increases.
    assert _relative_luminance(fixed) > _relative_luminance(faint)


def test_ensure_min_contrast_leaves_readable_colour_untouched():
    # Already well past the threshold -> returned as-is, not over-corrected.
    assert _ensure_min_contrast('#ffffff', '#000000', 4.5) == '#ffffff'
