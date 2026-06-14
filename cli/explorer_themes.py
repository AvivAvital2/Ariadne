"""Curated, high-contrast colour themes for the dry-run explorer.

The explorer used to offer Textual's built-in themes, several of which are
deliberately low-contrast (catppuccin, nord, solarized, rose-pine); on top of
that the explorer leaned on ``[dim]`` markup, so text blended into the
background and the whole UI read as a flat grey. This module vendors a curated
set of vibrant, verified-readable palettes sourced from the iTerm2-Color-Schemes
project (https://github.com/mbadolato/iTerm2-Color-Schemes, the
``windowsterminal/*.json`` exports) and converts each one into a Textual
:class:`~textual.theme.Theme`.

Every shipped theme is gated on WCAG contrast (``tests/test_explorer_themes.py``):
foreground vs background >= 4.5 (AA), and a *derived* muted colour that we
guarantee stays readable rather than relying on a blanket ``[dim]``.

The raw scheme dicts are kept verbatim (``_SCHEMES``) so their provenance is
obvious and the conversion stays the single, tested transform.
"""
from __future__ import annotations

from textual.color import Color
from textual.theme import Theme

# WCAG minimums. AA for normal text; muted/secondary text is held to the same
# bar so "dimmed" never means "invisible".
_TEXT_MIN = 4.5
_MUTED_MIN = 4.5


def _relative_luminance(color: str) -> float:
    """WCAG relative luminance of an ``#rrggbb`` colour (0 = black, 1 = white)."""
    r, g, b = Color.parse(color).normalized

    def _lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4

    return 0.2126 * _lin(r) + 0.7152 * _lin(g) + 0.0722 * _lin(b)


def _contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours (1.0 = identical, 21 = max)."""
    la, lb = _relative_luminance(a), _relative_luminance(b)
    hi, lo = max(la, lb), min(la, lb)
    return (hi + 0.05) / (lo + 0.05)


def _ensure_min_contrast(fg: str, bg: str, min_ratio: float) -> str:
    """``fg`` nudged away from ``bg`` until it clears ``min_ratio``.

    On a dark background we walk toward white, on a light one toward black, in
    small steps so the colour stays as close to the requested one as possible.
    A colour that already clears the ratio is returned unchanged.
    """
    if _contrast_ratio(fg, bg) >= min_ratio:
        return fg
    target = '#ffffff' if _relative_luminance(bg) < 0.5 else '#000000'
    src, dst = Color.parse(fg), Color.parse(target)
    factor = 0.0
    while factor < 1.0:
        factor += 0.05
        out = src.blend(dst, factor).hex
        if _contrast_ratio(out, bg) >= min_ratio:
            return out
    return Color.parse(target).hex


def iterm_to_textual(scheme: dict) -> Theme:
    """Convert an iTerm2 ``windowsterminal`` scheme dict into a Textual theme.

    Role mapping (terminal palette -> Textual semantic colour):
    background/foreground pass through; ``green/yellow/red`` become
    success/warning/error (the cost-bar gradient + totals lean on these);
    ``blue`` is the primary accent, ``cyan`` the secondary, ``purple`` the
    accent. ``surface``/``panel`` are derived by lifting the background toward
    the foreground so borders and panels read on both light and dark schemes.
    ``text-muted`` is derived from the foreground and contrast-clamped, and the
    selection colour drives the block cursor.
    """
    bg, fg = scheme['background'], scheme['foreground']
    dark = _relative_luminance(bg) < 0.5
    bg_c, fg_c = Color.parse(bg), Color.parse(fg)

    surface = bg_c.blend(fg_c, 0.07).hex
    panel = bg_c.blend(fg_c, 0.14).hex
    muted = _ensure_min_contrast(bg_c.blend(fg_c, 0.55).hex, bg, _MUTED_MIN)

    return Theme(
        name=scheme.get('slug') or scheme['name'].lower().replace(' ', '-'),
        primary=scheme['blue'],
        secondary=scheme['cyan'],
        accent=scheme['purple'],
        success=scheme['green'],
        warning=scheme['yellow'],
        error=scheme['red'],
        foreground=fg,
        background=bg,
        surface=surface,
        panel=panel,
        dark=dark,
        variables={
            'text-muted': muted,
            'block-cursor-background': scheme['selectionBackground'],
            'block-cursor-foreground': bg,
        },
    )


# --- Vendored schemes (iTerm2-Color-Schemes/windowsterminal/*.json) -----------
# Verbatim exports; ``slug`` is the explorer-facing theme id. Kept high-contrast
# and vibrant on purpose — the readability test enforces it.
_SCHEMES: list[dict] = [
    {
        'name': 'Dracula', 'slug': 'dracula',
        'black': '#21222c', 'red': '#ff5555', 'green': '#50fa7b',
        'yellow': '#f1fa8c', 'blue': '#bd93f9', 'purple': '#ff79c6',
        'cyan': '#8be9fd', 'white': '#f8f8f2', 'brightBlack': '#6272a4',
        'background': '#282a36', 'foreground': '#f8f8f2',
        'cursorColor': '#f8f8f2', 'selectionBackground': '#44475a',
    },
    {
        'name': 'Snazzy', 'slug': 'snazzy',
        'black': '#000000', 'red': '#fc4346', 'green': '#50fb7c',
        'yellow': '#f0fb8c', 'blue': '#49baff', 'purple': '#fc4cb4',
        'cyan': '#8be9fe', 'white': '#ededec', 'brightBlack': '#555555',
        'background': '#1e1f29', 'foreground': '#ebece6',
        'cursorColor': '#e4e4e4', 'selectionBackground': '#81aec6',
    },
    {
        'name': 'Catppuccin Mocha', 'slug': 'catppuccin-mocha',
        'black': '#45475a', 'red': '#f38ba8', 'green': '#a6e3a1',
        'yellow': '#f9e2af', 'blue': '#89b4fa', 'purple': '#f5c2e7',
        'cyan': '#94e2d5', 'white': '#a6adc8', 'brightBlack': '#585b70',
        'background': '#1e1e2e', 'foreground': '#cdd6f4',
        'cursorColor': '#f5e0dc', 'selectionBackground': '#585b70',
    },
    {
        'name': 'Ayu Mirage', 'slug': 'ayu-mirage',
        'black': '#171b24', 'red': '#ed8274', 'green': '#87d96c',
        'yellow': '#facc6e', 'blue': '#6dcbfa', 'purple': '#dabafa',
        'cyan': '#90e1c6', 'white': '#c7c7c7', 'brightBlack': '#686868',
        'background': '#1f2430', 'foreground': '#cccac2',
        'cursorColor': '#ffcc66', 'selectionBackground': '#409fff',
    },
    {
        'name': 'Tomorrow Night Eighties', 'slug': 'tomorrow-night-eighties',
        'black': '#000000', 'red': '#f2777a', 'green': '#99cc99',
        'yellow': '#ffcc66', 'blue': '#6699cc', 'purple': '#cc99cc',
        'cyan': '#66cccc', 'white': '#ffffff', 'brightBlack': '#595959',
        'background': '#2d2d2d', 'foreground': '#cccccc',
        'cursorColor': '#cccccc', 'selectionBackground': '#515151',
    },
    {
        'name': 'Night Owl', 'slug': 'night-owl',
        'black': '#011627', 'red': '#ef5350', 'green': '#22da6e',
        'yellow': '#addb67', 'blue': '#82aaff', 'purple': '#c792ea',
        'cyan': '#21c7a8', 'white': '#ffffff', 'brightBlack': '#575656',
        'background': '#011627', 'foreground': '#d6deeb',
        'cursorColor': '#7e57c2', 'selectionBackground': '#5f7e97',
    },
    {
        'name': 'Gruvbox Dark', 'slug': 'gruvbox-dark',
        'black': '#282828', 'red': '#cc241d', 'green': '#98971a',
        'yellow': '#d79921', 'blue': '#458588', 'purple': '#b16286',
        'cyan': '#689d6a', 'white': '#a89984', 'brightBlack': '#928374',
        'background': '#282828', 'foreground': '#ebdbb2',
        'cursorColor': '#ebdbb2', 'selectionBackground': '#665c54',
    },
    {
        'name': 'Monokai Soda', 'slug': 'monokai-soda',
        'black': '#1a1a1a', 'red': '#f4005f', 'green': '#98e024',
        'yellow': '#fa8419', 'blue': '#9d65ff', 'purple': '#f4005f',
        'cyan': '#58d1eb', 'white': '#c4c5b5', 'brightBlack': '#625e4c',
        'background': '#1a1a1a', 'foreground': '#c4c5b5',
        'cursorColor': '#f6f7ec', 'selectionBackground': '#343434',
    },
    {
        'name': 'TokyoNight', 'slug': 'tokyo-night',
        'black': '#15161e', 'red': '#f7768e', 'green': '#9ece6a',
        'yellow': '#e0af68', 'blue': '#7aa2f7', 'purple': '#bb9af7',
        'cyan': '#7dcfff', 'white': '#a9b1d6', 'brightBlack': '#414868',
        'background': '#1a1b26', 'foreground': '#c0caf5',
        'cursorColor': '#c0caf5', 'selectionBackground': '#33467c',
    },
    {
        'name': 'Synthwave Alpha', 'slug': 'synthwave-alpha',
        'black': '#241b30', 'red': '#e60a70', 'green': '#00986c',
        'yellow': '#adad3e', 'blue': '#6e29ad', 'purple': '#b300ad',
        'cyan': '#00b0b1', 'white': '#b9b1bc', 'brightBlack': '#7f7094',
        'background': '#241b30', 'foreground': '#f2f2e3',
        'cursorColor': '#f2f2e3', 'selectionBackground': '#6e29ad',
    },
    {
        'name': 'Catppuccin Latte', 'slug': 'catppuccin-latte',
        'black': '#5c5f77', 'red': '#d20f39', 'green': '#40a02b',
        'yellow': '#df8e1d', 'blue': '#1e66f5', 'purple': '#ea76cb',
        'cyan': '#179299', 'white': '#acb0be', 'brightBlack': '#6c6f85',
        'background': '#eff1f5', 'foreground': '#4c4f69',
        'cursorColor': '#dc8a78', 'selectionBackground': '#acb0be',
    },
    {
        'name': 'Material', 'slug': 'material',
        'black': '#212121', 'red': '#b7141f', 'green': '#457b24',
        'yellow': '#f6981e', 'blue': '#134eb2', 'purple': '#560088',
        'cyan': '#0e717c', 'white': '#afafaf', 'brightBlack': '#424242',
        'background': '#eaeaea', 'foreground': '#232322',
        'cursorColor': '#16afca', 'selectionBackground': '#c2c2c2',
    },
]

# Built once at import: the curated Theme objects + the lookup sets the explorer
# and its tests share.
CURATED_THEMES: list[Theme] = [iterm_to_textual(s) for s in _SCHEMES]

#: The default the explorer opens with when the user hasn't picked one — a
#: vibrant, instantly-recognisable palette rather than the terminal's own colours.
EXPLORER_DEFAULT_THEME = 'dracula'

#: Names the theme picker is allowed to show: the curated set plus the
#: terminal-default passthrough (Textual's ``ansi-dark``, the terminal's colours).
ALLOWED_THEME_NAMES: frozenset[str] = frozenset(
    {t.name for t in CURATED_THEMES} | {'ansi-dark'})
