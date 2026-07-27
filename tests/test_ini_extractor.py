"""Contract for INI/`.conf` catalog extraction.

`.conf` is overloaded (HOCON is canonical, but INI-style files —
Sphinx `theme.conf`, `setup.cfg`, systemd units — share the extension).
The catalog dispatch sniffs a `[section]` header and routes those to the
INI extractor (per-section + per-key elements); everything else stays on
the HOCON-or-file-index path.
"""
from __future__ import annotations

from pathlib import Path

from docgen.ini_extractor import looks_like_ini

_SPHINX_THEME = """\
[theme]
inherit = basic
stylesheet = css/theme.css

[options]
sticky_navigation = False
logo_only =
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    p = tmp_path / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding='utf-8')
    return p


def test_subtype_language_include_ini() -> None:
    from typing import get_args

    from docgen.catalog_extractor import Language, Subtype

    assert 'ini' in get_args(Language)
    assert 'ini_section' in get_args(Subtype)
    assert 'ini_key' in get_args(Subtype)


def test_sniff_detects_ini_section() -> None:
    assert looks_like_ini(_SPHINX_THEME)
    # A leading comment doesn't hide the section header.
    assert looks_like_ini('# a theme\n\n[theme]\ninherit = basic\n')
    # HOCON block form is NOT INI.
    assert not looks_like_ini('akka {\n  http { port = 8080 }\n}\n')
    # HOCON key form is NOT INI.
    assert not looks_like_ini('foo = "bar"\na.b.c = 1\n')
    # A single-line HOCON root array must not look like an INI section.
    assert not looks_like_ini('["a", "b", "c"]\n')


def test_ini_conf_produces_per_section_and_per_key_elements(tmp_path: Path) -> None:
    from docgen.catalog_extractor import extract_elements

    f = _write(tmp_path, 'theme.conf', _SPHINX_THEME)
    elements = extract_elements(f, tmp_path)

    by_qn = {e.qualified_name: e for e in elements}
    langs = {e.language for e in elements}
    assert langs == {'ini'}

    # sections
    section = next(e for e in elements if e.subtype == 'ini_section'
                   and e.qualified_name.endswith('.theme'))
    assert section.line_start == 1
    # keys, parented under their section
    inherit = next(e for e in elements if e.qualified_name.endswith('.theme.inherit'))
    assert inherit.subtype == 'ini_key'
    assert inherit.parent_qualified_name.endswith('.theme')
    assert inherit.line_start == 2
    # empty-value key is still an element
    assert any(e.qualified_name.endswith('.options.logo_only') for e in elements)
    # section keys don't leak across sections
    assert any(e.qualified_name.endswith('.options.sticky_navigation')
               for e in elements)
    assert not any(e.qualified_name.endswith('.theme.sticky_navigation')
                   for e in elements)


def test_hocon_conf_still_routes_to_hocon(tmp_path: Path) -> None:
    """The sniff must not hijack a genuine HOCON `.conf`."""
    from docgen.catalog_extractor import extract_elements

    f = _write(tmp_path, 'application.conf', 'akka {\n  port = 8080\n}\n')
    elements = extract_elements(f, tmp_path)
    assert elements and all(e.language == 'hocon' for e in elements)
