"""Per-language doc-type override (#2 backend).

A source can cap which doc types a given language receives — the doc-type
screen's per-format excludes (e.g. "don't generate ``architecture`` for YAML").
The override REPLACES the static ``LANGUAGE_DOC_TYPES`` default for that
language; unlisted languages keep their default. Slice 2a is the filter
primitive; config wiring, generation threading, and the web UI follow.
"""
from __future__ import annotations
import yaml
from config import Config

from docgen.prompts import filter_doc_types_for_language


def test_override_caps_a_language_below_its_static_default():
    """An override for a language replaces its static set: Python normally gets
    the full set, but an override of ('explanation',) caps it — so requesting
    architecture/qa too yields only explanation."""
    requested = ('explanation', 'architecture', 'qa')
    # No override → the static LANGUAGE_DOC_TYPES['python'] (full set) applies.
    assert filter_doc_types_for_language(requested, 'python') == (
        'explanation', 'architecture', 'qa',
    )
    # Override caps python to explanation only.
    assert filter_doc_types_for_language(
        requested, 'python', override={'python': ('explanation',)},
    ) == ('explanation',)


def test_override_only_affects_listed_languages():
    """An override entry for one language doesn't touch others — an unlisted
    language uses its static default, and an unknown language stays unfiltered."""
    requested = ('explanation', 'architecture')
    ov = {'yaml': ('explanation',)}
    # python not in the override → static default (full) → both kept.
    assert filter_doc_types_for_language(requested, 'python', override=ov) == (
        'explanation', 'architecture',
    )
    # yaml in the override → only explanation.
    assert filter_doc_types_for_language(requested, 'yaml', override=ov) == (
        'explanation',
    )
    # unknown language, no override entry → unfiltered (legacy behavior).
    assert filter_doc_types_for_language(requested, 'cobol', override=ov) == (
        'explanation', 'architecture',
    )
def test_doc_types_by_language_parses_from_config(tmp_path):
    """A source's ``doc_types_by_language`` parses into a {language: tuple} map —
    YAML lists become tuples, a bare scalar becomes a one-element tuple, and an
    unconfigured source defaults to empty. This is the per-format excludes the
    doc-type screen persists per source."""
    src = tmp_path / 'src'
    src.mkdir()
    (tmp_path / 'ariadne.yaml').write_text(
        'sources:\n'
        '  capped:\n'
        f'    path: {src}\n'
        '    doc_types_by_language:\n'
        '      python: [explanation]\n'
        '      yaml: [explanation, architecture]\n'
        '      html: explanation\n'
        '  plain:\n'
        f'    path: {src}\n',
        encoding='utf-8',
    )
    cfg = Config(tmp_path / 'ariadne.yaml')
    assert cfg.get_source_config('capped').doc_types_by_language == {
        'python': ('explanation',),
        'yaml': ('explanation', 'architecture'),
        'html': ('explanation',),
    }
    assert cfg.get_source_config('plain').doc_types_by_language == {}
def test_doc_types_by_language_round_trips_through_set_source_config(tmp_path):
    """set_source_config writes doc_types_by_language to ariadne.yaml as plain
    YAML lists (not python tuples), and a fresh Config re-reads it as the
    tuple-valued map — the persistence path behind the doc-type screen's
    per-format excludes."""
    src = tmp_path / 'src'
    src.mkdir()
    (tmp_path / 'ariadne.yaml').write_text(
        f'sources:\n  mylib:\n    path: {src}\n', encoding='utf-8')
    cfg = Config(tmp_path / 'ariadne.yaml')
    assert cfg.set_source_config(
        'mylib',
        doc_types_by_language={
            'python': ['explanation'],
            'yaml': ['explanation', 'architecture'],
        },
    )
    # Written as clean YAML lists, not !!python/tuple.
    raw = yaml.safe_load((tmp_path / 'ariadne.yaml').read_text(encoding='utf-8'))
    assert raw['sources']['mylib']['doc_types_by_language'] == {
        'python': ['explanation'],
        'yaml': ['explanation', 'architecture'],
    }
    # A fresh read coerces back to the tuple-valued override.
    reread = Config(tmp_path / 'ariadne.yaml')
    assert reread.get_source_config('mylib').doc_types_by_language == {
        'python': ('explanation',),
        'yaml': ('explanation', 'architecture'),
    }
