"""Human language aliases (java/scala/kotlin → jvm, javascript → typescript,
golang → go) live in the scip_languages registry, so spool eligibility derives
from one source and can't drift from it. And the unsupported-language complement
map can never silently overlap a registered language.
"""
from __future__ import annotations

from docgen.scip_languages import LANGUAGES, _EXT_TO_LANG


def _lang(name):
    return next(lang for lang in LANGUAGES if lang.name == name)


def test_registry_carries_human_aliases():
    assert {'java', 'scala', 'kotlin'} <= _lang('jvm').aliases
    assert 'javascript' in _lang('typescript').aliases
    assert 'golang' in _lang('go').aliases


def test_is_scip_eligible_covers_names_aliases_and_exts():
    from spools import is_scip_eligible

    for n in ('java', 'scala', 'kotlin', 'javascript', 'golang'):  # aliases
        assert is_scip_eligible(n), n
    for n in ('python', 'typescript', 'jvm', 'go', 'py', '.py', 'ts'):  # name/ext
        assert is_scip_eligible(n), n
    for n in ('ruby', 'rust', 'cobol', ''):  # genuinely unsupported
        assert not is_scip_eligible(n), n


def test_unsupported_ext_names_never_overlap_registry():
    # Locks the "keep in sync with scip_languages" caveat: an extension a
    # registered indexer covers must never also be named "unsupported".
    from spools import _UNSUPPORTED_CODE_EXT_NAMES
    overlap = set(_UNSUPPORTED_CODE_EXT_NAMES) & set(_EXT_TO_LANG)
    assert not overlap, f'{overlap}: both SCIP-supported and named unsupported'
