"""Pin Go as a first-class doc-type language.

Go has a registered ``scip-go`` indexer and catalog extraction (exactly
like JavaScript/Java), so it must be a full doc-type language — not fall
through the unknown-language fallback. This pins the three ``prompts``
tables AND the ``pricing`` estimator's language mirror, which had no
``.go`` case: a Go corpus was priced through the generic None-language
path instead of Go's own doc-type set, and fixing detection alone would
have capped it at ('explanation',). Both must agree with the real
extractor (``catalog_extractor._detect_language('*.go') == 'go'``).
"""
from __future__ import annotations

from pathlib import Path

from docgen.pricing import _detect_language, _supported_doc_types_for
from docgen.prompts import (
    LANGUAGE_DOC_TYPES,
    LANGUAGE_FENCE,
    LANGUAGE_FRAMING,
)

FULL_SET = {'explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'}


class TestGoIsFirstClass:
    def test_doc_types_full_set(self) -> None:
        # SCIP-indexed like JS/Java — Go earns the full set, not the
        # ('explanation',) fallback reserved for prose/data formats.
        assert 'go' in LANGUAGE_DOC_TYPES
        assert FULL_SET.issubset(set(LANGUAGE_DOC_TYPES['go']))

    def test_fence_and_framing(self) -> None:
        assert LANGUAGE_FENCE['go'] == 'go'
        assert LANGUAGE_FRAMING['go'] == 'Go'

    def test_pricing_detects_go_extension(self) -> None:
        # The estimator mirror must agree with the real extractor, else a
        # Go corpus is priced through the generic unknown-language branch.
        assert _detect_language(Path('pkg/server.go')) == 'go'

    def test_pricing_supported_doc_types_match_generation(self) -> None:
        # Generation's filter allows all requested types for go; the
        # estimator must not silently cap go at ('explanation',).
        assert FULL_SET.issubset(set(_supported_doc_types_for('go')))
