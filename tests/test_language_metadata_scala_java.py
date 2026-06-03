"""Pin LANGUAGE_DOC_TYPES / LANGUAGE_FENCE / LANGUAGE_FRAMING entries
for scala and java (SCIP follow-up #4, #5).

The fallback path in ``render_user_template`` and
``filter_doc_types_for_language`` produces correct-looking values for
unknown languages, but explicit entries make scala/java visible in the
table-of-supported-languages and prevent silent drift if the fallback
behavior ever changes.
"""
from __future__ import annotations


class TestLanguageDocTypes:
    def test_scala_has_explicit_entry(self) -> None:
        from docgen.prompts import LANGUAGE_DOC_TYPES

        assert 'scala' in LANGUAGE_DOC_TYPES
        types = set(LANGUAGE_DOC_TYPES['scala'])
        # Scala gets the full Python-equivalent set — implicits, traits, and
        # OO patterns benefit from explanation/architecture/qa/gotcha/diagram.
        assert {'explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'}.issubset(types)

    def test_java_has_explicit_entry(self) -> None:
        from docgen.prompts import LANGUAGE_DOC_TYPES

        assert 'java' in LANGUAGE_DOC_TYPES
        types = set(LANGUAGE_DOC_TYPES['java'])
        assert {'explanation', 'architecture', 'qa', 'catalog', 'gotcha', 'diagram'}.issubset(types)


class TestLanguageFence:
    def test_scala_fence(self) -> None:
        from docgen.prompts import LANGUAGE_FENCE

        assert LANGUAGE_FENCE['scala'] == 'scala'

    def test_java_fence(self) -> None:
        from docgen.prompts import LANGUAGE_FENCE

        assert LANGUAGE_FENCE['java'] == 'java'


class TestLanguageFraming:
    def test_scala_framing(self) -> None:
        """Framing word is what the prompt's prose substitutes for ``Python``
        when describing the source language. Capitalization matters for
        natural-sounding prose ('Scala module', not 'scala module').
        """
        from docgen.prompts import LANGUAGE_FRAMING

        assert LANGUAGE_FRAMING['scala'] == 'Scala'

    def test_java_framing(self) -> None:
        from docgen.prompts import LANGUAGE_FRAMING

        assert LANGUAGE_FRAMING['java'] == 'Java'
