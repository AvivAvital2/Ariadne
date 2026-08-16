"""Bounded source spans: exact, hash-verified, association-justified.

Reviewed-relevant source outside indexed symbol extents — leading
documentation, annotations, an associated file header — becomes an exact
span bound to the file's hash, and nothing else does: no whole files, no
distant comments, no silent truncation (bounds report gaps instead).
"""
from __future__ import annotations

import hashlib

from library.source_spans import (
    SourceSpan,
    SpanExtraction,
    dedup_spans,
    definition_adjacent_spans,
)

FILE_TEXT = """\
// Copyright example authors.
// Licensed for testing only.

package pkg

/** Preprocesses rules.
  * Runs before the write path.
  */
@Experimental
object Preprocess {
  def apply(): Unit = ()
}
"""


def extract(**overrides):
    parameters = dict(
        source_name="src1",
        file="pkg/preprocess.scala",
        file_text=FILE_TEXT,
        symbol="pkg.Preprocess",
        definition_line_start=10,
    )
    parameters.update(overrides)
    return definition_adjacent_spans(**parameters)


class TestLeadingDocumentation:
    def test_documentation_immediately_above_the_definition_is_retained(self):
        extraction = extract()

        docs = [span for span in extraction.spans
                if span.association_reason == "leading-documentation"]
        assert len(docs) == 1
        span = docs[0]
        assert (span.line_start, span.line_end) == (6, 8)
        assert "Preprocesses rules." in span.text
        assert span.associated_symbol == "pkg.Preprocess"

    def test_annotations_between_documentation_and_definition_are_spanned(
            self):
        extraction = extract()

        annotations = [span for span in extraction.spans
                       if span.association_reason == "annotation"]
        assert len(annotations) == 1
        assert (annotations[0].line_start, annotations[0].line_end) == (9, 9)
        assert annotations[0].text.strip() == "@Experimental"

    def test_distant_comments_are_excluded_by_contiguity(self):
        # The license banner is separated from the definition by the
        # package line; it never rides in on a leading-doc association.
        extraction = extract()

        for span in extraction.spans:
            assert "Copyright" not in span.text

    def test_blank_separation_beyond_the_bound_yields_no_span(self):
        text = "// stray note\n\n\n\ndef apply(): pass\n"
        extraction = definition_adjacent_spans(
            source_name="src1", file="a.py", file_text=text,
            symbol="pkg.apply", definition_line_start=5,
            max_blank_separation=2)

        assert extraction.spans == ()

    def test_oversize_documentation_reports_a_gap_instead_of_truncating(
            self):
        block = "\n".join(f"// line {index}" for index in range(50))
        text = block + "\ndef apply(): pass\n"
        extraction = definition_adjacent_spans(
            source_name="src1", file="a.py", file_text=text,
            symbol="pkg.apply", definition_line_start=51, max_height=40)

        assert extraction.spans == ()
        assert extraction.gaps
        assert extraction.gaps[0]["reason"] == "span-height-exceeded"


class TestFileHeader:
    def test_file_header_requires_explicit_association(self):
        implicit = extract()
        explicit = extract(include_file_header=True)

        assert not [span for span in implicit.spans
                    if span.association_reason == "file-header-documentation"]
        headers = [span for span in explicit.spans
                   if span.association_reason == "file-header-documentation"]
        assert len(headers) == 1
        assert (headers[0].line_start, headers[0].line_end) == (1, 2)


class TestProvenance:
    def test_text_is_byte_exact_and_bound_to_the_file_hash(self):
        extraction = extract()
        expected_hash = hashlib.sha256(FILE_TEXT.encode()).hexdigest()

        lines = FILE_TEXT.splitlines()
        for span in extraction.spans:
            assert span.source_hash == expected_hash
            assert span.text == "\n".join(
                lines[span.line_start - 1:span.line_end])
            assert span.source_name == "src1"

    def test_duplicate_spans_deduplicate_and_extraction_is_idempotent(self):
        first = extract()
        second = extract()

        assert first == second
        assert dedup_spans((*first.spans, *second.spans)) == first.spans
