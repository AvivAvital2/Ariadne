"""Exact source ranges are the proof behind SCIP coordinates."""
from __future__ import annotations

import hashlib

from library.source_materialization import materialize_citations
from library.structural_assembly import StructuralCitation


def _citation(file='m.py', line_start=2, line_end=3,
              call_site_file='m.py', call_site_line=1):
    return StructuralCitation(
        qualified_name='pkg.run', file=file, line_start=line_start,
        line_end=line_end, source_name='src1', relation='calls', hop=1,
        call_site_file=call_site_file, call_site_line=call_site_line)


def test_materialization_is_exact_scoped_deduplicated_and_hash_accounted(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    body = 'invoke()\nif ready:\n    persist(value)\nfinish()\n'
    (root / 'm.py').write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()

    result = materialize_citations(
        [_citation(), _citation()], {'src1': root},
        expected_hashes={'m.py': digest})

    assert [(item.kind, item.line_start, item.line_end, item.content)
            for item in result.excerpts] == [('definition', 2, 2, 'if ready:'), ('call_site', 1, 1, 'invoke()')]
    assert all(item.sha256 == digest for item in result.excerpts)
    assert result.gaps == ()
    (root / 'binary.py').write_bytes(b'\xff')
    outside = tmp_path / 'outside.py'
    outside.write_text('outside\n')
    gaps = materialize_citations([
        _citation(file='../outside.py'),
        _citation(file='missing.py'),
        _citation(file='binary.py'),
        _citation(line_start=0, line_end=1),
        StructuralCitation(
            qualified_name='other.run', file='x.py', line_start=1, line_end=1,
            source_name='src2', relation='calls', hop=1,
            call_site_file='x.py', call_site_line=1),
    ], {'src1': root})
    assert any('escapes source root' in gap for gap in gaps.gaps)
    assert any('source unavailable (FileNotFoundError)' in gap for gap in gaps.gaps)
    assert any('source unavailable (UnicodeDecodeError)' in gap for gap in gaps.gaps)
    assert any('invalid range' in gap for gap in gaps.gaps)
    assert any('source root unavailable' in gap for gap in gaps.gaps)

    mismatch = materialize_citations(
        [_citation()], {'src1': root}, expected_hashes={'m.py': 'wrong'})
    assert mismatch.excerpts == ()
    assert mismatch.gaps == ('src1:m.py: provenance hash mismatch',)
def test_materialization_resolves_a_unique_repository_prefix_and_rejects_ambiguity(tmp_path):
    root = tmp_path / "corpus"
    spark = root / "spark" / "sql" / "Flow.scala"
    spark.parent.mkdir(parents=True)
    spark.write_text("object Flow\n")

    unique = materialize_citations(
        [_citation(file="sql/Flow.scala", line_start=1, line_end=1,
                   call_site_file="sql/Flow.scala", call_site_line=1)],
        {"src1": root})

    assert [excerpt.content for excerpt in unique.excerpts] == ["object Flow", "object Flow"]
    delta = root / "delta" / "sql" / "Flow.scala"
    delta.parent.mkdir(parents=True)
    delta.write_text("object OtherFlow\n")
    ambiguous = materialize_citations(
        [_citation(file="sql/Flow.scala", line_start=1, line_end=1)],
        {"src1": root})
    assert ambiguous.excerpts == ()
    assert any("ambiguous repository path" in gap for gap in ambiguous.gaps)
def test_large_definition_extents_materialize_only_the_claim_anchor(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "large.py").write_text("\n".join(f"line {number}" for number in range(1, 101)) + "\n")
    citation = _citation(file="large.py", line_start=10, line_end=90,
                         call_site_file="large.py", call_site_line=5)

    result = materialize_citations([citation], {"src1": root})

    definition = next(item for item in result.excerpts if item.kind == "definition")
    assert (definition.line_start, definition.line_end, definition.content) == (10, 10, "line 10")
def test_compiler_frontier_lines_materialize_without_opening_the_whole_body(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    (root / "m.py").write_text("def commit():\n    prepare()\n    SetTransaction(query_id, batch_id)\n    finish()\n")

    result = materialize_citations(
        [_citation(file="m.py", line_start=1, line_end=4,
                   call_site_file="m.py", call_site_line=2)],
        {"src1": root},
        extra_ranges=(("src1", "m.py", 3, "body_edge"),))

    body = [excerpt for excerpt in result.excerpts if excerpt.kind == "body_edge"]
    assert [(item.line_start, item.line_end, item.content) for item in body] == [
        (3, 3, "    SetTransaction(query_id, batch_id)")]
def test_multiline_claim_witness_range_is_hash_accounted(tmp_path):
    root = tmp_path / "source"
    root.mkdir()
    body = "def run():\n    version = read()\n    if version >= batch:\n        return False\n"
    (root / "m.py").write_text(body)
    digest = hashlib.sha256(body.encode()).hexdigest()

    result = materialize_citations(
        [], {"src1": root},
        extra_ranges=(("src1", "m.py", 2, 4, "claim_witness"),))

    assert result.gaps == ()
    assert [(item.line_start, item.line_end, item.kind, item.content)
            for item in result.excerpts] == [(
                2, 4, "claim_witness",
                "    version = read()\n    if version >= batch:\n        return False")]
    assert result.excerpts[0].sha256 == digest
def test_definition_doc_header_travels_as_its_own_excerpt(tmp_path):
    """The comment block above a definition is part of what a reviewer reads.

    SCIP extents start at the definition line, so the header — annotation
    lines included — materializes as a ``doc_header`` excerpt with exact
    coordinates. A definition with plain code above contributes none, and a
    repeated definition contributes one.
    """
    root = tmp_path / 'source'
    root.mkdir()
    lines = (
        'import base',
        '',
        '/**',
        ' * Delegates all calls to the built-in session catalog directly.',
        ' */',
        '@Evolving',
        'class Delegate {',
        '  int f() { return 1; }',
        '}',
        'plain()',
        'def bare(): pass',
    )
    (root / 'm.py').write_text("\n".join(lines) + "\n")

    result = materialize_citations(
        [_citation(line_start=7, line_end=9),
         _citation(line_start=7, line_end=9),
         _citation(line_start=11, line_end=11)],
        {'src1': root},
        extra_ranges=(('src1', 'm.py', 7, 9, 'definition_body'),))

    headers = [item for item in result.excerpts if item.kind == 'doc_header']
    assert [(item.line_start, item.line_end) for item in headers] == [(3, 6)]
    assert headers[0].content == "\n".join((
        '/**',
        ' * Delegates all calls to the built-in session catalog directly.',
        ' */',
        '@Evolving'))
    assert result.gaps == ()


def test_doc_header_crosses_a_bounded_blank_separation(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    body = (
        '# Explains the writer fork.\n'
        '# Documented above a gap.\n'
        '\n'
        '\n'
        'def run():\n'
        '    pass\n')
    (root / 'm.py').write_text(body)

    result = materialize_citations(
        [_citation(line_start=5, line_end=6,
                   call_site_file='', call_site_line=0)],
        {'src1': root})

    headers = [item for item in result.excerpts
               if item.kind == 'doc_header']
    assert len(headers) == 1
    assert (headers[0].line_start, headers[0].line_end) == (1, 4)
    assert 'Explains the writer fork.' in headers[0].content


def test_doc_header_walk_stops_beyond_the_blank_bound(tmp_path):
    root = tmp_path / 'source'
    root.mkdir()
    body = (
        '# Stray distant note.\n'
        '\n'
        '\n'
        '\n'
        'def run():\n'
        '    pass\n')
    (root / 'm.py').write_text(body)

    result = materialize_citations(
        [_citation(line_start=5, line_end=6,
                   call_site_file='', call_site_line=0)],
        {'src1': root})

    assert not [item for item in result.excerpts
                if item.kind == 'doc_header']
