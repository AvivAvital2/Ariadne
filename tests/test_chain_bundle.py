"""Curating the bundle: the chain leads, documents follow.

The north star is ``index -> fetch document -> curate bundle -> formulate -> respond``.
Traversal (``library/structural_assembly.py``) produces the chain; this step attaches the
prose. Documents are fetched **per hop, by deterministic id** — ``_element_doc_id`` is a
pure function of ``(source, qualified_name)``, so no search and no embedding is involved.
The chain decides which documents are read, which is the inversion the north star asks for.

One budget is legitimate here and was not legitimate in traversal: an LLM context window is
a real constraint, whereas capping a graph walk by count truncated the chain itself. So
every hop always contributes its coordinates, and only *prose* is rationed.

Prose follows each citation's ``stop_reason`` — the traversal's own record of why it stopped
there. Inferring it from the trace's shape instead starved the wrong hops: a destination like
``writeAllChanges`` is a ``leaf``, and the leaf is usually where the work happens.

Synthetic fixtures only: source ``src1``.
"""
from __future__ import annotations

import pytest

from docgen.catalog_writer import _element_doc_id
from library import Library
from library.chain_bundle import curate_bundle
from library.structural_assembly import StructuralCitation

SOURCE = 'src1'


def _citation(qn, *, hop, line, stop_reason, file='pkg/alpha.py'):
    return StructuralCitation(
        qualified_name=qn, file=file, line_start=line, source_name=SOURCE,
        relation='calls', hop=hop, call_site_file='pkg/alpha.py',
        call_site_line=line + 100, stop_reason=stop_reason,
    )


# The walk descended into Alpha.run, found Beta.start to be a leaf, and stopped at
# Util.common because its fan-in sat at the descent boundary.
CHAIN = [
    _citation('pkg.alpha.Alpha.run', hop=1, line=5, stop_reason='descended'),
    _citation('pkg.beta.Beta.start', hop=2, line=3, file='pkg/beta.py',
              stop_reason='leaf'),
    _citation('pkg.aaa_util.Util.common', hop=1, line=2, file='pkg/aaa_util.py',
              stop_reason='plumbing'),
]


@pytest.fixture
def library(tmp_path):
    with Library(tmp_path / 'library.db') as lib:
        for qn, title, body in (
            ('pkg.alpha.Alpha.run', 'Alpha.run', 'Runs the alpha operation end to end.'),
            ('pkg.beta.Beta.start', 'Beta.start', 'Starts beta processing.'),
            # Util.common deliberately HAS a document, to prove rationing is by the
            # trace's structure and not by whether prose happens to exist.
            ('pkg.aaa_util.Util.common', 'Util.common', 'A shared helper.'),
        ):
            lib.add_document(
                content_type='catalog', title=title, content=body,
                source_files=[f'{qn.rsplit(".", 1)[0].replace(".", "/")}.py'],
                doc_id=_element_doc_id(SOURCE, qn), source_name=SOURCE,
            )
        yield lib


def test_every_hop_carries_coordinates_and_its_document(library):
    bundle = curate_bundle(library, CHAIN, source=SOURCE)

    assert [h.citation.qualified_name for h in bundle.hops] == [
        'pkg.alpha.Alpha.run', 'pkg.beta.Beta.start', 'pkg.aaa_util.Util.common',
    ]
    assert [h.title for h in bundle.hops] == ['Alpha.run', 'Beta.start', 'Util.common']
    assert bundle.documents_found == 3
    # order is the trace's order, never a relevance order
    assert [h.citation.hop for h in bundle.hops] == [1, 2, 1]


def test_prose_follows_the_stop_reason_the_walk_recorded(library):
    """`descended` and `leaf` earn prose; `plumbing` is cited and left alone."""
    bundle = curate_bundle(library, CHAIN, source=SOURCE)
    evidence = {h.citation.qualified_name: h.evidence for h in bundle.hops}

    assert evidence['pkg.alpha.Alpha.run'] == 'Runs the alpha operation end to end.'
    assert evidence['pkg.beta.Beta.start'] == 'Starts beta processing.'
    assert evidence['pkg.aaa_util.Util.common'] is None


def test_a_hop_with_no_document_still_carries_its_coordinates(library):
    """A symbol SCIP knows but the catalog never documented is still a real hop."""
    chain = [*CHAIN, _citation('pkg.ghost.Ghost.method', hop=2, line=9,
                               file='pkg/ghost.py', stop_reason='leaf')]

    bundle = curate_bundle(library, chain, source=SOURCE)
    ghost = bundle.hops[-1]

    assert ghost.document_id is None
    assert ghost.title is None
    assert (ghost.citation.file, ghost.citation.line_start) == ('pkg/ghost.py', 9)
    assert bundle.documents_found == 3


def test_the_context_budget_rations_prose_and_reports_what_it_dropped(library):
    bundle = curate_bundle(library, CHAIN, source=SOURCE, max_evidence_chars=30)

    kept = [h for h in bundle.hops if h.evidence]
    assert len(kept) == 1
    assert bundle.evidence_omitted == 1
    # coordinates survive rationing — they are what makes an answer checkable
    assert len(bundle.hops) == 3
    assert all(h.citation.line_start for h in bundle.hops)


def test_an_empty_chain_yields_an_empty_bundle_not_an_error(library):
    bundle = curate_bundle(library, [], source=SOURCE)

    assert bundle.hops == []
    assert bundle.documents_found == 0


def _join_theme(library, cluster_id, *, doc_id, qns, member_count, coherent=1,
                title='A theme'):
    """Register a theme with a summary document and the members that belong to it."""
    library.add_document(content_type='theme', title=title,
                         content='What this cluster is about.', doc_id=doc_id,
                         source_name=SOURCE)
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT INTO themes (cluster_id, doc_id, member_count, resolution, '
            'last_built_at, last_summarized_at, summary_hash, coherent) '
            "VALUES (?,?,?,1.0,'2026-01-01','2026-01-01','h',?)",
            (cluster_id, doc_id, member_count, coherent))
        for qn in qns:
            conn.execute(
                'INSERT INTO theme_members (cluster_id, element_id, weight, joined_at) '
                "VALUES (?,?,1.0,'2026-01-01')",
                (cluster_id, _element_doc_id(SOURCE, qn)))
        conn.commit()


def test_themes_map_the_chain_and_are_ordered_by_how_much_of_it_they_cover(library):
    """Per hop a 1,292-member theme says little; across the chain it is a map."""
    _join_theme(library, 'c-broad', doc_id='doc-broad', member_count=5520, coherent=0,
                title='Transaction Log Engine',
                qns=['pkg.alpha.Alpha.run', 'pkg.beta.Beta.start',
                     'pkg.aaa_util.Util.common'])
    _join_theme(library, 'c-narrow', doc_id='doc-narrow', member_count=839,
                title='MERGE INTO Row-Level DML Pipeline',
                qns=['pkg.beta.Beta.start'])

    bundle = curate_bundle(library, CHAIN, source=SOURCE)

    assert [(t.title, t.hops, t.member_count, t.coherent) for t in bundle.themes] == [
        ('Transaction Log Engine', 3, 5520, False),
        ('MERGE INTO Row-Level DML Pipeline', 1, 839, True),
    ]


def test_theme_breadth_is_reported_rather_than_silently_filtered(library):
    """A broad, incoherent cluster still names where the chain lives — flag, don't drop."""
    _join_theme(library, 'c-broad', doc_id='doc-broad', member_count=5520, coherent=0,
                title='Transaction Log Engine', qns=['pkg.alpha.Alpha.run'])

    bundle = curate_bundle(library, CHAIN, source=SOURCE)

    assert len(bundle.themes) == 1
    assert bundle.themes[0].coherent is False
    assert bundle.themes[0].member_count == 5520


def test_a_chain_touching_no_theme_reports_none(library):
    bundle = curate_bundle(library, CHAIN, source=SOURCE)

    assert bundle.themes == []
def _add_scip_symbol(library, qualified_name, *, file, line):
    """A scip_symbols row: the coordinates SCIP knows for a definition, and nothing else.

    There is deliberately no prose column to populate. SCIP parses the author's docstring
    and Ariadne drops it at ingest, because a docstring is not code.
    """
    with library._conn_provider.acquire() as conn:
        conn.execute(
            'INSERT OR REPLACE INTO scip_symbols (canonical_id, source_name, language, '
            'file, line_start, line_end, kind, display_name, qualified_name, '
            'parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (f'scip-python python {SOURCE} 0.1 `{qualified_name}`.', SOURCE, 'python',
             file, line, line + 8, 'Method', qualified_name.rsplit('.', 1)[-1],
             qualified_name, qualified_name.rsplit('.', 1)[0]))
        conn.commit()
def test_a_docstring_is_never_used_as_evidence(library):
    """A hop SCIP knows but the catalog never documented contributes coordinates only.

    A docstring is author prose, not code: it can be stale, aspirational, or contradict
    the function it sits above, and stage five verifies coordinates rather than claims — so
    prose nothing checks must not travel beside prose something does. Ariadne's trust model
    is code first, official documentation second for a spool only, everything else suspect,
    and a docstring is "everything else" whether the source is a spool or not.

    Enforced by absence rather than by a filter: ``scip_symbols`` has no column for it, so
    there is nothing here to leak.
    """
    _add_scip_symbol(library, 'pkg.ghost.Ghost.method', file='pkg/ghost.py', line=9)
    chain = [_citation('pkg.ghost.Ghost.method', hop=2, line=9,
                       file='pkg/ghost.py', stop_reason='leaf')]

    bundle = curate_bundle(library, chain, source=SOURCE)
    ghost = bundle.hops[0]

    assert ghost.evidence is None, 'no prose without a generated document'
    assert ghost.document_id is None
    assert (ghost.citation.file, ghost.citation.line_start) == ('pkg/ghost.py', 9), (
        'coordinates survive — they are the checkable part'
    )
