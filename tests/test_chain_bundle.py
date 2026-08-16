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
from library.chain_answer import render_spine
from library.chain_bundle import curate_bundle
from library.structural_assembly import StructuralCitation

SOURCE = 'src1'


def _citation(qn, *, hop, line, stop_reason, file='pkg/alpha.py', line_end=None):
    return StructuralCitation(
        qualified_name=qn, file=file, line_start=line, source_name=SOURCE,
        relation='calls', hop=hop, call_site_file='pkg/alpha.py',
        call_site_line=line + 100, stop_reason=stop_reason,
        # a real extent by default; stage two quotes a hop from it
        line_end=line + 6 if line_end is None else line_end,
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
def test_which_hops_carry_a_document_follows_the_stop_reason_the_walk_recorded(library):
    """``descended`` and ``leaf`` earn their document; ``plumbing`` is cited and left alone.

    The traversal already recorded why it stopped at each hop, so curation reads that
    instead of inventing a relevance score. An earlier version inferred it from the
    trace's shape and starved exactly the wrong hops: a destination like
    ``writeAllChanges`` is a ``leaf``, and the leaf is usually where the work happens.
    """
    bundle = curate_bundle(library, CHAIN, source=SOURCE)
    evidence = {h.citation.qualified_name: h.evidence for h in bundle.hops}

    assert evidence['pkg.alpha.Alpha.run'] == 'Runs the alpha operation end to end.'
    assert evidence['pkg.beta.Beta.start'] == 'Starts beta processing.'
    assert evidence['pkg.aaa_util.Util.common'] is None, (
        'plumbing is named, not explained')


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
def test_every_explained_hop_is_attached_and_the_renderer_bounds_the_prompt(library):
    """Curation has no size budget: ``render_spine`` is the only thing that bounds output.

    Measured across evidence caps of 6k, 12k, 20k, 60k and 200k chars, the rendered spine
    stayed at ~20,000 chars every time — the cap changed nothing about the prompt, only
    which hops were explained, decided before the renderer knew what it would keep. Two
    constants were guessing at one constraint and the one that binds was not the one being
    tuned.

    So every chain-material hop is attached, and the omission is reported once, by the
    stage that performs it: ``render_spine`` cuts from the tail, preserving execution
    order, and says how many hops it dropped.
    """
    bundle = curate_bundle(library, CHAIN, source=SOURCE)
    explained = [h for h in bundle.hops if h.evidence]

    assert len(explained) == 2, "both chain-material hops carry their document"
    assert all(h.citation.line_start for h in bundle.hops), 'coordinates survive'


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
def test_the_hop_carries_its_document_because_that_is_what_the_model_reads(library):
    """The document is the payload; the coordinates are the traceability. Not the reverse.

    This reverses what an earlier version of this file asserted — that a hop is quoted from
    source and a generated document must not travel. That confused two different things.
    Distrust of prose is about *authority for a claim*: a docstring or a human-authored
    guide can contradict the code, and search-retrieved prose concatenated as background is
    worse. A per-hop ``catalog`` document is neither — it is fetched by deterministic id
    from the symbol the walk actually reached, and its description is derived from that
    code.

    The division of labour is the point. A generated description can be wrong; a SCIP
    coordinate cannot. So the description is what the model reads, and ``file:line`` is what
    makes every claim checkable afterwards — which is why source text does not travel here
    and coordinates always do.
    """
    bundle = curate_bundle(library, CHAIN, source=SOURCE)
    beta = next(h for h in bundle.hops
                if h.citation.qualified_name == 'pkg.beta.Beta.start')

    assert beta.evidence == 'Starts beta processing.'
    assert beta.title == 'Beta.start'
    assert (beta.citation.file, beta.citation.line_start) == ('pkg/beta.py', 3)
def test_exact_source_ranges_travel_with_the_hop_into_formulation(library, tmp_path):
    root = tmp_path / 'source'
    (root / 'pkg').mkdir(parents=True)
    (root / 'pkg' / 'step.py').write_text(
        'call_step()\nif ready:\n    persist(value)\nfinish()\n')
    citation = _citation(
        'pkg.step.run', hop=1, line=2, line_end=3,
        file='pkg/step.py', stop_reason='leaf')
    citation = StructuralCitation(
        **{**citation.__dict__, 'call_site_file': 'pkg/step.py',
           'call_site_line': 1})

    bundle = curate_bundle(
        library, [citation], source=SOURCE, source_root=root)

    excerpts = bundle.hops[0].source_excerpts
    assert [(excerpt.kind, excerpt.content) for excerpt in excerpts] == [('definition', 'if ready:'), ('call_site', 'call_step()')]
    spine = render_spine(bundle)
    assert 'Source definition [pkg/step.py:2-2]' in spine
    assert 'if ready:' in spine
    assert 'Source call_site [pkg/step.py:1-1]' in spine
    assert bundle.source_gaps == ()
def test_deferred_bundle_keeps_document_ids_without_fetching_document_bodies(library):
    from library.chain_bundle import curate_bundle
    citation = _citation(
        "pkg.alpha.run", file="alpha.py", line=1, hop=1, stop_reason="leaf")

    bundle = curate_bundle(
        library, [citation], source=SOURCE, source_root=None,
        materialize_source=False, fetch_documents=False)

    assert bundle.hops[0].document_id is not None
    assert bundle.hops[0].evidence is None
    assert bundle.themes == []
def test_selected_definition_bodies_are_an_explicit_post_selection_payload(
        library, tmp_path):
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "step.py").write_text(
        "call_step()\nif ready:\n    persist(value)\nfinish()\n")
    citation = _citation(
        "pkg.step.run", hop=1, line=2, line_end=3,
        file="pkg/step.py", stop_reason="leaf")
    citation = StructuralCitation(
        **{**citation.__dict__, "call_site_file": "pkg/step.py",
           "call_site_line": 1})

    bundle = curate_bundle(
        library, [citation], source=SOURCE, source_root=root,
        materialize_definition_bodies=True)

    assert [(excerpt.kind, excerpt.line_start, excerpt.line_end, excerpt.content)
            for excerpt in bundle.hops[0].source_excerpts] == [
        ("definition", 2, 2, "if ready:"),
        ("call_site", 1, 1, "call_step()"),
        ("definition_body", 2, 3, "if ready:\n    persist(value)"),
    ]
def test_definition_body_materialization_is_scoped_to_chosen_symbols(
        library, tmp_path):
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "flow.py").write_text(
        "call()\nfirst body\nfirst end\nsecond body\nsecond end\n")
    first = _citation(
        "pkg.First.run", hop=1, line=2, line_end=3,
        file="pkg/flow.py", stop_reason="leaf")
    second = _citation(
        "pkg.Second.run", hop=1, line=4, line_end=5,
        file="pkg/flow.py", stop_reason="leaf")
    first = StructuralCitation(**{
        **first.__dict__, "call_site_file": "pkg/flow.py", "call_site_line": 1})
    second = StructuralCitation(**{
        **second.__dict__, "call_site_file": "pkg/flow.py", "call_site_line": 1})

    bundle = curate_bundle(
        library, [first, second], source=SOURCE, source_root=root,
        materialize_definition_bodies=True,
        definition_body_symbols=("pkg.Second.run",))

    bodies = {
        hop.citation.qualified_name: [
            excerpt.content for excerpt in hop.source_excerpts
            if excerpt.kind == "definition_body"]
        for hop in bundle.hops}
    assert bodies == {
        "pkg.First.run": [],
        "pkg.Second.run": ["second body\nsecond end"],
    }
def test_definition_body_slicing_keeps_semantic_and_compiler_edge_neighborhoods():
    from library.chain_bundle import slice_definition_body_excerpts
    from library.source_materialization import SourceExcerpt

    body = SourceExcerpt(
        source_name="repo", file="flow.py", line_start=10, line_end=19,
        kind="definition_body",
        content="\n".join((
            "def execute():",
            "    initialize()",
            "    joined = left.join(right)",
            "    decide_actions(joined)",
            "    audit_unrelated_state()",
            "    emit_rows(joined)",
            "    finalize_unrelated_state()",
            "    persist_result()",
            "    cleanup_unrelated_state()",
            "    return result",
        )),
        sha256="snapshot")

    sliced = slice_definition_body_excerpts(
        (body,), "When does the join happen and how are rows emitted?",
        evidence_lines={("flow.py", 17)}, context_lines=1,
        short_body_lines=0)

    assert [(item.kind, item.line_start, item.line_end) for item in sliced] == [
        ("definition_slice", 10, 17)]
    assert "joined = left.join(right)" in sliced[0].content
    assert "emit_rows(joined)" in sliced[0].content
    assert "persist_result()" in sliced[0].content
    assert "cleanup_unrelated_state()" not in sliced[0].content
def test_definition_body_slicing_uses_discriminating_code_tokens():
    from library.chain_bundle import slice_definition_body_excerpts
    from library.source_materialization import SourceExcerpt

    body = SourceExcerpt(
        source_name="repo", file="flow.py", line_start=1, line_end=7,
        kind="definition_body",
        content="\n".join((
            "def execute():",
            "    joined = left.join(right)",
            "    use(joined)",
            "    log(joined)",
            "    audit(joined)",
            "    cleanup(joined)",
            "    return joined",
        )),
        sha256="snapshot")

    sliced = slice_definition_body_excerpts(
        (body,), "Where does the join execute?", evidence_lines=set(),
        context_lines=1, short_body_lines=0)

    assert [(item.line_start, item.line_end) for item in sliced] == [(1, 3)]
    assert "log(joined)" not in sliced[0].content
def test_definition_body_slicing_preserves_causal_span_between_relevant_anchors():
    from library.chain_bundle import slice_definition_body_excerpts
    from library.source_materialization import SourceExcerpt

    lines = ["def execute():"] + [
        f"    unrelated_{index}()" for index in range(1, 60)]
    lines[9] = "    joined = left.join(right)"
    lines[29] = "    local_filter_without_query_vocabulary()"
    lines[49] = "    emit_rows(joined)"
    body = SourceExcerpt(
        source_name="repo", file="flow.py", line_start=1, line_end=60,
        kind="definition_body", content="\n".join(lines), sha256="snapshot")

    sliced = slice_definition_body_excerpts(
        (body,), "When does the join happen and how are rows emitted?",
        evidence_lines=set(), context_lines=1)

    assert [(item.line_start, item.line_end) for item in sliced] == [
        (1, 1), (9, 51)]
    assert "local_filter_without_query_vocabulary()" in sliced[1].content
    assert "unrelated_59()" not in sliced[1].content


def test_definition_body_slicing_keeps_short_selected_bodies_complete():
    from library.chain_bundle import slice_definition_body_excerpts
    from library.source_materialization import SourceExcerpt

    body = SourceExcerpt(
        source_name="repo", file="flow.py", line_start=20, line_end=24,
        kind="definition_body",
        content="def execute():\n    prepare()\n    classify()\n"
        "    filter_local()\n    return rows",
        sha256="snapshot")

    assert slice_definition_body_excerpts(
        (body,), "How are rows emitted?", evidence_lines=set()) == (body,)


def test_definition_body_slicing_preserves_full_body_when_scope_has_no_signal():
    from library.chain_bundle import slice_definition_body_excerpts
    from library.source_materialization import SourceExcerpt

    body = SourceExcerpt(
        source_name="repo", file="opaque.py", line_start=4, line_end=6,
        kind="definition_body", content="def f():\n    alpha()\n    omega()",
        sha256="snapshot")

    assert slice_definition_body_excerpts(
        (body,), "unrelated vocabulary", evidence_lines=set()) == (body,)
def test_curate_bundle_defers_semantic_slicing_to_story_chunks(library, tmp_path):
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    lines = ["call()", "def run():"] + [
        f"    unrelated_{index}()" for index in range(3, 61)]
    lines[30] = "    persist_result()"
    (root / "pkg" / "flow.py").write_text("\n".join(lines) + "\n")
    citation = StructuralCitation(
        qualified_name="pkg.Flow.run", file="pkg/flow.py",
        line_start=2, line_end=60, source_name=SOURCE,
        relation="localized", hop=1, call_site_file="pkg/flow.py",
        call_site_line=1, stop_reason="leaf")

    bundle = curate_bundle(
        library, [citation], source=SOURCE, source_root=root,
        materialize_definition_bodies=True,
        definition_body_symbols=("pkg.Flow.run",),
        definition_body_query="How is the result persisted?")

    bodies = [excerpt for excerpt in bundle.hops[0].source_excerpts
              if excerpt.kind == "definition_body"]
    assert [(item.line_start, item.line_end) for item in bodies] == [(2, 60)]
    assert "persist_result()" in bodies[0].content
def test_every_indexed_extent_of_a_selected_body_is_materialized(library, tmp_path):
    """Same-name overloads share a qualified name; the walk reaches one of them.

    The delegating stub's three lines must not stand in for the implementation
    overload where the causal statements live — selecting the body materializes
    every extent the index records for that name.
    """
    lines = ["# flow module"]
    lines += ["def write(data):", "    return write(data, [])", ""]      # 2-4
    lines += ["x = %d" % index for index in range(5, 12)]                # 5-11
    lines += ["def write(data, options):",                               # 12
              "    prepared = prepare(data)",
              "    sink.push(prepared, options)",
              "    return prepared"]                                     # 13-15
    source_file = tmp_path / "pkg" / "flow.py"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("\n".join(lines) + "\n")
    with library._conn_provider.acquire() as conn:
        for suffix, (line_start, line_end) in (
                ("().", (2, 3)), ("(+1).", (12, 15))):
            conn.execute(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, '
                'file, line_start, line_end, kind, display_name, qualified_name, '
                'parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (f'scip-python python {SOURCE} 0.1 `pkg.flow`/write{suffix}',
                 SOURCE, 'python', 'pkg/flow.py', line_start, line_end,
                 'Method', 'write', 'pkg.flow.write', 'pkg.flow'))
        conn.commit()
    chain = [_citation('pkg.flow.write', hop=1, line=2, file='pkg/flow.py',
                       line_end=3, stop_reason='leaf')]

    bundle = curate_bundle(
        library, chain, source=SOURCE, source_root=str(tmp_path),
        materialize_source=True, materialize_definition_bodies=True,
        definition_body_symbols=('pkg.flow.write',))

    bodies = {(excerpt.line_start, excerpt.line_end)
              for hop in bundle.hops for excerpt in hop.source_excerpts
              if excerpt.kind == 'definition_body'}
    assert (2, 3) in bodies
    assert (12, 15) in bodies
    rendered = "\n".join(
        excerpt.content for hop in bundle.hops
        for excerpt in hop.source_excerpts
        if excerpt.kind == 'definition_body')
    assert "sink.push(prepared, options)" in rendered
def test_doc_header_above_a_cited_definition_attaches_to_its_hop(library, tmp_path):
    """The header comment materializes — and must reach the hop that owns it.

    An excerpt no hop carries is proof no prompt ever sees.
    """
    lines = (
        "import base",
        "",
        "/**",
        " * Delegates all calls to the session catalog directly.",
        " */",
        "class Delegate {",
        "  int f() { return 1; }",
        "}",
    )
    source_file = tmp_path / "pkg" / "delegate.java"
    source_file.parent.mkdir(parents=True)
    source_file.write_text("\n".join(lines) + "\n")
    chain = [_citation('pkg.Delegate', hop=1, line=6, file='pkg/delegate.java',
                       line_end=8, stop_reason='leaf')]

    bundle = curate_bundle(
        library, chain, source=SOURCE, source_root=str(tmp_path),
        materialize_source=True)

    headers = [excerpt for hop in bundle.hops
               for excerpt in hop.source_excerpts
               if excerpt.kind == 'doc_header']
    assert [(item.line_start, item.line_end) for item in headers] == [(3, 5)]
    assert "session catalog directly" in headers[0].content
def test_selected_body_materializes_all_compiler_edge_sites_without_target_bodies(
        library, tmp_path):
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    (root / "other").mkdir(parents=True)
    (root / "pkg" / "flow.py").write_text("\n".join((
        "header", "def run():", "    first()", "    second()",
        "    third()", "    fourth()", "    fifth()", "    sixth()",
        "    return done", "")))
    (root / "other" / "flow.py").write_text("foreign_call()\n")
    caller = "scip-python python src1 0.1 `pkg`/Flow#run()."
    with library._conn_provider.acquire() as conn:
        conn.execute(
            "INSERT INTO scip_symbols (canonical_id, source_name, language, file, "
            "line_start, line_end, kind, display_name, qualified_name, "
            "parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (caller, SOURCE, "python", "pkg/flow.py", 2, 9, "Method",
             "run", "pkg.Flow.run", "pkg.Flow"))
        for index, line in enumerate(range(3, 9)):
            conn.execute(
                "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, "
                "edge_type, file, line, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                (caller, f"target-{index}", "call", "pkg/flow.py", line, "exact"))
        conn.execute(
            "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, "
            "edge_type, file, line, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            (caller, "foreign-target", "call", "other/flow.py", 1, "exact"))
        conn.commit()
    citation = _citation(
        "pkg.Flow.run", hop=1, line=2, line_end=9,
        file="pkg/flow.py", stop_reason="selected_route")

    bundle = curate_bundle(
        library, [citation], source=SOURCE, source_root=root,
        materialize_definition_bodies=True,
        definition_body_symbols=("pkg.Flow.run",))

    body_edges = sorted(
        (excerpt.file, excerpt.line_start, excerpt.content)
        for excerpt in bundle.hops[0].source_excerpts
        if excerpt.kind == "body_edge")
    assert body_edges == [
        ("other/flow.py", 1, "foreign_call()"),
        ("pkg/flow.py", 3, "    first()"),
        ("pkg/flow.py", 4, "    second()"),
        ("pkg/flow.py", 5, "    third()"),
        ("pkg/flow.py", 6, "    fourth()"),
        ("pkg/flow.py", 7, "    fifth()"),
        ("pkg/flow.py", 8, "    sixth()")]
def test_indexed_symbols_covered_by_selected_source_are_reported(library):
    from library.chain_bundle import BundleHop, indexed_symbols_covered_by_source
    from library.source_materialization import SourceExcerpt

    with library._conn_provider.acquire() as conn:
        rows = (
            ("owner", 1, 12, "pkg.Owner"),
            ("member", 4, 6, "pkg.Owner.decide"),
            ("partial", 7, 10, "pkg.Owner.partial"),
            ("outside", 20, 22, "pkg.Other.run"),
            ("local hidden", 4, 5, "local.hidden"),
        )
        for canonical, line_start, line_end, qualified_name in rows:
            conn.execute(
                "INSERT INTO scip_symbols (canonical_id, source_name, language, "
                "file, line_start, line_end, kind, display_name, qualified_name, "
                "parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (canonical, SOURCE, "python", "pkg/flow.py", line_start,
                 line_end, "Method", qualified_name.rsplit(".", 1)[-1],
                 qualified_name, "pkg.Owner"))
        conn.commit()
    excerpt = SourceExcerpt(
        source_name=SOURCE, file="pkg/flow.py", line_start=3, line_end=8,
        kind="definition_slice", content="selected source", sha256="proof")
    hop = BundleHop(
        citation=_citation("pkg.Owner", hop=0, line=1, line_end=12,
                           file="pkg/flow.py", stop_reason="selected_route"),
        source_excerpts=(excerpt,))

    assert indexed_symbols_covered_by_source(library, (hop,), source=SOURCE) == ("pkg.Owner.decide", "pkg.Owner.partial")
def test_compiler_edge_endpoints_in_selected_exact_source_are_reported(library):
    from library.chain_bundle import BundleHop, indexed_symbols_covered_by_source
    from library.source_materialization import SourceExcerpt

    with library._conn_provider.acquire() as conn:
        symbols = (
            ("caller", "pkg/flow.py", 1, 12, "pkg.Owner"),
            ("callee", "pkg/other.py", 30, 35, "pkg.External.run"),
            ("local hidden", "pkg/flow.py", 5, 5, "local.hidden"),
        )
        for canonical, file, line_start, line_end, qualified_name in symbols:
            conn.execute(
                "INSERT INTO scip_symbols (canonical_id, source_name, language, "
                "file, line_start, line_end, kind, display_name, qualified_name, "
                "parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (canonical, SOURCE, "python", file, line_start, line_end,
                 "Method", qualified_name.rsplit(".", 1)[-1], qualified_name,
                 "pkg.Owner"))
        for callee, line in (("callee", 5), ("local hidden", 6)):
            conn.execute(
                "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, "
                "edge_type, file, line, confidence) VALUES (?, ?, ?, ?, ?, ?)",
                ("caller", callee, "call", "pkg/flow.py", line, "exact"))
        conn.commit()
    excerpt = SourceExcerpt(
        source_name=SOURCE, file="pkg/flow.py", line_start=5, line_end=6,
        kind="body_edge", content="external()\nhidden()", sha256="proof")
    hop = BundleHop(
        citation=_citation("pkg.Owner", hop=0, line=1, line_end=12,
                           file="pkg/flow.py", stop_reason="selected_route"),
        source_excerpts=(excerpt,))

    assert indexed_symbols_covered_by_source(
        library, (hop,), source=SOURCE) == (
            "pkg.External.run", "pkg.Owner")
def test_curate_bundle_preserves_full_selected_body_for_downstream_dataflow(
        library, tmp_path):
    root = tmp_path / "source"
    (root / "pkg").mkdir(parents=True)
    lines = ["def run():"] + [f"    unrelated_{index}()" for index in range(1, 60)]
    lines[37] = "    current = transaction.version(key)"
    lines[38] = "    if current >= requested:"
    lines[39] = "        audit_skip()"
    lines[40] = "        return false"
    (root / "pkg" / "flow.py").write_text("\n".join(lines) + "\n")
    citation = StructuralCitation(
        qualified_name="pkg.Flow.run", file="pkg/flow.py",
        line_start=1, line_end=60, source_name=SOURCE,
        relation="localized", hop=0, call_site_file="pkg/flow.py",
        call_site_line=38, stop_reason="selected_route")

    bundle = curate_bundle(
        library, [citation], source=SOURCE, source_root=root,
        materialize_definition_bodies=True,
        definition_body_symbols=("pkg.Flow.run",),
        definition_body_query="How is a repeated request skipped?")

    bodies = [excerpt for hop in bundle.hops for excerpt in hop.source_excerpts
              if excerpt.kind == "definition_body"]
    assert [(body.line_start, body.line_end) for body in bodies] == [(1, 60)]
    assert "return false" in bodies[0].content
def test_contains_site_does_not_prove_owner_endpoint_from_member_line(library):
    from library.chain_bundle import BundleHop, indexed_symbols_covered_by_source
    from library.source_materialization import SourceExcerpt

    with library._conn_provider.acquire() as conn:
        for canonical, qualified_name, parent, start, end, kind in (
                ("owner-contained", "pkg.Owner", "pkg", 1, 20, "Class"),
                ("member-contained", "pkg.Owner.member", "pkg.Owner", 8, 12, "Method")):
            conn.execute(
                "INSERT INTO scip_symbols (canonical_id, source_name, language, "
                "file, line_start, line_end, kind, display_name, qualified_name, "
                "parent_qualified_name) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (canonical, SOURCE, "python", "pkg/owner.py", start, end, kind,
                 qualified_name.rsplit(".", 1)[-1], qualified_name, parent))
        conn.execute(
            "INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, "
            "edge_type, file, line, confidence) VALUES (?, ?, ?, ?, ?, ?)",
            ("owner-contained", "member-contained", "contains",
             "pkg/owner.py", 8, "exact"))
        conn.commit()
    excerpt = SourceExcerpt(
        source_name=SOURCE, file="pkg/owner.py", line_start=8, line_end=8,
        kind="definition", content="def member():", sha256="proof")
    hop = BundleHop(
        citation=_citation("pkg.Owner.member", hop=1, line=8, line_end=12,
                           file="pkg/owner.py", stop_reason="selected_owner_member"),
        source_excerpts=(excerpt,))

    covered = indexed_symbols_covered_by_source(
        library, (hop,), source=SOURCE)

    assert "pkg.Owner" not in covered
    assert "pkg.Owner.member" not in covered
