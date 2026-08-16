"""Caller resolution must still work when the indexer omits ``enclosing_range``.

``a11fbf7`` fixed the Python call graph by resolving a reference's caller to the
tightest definition whose ``enclosing_range`` (body extent) contains it. That is
correct for scip-python, which populates the field — and it silently broke Scala
and Java, because **scip-java emits no enclosing_range at all**.

Measured on the databricks spool: every one of the 37 definitions in
``MergeIntoCommand.scala`` has an empty ``enclosing_range``, so no definition
ever contains an in-body call, every caller resolves to None, and the edge is
dropped. ``ClassicMergeExecutor#writeAllChanges()`` has two real call sites in
that file (lines 138 and 155, both present in the raw index as references) and
zero edges in the graph. The file holds 4 edges in total; comparable Spark files
hold ~1,800.

So when a document supplies no enclosing ranges, the body extents are
synthesised positionally: definitions sorted by line, each running to the line
before the next. Approximate, but it needs no source file — which matters for an
installed spool whose corpus may not be on disk — and it restores the property
the whole mechanism depends on: a call inside a method attributes to that method.
"""
from __future__ import annotations


def _graph_with(doc, source='databricks', language='scala'):
    from docgen.scip_cross_source import CrossSourceGraph
    from docgen.scip_extractor import ScipIndex
    g = CrossSourceGraph()
    g.add_source(source, index=ScipIndex(documents=(doc,)), language=language)
    g.materialize()
    return g


def test_call_resolves_when_indexer_omits_enclosing_range() -> None:
    """The MergeIntoCommand -> writeAllChanges case, in miniature."""
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    caller = 'semanticdb maven . . p/MergeIntoCommand#run().'
    callee = 'semanticdb maven . . p/ClassicMergeExecutor#writeAllChanges().'
    other = 'semanticdb maven . . p/MergeIntoCommand#other().'

    doc = _ScipDoc(
        relative_path='MergeIntoCommand.scala',
        occurrences=(
            # scip-java style: identifier range only, NO enclosing_range
            _ScipOccurrence(symbol=caller, range=(10, 6, 10, 9),
                            is_definition=True, enclosing_range=()),
            _ScipOccurrence(symbol=other, range=(40, 6, 40, 11),
                            is_definition=True, enclosing_range=()),
            # the call, in run()'s body — between the two definitions
            _ScipOccurrence(symbol=callee, range=(20, 18, 20, 33),
                            is_definition=False),
            _ScipOccurrence(symbol=callee, range=(0, 0, 0, 5),
                            is_definition=True, enclosing_range=()),
        ),
        symbols=(
            _ScipSymbol(symbol=caller, kind='Method', display_name='run'),
            _ScipSymbol(symbol=other, kind='Method', display_name='other'),
            _ScipSymbol(symbol=callee, kind='Method', display_name='writeAllChanges'),
        ),
    )
    g = _graph_with(doc)
    edges = [e for e in g._edges
             if e.callee.canonical_id == callee and e.caller.canonical_id == caller]
    assert edges, (
        'a call inside run() must attribute to run() even though the index '
        'supplies no enclosing_range'
    )


def test_real_enclosing_range_still_wins() -> None:
    """scip-python supplies true body extents; those must not be overridden."""
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    caller = 'scip-python python proj 0.1 `svc`/caller().'
    callee = 'scip-python python proj 0.1 `svc`/callee().'
    doc = _ScipDoc(
        relative_path='svc.py',
        occurrences=(
            _ScipOccurrence(symbol=callee, range=(0, 4, 0, 10),
                            is_definition=True, enclosing_range=(0, 0, 1, 8)),
            _ScipOccurrence(symbol=caller, range=(3, 4, 3, 10),
                            is_definition=True, enclosing_range=(3, 0, 7, 12)),
            _ScipOccurrence(symbol=callee, range=(5, 8, 5, 14), is_definition=False),
        ),
        symbols=(
            _ScipSymbol(symbol=caller, kind='Function', display_name='caller'),
            _ScipSymbol(symbol=callee, kind='Function', display_name='callee'),
        ),
    )
    g = _graph_with(doc, source='proj', language='python')
    assert any(e.caller.canonical_id == caller and e.callee.canonical_id == callee
               for e in g._edges)


def test_reference_before_the_first_definition_is_still_dropped() -> None:
    """A file-scope reference has no enclosing definition and must stay dropped.

    Synthesising ranges must not invent a caller for an import or a top-level
    expression -- that would fabricate edges rather than recover them.
    """
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    callee = 'semanticdb maven . . p/Foo#bar().'
    definition = 'semanticdb maven . . p/Later#baz().'
    doc = _ScipDoc(
        relative_path='X.scala',
        occurrences=(
            _ScipOccurrence(symbol=callee, range=(2, 0, 2, 5), is_definition=False),
            _ScipOccurrence(symbol=definition, range=(50, 6, 50, 9),
                            is_definition=True, enclosing_range=()),
        ),
        symbols=(
            _ScipSymbol(symbol=callee, kind='Method', display_name='bar'),
            _ScipSymbol(symbol=definition, kind='Method', display_name='baz'),
        ),
    )
    g = _graph_with(doc)
    assert not [e for e in g._edges if e.callee.canonical_id == callee]


def test_parameters_on_the_signature_line_do_not_steal_the_method_body() -> None:
    """Both failures the synthetic fixture missed, caught on real scip-java data.

    scip-java defines a method AND each of its parameters on the signature
    line, and emits dense `local N` bindings through the body. Two consequences
    the first implementation got wrong:

    * ending a definition at `next entry - 1` ended `runMerge()` at the line
      before itself, so it enclosed nothing;
    * leaving locals/parameters in the candidate set made the nearest binding
      win, attributing the call to `local 30` instead of the method.
    """
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    method = 'semanticdb maven . . p/MergeIntoCommand#runMerge().'
    param = 'semanticdb maven . . p/MergeIntoCommand#runMerge().(spark)'
    later = 'semanticdb maven . . p/MergeIntoCommand#commit().'
    callee = 'semanticdb maven . . p/ClassicMergeExecutor#writeAllChanges().'

    doc = _ScipDoc(
        relative_path='MergeIntoCommand.scala',
        occurrences=(
            _ScipOccurrence(symbol=method, range=(84, 6, 84, 14),
                            is_definition=True, enclosing_range=()),
            _ScipOccurrence(symbol=param, range=(84, 20, 84, 25),
                            is_definition=True, enclosing_range=()),
            _ScipOccurrence(symbol='local 30', range=(120, 8, 120, 12),
                            is_definition=True, enclosing_range=()),
            _ScipOccurrence(symbol=callee, range=(137, 18, 137, 33),
                            is_definition=False),
            _ScipOccurrence(symbol=later, range=(201, 6, 201, 12),
                            is_definition=True, enclosing_range=()),
            # the callee's own definition — in the real corpus it lives in
            # ClassicMergeExecutor.scala; declared here so it is a known symbol
            _ScipOccurrence(symbol=callee, range=(400, 6, 400, 21),
                            is_definition=True, enclosing_range=()),
        ),
        symbols=(
            _ScipSymbol(symbol=method, kind='Method', display_name='runMerge'),
            _ScipSymbol(symbol=param, kind='Parameter', display_name='spark'),
            _ScipSymbol(symbol='local 30', kind='Variable', display_name='x'),
            _ScipSymbol(symbol=later, kind='Method', display_name='commit'),
            _ScipSymbol(symbol=callee, kind='Method', display_name='writeAllChanges'),
        ),
    )
    g = _graph_with(doc)
    callers = {e.caller.canonical_id for e in g._edges
               if e.callee.canonical_id == callee}
    assert method in callers, f'call must attribute to runMerge(); got {callers}'
    assert not any(c.startswith('local ') for c in callers), \
        'a local binding must never be recorded as a caller'
    assert param not in callers, 'a parameter must never be recorded as a caller'


def test_the_stored_symbol_carries_the_body_extent_not_the_identifier_line() -> None:
    """The extent the walk synthesises must also be PERSISTED, so a hop is quotable.

    Measured on the live store before this fix: **0 of 306,473 named databricks symbols
    had a multi-line extent** — every ``line_end`` equalled ``line_start``, because
    ``_collect_definitions`` stored the identifier occurrence. The synthesis existed only
    inside caller attribution and was thrown away, so ``get_element_body`` had one line to
    read and returned a 32-character signature fragment instead of a body.

    ``line_start`` must NOT move: it is the join key that resolves 95-100% of callables
    (design §2.2), and the catalog's element location is keyed on it.
    """
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    first = 'semanticdb maven . . p/C#first().'
    second = 'semanticdb maven . . p/C#second().'

    doc = _ScipDoc(
        relative_path='C.scala',
        occurrences=(
            # scip-java: identifier ranges only, no enclosing_range anywhere
            _ScipOccurrence(symbol=first, range=(9, 6, 9, 11),
                            is_definition=True, enclosing_range=()),
            _ScipOccurrence(symbol=second, range=(29, 6, 29, 12),
                            is_definition=True, enclosing_range=()),
            # a reference further down: the last definition may run to here, no further
            _ScipOccurrence(symbol=first, range=(40, 4, 40, 9), is_definition=False),
        ),
        symbols=(
            _ScipSymbol(symbol=first, kind='Method', display_name='first'),
            _ScipSymbol(symbol=second, kind='Method', display_name='second'),
        ),
    )
    g = _graph_with(doc)

    a = g._symbols[first]
    b = g._symbols[second]
    assert (a.line_start, a.line_end) == (10, 29), 'first() runs to the line before second()'
    assert (b.line_start, b.line_end) == (30, 41), (
        'the last definition runs to the last line the document mentions, never to a '
        'sentinel like 1<<30 that a body read would have to clamp'
    )


def test_a_real_enclosing_range_is_used_verbatim_and_start_still_holds() -> None:
    """Where the indexer supplies the body span (scip-python), it wins untouched."""
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    fn = 'scip-python python src1 0.1 `m`/fn().'
    doc = _ScipDoc(
        relative_path='m.py',
        occurrences=(
            _ScipOccurrence(symbol=fn, range=(4, 4, 4, 6), is_definition=True,
                            enclosing_range=(4, 0, 12, 0)),
        ),
        symbols=(_ScipSymbol(symbol=fn, kind='Function', display_name='fn'),),
    )
    g = _graph_with(doc, source='src1', language='python')

    sym = g._symbols[fn]
    assert sym.line_start == 5, 'the identifier line is the join key and must not move'
    assert sym.line_end == 13, 'the body end comes from enclosing_range'


def test_a_definition_without_an_enclosing_range_still_gets_a_body_in_a_mixed_doc() -> None:
    """The extent rule is PER DEFINITION, not per document.

    ``_collect_edges`` gates its synthesis on the whole document — correct there, because
    handing every ``local N`` a synthetic body would let it beat the enclosing method on
    in-body calls. Reusing that gate for *extents* left a hole: in a document where some
    definitions carry ``enclosing_range`` and others do not, the others fell back to their
    identifier line. In a prior Python index, 221 of 360 named symbols stayed
    single-line, and **all 221 lacked an enclosing_range** — none was genuinely one line.
    """
    from docgen.scip_extractor import _ScipDoc, _ScipOccurrence, _ScipSymbol

    with_range = 'scip-python python src1 0.1 `m`/withrange().'
    without = 'scip-python python src1 0.1 `m`/without().'

    doc = _ScipDoc(
        relative_path='m.py',
        occurrences=(
            _ScipOccurrence(symbol=with_range, range=(4, 4, 4, 13),
                            is_definition=True, enclosing_range=(4, 0, 9, 0)),
            # same document, no enclosing_range of its own
            _ScipOccurrence(symbol=without, range=(19, 4, 19, 11),
                            is_definition=True, enclosing_range=()),
            _ScipOccurrence(symbol=with_range, range=(30, 4, 30, 13),
                            is_definition=False),
        ),
        symbols=(
            _ScipSymbol(symbol=with_range, kind='Function', display_name='withrange'),
            _ScipSymbol(symbol=without, kind='Function', display_name='without'),
        ),
    )
    g = _graph_with(doc, source='src1', language='python')

    supplied = g._symbols[with_range]
    fallback = g._symbols[without]
    assert (supplied.line_start, supplied.line_end) == (5, 10), 'its own range wins'
    assert (fallback.line_start, fallback.line_end) == (20, 31), (
        'a definition the indexer gave no body must be reconstructed positionally, '
        'not collapsed onto its identifier line'
    )
