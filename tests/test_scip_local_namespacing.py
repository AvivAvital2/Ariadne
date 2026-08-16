"""A ``local N`` binding belongs to its document, not to the whole store.

SCIP numbers local bindings **per document**: the first local in every file is ``local 0``,
the second ``local 1``. Ariadne stores ``canonical_id`` as a global primary key, so one row
wins and every document that emitted that index points at it. Measured on the live store:

* 7,910 bare local rows
* edges from **4,446 distinct files** point at ``local 5`` alone
* ``local 0`` carries 16,061 in-edges
* **499,249 edges join symbols attributed to different sources** — 76,823 of them ``call``
  edges — and essentially all of them touch a local id

The consequence is not a slightly noisy graph. A walk that enters a fused local can leave
into any of thousands of unrelated files, in any repository. Scoping the id to its document
is what makes the first stage of the north star trustworthy.

Synthetic fixtures only.
"""
from __future__ import annotations

import re

from docgen.scip_cross_source import CrossSourceGraph
from docgen.scip_extractor import ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol

FIRST = 'scip-python python src1 0.1 `a`/first().'
SECOND = 'scip-python python src1 0.1 `b`/second().'
TARGET_A = 'scip-python python src1 0.1 `a`/hit_a().'
TARGET_B = 'scip-python python src1 0.1 `b`/hit_b().'
BARE_LOCAL = 'local 1'


def _doc(path, owner, target):
    """A document whose local binding calls something only this file should reach."""
    return _ScipDoc(
        relative_path=path,
        occurrences=(
            _ScipOccurrence(symbol=owner, range=(0, 4, 0, 9), is_definition=True,
                            enclosing_range=(0, 0, 20, 0)),
            _ScipOccurrence(symbol=target, range=(1, 4, 1, 9), is_definition=True,
                            enclosing_range=(1, 0, 3, 0)),
            # the same bare id in both files — SCIP numbers locals per document
            _ScipOccurrence(symbol=BARE_LOCAL, range=(5, 8, 5, 13),
                            is_definition=True),
            _ScipOccurrence(symbol=target, range=(6, 8, 6, 13), is_definition=False),
        ),
        symbols=(
            _ScipSymbol(symbol=owner, kind='Function', display_name=owner[-9:]),
            _ScipSymbol(symbol=target, kind='Function', display_name='hit'),
        ),
    )


def _graph():
    index = ScipIndex(documents=(
        _doc('a.py', FIRST, TARGET_A),
        _doc('b.py', SECOND, TARGET_B),
    ))
    graph = CrossSourceGraph()
    graph.add_source('src1', index=index, language='python')
    graph.materialize()
    return graph


def _is_bare(canonical_id: str) -> bool:
    """``local 1`` is bare; ``local src1:a.py:1`` is scoped.

    The scoped form keeps the ``local `` prefix on purpose, so every existing detector
    still fires — ``LIKE 'local %'`` in SQL, ``startswith('local ')`` in Python. So the
    invariant is not "nothing starts with local", it is "nothing is *only* local N".
    """
    return re.fullmatch(r'local \d+', canonical_id) is not None


def test_the_same_bare_local_in_two_documents_becomes_two_symbols():
    graph = _graph()

    locals_ = [sym for cid, sym in graph._symbols.items()
               if cid.startswith('local ')]
    assert len(locals_) == 2, 'one row per document, not one row shared'
    assert {sym.file for sym in locals_} == {'a.py', 'b.py'}


def test_no_bare_local_id_survives_ingest():
    """A bare id is the defect: it is what makes two documents collide."""
    graph = _graph()

    assert BARE_LOCAL not in graph._symbols
    assert not [cid for cid in graph._symbols if _is_bare(cid)]
    # and the scoped ids still announce themselves as locals to every consumer
    assert all(cid.startswith('local ')
               for cid in graph._symbols if 'local' in cid)


def test_a_local_id_still_resolves_within_its_own_document():
    """Scoping must not break the local's own edges — `dc138d3` kept locals as call
    sites deliberately, because a local-scope call site is a real dependency."""
    graph = _graph()

    a_files = {e.caller.file for e in graph._edges
               if e.callee.canonical_id == TARGET_A}
    assert a_files == {'a.py'}, 'a.py\'s local must not reach b.py\'s target'


def test_locals_in_different_documents_share_no_edges():
    """The fusion in miniature: before scoping, one local carried both files' edges."""
    graph = _graph()

    for edge in graph._edges:
        assert edge.caller.file == edge.callee.file, (
            f'{edge.caller.canonical_id} -> {edge.callee.canonical_id} crosses '
            f'{edge.caller.file} into {edge.callee.file}'
        )
