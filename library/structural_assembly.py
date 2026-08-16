"""Structural assembly — walk the SCIP call graph and return quotable coordinates.

This is the mechanism ``designs/answer-path.md`` §5 names: given seeds localized from
what retrieval returned, walk ``scip_edges`` and hand back ``file:line`` for every hop, so
an answer can carry a chain instead of a list of document titles.

**A chain is a path, not a ranked set.** The extent comes from SCIP — a body's call sites,
the callee's definition, the cycle — and the order comes from the call-site ``line``, which
is the execution sequence. An earlier version of this module collected a breadth-first
neighbourhood and truncated it against a count budget; that is top-k retrieval wearing a
graph's clothes, the same error design §4.1 forbids one layer up, and it inverted
``runMerge``'s own ordering. There is no output cap here.

Four rules are load-bearing, each measured (design §2.6, §2.7, §5.1):

* **Expand a seed to its declared members before walking.** A ``Class``/``Trait``/``Object``
  has no outgoing ``call`` edge — its methods do. Seeding on the type alone terminates the
  walk at hop 0, which is what recorded the call graph as contributing 0.0pp. Expansion
  keys on ``parent_qualified_name``, **never** on ``kind``: scip-python leaves ``kind`` and
  ``display_name`` blank on 96–99% of rows, so a kind-gated expansion silently no-ops on
  Python.
* **``call`` and ``type_ref`` edges, and never through a ``local N`` node.** What a body
  touches is part of what it does, and reading only ``call`` discarded 69% of the store.
  A type reference is **cited but not traversed**: touching a type is not executing it, and
  following one onward reaches whatever that type happens to touch. Traversed, they were
  15,216 of 25,313 hops at production width and left a MERGE question with its widest call
  sites in correlation statistics; the descent continues through ``call`` and
  ``implements``, which are what runs. The local rule is separate and absolute: ``canonical_id`` is a global primary key while
  SCIP numbers local bindings per document, so one ``local 5`` row is shared by every
  document that emitted that index — 19.5% of the call graph fuses through such nodes,
  welding unrelated files and repositories together, and 42% of ``type_ref`` edges point at
  one. The edge's own ``file``/``line`` still gives the site, so file-level impact is
  unaffected.
* **Stay inside the intended source.** ``scip_edges`` carries no ``source_name``, so the
  guard is a lookup in ``scip_symbols`` scoped to that source — applied to every hop
  *after* the raw edge fetch, never as a SQL join (see ``_body_edges``). Because the
  descent only follows what that lookup returned, a foreign symbol is neither cited nor
  walked through.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
# The path seam has ONE owner. This module used to carry its own copy of the
# candidate-and-verify rule, which is the duplication that let three
# consumers disagree about the same join in the first place.
from docgen.scip_paths import scip_paths_for  # noqa: F401

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlite3 import Connection
@dataclass(frozen=True)
class StructuralCitation:
    """One hop, quotable at a verifiable line.

    ``file``/``line_start`` locate the *definition* reached; ``call_site_file``/
    ``call_site_line`` locate where the call was made. Both matter: the first is what a
    reader opens, the second is what proves the edge.

    ``line_end`` closes the extent, which is what makes the hop quotable at all. Selecting
    only ``line_start`` meant the body extents the ingest rebuild reconstructed — and that
    ``definition_extents_present`` guards — could not reach the answer path.
    """

    qualified_name: str
    file: str
    line_start: int
    source_name: str
    relation: str
    hop: int
    call_site_file: str
    call_site_line: int
    #: Why the walk stopped here, straight from the traversal's own judgement:
    #: ``descended`` (walked into it) · ``leaf`` (its body calls nothing in-corpus) ·
    #: ``plumbing`` (fan-in at or above the descent boundary) · ``depth`` (would have
    #: descended, hit the limit) · ``revisit`` (already expanded elsewhere) ·
    #: ``reference`` (a type the body names -- cited and quoted, never walked through).
    #: Curation reads this instead of inventing a relevance score.
    stop_reason: str = 'descended'
    #: Last line of the definition's body. Equal to ``line_start`` when the indexer gave
    #: no extent and none could be reconstructed — such a hop cites but cannot quote.
    line_end: int = 0
    #: The body this hop was reached from — the other end of the edge, recorded at the moment
    #: the walk followed it. A chain is a path, and without this the citations are a set with
    #: depth numbers: recovering the parent afterwards means matching call-site coordinates
    #: back to definitions and guessing whenever a file holds several. The traversal has the
    #: caller in hand, so this is exact and costs nothing.
    parent_qualified_name: str = ''
@dataclass(frozen=True)
class FanOut:
    """A dispatch reported rather than walked, with the shape a caller can narrow by.

    ``implementations`` is **what the index can see** — a floor, never a census. Modules of
    a corpus can sit on disk and never reach the graph (615 code files measured on the live
    databricks corpus), and implementations can live in code the corpus does not contain,
    so no count here may be presented as a total.
    """

    qualified_name: str
    file: str
    line_start: int
    implementations: int
    #: ``(package, count)``, widest first. The package is the directory the implementations
    #: live in, because that is a dimension a caller can actually name back.
    by_package: tuple[tuple[str, int], ...] = ()
    #: How many sit under a test path — often the bulk, and rarely what was asked about.
    tests: int = 0
@dataclass(frozen=True)
class Truncation:
    """What the walk left out. Never silent (design §5 contract item 6)."""

    truncated: bool = False
    dropped: int = 0
    reason: str = ''
    unresolved_seeds: tuple[str, ...] = field(default_factory=tuple)
    #: Dispatches too wide to walk, reported with their shape instead. Not a drop: the
    #: count and the spread travel so the caller can narrow, which is the one thing a
    #: silent cap cannot do.
    fan_outs: tuple[FanOut, ...] = field(default_factory=tuple)


_BATCH = 400


def _is_local(canonical_id: str) -> bool:
    """A ``local N`` binding is not addressable and must not be a graph node."""
    return canonical_id.startswith('local ')


def _chunks(items: list[str], size: int = _BATCH):
    for i in range(0, len(items), size):
        yield items[i:i + size]
def _qualified_names(conn: 'Connection', source: str,
                     ids: list[str]) -> dict[str, str]:
    """canonical_id -> qualified_name, scoped to ``source``.

    Scoped in Python for the same reason as ``_locate``, and the flip is size-dependent so
    a one-seed test cannot see it: with a single id SQLite seeks the primary key in 0.03 ms,
    but by 30 ids it prefers ``idx_scip_symbols_source`` and walks every row the source owns
    — 297 ms, and 362 ms at the 195 seeds a real ``ask`` produces, against 0.85 ms for the
    same ids fetched by ``canonical_id`` alone.
    """
    found: dict[str, str] = {}
    for chunk in _chunks(ids):
        placeholders = ','.join('?' * len(chunk))
        rows = conn.execute(
            f'SELECT canonical_id, qualified_name, source_name FROM scip_symbols '
            f'WHERE canonical_id IN ({placeholders})',
            chunk,
        )
        found.update({cid: qn for cid, qn, owner in rows if owner == source})
    return found


def expand_to_members(conn: 'Connection', source: str,
                      ids: list[str], *, levels: int = 2) -> set[str]:
    """``ids`` plus everything declared inside them, via ``parent_qualified_name``.

    Keyed on the parent link and not on ``kind`` — see the module docstring. A type
    contributes its methods; a module contributes its functions.
    """
    named = [i for i in ids if not _is_local(i)]
    out = set(named)
    frontier = set(_qualified_names(conn, source, named).values())
    for _ in range(levels):
        if not frontier:
            break
        rows: list[tuple[str, str]] = []
        for chunk in _chunks(sorted(frontier)):
            placeholders = ','.join('?' * len(chunk))
            rows += conn.execute(
                f'SELECT canonical_id, qualified_name FROM scip_symbols '
                f'WHERE source_name = ? AND parent_qualified_name IN ({placeholders}) '
                f"AND canonical_id NOT LIKE 'local %'",
                [source, *chunk],
            ).fetchall()
        fresh = {cid for cid, _ in rows} - out
        out |= fresh
        frontier = {qn for cid, qn in rows if cid in fresh}
    return out
def _implementors(conn: 'Connection', symbol: str) -> list[tuple[str, str, int]]:
    """Who implements ``symbol`` — ``(implementor_id, file, line)``.

    Ingest stores the relation as ``implementor --implements--> interface``, so the
    implementations of an abstract member are its *callers* on that edge type. This is the
    SCIP relationship ``is_implementation``, and it is the only thing in the index that
    answers **which implementation runs** at a polymorphic call site. A walk that reads
    only ``call`` treats an abstract method as a dead end.

    Sorted in Python, **never** with ``ORDER BY`` in the SQL. Ordering by
    ``caller_canonical_id`` -- an indexed column -- let SQLite satisfy the sort by
    scanning every ``implements`` row in caller order and filtering each against the
    callee, instead of seeking the one callee it was asked about: 52.92 ms per call
    against 0.01 ms, and 9.7 of the 10.0 seconds a ``runMerge`` chain took. The result
    sets are tiny (39 rows across 178 calls on that chain), so the sort costs nothing
    here.
    """
    rows = [
        row for row in conn.execute(
            'SELECT caller_canonical_id, file, line FROM scip_edges '
            "WHERE callee_canonical_id = ? AND edge_type = 'implements'",
            (symbol,),
        )
        if not _is_local(row[0])
    ]
    return sorted(rows)


def _fan_in(conn: 'Connection', ids: list[str]) -> dict[str, int]:
    """How many callers each candidate has — the demotion signal.

    A symbol called from everywhere is plumbing; a specific hop has few callers. Measured
    on the live store, the separation is roughly 100x: ``LogicalPlan.<init>`` 375 callers
    and ``SparkSession.conf`` 188, against ``ClassicMergeExecutor.writeAllChanges`` 2.
    Effectively IDF for a call graph. It decides **descent only** — see
    ``DEFAULT_EXPAND_FAN_IN_MAX``; it never decides what is cited, and it never orders the
    output, because ordering by anything but the call-site line destroys the sequence.
    """
    counts = dict.fromkeys(ids, 0)
    for chunk in _chunks(ids):
        placeholders = ','.join('?' * len(chunk))
        for cid, seen in conn.execute(
                f'SELECT callee_canonical_id, COUNT(*) FROM scip_edges '
                f'WHERE callee_canonical_id IN ({placeholders}) '
                f"AND edge_type = 'call' GROUP BY callee_canonical_id",
                chunk):
            counts[cid] = seen
    return counts
#: Where a JVM build layout stops and the package begins.
_BUILD_LAYOUT = re.compile(r'(?:^|/)src/(?:main|test)/(?:scala|java|kotlin|resources)/')


def _package_label(file: str) -> str:
    """The package a file sits in, with the build layout removed.

    ``sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/Add.scala`` is
    ``org.apache.spark.sql.catalyst.expressions`` — what a caller would name back. The
    directory as stored repeats the module and carries ``src/main/scala`` on top of the
    package, and a sentence asking someone to choose an area cannot afford that noise.

    A layout with no ``src/main`` (Python, JS) keeps its directory, which is already the
    package there.
    """
    directory = file.rsplit('/', 1)[0] if '/' in file else file
    match = _BUILD_LAYOUT.search(directory + '/')
    if match:
        directory = (directory + '/')[match.end():].rstrip('/')
    return directory.replace('/', '.')


def _fan_out_for(conn: 'Connection', source: str, interface: str,
                 implementors: list[tuple[str, str, int]]) -> FanOut:
    """Describe a dispatch too wide to walk, from what the index holds about it.

    The package is taken from the implementation's **directory** rather than its qualified
    name: a caller narrowing a search says "the connector one" or "not the tests", and a
    directory is what that maps onto.
    """
    located = _locate(conn, source, [interface])
    qualified_name, file, line_start, _ = located.get(
        interface, (interface, '', 0, 0))
    impls = _locate(conn, source, [impl for impl, _, _ in implementors])
    packages: Counter = Counter()
    tests = 0
    for _qn, impl_file, _start, _end in impls.values():
        packages[_package_label(impl_file)] += 1
        if 'test' in impl_file.lower():
            tests += 1
    return FanOut(
        qualified_name=qualified_name, file=file, line_start=line_start,
        # what would have been cited: implementations this source owns
        implementations=len(impls),
        by_package=tuple(packages.most_common()), tests=tests)
def _locate(conn: 'Connection', source: str,
            ids: list[str]) -> dict[str, tuple[str, str, int, int]]:
    """canonical_id -> (qualified_name, file, line_start, line_end), scoped to ``source``.

    This lookup *is* the source guard: an id absent from it is external to the corpus or
    owned by another source, and is therefore neither cited nor descended into.

    The scoping happens in Python and **not** in the WHERE clause, which is a 1900x
    difference rather than a style choice. ``source_name`` holds three distinct values
    across 727,967 rows, so naming it as a predicate made SQLite prefer
    ``idx_scip_symbols_source`` over the primary key and walk every row that source owns
    on each batch: 251 ms per call, against 0.13 ms for the same ids fetched by
    ``canonical_id`` alone. On the ``runMerge`` chain that one choice was 14.2 of 24.0
    seconds, and it scaled with the walk -- 279 hops hid it, 884 did not. The guard is
    identical either way: a row owned by another source is dropped before the caller sees
    it.

    ``line_end`` travels because stage two quotes a hop from its extent.
    """
    out: dict[str, tuple[str, str, int, int]] = {}
    for chunk in _chunks(ids):
        placeholders = ','.join('?' * len(chunk))
        for cid, qn, file, line_start, line_end, owner in conn.execute(
                f'SELECT canonical_id, qualified_name, file, line_start, line_end, '
                f'source_name FROM scip_symbols WHERE canonical_id IN ({placeholders})',
                chunk):
            if owner != source:
                continue
            out[cid] = (qn, file, line_start or 0, line_end or 0)
    return out
def _body_edges(conn: 'Connection',
                caller: str) -> list[tuple[str, str, int, str]]:
    """One body's outgoing edges, **in source order** — (callee_id, file, line, edge_type).

    Both ``call`` and ``type_ref``. What a body touches is part of what it does: the types
    it constructs, the fields it reads. Reading only ``call`` discarded 69% of the store —
    1,810,941 ``type_ref`` edges against 777,182 calls — and left an answer able to say
    which methods ran but not what they operated on. 42% of them point at a ``local`` and
    are dropped here with every other local.

    The two are never merged: the citation records which it was, because calling a method
    and mentioning a type are different claims and synthesis must not confuse them.

    Deliberately **no join to ``scip_symbols``** to scope the callees: measured on the live
    store, a hop costs ~0 ms as a raw indexed fetch and **9.31 s** with the join, because
    against an ``IN`` list SQLite abandons ``idx_scip_edges_caller`` and scans. Scoping
    happens afterwards, in ``_locate``.

    Ordering by ``line`` is what turns a set of edges into a sequence: measured on
    ``MergeIntoCommand.runMerge``, the line-ordered calls read as the MERGE algorithm,
    with ``isInsertOnly`` at 124-127 sitting immediately before the three executors it
    chooses between (130 / 138 / 149). No ranking can recover that; ranking destroys it.
    """
    return [
        row for row in conn.execute(
            'SELECT callee_canonical_id, file, line, edge_type FROM scip_edges '
            "WHERE caller_canonical_id = ? AND edge_type IN ('call', 'type_ref') "
            'ORDER BY line, callee_canonical_id',
            (caller,),
        )
        if not _is_local(row[0])
    ]


#: Descend into a callee only when it has fewer callers than this.
#:
#: The one tuned number in this module, and it decides expansion only — never what is
#: cited. Measured on the live store, accessors and framework plumbing sit at 23-375
#: callers (``SnapshotDescriptor.protocol`` 23, ``OptimisticTransactionImpl.metadata``
#: 169, ``SQLMetric.value`` 185, ``SparkSession.conf`` 188, ``LogicalPlan.<init>`` 375)
#: while real steps in the chain sit at 1-2 (``prepareMergeSource`` 1,
#: ``ClassicMergeExecutor.writeAllChanges`` 2). Out-degree cannot separate them —
#: ``prepareMergeSource`` and ``SQLMetric.value`` both make 2 calls — so fan-in is the
#: discriminator, and this default sits inside the observed gap with margin either side.
DEFAULT_EXPAND_FAN_IN_MAX = 16
DEFAULT_DISCLOSE_ABOVE = 10
def chain_from_seeds(
    conn: 'Connection',
    symbols,
    *,
    source: str,
    depth: int = 3,
    expand_below_fan_in: int = DEFAULT_EXPAND_FAN_IN_MAX,
    disclose_above: int = DEFAULT_DISCLOSE_ABOVE,
) -> tuple[list[StructuralCitation], Truncation]:
    """Trace one chain covering every seed, sharing state across them.

    Cost is the graph reached, **not** seeds x graph. Measured on the rebuilt databricks
    store: 8 retrieved documents yield 195 seeds, and walking each independently took ~67s
    for a single ``ask`` -- every seed re-expanded members and re-walked ground the earlier
    ones had already covered. One ``expanded`` set across all seeds removes that.

    Order survives: roots are visited in file-and-line order and each body's edges in line
    order, so the result still reads as execution rather than as a set.
    """
    resolved = _locate(conn, source, sorted(set(symbols)))
    unresolved = tuple(sorted(set(symbols) - set(resolved)))

    roots = _locate(conn, source,
                    sorted(expand_to_members(conn, source, sorted(resolved))))
    citations: list[StructuralCitation] = []
    expanded: set[str] = set(roots)
    fan_outs: list[FanOut] = []
    deferred = 0

    def trace(caller: str, caller_name: str, hop: int, rows) -> None:
        nonlocal deferred
        located = _locate(conn, source, [callee for callee, _, _, _ in rows])
        fan_in = _fan_in(conn, list(located))
        for callee, site_file, site_line, edge_type in rows:
            if callee not in located:
                continue
            qualified_name, file, line_start, line_end = located[callee]
            onward = None
            if edge_type == 'type_ref':
                # Touching a type is not executing it. Cite it, do not continue through
                # it, and do not dispatch from it -- nothing runs on account of a name.
                reason = 'reference'
            elif fan_in.get(callee, 0) >= expand_below_fan_in:
                reason = 'plumbing'
            elif callee in expanded:
                reason = 'revisit'
            elif hop >= depth:
                reason = 'depth'
                deferred += 1
            else:
                onward = _body_edges(conn, callee)
                reason = 'descended' if onward else 'leaf'
            citations.append(StructuralCitation(
                qualified_name=qualified_name, file=file, line_start=line_start,
                source_name=source,
                relation='calls' if edge_type == 'call' else 'references',
                hop=hop,
                call_site_file=site_file, call_site_line=site_line,
                stop_reason=reason, line_end=line_end,
                parent_qualified_name=caller_name,
            ))
            if reason == 'descended':
                expanded.add(callee)
                trace(callee, qualified_name, hop + 1, onward)
            # Both a leaf and a descended hop can be overridden. Gating this on `leaf`
            # alone answered "which implementation runs" only for a method with no body
            # of its own, which is not where overriding lives: on the rebuilt store 7,176
            # descending hops have implementors, and the base body the walk descended into
            # is not what runs for a subtype.
            if reason in ('descended', 'leaf'):
                dispatch(callee, qualified_name, hop)

    def dispatch(interface: str, interface_name: str, hop: int) -> None:
        """Follow `implements` from a hop to the implementations that really run.

        Up to ``disclose_above`` implementations, every one is cited. A ceiling on what is
        *cited* would be the count budget this module's docstring rejects, and how widely an
        interface forks is itself part of the answer to which implementation runs.

        Past it the fan-out is **reported instead of walked**: a :class:`FanOut` records how
        many there are, which packages they sit in, and how many are tests. That is not the
        cap this module rejects — nothing is dropped silently and no subset is chosen — it is
        the difference between handing back 529 coordinates and saying "this forks 529 ways
        in the index; which area did you mean?". The count is what the index can see, so it
        is a floor.
        """
        nonlocal deferred
        implementors = _implementors(conn, interface)
        if not implementors:
            return
        if len(implementors) > disclose_above:
            fan_outs.append(_fan_out_for(conn, source, interface, implementors))
            return
        located = _locate(conn, source, [impl for impl, _, _ in implementors])
        for impl, site_file, site_line in implementors:
            if impl not in located or impl in expanded:
                continue
            qualified_name, file, line_start, line_end = located[impl]
            onward = _body_edges(conn, impl) if hop < depth else None
            if hop >= depth:
                deferred += 1
                reason = 'depth'
            else:
                reason = 'descended' if onward else 'leaf'
            citations.append(StructuralCitation(
                qualified_name=qualified_name, file=file, line_start=line_start,
                source_name=source, relation='implemented_by', hop=hop + 1,
                call_site_file=site_file, call_site_line=site_line,
                stop_reason=reason, line_end=line_end,
                parent_qualified_name=interface_name,
            ))
            if reason == 'descended':
                expanded.add(impl)
                trace(impl, qualified_name, hop + 2, onward)

    for caller in sorted(roots, key=lambda cid: (roots[cid][1], roots[cid][2])):
        trace(caller, roots[caller][0], 1, _body_edges(conn, caller))

    return citations, Truncation(
        truncated=bool(deferred) or bool(unresolved) or bool(fan_outs), dropped=deferred,
        reason=('depth' if deferred else 'unresolved seeds' if unresolved
                else 'fan-out' if fan_outs else ''),
        unresolved_seeds=unresolved, fan_outs=tuple(fan_outs),
    )
def chain_from(
    conn: 'Connection',
    symbol: str,
    *,
    source: str,
    depth: int = 3,
    expand_below_fan_in: int = DEFAULT_EXPAND_FAN_IN_MAX,
) -> tuple[list[StructuralCitation], Truncation]:
    """The single-seed case of ``chain_from_seeds``."""
    if not _locate(conn, source, [symbol]):
        return [], Truncation(truncated=True, reason='unresolved seed',
                              unresolved_seeds=(symbol,))
    return chain_from_seeds(conn, [symbol], source=source, depth=depth,
                            expand_below_fan_in=expand_below_fan_in)


def symbols_in_files(conn: 'Connection', source: str,
                     files: list[str]) -> list[str]:
    """Named symbols defined in ``files`` — the seed rule design §5.1 measures."""
    out: set[str] = set()
    for chunk in _chunks(files):
        placeholders = ','.join('?' * len(chunk))
        out |= {
            row[0] for row in conn.execute(
                f'SELECT canonical_id FROM scip_symbols WHERE source_name = ? '
                f'AND file IN ({placeholders}) '
                f"AND canonical_id NOT LIKE 'local %'",
                [source, *chunk],
            )
        }
    return sorted(out)


@dataclass(frozen=True)
class SeedSet:
    """Seeds plus an account of where they came from and what was lost."""

    seeds: list[str] = field(default_factory=list)
    from_files: int = 0
    from_mentions: int = 0
    ambiguous_mentions: int = 0
    unresolved_paths: tuple[str, ...] = ()


#: A capitalised identifier of four characters or more. Deliberately narrow: this feeds
#: the FALLBACK route only, and every match must then resolve to exactly one symbol.
_MENTION = re.compile(r'\b[A-Z][A-Za-z0-9]{3,}\b')


def _unique_symbol(conn: 'Connection', source: str, name: str) -> str | None:
    """The one symbol named ``name``, or ``None`` if none or several.

    Uniqueness is the guard that makes mention-seeding safe. Measured over the 32
    discarded files: 66 mentions resolve to exactly one symbol and **231 resolve to
    several** — without the guard those 231 are the "~711 spurious seeds per question"
    that made token-seeded SCIP contribute 0.8pp (design §5.1 rule 1). Same shape as the
    ``len(candidates) == 1`` guard that rejected 39,034 ambiguous external refs (§2.5).
    """
    rows = conn.execute(
        'SELECT canonical_id FROM scip_symbols WHERE source_name = ? '
        "AND display_name = ? AND canonical_id NOT LIKE 'local %' LIMIT 2",
        (source, name),
    ).fetchall()
    return rows[0][0] if len(rows) == 1 else None


def _document_field(document, name: str):
    return (document.get(name) if isinstance(document, dict)
            else getattr(document, name, None))


def seeds_from_documents(
    conn: 'Connection',
    documents,
    *,
    source: str,
    indexer_cwds: tuple[str, ...] = (),
    source_root: str | None = None,
) -> SeedSet:
    """Seeds for ``chain_from``, taken from what retrieval already returned.

    Two routes, in order of trust:

    1. **The file route** — symbols defined in the retrieved file. This is the seeding
       design §5.1 measures at production width, and it takes required-slot reach from
       35% to 59%.
    2. **The mention route, for a document whose named files the index cannot resolve** —
       prose has no code symbols in its own file but names them in its body, and 24 of the
       32 files the file route discards name at least one indexed symbol. Each mention
       must resolve to exactly one symbol (``_unique_symbol``); ambiguous ones are dropped
       and counted. 34,235 documents in the live databricks store qualify, mostly catalog
       entries for files outside the SCIP index.

    A document naming **no** files gets neither route. It has not pointed at code, and
    scraping it anyway made prose the largest single influence on the walk: measured at
    production width (8 documents, depth 3), one retrieved theme contributed 23 of 127
    seeds, and those 23 produced **21,713 of the chain's 25,313 hops** — seeded from the
    very names the theme's own caveats section calls false positives of its clustering
    (``Corr``, ``Covariance``, ``ParquetFilters``), which are graph hubs. A MERGE question
    came out with its widest call site in correlation statistics. Store-wide, 2,579 of the
    2,581 file-less documents are themes — Ariadne's own cluster summaries — so this is
    the trust rule applied to seeding, not a special case for one document.

    The mention route never runs for a document the file route already placed, so it
    cannot flood a seed set that was resolved structurally.
    """
    by_document: list[tuple[object, list[str]]] = []
    paths: list[str] = []
    for document in documents:
        files = [f for f in (_document_field(document, 'source_files') or ()) if f]
        by_document.append((document, files))
        for file in files:
            if file not in paths:
                paths.append(file)

    resolved, unresolved = ({}, ())
    if paths:
        resolved, unresolved = scip_paths_for(
            conn, paths, source=source, indexer_cwds=indexer_cwds,
            source_root=source_root)

    seeds = set(symbols_in_files(conn, source, sorted(set(resolved.values()))))
    from_mentions = 0
    ambiguous = 0
    considered: set[str] = set()
    for document, files in by_document:
        if not files:
            continue  # pointed at no code, so it introduces none
        if any(file in resolved for file in files):
            continue  # the file route placed this document; do not scrape it
        content = _document_field(document, 'content') or ''
        for name in sorted(set(_MENTION.findall(content))):
            if name in considered:
                continue
            considered.add(name)
            hit = _unique_symbol(conn, source, name)
            if hit is not None:
                if hit not in seeds:
                    seeds.add(hit)
                    from_mentions += 1
            elif conn.execute(
                    'SELECT 1 FROM scip_symbols WHERE source_name = ? '
                    "AND display_name = ? AND canonical_id NOT LIKE 'local %' LIMIT 1",
                    (source, name)).fetchone() is not None:
                ambiguous += 1

    return SeedSet(
        seeds=sorted(seeds), from_files=len(resolved), from_mentions=from_mentions,
        ambiguous_mentions=ambiguous, unresolved_paths=unresolved,
    )
