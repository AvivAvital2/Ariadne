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
from docgen.scip_paths import scip_paths_for# noqa: F401
from library.relation_semantics import edge_relation

if TYPE_CHECKING:# pragma: no cover - typing only
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
_REFERENCE_STOPWORDS = frozenset({
    "a", "an", "and", "as", "at", "by", "does", "for", "from",
    "how", "in", "into", "is", "it", "of", "on", "or", "that",
    "the", "to", "what", "when", "where", "which", "why", "with"})


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
@dataclass(frozen=True)
class CallerExpansion:
    roots: tuple[str, ...] = field(default_factory=tuple)
    citations: tuple[StructuralCitation, ...] = field(default_factory=tuple)
    gated_targets: tuple[str, ...] = field(default_factory=tuple)
    uncovered_seeds: tuple[str, ...] = field(default_factory=tuple)
def caller_roots(
    conn: 'Connection', symbols, *, source: str, depth: int = 2,
    fan_in_max: int = DEFAULT_EXPAND_FAN_IN_MAX,
) -> CallerExpansion:
    """Find source-scoped outer callers that can root one connected forward walk.

    Reverse expansion is bounded by depth and by the same fan-in rule as forward descent.
    Only the outermost ring becomes roots; walking forward from every intermediate caller
    would duplicate subtrees and turn a path back into a neighbourhood.
    """
    current = set(_locate(conn, source, [symbol for symbol in symbols
                                        if not _is_local(symbol)]))
    initial = set(current)
    covered: set[str] = set()
    visited = set(current)
    gated: set[str] = set()
    outer_edges: dict[str, tuple[str, str, int]] = {}
    for _ in range(max(depth, 0)):
        counts = _fan_in(conn, list(current))
        eligible = {symbol for symbol in current
                    if counts.get(symbol, 0) < fan_in_max}
        gated.update(current - eligible)
        rows: list[tuple[str, str, str, int]] = []
        for chunk in _chunks(sorted(eligible)):
            placeholders = ','.join('?' * len(chunk))
            rows.extend(conn.execute(
                f'SELECT caller_canonical_id, callee_canonical_id, file, line '
                f'FROM scip_edges WHERE callee_canonical_id IN ({placeholders}) '
                f"AND edge_type = 'call'", chunk))
        callers = _locate(conn, source, [caller for caller, _, _, _ in rows
                                         if not _is_local(caller)])
        callers = {canonical_id: location for canonical_id, location in callers.items()
                   if not _nonproduction_path(location[1])}
        targets = _locate(conn, source, [callee for _, callee, _, _ in rows])
        next_edges: dict[str, tuple[str, str, int]] = {}
        for caller, callee, file, line in rows:
            if caller not in callers or callee not in targets or caller in visited:
                continue
            candidate = (callee, file, line)
            previous = next_edges.get(caller)
            if previous is None or (file, line) < (previous[1], previous[2]):
                next_edges[caller] = candidate
        covered.update(callee for callee, _, _ in next_edges.values())
        if not next_edges:
            break
        outer_edges = next_edges
        current = set(next_edges)
        visited.update(current)

    roots = tuple(sorted(outer_edges, key=lambda symbol: (
        _locate(conn, source, [symbol])[symbol][1],
        _locate(conn, source, [symbol])[symbol][2], symbol)))
    citations: list[StructuralCitation] = []
    root_locations = _locate(conn, source, roots)
    target_locations = _locate(
        conn, source, [outer_edges[root][0] for root in roots])
    for root in roots:
        target, site_file, site_line = outer_edges[root]
        qualified_name, file, line_start, line_end = root_locations[root]
        citations.append(StructuralCitation(
            qualified_name=qualified_name, file=file, line_start=line_start,
            source_name=source, relation='called_by', hop=0,
            call_site_file=site_file, call_site_line=site_line,
            stop_reason='caller_root', line_end=line_end,
            parent_qualified_name=target_locations[target][0],
        ))
    return CallerExpansion(
        roots=roots, citations=tuple(citations),
        gated_targets=tuple(sorted(gated)),
        uncovered_seeds=tuple(sorted(initial - covered)),
    )
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
                relation=edge_relation(edge_type),
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
def qualified_caller_fanout(
        conn: "Connection", roots, *, source: str,
        callers_per_root: int = 1, child_per_caller: int = 16,
        fan_in_max: int = 8) -> tuple[StructuralCitation, ...]:
    """Add a selected body's nearest structural caller, then walk forward."""
    names = tuple(dict.fromkeys(str(root) for root in roots if root))
    selected = {}
    for chunk in _chunks(list(names)):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT canonical_id,qualified_name,file,line_start,line_end,"
                f"parent_qualified_name FROM scip_symbols "
                f"WHERE source_name=? AND qualified_name IN ({marks}) "
                "AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                [source, *chunk, "local *"]):
            if not _nonproduction_path(row[2]):
                selected[str(row[0])] = tuple(row)
    incoming = []
    for chunk in _chunks(sorted(selected)):
        marks = ",".join("?" * len(chunk))
        incoming.extend(conn.execute(
            f"SELECT caller_canonical_id,callee_canonical_id,file,line,edge_type "
            f"FROM scip_edges INDEXED BY idx_scip_edges_callee "
            f"WHERE callee_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','implements')", chunk))
    caller_ids = sorted({
        str(row[0]) for row in incoming if not _is_local(row[0])})
    callers = {}
    for chunk in _chunks(caller_ids):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT canonical_id,qualified_name,file,line_start,line_end,"
                f"parent_qualified_name,source_name FROM scip_symbols "
                f"WHERE canonical_id IN ({marks})", chunk):
            if row[6] == source and not _nonproduction_path(row[2]):
                callers[str(row[0])] = tuple(row[:6])
    by_target = {}
    for row in incoming:
        if (str(row[0]) in callers and str(row[1]) in selected
                and not _nonproduction_path(row[2])):
            by_target.setdefault(str(row[1]), []).append(row)
    citations = []
    caller_names = []
    seen = set()
    for target_id in sorted(
            by_target, key=lambda item: (
                selected[item][2], selected[item][3], selected[item][1])):
        rows = by_target[target_id]
        if len({str(row[0]) for row in rows}) > max(fan_in_max, 0):
            continue
        target = selected[target_id]
        target_name = str(target[1])
        target_file = str(target[2])
        target_owner = str(target[5] or target_name.rsplit(".", 1)[0])
        ranked = sorted(rows, key=lambda row: (
            -int(str(callers[str(row[0])][5]) == target_owner),
            -int(str(callers[str(row[0])][2]) == target_file),
            abs(int(row[3]) - int(target[3])),
            str(row[2]), int(row[3]), str(row[0])))
        for caller_id, _target, site_file, site_line, _edge_type in ranked[
                :max(callers_per_root, 0)]:
            caller = callers[str(caller_id)]
            caller_name = str(caller[1])
            key = (caller_name, target_name, str(site_file), int(site_line))
            if key in seen:
                continue
            seen.add(key)
            caller_names.append(caller_name)
            citations.append(StructuralCitation(
                qualified_name=caller_name, file=str(caller[2]),
                line_start=int(caller[3]), line_end=int(caller[4]),
                source_name=source, relation="called_by", hop=0,
                call_site_file=str(site_file), call_site_line=int(site_line),
                stop_reason="selected_caller",
                parent_qualified_name=target_name))
    if caller_names:
        citations.extend(qualified_call_fanout(
            conn, tuple(dict.fromkeys(caller_names)), source=source,
            per_root=child_per_caller, depth=2,
            recursive_per_root=4, max_recursive_total=24))
    return tuple(citations)
def qualified_reverse_reference_fanout(
        conn: "Connection", roots, *, source: str, question: str = "",
        per_root: int = 2, owner_per_root: int = 1,
        fan_in_max: int = 16, max_total: int = 32,
        lift_members = False, reserve_registrars = False) -> tuple[StructuralCitation, ...]:
    """Follow bounded incoming type references from a fact to its consumer and registrar."""
    names = tuple(dict.fromkeys(str(root) for root in roots if root))
    question_tokens = set(_compound_tokens(question)) - _REFERENCE_STOPWORDS
    if not names or not question_tokens or max_total <= 0:
        return ()

    def locate(target_names):
        found = {}
        for chunk in _chunks(list(target_names)):
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                    f"SELECT canonical_id,qualified_name,file,line_start,line_end,"
                    f"parent_qualified_name FROM scip_symbols "
                    f"WHERE source_name=? AND qualified_name IN ({marks}) "
                    "AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                    [source, *chunk, "local *"]):
                if not _nonproduction_path(row[2]):
                    found[str(row[0])] = tuple(row)
        return found
    requested_rows = locate(names)
    member_owner_by_name = {}
    for row in requested_rows.values():
        qualified = str(row[1])
        owner = str(row[5] or "")
        if owner and qualified.startswith(owner + "."):
            member_owner_by_name.setdefault(qualified, owner)
    indexed_owner_rows = locate(tuple(dict.fromkeys(member_owner_by_name.values())))
    indexed_owners = {str(row[1]) for row in indexed_owner_rows.values()}
    if lift_members:
        names = tuple(dict.fromkeys(
            member_owner_by_name.get(name, name)
            if member_owner_by_name.get(name) in indexed_owners else name
            for name in names))
    selected_rows = locate(names)
    selected_names = {str(row[1]) for row in selected_rows.values()}
    selected_owners = {
        str(row[5]) for row in selected_rows.values() if row[5]}

    def incoming(target_names, *, limit, excluded_names, excluded_owners,
                 remaining):
        if remaining <= 0:
            return []
        targets = locate(target_names)
        raw = []
        for chunk in _chunks(sorted(targets)):
            marks = ",".join("?" * len(chunk))
            raw.extend(conn.execute(
                f"SELECT caller_canonical_id,callee_canonical_id,file,line "
                f"FROM scip_edges INDEXED BY idx_scip_edges_callee "
                f"WHERE callee_canonical_id IN ({marks}) "
                "AND edge_type='type_ref'", chunk))
        caller_ids = sorted({
            str(row[0]) for row in raw if not _is_local(str(row[0]))})
        callers = {}
        for chunk in _chunks(caller_ids):
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                    f"SELECT canonical_id,qualified_name,file,line_start,line_end,"
                    f"parent_qualified_name,source_name FROM scip_symbols "
                    f"WHERE canonical_id IN ({marks})", chunk):
                if (row[6] == source
                        and not _nonproduction_path(str(row[2]))):
                    callers[str(row[0])] = tuple(row[:6])

        grouped = {}
        for caller_id, target_id, site_file, site_line in raw:
            caller = callers.get(str(caller_id))
            target = targets.get(str(target_id))
            if (caller is None or target is None
                    or _nonproduction_path(str(site_file))):
                continue
            caller_name = str(caller[1])
            caller_owner = str(caller[5] or "")
            target_name = str(target[1])
            target_owner = str(target[5] or "")
            same_owner_member = (
                bool(target_owner)
                and target_name.startswith(target_owner + ".")
                and caller_owner == target_owner)
            same_component = (
                str(site_file).split("/", 1)[0]
                == str(target[2]).split("/", 1)[0])
            if (caller_name in excluded_names
             or (caller_owner in excluded_owners and not same_owner_member)
             or caller_name == target_name
             or caller_owner == target_name):
                continue
            overlap = len(
                set(_compound_tokens(caller_name)) & question_tokens)
            if overlap == 0:
                continue
            grouped.setdefault(target_name, []).append((
                -int(same_owner_member), -int(same_component), -overlap,
                -int(str(site_file) == str(caller[2])),
                caller_name, str(caller[2]), int(caller[3]),
                int(caller[4]), caller_owner,
                str(site_file), int(site_line)))

        chosen = []
        target_order = sorted(
            grouped,
            key=lambda name: (
                -len(set(_compound_tokens(name)) & question_tokens), name))
        for target_name in target_order:
            rows = grouped[target_name]
            if len({row[4] for row in rows}) > max(fan_in_max, 0):
                continue
            seen_callers = set()
            for row in sorted(rows):
                caller_name = row[4]
                if caller_name in seen_callers:
                    continue
                seen_callers.add(caller_name)
                chosen.append((target_name, *row[4:]))
                if (len(seen_callers) >= max(limit, 0)
                        or len(chosen) >= remaining):
                    break
            if len(chosen) >= remaining:
                break
        return chosen
    registrar_reserve = (
        min(3, max(max_total // 2, 0)) if reserve_registrars else 0)
    first_budget = max(max_total - registrar_reserve, 0)

    first = incoming(
        names, limit=per_root,
        excluded_names=selected_names,
        excluded_owners=selected_owners,
        remaining=first_budget)
    first_names = {row[1] for row in first}
    first_owners = {row[5] for row in first if row[5]}
    second = incoming(
        tuple(sorted(first_owners)), limit=owner_per_root,
        excluded_names=selected_names | first_names,
        excluded_owners=selected_owners | first_owners,
        remaining=max(max_total - len(first), 0))

    citations = []
    seen = set()
    for target_name, caller_name, caller_file, start, end, _owner, site_file, site_line in (*first, *second):
        key = (target_name, caller_name, site_file, site_line)
        if key in seen:
            continue
        seen.add(key)
        citations.append(StructuralCitation(
            qualified_name=caller_name, file=caller_file,
            line_start=start, line_end=end, source_name=source,
            relation="referenced_by", hop=0,
            call_site_file=site_file, call_site_line=site_line,
            stop_reason="selected_reference_caller",
            parent_qualified_name=target_name))
    return tuple(citations)
def qualified_reference_fanout(
        conn: "Connection", roots, *, source: str, question: str = "",
        per_root: int = 8, owner_per_root: int = 2, depth: int = 2,
        recursive_per_root: int = 4,
        max_recursive_total: int = 16, excluded_targets = ()) -> tuple[StructuralCitation, ...]:
    # Expose distinct question-relevant type refs with bounded owner diversity.
    question_tokens = set(_compound_tokens(question)) - _REFERENCE_STOPWORDS
    excluded_names = {str(name) for name in excluded_targets if name}
    if not question_tokens:
        return ()
    frontier = list(dict.fromkeys(str(root) for root in roots if root))
    expanded = set()
    seen = set()
    citations = []
    recursive_emitted = 0
    for level in range(max(depth, 0)):
        current = [name for name in frontier if name not in expanded]
        if not current:
            break
        expanded.update(current)
        located_roots = {}
        for chunk in _chunks(current):
            marks = ",".join("?" * len(chunk))
            for canonical, qualified, file in conn.execute(
                    f"SELECT canonical_id,qualified_name,file FROM scip_symbols "
                    f"WHERE source_name=? AND qualified_name IN ({marks}) "
                    "AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                    [source, *chunk, "local *"]):
                if not _nonproduction_path(file):
                    located_roots[str(canonical)] = str(qualified)
        raw = []
        for chunk in _chunks(sorted(located_roots)):
            marks = ",".join("?" * len(chunk))
            raw.extend(conn.execute(
                f"SELECT caller_canonical_id,callee_canonical_id,file,line "
                f"FROM scip_edges INDEXED BY idx_scip_edges_caller "
                f"WHERE caller_canonical_id IN ({marks}) "
                "AND edge_type='type_ref'", chunk))
        target_ids = [
            row[1] for row in raw if not _is_local(row[1])]
        targets = _locate(conn, source, target_ids)
        target_families = {}
        for chunk in _chunks(sorted(set(target_ids))):
            marks = ",".join("?" * len(chunk))
            for canonical, qualified, owner in conn.execute(
                    f"SELECT canonical_id,qualified_name,parent_qualified_name "
                    f"FROM scip_symbols WHERE source_name=? "
                    f"AND canonical_id IN ({marks}) "
                    "AND canonical_id NOT GLOB ?",
                    [source, *chunk, "local *"]):
                qualified = str(qualified)
                owner = str(owner or "")
                target_families[str(canonical)] = (
                    owner if owner and qualified.startswith(owner + ".")
                    else qualified)
        candidate_tokens = {}
        for caller, target, site_file, _site_line in raw:
            location = targets.get(target)
            parent = located_roots.get(caller)
            if (location is None or parent is None
                    or location[0] in excluded_names
                    or _nonproduction_path(site_file)
                    or _nonproduction_path(location[1])):
                continue
            candidate_tokens.setdefault(parent, {})[
                location[0]] = set(_compound_tokens(location[0]))
        token_frequencies = {
            parent: Counter(
                token for tokens in targets_by_name.values()
                for token in tokens)
            for parent, targets_by_name in candidate_tokens.items()}
        ranked = []
        for caller, target, site_file, site_line in raw:
            location = targets.get(target)
            if (location is None or location[0] in excluded_names
                    or _nonproduction_path(site_file)):
                continue
            target_name, target_file, start, end = location
            if _nonproduction_path(target_file):
                continue
            parent = located_roots[caller]
            target_tokens = set(_compound_tokens(target_name))
            overlap_tokens = target_tokens & question_tokens
            overlap = len(overlap_tokens)
            if question_tokens and overlap == 0:
                continue
            frequencies = token_frequencies.get(parent, {})
            semantic_score = sum(
                1.0 / max(frequencies.get(token, 1), 1)
                for token in overlap_tokens)
            same_file = int(str(site_file) == str(target_file))
            target_family = target_families.get(str(target), target_name)
            ranked.append((
                parent, -semantic_score, -overlap, -same_file,
                str(site_file), int(site_line), target_family,
                target_name, target_file, int(start), int(end)))
        counts = {}
        owner_counts = {}
        seen_targets = set()
        next_frontier = []
        level_limit = (
            max(per_root, 0) if level == 0
            else max(recursive_per_root, 0))
        for (parent, _semantic, _overlap, _same_file,
             site_file, site_line, target_family, target_name,
             target_file, start, end) in sorted(ranked):
            key = (parent, target_name, site_file, site_line)
            owner_key = (parent, target_family)
            if (key in seen
             or (parent, target_name) in seen_targets
             or counts.get(parent, 0) >= level_limit
             or owner_counts.get(owner_key, 0) >= max(owner_per_root, 0)
             or (level > 0 and recursive_emitted
                 >= max(max_recursive_total, 0))):
                continue
            seen.add(key)
            seen_targets.add((parent, target_name))
            counts[parent] = counts.get(parent, 0) + 1
            owner_counts[owner_key] = owner_counts.get(owner_key, 0) + 1
            citations.append(StructuralCitation(
                qualified_name=target_name, file=target_file,
                line_start=start, line_end=end, source_name=source,
                relation="references", hop=level + 1,
                call_site_file=site_file, call_site_line=site_line,
                stop_reason="selected_reference",
                parent_qualified_name=parent))
            if level > 0:
                recursive_emitted += 1
            if (site_file == target_file
                    and target_name not in expanded
                    and target_name not in next_frontier):
                next_frontier.append(target_name)
        frontier = next_frontier
    return tuple(citations)
def qualified_owner_closure(conn: "Connection", roots, *, source: str) -> tuple[StructuralCitation, ...]:
    """Retain the proven canonical owner of each selected qualified member.

    A dotted name is not ownership evidence. This closure emits an owner only
    when SCIP records both parent_qualified_name and a matching canonical
    contains edge. It never expands the owner's other members.
    """
    names = tuple(dict.fromkeys(str(root) for root in roots if root))
    rows = []
    for chunk in _chunks(list(names)):
        marks = ",".join("?" * len(chunk))
        rows.extend(conn.execute(
            f"SELECT owner.qualified_name,owner.file,owner.line_start,owner.line_end,"
            f"member.qualified_name,member.file,member.line_start,member.line_end,"
            f"edge.file,edge.line "
            f"FROM scip_symbols AS member "
            f"JOIN scip_symbols AS owner "
            f"ON owner.source_name=member.source_name "
            f"AND owner.qualified_name=member.parent_qualified_name "
            f"JOIN scip_edges AS edge "
            f"ON edge.caller_canonical_id=owner.canonical_id "
            f"AND edge.callee_canonical_id=member.canonical_id "
            f"AND edge.edge_type='contains' "
            f"WHERE member.source_name=? "
            f"AND member.qualified_name IN ({marks}) "
            f"AND member.canonical_id NOT GLOB ? "
            f"AND owner.canonical_id NOT GLOB ? "
            f"ORDER BY owner.qualified_name,owner.canonical_id,"
            f"member.qualified_name,member.canonical_id,edge.file,edge.line",
            [source, *chunk, "local *", "local *"]))
    owners = []
    members = []
    seen_owners = set()
    seen_members = set()
    for (owner_name, owner_file, owner_start, owner_end,
         member_name, member_file, member_start, member_end,
         edge_file, edge_line) in rows:
        if (_nonproduction_path(owner_file)
                or _nonproduction_path(member_file)
                or _nonproduction_path(edge_file)):
            continue
        owner_key = (str(owner_name), str(owner_file), int(owner_start))
        if owner_key not in seen_owners:
            seen_owners.add(owner_key)
            owners.append(StructuralCitation(
                qualified_name=str(owner_name), file=str(owner_file),
                line_start=int(owner_start), line_end=int(owner_end or owner_start),
                source_name=source, relation="localized", hop=0,
                call_site_file=str(owner_file), call_site_line=int(owner_start),
                stop_reason="selected_owner"))
        member_key = (
            str(member_name), str(member_file), int(member_start),
            str(owner_name), str(edge_file), int(edge_line))
        if member_key not in seen_members:
            seen_members.add(member_key)
            members.append(StructuralCitation(
                qualified_name=str(member_name), file=str(member_file),
                line_start=int(member_start), line_end=int(member_end or member_start),
                source_name=source, relation="contains", hop=1,
                call_site_file=str(edge_file), call_site_line=int(edge_line),
                stop_reason="selected_owner_member",
                parent_qualified_name=str(owner_name)))
    return tuple((*owners, *members))
def qualified_call_fanout(conn: "Connection", roots, *, source: str,
                          per_root: int = 4, depth: int = 1,
                          max_total: int | None = None, recursive_per_root: int = 8, max_recursive_total: int = 48) -> tuple[StructuralCitation, ...]:
    """Expose bounded compiler calls, recursively, before semantic pruning."""
    names = tuple(dict.fromkeys(str(root) for root in roots if root))
    limit = (max(per_root, 0) * max(len(names), 1) + max(max_recursive_total, 0) if max_total is None else max(max_total, 0))
    citations = []
    seen = set()
    expanded = set()
    frontier = list(names)
    recursive_emitted = 0
    for level in range(max(depth, 0)):
        current = [name for name in frontier if name not in expanded]
        if not current or len(citations) >= limit or (level > 0 and recursive_emitted >= max(max_recursive_total, 0)):
            break
        expanded.update(current)
        located_roots = {}
        for chunk in _chunks(current):
            marks = ",".join("?" * len(chunk))
            for canonical, qualified, file in conn.execute(
                    f"SELECT canonical_id,qualified_name,file FROM scip_symbols "
                    f"WHERE source_name=? AND qualified_name IN ({marks}) "
                    "AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                    [source, *chunk, "local *"]):
                if not _nonproduction_path(file):
                    owners = located_roots.setdefault(str(qualified), [])
                    if str(canonical) not in owners:
                        owners.append(str(canonical))
        root_names = {
            canonical: qualified
            for qualified, canonicals in located_roots.items()
            for canonical in canonicals}
        raw = []
        for chunk in _chunks(sorted(root_names)):
            marks = ",".join("?" * len(chunk))
            raw.extend(conn.execute(
                f"SELECT caller_canonical_id,callee_canonical_id,file,line,edge_type "
                f"FROM scip_edges INDEXED BY idx_scip_edges_caller "
                f"WHERE caller_canonical_id IN ({marks}) "
                "AND edge_type IN ('call','implements')", chunk))
        targets = _locate(conn, source, [
            row[1] for row in raw if not _is_local(row[1])])
        ordered = []
        for caller, target, site_file, site_line, edge_type in raw:
            location = targets.get(target)
            if location is None or _nonproduction_path(site_file):
                continue
            target_name, target_file, start, end = location
            if _nonproduction_path(target_file):
                continue
            ordered.append((root_names[caller], target_name, str(site_file),
                            int(site_line), str(edge_type), target_file,
                            start, end))
        counts = {}
        recursive_counts = {}
        next_frontier = []
        for (parent, target, site_file, site_line, edge_type,
             target_file, start, end) in sorted(ordered, key=lambda row: (
                 row[0], 0 if row[4] == "call" else 1,
                 row[2], row[3], row[1])):
            key = (parent, target, edge_type)
            if key in seen or counts.get(parent, 0) >= max(per_root, 0) or len(citations) >= limit or (level > 0 and recursive_emitted >= max(max_recursive_total, 0)):
                continue
            seen.add(key)
            counts[parent] = counts.get(parent, 0) + 1
            citations.append(StructuralCitation(
                qualified_name=target, file=target_file, line_start=start,
                line_end=end, source_name=source,
                relation=edge_relation(edge_type),
                hop=level + 1, call_site_file=site_file,
                call_site_line=site_line,
                stop_reason="selected_route_fanout",
                parent_qualified_name=parent))
            if level > 0:
                recursive_emitted += 1
            if target not in expanded and target not in next_frontier and recursive_counts.get(parent, 0) < max(recursive_per_root, 0) and site_file.rsplit("/", 1)[0] == target_file.rsplit("/", 1)[0]:
                next_frontier.append(target)
                recursive_counts[parent] = recursive_counts.get(parent, 0) + 1
        frontier = next_frontier
    return tuple(citations)


def selected_route_call_fanout(conn: "Connection", matches, *, source: str,
                               per_root: int = 48) -> tuple[StructuralCitation, ...]:
    """Expose direct calls for every compiler route node selected upstream."""
    roots = tuple(dict.fromkeys((
        *(name for match in matches for name in match.clew.route),
        *(symbol for match in matches for _obligation, symbol in match.target_symbols),
    )))
    return qualified_call_fanout(
        conn, roots, source=source, per_root=per_root)
def citations_from_qualified_routes(conn: "Connection", routes, *,
                                    source: str) -> list[StructuralCitation]:
    """Materialize only compiler-verified consecutive edges from chosen routes."""
    qualified = tuple(dict.fromkeys(name for route in routes for name in route))
    by_name = {}
    for chunk in _chunks(list(qualified)):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT canonical_id, qualified_name, file, line_start, line_end "
                f"FROM scip_symbols WHERE source_name = ? AND qualified_name IN ({marks}) "
                f"AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                [source, *chunk, "local *"]):
            if not _nonproduction_path(row[2]):
                by_name.setdefault(str(row[1]), []).append(row)
    citations = []
    seen = set()
    for route in routes:
        previous = ()
        for hop, qualified_name in enumerate(route):
            located = tuple(by_name.get(qualified_name, ()))
            if not located:
                previous = ()
                continue
            if not previous:
                canonical_id, _name, file, line_start, line_end = located[0]
                citation = StructuralCitation(
                    qualified_name=qualified_name, file=file,
                    line_start=line_start, line_end=line_end,
                    source_name=source, relation="localized", hop=hop,
                    call_site_file=file, call_site_line=line_start,
                    stop_reason="selected_route")
                reachable = located
            else:
                parent_ids = [str(row[0]) for row in previous]
                target_ids = [str(row[0]) for row in located]
                parent_marks = ",".join("?" * len(parent_ids))
                target_marks = ",".join("?" * len(target_ids))
                edges = list(conn.execute(
                    f"SELECT caller_canonical_id,callee_canonical_id,edge_type,file,line "
                    f"FROM scip_edges WHERE caller_canonical_id IN ({parent_marks}) "
                    f"AND callee_canonical_id IN ({target_marks}) "
                    f"AND edge_type IN (\"call\",\"type_ref\") "
                    f"ORDER BY line,caller_canonical_id,callee_canonical_id",
                    [*parent_ids, *target_ids]))
                if not edges:
                    previous = located
                    continue
                caller_id, callee_id, edge_type, site_file, site_line = edges[0]
                current_by_id = {str(row[0]): row for row in located}
                canonical_id, _name, file, line_start, line_end = current_by_id[str(callee_id)]
                parent_name = next(
                    str(row[1]) for row in previous
                    if str(row[0]) == str(caller_id))
                citation = StructuralCitation(
                    qualified_name=qualified_name, file=file,
                    line_start=line_start, line_end=line_end,
                    source_name=source,
                    relation=edge_relation(edge_type),
                    hop=hop, call_site_file=site_file, call_site_line=site_line,
                    stop_reason="selected_route",
                    parent_qualified_name=parent_name)
                reached = {str(edge[1]) for edge in edges}
                reachable = tuple(
                    row for row in located if str(row[0]) in reached)
            key = (citation.qualified_name, citation.call_site_file,
                   citation.call_site_line, citation.relation)
            if key not in seen:
                seen.add(key)
                citations.append(citation)
            previous = reachable
    return citations
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
        files = [f for f in (_document_field(document, 'source_files') or ())
                 if f and not _nonproduction_path(str(f))]
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
            continue# pointed at no code, so it introduces none
        if any(file in resolved for file in files):
            continue# the file route placed this document; do not scrape it
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
def _compound_tokens(text: str) -> tuple[str, ...]:
    surface = (text or "").replace("_", " ")
    parts = re.findall(r'[A-Z]+(?=[A-Z][a-z]|[^A-Za-z]|$)|[A-Z]?[a-z]+|[0-9]+', surface)
    normalized = []
    aliases = {"writing": "write", "written": "write", "writes": "write",
               "streaming": "stream", "identifier": "id"}
    for part in parts:
        token = aliases.get(part.lower(), part.lower())
        if token.endswith("s") and len(token) > 4:
            token = token[:-1]
        normalized.append(token)
    return tuple(normalized)
_QUESTION_SYMBOL_STOPWORDS = frozenset({"a", "about", "actually", "does", "doesn", "gets", "how", "into", "later", "of", "rather", "s", "t", "than", "that", "the", "to", "version", "what", "would"})


def question_symbol_seeds(conn: "Connection", question: str, *, source: str,
                          limit: int = 4) -> tuple[str, ...]:
    """Resolve anchorless prose through compound identifiers in the SCIP symbol table."""
    entities = tuple(dict.fromkeys(name for name in re.findall(r"\b[A-Z][A-Za-z0-9]{2,}\b", question or "") if name.lower() not in _QUESTION_SYMBOL_STOPWORDS))
    if not entities or limit <= 0:
        return ()
    question_tokens = Counter(token for token in _compound_tokens(question) if token not in _QUESTION_SYMBOL_STOPWORDS)
    candidates: dict[str, tuple[int, str, str]] = {}
    for entity in entities[:1]:
        for canonical_id, display_name, qualified_name in conn.execute(
                "SELECT canonical_id, display_name, qualified_name FROM scip_symbols "
                'WHERE source_name = ? AND kind IN ("Class","Object","Interface","Trait","") AND display_name LIKE ? '
                'AND file NOT LIKE ? AND canonical_id NOT GLOB ?', (source, f"%{entity}%", "%/test/%", "local *")):
            overlap = sum(question_tokens[token] for token in set(_compound_tokens(display_name)))
            if overlap < 2:
                continue
            candidates[canonical_id] = (overlap, display_name, qualified_name)
    if not candidates:
        return ()
    best = max(score for score, _, _ in candidates.values())
    ranked = sorted(
        (canonical_id for canonical_id, value in candidates.items() if value[0] == best),
        key=lambda canonical_id: (candidates[canonical_id][1], candidates[canonical_id][2],
                                  canonical_id))
    refined = []
    for canonical_id in ranked[:limit]:
        qualified_name = candidates[canonical_id][2]
        members = []
        for member_id, display_name in conn.execute(
                "SELECT canonical_id, display_name FROM scip_symbols "
                "WHERE source_name = ? AND parent_qualified_name = ? "
                "AND file NOT LIKE ? AND canonical_id NOT GLOB ?",
                (source, qualified_name, "%/test/%", "local *")):
            score = sum(question_tokens[token]
                        for token in set(_compound_tokens(display_name)))
            if score:
                members.append((score, len(_compound_tokens(display_name)),
                                display_name, member_id))
        if members:
            members.sort(key=lambda item: (-item[0], item[1], item[2], item[3]))
            refined.append(members[0][3])
        else:
            refined.append(canonical_id)
    if refined:
        paired = []
        entity = entities[0]
        for canonical_id in ranked[:limit]:
            fork_display = candidates[canonical_id][1]
            if not fork_display.startswith(entity) or len(fork_display) <= len(entity):
                continue
            stock_display = fork_display[len(entity):]
            selected = conn.execute(
                "SELECT display_name FROM scip_symbols WHERE canonical_id = ? AND source_name = ?",
                (refined[0], source)).fetchone()
            if selected is None:
                continue
            for stock_parent, in conn.execute(
                    "SELECT qualified_name FROM scip_symbols WHERE source_name = ? "
                    "AND display_name = ? AND kind IN (\"Class\",\"Object\",\"Interface\",\"Trait\",\"\") "
                    "AND file NOT LIKE ? AND canonical_id NOT GLOB ? ORDER BY qualified_name",
                    (source, stock_display, "%/test/%", "local *")):
                row = conn.execute(
                    "SELECT canonical_id FROM scip_symbols WHERE source_name = ? "
                    "AND parent_qualified_name = ? AND display_name = ? "
                    "AND file NOT LIKE ? AND canonical_id NOT GLOB ? ORDER BY canonical_id LIMIT 1",
                    (source, stock_parent, selected[0], "%/test/%", "local *")).fetchone()
                if row is not None:
                    paired.append(row[0])
        refined.extend(item for item in paired if item not in refined)
    return tuple(refined)
def reference_bridges(
        conn: "Connection", citations, *, question: str = "", source: str,
        fan_in_max: int = 8, target_limit: int | None = None,
        caller_limit_per_target: int | None = None,
        exclude_callers=()) -> CallerExpansion:
    """Join bounded owners that reference the same question-relevant SCIP type."""
    question_tokens = set(_compound_tokens(question))
    referenced = sorted({citation.qualified_name for citation in citations
                         if citation.relation == "references"})
    token_frequency = {}
    for qualified_name in referenced:
        for token in set(_compound_tokens(qualified_name)):
            token_frequency[token] = token_frequency.get(token, 0) + 1
    rarity_ceiling = max(1, len(referenced) // 10)
    discriminating_tokens = {
        token for token in question_tokens
        if 0 < token_frequency.get(token, 0) <= rarity_ceiling}
    explicit_entities = re.findall(
        r"\b(?:[A-Z][A-Z0-9_]{1,}|[A-Za-z_]+[A-Z][A-Za-z0-9_]*)\b", question)
    entity_tokens = set(_compound_tokens(" ".join(explicit_entities)))
    ranking_tokens = entity_tokens or discriminating_tokens
    if target_limit is None:
        matching_targets = [
            qualified_name for qualified_name in referenced
            if set(_compound_tokens(qualified_name)) & ranking_tokens]
    else:
        matching_targets = [
            qualified_name for qualified_name in referenced
            if set(_compound_tokens(qualified_name)) & question_tokens]
    if entity_tokens and matching_targets:
        minimum_specificity = min(
            len(set(_compound_tokens(name)) - entity_tokens)
            for name in matching_targets)
        matching_targets = [
            name for name in matching_targets
            if len(set(_compound_tokens(name)) - entity_tokens)
            == minimum_specificity]
    targets = sorted(set(matching_targets))
    if not targets:
        return CallerExpansion()
    target_ids: dict[str, str] = {}
    for chunk in _chunks(targets):
        placeholders = ",".join("?" * len(chunk))
        for canonical_id, qualified_name, owner in conn.execute(
                f"SELECT canonical_id, qualified_name, source_name FROM scip_symbols "
                f"WHERE qualified_name IN ({placeholders})", chunk):
            if owner == source and not _is_local(canonical_id):
                target_ids[canonical_id] = qualified_name
    rows = []
    for chunk in _chunks(sorted(target_ids)):
        placeholders = ",".join("?" * len(chunk))
        rows.extend(conn.execute(
            f"SELECT caller_canonical_id, callee_canonical_id, file, line "
            f"FROM scip_edges WHERE callee_canonical_id IN ({placeholders}) "
            f"AND edge_type = ?", [*chunk, "type_ref"]))
    rows = [row for row in rows if not _nonproduction_path(row[2])]
    unique_rows = {}
    for row in sorted(rows, key=lambda item: (item[0], item[1], item[2], item[3])):
        unique_rows.setdefault((row[0], row[1]), row)
    by_target: dict[str, list] = {}
    for row in unique_rows.values():
        by_target.setdefault(row[1], []).append(row)
    gated = tuple(sorted(
        target for target, incoming in by_target.items()
        if len({row[0] for row in incoming}) > fan_in_max))
    eligible_by_target = {
        target: incoming
        for target, incoming in by_target.items()
        if 2 <= len({item[0] for item in incoming}) <= fan_in_max}
    if target_limit is not None:
        hop_by_name = {}
        for citation in citations:
            hop_by_name[citation.qualified_name] = max(
                hop_by_name.get(citation.qualified_name, 0),
                int(citation.hop or 0))
        ranked_targets = sorted(
            eligible_by_target,
            key=lambda target: (
                -len({
                    str(row[2]).strip("/").split("/", 1)[0]
                    for row in eligible_by_target[target]}),
                -len(set(_compound_tokens(target_ids[target]))
                     & question_tokens),
                -hop_by_name.get(target_ids[target], 0),
                target_ids[target]))
        allowed = set(ranked_targets[:max(target_limit, 0)])
        eligible_by_target = {
            target: incoming
            for target, incoming in eligible_by_target.items()
            if target in allowed}
    eligible = [
        row for target in sorted(
            eligible_by_target, key=lambda item: target_ids[item])
        for row in eligible_by_target[target]]
    callers = _locate(conn, source, [
        row[0] for row in eligible if not _is_local(row[0])])
    excluded = set(exclude_callers or ())
    if caller_limit_per_target is not None:
        limited = []
        for target in sorted(
                eligible_by_target, key=lambda item: target_ids[item]):
            candidates = [
                row for row in eligible_by_target[target]
                if row[0] in callers
                and callers[row[0]][0] not in excluded]
            target_owner = target_ids[target].rsplit(".", 1)[0]
            candidates.sort(key=lambda row: (
                -int(
                    callers[row[0]][0] == target_owner
                    or callers[row[0]][0].startswith(target_owner + ".")),
                -len(set(_compound_tokens(callers[row[0]][0])) & question_tokens),
                row[2], row[3], row[0]))
            limited.extend(
                candidates[:max(caller_limit_per_target, 0)])
        eligible = limited
    bridged = []
    seen = set()
    for caller, target, file, line in sorted(
            eligible, key=lambda row: (row[2], row[3], row[0])):
        if (caller not in callers
                or callers[caller][0] in excluded
                or (caller, target, file, line) in seen):
            continue
        seen.add((caller, target, file, line))
        qualified_name, definition_file, line_start, line_end = callers[caller]
        bridged.append(StructuralCitation(
            qualified_name=qualified_name, file=definition_file,
            line_start=line_start, source_name=source,
            relation="shared_reference", hop=0,
            call_site_file=file, call_site_line=line,
            stop_reason="reference_bridge", line_end=line_end,
            parent_qualified_name=target_ids[target]))
    return CallerExpansion(citations=tuple(bridged), gated_targets=gated)
def _obligation_neighbors(conn: "Connection", frontier: dict[str, str], *,
                          source: str) -> list:
    """Fetch both edge directions first, then localize endpoints in one batch.

    Joining ``scip_symbols`` while an edge ``IN`` frontier is active makes the
    runtime proportional to the joined neighborhood before the five-result beam
    can bound it. Endpoint indexes make the edge read cheap; ``_locate`` then
    performs primary-key symbol lookups and applies source ownership in Python.
    """
    raw = []
    for chunk in _chunks(sorted(frontier)):
        marks = ",".join("?" * len(chunk))
        raw.extend(conn.execute(
            f"SELECT caller_canonical_id,callee_canonical_id,file,line,edge_type "
            f"FROM scip_edges INDEXED BY idx_scip_edges_caller "
            f"WHERE caller_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','type_ref','implements')",
            chunk))
        raw.extend(
            (callee, caller, file, line, "incoming_" + edge_type)
            for caller, callee, file, line, edge_type in conn.execute(
                f"SELECT caller_canonical_id,callee_canonical_id,file,line,edge_type "
                f"FROM scip_edges INDEXED BY idx_scip_edges_callee "
                f"WHERE callee_canonical_id IN ({marks}) "
                "AND edge_type IN ('call','type_ref','implements')",
                chunk))
    located = _locate(
        conn, source, [row[1] for row in raw if not _is_local(row[1])])
    rows = []
    for caller, target, site_file, site_line, edge_type in raw:
        target_location = located.get(target)
        if target_location is None:
            continue
        target_name, target_file, start, end = target_location
        rows.append((caller, target, site_file, site_line, edge_type,
                     target_name, target_file, start, end))
    return rows
def _shared_reference_callers(conn: "Connection", target: str, *,
                              source: str) -> tuple:
    """Return a non-ubiquitous type's callers without an edge/symbol join."""
    raw = list(conn.execute(
        "SELECT caller_canonical_id,file,line FROM scip_edges "
        "INDEXED BY idx_scip_edges_callee WHERE callee_canonical_id=? "
        "AND edge_type='type_ref' "
        "ORDER BY caller_canonical_id, file, line LIMIT 9", (target,)))
    if len(raw) > 8:
        return ()
    located = _locate(conn, source, [row[0] for row in raw])
    return tuple(
        (caller, site_file, site_line, *located[caller])
        for caller, site_file, site_line in raw
        if caller in located)
def obligation_reference_closure(conn: "Connection", matches, *, source: str,
                                 question: str, per_depth: int = 5, depth: int = 2):
    """Bounded bidirectional SCIP closure around each selected obligation route."""
    groups = {}
    obligation_surfaces = {}
    for match in matches:
        for obligation in match.obligations:
            groups.setdefault(obligation, set()).update(match.clew.route)
        for obligation, text in match.obligation_texts:
            if text:
                obligation_surfaces.setdefault(obligation, []).append(text)
        for obligation, symbol in match.target_symbols:
            groups.setdefault(obligation, set()).add(symbol)
    question_tokens = set(_compound_tokens(question))
    citations = []
    seen_edges = set()
    for obligation in sorted(groups):
        obligation_text = " ".join(obligation_surfaces.get(obligation, ()))
        obligation_tokens = set(_compound_tokens(obligation_text)) or question_tokens
        explicit_entities = re.findall(
            r"\b(?:[A-Z][A-Z0-9_]{1,}|[A-Za-z_]+[A-Z][A-Za-z0-9_]*)\b",
            question + " " + obligation_text)
        entity_tokens = set(_compound_tokens(" ".join(explicit_entities)))
        frontier = {}
        names = sorted(groups[obligation])
        for chunk in _chunks(names):
            marks = ",".join("?" * len(chunk))
            for canonical_id, qualified_name in conn.execute(
                    f"SELECT canonical_id, qualified_name FROM scip_symbols "
                    f"WHERE source_name = ? AND qualified_name IN ({marks}) "
                    "AND canonical_id NOT GLOB ?", [source, *chunk, "local *"]):
                frontier[canonical_id] = qualified_name
        visited = set(frontier)
        for level in range(max(depth, 0)):
            candidates = _obligation_neighbors(conn, frontier, source=source)
            unique_candidates = {}
            for row in sorted(candidates, key=lambda item: (
                    item[0], item[1], item[4], item[2], item[3])):
                unique_candidates.setdefault((row[0], row[1], row[4]), row)
            candidates = list(unique_candidates.values())
            candidate_names = {row[5] for row in candidates}
            token_frequency = {}
            for candidate_name in candidate_names:
                for token in set(_compound_tokens(candidate_name)):
                    token_frequency[token] = token_frequency.get(token, 0) + 1
            rarity_ceiling = max(1, len(candidate_names) // 10)
            discriminating_tokens = {
                token for token in obligation_tokens
                if 0 < token_frequency.get(token, 0) <= rarity_ceiling}
            ranking_tokens = entity_tokens or discriminating_tokens or obligation_tokens
            ranked = []
            for row in candidates:
                if (row[1] in visited or _nonproduction_path(row[2])
                        or _nonproduction_path(row[6])):
                    continue
                overlap = len(set(_compound_tokens(row[5])) & ranking_tokens)
                kind_rank = (0 if row[4] == "call" else
                             1 if row[4] == "type_ref" else 2)
                specificity = len(set(_compound_tokens(row[5])) - ranking_tokens)
                ranked.append((-overlap, specificity, kind_rank, row[2], row[3],
                               row[5], row[0], row[1], row))
            next_frontier = {}
            for (_score, _specificity, _kind, _file, _line, _name,
                 _caller_id, _target_id, row) in sorted(ranked)[:max(per_depth, 0)]:
                (caller, target, site_file, site_line, edge_type, target_name,
                 target_file, start, end) = row
                parent_name = frontier[caller]
                key = (parent_name, target_name, site_file, site_line, edge_type)
                if key not in seen_edges:
                    seen_edges.add(key)
                    citations.append(StructuralCitation(
                        qualified_name=target_name, file=target_file, line_start=start,
                        line_end=end, source_name=source,
                        relation=edge_relation(edge_type),
                        hop=level + 1, call_site_file=site_file,
                        call_site_line=site_line,
                        stop_reason="obligation_reference",
                        parent_qualified_name=parent_name))
                visited.add(target)
                next_frontier[target] = target_name
                if edge_type != "type_ref":
                    continue
                incoming = _shared_reference_callers(
                    conn, target, source=source)
                for (other, other_site, other_line, other_name, other_file,
                     other_start, other_end) in incoming:
                    if (other in visited or _nonproduction_path(other_site)
                            or _nonproduction_path(other_file)):
                        continue
                    other_key = (other_name, target_name, other_site,
                                 other_line, "shared")
                    if other_key in seen_edges:
                        continue
                    seen_edges.add(other_key)
                    citations.append(StructuralCitation(
                        qualified_name=other_name, file=other_file,
                        line_start=other_start, line_end=other_end,
                        source_name=source, relation="shared_reference",
                        hop=level + 1, call_site_file=other_site,
                        call_site_line=other_line,
                        stop_reason="obligation_reference",
                        parent_qualified_name=target_name))
            frontier = next_frontier
            if not frontier:
                break
    return tuple(citations)
def connect_obligation_targets(conn: "Connection", matches, *, source: str,
                               max_depth: int = 4, max_frontier: int = 200):
    """Materialize bounded directed SCIP paths between selected obligation targets."""
    grouped = {}
    for match in matches:
        for obligation, symbol in match.target_symbols:
            grouped.setdefault(obligation, [])
            if symbol not in grouped[obligation]:
                grouped[obligation].append(symbol)
    routes = []
    for symbols in grouped.values():
        if len(symbols) < 2:
            continue
        ids = {}
        for chunk in _chunks(symbols):
            marks = ",".join("?" * len(chunk))
            for canonical, qualified in conn.execute(
                    f"SELECT canonical_id,qualified_name FROM scip_symbols "
                    f"WHERE source_name=? AND qualified_name IN ({marks}) "
                    "AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                    [source, *chunk, "local *"]):
                # Every canonical instance of a qualified name stays a
                # search endpoint until an actual edge resolves identity.
                ids.setdefault(qualified, [])
                if canonical not in ids[qualified]:
                    ids[qualified].append(canonical)
        def directed(start, finish):
            if start not in ids or finish not in ids:
                return []
            targets = set(ids[finish])
            paths = {canonical: [canonical] for canonical in ids[start]}
            visited = set(paths)
            for _depth in range(max(max_depth, 0)):
                frontier = sorted(paths)[:max(max_frontier, 0)]
                if not frontier:
                    break
                next_paths = {}
                for chunk in _chunks(frontier):
                    marks = ",".join("?" * len(chunk))
                    rows = conn.execute(
                        f"SELECT caller_canonical_id,callee_canonical_id FROM scip_edges "
                        f"WHERE caller_canonical_id IN ({marks}) "
                        "AND edge_type IN ('call','type_ref') ORDER BY caller_canonical_id,callee_canonical_id",
                        chunk)
                    for caller, callee in rows:
                        if caller not in paths or callee in visited:
                            continue
                        next_paths.setdefault(callee, [*paths[caller], callee])
                hits = sorted(
                    target for target in targets if target in next_paths)
                if hits:
                    names = _qualified_names(conn, source, next_paths[hits[0]])
                    return [names[item] for item in next_paths[hits[0]]
                            if item in names]
                visited.update(next_paths)
                paths = next_paths
            return []
        root = symbols[0]
        for target in symbols[1:]:
            route = directed(root, target) or directed(target, root)
            if len(route) > 1:
                routes.append(route)
    return tuple(citations_from_qualified_routes(conn, routes, source=source))


def localized_citations(conn: "Connection", symbols, *, source: str) -> tuple[StructuralCitation, ...]:
    """Carry definitions selected by canonical SCIP ID or qualified name."""
    requested = list(dict.fromkeys(
        str(symbol) for symbol in symbols if not _is_local(str(symbol))))
    canonical_ids = set(requested)
    for chunk in _chunks(requested):
        marks = ",".join("?" * len(chunk))
        canonical_ids.update(row[0] for row in conn.execute(
            f"SELECT canonical_id FROM scip_symbols WHERE source_name=? "
            f"AND qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
            [source, *chunk, "local *"]))
    located = _locate(conn, source, sorted(canonical_ids))
    citations = []
    for symbol in sorted(located, key=lambda item: (located[item][1], located[item][2], item)):
        qualified_name, file, line_start, line_end = located[symbol]
        citations.append(StructuralCitation(
            qualified_name=qualified_name, file=file, line_start=line_start,
            source_name=source, relation="localized", hop=0,
            call_site_file=file, call_site_line=line_start,
            stop_reason="question_symbol", line_end=line_end))
    return tuple(citations)
def question_entity_symbol_seeds(conn: "Connection", question: str, *,
                                 source: str, limit: int = 12) -> tuple[str, ...]:
    """Resolve ambiguous code-shaped question entities into diversified SCIP owners."""
    entities = [entity for entity in re.findall(
        r"\b(?:[A-Z][A-Z0-9_]{1,}|[A-Za-z_]+[A-Z][A-Za-z0-9_]*)\b",
        question or "") if set(_compound_tokens(entity)) -
        (_QUESTION_SYMBOL_STOPWORDS | {"not"})]
    if not entities or limit <= 0:
        return ()
    question_tokens = set(_compound_tokens(question)) - _QUESTION_SYMBOL_STOPWORDS
    ranked_by_entity = []
    for entity in dict.fromkeys(entities):
        entity_tokens = set(_compound_tokens(entity))
        candidates = {}
        for row in conn.execute(
                "SELECT canonical_id,qualified_name,file,line_start,kind,parent_qualified_name "
                "FROM scip_symbols WHERE source_name=? AND qualified_name LIKE ? "
                "AND canonical_id NOT GLOB ? LIMIT 5000",
                (source, f"%{entity}%", "local *")):
            if not _nonproduction_path(row[2]):
                candidates[row[0]] = row
        ranked = sorted(candidates.values(), key=lambda row: (
            -len(set(_compound_tokens(row[1])) & entity_tokens),
            -len(set(_compound_tokens(row[1])) & question_tokens),
            0 if row[4] in {"Method", "Function", "Constructor"} else 1,
            len(set(_compound_tokens(row[1]))), row[1], row[2], row[3]))
        family = []
        owners = set()
        for row in ranked:
            owner = row[5] or row[1].rsplit(".", 1)[0]
            if owner in owners:
                continue
            owners.add(owner)
            family.append(row[0])
            if len(family) == limit:
                break
        ranked_by_entity.append(family)
    selected = []
    depth = 0
    while len(selected) < limit and any(depth < len(family) for family in ranked_by_entity):
        for family in ranked_by_entity:
            if depth < len(family) and family[depth] not in selected:
                selected.append(family[depth])
                if len(selected) == limit:
                    break
        depth += 1
    return tuple(selected)
def question_ranked_seeds(conn: "Connection", symbols, question: str, *,
                          source: str, limit: int = 24) -> tuple[str, ...]:
    """Bound document fallback seeds by lexical evidence from SCIP identities.

    Canonical ids are selective and globally unique.  Fetch by that primary key first,
    then apply the source guard in Python; combining the low-cardinality source predicate
    with a large ``IN`` list makes SQLite scan the entire source once per batch.
    """
    ordered = tuple(dict.fromkeys(symbols))
    if not ordered or limit <= 0:
        return ()
    question_tokens = set(_compound_tokens(question)) - _QUESTION_SYMBOL_STOPWORDS
    if not question_tokens:
        return ordered
    rows = []
    for chunk in _chunks(list(ordered)):
        placeholders = ",".join("?" * len(chunk))
        rows.extend(
            (canonical_id, display_name, qualified_name, file)
            for canonical_id, display_name, qualified_name, file, owner
            in conn.execute(
                f"SELECT canonical_id, display_name, qualified_name, file, source_name "
                f"FROM scip_symbols WHERE canonical_id IN ({placeholders})",
                chunk)
            if owner == source)
    scored = []
    lowered = (question or "").lower()
    for canonical_id, display_name, qualified_name, file in rows:
        identity_tokens = set(_compound_tokens(f"{display_name} {qualified_name} {file}"))
        overlap = len(question_tokens & identity_tokens)
        exact = 1 if qualified_name.lower() in lowered else 0
        if overlap or exact:
            partition = (file or "").split("/", 1)[0]
            scored.append((overlap + exact * 4, overlap, qualified_name,
                           canonical_id, partition))
    if not scored:
        return ordered
    scored.sort(key=lambda item: (-item[0], -item[1], item[2], item[3]))
    by_partition = {}
    for item in scored:
        by_partition.setdefault(item[4], []).append(item)
    partitions = sorted(
        by_partition,
        key=lambda name: (-by_partition[name][0][0],
                          -by_partition[name][0][1], name))
    chosen = []
    offset = 0
    while len(chosen) < limit:
        added = False
        for partition in partitions:
            items = by_partition[partition]
            if offset < len(items):
                chosen.append(items[offset])
                added = True
                if len(chosen) == limit:
                    break
        if not added:
            break
        offset += 1
    return tuple(item[3] for item in chosen)
def _nonproduction_path(path: str) -> bool:
    normalized = f"/{(path or '').lower().strip('/')}"
    return any(segment in normalized for segment in (
        "/test/", "/tests/", "/benchmark/", "/benchmarks/",
        "/target/", "/generated/"))
def selected_route_branch_fanout(conn: "Connection", matches, *, source: str,
                                 roots = (), per_root: int = 8,
                                 child_per_sibling: int = 24) -> tuple[StructuralCitation, ...]:
    """Recover nearby same-owner branch alternatives through their direct caller.

    A selected method may be one arm of a compiler-visible decision.  Follow its
    incoming call to the decision owner, retain only sibling callees owned by the
    same implementation, rank them by call-site proximity, then expose their direct
    dependencies.  This discovers alternate execution paths without widening to
    unrelated callers or vocabulary matches.
    """
    names = tuple(dict.fromkeys((
        *(str(name) for name in roots if name),
        *(name for match in matches for name in match.clew.route),
        *(symbol for match in matches
          for _obligation, symbol in match.target_symbols),
    )))
    if not names or per_root <= 0:
        return ()
    selected = {}
    for chunk in _chunks(list(names)):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT canonical_id,qualified_name,file,line_start,line_end,"
                f"parent_qualified_name FROM scip_symbols WHERE source_name=? "
                f"AND qualified_name IN ({marks}) AND canonical_id NOT GLOB ?",
                [source, *chunk, "local *"]):
            if not _nonproduction_path(row[2]):
                selected.setdefault(str(row[0]), tuple(row))
    incoming = []
    for chunk in _chunks(sorted(selected)):
        marks = ",".join("?" * len(chunk))
        incoming.extend(conn.execute(
            f"SELECT caller_canonical_id,callee_canonical_id,file,line,edge_type "
            f"FROM scip_edges INDEXED BY idx_scip_edges_callee "
            f"WHERE callee_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','implements')", chunk))
    caller_ids = tuple(dict.fromkeys(str(row[0]) for row in incoming
                                     if not _is_local(row[0])))
    caller_locations = _locate(conn, source, caller_ids)
    outgoing = []
    for chunk in _chunks(list(caller_locations)):
        marks = ",".join("?" * len(chunk))
        outgoing.extend(conn.execute(
            f"SELECT caller_canonical_id,callee_canonical_id,file,line,edge_type "
            f"FROM scip_edges INDEXED BY idx_scip_edges_caller "
            f"WHERE caller_canonical_id IN ({marks}) "
            "AND edge_type IN ('call','implements')", chunk))
    target_ids = tuple(dict.fromkeys(str(row[1]) for row in outgoing
                                     if not _is_local(row[1])))
    targets = {}
    for chunk in _chunks(list(target_ids)):
        marks = ",".join("?" * len(chunk))
        for row in conn.execute(
                f"SELECT canonical_id,qualified_name,file,line_start,line_end,"
                f"parent_qualified_name FROM scip_symbols WHERE source_name=? "
                f"AND canonical_id IN ({marks}) AND canonical_id NOT GLOB ?",
                [source, *chunk, "local *"]):
            if not _nonproduction_path(row[2]):
                targets[str(row[0])] = tuple(row)
    outgoing_by_caller = {}
    for row in outgoing:
        outgoing_by_caller.setdefault(str(row[0]), []).append(tuple(row))
    citations = []
    sibling_names = []
    seen = set()
    for caller_id, selected_id, selected_site_file, selected_site_line, _edge_type in sorted(
            incoming, key=lambda row: (str(row[1]), str(row[0]), int(row[3]))):
        caller = caller_locations.get(str(caller_id))
        selected_row = selected.get(str(selected_id))
        if caller is None or selected_row is None or _nonproduction_path(selected_site_file):
            continue
        selected_owner = str(selected_row[5] or selected_row[1].rsplit(".", 1)[0])
        caller_name, caller_file, caller_start, caller_end = caller
        caller_key = (caller_name, caller_file, caller_start)
        siblings = []
        for out_row in outgoing_by_caller.get(str(caller_id), ()):
            _out_caller, target_id, site_file, site_line, edge_type = out_row
            target = targets.get(str(target_id))
            if (target is None or str(target_id) == str(selected_id)
                    or _nonproduction_path(site_file)):
                continue
            target_owner = str(target[5] or target[1].rsplit(".", 1)[0])
            if target_owner != selected_owner:
                continue
            siblings.append((abs(int(site_line) - int(selected_site_line)),
                             int(site_line), str(target[1]), out_row, target))
        if not siblings:
            continue
        if caller_key not in seen:
            seen.add(caller_key)
            citations.append(StructuralCitation(
                qualified_name=caller_name, file=caller_file,
                line_start=caller_start, line_end=caller_start,
                source_name=source, relation="localized", hop=0,
                stop_reason="selected_branch_caller",
                call_site_file="", call_site_line=0))
        for _distance, _line, target_name, out_row, target in sorted(siblings)[:per_root]:
            _out_caller, _target_id, site_file, site_line, edge_type = out_row
            _canonical, _qualified, target_file, start, end, _parent = target
            key = (caller_name, target_name, str(edge_type), str(site_file), int(site_line))
            if key in seen:
                continue
            seen.add(key)
            sibling_names.append(target_name)
            citations.append(StructuralCitation(
                qualified_name=target_name, file=str(target_file),
                line_start=int(start), line_end=int(start), source_name=source,
                relation=edge_relation(edge_type),
                hop=1, call_site_file=str(site_file), call_site_line=int(site_line),
                stop_reason="selected_branch_sibling",
                parent_qualified_name=caller_name))
    if sibling_names:
        citations.extend(qualified_call_fanout(
            conn, sibling_names, source=source,
            per_root=child_per_sibling, depth=1,
            recursive_per_root=0, max_recursive_total=0))
    unique = []
    emitted = set()
    for citation in citations:
        key = (citation.qualified_name, citation.parent_qualified_name,
               citation.relation, citation.call_site_file, citation.call_site_line)
        if key not in emitted:
            emitted.add(key)
            unique.append(citation)
    return tuple(unique)
def qualified_same_owner_reference_fanout(
        conn: "Connection", roots, *, source: str, question: str = "",
        per_root: int = 4, excluded_targets = ()) -> tuple[StructuralCitation, ...]:
    # Keep one bounded reference layer whose endpoints share a SCIP owner.
    references = qualified_reference_fanout(
        conn, roots, source=source, question=question,
        per_root=per_root, depth=2,
        recursive_per_root=per_root, max_recursive_total=per_root,
        excluded_targets=excluded_targets)
    names = tuple(dict.fromkeys((
        *(str(root) for root in roots if root),
        *(citation.qualified_name for citation in references))))
    owners = {}
    for chunk in _chunks(list(names)):
        marks = ",".join("?" * len(chunk))
        for qualified_name, parent_name in conn.execute(
                f"SELECT qualified_name,parent_qualified_name "
                f"FROM scip_symbols WHERE source_name=? "
                f"AND qualified_name IN ({marks}) "
                "AND canonical_id NOT GLOB ?",
                [source, *chunk, "local *"]):
            if parent_name:
                owners.setdefault(str(qualified_name), set()).add(
                    str(parent_name))
    return tuple(
        citation for citation in references
        if owners.get(citation.parent_qualified_name, set())
        & owners.get(citation.qualified_name, set()))


@dataclass(frozen=True)
class ObligationExpansion:
    """Evidence and diagnostics from obligation-seeded reverse expansion."""

    citations: tuple = ()
    retained_candidates: tuple = ()
    reserve_candidates: tuple = ()
    truncated_seeds: tuple = ()
    reasons: dict = field(default_factory=dict)


def obligation_seeded_expansion(
        conn: "Connection", matches, *, source: str,
        question_seed_ids=(), catalog_seed_ids=(),
        depth: int = 2, forward_depth: int = 0, per_seed_limit: int = 8,
        reserve_limit: int = 16) -> ObligationExpansion:
    """Bounded reverse entry discovery seeded by what obligations need.

    Seeds are the union of resolved obligation targets, clew-route
    endpoints, and question/catalog symbols — every canonical id per
    qualified name — plus exactly one structural ownership bridge
    (member -> exact owner) so a registrar that references the owning
    type is reachable when the route ends at a member; sibling members
    never enter. Reverse ``call`` edges recover entries (``called_by``),
    reverse ``type_ref`` edges recover registrars (``shared_reference``);
    an incoming reference is never represented as a forward one. Per-seed
    shortlists are caps on citation, not silent deletion: overflow goes
    to a bounded reserve and every cap event is recorded in ``reasons``.
    """
    seed_names: list = []
    for match in matches:
        route = [str(name) for name in getattr(match.clew, "route", ())]
        endpoints = [name for name in (*route[:1], *route[-1:]) if name]
        for name in (*endpoints,
                     *(str(symbol) for _obligation, symbol
                       in match.target_symbols if symbol)):
            if name not in seed_names:
                seed_names.append(name)
    located: dict = {}
    seed_ids: list = []
    owner_names: list = []

    def resolve_seed_names(names, collect_owners):
        for chunk in _chunks([name for name in names if name]):
            marks = ",".join("?" * len(chunk))
            for row in conn.execute(
                    f"SELECT canonical_id, qualified_name, file, line_start, "
                    f"line_end, parent_qualified_name FROM scip_symbols "
                    f"WHERE source_name = ? AND qualified_name IN ({marks}) "
                    f"AND canonical_id NOT GLOB ? ORDER BY canonical_id",
                    [source, *chunk, "local *"]):
                canonical = str(row[0])
                if canonical not in located:
                    seed_ids.append(canonical)
                    located[canonical] = (row[1], row[2], row[3], row[4])
                parent = str(row[5] or "")
                if (collect_owners and parent and parent not in seed_names
                        and parent not in owner_names):
                    owner_names.append(parent)

    resolve_seed_names(seed_names, collect_owners=True)
    resolve_seed_names(owner_names, collect_owners=False)
    for canonical in (*question_seed_ids, *catalog_seed_ids):
        if canonical and str(canonical) not in located:
            seed_ids.append(str(canonical))
    for canonical, location in _locate(
            conn, source,
            [cid for cid in seed_ids if cid not in located]).items():
        located[canonical] = location

    citations: list = []
    retained: list = []
    reserve: list = []
    truncated: list = []
    reasons: dict = {}
    frontier = list(dict.fromkeys(seed_ids))
    visited_callers = set(frontier)
    visited_edges: set = set()
    for level in range(1, max(int(depth), 0) + 1):
        rows = []
        for chunk in _chunks(sorted(frontier)):
            marks = ",".join("?" * len(chunk))
            rows.extend(conn.execute(
                f"SELECT caller_canonical_id, callee_canonical_id, file, "
                f"line, edge_type FROM scip_edges "
                f"WHERE callee_canonical_id IN ({marks}) "
                f"AND edge_type IN ('call', 'type_ref')", chunk))
        by_seed: dict = {}
        for caller, callee, file, line, edge_type in rows:
            if _is_local(caller) or caller == callee:
                continue
            key = (caller, callee, edge_type, file, line)
            if key in visited_edges:
                continue
            visited_edges.add(key)
            by_seed.setdefault(str(callee), []).append(
                (str(file), int(line), str(caller), str(edge_type)))
        caller_locations = _locate(conn, source, sorted({
            caller for pairs in by_seed.values()
            for _file, _line, caller, _edge in pairs}))
        next_frontier: list = []
        for seed in sorted(by_seed):
            seed_location = located.get(seed)
            if seed_location is None:
                continue
            candidates = []
            for file, line, caller, edge_type in sorted(by_seed[seed]):
                location = caller_locations.get(caller)
                if location is None or _nonproduction_path(location[1]):
                    continue
                candidates.append((file, line, caller, edge_type, location))
            kept = candidates[:max(int(per_seed_limit), 0)]
            spill = candidates[len(kept):]
            reserved_here = [
                caller for _file, _line, caller, _edge, _loc in spill
            ][:max(int(reserve_limit) - len(reserve), 0)]
            reserve.extend(reserved_here)
            if spill:
                truncated.append(seed)
            reasons[seed] = {
                "available": len(candidates),
                "retained": len(kept),
                "reserve": len(reserved_here),
                "discarded": len(spill) - len(reserved_here)}
            for file, line, caller, edge_type, location in kept:
                caller_qualified, caller_file, caller_start, caller_end = (
                    location)
                seed_qualified, seed_file, seed_start, seed_end = (
                    seed_location)
                citations.append(StructuralCitation(
                    qualified_name=str(caller_qualified),
                    file=str(caller_file),
                    line_start=int(caller_start), source_name=source,
                    relation=edge_relation("incoming_" + edge_type),
                    hop=level, call_site_file="", call_site_line=0,
                    stop_reason="obligation_entry",
                    line_end=int(caller_end)))
                citations.append(StructuralCitation(
                    qualified_name=str(seed_qualified), file=str(seed_file),
                    line_start=int(seed_start), source_name=source,
                    relation=edge_relation(edge_type), hop=level + 1,
                    call_site_file=file, call_site_line=line,
                    stop_reason="reference", line_end=int(seed_end),
                    parent_qualified_name=str(caller_qualified)))
                retained.append(caller)
                located.setdefault(caller, location)
                if caller not in visited_callers:
                    visited_callers.add(caller)
                    next_frontier.append(caller)
        frontier = next_frontier
        if not frontier:
            break
    if max(int(forward_depth), 0) > 0:
        # Diagnostic/ablation mode only: production defaults
        # keep forward continuation off, and every enabled
        # run says so in its diagnostics.
        reasons["forward:enabled"] = {
            "depth": int(forward_depth)}
    forward_frontier = list(dict.fromkeys(seed_ids))
    visited_forward = set(forward_frontier)
    for level in range(1, max(int(forward_depth), 0) + 1):
        rows = []
        for chunk in _chunks(sorted(forward_frontier)):
            marks = ",".join("?" * len(chunk))
            rows.extend(conn.execute(
                f"SELECT caller_canonical_id, callee_canonical_id, "
                f"file, line, edge_type FROM scip_edges "
                f"WHERE caller_canonical_id IN ({marks}) "
                f"AND edge_type IN ('call', 'type_ref')", chunk))
        by_caller: dict = {}
        for caller, callee, file, line, edge_type in rows:
            if _is_local(callee) or caller == callee:
                continue
            key = (caller, callee, edge_type, file, line, "fwd")
            if key in visited_edges:
                continue
            visited_edges.add(key)
            by_caller.setdefault(str(caller), []).append(
                (str(file), int(line), str(callee), str(edge_type)))
        callee_locations = _locate(conn, source, sorted({
            callee for pairs in by_caller.values()
            for _file, _line, callee, _edge in pairs}))
        next_forward: list = []
        for caller in sorted(by_caller):
            caller_location = located.get(caller)
            if caller_location is None:
                continue
            candidates = []
            for file, line, callee, edge_type in sorted(by_caller[caller]):
                location = callee_locations.get(callee)
                if location is None or _nonproduction_path(location[1]):
                    continue
                candidates.append((file, line, callee, edge_type, location))
            kept = candidates[:max(int(per_seed_limit), 0)]
            spill = candidates[len(kept):]
            if spill:
                truncated.append(caller)
            reasons[f"forward:{caller}"] = {
                "available": len(candidates),
                "retained": len(kept),
                "reserve": 0,
                "discarded": len(spill)}
            caller_qualified, caller_file, caller_start, caller_end = (
                caller_location)
            for file, line, callee, edge_type, location in kept:
                (callee_qualified, callee_file, callee_start,
                 callee_end) = location
                citations.append(StructuralCitation(
                    qualified_name=str(caller_qualified),
                    file=str(caller_file),
                    line_start=int(caller_start), source_name=source,
                    relation=edge_relation(edge_type),
                    hop=level, call_site_file="", call_site_line=0,
                    stop_reason="obligation_continuation",
                    line_end=int(caller_end)))
                citations.append(StructuralCitation(
                    qualified_name=str(callee_qualified),
                    file=str(callee_file),
                    line_start=int(callee_start), source_name=source,
                    relation=edge_relation(edge_type), hop=level + 1,
                    call_site_file=file, call_site_line=line,
                    stop_reason="reference", line_end=int(callee_end),
                    parent_qualified_name=str(caller_qualified)))
                retained.append(callee)
                located.setdefault(callee, location)
                if callee not in visited_forward:
                    visited_forward.add(callee)
                    next_forward.append(callee)
        forward_frontier = next_forward
        if not forward_frontier:
            break
    return ObligationExpansion(
        citations=tuple(citations),
        retained_candidates=tuple(dict.fromkeys(retained)),
        reserve_candidates=tuple(dict.fromkeys(reserve)),
        truncated_seeds=tuple(dict.fromkeys(truncated)),
        reasons=reasons)


def facet_symbol_seeds(conn: "Connection", identifiers, *,
                       source: str, per_identifier: int = 8) -> tuple:
    """Canonical ids matching facet identifiers exactly, by name channels.

    Channels per identifier: exact qualified name, exact display name,
    and exact qualified-name suffix (``.identifier``) — never fuzzy
    matching, so an identifier the question does not contain can never
    seed. Underscores are escaped so snake_case identifiers stay literal
    instead of acting as LIKE wildcards.
    """
    seeds: list = []
    for identifier in identifiers:
        name = str(identifier).strip().strip("`")
        if not name:
            continue
        escaped = (name.replace("\\", "\\\\")
                   .replace("%", "\\%").replace("_", "\\_"))
        rows = conn.execute(
            "SELECT canonical_id FROM scip_symbols WHERE source_name = ? "
            "AND (qualified_name = ? OR display_name = ? "
            "OR qualified_name LIKE ? ESCAPE '\\') "
            "AND canonical_id NOT GLOB ? ORDER BY canonical_id LIMIT ?",
            (source, name, name, f"%.{escaped}", "local *",
             max(int(per_identifier), 0))).fetchall()
        for row in rows:
            if row[0] not in seeds:
                seeds.append(row[0])
    return tuple(seeds)
