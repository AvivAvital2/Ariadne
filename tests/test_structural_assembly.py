"""Structural assembly: the call chain, in execution order, straight from SCIP.

Two findings drive these demands.

**Seeding.** A seed naming a *type* has no outgoing ``call`` edge — its methods do — so a
walk seeded on the type alone terminates at hop 0. That is what recorded the call graph as
contributing 0.0pp in ``designs/answer-path.md`` §5.1. Expansion keys on
``parent_qualified_name`` and never on ``kind``, which scip-python leaves blank on 96–99%
of rows.

**Order.** A chain is a path, not a ranked set. Ordering by call-site ``line`` is what makes
it a sequence: on the live store ``MergeIntoCommand.runMerge``'s line-ordered calls read as
the MERGE algorithm, with ``isInsertOnly`` immediately before the three executors it selects
between. Ranking by anything else destroys that, so nothing here caps or reorders the output.

The corpus reproduces the live defects in miniature: one ``local 1`` row shared between
files, a second source reachable by a raw edge (``scip_edges`` has no ``source_name``), a
callee with no definition at all, a duplicate call site, and a cycle.

Synthetic fixtures only: sources ``src1``/``src2``, packages ``pkg.*``/``pkg2.*``.
"""
from __future__ import annotations

import sqlite3

import pytest

from library.scip import init_scip_schema
from library.structural_assembly import (
    _package_label,
    chain_from,
    expand_to_members,
)

ALPHA_TYPE = 'scip-python python src1 0.1 `pkg.alpha`/Alpha#'
ALPHA_RUN = 'scip-python python src1 0.1 `pkg.alpha`/Alpha#run().'
BETA_START = 'scip-python python src1 0.1 `pkg.beta`/Beta#start().'
GAMMA_FINISH = 'scip-python python src1 0.1 `pkg.gamma`/Gamma#finish().'
ZULU_SPECIFIC = 'scip-python python src1 0.1 `pkg.zulu`/Zulu#specific().'
UTIL_COMMON = 'scip-python python src1 0.1 `pkg.aaa_util`/Util#common().'
OMEGA_DEEP = 'scip-python python src1 0.1 `pkg.omega`/Omega#deep().'
OTHER_THING = 'scip-python python src2 0.1 `pkg2.other`/Other#thing().'
EXTERNAL = 'semanticdb maven org.example lib 1.0 org/example/Lib#absent().'
SHARED_LOCAL = 'local 1'
EXTRA_CALLERS = (
    'scip-python python src1 0.1 `pkg.caller1`/Caller#go().',
    'scip-python python src1 0.1 `pkg.caller2`/Caller#go().',
)
# Util.common ends with 3 callers; every real step here has 1 or 2.
EXPAND_BELOW = 3


def _symbol(conn, cid, *, file, line_start, qn, parent='',
            kind='', display='', source='src1'):
    conn.execute(
        'INSERT INTO scip_symbols (canonical_id, source_name, language, file, '
        'line_start, line_end, kind, display_name, qualified_name, '
        'parent_qualified_name) VALUES (?,?,?,?,?,?,?,?,?,?)',
        (cid, source, 'python', file, line_start, line_start + 2, kind, display,
         qn, parent),
    )


def _edge(conn, caller, callee, *, line, file='pkg/alpha.py', edge_type='call'):
    conn.execute(
        'INSERT INTO scip_edges (caller_canonical_id, callee_canonical_id, '
        'edge_type, file, line, confidence) VALUES (?,?,?,?,?,?)',
        (caller, callee, edge_type, file, line, 'exact'),
    )


@pytest.fixture
def conn():
    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    # `kind` and `display_name` stay BLANK: the scip-python shape, against which a
    # kind-gated expansion would silently no-op.
    _symbol(c, ALPHA_TYPE, file='pkg/alpha.py', line_start=1,
            qn='pkg.alpha.Alpha', parent='pkg.alpha')
    _symbol(c, ALPHA_RUN, file='pkg/alpha.py', line_start=5,
            qn='pkg.alpha.Alpha.run', parent='pkg.alpha.Alpha')
    _symbol(c, UTIL_COMMON, file='pkg/aaa_util.py', line_start=2,
            qn='pkg.aaa_util.Util.common', parent='pkg.aaa_util.Util')
    _symbol(c, BETA_START, file='pkg/beta.py', line_start=3,
            qn='pkg.beta.Beta.start', parent='pkg.beta.Beta')
    _symbol(c, ZULU_SPECIFIC, file='pkg/zulu.py', line_start=8,
            qn='pkg.zulu.Zulu.specific', parent='pkg.zulu.Zulu')
    _symbol(c, GAMMA_FINISH, file='pkg/gamma.py', line_start=9,
            qn='pkg.gamma.Gamma.finish', parent='pkg.gamma.Gamma')
    _symbol(c, OMEGA_DEEP, file='pkg/omega.py', line_start=12,
            qn='pkg.omega.Omega.deep', parent='pkg.omega.Omega')
    _symbol(c, OTHER_THING, file='other/thing.py', line_start=6,
            qn='pkg2.other.Other.thing', parent='pkg2.other.Other',
            source='src2')
    # ONE `local 1` row, as the live store holds one `local 5` row pointed at by
    # edges from 4,446 distinct files.
    _symbol(c, SHARED_LOCAL, file='pkg/alpha.py', line_start=6, qn='local 1')

    # Alpha.run's body — inserted deliberately out of line order.
    _edge(c, ALPHA_RUN, ZULU_SPECIFIC, line=30)
    _edge(c, ALPHA_RUN, UTIL_COMMON, line=10)
    _edge(c, ALPHA_RUN, BETA_START, line=20)
    _edge(c, ALPHA_RUN, ZULU_SPECIFIC, line=35)     # same callee, second call site
    _edge(c, ALPHA_RUN, SHARED_LOCAL, line=45)      # a shared local: never a node
    _edge(c, ALPHA_RUN, OTHER_THING, line=50)       # another source
    _edge(c, ALPHA_RUN, EXTERNAL, line=55)          # no definition anywhere
    # nested bodies
    _edge(c, ZULU_SPECIFIC, GAMMA_FINISH, line=40, file='pkg/zulu.py')
    _edge(c, GAMMA_FINISH, ALPHA_RUN, line=60, file='pkg/gamma.py')   # cycle
    _edge(c, UTIL_COMMON, OMEGA_DEEP, line=50, file='pkg/aaa_util.py')
    _edge(c, SHARED_LOCAL, OMEGA_DEEP, line=7)
    _edge(c, OTHER_THING, OMEGA_DEEP, line=8, file='other/thing.py')
    # two more callers make Util.common ubiquitous
    for n, caller in enumerate(EXTRA_CALLERS, start=1):
        _symbol(c, caller, file=f'pkg/caller{n}.py', line_start=1,
                qn=f'pkg.caller{n}.Caller.go', parent=f'pkg.caller{n}.Caller')
        _edge(c, caller, UTIL_COMMON, line=3, file=f'pkg/caller{n}.py')
    c.commit()
    yield c
    c.close()


def _trace(conn, **kw):
    kw.setdefault('depth', 3)
    kw.setdefault('expand_below_fan_in', EXPAND_BELOW)
    return chain_from(conn, ALPHA_TYPE, source='src1', **kw)


def test_the_chain_follows_call_site_order_and_nests_depth_first(conn):
    """The whole contract in one sequence: line order, and callees inline after them."""
    citations, truncation = _trace(conn)

    assert [(c.qualified_name, c.hop, c.call_site_line) for c in citations] == [
        ('pkg.aaa_util.Util.common', 1, 10),
        ('pkg.beta.Beta.start', 1, 20),
        ('pkg.zulu.Zulu.specific', 1, 30),
        ('pkg.gamma.Gamma.finish', 2, 40),   # Zulu's own call, immediately after it
        ('pkg.alpha.Alpha.run', 3, 60),      # Gamma's call back — the cycle, cited once
        ('pkg.zulu.Zulu.specific', 1, 35),   # only now back to Alpha.run's next line
    ]
    assert not truncation.truncated
    first = citations[0]
    assert (first.file, first.line_start) == ('pkg/aaa_util.py', 2)
    assert (first.call_site_file, first.relation) == ('pkg/alpha.py', 'calls')


def test_a_callee_is_cited_at_every_call_site_but_descended_into_once(conn):
    """Two call sites are two pieces of evidence; the body is only worth walking once."""
    citations, _ = _trace(conn)
    names = [c.qualified_name for c in citations]

    assert names.count('pkg.zulu.Zulu.specific') == 2
    assert names.count('pkg.gamma.Gamma.finish') == 1


def test_plumbing_is_cited_but_never_descended_into(conn):
    """Dropping ubiquitous callees would lose branch conditions, so cite and stop."""
    citations, _ = _trace(conn)
    names = [c.qualified_name for c in citations]

    assert 'pkg.aaa_util.Util.common' in names
    assert 'pkg.omega.Omega.deep' not in names


def test_a_shared_local_binding_is_never_cited_or_traversed(conn):
    """`local 1` is one row shared across files — a path through it is an artefact."""
    citations, _ = _trace(conn, depth=5)
    names = [c.qualified_name for c in citations]

    assert 'local 1' not in names
    assert 'pkg.omega.Omega.deep' not in names


def test_a_foreign_source_is_neither_cited_nor_traversed(conn):
    """`Other.thing` calls back into src1; a guard applied only at citation would leak."""
    citations, _ = _trace(conn, depth=5)
    names = [c.qualified_name for c in citations]

    assert 'pkg2.other.Other.thing' not in names
    assert 'pkg.omega.Omega.deep' not in names


def test_a_callee_with_no_definition_is_skipped(conn):
    """An external moniker resolves nowhere in the corpus and cannot be quoted."""
    citations, _ = _trace(conn)

    assert all('Lib' not in c.qualified_name for c in citations)


def test_depth_stops_the_descent_and_reports_what_was_deferred(conn):
    """Nothing silent: the call sites left unexpanded come back as a count."""
    citations, truncation = _trace(conn, depth=1)

    assert [(c.qualified_name, c.hop) for c in citations] == [
        ('pkg.aaa_util.Util.common', 1),
        ('pkg.beta.Beta.start', 1),
        ('pkg.zulu.Zulu.specific', 1),
        ('pkg.zulu.Zulu.specific', 1),
    ]
    assert truncation.truncated
    assert truncation.dropped >= 1
    assert truncation.reason == 'depth'


def test_expansion_keys_on_the_parent_link_not_on_kind(conn):
    """`kind` is blank throughout this corpus; the member must still be found."""
    expanded = expand_to_members(conn, 'src1', [ALPHA_TYPE])

    assert {ALPHA_TYPE, ALPHA_RUN} <= expanded
    assert SHARED_LOCAL not in expanded


def test_an_unresolvable_seed_is_reported_not_silently_empty(conn):
    """A seed absent from this source is named, not swallowed."""
    citations, truncation = chain_from(
        conn, 'scip-python python src1 0.1 `pkg.absent`/Missing#', source='src1')

    assert citations == []
    assert truncation.truncated
    assert truncation.unresolved_seeds == (
        'scip-python python src1 0.1 `pkg.absent`/Missing#',)


def test_each_hop_records_why_the_walk_stopped_there(conn):
    """The traversal's own judgement travels with the citation, for curation to read."""
    citations, _ = _trace(conn)
    reasons = [(c.qualified_name.split('.')[-1], c.stop_reason) for c in citations]

    assert reasons == [
        ('common', 'plumbing'),   # fan-in at the boundary: cited, not expanded
        ('start', 'leaf'),        # its body calls nothing in-corpus
        ('specific', 'descended'),
        ('finish', 'descended'),
        ('run', 'revisit'),       # the cycle: already expanded
        ('specific', 'revisit'),  # second call site, body already walked
    ]


def test_a_hop_stopped_by_the_depth_limit_says_so(conn):
    """At the limit the body was never queried, so `depth` is honest where `leaf` is not."""
    citations, truncation = _trace(conn, depth=1)

    assert [c.stop_reason for c in citations] == [
        'plumbing', 'depth', 'depth', 'depth',
    ]
    assert truncation.reason == 'depth'


class TestManySeedsAreOneWalk:
    """Walking N seeds must cost the graph reached, not N x the graph.

    Measured on the rebuilt databricks store: 8 retrieved documents yield **195 seeds**,
    and walking them one at a time took ~67s for a single `ask` — each seed re-expanded
    members and re-walked ground the previous seeds had already covered. Shared state makes
    the cost proportional to what is reached.
    """

    def test_a_body_reachable_from_two_seeds_is_walked_once(self, conn):
        from library.structural_assembly import chain_from_seeds

        # Both ALPHA_RUN and ZULU_SPECIFIC reach GAMMA_FINISH's body.
        citations, _ = chain_from_seeds(
            conn, [ALPHA_TYPE, ZULU_SPECIFIC], source='src1', depth=3,
            expand_below_fan_in=EXPAND_BELOW)
        descended = [c for c in citations if c.stop_reason == 'descended']

        assert len(descended) == len({c.qualified_name for c in descended}), (
            'a body expanded twice means the seeds did not share state')

    def test_the_result_is_the_union_and_still_in_call_site_order(self, conn):
        from library.structural_assembly import chain_from_seeds

        one, _ = chain_from_seeds(conn, [ALPHA_TYPE], source='src1', depth=3,
                                  expand_below_fan_in=EXPAND_BELOW)
        both, _ = chain_from_seeds(conn, [ALPHA_TYPE, GAMMA_FINISH], source='src1',
                                   depth=3, expand_below_fan_in=EXPAND_BELOW)

        assert {c.qualified_name for c in one} <= {c.qualified_name for c in both}
        # within each root's trace, call sites still ascend
        assert both == sorted(both, key=lambda c: both.index(c))

    def test_an_unresolvable_seed_among_good_ones_does_not_lose_the_others(self, conn):
        from library.structural_assembly import chain_from_seeds

        citations, truncation = chain_from_seeds(
            conn, ['scip-python python src1 0.1 `pkg.absent`/Missing#', ALPHA_TYPE],
            source='src1', depth=2, expand_below_fan_in=EXPAND_BELOW)

        assert citations, 'the good seed must still produce a chain'
        assert truncation.unresolved_seeds == (
            'scip-python python src1 0.1 `pkg.absent`/Missing#',)

    def test_chain_from_is_the_single_seed_case(self, conn):
        from library.structural_assembly import chain_from, chain_from_seeds

        single, _ = chain_from(conn, ALPHA_TYPE, source='src1', depth=3,
                              expand_below_fan_in=EXPAND_BELOW)
        via_seeds, _ = chain_from_seeds(conn, [ALPHA_TYPE], source='src1', depth=3,
                                        expand_below_fan_in=EXPAND_BELOW)

        assert single == via_seeds


class TestDispatchIsFollowed:
    """`implements` answers *which implementation runs* — the reason it was ingested.

    98,976 of these edges exist in the rebuilt store and the walk read none of them: it
    filtered `edge_type = 'call'`, so a hop landing on an abstract method was a dead end
    labelled `leaf`. That is the whole question at a polymorphic call site, and for
    `runMerge` it is the difference between naming `ClassicMergeExecutor` and stopping at
    the trait.
    """

    @pytest.fixture
    def dispatch(self, conn):
        iface = 'scip-python python src1 0.1 `pkg.proto`/Proto#run().'
        impl_a = 'scip-python python src1 0.1 `pkg.a`/AImpl#run().'
        impl_b = 'scip-python python src1 0.1 `pkg.b`/BImpl#run().'
        inner = 'scip-python python src1 0.1 `pkg.a`/AImpl#step().'
        _symbol(conn, iface, file='pkg/proto.py', line_start=4, qn='pkg.proto.Proto.run',
                parent='pkg.proto.Proto')
        _symbol(conn, impl_a, file='pkg/a.py', line_start=10, qn='pkg.a.AImpl.run',
                parent='pkg.a.AImpl')
        _symbol(conn, impl_b, file='pkg/b.py', line_start=20, qn='pkg.b.BImpl.run',
                parent='pkg.b.BImpl')
        _symbol(conn, inner, file='pkg/a.py', line_start=30, qn='pkg.a.AImpl.step',
                parent='pkg.a.AImpl')
        # Alpha.run calls the ABSTRACT method; the implementors are what actually run.
        _edge(conn, ALPHA_RUN, iface, line=12)
        _edge(conn, impl_a, iface, line=10, file='pkg/a.py', edge_type='implements')
        _edge(conn, impl_b, iface, line=20, file='pkg/b.py', edge_type='implements')
        _edge(conn, impl_a, inner, line=11, file='pkg/a.py')
        conn.commit()
        return conn, iface

    def test_an_abstract_hop_reaches_its_implementors(self, dispatch):
        conn, iface = dispatch
        citations, _ = _trace(conn)
        names = [c.qualified_name for c in citations]

        assert 'pkg.proto.Proto.run' in names, 'the abstract call site is still a hop'
        assert 'pkg.a.AImpl.run' in names and 'pkg.b.BImpl.run' in names, (
            'both implementations are what could actually run')

    def test_an_implementor_hop_is_typed_as_dispatch_not_as_a_call(self, dispatch):
        conn, _ = dispatch
        citations, _ = _trace(conn)
        impl = next(c for c in citations if c.qualified_name == 'pkg.a.AImpl.run')

        assert impl.relation == 'implemented_by'

    def test_the_walk_continues_through_the_implementation(self, dispatch):
        conn, _ = dispatch
        citations, _ = _trace(conn, depth=4)

        assert 'pkg.a.AImpl.step' in [c.qualified_name for c in citations], (
            'the body that actually runs is the point of following dispatch')

    def test_a_leaf_with_no_implementors_is_still_a_leaf(self, conn):
        citations, _ = _trace(conn)
        beta = next(c for c in citations if c.qualified_name == 'pkg.beta.Beta.start')

        assert beta.stop_reason == 'leaf'
        assert beta.relation == 'calls'
class TestDispatchFollowsAnOverriddenMethodToo:
    """An override is not only an abstract method's business.

    ``dispatch()`` ran only for a hop the walk labelled ``leaf`` -- a body that calls
    nothing in-corpus. A method that HAS a body and is ALSO overridden descends into the
    base implementation, and the chain never names the override. That is the
    template-method and overridden-hook shape, and it is common: measured on the rebuilt
    store, 7,176 of 53,823 descending hops (13.3%) have implementors that were never shown.

    Following them is cheap -- median 1 implementor, p90 2, mean 1.65 -- but the tail is
    not (max 529 on the same store), so a cap applies and is REPORTED rather than
    silently swallowing the rest.
    """

    @pytest.fixture
    def overridden(self, conn):
        base = 'scip-python python src1 0.1 `pkg.base`/Base#handle().'
        override = 'scip-python python src1 0.1 `pkg.sub`/Sub#handle().'
        base_inner = 'scip-python python src1 0.1 `pkg.base`/Base#prepare().'
        _symbol(conn, base, file='pkg/base.py', line_start=4,
                qn='pkg.base.Base.handle', parent='pkg.base.Base')
        _symbol(conn, override, file='pkg/sub.py', line_start=10,
                qn='pkg.sub.Sub.handle', parent='pkg.sub.Sub')
        _symbol(conn, base_inner, file='pkg/base.py', line_start=40,
                qn='pkg.base.Base.prepare', parent='pkg.base.Base')
        _edge(conn, ALPHA_RUN, base, line=12)
        # The base HAS a body, so the walk descends and used to stop considering overrides.
        _edge(conn, base, base_inner, line=5, file='pkg/base.py')
        _edge(conn, override, base, line=10, file='pkg/sub.py', edge_type='implements')
        conn.commit()
        return conn

    def test_an_overridden_method_with_a_body_still_names_its_override(self, overridden):
        citations, _ = _trace(overridden, depth=4)
        names = [c.qualified_name for c in citations]

        assert 'pkg.base.Base.prepare' in names, 'the base body is still walked'
        assert 'pkg.sub.Sub.handle' in names, (
            'the override is what runs for a Sub, so the chain must name it')
        override = next(c for c in citations
                        if c.qualified_name == 'pkg.sub.Sub.handle')
        assert override.relation == 'implemented_by'
    @pytest.fixture
    def wide(self, conn):
        """One interface with ten implementors — right at the threshold."""
        iface = 'scip-python python src1 0.1 `pkg.wide`/Wide#run().'
        _symbol(conn, iface, file='pkg/wide.py', line_start=4,
                qn='pkg.wide.Wide.run', parent='pkg.wide.Wide')
        _edge(conn, ALPHA_RUN, iface, line=12)
        for index in range(10):
            impl = f'scip-python python src1 0.1 `pkg.i{index}`/I{index}#run().'
            _symbol(conn, impl, file=f'pkg/i{index}.py', line_start=5,
                    qn=f'pkg.i{index}.I{index}.run', parent=f'pkg.i{index}.I{index}')
            _edge(conn, impl, iface, line=5, file=f'pkg/i{index}.py',
                  edge_type='implements')
        conn.commit()
        return conn
    def test_a_wide_dispatch_cites_every_implementor(self, wide):
        """Ten implementors are all cited: this is the boundary, not a cap.

        A count budget on the walk is still the error this module's docstring rejects, and
        how many implementations an interface has is part of the answer to which one runs.
        What changed is only the far tail: above ``DEFAULT_DISCLOSE_ABOVE`` the walk reports
        the fan-out and its shape instead of expanding it, so the caller can narrow (see
        ``test_a_dispatch_beyond_the_threshold_is_reported_not_expanded``). Every dispatch at
        or below the threshold takes this path -- every implementor cited, nothing dropped.
        """
        citations, _ = _trace(wide, depth=4)
        implementors = [c for c in citations if c.relation == 'implemented_by']

        assert {c.qualified_name for c in implementors} == {
            f'pkg.i{index}.I{index}.run' for index in range(10)}
    @pytest.fixture
    def wider(self, conn):
        """Twelve implementors — past the threshold — split across two packages."""
        iface = 'scip-python python src1 0.1 `pkg.wide`/Wide#run().'
        _symbol(conn, iface, file='pkg/wide.py', line_start=4,
                qn='pkg.wide.Wide.run', parent='pkg.wide.Wide')
        _edge(conn, ALPHA_RUN, iface, line=12)
        for index in range(12):
            area = 'main' if index < 8 else 'tests'
            impl = f'scip-python python src1 0.1 `pkg.{area}.i{index}`/I{index}#run().'
            _symbol(conn, impl, file=f'pkg/{area}/i{index}.py', line_start=5,
                    qn=f'pkg.{area}.i{index}.I{index}.run',
                    parent=f'pkg.{area}.i{index}.I{index}')
            _edge(conn, impl, iface, line=5, file=f'pkg/{area}/i{index}.py',
                  edge_type='implements')
        conn.commit()
        return conn

    def test_a_dispatch_beyond_the_threshold_is_reported_not_expanded(self, wider):
        """Hundreds of answers is not an answer. The count is the finding; the caller narrows.

        Nothing is dropped silently -- that is the whole difference from a cap. The number
        is stated, the shape travels so the caller has something to narrow *by*, and the
        interface where the chain forks is still cited so the chain does not just end.
        """
        citations, truncation = _trace(wider, depth=4)

        assert not [c for c in citations if c.relation == 'implemented_by'], (
            'a fan-out this wide is described, not walked')
        assert 'pkg.wide.Wide.run' in [c.qualified_name for c in citations], (
            'the interface where the chain forks is still a hop')
        assert len(truncation.fan_outs) == 1
        fan_out = truncation.fan_outs[0]
        assert fan_out.qualified_name == 'pkg.wide.Wide.run'
        assert fan_out.implementations == 12
        assert dict(fan_out.by_package)['pkg.tests'] == 4, (
            'the breakdown is what makes the question answerable')
    def test_an_implementation_names_the_interface_it_was_reached_through(self, wide):
        """`implements` is followed from a hop, so that hop is the parent."""
        citations, _ = _trace(wide, depth=4)
        implementors = [c for c in citations if c.relation == 'implemented_by']

        assert implementors, 'fixture must produce implementors'
        assert {c.parent_qualified_name for c in implementors} == {'pkg.wide.Wide.run'}
class TestTypeReferencesAreCitedNotTraversed:
    """`type_ref` is 69% of the graph and stage one read none of it.

    1,810,941 edges against 777,182 calls. A body's type and field references are what it
    *touches* -- the types it constructs, the fields it reads -- and an answer about what a
    function does is poorer without them. 42% of them point at a local and are not
    addressable; the rest resolve to real definitions, 576 of them from the 151 symbols of
    the `runMerge` chain alone.

    They are hops like any other: cited in call-site line order so the sequence survives,
    and typed `references` so curation can tell touching a type from calling a method.

    What they are **not** is a step the walk continues through. Touching a type is not
    executing it. Measured at production width (8 documents, source `databricks`, depth 3),
    traversing them made references 15,216 of 25,313 hops and turned the call chain into a
    sweep of the type neighbourhood: a question about MERGE came out with its widest call
    sites in correlation statistics. A `call` edge reaches what runs next; a type reached
    onward reaches whatever that type happens to touch, which in a Scala corpus is most of
    it.
    """

    @pytest.fixture
    def touching(self, conn):
        dataset = 'scip-python python src1 0.1 `pkg.data`/Dataset#'
        _symbol(conn, dataset, file='pkg/data.py', line_start=3,
                qn='pkg.data.Dataset', parent='pkg.data')
        # Alpha.run touches Dataset, between the two calls its body already makes.
        _edge(conn, ALPHA_RUN, dataset, line=15, edge_type='type_ref')
        conn.commit()
        return conn

    def test_a_type_reference_is_a_hop(self, touching):
        citations, _ = _trace(touching)

        assert 'pkg.data.Dataset' in [c.qualified_name for c in citations], (
            'a type the body touches is part of what the body does')

    def test_a_type_reference_is_typed_as_a_reference_not_a_call(self, touching):
        """The distinction is the point: synthesis must not report a mention as a call."""
        citations, _ = _trace(touching)
        hop = next(c for c in citations if c.qualified_name == 'pkg.data.Dataset')

        assert hop.relation == 'references'

    def test_reference_hops_keep_call_site_line_order(self, touching):
        """Order is the sequence. A reference at line 15 sits where line 15 belongs."""
        citations, _ = _trace(touching)
        hop1 = [(c.qualified_name, c.call_site_line) for c in citations if c.hop == 1]

        assert hop1 == sorted(hop1, key=lambda pair: pair[1])
        assert ('pkg.data.Dataset', 15) in hop1

    def test_a_reference_to_a_local_is_not_a_hop(self, conn):
        """42% of `type_ref` edges point at a local, which is not addressable."""
        _edge(conn, ALPHA_RUN, SHARED_LOCAL, line=16, edge_type='type_ref')
        conn.commit()

        citations, _ = _trace(conn)

        assert not [c for c in citations if 'local' in c.qualified_name]
    def test_a_type_reference_does_not_extend_the_walk(self, touching):
        """The chain stops at the type it touches, and says so with `reference`."""
        beyond = 'scip-python python src1 0.1 `pkg.data`/Dataset#load().'
        _symbol(touching, beyond, file='pkg/data.py', line_start=8,
                qn='pkg.data.Dataset.load', parent='pkg.data.Dataset')
        _edge(touching, 'scip-python python src1 0.1 `pkg.data`/Dataset#', beyond,
              line=9, file='pkg/data.py')
        touching.commit()

        citations, _ = _trace(touching)
        names = [c.qualified_name for c in citations]

        assert 'pkg.data.Dataset' in names, 'the type it touches is still cited'
        assert 'pkg.data.Dataset.load' not in names, (
            'what a referenced type calls is not part of this chain')
        hop = next(c for c in citations if c.qualified_name == 'pkg.data.Dataset')
        assert hop.stop_reason == 'reference'

    def test_a_referenced_interface_does_not_dispatch_to_its_implementors(self, touching):
        """`implements` answers which implementation *runs*. A reference runs nothing.

        This is the sharper half: a type reference to an interface previously fanned out to
        every implementor it had — measured at 529 for one live interface — none of which
        is executing on account of a type being named.
        """
        iface = 'scip-python python src1 0.1 `pkg.data`/Dataset#'
        impl = 'scip-python python src1 0.1 `pkg.impl`/CsvDataset#'
        _symbol(touching, impl, file='pkg/impl.py', line_start=4,
                qn='pkg.impl.CsvDataset', parent='pkg.impl')
        _edge(touching, impl, iface, line=4, file='pkg/impl.py',
              edge_type='implements')
        touching.commit()

        citations, _ = _trace(touching)

        assert 'pkg.impl.CsvDataset' not in [c.qualified_name for c in citations], (
            'a named type does not dispatch — nothing is running')
def test_a_citation_carries_the_body_extent_not_just_its_start(conn):
    """Stage two quotes a hop, which needs both ends of the definition.

    ``_locate`` selected ``line_start`` alone, so the extent the ingest rebuild exists to
    reconstruct — guarded by ``definition_extents_present`` — could not reach the answer
    path at all. A citation that cannot say where a definition ends cannot be quoted from.
    """
    citations, _ = _trace(conn)
    alpha = next(c for c in citations
                 if c.qualified_name == 'pkg.aaa_util.Util.common')

    assert alpha.line_end >= alpha.line_start
    assert alpha.line_end == 4, (
        f'the fixture gives Util.common lines 2-4; got {alpha.line_end}')
def test_a_package_label_drops_the_build_layout():
    """`org.apache.spark.sql.catalyst.expressions`, not `sql.catalyst.src.main.scala.org...`.

    A caller narrowing a fan-out names a package. The directory as stored carries the module
    and the JVM build layout on top of the package, which is noise in a sentence asking
    someone to choose an area.
    """
    jvm = ('sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/'
           'expressions/Add.scala')

    assert _package_label(jvm) == 'org.apache.spark.sql.catalyst.expressions'
    assert _package_label('sql/core/src/test/scala/org/apache/spark/sql/X.scala') == (
        'org.apache.spark.sql'), 'a test tree carries the same package'
    assert _package_label('pkg/tests/i8.py') == 'pkg.tests', (
        'a layout without src/main falls back to the directory itself')
class TestAHopRecordsWhereItCameFrom:
    """A chain is a path, so each hop must know which body reached it.

    Without this the citation list is a set with depth numbers: reconstructing the parent
    afterwards means matching call-site coordinates back to definitions and guessing when a
    file has several, which is what an earlier measurement had to do. The walk holds the
    caller in its hand at the moment it follows the edge — recording it there is exact and
    free, and it is what lets a path to a terminal be computed at all.
    """

    def test_a_hop_names_the_body_that_reached_it(self, conn):
        citations, _ = _trace(conn)
        by_name = {c.qualified_name: c for c in citations}

        assert by_name['pkg.beta.Beta.start'].parent_qualified_name == 'pkg.alpha.Alpha.run'

    def test_a_first_ring_hop_names_the_seed_it_expanded_from(self, conn):
        citations, _ = _trace(conn)
        first_ring = [c for c in citations if c.hop == 1]

        assert all(c.parent_qualified_name for c in first_ring), (
            'every hop has a parent; a root ring came from the root')

