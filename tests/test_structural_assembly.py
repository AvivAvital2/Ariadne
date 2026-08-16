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
caller_roots)

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
        from library.structural_assembly import chain_from_seeds, caller_roots

        # Both ALPHA_RUN and ZULU_SPECIFIC reach GAMMA_FINISH's body.
        citations, _ = chain_from_seeds(
            conn, [ALPHA_TYPE, ZULU_SPECIFIC], source='src1', depth=3,
            expand_below_fan_in=EXPAND_BELOW)
        descended = [c for c in citations if c.stop_reason == 'descended']

        assert len(descended) == len({c.qualified_name for c in descended}), (
            'a body expanded twice means the seeds did not share state')

    def test_the_result_is_the_union_and_still_in_call_site_order(self, conn):
        from library.structural_assembly import chain_from_seeds, caller_roots

        one, _ = chain_from_seeds(conn, [ALPHA_TYPE], source='src1', depth=3,
                                  expand_below_fan_in=EXPAND_BELOW)
        both, _ = chain_from_seeds(conn, [ALPHA_TYPE, GAMMA_FINISH], source='src1',
                                   depth=3, expand_below_fan_in=EXPAND_BELOW)

        assert {c.qualified_name for c in one} <= {c.qualified_name for c in both}
        # within each root's trace, call sites still ascend
        assert both == sorted(both, key=lambda c: both.index(c))

    def test_an_unresolvable_seed_among_good_ones_does_not_lose_the_others(self, conn):
        from library.structural_assembly import chain_from_seeds, caller_roots

        citations, truncation = chain_from_seeds(
            conn, ['scip-python python src1 0.1 `pkg.absent`/Missing#', ALPHA_TYPE],
            source='src1', depth=2, expand_below_fan_in=EXPAND_BELOW)

        assert citations, 'the good seed must still produce a chain'
        assert truncation.unresolved_seeds == (
            'scip-python python src1 0.1 `pkg.absent`/Missing#',)

    def test_chain_from_is_the_single_seed_case(self, conn):
        from library.structural_assembly import chain_from, chain_from_seeds, caller_roots

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
class TestCallerExpansion:
    def test_downstream_seed_recovers_connected_outer_root_and_accounts_for_gate(self, conn):
        expansion = caller_roots(
            conn, [GAMMA_FINISH], source='src1', depth=2,
            fan_in_max=EXPAND_BELOW)

        assert expansion.roots == (ALPHA_RUN,)
        assert [citation.qualified_name for citation in expansion.citations] == [
            'pkg.alpha.Alpha.run']
        assert expansion.citations[0].relation == 'called_by'
        assert expansion.citations[0].call_site_line == 30

        gated = caller_roots(
            conn, [UTIL_COMMON], source='src1', depth=1,
            fan_in_max=EXPAND_BELOW)
        assert gated.roots == ()
        assert gated.gated_targets == (UTIL_COMMON,)
        exhausted = caller_roots(
            conn, [GAMMA_FINISH], source='src1', depth=3,
            fan_in_max=EXPAND_BELOW)
        assert exhausted.roots == (ALPHA_RUN,)
        assert caller_roots(
            conn, [GAMMA_FINISH], source='src1', depth=0,
            fan_in_max=EXPAND_BELOW).roots == ()
        scoped = caller_roots(
            conn, [OMEGA_DEEP], source='src1', depth=1,
            fan_in_max=10)
        assert scoped.roots == (UTIL_COMMON,)
        assert all(citation.source_name == 'src1' for citation in scoped.citations)
class TestQuestionSymbolLocalization:
    def test_compound_identifier_beats_a_short_product_name_collision(self, conn):
        from library.structural_assembly import question_symbol_seeds

        _symbol(conn, "delta-writer", file="spark/DeltaWriter.java",
                qn="org.spark.connector.DeltaWriter", line_start=1, display = 'DeltaWriter', kind = 'Object')
        _symbol(conn, "delta-file-writer", file="delta/DeltaFileFormatWriter.scala",
                qn="org.delta.files.DeltaFileFormatWriter", line_start=1, display = 'DeltaFileFormatWriter', kind = 'Object')

        seeds = question_symbol_seeds(
            conn,
            "Delta uses forked file-writing machinery instead of Spark stock writer",
            source="src1")

        assert seeds == ("delta-file-writer",)
        from library.structural_assembly import localized_citations
        localized = localized_citations(conn, seeds, source="src1")
        assert [citation.qualified_name for citation in localized] == [
            "org.delta.files.DeltaFileFormatWriter"]


class TestSharedReferenceBridge:
    def test_a_shared_constant_joins_its_getter_to_its_setter(self, conn):
        from library.structural_assembly import StructuralCitation, reference_bridges

        _symbol(conn, "query-key", file="api/StreamExecution.scala",
                qn="stream.StreamExecution.QUERY_ID_KEY", line_start=70, display = 'QUERY_ID_KEY')
        _symbol(conn, "setter", file="spark/StreamExecution.scala",
                qn="stream.StreamExecution.runStream", line_start=250, display = 'runStream')
        _symbol(conn, "getter", file="delta/DeltaSink.scala",
                qn="delta.DeltaSink.queryId", line_start=60, display = 'queryId')
        conn.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            ("setter", "query-key", "type_ref", "spark/StreamExecution.scala", 292, "exact"))
        conn.execute(
            "INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
            ("getter", "query-key", "type_ref", "delta/DeltaSink.scala", 66, "exact"))
        target = StructuralCitation(
            qualified_name="stream.StreamExecution.QUERY_ID_KEY",
            file="api/StreamExecution.scala", line_start=70, source_name="src1",
            relation="references", hop=2, call_site_file="delta/DeltaSink.scala",
            call_site_line=66, stop_reason="reference")

        bridge = reference_bridges(conn, [target], source="src1", question = 'stream identifier survives restart')

        assert [citation.qualified_name for citation in bridge.citations] == [
            "delta.DeltaSink.queryId", "stream.StreamExecution.runStream"]
        assert {citation.call_site_line for citation in bridge.citations} == {66, 292}
        assert all(citation.relation == "shared_reference" for citation in bridge.citations)
        assert len(reference_bridges(conn, [target], source="src1", question="stream key plumbing").citations) == 2
def test_fork_localization_pairs_the_same_member_on_the_stock_scip_type(conn):
    from library.structural_assembly import question_symbol_seeds
    _symbol(conn, "delta-file-writer", file="delta/DeltaFileFormatWriter.scala",
            qn="org.delta.files.DeltaFileFormatWriter", line_start=1,
            display="DeltaFileFormatWriter", kind="Object")
    _symbol(conn, "delta-file-write", file="delta/DeltaFileFormatWriter.scala",
            qn="org.delta.files.DeltaFileFormatWriter.write", line_start=10,
            display="write", parent="org.delta.files.DeltaFileFormatWriter")
    _symbol(conn, "stock-file-writer", file="spark/FileFormatWriter.scala",
            qn="org.spark.files.FileFormatWriter", line_start=1,
            display="FileFormatWriter", kind="Object")
    _symbol(conn, "stock-file-write", file="spark/FileFormatWriter.scala",
            qn="org.spark.files.FileFormatWriter.write", line_start=10,
            display="write", parent="org.spark.files.FileFormatWriter")

    seeds = question_symbol_seeds(
        conn, "Delta uses forked file-writing machinery instead of Spark stock writer",
        source="src1")

    assert seeds == ("delta-file-write", "stock-file-write")
def test_question_ranked_seeds_bound_a_broad_document_fallback(conn):
    from library.structural_assembly import question_ranked_seeds
    relevant = "scip-python python src1 0.1 `delta/ScanWithDeletionVectors`#"
    _symbol(conn, relevant, file="delta/Scan.scala",
            qn="delta.ScanWithDeletionVectors", line_start=1,
            display="ScanWithDeletionVectors", kind="Object")
    noise = []
    for number in range(40):
        canonical = f"scip-python python src1 0.1 `unrelated{number}/Utility`#"
        noise.append(canonical)
        _symbol(conn, canonical, file=f"noise/{number}.scala",
                qn=f"unrelated{number}.Utility", line_start=1,
                display="Utility", kind="Object")

    ranked = question_ranked_seeds(
        conn, [relevant, *noise],
        "How does the deletion-vector scan filter deleted rows?",
        source="src1", limit=8)

    assert ranked == (relevant,)
    assert question_ranked_seeds(conn, noise, "why?", source="src1") == tuple(noise)
def test_question_ranked_seeds_preserve_relevant_source_partitions(conn):
    from library.structural_assembly import question_ranked_seeds

    delta_best = "delta-best"
    delta_second = "delta-second"
    delta_third = "delta-third"
    spark_best = "spark-best"
    spark_second = "spark-second"
    _symbol(conn, delta_best, file="spark/src/DeltaMerge.scala",
            qn="delta.DeltaMergeRewrite", line_start=1,
            display="DeltaMergeRewrite", kind="Object")
    _symbol(conn, delta_second, file="spark/src/DeltaMergeHelper.scala",
            qn="delta.DeltaMergeHelper", line_start=1,
            display="DeltaMergeHelper", kind="Object")
    _symbol(conn, delta_third, file="spark/src/DeltaMergePlan.scala",
            qn="delta.DeltaMergePlan", line_start=1,
            display="DeltaMergePlan", kind="Object")
    _symbol(conn, spark_best, file="sql/core/MergeRewrite.scala",
            qn="spark.MergeRewrite", line_start=1,
            display="MergeRewrite", kind="Object")
    _symbol(conn, spark_second, file="sql/core/MergePlan.scala",
            qn="spark.MergePlan", line_start=1,
            display="MergePlan", kind="Object")

    ranked = question_ranked_seeds(
        conn, [delta_best, delta_second, delta_third, spark_best, spark_second],
        "Why does Delta MERGE use a different rewrite than Spark MERGE?",
        source="src1", limit=4)

    assert delta_best in ranked
    assert delta_second in ranked
    assert spark_best in ranked
    assert spark_second in ranked
def test_reference_bridge_excludes_test_and_suite_callers(conn):
    from library.structural_assembly import StructuralCitation, reference_bridges

    _symbol(conn, "stable-key", file="api/StreamExecution.scala",
            qn="stream.StreamExecution.QUERY_ID_KEY", line_start=10,
            display="QUERY_ID_KEY")
    _symbol(conn, "producer", file="spark/StreamExecution.scala",
            qn="stream.StreamExecution.runStream", line_start=20,
            display="runStream")
    _symbol(conn, "consumer", file="delta/DeltaSink.scala",
            qn="delta.DeltaSink.queryId", line_start=30, display="queryId")
    _symbol(conn, "test-caller", file="delta/src/test/DeltaSinkSuite.scala",
            qn="delta.DeltaSinkSuite.queryId", line_start=40,
            display="queryId")
    for caller, file, line in (
            ("producer", "spark/StreamExecution.scala", 25),
            ("consumer", "delta/DeltaSink.scala", 35),
            ("test-caller", "delta/src/test/DeltaSinkSuite.scala", 45)):
        conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                     (caller, "stable-key", "type_ref", file, line, "exact"))
    target = StructuralCitation(
        qualified_name="stream.StreamExecution.QUERY_ID_KEY",
        file="api/StreamExecution.scala", line_start=10, source_name="src1",
        relation="references", hop=1, call_site_file="delta/DeltaSink.scala",
        call_site_line=35, stop_reason="reference")

    bridged = reference_bridges(
        conn, [target], source="src1",
        question="Which streaming identifier survives a restart?")

    names = [citation.qualified_name for citation in bridged.citations]
    assert "stream.StreamExecution.runStream" in names
    assert "delta.DeltaSink.queryId" in names
    assert not any("Suite" in name for name in names)
def test_document_seeding_ignores_nonproduction_source_files(conn):
    from library.structural_assembly import seeds_from_documents
    production = "production-seed"
    suite = "suite-seed"
    _symbol(conn, production, file="src/main/DeltaSink.scala", line_start=1,
            qn="delta.DeltaSink.addBatch", display="addBatch")
    _symbol(conn, suite, file="src/test/DeltaSinkSuite.scala", line_start=1,
            qn="delta.DeltaSinkSuite.testReplay", display="testReplay")

    seeds = seeds_from_documents(
        conn,
        [{"source_files": ["src/main/DeltaSink.scala"]},
         {"source_files": ["src/test/DeltaSinkSuite.scala"]}],
        source="src1")

    assert production in seeds.seeds
    assert suite not in seeds.seeds


def test_caller_roots_never_promotes_nonproduction_callers(conn):
    target = "production-target"
    production_caller = "production-caller"
    suite_caller = "suite-caller"
    _symbol(conn, target, file="src/main/DeltaSink.scala", line_start=20,
            qn="delta.DeltaSink.commit", display="commit")
    _symbol(conn, production_caller, file="src/main/DeltaSink.scala", line_start=5,
            qn="delta.DeltaSink.addBatch", display="addBatch")
    _symbol(conn, suite_caller, file="src/test/DeltaSinkSuite.scala", line_start=5,
            qn="delta.DeltaSinkSuite.testReplay", display="testReplay")
    _edge(conn, production_caller, target, line=10, file="src/main/DeltaSink.scala")
    _edge(conn, suite_caller, target, line=10, file="src/test/DeltaSinkSuite.scala")

    expansion = caller_roots(conn, [target], source="src1", depth=1)

    assert production_caller in expansion.roots
    assert suite_caller not in expansion.roots
def test_verified_route_citations_materialize_only_consecutive_scip_edges(conn):
    from library.structural_assembly import citations_from_qualified_routes
    entry, commit, noise = "route-entry", "route-commit", "route-noise"
    _symbol(conn, entry, file="entry.py", line_start=1,
            qn="pkg.Entry.run", display="run")
    _symbol(conn, commit, file="store.py", line_start=10,
            qn="pkg.Store.commit", display="commit")
    _symbol(conn, noise, file="noise.py", line_start=20,
            qn="pkg.Noise.expand", display="expand")
    _edge(conn, entry, commit, line=5, file="entry.py")
    _edge(conn, entry, noise, line=6, file="entry.py")

    citations = citations_from_qualified_routes(
        conn, [("pkg.Entry.run", "pkg.Store.commit")], source="src1")

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Entry.run", "pkg.Store.commit"]
    assert citations[1].call_site_file == "entry.py"
    assert citations[1].call_site_line == 5
def test_obligation_reference_closure_adds_relevant_shared_reference_callers(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure
    constant = "scip-python python src1 0.1 `pkg.shared`/QUERY_ID_KEY."
    consumer = "scip-python python src1 0.1 `pkg.stream`/StreamExecution#start()."
    _symbol(conn, constant, file="pkg/shared.py", line_start=2,
            qn="pkg.shared.QUERY_ID_KEY", parent="pkg.shared")
    _symbol(conn, consumer, file="pkg/stream.py", line_start=4,
            qn="pkg.stream.StreamExecution.start", parent="pkg.stream.StreamExecution")
    _edge(conn, ALPHA_RUN, constant, line=7, file="pkg/alpha.py", edge_type="type_ref")
    _edge(conn, consumer, constant, line=8, file="pkg/stream.py", edge_type="type_ref")
    match = ClewMatch(clew=Clew(id="route", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run", route=["pkg.alpha.Alpha.run"],
        files=["pkg/alpha.py"]), similarity=1.0, obligations=(1,))

    citations = obligation_reference_closure(
        conn, [match], source="src1", question="stable query id after restart")

    assert any(c.qualified_name == "pkg.shared.QUERY_ID_KEY" for c in citations)
    assert any(c.qualified_name == "pkg.stream.StreamExecution.start" for c in citations)
    assert {"references", "shared_reference"} <= {c.relation for c in citations}
def test_obligation_reference_closure_adds_question_relevant_direct_calls(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure
    join = "scip-python python src1 0.1 `pkg.join`/Executor#join()."
    _symbol(conn, join, file="pkg/join.py", line_start=2,
            qn="pkg.join.Executor.join", parent="pkg.join.Executor")
    _edge(conn, ALPHA_RUN, join, line=7, file="pkg/alpha.py", edge_type="call")
    match = ClewMatch(clew=Clew(id="route", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run", route=["pkg.alpha.Alpha.run"],
        files=["pkg/alpha.py"]), similarity=1.0, obligations=(1,))

    citations = obligation_reference_closure(
        conn, [match], source="src1", question="when does the join run")

    assert any(c.qualified_name == "pkg.join.Executor.join" and c.relation == "calls"
               for c in citations)
def test_obligation_reference_closure_crosses_an_unnamed_intermediate(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure
    middle = "scip-python python src1 0.1 `pkg.exec`/Executor#prepare()."
    join = "scip-python python src1 0.1 `pkg.exec`/Executor#join()."
    _symbol(conn, middle, file="pkg/exec.py", line_start=2,
            qn="pkg.exec.Executor.prepare", parent="pkg.exec.Executor")
    _symbol(conn, join, file="pkg/exec.py", line_start=8,
            qn="pkg.exec.Executor.join", parent="pkg.exec.Executor")
    _edge(conn, ALPHA_RUN, middle, line=7, file="pkg/alpha.py", edge_type="call")
    _edge(conn, middle, join, line=9, file="pkg/exec.py", edge_type="call")
    match = ClewMatch(clew=Clew(id="route", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run", route=["pkg.alpha.Alpha.run"],
        files=["pkg/alpha.py"]), similarity=1.0, obligations=(1,))

    citations = obligation_reference_closure(
        conn, [match], source="src1", question="when does the join run")

    assert any(c.qualified_name == "pkg.exec.Executor.prepare" for c in citations)
    assert any(c.qualified_name == "pkg.exec.Executor.join" and c.hop == 2 for c in citations)
def test_obligation_search_reaches_an_upstream_planning_caller(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure
    planner = "scip-python python src1 0.1 `pkg.plan`/MergeRows#plan()."
    _symbol(conn, planner, file="pkg/plan.py", line_start=2,
            qn="pkg.plan.MergeRows.plan", parent="pkg.plan.MergeRows")
    _edge(conn, planner, ALPHA_RUN, line=7, file="pkg/plan.py", edge_type="type_ref")
    match = ClewMatch(clew=Clew(id="route", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run", route=["pkg.alpha.Alpha.run"],
        files=["pkg/alpha.py"]), similarity=1.0, obligations=(1,),
        obligation_texts=((1, "C1 logical MergeRows planning stage"),))

    citations = obligation_reference_closure(
        conn, [match], source="src1", question="how is the operation planned")

    assert any(c.qualified_name == "pkg.plan.MergeRows.plan" for c in citations)
def test_reference_bridge_connects_analyzer_owners_through_a_shared_plan_type(conn):
    from library.structural_assembly import StructuralCitation, reference_bridges
    _symbol(conn, "merge-plan", file="plans/DeltaMergeInto.scala",
            qn="plans.DeltaMergeInto", line_start=1, display="DeltaMergeInto")
    _symbol(conn, "analysis", file="delta/DeltaAnalysis.scala",
            qn="delta.DeltaAnalysis.apply", line_start=10, display="apply")
    _symbol(conn, "preprocess", file="delta/PreprocessTableMerge.scala",
            qn="delta.PreprocessTableMerge.apply", line_start=20, display="apply")
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("analysis", "merge-plan", "type_ref", "delta/DeltaAnalysis.scala", 12, "exact"))
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("preprocess", "merge-plan", "type_ref", "delta/PreprocessTableMerge.scala", 22, "exact"))
    target = StructuralCitation(
        qualified_name="plans.DeltaMergeInto", file="plans/DeltaMergeInto.scala",
        line_start=1, source_name="src1", relation="references", hop=1,
        call_site_file="delta/DeltaAnalysis.scala", call_site_line=12,
        stop_reason="reference")

    bridge = reference_bridges(
        conn, [target], source="src1",
        question="Why does Delta MERGE take a different analysis path?")

    assert {citation.qualified_name for citation in bridge.citations} == {
        "delta.DeltaAnalysis.apply", "delta.PreprocessTableMerge.apply"}
def test_reference_bridge_uses_rare_question_terms_not_common_namespace_terms(conn):
    from library.structural_assembly import StructuralCitation, reference_bridges
    citations = []
    for index, name in enumerate(("DeltaMergePlan", "DeltaReadPlan",
                                  "DeltaWritePlan", "DeltaScanPlan"), 1):
        target = f"target-{index}"
        _symbol(conn, target, file=f"plans/{name}.scala", qn=f"plans.{name}",
                line_start=index, display=name)
        citations.append(StructuralCitation(
            qualified_name=f"plans.{name}", file=f"plans/{name}.scala",
            line_start=index, source_name="src1", relation="references", hop=1,
            call_site_file="owner.scala", call_site_line=index, stop_reason="reference"))
        for suffix in ("a", "b"):
            caller = f"caller-{index}-{suffix}"
            _symbol(conn, caller, file=f"owners/{caller}.scala",
                    qn=f"owners.{caller}", line_start=index, display=caller)
            conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                         (caller, target, "type_ref", f"owners/{caller}.scala",
                          index, "exact"))

    bridge = reference_bridges(
        conn, citations, source="src1",
        question="Why does the Delta MERGE take a different path?")

    assert {citation.parent_qualified_name for citation in bridge.citations} == {
        "plans.DeltaMergePlan"}
    assert len(bridge.citations) == 2
def test_obligation_closure_prefers_rare_plan_term_over_common_namespace(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure
    _symbol(conn, "analysis", file="DeltaAnalysis.scala",
            qn="delta.DeltaAnalysis.apply", line_start=1, display="apply")
    _symbol(conn, "merge-plan", file="DeltaMergeInto.scala",
            qn="plans.DeltaMergeInto", line_start=2, display="DeltaMergeInto")
    conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                 ("analysis", "merge-plan", "type_ref", "DeltaAnalysis.scala", 3, "exact"))
    for index in range(12):
        target = f"noise-{index}"
        _symbol(conn, target, file=f"DeltaNoise{index}.scala",
                qn=f"delta.DeltaNoise{index}.run", line_start=index + 10,
                display=f"DeltaNoise{index}")
        conn.execute("INSERT INTO scip_edges VALUES (?,?,?,?,?,?)",
                     ("analysis", target, "call", "DeltaAnalysis.scala",
                      index + 10, "exact"))
    match = ClewMatch(clew=Clew(
        id="analysis", source_name="src1", entry_symbol="delta.DeltaAnalysis.apply",
        route=["delta.DeltaAnalysis.apply"], files=["DeltaAnalysis.scala"]),
        similarity=1.0, obligations=(1,),
        obligation_texts=((1, "Explain how Delta MERGE changes analysis path"),))

    citations = obligation_reference_closure(
        conn, [match], source="src1",
        question="Why does Delta MERGE take another analysis path?", per_depth=1)

    assert citations[0].qualified_name == "plans.DeltaMergeInto"
def test_entity_symbol_beam_diversifies_ambiguous_operation_owners(conn):
    from library.structural_assembly import question_entity_symbol_seeds
    for canonical, qn, kind, parent in (
            ("delta-analysis", "delta.DeltaAnalysis.stripTempViewForMerge", "Method", "delta.DeltaAnalysis"),
            ("preprocess", "delta.PreprocessTableMerge.apply", "Method", "delta.PreprocessTableMerge"),
            ("rewrite", "spark.RewriteMergeIntoTable.apply", "Method", "spark.RewriteMergeIntoTable"),
            ("duplicate", "spark.RewriteMergeIntoTable.validateMerge", "Method", "spark.RewriteMergeIntoTable"),
            ("generic", "other.MergeWriter.write", "Method", "other.MergeWriter")):
        _symbol(conn, canonical, file=f"{canonical}.scala", qn=qn,
                line_start=1, display=qn.rsplit(".", 1)[-1], kind=kind, parent=parent)

    seeds = question_entity_symbol_seeds(
        conn, "Why does Delta MERGE use analysis instead of the normal rewrite?",
        source="src1", limit=4)

    assert {"delta-analysis", "preprocess", "rewrite"}.issubset(seeds)
    assert not {"rewrite", "duplicate"}.issubset(seeds)
def test_question_ranked_seeds_fetches_by_primary_key_before_source_guard(conn):
    from library.structural_assembly import question_ranked_seeds

    conn.execute(
        "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("wanted", "src1", "python", "merge.py", 1, 2, "Method", "mergeRows",
         "pkg.Merge.mergeRows", "pkg.Merge"))
    conn.execute(
        "INSERT INTO scip_symbols VALUES (?,?,?,?,?,?,?,?,?,?)",
        ("foreign", "src2", "python", "merge.py", 1, 2, "Method", "mergeRows",
         "other.Merge.mergeRows", "other.Merge"))

    class RecordingConnection:
        def __init__(self, wrapped):
            self.wrapped = wrapped
            self.queries = []

        def execute(self, sql, parameters=()):
            self.queries.append(sql)
            return self.wrapped.execute(sql, parameters)

    recording = RecordingConnection(conn)

    ranked = question_ranked_seeds(
        recording, ["wanted", "foreign"], "how are merge rows handled?",
        source="src1")

    assert ranked == ("wanted",)
    symbol_queries = [query for query in recording.queries
                      if "FROM scip_symbols" in query]
    assert symbol_queries
    assert all("WHERE canonical_id IN" in query for query in symbol_queries)
    assert all("WHERE source_name" not in query for query in symbol_queries)
def test_obligation_reference_closure_fetches_edges_through_endpoint_indexes(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure

    target = "indexed-closure-target"
    _symbol(conn, target, file="pkg/target.py", line_start=20,
            qn="pkg.Target.finish", parent="pkg.Target")
    _edge(conn, ALPHA_RUN, target, line=9, edge_type="call")
    match = ClewMatch(clew=Clew(
        id="indexed", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run",
        route=["pkg.alpha.Alpha.run"], files=["pkg/alpha.py"]),
        similarity=1.0, obligations=(1,))
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        citations = obligation_reference_closure(
            conn, [match], source="src1", question="finish target", depth=1)
    finally:
        conn.set_trace_callback(None)

    edge_queries = [sql for sql in statements if "FROM scip_edges" in sql]
    assert citations
    assert edge_queries
    assert all("INDEXED BY idx_scip_edges_" in sql for sql in edge_queries)
    assert all("JOIN scip_symbols" not in sql for sql in edge_queries)
def test_obligation_neighbor_fetch_preserves_both_directions_and_source_scope(conn):
    from library.structural_assembly import _obligation_neighbors

    outbound = "closure-outbound"
    inbound = "closure-inbound"
    foreign = "closure-foreign"
    _symbol(conn, outbound, file="pkg/out.py", line_start=20,
            qn="pkg.Out.finish", parent="pkg.Out")
    _symbol(conn, inbound, file="pkg/in.py", line_start=30,
            qn="pkg.In.prepare", parent="pkg.In")
    _symbol(conn, foreign, file="foreign.py", line_start=40,
            qn="pkg2.Foreign.run", parent="pkg2.Foreign", source="src2")
    _edge(conn, ALPHA_RUN, outbound, line=9, edge_type="call")
    _edge(conn, inbound, ALPHA_RUN, line=10, file="pkg/in.py",
          edge_type="type_ref")
    _edge(conn, ALPHA_RUN, foreign, line=11, edge_type="call")

    rows = _obligation_neighbors(
        conn, {ALPHA_RUN: "pkg.alpha.Alpha.run"}, source="src1")

    assert {
        ("pkg.Out.finish", "call"),
        ("pkg.In.prepare", "incoming_type_ref"),
    } <= {(row[5], row[4]) for row in rows}
    assert all(row[5] != "pkg2.Foreign.run" for row in rows)
    assert all(row[5] != "local 1" for row in rows)
def test_shared_reference_fetch_gates_ubiquitous_type_without_symbol_join(conn):
    from library.structural_assembly import _shared_reference_callers

    target = "ubiquitous-reference-target"
    _symbol(conn, target, file="pkg/shared.py", line_start=1,
            qn="pkg.Shared.KEY", parent="pkg.Shared")
    for index in range(9):
        caller = f"ubiquitous-caller-{index}"
        _symbol(conn, caller, file=f"pkg/caller{index}.py", line_start=index + 2,
                qn=f"pkg.Caller{index}.run", parent=f"pkg.Caller{index}")
        _edge(conn, caller, target, line=index + 2,
              file=f"pkg/caller{index}.py", edge_type="type_ref")
    statements = []
    conn.set_trace_callback(statements.append)
    try:
        callers = _shared_reference_callers(conn, target, source="src1")
    finally:
        conn.set_trace_callback(None)

    assert callers == ()
    edge_queries = [sql for sql in statements if "FROM scip_edges" in sql]
    assert len(edge_queries) == 1
    assert "INDEXED BY idx_scip_edges_callee" in edge_queries[0]
    assert "JOIN" not in edge_queries[0]
def test_obligation_closure_accepts_empty_text_and_explicit_targets(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure

    match = ClewMatch(clew=Clew(
        id="explicit", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run",
        route=["pkg.alpha.Alpha.run"], files=["pkg/alpha.py"]),
        similarity=1.0, obligations=(1,), obligation_texts=((1, ""),),
        target_symbols=((1, "pkg.beta.Beta.start"),))

    assert obligation_reference_closure(
        conn, [match], source="src1", question="", depth=0) == ()
def test_obligation_closure_deduplicates_edges_across_obligations(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure

    target = "deduplicated-reference-target"
    peer = "deduplicated-reference-peer"
    _symbol(conn, target, file="pkg/shared.py", line_start=2,
            qn="pkg.Shared.QUERY_ID", parent="pkg.Shared")
    _symbol(conn, peer, file="pkg/peer.py", line_start=3,
            qn="pkg.Peer.resume", parent="pkg.Peer")
    _edge(conn, ALPHA_RUN, target, line=8, edge_type="type_ref")
    _edge(conn, peer, target, line=9, file="pkg/peer.py", edge_type="type_ref")
    match = ClewMatch(clew=Clew(
        id="duplicate", source_name="src1",
        entry_symbol="pkg.alpha.Alpha.run",
        route=["pkg.alpha.Alpha.run"], files=["pkg/alpha.py"]),
        similarity=1.0, obligations=(1, 2),
        obligation_texts=((1, "query id"), (2, "query id")))

    citations = obligation_reference_closure(
        conn, [match], source="src1", question="query id", depth=1)

    keys = [(citation.qualified_name, citation.relation,
             citation.call_site_file, citation.call_site_line)
            for citation in citations]
    assert len(keys) == len(set(keys))
    assert any(citation.qualified_name == "pkg.Peer.resume"
               and citation.relation == "shared_reference"
               for citation in citations)
def test_selected_route_call_fanout_preserves_direct_compiler_calls(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import selected_route_call_fanout
    expected = set()
    for index in range(9):
        canonical = f"direct-{index}"
        qualified = f"pkg.Target{index}.run"
        _symbol(conn, canonical, file=f"pkg/target{index}.py", line_start=index + 20,
                qn=qualified, parent=f"pkg.Target{index}")
        _edge(conn, ALPHA_RUN, canonical, line=index + 30,
              file="pkg/alpha.py", edge_type="call")
        expected.add(qualified)
    match = ClewMatch(clew=Clew(
        id="route", source_name="src1", entry_symbol="pkg.alpha.Alpha.run",
        route=["pkg.alpha.Alpha.run"], files=["pkg/alpha.py"]),
        similarity=1.0, obligations=(1,))

    citations = selected_route_call_fanout(
        conn, [match], source="src1", per_root=16)

    assert expected.issubset({citation.qualified_name for citation in citations})
    assert all(citation.parent_qualified_name == "pkg.alpha.Alpha.run"
               for citation in citations)
    assert all(citation.stop_reason == "selected_route_fanout"
               for citation in citations)


def test_selected_route_call_fanout_is_bounded_per_root(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import selected_route_call_fanout
    for index in range(7):
        canonical = f"bounded-{index}"
        _symbol(conn, canonical, file=f"pkg/bounded{index}.py", line_start=index + 2,
                qn=f"pkg.Bounded{index}.run", parent=f"pkg.Bounded{index}")
        _edge(conn, ALPHA_RUN, canonical, line=index + 2,
              file="pkg/alpha.py", edge_type="call")
    match = ClewMatch(clew=Clew(
        id="route", source_name="src1", entry_symbol="pkg.alpha.Alpha.run",
        route=["pkg.alpha.Alpha.run"], files=["pkg/alpha.py"]), similarity=1.0)

    citations = selected_route_call_fanout(
        conn, [match], source="src1", per_root=3)

    assert len(citations) == 3
def test_selected_route_branch_fanout_recovers_same_owner_sibling_and_child(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import selected_route_branch_fanout
    caller = "branch-caller"
    selected = "branch-selected"
    sibling = "branch-sibling"
    child = "branch-child"
    unrelated = "branch-unrelated"
    _symbol(conn, caller, file="pkg/flow.py", line_start=1,
            qn="pkg.Flow.apply", parent="pkg.Flow")
    _symbol(conn, selected, file="pkg/flow.py", line_start=10,
            qn="pkg.Flow.fastPath", parent="pkg.Flow")
    _symbol(conn, sibling, file="pkg/flow.py", line_start=20,
            qn="pkg.Flow.safePath", parent="pkg.Flow")
    _symbol(conn, child, file="pkg/flow.py", line_start=30,
            qn="pkg.Flow.persist", parent="pkg.Flow")
    _symbol(conn, unrelated, file="pkg/noise.py", line_start=40,
            qn="pkg.Noise.run", parent="pkg.Noise")
    _edge(conn, caller, selected, line=5, file="pkg/flow.py")
    _edge(conn, caller, sibling, line=6, file="pkg/flow.py")
    _edge(conn, caller, unrelated, line=7, file="pkg/flow.py")
    _edge(conn, sibling, child, line=22, file="pkg/flow.py")
    match = ClewMatch(clew=Clew(
        id="branch", source_name="src1", entry_symbol="pkg.Flow.fastPath",
        route=["pkg.Flow.fastPath"], files=["pkg/flow.py"]), similarity=1.0)

    citations = selected_route_branch_fanout(conn, [match], source="src1")
    names = [citation.qualified_name for citation in citations]

    assert "pkg.Flow.apply" in names
    assert "pkg.Flow.safePath" in names
    assert "pkg.Flow.persist" in names
    assert "pkg.Noise.run" not in names
    sibling_citation = next(
        citation for citation in citations
        if citation.qualified_name == "pkg.Flow.safePath")
    child_citation = next(
        citation for citation in citations
        if citation.qualified_name == "pkg.Flow.persist")
    assert sibling_citation.parent_qualified_name == "pkg.Flow.apply"
    assert child_citation.parent_qualified_name == "pkg.Flow.safePath"


def test_selected_route_branch_fanout_prefers_nearby_branch_siblings(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import selected_route_branch_fanout
    caller = "near-caller"
    selected = "near-selected"
    _symbol(conn, caller, file="pkg/near.py", line_start=1,
            qn="pkg.Near.apply", parent="pkg.Near")
    _symbol(conn, selected, file="pkg/near.py", line_start=10,
            qn="pkg.Near.chosen", parent="pkg.Near")
    _edge(conn, caller, selected, line=50, file="pkg/near.py")
    for index, line in enumerate((5, 49, 51, 90)):
        canonical = f"near-sibling-{index}"
        _symbol(conn, canonical, file="pkg/near.py", line_start=20 + index,
                qn=f"pkg.Near.path{index}", parent="pkg.Near")
        _edge(conn, caller, canonical, line=line, file="pkg/near.py")
    match = ClewMatch(clew=Clew(
        id="near", source_name="src1", entry_symbol="pkg.Near.chosen",
        route=["pkg.Near.chosen"], files=["pkg/near.py"]), similarity=1.0)

    citations = selected_route_branch_fanout(
        conn, [match], source="src1", per_root=2)
    names = {citation.qualified_name for citation in citations}

    assert {"pkg.Near.path1", "pkg.Near.path2"} <= names
    assert "pkg.Near.path0" not in names
    assert "pkg.Near.path3" not in names
def test_fanout_follows_every_overload_of_a_named_root(conn):
    """Overloads share a qualified name but not a canonical id.

    The delegating stub is usually indexed first; a fanout keyed on one
    canonical id per name would only ever see the stub's edges and the
    implementation overload's direct calls would silently vanish.
    """
    from library.structural_assembly import qualified_call_fanout

    stub = 'scip-python python src1 0.1 `pkg.flow`/Flow#write().'
    implementation = 'scip-python python src1 0.1 `pkg.flow`/Flow#write(+1).'
    sink = 'scip-python python src1 0.1 `pkg.flow`/Sink#push().'
    _symbol(conn, stub, file='pkg/flow.py', line_start=2,
            qn='pkg.flow.Flow.write')
    _symbol(conn, implementation, file='pkg/flow.py', line_start=12,
            qn='pkg.flow.Flow.write')
    _symbol(conn, sink, file='pkg/flow.py', line_start=30,
            qn='pkg.flow.Sink.push')
    _edge(conn, stub, implementation, line=3, file='pkg/flow.py')
    _edge(conn, implementation, sink, line=14, file='pkg/flow.py')
    conn.commit()

    citations = qualified_call_fanout(
        conn, ('pkg.flow.Flow.write',), source='src1')

    reached = {(citation.qualified_name, citation.call_site_line)
               for citation in citations}
    assert ('pkg.flow.Sink.push', 14) in reached
def test_qualified_call_fanout_recurses_with_a_global_bound(conn):
    from library.structural_assembly import qualified_call_fanout

    root = "recursive-root"
    middle = "recursive-middle"
    leaf = "recursive-leaf"
    beyond = "recursive-beyond"
    _symbol(conn, root, file="pkg/flow/root.py", line_start=1,
            qn="pkg.Root.start", parent="pkg.Root")
    _symbol(conn, middle, file="pkg/flow/middle.py", line_start=10,
            qn="pkg.Middle.run", parent="pkg.Middle")
    _symbol(conn, leaf, file="pkg/flow/leaf.py", line_start=20,
            qn="pkg.Leaf.run", parent="pkg.Leaf")
    _symbol(conn, beyond, file="pkg/flow/beyond.py", line_start=30,
            qn="pkg.Beyond.run", parent="pkg.Beyond")
    _edge(conn, root, middle, line=5, file="pkg/flow/root.py")
    _edge(conn, middle, leaf, line=12, file="pkg/flow/middle.py")
    _edge(conn, leaf, beyond, line=22, file="pkg/flow/leaf.py")
    conn.commit()

    citations = qualified_call_fanout(
        conn, ("pkg.Root.start",), source="src1",
        depth=2, max_total=2)

    assert [(item.parent_qualified_name, item.qualified_name, item.hop)
            for item in citations] == [
        ("pkg.Root.start", "pkg.Middle.run", 1),
        ("pkg.Middle.run", "pkg.Leaf.run", 2),
    ]
def test_qualified_call_fanout_does_not_recurse_into_external_direct_calls(conn):
    from library.structural_assembly import qualified_call_fanout

    root = "local-root"
    external = "external-call"
    noise = "external-noise"
    _symbol(conn, root, file="app/flow.py", line_start=1,
            qn="app.Flow.run", parent="app.Flow")
    _symbol(conn, external, file="vendor/client.py", line_start=10,
            qn="vendor.Client.send", parent="vendor.Client")
    _symbol(conn, noise, file="vendor/noise.py", line_start=20,
            qn="vendor.Noise.emit", parent="vendor.Noise")
    _edge(conn, root, external, line=5, file="app/flow.py")
    _edge(conn, external, noise, line=12, file="vendor/client.py")
    conn.commit()

    citations = qualified_call_fanout(
        conn, ("app.Flow.run",), source="src1", depth=3, max_total=10)

    assert "vendor.Client.send" in {item.qualified_name for item in citations}
    assert "vendor.Noise.emit" not in {item.qualified_name for item in citations}
def test_qualified_call_fanout_bounds_recursive_roots_separately(conn):
    from library.structural_assembly import qualified_call_fanout

    root = "budget-root"
    _symbol(conn, root, file="pkg/flow/root.py", line_start=1,
            qn="pkg.Budget.start", parent="pkg.Budget")
    for index in range(3):
        middle = f"budget-middle-{index}"
        leaf = f"budget-leaf-{index}"
        _symbol(conn, middle, file=f"pkg/flow/middle{index}.py", line_start=10,
                qn=f"pkg.Middle{index}.run", parent=f"pkg.Middle{index}")
        _symbol(conn, leaf, file=f"pkg/flow/leaf{index}.py", line_start=20,
                qn=f"pkg.Leaf{index}.run", parent=f"pkg.Leaf{index}")
        _edge(conn, root, middle, line=5 + index, file="pkg/flow/root.py")
        _edge(conn, middle, leaf, line=12, file=f"pkg/flow/middle{index}.py")
    conn.commit()

    citations = qualified_call_fanout(
        conn, ("pkg.Budget.start",), source="src1", depth=2,
        max_total=20, recursive_per_root=1)
    names = {item.qualified_name for item in citations}

    assert {f"pkg.Middle{i}.run" for i in range(3)} <= names
    assert "pkg.Leaf0.run" in names
    assert "pkg.Leaf1.run" not in names
    assert "pkg.Leaf2.run" not in names
def test_qualified_call_fanout_caps_recursion_without_dropping_direct_sites(conn):
    from library.structural_assembly import qualified_call_fanout

    root = "recursive-budget-root"
    _symbol(conn, root, file="pkg/flow/root.py", line_start=1,
            qn="pkg.RecursiveBudget.start", parent="pkg.RecursiveBudget")
    for index in range(3):
        middle = f"recursive-budget-middle-{index}"
        leaf = f"recursive-budget-leaf-{index}"
        _symbol(conn, middle, file=f"pkg/flow/middle{index}.py", line_start=10,
                qn=f"pkg.RMiddle{index}.run", parent=f"pkg.RMiddle{index}")
        _symbol(conn, leaf, file=f"pkg/flow/leaf{index}.py", line_start=20,
                qn=f"pkg.RLeaf{index}.run", parent=f"pkg.RLeaf{index}")
        _edge(conn, root, middle, line=5 + index, file="pkg/flow/root.py")
        _edge(conn, middle, leaf, line=12, file=f"pkg/flow/middle{index}.py")
    conn.commit()

    citations = qualified_call_fanout(
        conn, ("pkg.RecursiveBudget.start",), source="src1", depth=2,
        recursive_per_root=3, max_recursive_total=1)
    names = {item.qualified_name for item in citations}

    assert {f"pkg.RMiddle{i}.run" for i in range(3)} <= names
    assert len({name for name in names if "RLeaf" in name}) == 1
def test_qualified_owner_closure_adds_only_proven_owner_edge(conn):
    from library.structural_assembly import qualified_owner_closure

    owner = "owner-flow"
    member = "owner-flow-run"
    sibling = "owner-flow-noise"
    _symbol(conn, owner, file="pkg/flow.py", line_start=1,
            qn="pkg.Flow", parent="pkg")
    _symbol(conn, member, file="pkg/flow.py", line_start=10,
            qn="pkg.Flow.run", parent="pkg.Flow")
    _symbol(conn, sibling, file="pkg/flow.py", line_start=20,
            qn="pkg.Flow.noise", parent="pkg.Flow")
    _edge(conn, owner, member, line=10, file="pkg/flow.py", edge_type="contains")
    _edge(conn, owner, sibling, line=20, file="pkg/flow.py", edge_type="contains")
    conn.commit()

    citations = qualified_owner_closure(
        conn, ("pkg.Flow.run",), source="src1")

    assert [(item.qualified_name, item.relation,
             item.parent_qualified_name, item.call_site_line)
            for item in citations] == [
        ("pkg.Flow", "localized", "", 1),
        ("pkg.Flow.run", "contains", "pkg.Flow", 10),
    ]
def test_qualified_reference_fanout_prioritizes_question_relevant_type_refs(conn):
    from library.structural_assembly import qualified_reference_fanout

    relevant = "reference-query-id"
    incidental = "reference-metrics"
    external = "reference-external"
    _symbol(conn, relevant, file="pkg/query.py", line_start=20,
            qn="pkg.Stream.QUERY_ID_KEY", parent="pkg.Stream")
    _symbol(conn, incidental, file="pkg/metrics.py", line_start=30,
            qn="pkg.Metrics.timer", parent="pkg.Metrics")
    _symbol(conn, external, file="pkg/ext.py", line_start=40,
            qn="pkg.External.value", parent="pkg.External", source="src2")
    _edge(conn, ALPHA_RUN, incidental, line=6, edge_type="type_ref")
    _edge(conn, ALPHA_RUN, relevant, line=7, edge_type="type_ref")
    _edge(conn, ALPHA_RUN, external, line=8, edge_type="type_ref")
    conn.commit()

    citations = qualified_reference_fanout(
        conn, ("pkg.alpha.Alpha.run",), source="src1",
        question="stable query identifier", per_root=1)

    assert [(item.qualified_name, item.relation,
             item.parent_qualified_name, item.call_site_line)
            for item in citations] == [
        ("pkg.Stream.QUERY_ID_KEY", "references",
         "pkg.alpha.Alpha.run", 7),
    ]
def test_qualified_reference_fanout_recurses_only_through_same_file(conn):
    from library.structural_assembly import qualified_reference_fanout

    middle = "reference-query-state"
    leaf = "reference-query-metadata"
    remote = "reference-query-remote"
    remote_leaf = "reference-query-remote-leaf"
    _symbol(conn, middle, file="pkg/alpha.py", line_start=20,
            qn="pkg.Alpha.queryState", parent="pkg.Alpha")
    _symbol(conn, leaf, file="pkg/alpha.py", line_start=30,
            qn="pkg.Alpha.queryMetadata", parent="pkg.Alpha")
    _symbol(conn, remote, file="pkg/remote.py", line_start=40,
            qn="pkg.Remote.queryState", parent="pkg.Remote")
    _symbol(conn, remote_leaf, file="pkg/remote.py", line_start=50,
            qn="pkg.Remote.queryMetadata", parent="pkg.Remote")
    _edge(conn, ALPHA_RUN, middle, line=6, file="pkg/alpha.py",
          edge_type="type_ref")
    _edge(conn, middle, leaf, line=21, file="pkg/alpha.py",
          edge_type="type_ref")
    _edge(conn, ALPHA_RUN, remote, line=7, file="pkg/alpha.py",
          edge_type="type_ref")
    _edge(conn, remote, remote_leaf, line=41, file="pkg/remote.py",
          edge_type="type_ref")
    conn.commit()

    citations = qualified_reference_fanout(
        conn, ("pkg.alpha.Alpha.run",), source="src1",
        question="query metadata", per_root=4, depth=3,
        recursive_per_root=2, max_recursive_total=4)

    names = [citation.qualified_name for citation in citations]
    assert "pkg.Alpha.queryState" in names
    assert "pkg.Alpha.queryMetadata" in names
    assert "pkg.Remote.queryState" in names
    assert "pkg.Remote.queryMetadata" not in names
def test_qualified_reference_fanout_refuses_an_absent_semantic_query(conn):
    from library.structural_assembly import qualified_reference_fanout

    target = "reference-without-query"
    _symbol(conn, target, file="pkg/query.py", line_start=20,
            qn="pkg.Query.identifier", parent="pkg.Query")
    _edge(conn, ALPHA_RUN, target, line=7, edge_type="type_ref")
    conn.commit()

    assert qualified_reference_fanout(
        conn, ("pkg.alpha.Alpha.run",), source="src1",
        question="") == ()
def test_reference_bridge_target_limit_prefers_cross_component_join(conn):
    from library.structural_assembly import StructuralCitation, reference_bridges

    batch = "bridge-batch"
    stable = "bridge-stable"
    _symbol(conn, batch, file="shared/keys.py", line_start=1,
            qn="pkg.Keys.batchId", parent="pkg.Keys")
    _symbol(conn, stable, file="shared/keys.py", line_start=2,
            qn="pkg.Keys.stableIdentifier", parent="pkg.Keys")
    for canonical, qualified, file in (
            ("batch-a", "pkg.BatchA.run", "same/a.py"),
            ("batch-b", "pkg.BatchB.run", "same/b.py"),
            ("stable-a", "pkg.Engine.publish", "engine/a.py"),
            ("stable-b", "pkg.Sink.consume", "sink/b.py")):
        _symbol(conn, canonical, file=file, line_start=10,
                qn=qualified, parent=qualified.rsplit(".", 1)[0])
    _edge(conn, "batch-a", batch, line=11, file="same/a.py",
          edge_type="type_ref")
    _edge(conn, "batch-b", batch, line=11, file="same/b.py",
          edge_type="type_ref")
    _edge(conn, "stable-a", stable, line=11, file="engine/a.py",
          edge_type="type_ref")
    _edge(conn, "stable-b", stable, line=11, file="sink/b.py",
          edge_type="type_ref")
    conn.commit()
    refs = [
        StructuralCitation(
            qualified_name="pkg.Keys.batchId", file="shared/keys.py",
            line_start=1, line_end=1, source_name="src1",
            relation="references", hop=1, call_site_file="root.py",
            call_site_line=1),
        StructuralCitation(
            qualified_name="pkg.Keys.stableIdentifier", file="shared/keys.py",
            line_start=2, line_end=2, source_name="src1",
            relation="references", hop=2, call_site_file="root.py",
            call_site_line=2),
    ]

    result = reference_bridges(
        conn, refs, source="src1",
        question="stable identifier and batch replay",
        target_limit=1, caller_limit_per_target=2)

    assert {citation.qualified_name for citation in result.citations} == {
        "pkg.Engine.publish", "pkg.Sink.consume"}
def test_limited_reference_bridges_do_not_drop_a_common_question_token(conn):
    from library.structural_assembly import StructuralCitation, reference_bridges

    refs = []
    for index, name in enumerate(("primaryIdentifier", "secondaryIdentifier")):
        target = f"common-target-{index}"
        _symbol(conn, target, file=f"shared/key{index}.py", line_start=1,
                qn=f"pkg.Keys.{name}", parent="pkg.Keys")
        refs.append(StructuralCitation(
            qualified_name=f"pkg.Keys.{name}",
            file=f"shared/key{index}.py", line_start=1, line_end=1,
            source_name="src1", relation="references", hop=1,
            call_site_file="root.py", call_site_line=index + 1))
        for side in ("engine", "sink"):
            caller = f"common-caller-{index}-{side}"
            _symbol(conn, caller, file=f"{side}/c{index}.py", line_start=10,
                    qn=f"pkg.{side.title()}{index}.run",
                    parent=f"pkg.{side.title()}{index}")
            _edge(conn, caller, target, line=11,
                  file=f"{side}/c{index}.py", edge_type="type_ref")
    conn.commit()

    result = reference_bridges(
        conn, refs, source="src1", question="identifier survives",
        target_limit=1, caller_limit_per_target=2)

    assert len(result.citations) == 2
def test_limited_reference_bridge_prefers_the_target_owner_publisher(conn):
    from library.structural_assembly import StructuralCitation, reference_bridges

    target = "owner-key"
    owner = "owner-publisher"
    consumer = "external-consumer"
    _symbol(conn, target, file="stream/key.py", line_start=1,
            qn="pkg.Stream.KEY", parent="pkg.Stream")
    _symbol(conn, owner, file="z/stream.py", line_start=10,
            qn="pkg.Stream.publish", parent="pkg.Stream")
    _symbol(conn, consumer, file="a/consumer.py", line_start=10,
            qn='pkg.StreamConsumer.read', parent='pkg.StreamConsumer')
    _edge(conn, owner, target, line=11, file="z/stream.py",
          edge_type="type_ref")
    _edge(conn, consumer, target, line=11, file="a/consumer.py",
          edge_type="type_ref")
    conn.commit()
    ref = StructuralCitation(
        qualified_name="pkg.Stream.KEY", file="stream/key.py",
        line_start=1, line_end=1, source_name="src1",
        relation="references", hop=1, call_site_file="sink.py",
        call_site_line=1)

    result = reference_bridges(
        conn, [ref], source="src1", question="stream key",
        target_limit=1, caller_limit_per_target=1)

    assert [citation.qualified_name for citation in result.citations] == [
        "pkg.Stream.publish"]
def test_qualified_caller_fanout_prefers_same_owner_and_walks_forward(conn):
    from library.structural_assembly import qualified_caller_fanout

    target = "caller-target"
    owner_caller = "caller-owner"
    external_caller = "caller-external"
    sibling = "caller-sibling"
    _symbol(conn, target, file="pkg/flow.py", line_start=20,
            qn="pkg.Flow.process", parent="pkg.Flow")
    _symbol(conn, owner_caller, file="pkg/flow.py", line_start=10,
            qn="pkg.Flow.execute", parent="pkg.Flow")
    _symbol(conn, external_caller, file="other/use.py", line_start=10,
            qn="other.Use.run", parent="other.Use")
    _symbol(conn, sibling, file="pkg/flow.py", line_start=30,
            qn="pkg.Flow.alternate", parent="pkg.Flow")
    _edge(conn, owner_caller, target, line=11, file="pkg/flow.py")
    _edge(conn, external_caller, target, line=11, file="other/use.py")
    _edge(conn, owner_caller, sibling, line=12, file="pkg/flow.py")
    conn.commit()

    citations = qualified_caller_fanout(
        conn, ("pkg.Flow.process",), source="src1",
        callers_per_root=1, child_per_caller=4)

    assert citations[0].qualified_name == "pkg.Flow.execute"
    assert citations[0].relation == "called_by"
    assert citations[0].parent_qualified_name == "pkg.Flow.process"
    assert "pkg.Flow.alternate" in {
        citation.qualified_name for citation in citations}
    assert "other.Use.run" not in {
        citation.qualified_name for citation in citations}
def test_reverse_reference_fanout_finds_external_consumer_and_its_registrar(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    target = "merge-plan"
    self_user = "merge-plan-self"
    rule_owner = "merge-rule-owner"
    rule_user = "merge-rule-user"
    registrar = "merge-extension"
    ambiguous = "merge-builder"
    _symbol(conn, target, file="plan.py", line_start=20,
            qn="pkg.MergePlan", parent="pkg.Plan")
    _symbol(conn, self_user, file="plan.py", line_start=30,
            qn="pkg.Plan.apply", parent="pkg.Plan")
    _symbol(conn, rule_owner, file="rule.py", line_start=1,
            qn="pkg.MergeRule", parent="pkg")
    _symbol(conn, rule_user, file="rule.py", line_start=10,
            qn="pkg.MergeRule.prepare", parent="pkg.MergeRule")
    _symbol(conn, registrar, file="extension.py", line_start=10,
            qn="pkg.MergeExtension.install", parent="pkg.MergeExtension")
    _symbol(conn, ambiguous, file="generated/builder.py", line_start=10,
            qn="pkg.MergeBuilder.make", parent="pkg.MergeBuilder")
    _edge(conn, self_user, target, line=31, file="plan.py",
          edge_type="type_ref")
    _edge(conn, rule_user, target, line=11, file="rule.py",
          edge_type="type_ref")
    _edge(conn, rule_owner, rule_user, line=10, file="rule.py",
          edge_type="contains")
    _edge(conn, registrar, rule_owner, line=11, file="extension.py",
          edge_type="type_ref")
    _edge(conn, ambiguous, rule_owner, line=11, file="builder.py",
          edge_type="type_ref")
    conn.commit()

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.MergePlan",), source="src1",
        question="why does the merge plan pass through the rule extension",
        per_root=2, owner_per_root=1)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.MergeRule.prepare", "pkg.MergeExtension.install"]
    assert [citation.parent_qualified_name for citation in citations] == [
        "pkg.MergePlan", "pkg.MergeRule"]
    assert all(
        citation.relation == "referenced_by"
        and citation.stop_reason == "selected_reference_caller"
        for citation in citations)
def test_reverse_reference_fanout_keeps_same_owner_consumer_of_selected_member(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    target = "stable-key"
    publisher = "stream-publisher"
    _symbol(conn, target, file="stream.py", line_start=2,
            qn="pkg.Stream.STABLE_KEY", parent="pkg.Stream")
    _symbol(conn, publisher, file="stream.py", line_start=20,
            qn="pkg.Stream.publish", parent="pkg.Stream")
    _edge(conn, publisher, target, line=24, file="stream.py",
          edge_type="type_ref")
    conn.commit()

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.Stream.STABLE_KEY",), source="src1",
        question="where does the stream publish the stable key",
        per_root=1, owner_per_root=0)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Stream.publish"]
    assert citations[0].parent_qualified_name == "pkg.Stream.STABLE_KEY"
def test_reverse_reference_fanout_prioritizes_same_owner_member_consumer(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    target = "stable-owner-key"
    publisher = "stable-owner-publisher"
    external = "stable-external-reader"
    _symbol(conn, target, file="stream.py", line_start=2,
            qn="pkg.Stream.STABLE_KEY", parent="pkg.Stream")
    _symbol(conn, publisher, file="stream.py", line_start=20,
            qn="pkg.Stream.publish", parent="pkg.Stream")
    _symbol(conn, external, file="external.py", line_start=20,
            qn="pkg.StableStreamKeyReader.readIdentifier",
            parent="pkg.StableStreamKeyReader")
    _edge(conn, publisher, target, line=24, file="stream.py",
          edge_type="type_ref")
    _edge(conn, external, target, line=24, file="external.py",
          edge_type="type_ref")
    conn.commit()

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.Stream.STABLE_KEY",), source="src1",
        question="stream stable key identifier",
        per_root=1, owner_per_root=0)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Stream.publish"]
def test_same_owner_reference_fanout_excludes_cross_owner_targets(conn):
    from library.structural_assembly import qualified_same_owner_reference_fanout

    root = "owner-ref-root"
    local = "owner-ref-local"
    external = "owner-ref-external"
    _symbol(conn, root, file="flow.py", line_start=20,
            qn="pkg.Flow.publish", parent="pkg.Flow")
    _symbol(conn, local, file="flow.py", line_start=2,
            qn="pkg.Flow.id", parent="pkg.Flow")
    _symbol(conn, external, file="metadata.py", line_start=2,
            qn="pkg.Metadata.stableIdentifier", parent="pkg.Metadata")
    _edge(conn, root, local, line=24, file="flow.py", edge_type="type_ref")
    _edge(conn, root, external, line=25, file="flow.py", edge_type="type_ref")
    conn.commit()

    citations = qualified_same_owner_reference_fanout(
        conn, ("pkg.Flow.publish",), source="src1",
        question="stable flow id metadata", per_root=4)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Flow.id"]
def test_selected_route_branch_fanout_accepts_qualified_roots(conn):
    from library.structural_assembly import selected_route_branch_fanout

    chosen = "qualified-branch-chosen"
    sibling = "qualified-branch-sibling"
    caller = "qualified-branch-caller"
    _symbol(conn, chosen, file="decision.py", line_start=20,
            qn="pkg.Decision.chosen", parent="pkg.Decision")
    _symbol(conn, sibling, file="decision.py", line_start=30,
            qn="pkg.Decision.alternate", parent="pkg.Decision")
    _symbol(conn, caller, file="decision.py", line_start=10,
            qn="pkg.Decision.choose", parent="pkg.Decision")
    _edge(conn, caller, chosen, line=12, file="decision.py")
    _edge(conn, caller, sibling, line=13, file="decision.py")
    conn.commit()

    citations = selected_route_branch_fanout(
        conn, (), source="src1", roots=("pkg.Decision.chosen",),
        per_root=1, child_per_sibling=0)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Decision.choose", "pkg.Decision.alternate"]
    assert all(citation.line_end == citation.line_start for citation in citations)
def test_reference_fanout_duplicate_sites_do_not_consume_distinct_target_budget(conn):
    from library.structural_assembly import qualified_reference_fanout

    root = "duplicate-ref-root"
    repeated = "duplicate-ref-repeated"
    distinct = "duplicate-ref-distinct"
    _symbol(conn, root, file="flow.py", line_start=10,
            qn="pkg.Flow.analyze", parent="pkg.Flow")
    _symbol(conn, repeated, file="types.py", line_start=2,
            qn="pkg.MergeTable", parent="pkg")
    _symbol(conn, distinct, file="types.py", line_start=20,
            qn="pkg.MergePlan", parent="pkg")
    for line in range(11, 19):
        _edge(conn, root, repeated, line=line, file="flow.py",
              edge_type="type_ref")
    _edge(conn, root, distinct, line=19, file="flow.py",
          edge_type="type_ref")
    conn.commit()

    citations = qualified_reference_fanout(
        conn, ("pkg.Flow.analyze",), source="src1",
        question="merge table plan", per_root=2, depth=1,
        recursive_per_root=0, max_recursive_total=0)

    assert {citation.qualified_name for citation in citations} == {"pkg.MergePlan", "pkg.MergeTable"}
def test_reference_fanout_diversifies_target_owner_families(conn):
    from library.structural_assembly import qualified_reference_fanout

    root = "family-ref-root"
    _symbol(conn, root, file="flow.py", line_start=10,
            qn="pkg.Flow.analyze", parent="pkg.Flow")
    for index, member in enumerate(("path", "log", "catalog")):
        target = f"family-ref-table-{member}"
        _symbol(conn, target, file="table.py", line_start=index + 2,
                qn=f"pkg.MergeTable.{member}", parent="pkg.MergeTable")
        _edge(conn, root, target, line=11 + index, file="flow.py",
              edge_type="type_ref")
    plan = "family-ref-plan"
    _symbol(conn, plan, file="plan.py", line_start=20,
            qn="pkg.MergePlan", parent="pkg")
    _edge(conn, root, plan, line=20, file="flow.py", edge_type="type_ref")
    conn.commit()

    citations = qualified_reference_fanout(
        conn, ("pkg.Flow.analyze",), source="src1",
        question="merge table plan", per_root=3, owner_per_root=2,
        depth=1, recursive_per_root=0, max_recursive_total=0)

    assert "pkg.MergePlan" in {
        citation.qualified_name for citation in citations}
    assert len([
        citation for citation in citations
        if citation.qualified_name.startswith("pkg.MergeTable.")]) == 2
def test_reference_fanout_prefers_discriminating_question_tokens(conn):
    from library.structural_assembly import qualified_reference_fanout

    root = "weighted-ref-root"
    _symbol(conn, root, file="flow.py", line_start=10,
            qn="pkg.Flow.analyze", parent="pkg.Flow")
    for index, member in enumerate(("path", "log", "catalog", "snapshot")):
        target = f"weighted-ref-table-{member}"
        _symbol(conn, target, file="table.py", line_start=index + 2,
                qn=f"pkg.DeltaTableV2.{member}", parent="pkg.DeltaTableV2")
        _edge(conn, root, target, line=11 + index, file="flow.py",
              edge_type="type_ref")
    plan = "weighted-ref-plan"
    _symbol(conn, plan, file="plan.py", line_start=20,
            qn="pkg.DeltaMergePlan", parent="pkg")
    _edge(conn, root, plan, line=20, file="flow.py", edge_type="type_ref")
    conn.commit()

    citations = qualified_reference_fanout(
        conn, ("pkg.Flow.analyze",), source="src1",
        question="delta merge table v2", per_root=1,
        depth=1, recursive_per_root=0, max_recursive_total=0)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.DeltaMergePlan"]
def test_reverse_reference_fanout_lifts_member_reference_to_owner_type(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    owner = "plan-owner"
    member = "plan-member"
    consumer = "plan-consumer"
    _symbol(conn, owner, file="pkg/plan.py", line_start=1,
            qn="pkg.Plan", kind="Class", parent="pkg")
    _symbol(conn, member, file="pkg/plan.py", line_start=4,
            qn="pkg.Plan.target", parent="pkg.Plan")
    _symbol(conn, consumer, file="pkg/prepare.py", line_start=10,
            qn="pkg.Prepare.apply", parent="pkg.Prepare")
    _edge(conn, consumer, owner, line=12, file="pkg/prepare.py",
          edge_type="type_ref")

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.Plan.target",), source="src1",
        question="prepare plan target", per_root=2, max_total=4, lift_members = True)

    assert any(
        citation.parent_qualified_name == "pkg.Plan"
        and citation.qualified_name == "pkg.Prepare.apply"
        for citation in citations)
def test_reverse_reference_fanout_prefers_same_repository_consumer(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    target = "repo-target"
    local = "repo-local-consumer"
    foreign = "repo-foreign-consumer"
    _symbol(conn, target, file="engine/src/plan.py", line_start=1,
            qn="pkg.Plan", kind="Class", parent="pkg")
    _symbol(conn, local, file="engine/src/prepare.py", line_start=10,
            qn="pkg.z.PrepareMerge.apply", parent="pkg.z.PrepareMerge")
    _symbol(conn, foreign, file="adapter/src/prepare.py", line_start=10,
            qn="pkg.a.PrepareMerge.apply", parent="pkg.a.PrepareMerge")
    _edge(conn, local, target, line=12, file="engine/src/prepare.py",
          edge_type="type_ref")
    _edge(conn, foreign, target, line=12, file="adapter/src/prepare.py",
          edge_type="type_ref")

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.Plan",), source="src1",
        question="prepare merge plan", per_root=1, max_total=1)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.z.PrepareMerge.apply"]
def test_reverse_reference_fanout_can_replace_member_with_indexed_owner(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    owner = "lift-owner"
    member = "lift-member"
    child = "lift-child"
    consumer = "lift-consumer"
    _symbol(conn, owner, file="core/plan.py", line_start=1,
            qn="pkg.Plan", kind="Class", parent="pkg")
    _symbol(conn, member, file="core/plan.py", line_start=4,
            qn="pkg.Plan.target", parent="pkg.Plan")
    _symbol(conn, child, file="core/plan.py", line_start=8,
            qn="pkg.Plan.children", parent="pkg.Plan")
    _symbol(conn, consumer, file="core/prepare.py", line_start=10,
            qn="pkg.Prepare.apply", parent="pkg.Prepare")
    _edge(conn, child, member, line=9, file="core/plan.py",
          edge_type="type_ref")
    _edge(conn, consumer, owner, line=12, file="core/prepare.py",
          edge_type="type_ref")

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.Plan.target",), source="src1",
        question="prepare plan target", per_root=1, max_total=1,
        lift_members=True)

    assert [(citation.parent_qualified_name, citation.qualified_name)
            for citation in citations] == [("pkg.Plan", "pkg.Prepare.apply")]


def test_reverse_reference_fanout_reserves_capacity_for_registrar(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    for canonical, qn, parent, file in (
            ("reserve-a", "pkg.APlan", "pkg", "core/a.py"),
            ("reserve-b", "pkg.BPlan", "pkg", "core/b.py"),
            ("reserve-owner-a", "pkg.AConsumer", "pkg", "core/a_consumer.py"),
            ("reserve-owner-b", "pkg.BConsumer", "pkg", "core/b_consumer.py"),
            ("reserve-consumer-a", "pkg.AConsumer.prepare", "pkg.AConsumer", "core/a_consumer.py"),
            ("reserve-consumer-b", "pkg.BConsumer.prepare", "pkg.BConsumer", "core/b_consumer.py"),
            ("reserve-registrar", "pkg.Extension.install", "pkg.Extension", "core/extension.py")):
        _symbol(conn, canonical, file=file, line_start=1, qn=qn,
                parent=parent, kind="Class" if "." not in qn[4:] else "Method")
    _edge(conn, "reserve-consumer-a", "reserve-a", line=2,
          file="core/a_consumer.py", edge_type="type_ref")
    _edge(conn, "reserve-consumer-b", "reserve-b", line=2,
          file="core/b_consumer.py", edge_type="type_ref")
    _edge(conn, "reserve-registrar", "reserve-owner-a", line=2,
          file="core/extension.py", edge_type="type_ref")

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.APlan", "pkg.BPlan"), source="src1",
        question="prepare plan extension", per_root=1,
        owner_per_root=1, max_total=2, reserve_registrars = True)

    assert [(citation.parent_qualified_name, citation.qualified_name)
            for citation in citations] == [
        ("pkg.APlan", "pkg.AConsumer.prepare"),
        ("pkg.AConsumer", "pkg.Extension.install")]
def test_reference_fanout_ignores_prose_stopwords_when_ranking_members(conn):
    from library.structural_assembly import qualified_reference_fanout

    root = "stopword-root"
    noise = "stopword-noise"
    wanted = "stopword-wanted"
    _symbol(conn, root, file="core/engine.py", line_start=1,
            qn="pkg.Engine.run", parent="pkg.Engine")
    _symbol(conn, noise, file="core/engine.py", line_start=10,
            qn="pkg.Engine.deleteOnStop", parent="pkg.Engine")
    _symbol(conn, wanted, file="core/engine.py", line_start=20,
            qn="pkg.Engine.stableId", parent="pkg.Engine")
    _edge(conn, root, noise, line=3, file="core/engine.py",
          edge_type="type_ref")
    _edge(conn, root, wanted, line=4, file="core/engine.py",
          edge_type="type_ref")

    citations = qualified_reference_fanout(
        conn, ("pkg.Engine.run",), source="src1",
        question="what identifier does it rely on after restart",
        per_root=1, depth=1)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Engine.stableId"]
def test_reference_fanout_excludes_retained_target_before_owner_budget(conn):
    from library.structural_assembly import qualified_reference_fanout

    root = "exclude-root"
    retained = "exclude-retained"
    wanted = "exclude-wanted"
    _symbol(conn, root, file="core/stream.py", line_start=1,
            qn="pkg.Stream.publish", parent="pkg.Stream")
    _symbol(conn, retained, file="core/stream.py", line_start=4,
            qn="pkg.Stream.ROUTE_ID", parent="pkg.Stream")
    _symbol(conn, wanted, file="core/stream.py", line_start=8,
            qn="pkg.Stream.id", parent="pkg.Stream")
    _edge(conn, root, retained, line=2, file="core/stream.py",
          edge_type="type_ref")
    _edge(conn, root, wanted, line=3, file="core/stream.py",
          edge_type="type_ref")

    citations = qualified_reference_fanout(
        conn, ("pkg.Stream.publish",), source="src1",
        question="which stream identifier survives", per_root=2,
        owner_per_root=1, depth=1,
        excluded_targets=("pkg.Stream.ROUTE_ID",))

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Stream.id"]
def test_same_owner_reference_fanout_follows_one_same_file_dependency_layer(conn):
    from library.structural_assembly import qualified_same_owner_reference_fanout

    for canonical, qn, line in (
            ("local-root", "pkg.Stream.publish", 1),
            ("local-id", "pkg.Stream.id", 10),
            ("local-metadata", "pkg.Stream.metadata", 20),
            ("local-store", "pkg.Stream.store", 30)):
        _symbol(conn, canonical, file="core/stream.py", line_start=line,
                qn=qn, parent="pkg.Stream")
    _edge(conn, "local-root", "local-id", line=3,
          file="core/stream.py", edge_type="type_ref")
    _edge(conn, "local-id", "local-metadata", line=12,
          file="core/stream.py", edge_type="type_ref")
    _edge(conn, "local-metadata", "local-store", line=22,
          file="core/stream.py", edge_type="type_ref")

    citations = qualified_same_owner_reference_fanout(
        conn, ("pkg.Stream.publish",), source="src1",
        question="stream identifier metadata", per_root=2)

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Stream.id", "pkg.Stream.metadata"]
def test_arbitrary_explicit_entity_can_seed_a_unique_compound_symbol(conn):
    from library.structural_assembly import question_symbol_seeds
    _symbol(conn, "widget-processor", file="app/WidgetStreamProcessor.py",
            qn="app.WidgetStreamProcessor", line_start=1,
            display="WidgetStreamProcessor", kind="Class")
    _symbol(conn, "widget-reconstruct", file="app/WidgetStreamProcessor.py",
            qn="app.WidgetStreamProcessor.reconstructSession", line_start=12,
            display="reconstructSession", parent="app.WidgetStreamProcessor")

    seeds = question_symbol_seeds(
        conn, "How does Widget reconstruct streaming sessions?", source="src1")

    assert seeds == ("widget-reconstruct",)
def test_qualified_call_fanout_defaults_to_four_direct_dependencies(conn):
    from library.structural_assembly import qualified_call_fanout
    root = "bounded-default-root"
    _symbol(conn, root, file="pkg/root.py", line_start=1,
            qn="pkg.Root.run", parent="pkg.Root")
    for index in range(6):
        target = f"bounded-default-target-{index}"
        _symbol(conn, target, file=f"pkg/target{index}.py", line_start=20,
                qn=f"pkg.Target{index}.run", parent=f"pkg.Target{index}")
        _edge(conn, root, target, line=10 + index, file="pkg/root.py")
    conn.commit()

    citations = qualified_call_fanout(
        conn, ("pkg.Root.run",), source="src1")

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Target0.run", "pkg.Target1.run", "pkg.Target2.run",
        "pkg.Target3.run"]
def test_reverse_reference_fanout_honors_per_root_for_distinct_callers(conn):
    from library.structural_assembly import qualified_reverse_reference_fanout

    _symbol(conn, "target-two", file="model.py", line_start=2,
            qn="pkg.Model.stableIdentifier", parent="pkg.Model")
    _symbol(conn, "caller-one", file="first.py", line_start=10,
            qn="pkg.First.readStableIdentifier", parent="pkg.First")
    _symbol(conn, "caller-two", file="second.py", line_start=20,
            qn="pkg.Second.writeStableIdentifier", parent="pkg.Second")
    _edge(conn, "caller-one", "target-two", line=12, file="first.py",
          edge_type="type_ref")
    _edge(conn, "caller-two", "target-two", line=22, file="second.py",
          edge_type="type_ref")
    conn.commit()

    citations = qualified_reverse_reference_fanout(
        conn, ("pkg.Model.stableIdentifier",), source="src1",
        question="first second stable identifier", per_root=2,
        owner_per_root=0)

    assert {citation.qualified_name for citation in citations} == {
        "pkg.First.readStableIdentifier", "pkg.Second.writeStableIdentifier"}
def test_verified_route_citations_keep_all_overloads_until_edge_resolution(conn):
    from library.structural_assembly import citations_from_qualified_routes

    _symbol(conn, "entry-a", file="entry.scala", line_start=1,
            qn="pkg.Entry.run", display="run")
    _symbol(conn, "entry-z", file="entry.scala", line_start=20,
            qn="pkg.Entry.run", display="run")
    _symbol(conn, "commit", file="store.scala", line_start=40,
            qn="pkg.Store.commit", display="commit")
    _edge(conn, "entry-z", "commit", line=25, file="entry.scala")

    citations = citations_from_qualified_routes(
        conn, [("pkg.Entry.run", "pkg.Store.commit")], source="src1")

    assert [citation.qualified_name for citation in citations] == [
        "pkg.Entry.run", "pkg.Store.commit"]
    assert citations[1].call_site_line == 25


def test_obligation_reference_closure_labels_incoming_calls_as_called_by(conn):
    from library.clews import Clew, ClewMatch
    from library.structural_assembly import obligation_reference_closure

    _symbol(conn, "incoming-target", file="target.py", line_start=20,
            qn="pkg.Target.execute", parent="pkg.Target")
    _symbol(conn, "incoming-caller", file="caller.py", line_start=2,
            qn="pkg.Caller.invokeTarget", parent="pkg.Caller")
    _edge(conn, "incoming-caller", "incoming-target", line=7,
          file="caller.py", edge_type="call")
    match = ClewMatch(clew=Clew(
        id="target", source_name="src1", entry_symbol="pkg.Target.execute",
        route=["pkg.Target.execute"], files=["target.py"]),
        similarity=1.0, obligations=(1,),
        obligation_texts=((1, "C1 caller invokes target"),))

    citations = obligation_reference_closure(
        conn, [match], source="src1", question="which caller invokes target",
        per_depth=5, depth=1)

    incoming = next(citation for citation in citations
                    if citation.qualified_name == "pkg.Caller.invokeTarget")
    assert incoming.parent_qualified_name == "pkg.Target.execute"
    assert incoming.relation == "called_by"
def test_nested_execution_enclosure_bridge_recovers_constructor_and_outer_entry(conn):
    from library.structural_assembly import nested_execution_enclosure_bridges

    outer = "pkg.Executor"
    nested = "pkg.Executor.BatchIterator"
    selected = "nested-emit"
    constructor = "nested-init"
    worker = "outer-process"
    entry = "outer-execute"
    noise = "other-process"
    _symbol(conn, "outer-owner", file="pkg/executor.py", line_start=1,
            qn=outer, parent="pkg")
    _symbol(conn, "nested-owner", file="pkg/executor.py", line_start=5,
            qn=nested, parent=outer)
    _edge(conn, "outer-owner", "nested-owner", line=5,
          file="pkg/executor.py", edge_type="contains")
    _symbol(conn, selected, file="pkg/executor.py", line_start=40,
            qn="pkg.Executor.BatchIterator.emit", parent=nested)
    _symbol(conn, constructor, file="pkg/executor.py", line_start=30,
            qn="pkg.Executor.BatchIterator.<init>", parent=nested)
    _symbol(conn, worker, file="pkg/executor.py", line_start=20,
            qn="pkg.Executor.process", parent=outer)
    _symbol(conn, entry, file="pkg/executor.py", line_start=10,
            qn="pkg.Executor.execute", parent=outer)
    _symbol(conn, noise, file="other/worker.py", line_start=10,
            qn="pkg.Other.process", parent="pkg.Other")
    _edge(conn, worker, constructor, line=24, file="pkg/executor.py")
    _edge(conn, entry, worker, line=14, file="pkg/executor.py")
    _edge(conn, noise, constructor, line=12, file="other/worker.py")
    conn.commit()

    citations = nested_execution_enclosure_bridges(
        conn, ("pkg.Executor.BatchIterator.emit",), source="src1")

    assert [(citation.qualified_name, citation.parent_qualified_name,
             citation.relation, citation.stop_reason) for citation in citations] == [
        ("pkg.Executor.process", "pkg.Executor.BatchIterator.<init>",
         "called_by", "selected_nested_constructor_caller"),
        ("pkg.Executor.execute", "pkg.Executor.process",
         "called_by", "selected_nested_execution_entry"),
    ]
def test_nested_execution_enclosure_bridge_requires_a_direct_constructor_call(conn):
    from library.structural_assembly import nested_execution_enclosure_bridges

    _symbol(conn, "selected", file="pkg/worker.py", line_start=30,
            qn="pkg.Worker.Nested.emit", parent="pkg.Worker.Nested")
    _symbol(conn, "caller", file="pkg/worker.py", line_start=10,
            qn="pkg.Worker.execute", parent="pkg.Worker")
    conn.commit()

    assert nested_execution_enclosure_bridges(
        conn, ("pkg.Worker.Nested.emit",), source="src1") == ()
def test_nested_execution_enclosure_bridge_requires_compiler_ownership(conn):
    from library.structural_assembly import nested_execution_enclosure_bridges

    outer = "pkg.Worker"
    nested = "pkg.Worker.Nested"
    _symbol(conn, "outer-owner", file="pkg/worker.py", line_start=1,
            qn=outer, parent="pkg")
    _symbol(conn, "nested-owner", file="pkg/worker.py", line_start=5,
            qn=nested, parent=outer)
    _symbol(conn, "selected", file="pkg/worker.py", line_start=30,
            qn="pkg.Worker.Nested.emit", parent=nested)
    _symbol(conn, "constructor", file="pkg/worker.py", line_start=20,
            qn="pkg.Worker.Nested.<init>", parent=nested)
    _symbol(conn, "worker", file="pkg/worker.py", line_start=10,
            qn="pkg.Worker.process", parent=outer)
    _symbol(conn, "entry", file="pkg/worker.py", line_start=7,
            qn="pkg.Worker.execute", parent=outer)
    _edge(conn, "worker", "constructor", line=12, file="pkg/worker.py")
    _edge(conn, "entry", "worker", line=8, file="pkg/worker.py")
    conn.commit()

    assert nested_execution_enclosure_bridges(
        conn, ("pkg.Worker.Nested.emit",), source="src1") == ()

