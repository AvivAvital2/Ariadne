from dataclasses import dataclass, field
from library.chain_confidence import (
    assess_chain_confidence,
    assess_obligation_coverage,
)


@dataclass
class Hop:
    source_excerpts: tuple = ()
    citation: object = None


@dataclass
class Evidence:
    hops: tuple = ()
    source_gaps: tuple = ()
    caller_frontiers: tuple = ()
    unresolved_paths: tuple = ()
    truncation_reason: str = ""
    fan_outs: tuple = ()
    mandatory_fan_outs: tuple | None = None


def test_confidence_requires_a_complete_materialized_supported_chain():
    absent = assess_chain_confidence(None, claims_total=0, supported_claims=0)
    assert absent.level == "low"
    assert absent.reasons == ("no compiler chain",)

    unsupported = assess_chain_confidence(
        Evidence(hops=(Hop(("source",)),)), claims_total=2, supported_claims=1)
    assert unsupported.level == "low"
    assert '1 rejected formulation claim' in unsupported.reasons

    incomplete = assess_chain_confidence(
        Evidence(hops=(Hop(("source",)), Hop(),),
                 source_gaps=("missing source",), caller_frontiers=("dispatch",),
                 unresolved_paths=("missing.py",), truncation_reason="window"),
        claims_total=2, supported_claims=2)
    assert incomplete.level == 'low'
    assert incomplete.reasons == (
        "source materialized for 1/2 hops", "1 source gap",
        "1 caller frontier", "1 unresolved path", "window")

    complete = assess_chain_confidence(
        Evidence(hops=(Hop(("source",)), Hop(("source",))),),
        claims_total=2, supported_claims=2)
    assert complete.level == "high"
    assert complete.reasons == ()
def test_confidence_covers_empty_and_plural_claim_boundaries():
    empty_chain = assess_chain_confidence(Evidence(), claims_total=0, supported_claims=0)
    assert empty_chain.reasons == ("no compiler chain",)
    no_claims = assess_chain_confidence(
        Evidence(hops=(Hop(("source",)),)), claims_total=0, supported_claims=0)
    assert no_claims.reasons == ("no supported claims",)
    unsupported = assess_chain_confidence(
        Evidence(hops=(Hop(("source",)),)), claims_total=3, supported_claims=1)
    assert unsupported.reasons == ('2 rejected formulation claims',)
def test_completeness_fails_closed_on_every_unresolved_functional_path():
    from library.chain_confidence import assess_chain_completeness
    evidence = Evidence(
        hops=(Hop(("source",)), Hop()),
        source_gaps=("missing source",),
        caller_frontiers=("dispatch",),
        unresolved_paths=("missing.py",),
        truncation_reason="chain truncated", fan_outs = ("25 implementations",), mandatory_fan_outs = ("25 implementations",))

    result = assess_chain_completeness(
        evidence, claims_total=4, supported_claims=3)

    assert result.complete is False
    assert result.reasons == (
        "source materialized for 1/2 hops",
        "1 source gap", "1 caller frontier", "1 unresolved fork", "chain truncated")


def test_completeness_requires_claims_and_a_closed_materialized_chain():
    from library.chain_confidence import assess_chain_completeness
    closed = Evidence(hops=(Hop(("source",)), Hop(("source",))))
    assert assess_chain_completeness(
        closed, claims_total=2, supported_claims=2).complete is True
    empty = assess_chain_completeness(
        closed, claims_total=0, supported_claims=0)
    assert empty.complete is True
    assert empty.reasons == ()
@dataclass
class Citation:
    qualified_name: str = "child.run"
    relation: str = "calls"
    parent_qualified_name: str = "parent.run"
    file: str = "child.py"
    line_start: int = 2
    call_site_file: str = "parent.py"
    call_site_line: int = 5


def test_transition_accounting_is_topological_not_prose_based():
    from library.chain_confidence import account_transitions
    root = Hop(("source",), Citation(relation="localized", parent_qualified_name=""))
    connected = Hop(("source",), Citation())
    disconnected = Hop(("source",), Citation(parent_qualified_name=""))
    evidence = Evidence(hops=(root, connected, disconnected))

    result = account_transitions(evidence)

    assert result.total == 2
    assert result.accounted == 1
    assert result.unaccounted == 1


def test_formulation_coverage_is_separate_from_chain_completeness():
    from library.chain_confidence import assess_chain_completeness, assess_formulation_coverage
    evidence = Evidence(hops=(Hop(("source",), Citation()),))

    chain = assess_chain_completeness(evidence, claims_total=3, supported_claims=1)
    formulation = assess_formulation_coverage(claims_total=3, supported_claims=1)

    assert chain.complete is True
    assert chain.reasons == ()
    assert formulation.complete is False
    assert formulation.reasons == ("2 rejected formulation claims",)
def test_scope_completeness_owns_unresolved_retrieval_paths():
    from library.chain_confidence import assess_chain_completeness, assess_scope_completeness
    evidence = Evidence(
        hops=(Hop(("source",), Citation()),),
        unresolved_paths=("docs/not-indexed.md",))

    chain = assess_chain_completeness(evidence)
    scope = assess_scope_completeness(evidence)

    assert chain.complete is True
    assert chain.reasons == ()
    assert scope.complete is False
    assert scope.reasons == ("unresolved positioning path: docs/not-indexed.md",)
def test_completeness_requires_connected_mandatory_endpoints():
    from library.chain_confidence import assess_chain_completeness
    disconnected = Evidence(hops=(
        Hop(("source",), Citation(
            qualified_name="entry.run", relation="localized",
            parent_qualified_name="", file="entry.py", line_start=1,
            call_site_file="", call_site_line=0)),
    ))

    result = assess_chain_completeness(disconnected)

    assert result.complete is False
    assert result.reasons == ("1 disconnected mandatory endpoint",)

def test_completeness_rejects_an_executable_frontier_on_the_mandatory_component():
    from library.chain_confidence import assess_chain_completeness
    entry = Hop(("source",), Citation(
        qualified_name="entry.run", relation="localized",
        parent_qualified_name="", file="entry.py", line_start=1,
        call_site_file="", call_site_line=0))
    frontier_citation = Citation(
        qualified_name="db.commit", relation="calls",
        parent_qualified_name="entry.run", file="db.py", line_start=4,
        call_site_file="entry.py", call_site_line=2)
    frontier_citation.stop_reason = "depth"
    frontier = Hop(("source",), frontier_citation)

    result = assess_chain_completeness(Evidence(hops=(entry, frontier)))

    assert result.complete is False
    assert "1 executable frontier" in result.reasons
def test_display_confidence_reports_independent_dimensions():
    from library.chain_confidence import derive_display_confidence

    assert derive_display_confidence(chain_complete=True, formulation_complete=True,
                                     scope_complete=True) == "high"
    assert derive_display_confidence(chain_complete=True, formulation_complete=True,
                                     scope_complete=False) == "medium"
    assert derive_display_confidence(chain_complete=False, formulation_complete=True,
                                     scope_complete=True) == "low"
    assert derive_display_confidence(chain_complete=True, formulation_complete=False,
                                     scope_complete=True) == "low"
def test_selection_coverage_requires_roots_not_every_alternative_route():
    from library.chain_confidence import assess_selection_coverage
    candidates = {
        "R1": ["pkg.Entry.run", "pkg.Store.save"],
        "R2": ["pkg.Entry.run", "pkg.Cache.save"],
        "R3": ["pkg.Noise.run"],
    }

    covered = assess_selection_coverage(
        candidates, ("R1",), required_symbols=("pkg.Entry.run",))
    missing = assess_selection_coverage(
        candidates, ("R3",), required_symbols=("pkg.Entry.run",))
    empty = assess_selection_coverage(candidates, ())

    assert covered.complete is True
    assert missing.complete is False
    assert missing.reasons == ("1 required route symbol missing",)
    assert empty.complete is False
    assert empty.reasons == ("no candidate route selected",)


def test_obligation_coverage_fails_when_a_planned_obligation_lacks_proof():
    coverage = assess_obligation_coverage(
        ((1, 'pkg.Pipeline.run'), (2, 'pkg.Registrar.register')),
        represented_symbols={'pkg.Pipeline.run'})

    assert not coverage.complete
    assert any('C2' in reason for reason in coverage.reasons)


def test_obligation_coverage_passes_when_every_obligation_is_represented():
    coverage = assess_obligation_coverage(
        ((1, 'pkg.Pipeline.run'), (1, 'pkg.Step.apply')),
        represented_symbols={'pkg.Step.apply'})

    assert coverage.complete
    assert coverage.reasons == ()
