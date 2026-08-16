"""Every returned code claim is accounted against supplied SCIP evidence."""
from __future__ import annotations

from library.chain_answer import AnswerEvidence
from library.claim_validation import repair_prompt, validate_claims


def test_claim_ledger_accounts_support_repairs_and_drops_unproven_lines():
    evidence = AnswerEvidence(locations=frozenset({'src/a.py:7', 'src/b.py:11'}))
    draft = '\n'.join([
        'The caller invokes the writer at src/a.py:7.',
        'The database commits at invented.py:99.',
        'The cache is always durable.',
    ])

    ledger = validate_claims(draft, evidence)

    assert [claim.supported for claim in ledger.claims] == [True, False, False]
    assert ledger.claims[0].locations == ('src/a.py:7',)
    assert ledger.claims[1].reason == 'unsupported location: invented.py:99'
    assert ledger.claims[2].reason == 'claim has no evidence coordinate'
    assert not ledger.valid
    assert ledger.supported_answer() == 'The caller invokes the writer at src/a.py:7.'
    prompt = repair_prompt(draft, 'CHAIN EVIDENCE', ledger)
    assert 'CHAIN EVIDENCE' in prompt
    assert 'invented.py:99' in prompt
    assert 'Return only repaired claims' in prompt
def test_blank_answer_has_no_unaccounted_claims():
    ledger = validate_claims("\n  \n", AnswerEvidence())
    assert ledger.valid
    assert ledger.claims == ()
    assert ledger.gaps == ()
def test_transition_claims_are_deterministically_derived_from_scip_edges():
    from library.chain_answer import locations_for
    from library.claim_validation import transition_claims
    from library.structural_assembly import StructuralCitation
    citation = StructuralCitation(
        qualified_name="db.Writer.commit", file="db/writer.py", line_start=20,
        source_name="src", relation="calls", hop=1,
        call_site_file="api/endpoint.py", call_site_line=11,
        parent_qualified_name="api.Endpoint.post", line_end=30)
    evidence = AnswerEvidence(
        bundle_citations=[citation], locations=locations_for([citation]))

    ledger = transition_claims(evidence)

    assert ledger.valid
    assert ledger.claims[0].text == (
        "api.Endpoint.post calls db.Writer.commit at api/endpoint.py:11; "
        "db.Writer.commit is defined at db/writer.py:20.")
    assert ledger.claims[0].locations == ("api/endpoint.py:11", "db/writer.py:20")
def test_claim_ledger_ignores_explanatory_structure_but_rejects_unanchored_code_claims():
    evidence = AnswerEvidence(locations=frozenset({"src/worker.py:12"}))
    ledger = validate_claims(
        "Summary\n`Worker.run` calls `Store.save`.\n"
        "The transition is shown at src/worker.py:12.", evidence)

    assert len(ledger.claims) == 2
    assert ledger.claims[0].supported is False
    assert ledger.claims[0].reason == "code claim has no evidence coordinate"
    assert ledger.claims[1].supported is True
def test_filter_supported_revalidates_the_answer_that_is_actually_returned():
    from library.claim_validation import filter_supported

    evidence = AnswerEvidence(locations=frozenset({"src/worker.py:12"}))
    draft = "Supported at src/worker.py:12.\nUnsupported assertion."
    original = validate_claims(draft, evidence)

    answer, returned = filter_supported(draft, evidence, original)

    assert answer == "Supported at src/worker.py:12."
    assert returned.valid
    assert len(returned.claims) == 1
    assert returned.claims[0].supported
def test_transition_claims_preserve_structural_relation_and_reverse_direction():
    from library.chain_answer import AnswerEvidence, locations_for
    from library.claim_validation import transition_claims
    from library.structural_assembly import StructuralCitation

    citations = [
        StructuralCitation(
            qualified_name="pkg.Owner.member", file="core/owner.py", line_start=8,
            source_name="src", relation="contains", hop=1,
            call_site_file="core/owner.py", call_site_line=8,
            parent_qualified_name="pkg.Owner", line_end=12),
        StructuralCitation(
            qualified_name="pkg.Caller.run", file="api/caller.py", line_start=3,
            source_name="src", relation="called_by", hop=0,
            call_site_file="api/caller.py", call_site_line=7,
            parent_qualified_name="pkg.Target.execute", line_end=9),
        StructuralCitation(
            qualified_name="pkg.Reader.use", file="api/reader.py", line_start=4,
            source_name="src", relation="referenced_by", hop=0,
            call_site_file="api/reader.py", call_site_line=6,
            parent_qualified_name="pkg.Model.id", line_end=8),
    ]
    evidence = AnswerEvidence(
        bundle_citations=citations, locations=locations_for(citations))

    claims = [claim.text for claim in transition_claims(evidence).claims]

    assert claims[0].startswith("pkg.Owner contains pkg.Owner.member at ")
    assert claims[1].startswith("pkg.Target.execute is called by pkg.Caller.run at ")
    assert claims[2].startswith("pkg.Model.id is referenced by pkg.Reader.use at ")
    assert all("pkg.Owner calls pkg.Owner.member" not in claim for claim in claims)
