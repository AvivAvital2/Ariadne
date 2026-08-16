"""Deterministic claim accounting for answers formulated from SCIP evidence."""
from __future__ import annotations
import re

from dataclasses import dataclass, field
from library.relation_semantics import transition_verb


@dataclass(frozen=True)
class ClaimRecord:
    text: str
    locations: tuple[str, ...] = field(default_factory=tuple)
    supported: bool = False
    reason: str = ''


@dataclass(frozen=True)
class ClaimLedger:
    claims: tuple[ClaimRecord, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        return all(claim.supported for claim in self.claims)

    @property
    def gaps(self) -> tuple[str, ...]:
        return tuple(claim.reason for claim in self.claims if not claim.supported)

    def supported_answer(self) -> str:
        return '\n'.join(claim.text for claim in self.claims if claim.supported)
def _is_structural_prose(text: str) -> bool:
    stripped = text.strip().strip("#*:- ")
    return stripped.lower() in {
        "summary", "overview", "evidence", "verbatim scip evidence",
        "code chain", "answer",
    } or text.strip().startswith("```")


def validate_claims(answer: str, evidence) -> ClaimLedger:
    """Account every non-empty answer line against coordinates supplied to formulation."""
    from library.chain_answer import resolve_location

    claims: list[ClaimRecord] = []
    pattern = re.compile(r'([\w./-]+\.[A-Za-z]{1,8}:\d+)')
    for raw in (answer or '').splitlines():
        text = raw.strip()
        if not text:
            continue
        named = tuple(pattern.findall(text))
        if not named:
            if _is_structural_prose(text):
                continue
            reason = ("code claim has no evidence coordinate"
                      if "`" in text else "claim has no evidence coordinate")
            claims.append(ClaimRecord(text=text, reason=reason))
            continue
        unsupported = tuple(location for location in named
                            if resolve_location(location, evidence.locations) is None)
        if unsupported:
            claims.append(ClaimRecord(
                text=text, locations=named,
                reason=f'unsupported location: {unsupported[0]}'))
            continue
        claims.append(ClaimRecord(text=text, locations=named, supported=True))
    return ClaimLedger(claims=tuple(claims))
def repair_prompt(draft: str, chain_prompt: str, ledger: ClaimLedger) -> str:
    """Constrain one repair attempt to the original evidence and rejected reasons."""
    issues = '\n'.join(f'- {reason}' for reason in ledger.gaps)
    return (
        'Repair the draft using only the same compiler-derived evidence. '
        'Return only repaired claims, one claim per line, with file:line for every claim. '
        'Remove anything the evidence cannot prove.\n\n'
        f'Rejected reasons:\n{issues}\n\nDraft:\n{draft}\n\n{chain_prompt}')
def transition_claims(evidence) -> ClaimLedger:
    """Translate SCIP edges into claims without asking a language model to infer them."""
    claims: list[ClaimRecord] = []
    for hop in tuple(getattr(evidence, "bundle_citations", ()) or ()):
        if hop.relation == "localized" or not hop.parent_qualified_name:
            continue
        verb = transition_verb(hop.relation)
        call_site = f"{hop.call_site_file}:{hop.call_site_line}"
        definition = f"{hop.file}:{hop.line_start}"
        text = (
            f"{hop.parent_qualified_name} {verb} {hop.qualified_name} at {call_site}; "
            f"{hop.qualified_name} is defined at {definition}.")
        claims.append(ClaimRecord(
            text=text, locations=(call_site, definition), supported=True))
    return ClaimLedger(claims=tuple(claims))
def filter_supported(answer: str, evidence, ledger: ClaimLedger) -> tuple[str, ClaimLedger]:
    """Drop rejected draft lines and grade the exact answer that will be returned."""
    if ledger.valid:
        return answer, ledger
    supported = ledger.supported_answer()
    return supported, validate_claims(supported, evidence)
