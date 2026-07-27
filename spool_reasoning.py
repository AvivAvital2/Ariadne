"""Reasoning-path machinery (slice f1).

The §7 flow with the LLM behind explicit seams: decompose along the
Spool-supplied concern taxonomy (#2) → assemble the capability profile
at reasoning time (#1 — probes and spans-matched guardrails, all
through the e1 signal evaluator, evidence-bearing) → consistency-pass
the selected methods (#4 — requires/provides token matching; unmet =
a consult point for the USER, never auto-resolved) → compose the
serializable construction plan. Selection itself is an explicit input
(f2's LLM fills it). Design: designs/spool-environment-plugin.md §7 · §9.
"""
from dataclasses import dataclass
from pathlib import Path

from guardrails import GuardrailResult, evaluate_guardrail, evaluate_signal

BASELINE_PROVIDER = 'baseline'


class ReasoningError(Exception):
    """A reasoning-input violation — always loud, never masked."""


# ---------------------------------------------------------------- taxonomy

@dataclass(frozen=True)
class Concern:
    """One dimension that matters for this environment (§9 #2)."""
    name: str
    description: str = ''
    probes: tuple = ()          # e1 signal dicts, evaluated structurally


@dataclass(frozen=True)
class ConcernTaxonomy:
    environment: str
    concerns: tuple


def load_taxonomy(path) -> ConcernTaxonomy:
    """Load a Spool's concern taxonomy YAML; loud on malformed concerns."""
    from spools import load_yaml_mapping
    data = load_yaml_mapping(path, ReasoningError)
    concerns = []
    for raw in data.get('concerns') or []:
        if not (raw or {}).get('name'):
            raise ReasoningError(
                f'taxonomy {path}: every concern needs a name, got {raw!r}',
            )
        concerns.append(Concern(
            name=raw['name'],
            description=raw.get('description', ''),
            probes=tuple(raw.get('probes') or ()),
        ))
    if not data.get('environment') or not concerns:
        raise ReasoningError(
            f'taxonomy {path} must declare `environment` and at least '
            f'one concern',
        )
    return ConcernTaxonomy(
        environment=str(data['environment']),
        concerns=tuple(concerns),
    )


# ------------------------------------------------------- capability profile

@dataclass(frozen=True)
class ConcernFinding:
    """What the user's code shows along one concern: probe evidence plus
    the guardrails whose ``spans`` hit the concern, each evaluated."""
    concern: Concern
    probe_results: tuple = ()
    guardrails: tuple = ()      # GuardrailResult


@dataclass(frozen=True)
class CapabilityProfile:
    """Assembled at reasoning time along the taxonomy (§9 #1) — never a
    fixed, environment-agnostic extraction."""
    environment: str
    findings: tuple

    def finding(self, concern_name: str) -> ConcernFinding:
        for finding in self.findings:
            if finding.concern.name == concern_name:
                return finding
        raise ReasoningError(
            f'no concern {concern_name!r} in the {self.environment} profile',
        )


def assemble_capability_profile(
    taxonomy: ConcernTaxonomy, guardrails, view,
) -> CapabilityProfile:
    """Project the code's catalog along the taxonomy's concerns."""
    findings = []
    for concern in taxonomy.concerns:
        probe_results = tuple(
            evaluate_signal(probe, view) for probe in concern.probes
        )
        matched = tuple(
            evaluate_guardrail(g, view)
            for g in guardrails
            if concern.name in g.spans
        )
        findings.append(ConcernFinding(
            concern=concern,
            probe_results=probe_results,
            guardrails=matched,
        ))
    return CapabilityProfile(
        environment=taxonomy.environment, findings=tuple(findings),
    )


# --------------------------------------------------------- consistency (#4)

@dataclass(frozen=True)
class ConsultPoint:
    """An unmet requirement — surfaced to the user, never auto-resolved."""
    guardrail: str
    requirement: str


@dataclass(frozen=True)
class ConsistencyReport:
    met: dict                    # requirement -> provider guardrail | 'baseline'
    consult_points: tuple


def consistency_pass(selected, *, baseline: frozenset) -> ConsistencyReport:
    """Check every selected method's ``requires`` against the others'
    ``provides`` and the baseline facts (§9 #4)."""
    providers = {}
    for guardrail in selected:
        for token in guardrail.provides:
            providers.setdefault(token, guardrail.name)
    met = {}
    consult_points = []
    for guardrail in selected:
        for requirement in guardrail.requires:
            if requirement in providers:
                met[requirement] = providers[requirement]
            elif requirement in baseline:
                met[requirement] = BASELINE_PROVIDER
            else:
                consult_points.append(ConsultPoint(
                    guardrail=guardrail.name, requirement=requirement,
                ))
    return ConsistencyReport(met=met, consult_points=tuple(consult_points))


# ------------------------------------------------------------------- plan

@dataclass(frozen=True)
class PlanStep:
    concern: str
    selected: str | None
    why: str = ''
    evidence: tuple = ()


@dataclass(frozen=True)
class ConstructionPlan:
    """The grounded, symbol-cited output shape (§8 v1). Consult points are
    data; presentation is the §17 surface."""
    goal: str
    environment: str
    steps: tuple
    consult_points: tuple

    def step(self, concern_name: str) -> PlanStep:
        for step in self.steps:
            if step.concern == concern_name:
                return step
        raise ReasoningError(f'no plan step for concern {concern_name!r}')

    def to_dict(self) -> dict:
        return {
            'goal': self.goal,
            'environment': self.environment,
            'steps': [
                {
                    'concern': s.concern,
                    'selected': s.selected,
                    'why': s.why,
                    'evidence': list(s.evidence),
                }
                for s in self.steps
            ],
            'consult_points': [
                {'guardrail': c.guardrail, 'requirement': c.requirement}
                for c in self.consult_points
            ],
        }


def render_plan_markdown(plan, *, provenance=None) -> str:
    """The one output shape (§19.1) — interactive callers relay the
    Decisions-needed items as questions; headless callers ship this as-is."""
    lines = [f'# Plan: {plan.goal}', '']
    for step in plan.steps:
        lines.append(f'## {step.concern}')
        if step.selected is None:
            lines.append('No method selected for this concern.')
        else:
            lines.append(f'**Selected:** `{step.selected}`')
            lines.append(f'**Why:** {step.why}')
            for evidence in step.evidence:
                lines.append(f'- {evidence}')
        lines.append('')
    if plan.consult_points:
        lines.append('## Decisions needed')
        for point in plan.consult_points:
            lines.append(
                f'- `{point.guardrail}` requires **{point.requirement}** '
                f'— nothing selected provides it; confirm how it will be '
                f'satisfied.',
            )
        lines.append('')
    lines.append('---')
    lines.append(
        '_Evidence confidence: verified (structural) — every citation was '
        'checked against the indexed code._',
    )
    if provenance:
        lines.append(f'_Provenance: {provenance}_')
    return '\n'.join(lines)


def render_selection_request(goal, profile) -> str:
    """The compact digest the selection LLM sees (§19.3): candidates with
    their contracts, plus the strict-JSON output instructions."""
    lines = [
        f'Goal: {goal}',
        f'Environment: {profile.environment}',
        '',
        'For each concern below, pick the best-fitting candidate method '
        'by name, or null when none applies. You may ONLY name listed '
        'candidates.',
        '',
    ]
    for finding in profile.findings:
        lines.append(f'concern: {finding.concern.name} — '
                     f'{finding.concern.description}')
        for result in finding.guardrails:
            g = result.guardrail
            lines.append(
                f'  candidate: {g.name} ({g.kind}; fired={result.result.fired})',
            )
            lines.append(f'    recommendation: {g.recommendation}')
            if g.requires:
                lines.append(f"    requires: {', '.join(g.requires)}")
            if g.provides:
                lines.append(f"    provides: {', '.join(g.provides)}")
        lines.append('')
    lines.append(
        'Reply with ONLY a JSON object of the shape '
        '{"selections": {"<concern>": "<candidate-name>" | null, ...}, '
        '"assumed_baseline": ["<token>", ...], "notes": "<string>"}.',
    )
    return '\n'.join(lines)


@dataclass(frozen=True)
class SelectionResult:
    selections: dict
    assumed_baseline: tuple = ()
    notes: str = ''


def _parse_selection_reply(reply: str, concern_names, catalog_names) -> SelectionResult:
    import json

    try:
        data = json.loads(reply)
    except (TypeError, ValueError) as exc:
        raise ReasoningError(f'reply is not valid JSON: {exc}') from exc
    if not isinstance(data, dict) or not isinstance(data.get('selections'), dict):
        raise ReasoningError('reply must be a JSON object with a '
                             '"selections" mapping')
    selections = data['selections']
    for concern, name in selections.items():
        if concern not in concern_names:
            raise ReasoningError(f'unknown concern {concern!r}')
        if name is not None and name not in catalog_names:
            raise ReasoningError(
                f'selection {name!r} for {concern!r} is not a listed '
                f'candidate — only catalog methods may be selected',
            )
    return SelectionResult(
        selections=dict(selections),
        assumed_baseline=tuple(data.get('assumed_baseline') or ()),
        notes=str(data.get('notes') or ''),
    )


def select_methods(goal, profile, guardrails, *, llm,
                   max_repairs: int = 1) -> SelectionResult:
    """The (f2) selection loop (§19.3): render the digest, call the LLM,
    validate strictly; one repair re-prompt carrying the error; then loud.

    Catalog-only selection is the structural row-sharding protection: a
    method the catalog doesn't contain cannot be selected, whatever the
    LLM replies.
    """
    concern_names = {f.concern.name for f in profile.findings}
    catalog_names = {g.name for g in guardrails}
    prompt = render_selection_request(goal, profile)
    reply = llm(prompt)
    last_error = None
    for attempt in range(max_repairs + 1):
        try:
            return _parse_selection_reply(reply, concern_names, catalog_names)
        except ReasoningError as exc:
            last_error = exc
            if attempt < max_repairs:
                reply = llm(
                    f'{prompt}\n\nYour previous reply was invalid: {exc}. '
                    f'Reply with ONLY the JSON object.'
                )
    raise ReasoningError(
        f'selection reply still invalid after {max_repairs} repair(s): '
        f'{last_error}',
    )


def compose_plan(*, goal, profile: CapabilityProfile, selections: dict,
                 guardrails, baseline: frozenset) -> ConstructionPlan:
    """Assemble the plan from explicit per-concern selections.

    ``selections`` maps concern name -> selected guardrail name (or None
    for "nothing to do on this concern"). The why is the guardrail's
    rationale + recommendation; evidence comes from the profile's
    evaluation. The consistency pass runs over the selected methods.
    """
    by_name = {g.name: g for g in guardrails}
    steps = []
    selected_guardrails = []
    for finding in profile.findings:
        name = selections.get(finding.concern.name)
        if name is None:
            steps.append(PlanStep(concern=finding.concern.name, selected=None))
            continue
        guardrail = by_name.get(name)
        if guardrail is None:
            raise ReasoningError(
                f'selection for {finding.concern.name!r} names unknown '
                f'guardrail {name!r}',
            )
        # A method may serve several concerns; its requires/provides
        # enter the consistency pass once (no duplicate consult points).
        if guardrail not in selected_guardrails:
            selected_guardrails.append(guardrail)
        evidence = tuple(
            e
            for result in finding.guardrails
            if result.guardrail.name == name
            for e in result.result.evidence
        )
        steps.append(PlanStep(
            concern=finding.concern.name,
            selected=name,
            why=f'{guardrail.recommendation} — {guardrail.rationale}',
            evidence=evidence,
        ))
    report = consistency_pass(selected_guardrails, baseline=baseline)
    return ConstructionPlan(
        goal=goal,
        environment=profile.environment,
        steps=tuple(steps),
        consult_points=report.consult_points,
    )
