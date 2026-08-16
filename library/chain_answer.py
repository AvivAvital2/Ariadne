"""The bridge into synthesis: what the LLM receives, and what the response carries.

``index -> fetch document -> curate bundle -> formulate with LLM -> return response``.
Stages one to three produce a chain and its documents; this renders them for stage four and
carries the coordinates into stage five.

The whole point is the inversion: **the evidence is the spine and prose is commentary.**
Before this, ``ask`` retrieved eight documents, concatenated them and synthesized, so no
answer could name a line and the store's 2.5M compiler-precise edges never reached the
model. Here the chain is the structure of the prompt, in execution order, and prose hangs
off individual hops.

Order is not cosmetic. ``MergeIntoCommand.runMerge``'s hops read as the MERGE algorithm —
metadata checks, then ``isInsertOnly``, then the three executors it chooses between — only
because they are ordered by call-site line. Ranking them by anything destroys that, which is
why nothing here reorders.

``unsupported_locations`` is the one guard: an answer may not name a ``file:line`` the bundle
does not contain. Deterministic, no judge, because the recurring failure is output that reads
correctly and cites something that was never there.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from library import Library
    from library.chain_bundle import ChainBundle

#: ``path/to/file.ext:123`` as it appears in prose.
_LOCATION = re.compile(r'\b([\w./-]+\.[A-Za-z]{1,6}):(\d+)\b')


@dataclass(frozen=True)
class AnswerEvidence:
    """Everything stage four needs and stage five returns."""

    spine: str = ''
    bundle_citations: list = field(default_factory=list)
    themes: list = field(default_factory=list)
    locations: frozenset = frozenset()
    unresolved_paths: tuple = ()
    truncation_reason: str = ''

    def citations(self) -> list[dict]:
        """The response payload — coordinates, not prose."""
        return [
            {
                'qualified_name': hop.qualified_name,
                'file': hop.file,
                'line': hop.line_start,
                'relation': hop.relation,
                'hop': hop.hop,
                'call_site': f'{hop.call_site_file}:{hop.call_site_line}',
                'stop_reason': hop.stop_reason,
            }
            for hop in self.bundle_citations
        ]


#: Characters of chain the prompt may carry. An LLM context window is a real constraint —
#: unlike a graph walk, where a count cap was wrong and truncated the chain itself.
#: Measured on the rebuilt databricks store: 8 retrieved documents give 279 hops and
#: ~18,285 tokens of spine at depth 3 (22,892 at depth 4), on top of the ~17k-token
#: document context, so unbounded the chain dominates the prompt it was meant to anchor.
DEFAULT_MAX_SPINE_CHARS = 20_000


def render_spine(bundle: 'ChainBundle',
                 max_chars: int = DEFAULT_MAX_SPINE_CHARS) -> str:
    """The chain as synthesis sees it: execution order, nested, coordinates on every hop.

    Indentation carries depth, so the shape of the call structure survives into the prompt
    rather than being flattened into a list.

    Two economies, because 60% of live hops carry no new structure — 100 ``plumbing`` and
    67 ``revisit`` of 279:

    * a ``revisit`` hop renders as one short line. Its body was already shown above, so
      repeating the definition is pure cost, but the **call site stays** because that is
      new evidence of a second invocation.
    * the whole spine is bounded, and the cut is a **prefix**: hops are dropped from the
      tail and the omission is stated. Ranking instead of truncating would destroy
      execution order, which is the one property that makes the chain explicable.
    """
    if not bundle.hops:
        return ''
    lines: list[str] = []
    used = 0
    omitted = 0
    for entry in bundle.hops:
        hop = entry.citation
        indent = '  ' * (hop.hop - 1)
        if hop.stop_reason == 'revisit':
            rendered = [f'{indent}{hop.qualified_name.rsplit(".", 1)[-1]} '
                        f'(already shown) called again at '
                        f'{hop.call_site_file}:{hop.call_site_line}']
        else:
            rendered = [
                f'{indent}{hop.qualified_name}  [{hop.file}:{hop.line_start}]'
                f'  called at {hop.call_site_file}:{hop.call_site_line}'
            ]
            if entry.evidence:
                rendered.append(f'{indent}    {entry.evidence.strip()}')
        cost = sum(len(line) + 1 for line in rendered)
        if used + cost > max_chars and lines:
            omitted = len(bundle.hops) - bundle.hops.index(entry)
            break
        lines += rendered
        used += cost
    if omitted:
        lines.append(f'... {omitted} further hop(s) omitted to fit the context; '
                     f'the chain continues beyond what is shown.')
    if bundle.themes:
        lines.append('')
        lines.append('This chain runs through:')
        for theme in bundle.themes:
            breadth = '' if theme.coherent else ' (broad)'
            lines.append(f'  {theme.hops} hop(s) in "{theme.title}"{breadth}')
    return '\n'.join(lines)


def evidence_for(
    library: 'Library',
    documents,
    *,
    source: str,
    depth: int = 3,
    max_evidence_chars: int | None = None,
    max_spine_chars: int = DEFAULT_MAX_SPINE_CHARS,
) -> AnswerEvidence:
    """Run stages one to three over what retrieval returned.

    Seeds come from the retrieved documents — the localization step measured at production
    width, which takes required-slot reach from 35% to 59% — and the walk expands them,
    because a seed naming a type has no outgoing call edge of its own.
    """
    from docgen.scip_paths import indexer_cwds
    from library.chain_bundle import DEFAULT_MAX_EVIDENCE_CHARS, curate_bundle
    from library.structural_assembly import chain_from_seeds, seeds_from_documents

    root = None
    try:
        from config import get_config
        root = str(get_config().get_all_source_paths().get(source) or '') or None
    except Exception:  # noqa: BLE001 — config is optional for a scoped call
        root = None
    cwds = indexer_cwds(root) if root else ()

    with library._conn_provider.acquire() as conn:
        seed_set = seeds_from_documents(
            conn, documents, source=source, indexer_cwds=cwds, source_root=root)
        # ONE walk over every seed. Walking them independently cost ~67s for a single
        # `ask` at production width (8 documents -> 195 seeds), because each seed
        # re-expanded members and re-walked ground the previous ones had covered.
        citations, _truncation = chain_from_seeds(
            conn, seed_set.seeds, source=source, depth=depth)

    if not citations:
        return AnswerEvidence(unresolved_paths=seed_set.unresolved_paths)

    bundle = curate_bundle(
        library, citations, source=source,
        max_evidence_chars=(max_evidence_chars
                            if max_evidence_chars is not None
                            else DEFAULT_MAX_EVIDENCE_CHARS))
    spine_shown = render_spine(bundle, max_spine_chars)
    # Locations stay the FULL bundle: a hop cut from the prompt is still real, so an
    # answer naming it must not be called a fabrication.
    locations = frozenset(
        f'{hop.citation.file}:{hop.citation.line_start}' for hop in bundle.hops)
    return AnswerEvidence(
        spine=spine_shown,
        bundle_citations=[hop.citation for hop in bundle.hops],
        themes=list(bundle.themes),
        locations=locations,
        unresolved_paths=seed_set.unresolved_paths,
        truncation_reason=_why_truncated(bundle, spine_shown),
    )


def unsupported_locations(answer: str, evidence: AnswerEvidence) -> tuple[str, ...]:
    """Every ``file:line`` the answer names that the bundle does not contain.

    The line is part of the claim, so the right file at the wrong line is unsupported —
    the same standard the eval's admissibility gate applies. An answer naming no location
    is not reported: saying nothing locational is honest, inventing one is not.
    """
    return tuple(
        found for found in
        (f'{m.group(1)}:{m.group(2)}' for m in _LOCATION.finditer(answer or ''))
        if found not in evidence.locations
    )


def _why_truncated(bundle: 'ChainBundle', spine: str) -> str:
    """What the caller should know was left out, if anything."""
    reasons = []
    if 'omitted to fit the context' in spine:
        reasons.append('chain truncated')
    if bundle.evidence_omitted:
        reasons.append('evidence omitted')
    return '; '.join(reasons)
