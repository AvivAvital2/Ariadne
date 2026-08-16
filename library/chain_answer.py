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
    #: The curated hops themselves, so a consumer can offer them as a menu before
    #: spending them (:mod:`library.chain_menu`). ``bundle_citations`` stays because the
    #: response payload is built from coordinates alone.
    hops: tuple = ()
    #: Dispatches the walk reported instead of expanding — see
    #: :func:`library.chain_disclosure.describe_fan_out`. Structured, not prose: the
    #: wording belongs to the presentation layer.
    fan_outs: tuple = ()
    def citations(self) -> list[dict]:
        """The response payload — coordinates, not prose, and each one once.

        Deduplicated by definition. Measured at production width the chain reached 2,645
        hops across 973 distinct symbols, and returning all 2,645 shipped 939,389 characters
        (~235,000 tokens) to the caller for one question. A definition reached from a second
        call site is new evidence *in the spine*, where the call site is shown; in a list of
        coordinates it is the same coordinate twice.
        """
        payload: list[dict] = []
        seen: set[tuple[str, str, int]] = set()
        for hop in self.bundle_citations:
            key = (hop.qualified_name, hop.file, hop.line_start)
            if key in seen:
                continue
            seen.add(key)
            payload.append({
                'qualified_name': hop.qualified_name,
                'file': hop.file,
                'line': hop.line_start,
                'relation': hop.relation,
                'hop': hop.hop,
                'call_site': f'{hop.call_site_file}:{hop.call_site_line}',
                'stop_reason': hop.stop_reason,
            })
        return payload

    def cited_by(self, answer: str) -> list[dict]:
        """Only the coordinates the answer actually used, in chain order.

        What a caller needs is what the claims rest on. The rest of the chain is described by
        :meth:`summary` rather than enumerated — an answer names a few dozen locations out of
        the hundreds the walk covered.
        """
        resolved = {
            resolve_location(f'{match.group(1)}:{match.group(2)}', self.locations)
            for match in _LOCATION.finditer(answer or '')}
        resolved.discard(None)
        # Either end of the edge: an answer that cites the call site is pointing at this hop
        # just as surely as one that cites the definition.
        return [entry for entry in self.citations()
                if f'{entry["file"]}:{entry["line"]}' in resolved
                or entry['call_site'] in resolved]

    def summary(self) -> dict:
        """The chain's shape, for a caller who should not be handed every coordinate."""
        return {
            'hops': len(self.bundle_citations),
            'symbols': len({hop.qualified_name for hop in self.bundle_citations}),
            'files': len({hop.file for hop in self.bundle_citations}),
            'locations': len(self.locations),
            'forks': [
                {
                    'qualified_name': fan_out.qualified_name,
                    'file': fan_out.file,
                    'line': fan_out.line_start,
                    'implementations_in_index': fan_out.implementations,
                }
                for fan_out in self.fan_outs
            ],
            'truncation': self.truncation_reason,
        }

ANSWER_MAX_TOKENS = 1024


def spine_budget_chars(model: 'str | None' = None, *,
                       reserved_output_tokens: int = ANSWER_MAX_TOKENS) -> 'int | None':
    """Characters of chain the prompt may carry: whatever the context window leaves.

    Derived, not declared. The number this replaced -- 20,000 characters -- was set when
    the prompt still carried the retrieved documents beside the chain; measured on a
    production question, those eight documents were 15,754 tokens of prose. Nothing but
    the chain travels now, so that premise is void, and the only claimants on the window
    are the answer reserved inside it and the instructions that frame it.

    ``None`` means the window is unknown, and then nothing is cut. A model absent from
    :data:`docgen.pricing.MODEL_CONTEXT_WINDOW_TOKENS` is reported rather than assigned a
    believable limit, and an over-long prompt fails at the API -- the boundary that owns
    the rule -- instead of being quietly trimmed here.

    This bounds the prompt; it does not make the chain the right size. Measured at
    production width (8 retrieved documents, source ``databricks``, depth 3) the uncapped
    spine is 2,645 hops and ~1.34M tokens -- down from 25,313 hops and ~13.9M once prose
    stopped seeding the walk and type references stopped being traversed -- so a 1M-token
    window admits it whole. What each hop contributes is its ``catalog`` description —
    measured, 883 distinct documents for 2,645 hops, ~88,600 tokens — and its coordinates.
    """
    from docgen.pricing import (
        CHARS_PER_TOKEN,
        PROMPT_OVERHEAD_TOKENS,
        context_window_tokens,
    )
    if model is None:
        try:
            from config import get_config
            model = get_config().model
        except Exception:  # noqa: BLE001 -- a scoped call need not have a config
            return None
    window = context_window_tokens(model or '')
    if window is None:
        return None
    return int(max(window - reserved_output_tokens - PROMPT_OVERHEAD_TOKENS, 0)
               * CHARS_PER_TOKEN)
#: A line reference with no file: ``(and again at :166)``. The colon must not follow a path
#: or word character, which is what separates it from ``File.scala:166``, ``12:30`` and
#: ``ratio 3:4``.
_BARE_LINE = re.compile(r'(?<![\w./-]):(\d+)\b')


def expand_bare_lines(answer: str) -> str:
    """Give every bare ``:166`` the file named before it.

    An answer writes "invoked at MergeIntoCommand.scala:130 (and again at :166)". The second
    reference is a real claim about a real line, and it inherits its file from the first —
    but :data:`_LOCATION` requires ``file.ext:line``, so as written it is invisible to
    :func:`unsupported_locations` and to :meth:`AnswerEvidence.cited_by`. It is neither
    checked nor returned to the caller.

    Done here rather than in the prompt: instructing the model to repeat the file name spends
    tokens on every answer and depends on it complying. Rewriting afterwards costs nothing
    and always holds.

    A bare reference is expanded **whether or not the result resolves**. Expanding only the
    ones that check out would hide a wrong claim from the guard, which is backwards — the
    point is that every coordinate the answer states gets verified.
    """
    if not answer:
        return answer
    result: list[str] = []
    position = 0
    antecedent = ''
    for match in re.finditer(r'([\w./-]+\.[A-Za-z]{1,6}):(\d+)|(?<![\w./-]):(\d+)\b',
                             answer):
        if match.group(1):                      # a full coordinate: it sets the antecedent
            antecedent = match.group(1)
            continue
        if not antecedent:                      # nothing to inherit from
            continue
        result.append(answer[position:match.start()])
        result.append(f'{antecedent}:{match.group(3)}')
        position = match.end()
    result.append(answer[position:])
    return ''.join(result)


def locations_for(hops) -> frozenset:
    """Every coordinate the spine puts in front of the model — definitions and call sites.

    Both are admissible because both are shown. The prompt renders the site as ``called at file:line`` for a call
    edge and ``referenced at file:line`` for a type reference, and tells the model that
    site is what the index recorded, so citing either is citing what the chain showed. A live run cited five call sites and the guard reported all five as
    invented, because this set held definitions only.
    """
    coordinates: set[str] = set()
    for entry in hops:
        citation = getattr(entry, 'citation', entry)
        coordinates.add(f'{citation.file}:{citation.line_start}')
        if citation.call_site_file:
            coordinates.add(f'{citation.call_site_file}:{citation.call_site_line}')
    return frozenset(coordinates)


def resolve_location(named: str, locations) -> 'str | None':
    """The coordinate ``named`` refers to, or ``None`` when that cannot be decided.

    An answer writes ``InsertOnlyMergeExecutor.scala:53``; the index stores a 78-character
    path. A file name and line matching exactly one known coordinate **is** that coordinate,
    and the index is what says so. Two matches cannot be pinned, so they are reported rather
    than assumed — the point of this check is that a location is verified, not believed.

    Measured on a live answer: 11 of 11 shortened citations resolved to exactly one
    coordinate, and every one had been reported as a fabrication before this existed.
    """
    if named in locations:
        return named
    suffix = f'/{named}'
    matches = [known for known in locations if known.endswith(suffix)]
    return matches[0] if len(matches) == 1 else None


def render_spine(bundle: 'ChainBundle', max_chars: 'int | None' = None) -> str:
    """The chain as synthesis sees it: execution order, nested, coordinates on every hop.

    Indentation carries depth, so the shape of the call structure survives into the prompt
    rather than being flattened into a list.

    Two economies, because not every hop carries a body worth repeating -- measured at
    production width, 157 of 2,645 hops are ``revisit``:

    * a ``revisit`` hop renders as one short line. Its body was already explained above,
      so repeating the definition is pure cost, but the **call site stays** because that is
      new evidence of a second invocation. For the same reason a symbol's description is
      rendered once however many hops reach it, while every call site is kept.
    * the spine is bounded by what the context window leaves (:func:`spine_budget_chars`),
      and the cut is a **prefix**: hops are dropped from the tail and the omission is
      stated. Ranking instead of truncating would destroy execution order, which is the one
      property that makes the chain explicable. ``max_chars=None`` cuts nothing, which is
      what an unknown window gets.
    """
    if not bundle.hops:
        return ''
    lines: list[str] = []
    described: set[str] = set()
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
            site = 'referenced at' if hop.relation == 'references' else 'called at'
            rendered = [
                f'{indent}{hop.qualified_name}  [{hop.file}:{hop.line_start}]'
                f'  {site} {hop.call_site_file}:{hop.call_site_line}'
            ]
            # A description belongs to a symbol, not to a hop. The same type reached from
            # four bodies is four hops — the call sites are four separate pieces of evidence
            # — but its description does not change between them, and at production width
            # 1,788 document-carrying hops held only 883 distinct documents. This is the rule
            # ``revisit`` applies to a body, applied to a description.
            if entry.evidence and hop.qualified_name not in described:
                described.add(hop.qualified_name)
                rendered.append(f'{indent}    {entry.evidence.strip()}')
        cost = sum(len(line) + 1 for line in rendered)
        if max_chars is not None and used + cost > max_chars and lines:
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
    max_spine_chars: 'int | None' = None,
    clew_symbols=(),
) -> AnswerEvidence:
    """Run stages one to three over what retrieval returned.

    Seeds come from the retrieved documents — the localization step measured at production
    width, which takes required-slot reach from 35% to 59% — and the walk expands them,
    because a seed naming a type has no outgoing call edge of its own.

    ``clew_symbols`` adds the qualified names of a route retrieval matched (see
    :mod:`library.clews`). A document tells the walk which *file* to start in; a clew tells it
    which *path*, which is the one thing document seeding cannot express — the walk itself has
    never seen the question. Measured on the databricks pack, pooled clew strategies contain
    92.8% of the symbols answer keys require against 66.0% for one document-seeded walk.

    Additive, deliberately: clew seeds are unioned with document seeds rather than replacing
    them, so a badly matched clew widens the walk instead of misplacing it. The same rule the
    menu follows — a bad selection costs tokens, never evidence.
    """
    from docgen.scip_paths import indexer_cwds
    from library.chain_bundle import curate_bundle
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
        seeds = list(seed_set.seeds)
        if clew_symbols:
            # Resolved by name, and scoped in Python: `source_name` holds few distinct values
            # over hundreds of thousands of rows, so naming it in the WHERE clause makes
            # SQLite prefer that index and scan (`_locate` records the 1900x measurement).
            listed = sorted({str(name) for name in clew_symbols})
            for start in range(0, len(listed), 300):
                chunk = listed[start:start + 300]
                placeholders = ','.join('?' * len(chunk))
                seeds += [row[0] for row in conn.execute(
                    f'SELECT canonical_id, source_name FROM scip_symbols '
                    f'WHERE qualified_name IN ({placeholders}) '
                    f"AND canonical_id NOT LIKE 'local %'", chunk)
                    if row[1] == source]
        # ONE walk over every seed. Walking them independently cost ~67s for a single
        # `ask` at production width (8 documents -> 195 seeds), because each seed
        # re-expanded members and re-walked ground the previous ones had covered.
        citations, truncation = chain_from_seeds(
            conn, seeds, source=source, depth=depth)

    if not citations:
        return AnswerEvidence(unresolved_paths=seed_set.unresolved_paths,
                              fan_outs=truncation.fan_outs)

    bundle = curate_bundle(library, citations, source=source)
    budget = (max_spine_chars if max_spine_chars is not None
              else spine_budget_chars())
    spine_shown = render_spine(bundle, budget)
    # Locations stay the FULL bundle: a hop cut from the prompt is still real, so an
    # answer naming it must not be called a fabrication.
    locations = locations_for(bundle.hops)
    return AnswerEvidence(
        spine=spine_shown,
        bundle_citations=[hop.citation for hop in bundle.hops],
        themes=list(bundle.themes),
        locations=locations,
        unresolved_paths=seed_set.unresolved_paths,
        truncation_reason=_why_truncated(bundle, spine_shown),
        # a fork the walk declined to expand is evidence about the code, so it travels
        fan_outs=truncation.fan_outs,
        hops=tuple(bundle.hops),
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
        if resolve_location(found, evidence.locations) is None
    )


def _why_truncated(bundle: 'ChainBundle', spine: str) -> str:
    """What the caller should know was left out, if anything."""
    # One budget, one report. Curation attaches every explained hop; the spine is the only
    # stage that drops anything, and it says so in the text it returns.
    return 'chain truncated' if 'omitted to fit the context' in spine else ''
