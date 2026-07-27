"""Spool lens router — pure routing rules (designs/spool-lens-router.md §3).

Composes with the aisle theme prefilter (root ``spool_router``): the
prefilter answers WHICH spool(s) a question could touch (the north-star §5
"intrinsic signature" tier); this module answers HOW the picked spool
participates — the regime. Deterministic: no I/O, no embeddings, no LLM.
Every rule here was ratified from the measured router-signal battery
(10/10 regimes on live data).
"""
import re

from attrs import frozen

from library.word_tokens import segment_word_tokens

REPO_ONLY = 'repo-only'
FUSE = 'fuse'
EXPERT_ONLY = 'expert-only'
HONEST_GAP = 'honest-gap'

_STRONG_CLASSES = frozenset({'api', 'product'})


@frozen
class EntityHit:
    """One crisp resolution of a question term against one side's entity
    index: which layer matched (symbol/title/heading/alias) and the term's
    class (api/product/phrase — expert-only weighs classes, §3 rule 3)."""
    term: str
    layer: str
    entity_class: str


@frozen
class RouteResult:
    """The router's decision plus the evidence it stands on — the labeled
    synthesis cites the crisp hits, and telemetry logs them for
    calibration."""
    regime: str
    crisp_repo: tuple
    crisp_spool: tuple
    subject_named: bool
    fallback_enabled: bool


def is_distinctive(term: str) -> bool:
    """Only distinctive terms may resolve: multi-word phrases and
    symbol-shaped tokens (case-mixed or underscored). Bare common lowercase
    words ('pipeline', 'memory') never route — measured, they false-fused
    the surface-vocabulary battery traps."""
    term = term.strip()
    if ' ' in term:
        return True
    return '_' in term or (term != term.lower() and term != term.upper())


def classify_entity(term: str, layer: str) -> str:
    """api: a single symbol-shaped token resolved via the symbol layer.
    product: capitalized word(s) — proper-noun shaped — via any layer.
    phrase: any other distinctive term, WHATEVER layer resolved it (a
    lowercase property phrase that happens to match a symbol is still a
    phrase for weighting). Expert-only requires a strong (api/product)
    spool hit and no strong repo hit; phrases never decide."""
    words = term.split()
    if layer == 'symbol' and len(words) == 1:
        return 'api'
    if words and all(w[:1].isupper() for w in words):
        return 'product'
    return 'phrase'


def symbol_matches(term: str, qualified_names, limit: int = 3) -> list:
    """The qualified names ``term`` resolves to, under the rule-1 structural
    tiers: exact last segment, case-folded last segment, exact inner
    segment, or term-words a subset of the last segment's words (two or
    more term words — a one-word fragment is not entity resolution). NEVER
    typo-distance — the suggestion matcher's difflib tier exists for humans
    recovering from typos, and misusing it as a resolver mis-fused 4/10
    battery questions. Retrieval admits the matched elements' docs; the
    router only needs :func:`match_symbol`, the boolean view."""
    term = term.strip()
    forms = {term}
    if ' ' in term:
        forms.add(term.replace(' ', '_'))
        forms.add(term.replace(' ', ''))
    term_tokens = segment_word_tokens(term)
    out = []
    for qn in qualified_names:
        segments = qn.split('.')
        last = segments[-1]
        matched = any(
            last == f or last.lower() == f.lower() or f in segments[:-1]
            for f in forms
        ) or (
            len(term_tokens) >= 2
            and term_tokens <= segment_word_tokens(last)
        )
        if matched:
            out.append(qn)
            if limit is not None and len(out) >= limit:
                break
    return out


def match_symbol(term: str, qualified_names) -> bool:
    """Boolean view of :func:`symbol_matches` — same tiers, first hit wins."""
    return bool(symbol_matches(term, qualified_names, limit=1))


_STOP_WORDS = frozenset(
    'a an and are as at be by can do does for from how i in is it my of on '
    'or our should that the this to we what when where which will with you '
    'your'.split()
)

_WORD_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_'-]*")


def extract_candidate_terms(question: str, max_words: int = 3) -> list:
    """Deterministic entity-candidate extraction — question word n-grams
    (longest first), trimmed at stop-word boundaries, filtered to
    distinctive terms. This replaces any LLM in routing: garbage n-grams
    are harmless because resolution against the entity index is the real
    gate."""
    words = _WORD_RE.findall(question)
    out, seen = [], set()
    for n in range(min(max_words, len(words)), 0, -1):
        for i in range(len(words) - n + 1):
            gram = words[i:i + n]
            if (gram[0].lower() in _STOP_WORDS
                    or gram[-1].lower() in _STOP_WORDS):
                continue
            term = ' '.join(gram)
            if not is_distinctive(term):
                continue
            key = term.lower()
            if key not in seen:
                seen.add(key)
                out.append(term)
    return out


def is_subject_named(question: str, subject_names) -> bool:
    """Rule 2's trigger: the question literally names the scoped source (or
    an alias) as a word — 'acme-core' matches, 'acme-corex' does not."""
    q = question.lower()
    return any(
        re.search(
            r'(?<![a-z0-9])' + re.escape(name.lower()) + r'(?![a-z0-9])', q)
        for name in subject_names if name
    )


def route_question(question: str, *, subject_names, repo_index,
                   spool_index) -> 'RouteResult':
    """Extract → resolve both sides → derive the regime. Indexes are
    duck-typed (``resolve(term) -> hits``); hits deduplicate by
    (term, layer) so overlapping n-grams can't inflate the evidence."""
    named = is_subject_named(question, subject_names)
    repo_hits, spool_hits, seen = [], [], set()
    for term in extract_candidate_terms(question):
        for side, hits in (('repo', repo_index.resolve(term)),
                           ('spool', spool_index.resolve(term))):
            for hit in hits:
                key = (side, hit.term.lower(), hit.layer)
                if key in seen:
                    continue
                seen.add(key)
                (repo_hits if side == 'repo' else spool_hits).append(hit)
    repo_hits = _drop_subsumed(repo_hits, repo_hits + spool_hits)
    spool_hits = _drop_subsumed(spool_hits, repo_hits + spool_hits)
    return derive_regime(
        subject_named=named, repo_hits=repo_hits, spool_hits=spool_hits)


def _drop_subsumed(hits, all_hits):
    """Maximal munch: a term that is a word-bounded sub-span of a LONGER
    resolved term (on either side) is a fragment of that entity, not an
    entity — drop its hits everywhere ('Quantum' inside 'Quantum Mesh')."""
    resolved = {h.term.lower() for h in all_hits}

    def subsumed(hit):
        t = hit.term.lower()
        return any(
            other != t and re.search(
                r'(?<![a-z0-9])' + re.escape(t) + r'(?![a-z0-9])', other)
            for other in resolved
        )

    return [h for h in hits if not subsumed(h)]


def _strong(hits) -> bool:
    return any(h.entity_class in _STRONG_CLASSES for h in hits)


def derive_regime(*, subject_named: bool, repo_hits, spool_hits) -> RouteResult:
    """§3 regime derivation — total over its inputs.

    The repo path always runs (hop one, in the caller's scope); this
    decides the SPOOL's participation and whether the repo TAKE is dropped
    from the spool question's context. Rules, in order: no crisp signal
    anywhere → repo-only with gated fallback when the subject is named,
    honest-gap otherwise; spool silent when it has no crisp hit;
    expert-only only when the subject is NOT named, the spool has a strong
    (api/product) hit, and the repo has none — repo phrase hits don't block
    (a consumer repo genuinely resolving generic property vocabulary must
    not veto the expert). Everything else fuses: ambiguity degrades toward
    fuse-with-attribution, because labeled inclusion is harmless and silent
    dropping is not.
    """
    repo = tuple(repo_hits)
    spool = tuple(spool_hits)
    if not repo and not spool:
        regime = REPO_ONLY if subject_named else HONEST_GAP
        return RouteResult(regime, (), (), subject_named, True)
    if not spool:
        return RouteResult(REPO_ONLY, repo, (), subject_named, False)
    if not subject_named and _strong(spool) and not _strong(repo):
        return RouteResult(EXPERT_ONLY, repo, spool, subject_named, False)
    return RouteResult(FUSE, repo, spool, subject_named, False)
