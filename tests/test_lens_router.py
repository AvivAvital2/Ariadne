"""Slice 1 lens-router core — the three ratified rules + regime derivation.

Evolutionary tests, one per code piece (designs/spool-lens-router.md §3),
encoding the router-signal battery's measured cases as neutral synthetic
fixtures. Distinct from tests/test_spool_router.py (the aisle theme
PREFILTER — which spool could this question touch); this router decides how
the picked spool participates.
"""
from __future__ import annotations

import library.lens_router as lens_router


def _hit(term, layer='title', entity_class='product'):
    return lens_router.EntityHit(
        term=term, layer=layer, entity_class=entity_class)


class TestEntityRules:
    def test_distinctiveness(self):
        # Multi-word phrases and symbol-shaped tokens are distinctive; bare
        # common lowercase words never are (they never resolve or route).
        assert lens_router.is_distinctive('quantum mesh')
        assert lens_router.is_distinctive('FrobnicateSelector')
        assert lens_router.is_distinctive('shared_ledger')
        assert not lens_router.is_distinctive('pipeline')
        assert not lens_router.is_distinctive('memory')

    def test_entity_classes(self):
        # api: resolved via the symbol layer; product: capitalized word(s)
        # via title/heading/alias layers; phrase: other distinctive terms.
        assert lens_router.classify_entity(
            'FrobnicateSelector', layer='symbol') == 'api'
        assert lens_router.classify_entity(
            'Quantum Mesh', layer='title') == 'product'
        assert lens_router.classify_entity(
            'Envcloud', layer='title') == 'product'
        assert lens_router.classify_entity(
            'Quantum Mesh', layer='alias') == 'product'
        assert lens_router.classify_entity(
            'shared ledger', layer='heading') == 'phrase'

    def test_structural_symbol_match_no_typo_tier(self):
        # Rule 1: exact last segment / case-folded / inner segment /
        # word-subset — and NEVER typo-distance (the suggestion matcher's
        # difflib tier mis-fused 4/10 battery questions when misused as a
        # resolver).
        names = ['pkg.mod.FrobnicateSelector', 'pkg.mod.QuantumMesh.helper',
                 'pkg.other.SHARED_LEDGER_LIMIT']
        match = lens_router.match_symbol
        assert match('FrobnicateSelector', names)      # exact last segment
        assert match('frobnicateselector', names)      # case-folded
        assert match('QuantumMesh', names)             # inner segment
        assert match('shared ledger', names)           # word-subset
        assert not match('FrobnicateSelectr', names)   # typo: NOT a route
        assert not match('unrelated thing', names)

    def test_symbol_matches_returns_matching_names_same_tiers(self):
        # Retrieval needs the matched names (to admit their element docs);
        # match_symbol stays the boolean view of the SAME tiers.
        names = ['pkg.mod.FrobnicateSelector', 'pkg.mod.QuantumMesh.helper',
                 'pkg.other.SHARED_LEDGER_LIMIT', 'pkg.mod.score_delta']
        assert lens_router.symbol_matches('shared ledger', names) == [
            'pkg.other.SHARED_LEDGER_LIMIT']
        assert lens_router.symbol_matches('QuantumMesh', names) == [
            'pkg.mod.QuantumMesh.helper']
        assert lens_router.symbol_matches('Delta', names) == []
        many = [f'pkg.m{i}.FrobnicateSelector' for i in range(9)]
        assert len(lens_router.symbol_matches(
            'FrobnicateSelector', many, limit=4)) == 4

    def test_single_token_never_matches_as_word_subset(self):
        # A one-word term must not resolve as a fragment of a longer
        # symbol ({delta} ⊆ {score, delta} is not entity resolution) —
        # live end-to-end regression: bare 'Delta' resolved in the consumer
        # repo and blocked expert-only.
        assert not lens_router.match_symbol('Delta', ['pkg.mod.score_delta'])
        assert lens_router.match_symbol('Delta', ['pkg.mod.Delta'])

    def test_classification_follows_term_shape_through_symbol_layer(self):
        # A lowercase phrase is a phrase no matter which layer resolved it;
        # a capitalized multi-word stays product even via symbols. Only a
        # single symbol-shaped token is 'api'.
        assert lens_router.classify_entity('same table', layer='symbol') == 'phrase'
        assert lens_router.classify_entity('Delta Lake', layer='symbol') == 'product'


class TestRegimeDerivation:
    """The battery's cases, synthetically. subject_named = the question
    literally names the scoped source; the repo path is ALWAYS active — the
    router decides only the spool's participation and mode."""

    def test_pure_code_repo_only(self):
        r = lens_router.derive_regime(
            subject_named=True,
            repo_hits=[_hit('combinatorial pruning', 'title', 'phrase')],
            spool_hits=[],
        )
        assert r.regime == 'repo-only'
        assert not r.fallback_enabled     # crisp repo signal: no probing
        assert r.primary == 'repo'

    def test_seam_named_subject_fuses(self):
        r = lens_router.derive_regime(
            subject_named=True,
            repo_hits=[],
            spool_hits=[_hit('Quantum Mesh', 'title', 'product')],
        )
        assert r.regime == 'fuse'
        # RATIFIED dominance rule (bidirectional lens): the environment owns
        # the question's strong entities and the repo has none — the spool
        # becomes the PRIMARY ranked channel; the named repo rides as the
        # capped, labeled lens.
        assert r.primary == 'spool'

    def test_pure_target_expert_only_over_repo_phrase(self):
        # Entity-class weighting: spool product entity vs repo
        # property-phrase only, subject not named -> drop the repo take.
        r = lens_router.derive_regime(
            subject_named=False,
            repo_hits=[_hit('shared ledger', 'heading', 'phrase')],
            spool_hits=[_hit('Quantum Mesh', 'title', 'product'),
                        _hit('shared ledger', 'heading', 'phrase')],
        )
        assert r.regime == 'expert-only'
        assert r.primary == 'spool'

    def test_repo_api_hit_blocks_expert_only(self):
        # A repo api/product-class hit keeps the repo take (fuse), even
        # unnamed — the safety asymmetry degrades toward inclusion.
        r = lens_router.derive_regime(
            subject_named=False,
            repo_hits=[_hit('FrobnicateSelector', 'symbol', 'api')],
            spool_hits=[_hit('Quantum Mesh', 'title', 'product')],
        )
        assert r.regime == 'fuse'
        assert r.primary == 'repo'        # both strong: the artifact anchors

    def test_named_subject_blocks_expert_only(self):
        r = lens_router.derive_regime(
            subject_named=True,
            repo_hits=[],
            spool_hits=[_hit('Quantum Mesh', 'title', 'product')],
        )
        assert r.regime != 'expert-only'

    def test_no_signal_named_subject_repo_only_with_fallback(self):
        r = lens_router.derive_regime(
            subject_named=True, repo_hits=[], spool_hits=[])
        assert r.regime == 'repo-only'
        assert r.fallback_enabled         # gated semantic probing allowed
        assert r.primary == 'repo'

    def test_no_signal_unnamed_honest_gap(self):
        r = lens_router.derive_regime(
            subject_named=False, repo_hits=[], spool_hits=[])
        assert r.regime == 'honest-gap'
        assert r.fallback_enabled

    def test_spool_phrase_only_never_drops_repo(self):
        # Spool crisp but only phrase-class: fuse, not expert-only.
        r = lens_router.derive_regime(
            subject_named=False,
            repo_hits=[],
            spool_hits=[_hit('shared ledger', 'heading', 'phrase')],
        )
        assert r.regime == 'fuse'
        assert r.primary == 'repo'        # phrase never decides dominance

    def test_result_carries_the_evidence(self):
        repo = [_hit('combinatorial pruning', 'title', 'phrase')]
        spool = [_hit('Quantum Mesh', 'title', 'product')]
        r = lens_router.derive_regime(
            subject_named=True, repo_hits=repo, spool_hits=spool)
        assert r.crisp_repo == tuple(repo)
        assert r.crisp_spool == tuple(spool)
        assert r.subject_named is True


class TestExtraction:
    def test_ngrams_trim_stopwords_and_keep_distinctive(self):
        terms = lens_router.extract_candidate_terms(
            'What should I watch for when shipping data to Quantum Mesh?')
        assert 'Quantum Mesh' in terms
        assert 'shipping data' in terms
        assert 'What should' not in terms          # stop-word boundary
        assert 'for' not in terms                  # stop word
        assert 'data' not in terms                 # bare common word

    def test_imperative_request_verbs_are_stopwords(self):
        # LOOKUP-probe wart (live): sentence-initial 'Show' resolved as a
        # crisp entity via Dataset.show / 'SHOW COLUMNS' docs and dragged
        # junk into a name lookup. Request verbs carry no entity meaning in
        # questions — they are stop words, as boundaries AND as bare terms.
        terms = lens_router.extract_candidate_terms(
            'Show me the Quantum Mesh overview')
        assert 'Show' not in terms
        assert 'Quantum Mesh' in terms             # the real entity survives
        for verb in ('List', 'Explain', 'Describe', 'Display'):
            assert verb not in lens_router.extract_candidate_terms(
                f'{verb} the shared ledger limits')

    def test_symbol_shaped_single_tokens_survive(self):
        terms = lens_router.extract_candidate_terms(
            'Is FrobnicateSelector still available?')
        assert 'FrobnicateSelector' in terms

    def test_subject_detection_word_bounded(self):
        assert lens_router.is_subject_named(
            'how does acme-core keep features?', ['acme-core'])
        assert not lens_router.is_subject_named(
            'how does acme-corex keep features?', ['acme-core'])
        assert not lens_router.is_subject_named(
            'how does the mesh work?', ['acme-core'])


class TestRouteQuestion:
    """End-to-end orchestration over duck-typed entity indexes."""

    class _Index:
        def __init__(self, hits_by_term):
            self._hits = {k.lower(): tuple(v) for k, v in hits_by_term.items()}

        def resolve(self, term):
            return self._hits.get(term.lower(), ())

    def test_seam_question_fuses(self):
        repo = self._Index({})
        spool = self._Index({
            'Quantum Mesh': [_hit('Quantum Mesh', 'title', 'product')]})
        r = lens_router.route_question(
            'What should I watch for when shipping acme-core data to Quantum Mesh?',
            subject_names=['acme-core'], repo_index=repo, spool_index=spool)
        assert r.regime == 'fuse'
        assert r.subject_named

    def test_pure_target_question_expert_only(self):
        repo = self._Index({
            'shared ledger': [_hit('shared ledger', 'heading', 'phrase')]})
        spool = self._Index({
            'Quantum Mesh': [_hit('Quantum Mesh', 'title', 'product')],
            'shared ledger': [_hit('shared ledger', 'heading', 'phrase')]})
        r = lens_router.route_question(
            'How does Quantum Mesh handle two jobs writing to the shared ledger?',
            subject_names=['acme-core'], repo_index=repo, spool_index=spool)
        assert r.regime == 'expert-only'

    def test_no_signal_scoped_question_repo_only_with_fallback(self):
        r = lens_router.route_question(
            'How does acme-core handle failures?',
            subject_names=['acme-core'],
            repo_index=self._Index({}), spool_index=self._Index({}))
        assert r.regime == 'repo-only'
        assert r.fallback_enabled

    def test_fragments_of_resolved_entities_are_subsumed(self):
        # Maximal munch: 'Quantum' resolving alone (either side) is a
        # fragment of the resolved entity 'Quantum Mesh' and must be
        # dropped everywhere — else a repo-side fragment blocks
        # expert-only (live end-to-end regression).
        repo = self._Index({
            'Quantum': [_hit('Quantum', 'symbol', 'api')]})
        spool = self._Index({
            'Quantum Mesh': [_hit('Quantum Mesh', 'title', 'product')],
            'Quantum': [_hit('Quantum', 'title', 'product')]})
        r = lens_router.route_question(
            'How does Quantum Mesh handle concurrent writes?',
            subject_names=['acme-core'], repo_index=repo, spool_index=spool)
        assert r.regime == 'expert-only'


class TestEnvironmentNameTerms:
    def test_route_dont_admit_helpers(self):
        # A spool's OWN names (environment id, component keys, corpus keys,
        # recipe aliases) are excellent ROUTING signals but degenerate doc
        # SELECTORS inside their own corpus (they match READMEs, code-of-
        # conducts, namespace markers by construction). The name set feeds
        # admission filtering only — derive_regime never sees it.
        from types import SimpleNamespace

        resolution = SimpleNamespace(registered={
            'fakebricks': SimpleNamespace(manifest=SimpleNamespace(
                environment='fakebricks',
                runtime_components={'quantumcore': '2.0'},
                corpus_shas={'quantumcore': 'abc', 'mesh-sdk': 'def'},
                name_aliases=['quantum mesh'],
            )),
        })
        names = lens_router.environment_name_terms(resolution)
        assert names == frozenset(
            {'fakebricks', 'quantumcore', 'mesh-sdk', 'quantum mesh'})

        hits = [
            _hit('Quantum Mesh', 'title', 'product'),      # alias -> dropped
            _hit('fakebricks', 'heading', 'product'),      # env id -> dropped
            _hit('ConflictChecker', 'symbol', 'api'),      # real entity kept
            _hit('same table', 'symbol', 'phrase'),        # phrase kept
        ]
        kept = lens_router.admissible_hits(hits, names)
        assert [h.term for h in kept] == ['ConflictChecker', 'same table']

    def test_tolerant_resolution_shapes(self):
        from types import SimpleNamespace

        # dict-shaped manifest + missing fields never crash.
        resolution = SimpleNamespace(registered={
            'envx': {'environment': 'envx'},
        })
        assert lens_router.environment_name_terms(resolution) == frozenset(
            {'envx'})
        assert lens_router.environment_name_terms(
            SimpleNamespace()) == frozenset()
