"""Search and ranking logic for AriadneService."""
from __future__ import annotations

import hashlib
import json
import logging

from ariadne_mcp.models import DocumentResult, SearchResponse, SectionResult

_logger = logging.getLogger(__name__)


def _is_catalog_bloat(title: str) -> bool:
    """Low-value decompiled/boilerplate catalog docs (local-variable accessor
    stubs like ``local 169``, ``@SerialVersionUID`` lines, anonymous-class
    placeholders) that dilute a spool's signal. Demoted in ground ranking; the
    deeper fix is excluding them at build-time catalog extraction."""
    t = (title or '').strip()
    if 'SerialVersionUID' in t or t.lower().startswith('anonymous class'):
        return True
    return t.startswith('local ') and t[len('local '):].strip().isdigit()


def _trim_related_documents(content: str, max_links: int = 5) -> str:
    """Trim the '## Related Documents' section to reduce response bloat.

    Auto-generated cross-references can account for 30-60% of content size.
    Prioritize import-based links ("References X") over title-mention links
    ("Mentions 'X'"), since imports indicate real code dependencies while
    mentions are often noisy string matches on short titles.
    """
    marker = '\n## Related Documents'
    idx = content.find(marker)
    if idx == -1:
        return content
    before = content[:idx]
    related = content[idx:]
    lines = related.split('\n')

    # Separate import-based links from mention-based ones
    import_links = []
    mention_links = []
    for line in lines[1:]:
        if not line.startswith('- '):
            continue
        if 'References ' in line:
            import_links.append(line)
        elif 'Mentions ' in line:
            mention_links.append(line)
        else:
            import_links.append(line)  # unknown type → keep

    # Prioritize import links, fill remainder with mentions
    kept = import_links[:max_links]
    remaining = max_links - len(kept)
    if remaining > 0:
        kept.extend(mention_links[:remaining])

    total = len(import_links) + len(mention_links)
    # lines[0] is empty (from leading \n in marker), lines[1] is the header
    result = ['', lines[1] if len(lines) > 1 else '## Related Documents', '']
    result.extend(kept)
    if total > max_links:
        result.append(f'- ... and {total - len(kept)} more')
    return before + '\n'.join(result)


# Version of the retrieval semantics baked into every persistent cache key.
# BUMP THIS whenever a fix changes what the CORRECT result set is for an
# identical query (scope gates, ranking weights, admission rules): the spool
# fingerprint only tracks the enabled-spool SET, so without a version
# component a fixed leak keeps replaying from query_cache for its 30-day TTL.
# v2 = 2026-07-26 (member-grounded theme gate, spool provenance rank,
# per-source suggestion pool). v3 = lens routing wired (regimes change every
# routed result set). v4 = entity-cap + reserved semantic fill in
# select_spool_docs (fuse/expert-only result sets change). v5 =
# surface-scoped fallback (no-crisp result sets change where tags exist).
_RETRIEVAL_CACHE_VERSION = 5

class SearchMixin:
    """Search implementation with multi-phase ranking.

    Expects the composed class to provide:
    - self.library: Library
    - self.embedding_service: EmbeddingService
    - self.config: Config
    - self._cache_key(), self._query_cache: dict
    - self.get_branch()
    """

    @staticmethod
    def _persistent_cache_key(
        func_name: str, branch: str | None, *args: object, **kwargs: object,
    ) -> str:
        """Deterministic SHA256 cache key: retrieval version + branch + args."""
        raw = json.dumps(
            [_RETRIEVAL_CACHE_VERSION, func_name, branch or '', list(args),
             sorted((kwargs or {}).items())],
            sort_keys=True, default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

    def expand(self, event_id: int) -> dict:
        """Return full content for the docs of a prior search event.

        CRIT-12: re-fetching by id must honor the CURRENT spool scope. A
        doc whose source is a spool that is no longer registered (disabled
        or updated away) is dropped to ``missing_document_ids`` — otherwise
        a stale ``event_id`` would leak a disabled/malicious spool's full
        content, bypassing the disable-remediation. Non-spool docs are
        unaffected.
        """
        from spools import active_spool_sources, is_spool_source

        event = self.library.get_usage_event(event_id)
        if event is None:
            return {'error': f'event_id {event_id} not found', 'event_id': event_id}
        doc_ids = event.get('returned_document_ids') or []
        if not doc_ids:
            return {
                'event_id': event_id,
                'original_query': event.get('query'),
                'documents': [],
                'note': 'original search returned no documents',
            }

        active_spools = active_spool_sources(self.config)
        found = {d.id: d for d in self.library.get_documents_batch(doc_ids)}

        documents: list[dict] = []
        missing: list[str] = []
        for doc_id in doc_ids:
            doc = found.get(doc_id)
            src = getattr(doc, 'source_name', None) if doc else None
            # Missing, or a spool no longer in scope (disabled/updated) — a
            # stale event_id must not resurface a disabled spool's content.
            if doc is None or (is_spool_source(src) and src not in active_spools):
                missing.append(doc_id)
                continue
            documents.append({
                'id': getattr(doc, 'id', doc_id),
                'title': getattr(doc, 'title', None),
                'content': getattr(doc, 'content', None),
                'content_type': getattr(doc, 'content_type', None),
                'source_files': getattr(doc, 'source_files', []),
            })

        result: dict = {
            'event_id': event_id,
            'original_query': event.get('query'),
            'documents': documents,
        }
        if missing:
            result['missing_document_ids'] = missing
        return result

    async def search(
        self,
        query: str | None = None,
        feature: str | None = None,
        branch: str | None = None,
        status: str | None = None,
        limit: int = 10,
        context_file: str | None = None,
        sections_only: bool = False,
        role: str = 'developer',
        source: str | None = None,
    ) -> SearchResponse:
        """Search documents within the resolved source's closure.

        ``source`` selects which source's closure scopes the search;
        if omitted it resolves from cwd then ``config.default_source``.
        See ``designs/directional-closure-scoping.md`` for the closure
        rule.

        ``role`` (default ``'developer'``) gates whether
        ``content_type='audience_response'`` rows participate. When
        ``role='developer'``, those rows are excluded — developers
        always want the dev baseline. When ``role`` is non-default
        (currently only ``'product_manager'`` supported), only
        audience_response rows whose ``metadata.audience`` matches
        the requested role participate.
        """
        # ``source`` is part of the cache keys so a query asked from
        # product's scope and the same query from extension's scope
        # don't collide (their closures differ, so their result sets
        # legitimately differ). CRIT-11: the enabled-spool fingerprint is
        # in the key too, so enabling/disabling/updating a spool shifts the
        # key → a stale cached result can't outlive the spool that shaped
        # it (in particular, disabling a bad spool evicts its content).
        from spools import resolve_spools
        spool_resolution = resolve_spools(self.config)
        spool_fp = spool_resolution.fingerprint()
        key = self._cache_key('search', query, feature, branch, status, limit, context_file, sections_only, role, source, spool_fp)
        if key in self._query_cache:
            return self._query_cache[key]

        # Persistent cache (cross-session, branch-aware, 30-day TTL)
        p_key = self._persistent_cache_key('search', branch, query, feature, status, limit, context_file, sections_only, role, source, spool_fp)
        cached_json = self.library.cache_get(p_key)
        if cached_json is not None:
            try:
                result = SearchResponse.model_validate_json(cached_json)
                self._query_cache[key] = result  # Populate in-memory too
                return result
            except Exception:
                pass  # Corrupt cache entry — recompute

        result = await self._search_uncached(query, feature, branch, status, limit, context_file, sections_only, role=role, source=source, spool_resolution=spool_resolution)
        self._query_cache[key] = result

        # Store in persistent cache
        try:
            self.library.cache_put(p_key, branch or '', query or '', result.model_dump_json())
        except Exception:
            _logger.debug('Failed to persist cache entry', exc_info=True)

        return result

    def _lens_route(self, query, lite_docs, spool_sources, spool_fp):
        """Deterministic lens routing (designs/spool-lens-router.md §3-§4):
        build/cache the two entity indexes for this (fingerprint, sources)
        pair and derive the regime. No LLM, no embeddings — routing must
        never depend on a degraded environment."""
        from spools import is_spool_source

        from library.lens_router import route_question
        from library.spool_entity_index import build_entity_index

        repo_sources = sorted({
            d.source_name for d in lite_docs
            if d.source_name and not is_spool_source(d.source_name)
        })
        corpus_names = {s.split(':', 1)[1] for s in spool_sources if ':' in s}
        spool_index_sources = sorted(set(spool_sources) | corpus_names)
        cache = getattr(self, '_lens_index_cache', None)
        if cache is None:
            cache = self._lens_index_cache = {}
        key = (spool_fp, tuple(repo_sources), tuple(spool_index_sources))
        pair = cache.get(key)
        if pair is None:
            pair = (
                build_entity_index(self.library, repo_sources or ['']),
                build_entity_index(self.library, spool_index_sources),
            )
            cache[key] = pair
        repo_index, spool_index = pair
        return route_question(
            query, subject_names=repo_sources,
            repo_index=repo_index, spool_index=spool_index,
        )

    async def _lens_ranked_ids(self, route, repo_ids, retrieval_sources,
                               query, limit, weights,
                               surface_restrict=None):
        """Regime → (ranked ids, connection labels) (design §5-§6).
        repo-only: spool silent;
        fuse: repo ranking + categorical spool contributions; expert-only:
        the spool contributions ARE the results (repo take dropped; degrades
        to repo ranking only if the environment can resolve no entity docs —
        never empty for structural reasons); no-crisp regimes: repo ranking
        + the gated semantic fallback when enabled."""
        from library.lens_retrieval import (
            fallback_spool_docs,
            select_spool_docs,
        )

        matrix = None
        try:
            matrix = self._get_embedding_matrix()
        except Exception:
            _logger.debug('lens: embedding matrix unavailable', exc_info=True)

        async def _repo_ranked():
            if not repo_ids:
                return []
            return await self._rank_ids_by_embedding(
                repo_ids, query, limit, weights)

        contributions = []
        if route.regime in ('fuse', 'expert-only'):
            query_embedding = None
            try:
                query_embedding = await self.embedding_service.embed(query)
            except Exception:
                _logger.debug(
                    'lens: query embed failed — entity admissions only',
                    exc_info=True)
            spool_limit = (
                limit if route.regime == 'expert-only' else max(2, limit // 2)
            )
            contributions = select_spool_docs(
                self.library, matrix, retrieval_sources, route.crisp_spool,
                query_embedding=query_embedding, limit=spool_limit,
            )

        if route.regime == 'expert-only':
            if contributions:
                ordered = [c.doc_id for c in contributions][:limit]
            else:
                ordered = [d for d, _ in await _repo_ranked()][:limit]
        else:
            repo_ranked = [d for d, _ in await _repo_ranked()]
            if route.regime == 'fuse':
                pass                          # contributions already selected
            elif route.fallback_enabled:
                contributions = fallback_spool_docs(
                    self.library, matrix, retrieval_sources,
                    repo_ranked[:3], limit=2,
                    restrict_to=surface_restrict)
            else:
                contributions = []
            extra = [c.doc_id for c in contributions]
            keep = max(1, limit - len(extra)) if extra else limit
            seen = set(repo_ranked[:keep])
            ordered = (
                repo_ranked[:keep]
                + [e for e in extra if e not in seen]
            )[:limit]
        kept = set(ordered)
        connections = {
            c.doc_id: f'{c.connection}({c.detail})'
            for c in contributions if c.doc_id in kept
        }
        n = len(ordered)
        scored = [(did, (n - i) / n if n else 0.0)
                  for i, did in enumerate(ordered)]
        return scored, connections

    async def _search_uncached(
        self,
        query: str | None = None,
        feature: str | None = None,
        branch: str | None = None,
        status: str | None = None,
        limit: int = 10,
        context_file: str | None = None,
        sections_only: bool = False,
        role: str = 'developer',
        source: str | None = None,
        spool_resolution=None,
    ) -> SearchResponse:
        """Uncached search implementation."""
        # Build the closure-scoped library view first so every read path
        # below cannot leak rows from outside the closure.
        scoped = self._resolve_scope(source)

        # Resolve the enabled-spool set ONCE per request: search() passes down
        # the resolution it already built for the cache key, so the fingerprint
        # (PM audience gate) and the source ids (tier-2 partition) come off one
        # resolution instead of re-reading manifests. A direct caller (or test)
        # that omits it falls back to resolving here.
        if spool_resolution is None:
            from spools import resolve_spools
            spool_resolution = resolve_spools(self.config)
        spool_sources = spool_resolution.scope_sources()
        spool_fp = spool_resolution.fingerprint()

        # Phase 1: Filter using lightweight metadata (no content/embeddings loaded)
        lite_docs = scoped.list_documents_lite()

        # Role filter — applied BEFORE other filters so the candidate
        # set is correct for ranking. ``content_type='audience_response''
        # rows are excluded for the default developer role (those rows
        # are the shallower audience-adapted siblings of the dev baseline;
        # developers always want the dev content). For non-default roles,
        # include audience_response rows for the matching audience only;
        # dev baseline rows still participate because the adapter consumes
        # them as context on cache miss.
        if role == 'developer':
            lite_docs = [
                d for d in lite_docs
                if d.content_type != 'audience_response'
            ]
        else:
            # CRIT-11 (PM-search gap): an audience_response row also carries
            # the spool fingerprint it was synthesized under. Include it only
            # when that fingerprint matches the CURRENT enabled-spool set —
            # otherwise a PM answer shaped by a now-disabled spool would leak
            # through search (the disable-remediation must hold here too, not
            # just in the ask cache lookup). Legacy/no-spool rows carry '' and
            # match when no spool is active.
            from ariadne_mcp.service_analysis import audience_row_matches
            lite_docs = [
                d for d in lite_docs
                if (
                    d.content_type != 'audience_response'
                    or audience_row_matches(d.metadata, role, spool_fp)
                )
            ]

        if status:
            lite_docs = [d for d in lite_docs if d.metadata.get('status', 'stable') == status]

        if branch:
            lite_docs = self._filter_by_branch(lite_docs, branch)

        if feature:
            lite_docs = self._filter_by_feature(lite_docs, feature)

        # Phase 2: For embedding ranking, load just the IDs that pass filters
        # Full document content is loaded only for the final top-k results
        from spools import partition_tier2
        # HIGH-1: gate on BOTH axes — only spool-origin official docs are
        # held back as tier-2; a user's own 'official' doc stays tier-1.
        tier1_docs, tier2_docs = partition_tier2(lite_docs, spool_sources)
        candidate_ids = [d.id for d in tier1_docs]
        tier2_ids = [d.id for d in tier2_docs]
        from library.search import provenance_weight
        weights = {d.id: provenance_weight(d.metadata, d.source_name) for d in lite_docs}

        # Context-aware boosting: if working on a specific file, boost
        # docs for that file and its graph neighbors
        context_boost_ids: set[str] = set()
        if context_file:
            context_docs = scoped.find_documents_by_source_files([context_file])
            context_boost_ids = {d.id for d in context_docs}
            # Also boost graph neighbors' docs
            try:
                related = scoped.get_related(context_file, max_hops=1, limit=20)
                context_boost_ids.update(r['id'] for r in related)
            except Exception:
                pass

        # Score and rank by query (treat empty/whitespace as no query)
        effective_query = query.strip() if query else ''
        spool_gap_hint_text = None
        lens_connections: dict | None = None
        if effective_query:
            if spool_sources:
                # Lens routing (designs/spool-lens-router.md §6). On routed
                # questions the REGIME replaces the anchored premise ("the
                # spool never replaces the repo"), the scarcity gate ("spool
                # only when the repo runs thin"), and the tier-2 holdback —
                # those philosophies contradict expert-only and the
                # corrector (a confidently-wrong repo layer kept the gate
                # closed). Routing is deterministic: entity resolution over
                # both catalogs + subject anchor + entity classes; ranked
                # order is encoded as descending scores exactly like the
                # branch it replaces.
                from spools import is_spool_source, spool_gap_hint
                route = self._lens_route(
                    effective_query, lite_docs, spool_sources, spool_fp)
                repo_ids = [d.id for d in lite_docs
                            if not is_spool_source(d.source_name)]
                corpus_names = {
                    s.split(':', 1)[1] for s in spool_sources if ':' in s}
                retrieval_sources = sorted(set(spool_sources) | corpus_names)
                surface_restrict = None
                try:
                    from library.surface_tags import (
                        docs_for_surfaces,
                        surfaces_for_question,
                        surfaces_from_resolution,
                    )
                    _surfaces = surfaces_from_resolution(spool_resolution)
                    _question_surfaces = (
                        surfaces_for_question(effective_query, _surfaces)
                        if _surfaces else []
                    )
                    if _question_surfaces:
                        with self.library._conn_provider.acquire() as _conn:
                            _tagged = docs_for_surfaces(
                                _conn, sorted(corpus_names),
                                _question_surfaces)
                        surface_restrict = _tagged or None
                except Exception:
                    _logger.debug(
                        'lens: surface restriction unavailable',
                        exc_info=True)
                ranked_ids, lens_connections = await self._lens_ranked_ids(
                    route, repo_ids, retrieval_sources, effective_query,
                    limit, weights, surface_restrict=surface_restrict)
                if route.regime == 'honest-gap':
                    spool_doc_ids = {d.id for d in lite_docs
                                     if is_spool_source(d.source_name)}
                    if not (spool_doc_ids
                            & {did for did, _ in ranked_ids}):
                        spool_gap_hint_text = spool_gap_hint(
                            gate_opened=True, tier2_present=False,
                            spools_registered=True)
            else:
                # No spool registered — unchanged scarcity-gate path
                # (byte-identical to pre-anchored behavior for every non-spool
                # project).
                from spools import _GATE_MIN_STRONG_HITS, rank_with_scarcity_gate

                async def _rank(ids):
                    # HIGH-3: the scarcity check needs at least
                    # _GATE_MIN_STRONG_HITS candidates to judge "is code
                    # scarce"; floor the window so a small caller limit (e.g. 1)
                    # can't shrink it below that and force the gate open.
                    return await self._rank_ids_by_embedding(ids, effective_query, max(limit * 2, _GATE_MIN_STRONG_HITS), weights)

                gated = await rank_with_scarcity_gate(_rank, candidate_ids, tier2_ids)
                ranked_ids = gated.ranked
                from spools import spool_gap_hint
                spool_gap_hint_text = spool_gap_hint(
                    gate_opened=gated.gate_opened,
                    tier2_present=bool(tier2_ids),
                    spools_registered=bool(spool_sources),
                )

            # Apply context boost (shared by both ranking paths)
            if context_boost_ids:
                boosted = [(did, (score or 0) + (0.15 if did in context_boost_ids else 0.0))
                           for did, score in ranked_ids]
                boosted.sort(key=lambda x: x[1], reverse=True)
                top_ids_with_scores = boosted[:limit]
            else:
                top_ids_with_scores = ranked_ids[:limit]

            # Phase 4: Load full content only for the final top-k
            top_ids = [did for did, _ in top_ids_with_scores]
            score_map = dict(top_ids_with_scores)
            top_docs = scoped.get_documents_batch(top_ids)
            scored_docs = [(d, score_map.get(d.id)) for d in top_docs]
        else:
            # No query — return first N docs (load full content for those only)
            top_ids = candidate_ids[:limit]
            top_docs = scoped.get_documents_batch(top_ids) if top_ids else []
            scored_docs = [(d, None) for d in top_docs]

        doc_ids = [d.id for d, _ in scored_docs]
        event_id = self.library.log_usage(
            'ariadne_search', query or feature or '', len(scored_docs),
            document_ids=doc_ids,
        )

        # Budget check: estimate token cost of full-content output.
        # If it exceeds response_token_budget, downgrade to sections and mark                                                                                           
        # truncated=True so the caller can invoke ariadne_expand(event_id).
        truncated = False                                                                                                                                               
        if not sections_only and effective_query:
            try:                                                                                                                                                        
                from config import get_config
                _budget = int(get_config().response_token_budget or 0)
            except Exception:                                                                                                                                           
                _budget = 0
            if _budget > 0:                                                                                                                                             
                _estimate = sum(
                    (len(d.content) // 4) for d in top_docs if d.content
                )                                                                                                                                                       
                if _estimate > _budget:
                    sections_only = True                                                                                                                                
                    truncated = True

        # Build section-filtered results when requested
        section_data: dict[str, list[SectionResult]] | None = None
        if sections_only and effective_query:
            section_data = await self._select_relevant_sections(doc_ids, effective_query)

        documents: list[DocumentResult] = []
        for d, score in scored_docs:
            if section_data is not None and d.id in section_data and section_data[d.id]:
                # Return only relevant sections instead of full content
                secs = section_data[d.id]
                kept_headings = {s.heading for s in secs}
                all_sections = self.library.get_sections(d.id)
                omitted = [s for s in all_sections if s.heading not in kept_headings]
                assembled = '\n\n'.join(s.content for s in secs)
                if omitted:
                    assembled += f'\n\n[... {len(omitted)} sections omitted ...]'
                documents.append(DocumentResult(
                    id=d.id, title=d.title, content_type=d.content_type,
                    content=assembled, source_files=d.source_files,
                    metadata=d.metadata, score=score, sections=secs,
                    source_name=d.source_name,
                ))
            else:
                documents.append(DocumentResult(
                    id=d.id, title=d.title, content_type=d.content_type,
                    content=_trim_related_documents(d.content),
                    source_files=d.source_files, metadata=d.metadata, score=score,
                    source_name=d.source_name,
                ))

        return SearchResponse(
            documents=documents,
            event_id=event_id,
            suggested_queries=self._suggest_queries(effective_query, scored_docs) if effective_query else None,
            improvement_hint=spool_gap_hint_text or self._improvement_hint(scored_docs, effective_query),
            truncated=truncated,
            spool_connections=lens_connections or None,                                                                                                                                        
        )

    @staticmethod
    def _suggest_queries(query: str, results: list[tuple]) -> list[str] | None:
        """Suggest alternative queries when results are few or low-scoring.

        Returns None if results are good enough. Returns up to 3 suggestions otherwise.
        """
        # Only suggest if few results or all scores are low
        if len(results) >= 3:
            scores = [s for _, s in results if s is not None]
            if scores and max(scores) > 0.5:
                return None  # Good results, no suggestions needed

        # Generate suggestions from query words
        words = query.lower().split()
        if len(words) < 2:
            return None

        suggestions = []
        # Try individual key words (skip common words)
        skip = {'the', 'a', 'an', 'in', 'of', 'for', 'to', 'how', 'does', 'what', 'is', 'are', 'with'}
        key_words = [w for w in words if w not in skip and len(w) > 2]
        for w in key_words[:3]:
            suggestions.append(w)

        # Try pairs of key words
        if len(key_words) >= 2:
            suggestions.append(f'{key_words[0]} {key_words[1]}')

        return suggestions[:3] if suggestions else None

    @staticmethod
    def _improvement_hint(results: list[tuple], query: str) -> str | None:
        """Generate a hint when the library could be improved.

        Returns a hint string when results are poor, suggesting the caller
        use ariadne_contribute to save useful findings back to the library.
        """
        if not query:
            return None

        # No results at all
        if not results:
            return (
                f'No docs found for "{query}". If you find the answer through code exploration, '
                'consider saving it with ariadne_contribute so future searches succeed.'
            )

        # All results have low scores
        scores = [s for _, s in results if s is not None]
        if scores and max(scores) < 0.35:
            return (
                f'Results for "{query}" have low relevance scores (max {max(scores):.2f}). '
                'If you find better information, contribute it back with ariadne_contribute.'
            )

        return None

    async def _rank_ids_by_embedding(self, doc_ids: list[str], query: str, limit: int, weights=None) -> list[tuple[str, float]]:
        """Rank document IDs by embedding similarity, via the strategy chosen for the candidate count."""
        try:
            from library.embedding_ranking import select_ranker

            query_embedding = await self.embedding_service.embed(query)
            ranker = select_ranker(len(doc_ids), self._get_embedding_matrix, self.library)
            ranked = ranker.rank(query_embedding, doc_ids, limit, weights=weights)
            if ranked:
                return ranked
            return [(did, 0.0) for did in self._rank_by_query_ids(doc_ids, query, limit)]
        except Exception:
            _logger.debug('Embedding ranking failed, falling back to text matching', exc_info=True)
            return [(did, 0.0) for did in self._rank_by_query_ids(doc_ids, query, limit)]

    async def _anchored_rank_ids(self, anchor_ids, ground_ids, query, limit, weights=None):
        """Anchor-then-ground ranking: the user repo (``anchor_ids``) is the
        protected subject; the spool (``ground_ids``) is subordinate context,
        admitted by both query and anchor similarity, diversified and gated.
        See ``designs/spool-anchored-retrieval.md``.

        Falls back to flat embedding ranking over anchor+ground when the
        embedding matrix is unavailable, so a missing matrix never regresses to
        empty results.
        """
        import numpy as np

        from library.anchored_retrieval import anchored_rank

        query_embedding = await self.embedding_service.embed(query)
        floor = max(1, limit // 2)          # the repo keeps at least half the slots
        window_a = max(limit * 2, 20)       # per-side pre-rank windows keep the
        window_g = max(limit * 5, 50)       # anchored combine O(window), not O(N)

        def _norm(vec):
            arr = np.asarray(vec, dtype=np.float32)
            mag = float(np.linalg.norm(arr))
            return arr / mag if mag else arr

        def _pairs(ids, emb_fn):
            out = []
            for did in ids:
                vec = emb_fn(did)
                if vec is not None:
                    out.append((did, _norm(vec)))
            return out

        matrix = None
        try:
            matrix = self._get_embedding_matrix()
        except Exception:
            _logger.debug('anchored rank: matrix load failed', exc_info=True)

        if matrix is not None:
            # Pre-rank each side by query similarity via the fast batch matrix,
            # keeping only a small window per side. The anchor-similarity combine
            # then runs over ~window docs, not the whole (~200k-doc) corpus.
            top_anchor = matrix.rank(query_embedding, list(anchor_ids), window_a)
            top_ground = matrix.rank(query_embedding, list(ground_ids), window_g)

            def _emb(did):
                row = matrix.id_to_row.get(did)
                return matrix.M[row] if row is not None else None

            anchor_pairs = _pairs([did for did, _ in top_anchor], _emb)
            ground_pairs = _pairs([did for did, _ in top_ground], _emb)
        elif len(anchor_ids) + len(ground_ids) > 2000:
            # No matrix over a large corpus — can't anchor cheaply; degrade to a
            # flat embedding ranking rather than hang. (The matrix is present in
            # normal operation; this guards the degraded path.)
            ranked = await self._rank_ids_by_embedding(
                list(anchor_ids) + list(ground_ids), query, limit, weights)
            return [did for did, _ in ranked]
        else:
            # Small candidate sets (tests / tiny stores) — embeddings from the DB.
            emb_map = self.library.get_embeddings_for_ids(
                list(anchor_ids) + list(ground_ids))

            def _emb(did):
                return emb_map.get(did)

            anchor_pairs = _pairs(anchor_ids, _emb)
            ground_pairs = _pairs(ground_ids, _emb)

        if not anchor_pairs and not ground_pairs:
            return (list(anchor_ids)[:floor] + list(ground_ids)
                    + list(anchor_ids)[floor:])[:limit]
        return anchored_rank(
            _norm(query_embedding), anchor_pairs, ground_pairs,
            limit=limit, anchor_floor=floor, weights=weights or {},
        )

    def _rank_by_query_ids(self, doc_ids: list[str], query: str, limit: int) -> list[str]:
        """Fallback text ranking that returns doc IDs."""
        # Load lite metadata for scoring
        query_lower = query.lower()
        query_words = query_lower.split()
        lite_docs = self.library.list_documents_lite()
        id_set = set(doc_ids)
        scored = []
        for doc in lite_docs:
            if doc.id not in id_set:
                continue
            score = 0.0
            title_lower = doc.title.lower()
            if title_lower == query_lower:
                score += 10
            elif query_lower in title_lower:
                score += 5
            for word in query_words:
                if word in title_lower:
                    score += 2
            if score > 0:
                scored.append((doc.id, score))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [did for did, _ in scored[:limit]]

    @staticmethod
    def _filter_by_branch(docs: list, branch: str) -> list:
        from library import filter_by_branch
        return filter_by_branch(docs, branch)

    @staticmethod
    def _filter_by_feature(docs: list, feature: str) -> list:
        feature_lower = feature.lower()
        filtered = []
        for doc in docs:
            doc_feature = doc.metadata.get('feature', '')
            if doc_feature and feature_lower in doc_feature.lower():
                filtered.append(doc)
                continue

            doc_aliases = doc.metadata.get('aliases', [])
            if isinstance(doc_aliases, list):
                for alias in doc_aliases:
                    if feature_lower in alias.lower():
                        filtered.append(doc)
                        break
        return filtered

    _SECTION_SIMILARITY_THRESHOLD = 0.3

    async def _select_relevant_sections(
        self,
        doc_ids: list[str],
        query: str,
    ) -> dict[str, list[SectionResult]]:
        """For each document, rank sections by embedding similarity and return relevant ones."""
        import numpy as np

        from search import batch_dot_similarity

        query_embedding = await self.embedding_service.embed(query)
        result: dict[str, list[SectionResult]] = {}

        for doc_id in doc_ids:
            section_embs = self.library.get_section_embeddings_for_doc(doc_id)
            if not section_embs:
                # No sections stored — fall back to full content
                result[doc_id] = []
                continue

            indices, embeddings = zip(*section_embs)
            emb_matrix = np.stack(embeddings)
            similarities = batch_dot_similarity(query_embedding, emb_matrix)

            # Get all sections for this doc to look up content
            sections = self.library.get_sections(doc_id)
            section_by_idx = {s.index: s for s in sections}

            relevant: list[SectionResult] = []
            for i, (sec_idx, sim) in enumerate(zip(indices, similarities)):
                if float(sim) >= self._SECTION_SIMILARITY_THRESHOLD:
                    sec = section_by_idx.get(sec_idx)
                    if sec:
                        relevant.append(SectionResult(
                            heading=sec.heading,
                            description=sec.description,
                            content=sec.content,
                            score=float(sim),
                        ))

            # Always include at least the top section (overview) if we have sections
            if not relevant and sections:
                sec = sections[0]
                relevant.append(SectionResult(
                    heading=sec.heading, description=sec.description,
                    content=sec.content, score=0.0,
                ))

            # Sort by original document order for readability
            sec_order = {s.heading: s.index for s in sections}
            relevant.sort(key=lambda s: sec_order.get(s.heading, 999))
            result[doc_id] = relevant

        return result
    
    def _get_embedding_matrix(self):
        """Load + freshness-check the shared embedding matrix; None falls back to SQLite."""
        from pathlib import Path

        from library.embedding_matrix import EmbeddingMatrix

        matrix_dir = Path(self.library._conn_provider.path).parent / '.ariadne'
        if not hasattr(self, '_embedding_matrix_cache'):
            self._embedding_matrix_cache = EmbeddingMatrix.load(matrix_dir)
        matrix = self._embedding_matrix_cache
        with self.library._conn_provider.acquire() as conn:
            if matrix is not None and matrix.is_fresh(conn):
                return matrix
            # Stale (or absent) in-process copy: the store changed after we
            # loaded — e.g. a `spools install`/`theme` run rebuilt the matrix
            # on disk. Re-load ONCE and swap it in, so semantic ranking
            # recovers without a server restart; still-stale -> None (SQLite
            # fallback), never a wrong-dimension rank.
            reloaded = EmbeddingMatrix.load(matrix_dir)
            if reloaded is not None and reloaded.is_fresh(conn):
                self._embedding_matrix_cache = reloaded
                return reloaded
        return None

    async def environment_considerations(
        self, anchor_doc_ids, *, limit=3, gate=0.55, diversity=0.5,
    ):
        """Spool docs most relevant to the anchor docs (a target file/symbol's
        OWN documentation) — the "environment considerations" a mechanistic
        tool (impact_radius / trace_flow) surfaces beside its result.

        Anchor-only ranking (relevance to the anchor, no natural-language
        query), relevance-gated so an unrelated target admits none. Returns
        ``[]`` when no spool is enabled or nothing clears the gate, so the
        tool's output is then exactly as before (no-harm).
        """
        import numpy as np

        from library.anchored_retrieval import select_ground
        from spools import is_spool_source, resolve_spools

        anchor_doc_ids = [d for d in (anchor_doc_ids or []) if d]
        if not anchor_doc_ids:
            return []
        spool_sources = resolve_spools(self.config).scope_sources()
        if not spool_sources:
            return []
        ground_ids = [
            d.id for d in self.library.list_documents_lite()
            if is_spool_source(d.source_name) and d.source_name in spool_sources
        ]
        if not ground_ids:
            return []

        def _norm(vec):
            arr = np.asarray(vec, dtype=np.float32)
            mag = float(np.linalg.norm(arr))
            return arr / mag if mag else arr

        anchor_emb_map = self.library.get_embeddings_for_ids(anchor_doc_ids)
        anchor_embs = [
            _norm(v) for v in (anchor_emb_map.get(i) for i in anchor_doc_ids)
            if v is not None
        ]
        if not anchor_embs:
            return []
        centroid = _norm(np.mean(np.stack(anchor_embs), axis=0))

        # Pre-window the ground by centroid similarity (fast matrix path), then
        # rank the window by relevance to the anchor docs — O(window), never a
        # full spool-corpus scan.
        window = max(limit * 8, 40)
        matrix = None
        try:
            matrix = self._get_embedding_matrix()
        except Exception:
            _logger.debug('env considerations: matrix load failed', exc_info=True)

        ground_pairs = []
        if matrix is not None:
            for did, _ in matrix.rank(centroid, ground_ids, window):
                row = matrix.id_to_row.get(did)
                if row is not None:
                    ground_pairs.append((did, _norm(matrix.M[row])))
        elif len(ground_ids) > 2000:
            return []  # no matrix over a large corpus — skip rather than scan
        else:
            emb_map = self.library.get_embeddings_for_ids(ground_ids)
            for did in ground_ids:
                vec = emb_map.get(did)
                if vec is not None:
                    ground_pairs.append((did, _norm(vec)))

        selected = select_ground(
            None, anchor_embs, ground_pairs,
            limit=limit, gate=gate, diversity=diversity,
        )
        notes = []
        for did in selected:
            doc = self.library.get_document(did)
            if doc is None:
                continue
            content = getattr(doc, 'content', '') or ''
            notes.append({
                'doc_id': did,
                'title': getattr(doc, 'title', did),
                'source': getattr(doc, 'source_name', None),
                'snippet': content[:200].strip(),
            })
        return notes
