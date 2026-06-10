"""Search and ranking logic for AriadneService."""
from __future__ import annotations

import hashlib
import json
import logging

from ariadne_mcp.models import DocumentResult, SearchResponse, SectionResult

_logger = logging.getLogger(__name__)


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
        """Deterministic SHA256 cache key that includes branch."""
        raw = json.dumps(
            [func_name, branch or '', list(args), sorted((kwargs or {}).items())],
            sort_keys=True, default=str,
        )
        return hashlib.sha256(raw.encode()).hexdigest()

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
        # legitimately differ).
        key = self._cache_key('search', query, feature, branch, status, limit, context_file, sections_only, role, source)
        if key in self._query_cache:
            return self._query_cache[key]

        # Persistent cache (cross-session, branch-aware, 30-day TTL)
        p_key = self._persistent_cache_key('search', branch, query, feature, status, limit, context_file, sections_only, role, source)
        cached_json = self.library.cache_get(p_key)
        if cached_json is not None:
            try:
                result = SearchResponse.model_validate_json(cached_json)
                self._query_cache[key] = result  # Populate in-memory too
                return result
            except Exception:
                pass  # Corrupt cache entry — recompute

        result = await self._search_uncached(query, feature, branch, status, limit, context_file, sections_only, role=role, source=source)
        self._query_cache[key] = result

        # Store in persistent cache
        try:
            self.library.cache_put(p_key, branch or '', query or '', result.model_dump_json())
        except Exception:
            _logger.debug('Failed to persist cache entry', exc_info=True)

        return result

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
    ) -> SearchResponse:
        """Uncached search implementation."""
        # Build the closure-scoped library view first so every read path
        # below cannot leak rows from outside the closure.
        scoped = self._resolve_scope(source)

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
            lite_docs = [
                d for d in lite_docs
                if (
                    d.content_type != 'audience_response'
                    or (d.metadata or {}).get('audience') == role
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
        candidate_ids = [d.id for d in lite_docs]

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
        if effective_query:
            # Phase 3: Load only embeddings for ranking (no content loaded yet)
            ranked_ids = await self._rank_ids_by_embedding(candidate_ids, effective_query, limit * 2)

            # Apply context boost
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
                ))
            else:
                documents.append(DocumentResult(
                    id=d.id, title=d.title, content_type=d.content_type,
                    content=_trim_related_documents(d.content),
                    source_files=d.source_files, metadata=d.metadata, score=score,
                ))

        return SearchResponse(
            documents=documents,
            event_id=event_id,
            suggested_queries=self._suggest_queries(effective_query, scored_docs) if effective_query else None,
            improvement_hint=self._improvement_hint(scored_docs, effective_query),
            truncated=truncated,                                                                                                                                        
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

    async def _rank_ids_by_embedding(self, doc_ids: list[str], query: str, limit: int) -> list[tuple[str, float]]:
        """Rank document IDs by embedding similarity, via the strategy chosen for the candidate count."""
        try:
            from library.embedding_ranking import select_ranker

            query_embedding = await self.embedding_service.embed(query)
            ranker = select_ranker(len(doc_ids), self._get_embedding_matrix, self.library)
            ranked = ranker.rank(query_embedding, doc_ids, limit)
            if ranked:
                return ranked
            return [(did, 0.0) for did in self._rank_by_query_ids(doc_ids, query, limit)]
        except Exception:
            _logger.debug('Embedding ranking failed, falling back to text matching', exc_info=True)
            return [(did, 0.0) for did in self._rank_by_query_ids(doc_ids, query, limit)]

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

        if not hasattr(self, '_embedding_matrix_cache'):
            matrix_dir = Path(self.library._conn_provider.path).parent / '.ariadne'
            self._embedding_matrix_cache = EmbeddingMatrix.load(matrix_dir)
        matrix = self._embedding_matrix_cache
        if matrix is None:
            return None
        with self.library._conn_provider.acquire() as conn:
            if not matrix.is_fresh(conn):
                return None
        return matrix
