"""Business logic facade for the Ariadne MCP server.

AriadneService is a lazy singleton that holds a single Library instance
for the server's lifetime, avoiding per-call open/close overhead.
All MCP tool handlers delegate here.

The service is composed from domain-specific mixins:
- SearchMixin: semantic search with multi-phase ranking
- AdminMixin: listing, sync, generation, feedback
- AnalysisMixin: issue analysis, Q&A, coverage, review, task context
- FacadesMixin: thin facades over Library, self-improvement
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

from ariadne_mcp.service_admin import AdminMixin
from ariadne_mcp.service_analysis import AnalysisMixin
from ariadne_mcp.service_facades import FacadesMixin
from ariadne_mcp.service_search import SearchMixin

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


def _confidence_from_scores(scores: list[float | None]) -> str:
    """Compute confidence level from similarity scores."""
    valid = [s for s in scores if s is not None]
    if not valid:
        return 'medium'
    best = max(valid)
    if best > 0.6:
        return 'high'
    if best > 0.4:
        return 'medium'
    return 'low'


class _UNSET:
    """Sentinel to distinguish 'not cached' from None."""


class AriadneService(
    SearchMixin,
    AdminMixin,
    AnalysisMixin,
    FacadesMixin,
):
    """Singleton facade over Library, Config, and git operations.

    Usage::

        svc = AriadneService.get()
        result = svc.search(query="feature system")
    """

    _instance: ClassVar[AriadneService | None] = None

    def __init__(self) -> None:
        self._library: object | None = None  # Library (lazy)
        self._config: object | None = None   # Config (lazy)
        self._embedding_service: object | None = None  # EmbeddingService (lazy, cached)
        self._cached_branch: str | None | type[_UNSET] = _UNSET
        self._branch_cached_at: float = 0.0
        self._query_cache: dict[int, Any] = {}

    # ------------------------------------------------------------------
    # Query cache (event-invalidated, no TTL)
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(func_name: str, *args: object, **kwargs: object) -> int:
        """Hash function name + arguments into a cache key."""
        return hash((func_name, args, tuple(sorted(kwargs.items()))))

    def clear_cache(self) -> None:
        """Invalidate all cached query results (in-memory and persistent)."""
        self._query_cache.clear()
        if self._library is not None:
            try:
                self.library.cache_clear()
            except Exception:
                pass  # Library may not be initialized yet

    @classmethod
    def get(cls) -> AriadneService:
        """Return the singleton instance, creating it on first call."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Lazy accessors
    # ------------------------------------------------------------------

    @property
    def config(self) -> 'Config':
        """Lazy-loaded Config instance."""
        if self._config is None:
            from config import get_config
            self._config = get_config()
        return self._config

    @property
    def library(self) -> 'Library':
        """Lazy-loaded Library instance (opened once, kept alive)."""
        if self._library is None:
            from library import Library
            self._library = Library(Path(self.config.db_path))
        return self._library

    @property
    def embedding_service(self) -> 'EmbeddingService':
        """Lazy-loaded, cached EmbeddingService (reuses httpx connection pool)."""
        if self._embedding_service is None:
            from embedding import EmbeddingConfig, EmbeddingService
            self._embedding_service = EmbeddingService(EmbeddingConfig())
        return self._embedding_service

    def get_branch(self, ttl_seconds: float = 30.0) -> str | None:
        """Get current git branch, cached with TTL to avoid per-call subprocess."""
        import time
        now = time.monotonic()
        if self._cached_branch is _UNSET or (now - self._branch_cached_at) > ttl_seconds:
            from git_ops import get_current_branch
            self._cached_branch = get_current_branch()
            self._branch_cached_at = now
        return self._cached_branch

    async def close(self) -> None:
        """Clean up resources (call on server shutdown)."""
        if self._embedding_service is not None:
            await self._embedding_service.close()
            self._embedding_service = None
        if self._library is not None:
            self._library.close()
            self._library = None

    def _resolve_source(self, source: str | None) -> str | None:
        """Resolve a source name, falling back to the default."""
        return source or self.config.default_source

    def _source_path(self, source_name: str) -> Path | None:
        """Get the resolved filesystem path for a source."""
        return self.config.get_source_path(source_name)

    def _resolve_scope(self, source: str | None) -> 'ScopedLibrary':
        """Resolve a source argument into a closure-scoped Library view.

        Delegates to :func:`scope_resolution.make_scoped_library` so the
        MCP service and the CLI dispatch share one resolution path.

        ``use_cwd=False``: the MCP server's process cwd is the Ariadne
        install (it's launched with a pinned ``--directory``), not the
        user's project, so cwd-based detection would always mis-resolve to
        whatever source contains Ariadne. The source must come from the
        explicit ``source`` argument (Claude's decomposition); an
        undetermined source fails closed with ``LookupError`` rather than
        silently answering from the wrong repo.
        """
        from scope_resolution import make_scoped_library
        return make_scoped_library(
            self.config, self.library, source, use_cwd=False,
        )
