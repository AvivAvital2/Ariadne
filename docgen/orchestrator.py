"""Orchestrator for documentation generation pipeline.

This module coordinates the entire documentation generation process,
from source analysis through validation and storage.
"""

from __future__ import annotations

import asyncio
import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from attrs import define, field, frozen

from docgen._legacy_analyzer import SourceAnalyzer
from docgen.crossref import CrossRefDetector, CrossReference, inject_related_section
from docgen.generator import (
    DocGenerator, GeneratedDoc, GeneratorConfig, PromptBundle,
)
from docgen.llm.anthropic import QuotaExhaustedError
from docgen.llm.factory import batch_strategy_for
from docgen.prompts import DocType
from docgen.staleness import StalenessTracker, find_catalog_files, find_python_files
from docgen.validator import ContentValidator, ValidationResult
from diagram_format import split_description_and_dot
from library import Library
from schema import Document
from writer import LibraryWriter

_logger = logging.getLogger(__name__)

# Number of times to re-roll a doc when validation fails. The first
# attempt is on top of this — total LLM calls per failing doc are
# 1 + MAX_VALIDATION_RETRIES. Naive re-roll relies on the model's
# default sampling variance producing a different sample; usually one
# retry suffices.
MAX_VALIDATION_RETRIES = 2


def _doc_provenance_for_files(source_files, low_confidence_languages) -> str:
    """Provenance for a doc from the language of the file(s) it documents:
    ``human-doc`` if any documented file is a low-confidence (human-authored)
    language, else ``code-derived``. The basis for provenance down-ranking.
    """
    from config import CODE_PROVENANCE, HUMAN_DOC_PROVENANCE, doc_provenance
    from docgen.catalog_extractor import _detect_language

    for f in source_files:
        if doc_provenance(_detect_language(Path(f)), low_confidence_languages) == HUMAN_DOC_PROVENANCE:
            return HUMAN_DOC_PROVENANCE
    return CODE_PROVENANCE


def _provenance_for_doc(gen_doc, low_confidence_languages) -> str:
    """A doc's provenance: prefer the language the generator stamped on the
    bundle (``metadata['language']``); fall back to detecting from the documented
    files for group/topic docs that span files and carry no single language.
    """
    lang = gen_doc.metadata.get('language')
    if lang is not None:
        from config import doc_provenance
        return doc_provenance(lang, low_confidence_languages)
    return _doc_provenance_for_files(gen_doc.source_files, low_confidence_languages)


@frozen
class OrchestratorConfig:
    """Configuration for the documentation generation orchestrator.

    Attributes:
        source_path: Path to source code directory.
        db_path: Path to the Ariadne library database.
        staleness_db_path: Path to the staleness tracking database.
        model: LLM model to use for generation.
        api_key: API key for the LLM provider.
        base_url: Base URL for the API endpoint.
        doc_types: Types of documentation to generate.
        concurrency: Maximum concurrent LLM requests.
        validate: Whether to validate generated content.
        inject_crossrefs: Whether to inject cross-references.
        force_regenerate: Regenerate even if not stale.
        dry_run: If True, don't write to database.
        source_name: Name of the source (for dependency lookup).
        dependencies: List of source names this source depends on.
    """

    source_path: Path
    db_path: Path = Path('ariadne.db')
    staleness_db_path: Path = Path('ariadne_staleness.db')
    model: str = 'gpt-5.2'
    api_key: str | None = None
    base_url: str = 'https://api.openai.com/v1'
    # LLM backend ("openai" | "anthropic"). Threaded through to
    # GeneratorConfig.provider; controls which client implementation
    # the generator uses. Default "openai" preserves historical behavior.
    provider: str = 'openai'
    doc_types: tuple[DocType, ...] = ('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram')
    doc_types_by_language: dict[str, tuple[str, ...]] = field(factory=dict)
    concurrency: int = 3
    validate: bool = True
    inject_crossrefs: bool = True
    force_regenerate: bool = False
    dry_run: bool = False
    source_name: str | None = None
    dependencies: tuple[str, ...] = ()
    target_path: Path | None = None
    themes_enabled: bool = True
    # Commit-diff gate (set by generate/onboard from the source's sync_state):
    # source-relative posix paths changed since the last synced commit. When
    # not None, run() generates exactly this set (∩ discovery) and skips the
    # hash/doc-type staleness check — the commit is the source of truth.
    # ``None`` keeps the legacy staleness path; ``force_regenerate`` overrides.
    restrict_to_files: frozenset[str] | None = None
    # Glob patterns matched against Path.match to exclude files from the
    # discovery walk. Set per-source via ``exclude:`` in ariadne.yaml; the
    # CLI threads it through. Use to keep secrets / credentials out of
    # prompts and the docs DB even when their extension is otherwise
    # catalog-eligible (e.g. .json/.yaml/.py).
    exclude_patterns: tuple[str, ...] = ()
    # Directory NAMES (not patterns) pruned from the discovery walk at
    # every depth. Set per-source via ``exclude_dirs:`` in ariadne.yaml.
    # More efficient and reliable than ``exclude_patterns`` for "exclude
    # this whole tree" cases (docs/, generated/, target/, vendor/).
    exclude_dir_names: tuple[str, ...] = ()
    # source_config is the per-source SCIP configuration (from
    # ``Config.get_source_scip_config``). When set, the catalog-driven
    # path routes Scala/Java files through ``scip_extractor`` so the
    # generator sees ElementInfo carrying SCIP-derived structured
    # documentation. None for sources that don't declare SCIP — the
    # legacy/Python-only path is unaffected.
    source_config: object | None = None
    # catalog_only_generator gates the catalog-driven generation path
    # (Catalog transition Phase 2). Default flipped to True in
    # Phase 2 Change 1 — the legacy SourceAnalyzer is Python-only and
    # SyntaxErrors on .md/.yaml/.json/Scala/Java files which the
    # multi-language catalog walker now surfaces by default. Sources
    # can still pass ``catalog_only_generator=False`` for emergency
    # rollback to the legacy path until ``analyzer.py`` /
    # ``metadata.py`` are renamed to ``_legacy_*.py`` in Phase 4.
    catalog_only_generator: bool = True

    # Batch API mode. ``"auto"`` defaults to batch when planned LLM calls
    # exceed ``auto_batch_threshold``; ``"always"`` forces batch (use for
    # large reruns); ``"never"`` forces sync (use for incremental edits
    # where 24h latency is unacceptable). Only meaningful for the Anthropic
    # provider — OpenAI batch is a separate API not yet wired here.
    #
    # NOTE: as of this writing, the runtime dispatch is NOT implemented —
    # ``run()`` always uses the streaming asyncio.Semaphore path regardless
    # of this setting. The ``BATCH_DISPATCH_IMPLEMENTED`` module flag below
    # gates whether the dry-run cost estimator claims batch pricing; until
    # the flag flips True, the estimator reports sync prices even on
    # ``batch_mode='always'`` to avoid billing-surprise hazards.
    batch_mode: str = 'auto'
    auto_batch_threshold: int = 200
    # Batched validation-retry: how many extra batch rounds to re-roll
    # validation failures (0 disables retrying; the initial failure count is
    # still reported). Each round is a full batch cycle — see
    # ``_batch_validation_retry``.
    max_batch_validation_retries: int = 3

    def __attrs_post_init__(self) -> None:
        """Resolve source_path to prevent relative path mismatches."""
        object.__setattr__(self, 'source_path', self.source_path.resolve())

    def config_hash(self) -> str:
        """Stable hash of the run-config attributes that determine
        batch contents.

        Used by ``_dispatch_batch`` to record/retrieve pending
        batches. The orchestrator's resume path refuses to adopt a
        pending batch whose hash mismatches the current run —
        submitting under one set of doc_types and resuming under
        another would land docs the user didn't ask for.
        """
        import hashlib

        h = hashlib.sha256()
        parts = [
            self.provider,
            self.model,
            '|'.join(sorted(self.doc_types)),
            str(self.source_path),
        ]
        h.update('||'.join(parts).encode('utf-8'))
        return h.hexdigest()
    ignore_staleness: bool | tuple[str, ...] = False


# ---------------------------------------------------------------------------
# Feature flag: batch runtime dispatch (flipped in #45.9)
# ---------------------------------------------------------------------------
#
# True since #45.9 — ``DocGenOrchestrator.run`` now forks to
# ``_run_batch`` when ``batch_mode`` resolves to batch (#45.8). The
# CLI's dry-run cost estimator consults this flag and now legitimately
# claims the 50% Anthropic batch discount because the runtime actually
# takes it.
#
# Kept as a module-level constant rather than removed entirely so the
# resolver's ``apply_dispatch_gate`` retains its pinch-point shape —
# a future temporary rollback (e.g., emergency revert) just toggles
# this back to False without unwinding the dispatch code. Once the
# batch path has settled in production, the flag and gate can be
# removed together.
BATCH_DISPATCH_IMPLEMENTED: bool = True


# ---------------------------------------------------------------------------
# Validation retry policy in batch mode (#45.6)
# ---------------------------------------------------------------------------
#
# Both dispatch paths now retry validation failures — but differently:
# streaming re-rolls inline via ``_validate_with_retry``; batch re-submits the
# failing prompts as FRESH BATCHES (``_batch_validation_retry``), staying in
# batch so the 50% discount is preserved (the original Decision-1a objection was
# that a *sync* retry would erase it — a batched retry does not). Rounds are
# capped by ``OrchestratorConfig.max_batch_validation_retries`` and the summary
# bills them. Set this False to hard-disable the batch retry path (the
# per-round cap via config is the normal knob).
BATCH_VALIDATION_RETRY: bool = True


@frozen
class GenerationResult:
    """Result of generating documentation for a single file.

    Attributes:
        source_path: Path to the source file.
        docs_generated: Number of documents generated.
        docs_failed: Number of documents that failed generation.
        validation_results: Validation results for each document.
        doc_ids: IDs of successfully created documents.
    """

    source_path: Path
    docs_generated: int
    docs_failed: int
    validation_results: tuple[ValidationResult, ...] = ()
    doc_ids: tuple[str, ...] = ()
    # Validation retry telemetry (Phase: in-loop retry on validation
    # failure). All three default to 0 for backwards-compat.
    validation_initial_failures: int = 0
    validation_retry_attempts: int = 0
    validation_recovered: int = 0


@frozen
class PipelineResult:
    """Result of the entire generation pipeline.

    Attributes:
        files_processed: Number of source files processed.
        files_skipped: Number of files skipped (up-to-date).
        docs_created: Total documents created.
        docs_failed: Total documents that failed.
        errors: List of error messages.
        validation_results: All validation results from generation.
        validation_initial_failures: Total docs whose first validation
            attempt failed (sum across all files).
        validation_retry_attempts: Total retry calls made.
        validation_recovered: Docs that passed validation on a retry.
        cache_stats: Provider-reported cache outcome counters. None for
            providers without caching support (OpenAI today). Surfaced
            so the CLI can confirm caching is engaged without log-grep.
    """

    files_processed: int
    files_skipped: int
    docs_created: int
    docs_failed: int
    errors: tuple[str, ...] = ()
    validation_results: tuple[ValidationResult, ...] = ()
    validation_initial_failures: int = 0
    validation_retry_attempts: int = 0
    validation_recovered: int = 0
    # Estimated USD spent on batched validation-retry rounds (the "retry bill").
    validation_retry_cost_usd: float = 0.0
    cache_stats: object | None = None  # CacheStats — typed object to avoid import cycle
    aborted: bool = False
    abort_reason: str = ''
    # Files that were queued but not started by the time we aborted, so
    # the CLI can compute the resume cost without re-discovering.
    unprocessed_files: tuple[Path, ...] = ()


ProgressCallback = Callable[[str, int, int], None]


# Async-shaped confirm callback for first-run UX prompts (#45.7).
# Async because the CLI's default implementation calls ``input()``
# inside ``asyncio.to_thread`` so the event loop isn't blocked while
# the user types. Tests inject ``lambda _msg: True`` (after wrapping
# in an async fn) to bypass.
ConfirmCallback = Callable[[str], Awaitable[bool]]


@frozen
class BatchAbort:
    """Reason a batch dispatch couldn't complete cleanly.

    Returned alongside an empty results dict from ``_dispatch_batch``
    so the caller (``_run_batch``) knows to surface
    ``aborted=True`` in ``PipelineResult``. ``detail`` carries the
    batch_id when one exists, so the user can resume or clear it via
    ``ariadne batch`` CLI.
    """
    reason: str  # short — e.g. 'quota at submit', 'fetch failed after retries'
    detail: str  # longer — error message, batch_id for resume, etc.


@define
class DocGenOrchestrator:
    """Orchestrates the documentation generation pipeline.

    This class coordinates:
    - Source file discovery and analysis
    - Staleness checking
    - LLM-based documentation generation
    - Content validation
    - Cross-reference injection
    - Storage in the Ariadne library

    Example:
        >>> config = OrchestratorConfig(
        ...     source_path=Path("mylib"),
        ...     model="gpt-5.2",
        ... )
        >>> async with DocGenOrchestrator(config) as orchestrator:
        ...     result = await orchestrator.run()
        ...     print(f"Created {result.docs_created} documents")
    """

    config: OrchestratorConfig
    # Optional progress hook; set before ``async with`` so __aenter__ phases
    # are visible. Same signature as the run() callback so the CLI can wire
    # one bar to both. Setting via attribute keeps the constructor simple.
    progress_callback: ProgressCallback | None = None
    # First-run UX prompt for batch dispatch (#45.7). When wired,
    # ``run()`` invokes this before submitting the batch to confirm
    # the user accepts the up-to-24h SLA. ``None`` means "no
    # confirmation needed" — the test/CI default.
    confirm_callback: ConfirmCallback | None = None
    _library: Library | None = field(default=None, init=False)
    _writer: LibraryWriter | None = field(default=None, init=False)
    _generator: DocGenerator | None = field(default=None, init=False)
    _staleness: StalenessTracker | None = field(default=None, init=False)
    _doc_types_by_file: dict = field(factory=dict, init=False)
    _low_confidence_langs: tuple[str, ...] | None = field(default=None, init=False)
    _validator: ContentValidator = field(factory=ContentValidator, init=False)
    _analyzer: SourceAnalyzer = field(factory=SourceAnalyzer, init=False)
    # Cross-source SCIP graph loaded from ariadne.db at __aenter__.
    # Threaded into ``enrich_file`` so ``EnrichedFileBundle.scip``
    # carries cross-file callers/callees (Phase 2 Change 2). None when
    # the catalog path is off — no need to pay the load on legacy runs.
    _cross_source_graph: 'CrossSourceGraph | None' = field(default=None, init=False)
    # Cached ScopedLibrary view, built once per orchestrator lifetime.
    # ``self.config.source_name`` is immutable for the orchestrator's
    # __aenter__/__aexit__ scope, so the closure resolution is stable;
    # rebuilding it per-call (especially inside per-doc loops at
    # ~line 1917) was paying for Config.scope_closure's DFS + cycle
    # check on every iteration AND throwing away the lazy SCIP graph
    # cache inside ScopedLibrary.
    _scoped: 'object | None' = field(default=None, init=False)

    def _emit(self, message: str, current: int = 0, total: int = 0) -> None:
        """Fire the progress callback if set. Each phase boundary calls this
        BEFORE doing the work so users see what's about to happen, not what
        just finished.
        """
        if self.progress_callback is not None:
            self.progress_callback(message, current, total)

    def _scoped_lib(self):
        """Closure-scoped library view for this orchestrator's source.

        Built once and cached on ``self._scoped``. ``self.config.
        source_name`` is fixed for the orchestrator's lifetime, so the
        closure resolution doesn't need to re-run per call. The cache
        also preserves ScopedLibrary's lazy SCIP graph between
        invocations from the same orchestrator instance.
        """
        if self._scoped is not None:
            return self._scoped
        from config import get_config
        from scope_resolution import (
            make_global_scoped_library, make_scoped_library,
        )

        cfg = get_config()
        source = self.config.source_name
        if source is None:
            self._scoped = make_global_scoped_library(cfg, self._library)
        else:
            self._scoped = make_scoped_library(cfg, self._library, source)
        return self._scoped

    def _low_confidence_languages(self) -> tuple[str, ...]:
        """The source's human-prose languages (cached; resolved once, like
        :meth:`_scoped_lib`). Basis for each doc's provenance tag.
        """
        if self._low_confidence_langs is None:
            from config import DEFAULT_LOW_CONFIDENCE_DOC_LANGUAGES, get_config
            sc = get_config().get_source_config(self.config.source_name) if self.config.source_name else None
            self._low_confidence_langs = sc.low_confidence_doc_languages if sc else DEFAULT_LOW_CONFIDENCE_DOC_LANGUAGES
        return self._low_confidence_langs

    async def __aenter__(self) -> DocGenOrchestrator:
        """Enter async context and initialize components."""
        # Suppress SyntaxWarning emitted by ast.parse() on source files that
        # contain invalid escape sequences in string literals (e.g., "\d",
        # "\[", "\/") — common in older or non-Python-idiomatic codebases.
        # These are signals for that codebase's authors, not actionable on
        # our side, and they flood the terminal/log during catalog enrichment
        # of large source trees. Caller process scope; doesn't affect tests.
        import warnings
        warnings.filterwarnings('ignore', category=SyntaxWarning)

        # Library() runs schema migrations + an embedding-normalization scan
        # over every documents/chunks row. On a populated DB (5K+ docs) this
        # can take 10-30s; signaling here removes the "stuck on Starting" UX.
        self._emit('Opening library DB and running schema migrations...')
        self._library = Library(self.config.db_path)
        self._writer = LibraryWriter(self._library)

        # Load the cross-source SCIP graph once so each ``enrich_file``
        # call can attach ``ScipFileMetadata`` carrying cross-file
        # callers/callees. Skipped on the legacy path — enrich_file is
        # never invoked there. Empty DB still produces a valid empty
        # graph; no need for a "graph exists" preflight.
        if self.config.catalog_only_generator:
            from docgen.scip_cross_source import CrossSourceGraph
            graph = CrossSourceGraph()
            with self._library._conn_provider.acquire() as conn:
                graph.load_from(conn)
            self._cross_source_graph = graph

        # Resolve API key per provider — Anthropic uses ANTHROPIC_API_KEY,
        # everything else falls back to OPENAI_API_KEY (covers OpenAI direct
        # and OpenAI-compatible proxies).
        if self.config.api_key:
            api_key = self.config.api_key
        elif self.config.provider == 'anthropic':
            api_key = os.environ.get('ANTHROPIC_API_KEY', '')
        else:
            api_key = os.environ.get('OPENAI_API_KEY', '')

        # Resolve base URL: per-provider default unless explicitly overridden.
        # The OpenAI default lives in OrchestratorConfig.base_url; if the
        # user sticks with the default but selects Anthropic, swap to the
        # Anthropic default so requests don't 404 against api.openai.com.
        base_url = self.config.base_url
        if (
            self.config.provider == 'anthropic'
            and base_url == 'https://api.openai.com/v1'
        ):
            base_url = 'https://api.anthropic.com/v1'

        self._emit(f'Initializing LLM provider ({self.config.provider}, {self.config.model})...')
        gen_config = GeneratorConfig(
            model=self.config.model,
            api_key=api_key,
            base_url=base_url,
            doc_types=self.config.doc_types,
            provider=self.config.provider,
        )
        self._generator = DocGenerator(config=gen_config, analyzer=self._analyzer)
        await self._generator.__aenter__()

        self._emit('Opening staleness DB...')
        self._staleness = StalenessTracker(self.config.staleness_db_path, doc_types_by_language=self.config.doc_types_by_language)

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context and cleanup."""
        if self._generator:
            await self._generator.__aexit__(exc_type, exc_val, exc_tb)
        if self._writer:
            await self._writer.close()
        if self._staleness:
            self._staleness.close()

    async def run(
        self,
        progress_callback: ProgressCallback | None = None,
        crossref_progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Run the full documentation generation pipeline.

        Args:
            progress_callback: Optional per-file progress reporting.
            crossref_progress: Optional progress reporting for the crossref
                stage (fired per scoped doc as references are computed).
            progress_callback: Optional callback for progress reporting.
                               Called with (message, current, total).

        Returns:
            PipelineResult with summary statistics.
        """
        if not self._library or not self._generator or not self._staleness:
            msg = 'Orchestrator must be used as async context manager'
            raise RuntimeError(msg)

        # Use the run-scoped callback if provided; otherwise the one set
        # before __aenter__ stays in effect. Backward-compat with callers
        # that still pass progress_callback only to run().
        if progress_callback is not None:
            self.progress_callback = progress_callback

        # A pinned/immutable spool corpus already fully generated at this sha has
        # a frozen file set — nothing new to discover, nothing changed to
        # regenerate — so skip discovery + staleness + generation entirely.
        if self._generation_already_complete():
            self._emit(
                'Corpus unchanged (pinned) and already fully generated — '
                'skipping documentation.',
            )
            return PipelineResult(
                files_processed=0, files_skipped=0,
                docs_created=0, docs_failed=0,
            )
        # Not skipping: drop any stale completion marker so a partial/interrupted
        # run below never leaves one that would wrongly skip next time.
        self._invalidate_generation_marker()

        # Check dependency documentation exists
        if self.config.dependencies:
            self._emit('Checking dependency documentation...')
            self._check_dependency_docs()

        # File discovery — multi-language via find_catalog_files when the
        # catalog-driven path is on; legacy Python-only otherwise.
        self._emit('Discovering source files...')
        discover = (
            find_catalog_files
            if self.config.catalog_only_generator
            else find_python_files
        )
        # Merge user excludes with the canonical discovery defaults
        # (test files + vendored installers like get-pip.py — single-sourced
        # in config.DEFAULT_EXCLUDE_FILE_PATTERNS so no two-list drift).
        from config import DEFAULT_EXCLUDE_FILE_PATTERNS
        excludes = DEFAULT_EXCLUDE_FILE_PATTERNS + self.config.exclude_patterns

        search_root = self.config.source_path
        if self.config.target_path:
            full_target = search_root / self.config.target_path
            if full_target.is_file():
                all_files = [full_target]
            else:
                search_root = full_target
                all_files = discover(
                    search_root,
                    exclude_patterns=excludes,
                    exclude_dir_names=self.config.exclude_dir_names,
                )
        else:
            all_files = discover(
                search_root,
                exclude_patterns=excludes,
                exclude_dir_names=self.config.exclude_dir_names,
            )
        _logger.info('Found %d catalog files in %s', len(all_files), search_root)
        self._emit(f'Found {len(all_files)} files; checking staleness...')

        # Filter to stale/new files unless force regenerate.
        # Type-aware: a file with explanation but no architecture is stale
        # when --types architecture is requested, even with matching sha.
        # Eliminates the need for --force when adding doc types incrementally.
        files_to_process, self._doc_types_by_file = self._staleness.files_for_generation(
            all_files,
            base_path=self.config.source_path,
            requested_types=self.config.doc_types,
            library=self._library,
            is_exempt=self._staleness_exempt(),
            restrict_to_files=self.config.restrict_to_files,
            force=self.config.force_regenerate,
        )

        files_skipped = len(all_files) - len(files_to_process)
        _logger.info(
            'Processing %d files (%d up-to-date)',
            len(files_to_process),
            files_skipped,
        )

        # Phase C fork (#45.8): if batch dispatch resolves and is wired,
        # route to _run_batch instead of the streaming Semaphore loop.
        # The gate (apply_dispatch_gate) downgrades batch=True to sync
        # while BATCH_DISPATCH_IMPLEMENTED is False, so until #45.9
        # this branch is never taken in production.
        from docgen.batch_resolution import (
            apply_dispatch_gate,
            resolve_batch_decision,
        )

        planned_calls = (
            len(files_to_process) * len(self.config.doc_types)
        )
        batch_resolved, batch_reason = resolve_batch_decision(
            provider=self.config.provider,
            batch_mode=self.config.batch_mode,
            planned_calls=planned_calls,
            auto_threshold=self.config.auto_batch_threshold,
        )
        batch_resolved, batch_reason = apply_dispatch_gate(
            batch_resolved, batch_reason,
        )
        if batch_resolved:
            _logger.info('Batch dispatch resolved: %s', batch_reason)
            result = await self._run_batch(
                files_to_process, files_skipped,
                crossref_progress=crossref_progress,
            )
            return self._finalize_generation(result)
        _logger.info('Sync dispatch resolved: %s', batch_reason)

        if files_to_process:
            self._emit(
                f'Generating {len(files_to_process)} files '
                f'({files_skipped} up-to-date) — first {self.config.concurrency} '
                f'in flight...',
                0, len(files_to_process),
            )
        else:
            self._emit('Nothing to generate — all files up-to-date.', 0, 0)

        # Process files
        results: list[GenerationResult] = []
        errors: list[str] = []

        semaphore = asyncio.Semaphore(self.config.concurrency)
        progress_lock = asyncio.Lock()
        completed = 0
        total_to_process = len(files_to_process)

        # Abort coordination: if any file hits QuotaExhaustedError, set the
        # flag so subsequent files skip immediately. In-flight files finish
        # naturally — we don't kill them mid-call, just stop dispatching new.
        abort_event = asyncio.Event()
        abort_reason = ''
        unprocessed: list[Path] = []

        async def process_file(path: Path) -> GenerationResult | None:
            nonlocal completed, abort_reason
            if abort_event.is_set():
                # Already aborted — record this file as unprocessed and skip.
                async with progress_lock:
                    unprocessed.append(path)
                return None
            async with semaphore:
                # Re-check after acquiring (a peer may have aborted while we waited).
                if abort_event.is_set():
                    async with progress_lock:
                        unprocessed.append(path)
                    return None
                self._emit(
                    f'Processing {path.name}', completed, total_to_process,
                )
                try:
                    result = await self._process_file(path)
                except QuotaExhaustedError as e:
                    if not abort_event.is_set():
                        abort_reason = str(e) or 'anthropic quota exhausted'
                        abort_event.set()
                        _logger.error(
                            'Aborting run due to quota exhaustion: %s',
                            abort_reason,
                        )
                        self._emit(
                            f'Aborting: {abort_reason[:80]}',
                            completed, total_to_process,
                        )
                    async with progress_lock:
                        unprocessed.append(path)
                    return None
                except Exception as e:
                    _logger.error('Failed to process %s: %s', path, e)
                    errors.append(f'{path}: {e}')
                    result = None
            async with progress_lock:
                completed += 1
                self._emit(
                    f'Completed {path.name}', completed, total_to_process,
                )
            return result

        tasks = [process_file(path) for path in files_to_process]
        task_results = await asyncio.gather(*tasks)

        for result in task_results:
            if result:
                results.append(result)

        # Compute summary
        docs_created = sum(r.docs_generated for r in results)
        docs_failed = sum(r.docs_failed for r in results)
        val_initial_failures = sum(r.validation_initial_failures for r in results)
        val_retry_attempts = sum(r.validation_retry_attempts for r in results)
        val_recovered = sum(r.validation_recovered for r in results)

        # Aggregate validation results
        all_validation_results: list[ValidationResult] = []
        for r in results:
            all_validation_results.extend(r.validation_results)

        if abort_event.is_set():
            # Skip post-processing on abort — crossrefs/themes over a partial
            # set wastes work the resume run will redo. Mark and return.
            self._emit(
                f'Aborted: {abort_reason[:80]}',
                completed, total_to_process,
            )
        elif errors:
            self._emit(
                f'Generation hit {len(errors)} error(s) — skipping post-processing; existing themes/crossrefs preserved.',
                completed, total_to_process,
            )
        else:
            self._emit(
                'Generation complete; running post-processing...',
                len(files_to_process), len(files_to_process),
            )
            # Post-processing: themes first, then crossrefs. Themes are
            # useful on their own; crossrefs is an O(N²) hot spot and may
            # be skipped via flag or threshold guard.
            await self._post_process(
                results, crossref_progress=crossref_progress,
            )

        # Pull cache stats from the provider through the generator. The
        # generator stores the provider as `_provider` (private) and the
        # provider exposes `cache_stats`. Wrapped in getattr so generator
        # implementations without a provider field don't crash.
        cache_stats = None
        provider = getattr(self._generator, '_provider', None)
        if provider is not None:
            cache_stats = getattr(provider, 'cache_stats', None)

        return self._finalize_generation(PipelineResult(
            files_processed=len(files_to_process) - len(unprocessed),
            files_skipped=files_skipped,
            docs_created=docs_created,
            docs_failed=docs_failed,
            errors=tuple(errors),
            validation_results=tuple(all_validation_results),
            validation_initial_failures=val_initial_failures,
            validation_retry_attempts=val_retry_attempts,
            validation_recovered=val_recovered,
            cache_stats=cache_stats,
            aborted=abort_event.is_set(),
            abort_reason=abort_reason,
            unprocessed_files=tuple(unprocessed),
        ))

    def _generation_already_complete(self) -> bool:
        """True iff this source is a pinned/immutable spool corpus already fully
        generated at its current sha for the requested doc types — the case
        where discovery + staleness + generation can be skipped entirely."""
        if self.config.force_regenerate:
            return False
        source_name = self.config.source_name
        if not source_name:
            return False
        from config import get_config
        if not get_config().source_staleness_exempt(source_name):
            return False
        from docgen.generation_marker import (
            current_corpus_shas,
            generation_complete,
        )
        root = self.config.source_path
        if not current_corpus_shas(root):
            return False  # not a pinned/fetched corpus (no corpus-sha markers)
        return generation_complete(
            root / '.ariadne', root, requested_doc_types=self.config.doc_types,
        )

    def _invalidate_generation_marker(self) -> None:
        from docgen.generation_marker import invalidate_marker

        invalidate_marker(self.config.source_path / '.ariadne')

    def _write_generation_marker(self) -> None:
        """Record that docs are generated for this (pinned, staleness-exempt)
        corpus, so a later run skips discovery/generation. No-op for a
        non-exempt or non-pinned (no corpus-sha markers) source."""
        source_name = self.config.source_name
        if not source_name:
            return
        from config import get_config
        if not get_config().source_staleness_exempt(source_name):
            return
        from docgen.generation_marker import current_corpus_shas, write_marker

        root = self.config.source_path
        shas = current_corpus_shas(root)
        if shas:
            write_marker(
                root / '.ariadne', corpus_shas=shas, doc_types=self.config.doc_types,
            )

    def _finalize_generation(self, result: PipelineResult) -> PipelineResult:
        """Belt-and-suspenders marker write at run end (``_post_process`` already
        writes it BEFORE themes, so an abort mid-themes doesn't lose it). No-op
        on failure/abort, or for non-exempt / non-pinned sources."""
        if not (result.docs_failed or result.errors or result.aborted):
            self._write_generation_marker()
        return result

    async def _process_file(self, path: Path) -> GenerationResult:
        """Process a single source file.

        Routes through one of two paths:
        - Legacy (default): SourceAnalyzer + ``generate_for_module`` —
          Python only.
        - Catalog (when ``OrchestratorConfig.catalog_only_generator`` is
          True): catalog_extractor + enrichment + ``generate_from_elements`` —
          multi-language.
        """
        if not self._generator or not self._writer or not self._staleness:
            msg = 'Components not initialized'
            raise RuntimeError(msg)

        if self.config.catalog_only_generator:
            generated_docs, line_count = await self._catalog_generate(path)
        else:
            generated_docs, line_count = await self._legacy_generate(path)

        if generated_docs is None:
            # Pre-generation failure (SyntaxError, unsupported file, etc.).
            return GenerationResult(
                source_path=path,
                docs_generated=0,
                docs_failed=1,
            )

        if not generated_docs:
            return GenerationResult(
                source_path=path,
                docs_generated=0,
                docs_failed=len(self.config.doc_types),
            )

        # Validate (with retry on failure) and store
        validation_results: list[ValidationResult] = []
        doc_ids: list[str] = []
        failed = 0
        val_initial_failures = 0
        val_retry_attempts = 0
        val_recovered = 0

        for gen_doc in generated_docs:
            if self.config.validate:
                async def _regen(p=path, dt=gen_doc.doc_type):
                    return await self._regenerate_doc(p, dt)

                final_doc, result, retries = await self._validate_with_retry(
                    gen_doc, _regen,
                )
                validation_results.append(result)

                if retries > 0:
                    val_initial_failures += 1
                    val_retry_attempts += retries
                    if final_doc is not None:
                        val_recovered += 1
                    else:
                        _logger.warning(
                            'Validation failed for %s (%s) after %d retries: %d errors',
                            gen_doc.title, gen_doc.doc_type, retries, result.errors,
                        )

                if final_doc is None:
                    failed += 1
                    continue
                gen_doc = final_doc

            if not self.config.dry_run:
                doc = await self._store_document(gen_doc)
                if doc:
                    doc_ids.append(doc.id)
                else:
                    failed += 1

        if doc_ids and not self.config.dry_run:
            await self._staleness.record_documentation_async(
                path, doc_ids, base_path=self.config.source_path
            )

        return GenerationResult(
            source_path=path,
            docs_generated=len(doc_ids),
            docs_failed=failed,
            validation_results=tuple(validation_results),
            doc_ids=tuple(doc_ids),
            validation_initial_failures=val_initial_failures,
            validation_retry_attempts=val_retry_attempts,
            validation_recovered=val_recovered,
        )

    def _doc_types_for(self, path):
        """Doc types to (re)generate for ``path``: the per-file narrowed set from
        staleness (only the missing/changed types), or the full configured set
        when not narrowed (force/restrict, or an unmapped file) — then capped by
        any per-language override (``doc_types_by_language``: the doc-type
        screen's per-format excludes)."""
        narrowed = getattr(self, "_doc_types_by_file", {}).get(path)
        base = tuple(narrowed) if narrowed else self.config.doc_types
        override = self.config.doc_types_by_language
        if override:
            from docgen.catalog_extractor import _detect_language
            from docgen.prompts import filter_doc_types_for_language
            base = filter_doc_types_for_language(base, _detect_language(path), override)
        return base

    async def _legacy_generate(
        self, path: Path,
    ) -> tuple[list[GeneratedDoc] | None, int]:
        """Legacy SourceAnalyzer path (Python only).

        Returns (generated_docs | None, line_count). None signals a
        pre-generation failure (SyntaxError) so the caller surfaces it.
        """
        try:
            metadata = self._analyzer.analyze_file(path)
        except SyntaxError as e:
            _logger.warning('Syntax error in %s: %s', path, e)
            return None, 0

        if metadata.line_count > 200 and not self.config.dry_run:
            self._store_file_map(metadata)

        generated_docs = await self._generator.generate_for_module(
            metadata, self._doc_types_for(path),
        )
        return generated_docs, metadata.line_count

    async def _catalog_generate(
        self, path: Path,
    ) -> tuple[list[GeneratedDoc] | None, int]:
        """Catalog-driven path (multi-language) — Phase 2.5 feature flag.

        Builds an EnrichedFileBundle via ``catalog_enrich.enrich_file`` and
        dispatches to ``DocGenerator.generate_from_elements``. Doc types
        are filtered per language inside the generator. ``source_config``
        is forwarded so Scala/Java route through SCIP.
        """
        from docgen.catalog_enrich import enrich_file

        bundle = enrich_file(
            path,
            source_root=self.config.source_path,
            source_config=self.config.source_config,
            cross_source_graph=self._cross_source_graph,
        )
        if bundle is None:
            _logger.warning(
                'Could not build bundle for %s (unsupported extension or read failed)',
                path,
            )
            return None, 0

        generated_docs = await self._generator.generate_from_elements(
            bundle, doc_types=self._doc_types_for(path),
        )
        return generated_docs, bundle.line_count

    # -------------------------------------------------------------------
    # Build-only path (#45.4) — collect prompts upfront for batch dispatch.
    # _build_prompts_for_file mirrors the legacy/catalog fork in
    # _process_file, but produces PromptBundles instead of generated docs.
    # _collect_prompts walks a list of files, collecting all prompts
    # plus a file→prompt-indices map for later result reassembly.
    # -------------------------------------------------------------------

    async def _build_prompts_for_file(
        self, path: Path,
    ) -> list[PromptBundle]:
        """Build PromptBundles for one file via legacy or catalog path.

        Mirrors the dispatch logic in :meth:`_process_file`. May raise
        ``SyntaxError`` if the legacy analyzer can't parse the file —
        the caller (:meth:`_collect_prompts`) catches and routes to
        ``pre_gen_failed`` so one bad file doesn't forfeit the batch.

        Returns ``[]`` if the catalog enrichment fails (unsupported
        extension or read error) so the caller can route those to
        ``pre_gen_failed`` too — same fall-through as the streaming
        catalog path.

        File-map sidecars for files >200 lines are written here for
        feature parity with ``_legacy_generate``. Without this, batch
        runs would silently drop file_map sidecars that streaming runs
        produce.
        """
        if not self._generator:
            msg = 'Components not initialized'
            raise RuntimeError(msg)

        if self.config.catalog_only_generator:
            from docgen.catalog_enrich import enrich_file
            bundle = enrich_file(
                path,
                source_root=self.config.source_path,
                source_config=self.config.source_config,
                cross_source_graph=self._cross_source_graph,
            )
            if bundle is None:
                _logger.warning(
                    'Could not build bundle for %s '
                    '(unsupported extension or read failed)',
                    path,
                )
                return []
            return await self._generator.build_prompts_for_bundle(
                bundle, doc_types=self._doc_types_for(path),
            )

        # Legacy path: analyze_file may raise SyntaxError; let it
        # propagate so _collect_prompts can route to pre_gen_failed.
        metadata = self._analyzer.analyze_file(path)
        if metadata.line_count > 200 and not self.config.dry_run:
            self._store_file_map(metadata)
        return await self._generator.build_prompts_for_module(
            metadata, self._doc_types_for(path),
        )

    async def _collect_prompts(
        self, files: list[Path],
    ) -> tuple[list[PromptBundle], dict[Path, list[int]], list[Path]]:
        """Collect prompts across all files for batch dispatch.

        Args:
            files: Source files to process.

        Returns:
            ``(prompts, file_to_idxs, pre_gen_failed)``:

            - ``prompts``: flat list of ``PromptBundle``, one per
              ``(file, doc_type)`` pair. Order is deterministic
              (caller-supplied file order, then per-file doc_type
              order from the build helpers).
            - ``file_to_idxs``: ``file → list of indices into
              prompts``. The orchestrator's ``_assemble_and_store``
              (#45.6) uses this to map per-prompt batch responses
              back to per-file ``GenerationResult`` objects.
            - ``pre_gen_failed``: files that couldn't be analyzed
              (SyntaxError, unsupported extension, missing bundle).
              Counted as failures in the final ``PipelineResult``.
        """
        prompts: list[PromptBundle] = []
        file_to_idxs: dict[Path, list[int]] = {}
        pre_gen_failed: list[Path] = []

        for path in files:
            try:
                file_prompts = await self._build_prompts_for_file(path)
            except SyntaxError as e:
                _logger.warning('Syntax error in %s: %s', path, e)
                pre_gen_failed.append(path)
                continue
            if not file_prompts:
                # Catalog enrichment failed or no doc_types apply —
                # equivalent to a pre-gen failure in the legacy path.
                pre_gen_failed.append(path)
                continue

            idxs: list[int] = []
            for p in file_prompts:
                idxs.append(len(prompts))
                prompts.append(p)
            file_to_idxs[path] = idxs

        return prompts, file_to_idxs, pre_gen_failed

    # -------------------------------------------------------------------
    # Batch dispatch (#45.5) — submit, poll, fetch with retry.
    # _dispatch_batch wraps the full provider interaction; quota at
    # any phase routes to BatchAbort. _fetch_with_retry is a focused
    # helper for the post-success fetch with exp-backoff.
    # -------------------------------------------------------------------

    def _themes_batch_strategy(self):
        """Batch strategy for theme summarization when the run chose batch mode,
        so a batch onboard batches themes too — not just generation/embeddings.
        Themes run in _post_process (themes_enabled defaults on), so without
        this they always summarized live at full price despite the batch choice.
        None (live) otherwise, or when the provider has no batch backend."""
        if str(self.config.batch_mode or '').lower() not in ('always', 'batch'):
            return None
        return self._batch_strategy()

    def _batch_strategy(self):
        """The configured provider's BatchStrategy, or None if the generator
        has no provider. Batch dispatch goes through this — the provider
        carries no batch methods (they live in the per-provider strategy,
        selected by ``batch_strategy_for``)."""
        provider = getattr(self._generator, '_provider', None)
        if provider is None:
            return None
        try:
            return batch_strategy_for(provider)
        except ValueError:
            # Provider type with no batch backend — degrade to the "no
            # strategy" path the callers already handle (BatchAbort /
            # PipelineResult), rather than letting a ValueError crash the run.
            _logger.warning(
                'No batch strategy for provider type %s; treating as no '
                'batch backend (sync only)', type(provider).__name__,
            )
            return None

    async def _fetch_with_retry(
        self, strategy: object, batch_id: str, max_retries: int = 5,
    ) -> dict[str, str | None] | None:
        """Fetch batch results with exp-backoff retry on transient
        HTTP errors.

        Once a batch is submitted we've paid Anthropic for it, so
        losing the results to a transient network blip is
        unacceptable. ``max_retries`` attempts with exponential
        backoff capped at 60s give the network ~2 minutes of grace
        before we surface a hard failure (caller's abort path).

        ``QuotaExhaustedError`` propagates immediately — the caller's
        abort path handles it, and retrying a quota error just burns
        attempts. Other ``httpx`` errors retry.
        """
        import httpx

        delay = 1.0
        last_error: Exception | None = None
        for attempt in range(max_retries):
            try:
                return await strategy.fetch_batch_results(batch_id)
            except QuotaExhaustedError:
                raise
            except httpx.HTTPError as e:
                last_error = e
                _logger.warning(
                    'Batch fetch attempt %d/%d failed: %s',
                    attempt + 1, max_retries, e,
                )
                if attempt + 1 < max_retries:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, 60.0)

        _logger.error(
            'Batch fetch failed after %d retries: %s',
            max_retries, last_error,
        )
        return None

    def _record_pending_batch(
        self,
        batch_id: str,
        prompts: list[PromptBundle],
        file_to_idxs: dict[Path, list[int]],
        config_hash: str,
    ) -> None:
        """Serialize prompts + file_to_idxs to JSON and persist via
        StalenessTracker.

        Path keys in ``file_to_idxs`` are stringified for JSON. The
        orchestrator's resume path reads these back via
        ``find_pending_batch`` and restores ``Path`` instances by
        passing the strings through ``Path()`` again — round-trip
        is lossy only on Windows-vs-POSIX path separators, which is
        acceptable for a single-machine resume scenario.
        """
        import json

        if not self._staleness:
            return

        prompts_payload = [
            {
                'file': str(p.file),
                'doc_type': p.doc_type,
                'system_prompt': p.system_prompt,
                'user_prompt': p.user_prompt,
                'title': p.title,
                'metadata': p.metadata,
            }
            for p in prompts
        ]
        file_to_idxs_payload = {
            str(path): idxs for path, idxs in file_to_idxs.items()
        }
        self._staleness.record_pending_batch(
            batch_id=batch_id,
            prompts_json=json.dumps(prompts_payload),
            file_to_idxs_json=json.dumps(file_to_idxs_payload),
            config_hash=config_hash,
        )

    async def _dispatch_batch(
        self,
        prompts: list[PromptBundle],
        file_to_idxs: dict[Path, list[int]],
        config_hash: str,
        record_pending: bool = True,
    ) -> tuple[dict[str, str | None], BatchAbort | None]:
        """Submit, poll, fetch a batch of prompts.

        ``record_pending`` (default True) records the batch in
        ``pending_batches`` for crash-resume. Validation-retry sub-batches pass
        False: they are best-effort re-rolls, and recording a SUBSET batch under
        the run's ``config_hash`` would let resume adopt it as the main batch.

        Returns ``(results_by_cid, abort_or_None)``. On abort:
        - Quota at submit: no batch was created; nothing to clean
          up; pending_batches stays empty.
        - Quota at poll: ``batch_id`` IS recorded in pending_batches.
          The user can resume on the next ``ariadne generate`` (which
          consults ``find_pending_batch``) or clear via
          ``ariadne batch clear``. Clearing here would forfeit the
          paid-for batch.
        - Fetch failure after retries: same as quota-at-poll —
          batch_id stays in pending_batches; user can retry or clear.

        On success the pending_batches row is cleared so it doesn't
        stick around as an orphan.
        """
        from docgen.llm.anthropic import BatchRequest

        if not self._generator or not self._staleness:
            msg = 'Components not initialized'
            raise RuntimeError(msg)

        strategy = self._batch_strategy()
        if strategy is None:
            return {}, BatchAbort(
                reason='no provider',
                detail='Generator has no LLM provider',
            )

        requests = [
            BatchRequest(
                custom_id=str(i),
                system_prompt=p.system_prompt,
                user_prompt=p.user_prompt,
            )
            for i, p in enumerate(prompts)
        ]

        self._emit(f'Submitting {len(prompts)} prompts to the batch API...', 0, 0)

        try:
            submission = await strategy.submit_batch(requests)
        except QuotaExhaustedError as e:
            return {}, BatchAbort(
                reason='quota exhausted at submit',
                detail=str(e) or 'anthropic quota exhausted',
            )

        # Persist for resume — record AFTER submit so we have a real
        # batch_id, BEFORE poll so a crash mid-poll can resume. Retry
        # sub-batches skip this (see ``record_pending``).
        if record_pending:
            self._record_pending_batch(
                batch_id=submission.batch_id,
                prompts=prompts,
                file_to_idxs=file_to_idxs,
                config_hash=config_hash,
            )

        # Poll with progress wired to _emit.
        total = len(prompts)

        def on_poll_progress(
            processing: int, succeeded: int, errored: int,
        ) -> None:
            done = succeeded + errored
            if done > 0:
                # Real per-request counts have landed — drive a determinate
                # bar so the user sees the Anthropic-style 1200/1800
                # progression as requests finish.
                in_flight = f' ({processing} in flight)' if processing else ''
                self._emit(
                    f'Batch processing: {done}/{total} done{in_flight}',
                    done, total,
                )
            elif processing > 0:
                # Actively processing but nothing finished yet. Pulse rather
                # than park a determinate bar at 0/N (which reads as frozen).
                self._emit(
                    f'Batch processing — {processing} of {total} in flight...',
                    0, 0,
                )
            else:
                # All-zero counts: queued by the provider, awaiting
                # processing. Pulse here too so the bar stays alive.
                self._emit(
                    f'Batch queued — awaiting processing ({total} prompts)',
                    0, 0,
                )

        try:
            await strategy.poll_batch(
                submission.batch_id,
                on_progress=on_poll_progress,
            )
        except QuotaExhaustedError as e:
            # Don't clear — user resumes via run() or clears via CLI.
            return {}, BatchAbort(
                reason='quota exhausted during poll',
                detail=f'batch_id={submission.batch_id}; {e}',
            )

        self._emit(f'Batch ended — downloading {total} results...', 0, 0)
        results = await self._fetch_with_retry(
            strategy, submission.batch_id,
        )
        if results is None:
            return {}, BatchAbort(
                reason='fetch failed after retries',
                detail=(
                    f'batch_id={submission.batch_id}; clear with '
                    f'`ariadne batch clear {submission.batch_id}`'
                ),
            )

        # Success — clear pending row to avoid orphan accumulation.
        if record_pending:
            self._staleness.clear_pending_batch(submission.batch_id)
        return results, None

    def _estimate_batch_cost(self, prompt, response_text) -> float:
        """Estimated USD for re-rolling one prompt at batch pricing.

        Approximate by design: batch results don't surface exact token counts,
        so we use the shared chars→tokens heuristic × the model's rate × the
        batch discount — the same basis as the dry-run estimator."""
        from docgen.pricing import (
            CHARS_PER_TOKEN,
            LLM_PRICING,
            _BATCH_DISCOUNT,
        )
        rates = LLM_PRICING.get(self.config.model)
        if rates is None:
            return 0.0
        in_rate, out_rate = rates
        in_chars = len(prompt.system_prompt or '') + len(prompt.user_prompt or '')
        out_chars = len(response_text or '')
        in_tok = in_chars / CHARS_PER_TOKEN
        out_tok = out_chars / CHARS_PER_TOKEN
        return (in_tok * in_rate + out_tok * out_rate) / 1_000_000 * _BATCH_DISCOUNT

    async def _batch_validation_retry(self, prompts, results_by_cid) -> dict:
        """Re-roll validation-failing prompts as FRESH batches, up to
        ``config.max_batch_validation_retries`` rounds, merging recoveries into
        ``results_by_cid`` in place. Stays in batch (keeps the discount); the
        retry batches are dispatched WITHOUT the resume machinery so a crash
        can't leave a subset batch under the run's config_hash.

        Returns ``{'initial_failures': set[int], 'attempts': dict[int,int],
        'recovered': set[int], 'cost_usd': float}``. ``initial_failures`` is
        computed even when the round budget is 0, so the summary is accurate.
        """
        empty = {'initial_failures': set(), 'attempts': {},
                 'recovered': set(), 'cost_usd': 0.0}
        if (not BATCH_VALIDATION_RETRY or not self.config.validate
                or self._generator is None):
            return empty

        def _is_invalid(idx: int) -> bool:
            text = results_by_cid.get(str(idx))
            if text is None:
                return False  # API-errored row (not a validation failure)
            doc = self._generator.assemble_doc(prompts[idx], text)
            return not self._validator.validate(
                doc.content, doc.doc_type, doc.title).is_valid

        failing = [i for i in range(len(prompts)) if _is_invalid(i)]
        if not failing:
            return empty

        initial = set(failing)
        attempts: dict[int, int] = {}
        recovered: set[int] = set()
        cost = 0.0
        max_rounds = self.config.max_batch_validation_retries

        round_num = 0
        while failing and round_num < max_rounds:
            round_num += 1
            self._emit(
                f'Batch validation retry {round_num}/{max_rounds}: '
                f're-rolling {len(failing)} doc(s)...', 0, 0)
            sub_prompts = [prompts[i] for i in failing]
            # record_pending=False: a retry sub-batch is best-effort, never the
            # resumable main batch.
            sub_results, abort = await self._dispatch_batch(
                sub_prompts, {}, '', record_pending=False)
            if abort is not None:
                self._emit(
                    f'Batch retry stopped ({abort.reason}); keeping '
                    f'recoveries so far.', 0, 0)
                break
            still_failing: list[int] = []
            for pos, idx in enumerate(failing):
                attempts[idx] = attempts.get(idx, 0) + 1
                new_text = sub_results.get(str(pos))
                cost += self._estimate_batch_cost(sub_prompts[pos], new_text)
                if new_text is None:
                    still_failing.append(idx)
                    continue
                doc = self._generator.assemble_doc(prompts[idx], new_text)
                if self._validator.validate(
                        doc.content, doc.doc_type, doc.title).is_valid:
                    results_by_cid[str(idx)] = new_text  # merge the recovery
                    recovered.add(idx)
                else:
                    still_failing.append(idx)
            failing = still_failing

        return {'initial_failures': initial, 'attempts': attempts,
                'recovered': recovered, 'cost_usd': cost}

    # -------------------------------------------------------------------
    # Result assembly (#45.6) — wrap batch responses, validate, store.
    # Validation failures were re-rolled first by ``_batch_validation_retry``
    # (batched retry); whatever still fails here is dropped, and the retry
    # telemetry is carried through on each GenerationResult.
    # -------------------------------------------------------------------

    async def _assemble_and_store(
        self,
        prompts: list[PromptBundle],
        file_to_idxs: dict[Path, list[int]],
        results_by_cid: dict[str, str | None],
        files_to_process: list[Path],
        pre_gen_failed: list[Path],
        retry: dict | None = None,
    ) -> tuple[list[GenerationResult], list[ValidationResult]]:
        """Wrap batch responses into GeneratedDocs, validate, store.

        Mirrors the per-file loop in :meth:`_process_file` but
        consumes batch results from ``results_by_cid`` instead of
        calling the generator. The custom_id used in batch is the
        prompt's index in ``prompts`` as a string, which the caller
        also stored in ``file_to_idxs`` as a list of ints.

        Validation-failing prompts were already re-rolled by
        ``_batch_validation_retry`` (batched retry) before this runs, so
        ``results_by_cid`` holds any recoveries; a doc still failing here is
        dropped. ``retry`` carries the per-index retry telemetry
        (initial-failure / attempts / recovered) threaded onto each result.

        ``docs_generated`` follows streaming's contract:
        ``len(doc_ids)``. Under ``dry_run=True``, ``_store_document``
        is skipped so ``doc_ids`` stays empty and ``docs_generated``
        is 0 — same as streaming.
        """
        if not self._generator or not self._staleness:
            msg = 'Components not initialized'
            raise RuntimeError(msg)

        _retry = retry or {
            'initial_failures': set(), 'attempts': {}, 'recovered': set()}
        all_results: list[GenerationResult] = []
        all_val_results: list[ValidationResult] = []

        # Storing each doc embeds it (one embedding call per doc), which
        # for a large batch is many seconds of work AFTER the download.
        # Emit per-file progress so the bar moves here instead of looking
        # frozen on the prior "downloading results" message.
        store_total = len(files_to_process)
        for store_done, path in enumerate(files_to_process):
            self._emit(
                f'Storing & embedding docs: {store_done}/{store_total} files...',
                store_done, store_total,
            )
            # Pre-gen failures: count one failure per requested doc_type.
            # No prompts to assemble.
            if path in pre_gen_failed:
                all_results.append(GenerationResult(
                    source_path=path,
                    docs_generated=0,
                    docs_failed=len(self.config.doc_types),
                ))
                continue

            idxs = file_to_idxs.get(path, [])
            if not idxs:
                # Defensive: file claimed in files_to_process but no
                # prompts mapped — treat as pre-gen failure.
                all_results.append(GenerationResult(
                    source_path=path,
                    docs_generated=0,
                    docs_failed=len(self.config.doc_types),
                ))
                continue

            val_results: list[ValidationResult] = []
            doc_ids: list[str] = []
            failed = 0

            for i in idxs:
                prompt = prompts[i]
                text = results_by_cid.get(str(i))
                if text is None:
                    # Result missing or explicitly None (per
                    # fetch_batch_results' contract for errored rows).
                    failed += 1
                    continue

                gen_doc = self._generator.assemble_doc(prompt, text)

                if self.config.validate:
                    # Retries already ran in _batch_validation_retry; a doc
                    # still invalid here is dropped.
                    val = self._validator.validate(
                        gen_doc.content, gen_doc.doc_type, gen_doc.title,
                    )
                    val_results.append(val)
                    if not val.is_valid:
                        failed += 1
                        continue

                if not self.config.dry_run:
                    doc = await self._store_document(gen_doc)
                    if doc:
                        doc_ids.append(doc.id)
                    else:
                        failed += 1

            if doc_ids and not self.config.dry_run:
                await self._staleness.record_documentation_async(
                    path, doc_ids, base_path=self.config.source_path,
                )

            all_results.append(GenerationResult(
                source_path=path,
                docs_generated=len(doc_ids),
                docs_failed=failed,
                validation_results=tuple(val_results),
                doc_ids=tuple(doc_ids),
                validation_initial_failures=len(
                    set(idxs) & _retry['initial_failures']),
                validation_retry_attempts=sum(
                    _retry['attempts'].get(i, 0) for i in idxs),
                validation_recovered=len(set(idxs) & _retry['recovered']),
            ))
            all_val_results.extend(val_results)

        return all_results, all_val_results

    # -------------------------------------------------------------------
    # Batch run orchestrator (#45.8) — ties #45.3-7 together.
    #
    # Phases (in order):
    # 1. Resume check: if a pending batch matches the current
    #    config_hash, fetch its results instead of resubmitting.
    # 2. First-run prompt: confirm_callback fires before submit so
    #    the user accepts the up-to-24h SLA.
    # 3. Collect prompts (#45.4) + dispatch (#45.5) + assemble (#45.6).
    # 4. Post-processing (themes + crossrefs) — same as streaming.
    # -------------------------------------------------------------------

    async def _run_batch(
        self,
        files_to_process: list[Path],
        files_skipped: int,
        crossref_progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Batch dispatch fork. Build → dispatch → assemble → post-process."""
        if not self._library or not self._generator or not self._staleness:
            msg = 'Components not initialized'
            raise RuntimeError(msg)

        config_hash = self.config.config_hash()

        # Phase 1: Resume check — adopt an in-flight batch if one
        # matches the current run's config.
        pending = self._staleness.find_pending_batch(config_hash)
        if pending is not None:
            _logger.info(
                'Resuming pending batch %s (matched config_hash)',
                pending.batch_id,
            )
            return await self._resume_from_pending(
                pending, files_skipped, crossref_progress=crossref_progress,
            )

        # Phase 2: First-run prompt for the up-to-24h SLA.
        if self.confirm_callback is not None:
            msg = (
                f'Batch dispatch will submit prompts for '
                f'{len(files_to_process)} files. '
                f'Anthropic batch SLA is up to 24 hours. Continue?'
            )
            accepted = await self.confirm_callback(msg)
            if not accepted:
                _logger.info('User declined batch run')
                return PipelineResult(
                    files_processed=0,
                    files_skipped=files_skipped,
                    docs_created=0,
                    docs_failed=0,
                    aborted=True,
                    abort_reason='user declined batch run',
                    unprocessed_files=tuple(files_to_process),
                )

        # Phase 3: collect → dispatch → assemble.
        prompts, file_to_idxs, pre_gen_failed = await self._collect_prompts(
            files_to_process,
        )

        if not prompts and not pre_gen_failed:
            self._emit('Nothing to generate — all files up-to-date.', 0, 0)
            return PipelineResult(
                files_processed=0,
                files_skipped=files_skipped,
                docs_created=0,
                docs_failed=0,
            )

        results_by_cid: dict[str, str | None] = {}
        if prompts:
            results_by_cid, abort = await self._dispatch_batch(
                prompts, file_to_idxs, config_hash,
            )
            if abort is not None:
                return PipelineResult(
                    files_processed=0,
                    files_skipped=files_skipped,
                    docs_created=0,
                    docs_failed=0,
                    aborted=True,
                    abort_reason=f'{abort.reason}: {abort.detail}',
                    unprocessed_files=tuple(files_to_process),
                )

        # Re-roll validation failures as fresh batches before the final
        # assemble/store (recoveries merged into results_by_cid).
        retry = await self._batch_validation_retry(prompts, results_by_cid)
        results, all_val_results = await self._assemble_and_store(
            prompts, file_to_idxs, results_by_cid,
            files_to_process, pre_gen_failed, retry=retry,
        )

        # Phase 4: post-processing — themes + crossrefs, same as streaming.
        self._emit(
            'Generation complete; running post-processing...',
            len(files_to_process), len(files_to_process),
        )
        await self._post_process(results, crossref_progress=crossref_progress)

        # Compute summary.
        docs_created = sum(r.docs_generated for r in results)
        docs_failed = sum(r.docs_failed for r in results)
        val_initial_failures = sum(
            r.validation_initial_failures for r in results
        )
        val_retry_attempts = sum(r.validation_retry_attempts for r in results)
        val_recovered = sum(r.validation_recovered for r in results)
        val_retry_cost = retry['cost_usd']

        cache_stats = None
        provider = getattr(self._generator, '_provider', None)
        if provider is not None:
            cache_stats = getattr(provider, 'cache_stats', None)

        return PipelineResult(
            files_processed=len(files_to_process) - len(pre_gen_failed),
            files_skipped=files_skipped,
            docs_created=docs_created,
            docs_failed=docs_failed,
            errors=(),
            validation_results=tuple(all_val_results),
            validation_initial_failures=val_initial_failures,
            validation_retry_attempts=val_retry_attempts,
            validation_recovered=val_recovered,
            validation_retry_cost_usd=val_retry_cost,
            cache_stats=cache_stats,
        )

    async def _resume_from_pending(
        self,
        pending,  # type: PendingBatch
        files_skipped: int,
        crossref_progress: ProgressCallback | None = None,
    ) -> PipelineResult:
        """Fetch + assemble an in-flight batch the previous run
        submitted but didn't finish.

        Skips submit + poll entirely — the batch is already done on
        Anthropic's side. On successful fetch + assemble, the
        pending row is cleared. Quota or fetch failures preserve the
        row for another retry.
        """
        import json

        if not self._generator or not self._staleness:
            msg = 'Components not initialized'
            raise RuntimeError(msg)

        strategy = self._batch_strategy()
        if strategy is None:
            return PipelineResult(
                files_processed=0,
                files_skipped=files_skipped,
                docs_created=0,
                docs_failed=0,
                aborted=True,
                abort_reason='no provider available for resume',
            )

        self._emit(
            f'Resuming pending batch {pending.batch_id}...', 0, 0,
        )

        # Reconstruct prompts + file_to_idxs from the persisted JSON.
        prompts_payload = json.loads(pending.prompts_json)
        file_to_idxs_payload = json.loads(pending.file_to_idxs_json)
        prompts = [
            PromptBundle(
                file=Path(p['file']),
                doc_type=p['doc_type'],
                system_prompt=p['system_prompt'],
                user_prompt=p['user_prompt'],
                title=p['title'],
                metadata=p['metadata'],
            )
            for p in prompts_payload
        ]
        file_to_idxs = {
            Path(k): v for k, v in file_to_idxs_payload.items()
        }
        files_resumed = list(file_to_idxs.keys())

        try:
            results_by_cid = await self._fetch_with_retry(
                strategy, pending.batch_id,
            )
        except QuotaExhaustedError as e:
            return PipelineResult(
                files_processed=0,
                files_skipped=files_skipped,
                docs_created=0,
                docs_failed=0,
                aborted=True,
                abort_reason=f'quota during fetch on resume: {e}',
                unprocessed_files=tuple(files_resumed),
            )

        if results_by_cid is None:
            return PipelineResult(
                files_processed=0,
                files_skipped=files_skipped,
                docs_created=0,
                docs_failed=0,
                aborted=True,
                abort_reason=(
                    f'fetch failed for resumed batch {pending.batch_id}'
                ),
                unprocessed_files=tuple(files_resumed),
            )

        # Successful fetch — clear the pending row.
        self._staleness.clear_pending_batch(pending.batch_id)

        retry = await self._batch_validation_retry(prompts, results_by_cid)
        results, all_val_results = await self._assemble_and_store(
            prompts, file_to_idxs, results_by_cid,
            files_to_process=files_resumed,
            pre_gen_failed=[],
            retry=retry,
        )

        # Post-processing (same as fresh batch run).
        await self._post_process(results, crossref_progress=crossref_progress)

        docs_created = sum(r.docs_generated for r in results)
        docs_failed = sum(r.docs_failed for r in results)

        # cache_stats lives on the provider (the strategy borrows it and
        # records into it during fetch), so read it off the provider — same as
        # the streaming + fresh-batch paths.
        provider = getattr(self._generator, '_provider', None)
        cache_stats = getattr(provider, 'cache_stats', None) if provider else None

        return PipelineResult(
            files_processed=len(files_resumed),
            files_skipped=files_skipped,
            docs_created=docs_created,
            docs_failed=docs_failed,
            validation_results=tuple(all_val_results),
            validation_initial_failures=sum(
                r.validation_initial_failures for r in results),
            validation_retry_attempts=sum(
                r.validation_retry_attempts for r in results),
            validation_recovered=sum(
                r.validation_recovered for r in results),
            validation_retry_cost_usd=retry['cost_usd'],
            cache_stats=cache_stats,
        )

    async def _validate_with_retry(
        self,
        gen_doc: GeneratedDoc,
        regenerate_fn,
    ) -> tuple[GeneratedDoc | None, ValidationResult, int]:
        """Validate ``gen_doc``; on failure, re-roll up to MAX_VALIDATION_RETRIES.

        ``regenerate_fn`` is awaited with no args to produce a fresh
        attempt — typically a single-doc-type LLM call. Naive retry: the
        prompt is identical; LLM sampling variance handles the rest.

        Returns ``(final_doc | None, last_result, retries_used)``. If the
        first attempt passes, retries_used is 0.
        """
        result = self._validator.validate(
            gen_doc.content, gen_doc.doc_type, gen_doc.title,
        )
        if result.is_valid:
            return gen_doc, result, 0

        retries_used = 0
        for _ in range(MAX_VALIDATION_RETRIES):
            retries_used += 1
            new_doc = await regenerate_fn()
            if new_doc is None:
                break
            result = self._validator.validate(
                new_doc.content, new_doc.doc_type, new_doc.title,
            )
            if result.is_valid:
                return new_doc, result, retries_used
            gen_doc = new_doc

        return None, result, retries_used

    async def _post_process(
        self,
        results: list[GenerationResult],
        crossref_progress: ProgressCallback | None = None,
    ) -> None:
        """Run themes (always) then crossrefs (gated by config flag).

        Themes are useful on their own and shouldn't gate on crossrefs
        finishing. Crossrefs is graph-based (O(N×K)) and reuses the
        progress callback to surface its per-doc work.

        Both phases announce their stage through ``crossref_progress``
        so the CLI bar fills the otherwise-silent window between
        per-file generation completing and crossrefs starting.
        """
        if self.config.dry_run:
            return

        # Docs are generated by the time post-processing runs. Record the
        # completion marker for a pinned spool corpus NOW — BEFORE the long,
        # interruptible theme summarize — so an abort during themes still lets
        # the next run skip discovery/generation for this unchanged corpus.
        if not any(r.docs_failed for r in results):
            self._write_generation_marker()

        # Themes first. Announce phase entry/exit on the bar AND surface
        # per-theme progress during the LLM summarize stage so the user
        # sees the bar tick from 0/N → N/N instead of staring at 0/0.
        from time import perf_counter
        _t0 = perf_counter()
        from docgen.themes import refresh_themes
        if crossref_progress is not None:
            crossref_progress('Themes: refreshing...', 0, 0)

        def _theme_progress(
            completed: int, total: int, cluster_id: str | None,
        ) -> None:
            if crossref_progress is None:
                return
            short = (cluster_id or '')[:12]
            desc = (
                f'Themes: summarizing [dim]{short}[/dim]'
                if short else 'Themes: summarize done'
            )
            crossref_progress(desc, completed, total)

        try:
            # Inherit the user's --concurrency for theme summarization too.
            # Without this, themes run at generate_themes' hardcoded default
            # of 4 even when the user passed -c 6 (or higher) for per-file
            # generation. Network-bound LLM calls scale the same way for
            # both phases — there's no good reason to split them.
            themes_summary = await refresh_themes(
                self._library, self._writer,
                enabled=self.config.themes_enabled,
                # Honor the run's batch choice for themes too (they run here,
                # in generate's post-process) — else --batch still summarizes
                # themes live at full price.
                batch_strategy=self._themes_batch_strategy(),
                summarize_kwargs={
                    'on_progress': _theme_progress,
                    'concurrency': self.config.concurrency,
                    # Theme summarization is generation — use the RUN's model
                    # (e.g. a spool build's spools_model), not the global
                    # default it would otherwise fall back to.
                    'model': self.config.model,
                },
            )
            if crossref_progress is not None and themes_summary:
                path = themes_summary.get('path', '?')
                summarized = themes_summary.get('summarized', 0)
                crossref_progress(
                    f'Themes: {path} ({summarized} summarized)', 0, 0,
                )
        except Exception as e:
            _logger.warning('refresh_themes failed: %s', e)
            if crossref_progress is not None:
                crossref_progress('Themes: failed (see log)', 0, 0)

        _logger.info('post-process timing: themes refresh %.2fs', perf_counter() - _t0)

        # Crossrefs second (and optional). The function itself fires
        # per-doc progress; we just announce the load stage here.
        if self.config.inject_crossrefs and results:
            if crossref_progress is not None:
                crossref_progress('Crossref: loading library...', 0, 0)
            _xref_t0 = perf_counter()
            await self._inject_crossrefs_scoped(
                progress_callback=crossref_progress,
            )
            _logger.info('post-process timing: crossref injection %.2fs', perf_counter() - _xref_t0)

    async def _regenerate_doc(
        self, path: Path, doc_type: DocType,
    ) -> GeneratedDoc | None:
        """Re-generate a single doc-type for ``path``.

        Used by validation retry. Routes through the same catalog/legacy
        branching as ``_process_file`` but only requests one doc_type so
        we don't pay for re-generating types that already passed.
        """
        if self.config.catalog_only_generator:
            from docgen.catalog_enrich import enrich_file
            bundle = enrich_file(
                path,
                source_root=self.config.source_path,
                source_config=self.config.source_config,
                cross_source_graph=self._cross_source_graph,
            )
            if bundle is None:
                return None
            docs = await self._generator.generate_from_elements(
                bundle, doc_types=(doc_type,),
            )
        else:
            try:
                metadata = self._analyzer.analyze_file(path)
            except SyntaxError:
                return None
            docs = await self._generator.generate_for_module(
                metadata, doc_types=(doc_type,),
            )
        return docs[0] if docs else None

    async def _store_document(self, gen_doc: GeneratedDoc) -> Document | None:
        """Store a generated document in the library.

        Args:
            gen_doc: The generated document.

        Returns:
            The created Document, or None if storage failed.
        """
        if not self._writer:
            return None

        source_name = self.config.source_name
        gen_doc.metadata['provenance'] = _provenance_for_doc(gen_doc, self._low_confidence_languages())
        # Deterministic doc_id keyed on (source, content_type, primary_key).
        # Re-running generate on the same file therefore updates the same row
        # rather than appending a duplicate (the prior UUID4 scheme produced
        # title-collision dupes when two files shared a module-name leaf).
        doc_id = self._compute_deterministic_doc_id(gen_doc, source_name)

        try:
            if gen_doc.doc_type == 'explanation':
                return await self._writer.add_explanation(
                    title=gen_doc.title,
                    content=gen_doc.content,
                    source_files=list(gen_doc.source_files),
                    metadata=gen_doc.metadata,
                    source_name=source_name,
                    doc_id=doc_id,
                )
            elif gen_doc.doc_type == 'architecture':
                return await self._writer.add_architecture(
                    title=gen_doc.title,
                    content=gen_doc.content,
                    source_files=list(gen_doc.source_files),
                    source_name=source_name,
                    metadata=gen_doc.metadata,
                    doc_id=doc_id,
                )
            elif gen_doc.doc_type == 'qa':
                return await self._writer.add_document(
                    content_type='qa',
                    title=gen_doc.title,
                    content=gen_doc.content,
                    source_files=list(gen_doc.source_files),
                    metadata=gen_doc.metadata,
                    source_name=source_name,
                    doc_id=doc_id,
                )
            elif gen_doc.doc_type == 'diagram':
                description, dot_code = split_description_and_dot(gen_doc.content)

                return await self._writer.add_diagram(
                    title=gen_doc.title,
                    description=description,
                    dot_code=dot_code,
                    source_files=list(gen_doc.source_files),
                    source_name=source_name,
                    metadata=gen_doc.metadata,
                    doc_id=doc_id,
                )
            else:
                return await self._writer.add_document(
                    content_type=gen_doc.doc_type,
                    title=gen_doc.title,
                    content=gen_doc.content,
                    source_files=list(gen_doc.source_files),
                    metadata=gen_doc.metadata,
                    source_name=source_name,
                    doc_id=doc_id,
                )
        except Exception as e:
            _logger.error('Failed to store document %s: %s', gen_doc.title, e)
            return None

    def _compute_deterministic_doc_id(
        self, gen_doc: GeneratedDoc, source_name: str | None,
    ) -> str:
        """Compute a stable doc_id for a GeneratedDoc.

        - Per-file docs: keyed on path (relative to source_path when possible
          so absolute prefix changes don't break determinism)
        - Group / package-level docs: keyed on ``"group:<package_name>"``
        - Topic docs: keyed on ``"topic:<topic_title>"``
        - Fallback: keyed on title (preserves the legacy title-only behavior
          for any doc that doesn't fit the above)
        """
        from schema import doc_id_for

        meta = gen_doc.metadata or {}
        if meta.get('group'):
            primary_key = f"group:{meta.get('package_name', gen_doc.title)}"
        elif meta.get('topic'):
            primary_key = f"topic:{meta.get('topic_title', gen_doc.title)}"
        elif gen_doc.source_files:
            primary_file = gen_doc.source_files[0]
            try:
                primary_key = str(
                    Path(primary_file).resolve().relative_to(
                        self.config.source_path,
                    )
                )
            except (ValueError, OSError):
                primary_key = primary_file
        else:
            primary_key = gen_doc.title

        return doc_id_for(
            source_name or 'unknown',
            gen_doc.doc_type,
            primary_key,
        )

    def _store_file_map(self, metadata: 'ModuleMetadata') -> None:
        """Generate and store a compact file map from AST metadata.

        File maps enable agents to query method names, line ranges, and
        dependencies without reading the full source file.
        """
        from schema import generate_deterministic_id

        filename = metadata.path.name
        title = f'File map: {filename} ({metadata.line_count} lines)'
        map_id = generate_deterministic_id('finding', title)

        # Skip if an identical map already exists (same source hash)
        existing = self._scoped_lib().get_document(map_id)
        if existing and existing.metadata.get('source_hash') == metadata.source_hash:
            return

        lines: list[str] = [f'# {title}\n']

        if metadata.classes:
            lines.append('## Classes')
            for cls in metadata.classes:
                methods = [m.name for m in cls.methods]
                bases = f"({', '.join(cls.bases)})" if cls.bases else ''
                lines.append(f"- **{cls.name}**{bases} (line {cls.lineno}): {', '.join(methods)}")
            lines.append('')

        top_funcs = [f for f in metadata.functions if not f.is_method]
        if top_funcs:
            lines.append('## Functions')
            for func in top_funcs:
                lines.append(f'- {func.name} (line {func.lineno})')
            lines.append('')

        if metadata.imports:
            modules = sorted({imp.module for imp in metadata.imports})
            lines.append(f"## Imports: {', '.join(modules)}")

        content = '\n'.join(lines)
        source_file = str(metadata.path)
        meta = {'source_hash': metadata.source_hash, 'auto_generated': True}

        try:
            if existing:
                self._library.update_document(map_id, content=content, metadata=meta)
            else:
                self._library.add_document(
                    content_type='finding',
                    title=title,
                    content=content,
                    source_files=[source_file],
                    metadata=meta,
                    doc_id=map_id,
                    source_name=self.config.source_name,
                )
            _logger.debug('Stored file map for %s', filename)
        except Exception:
            _logger.debug('Failed to store file map for %s', filename, exc_info=True)

    def _check_dependency_docs(self) -> None:
        """Check that dependency documentation exists and warn if missing.

        This includes both explicit dependencies and parent sources.
        """
        if not self._library:
            return

        from config import get_config

        cfg = get_config()

        # Get effective dependencies (includes parent if set)
        deps_to_check = list(self.config.dependencies)
        if self.config.source_name:
            source_config = cfg.get_source_config(self.config.source_name)
            if source_config and source_config.parent:
                if source_config.parent not in deps_to_check:
                    deps_to_check.insert(0, source_config.parent)

        for dep_name in deps_to_check:
            dep_docs_path = cfg.resolve_docs_path(dep_name)
            if not dep_docs_path.exists():
                _logger.warning(
                    "Dependency '%s' documentation not found at %s. "
                    "Run 'ariadne generate --source %s' first for better cross-references.",
                    dep_name,
                    dep_docs_path,
                    dep_name,
                )
            else:
                # Check if there are any markdown files
                md_files = list(dep_docs_path.glob('**/*.md'))
                if not md_files:
                    _logger.warning(
                        "Dependency '%s' documentation at %s has no markdown files. "
                        "Run 'ariadne generate --source %s' first.",
                        dep_name,
                        dep_docs_path,
                        dep_name,
                    )
                else:
                    _logger.info(
                        "Found %d docs for dependency '%s'",
                        len(md_files),
                        dep_name,
                    )

    async def _inject_crossrefs(self) -> None:
        """Inject cross-references into all documents."""
        if not self._library:
            return

        docs = self._library.list_documents()
        if not docs:
            return

        detector = CrossRefDetector(list(docs))
        doc_titles = {doc.id: doc.title for doc in docs}

        for doc in docs:
            refs = detector.get_all_references(doc)
            if refs:
                new_content = inject_related_section(doc.content, refs, doc_titles)
                if new_content != doc.content:
                    self._library.update_document(doc.id, content=new_content)

    async def _inject_crossrefs_scoped(
        self,
        progress_callback: ProgressCallback | None = None,
    ) -> None:
        """Inject cross-references using the precomputed doc_graph.

        Replaces the brute-force regex pass (``detect_mention_references``)
        with edge lookups via ``library.get_related`` — undirected BFS
        over imports / documents / topic_member / semantic_neighbor
        edges. Complexity drops from O(N²) substring scanning to
        O(N×K) where K is the BFS frontier (~10-50 per doc).

        ``progress_callback`` fires per processed doc as
        ``(message, current, total)`` so the CLI can render a bar.

        Edges relied on:
          - ``imports`` — written by ``build_graph`` from AST parsing
          - ``documents`` / ``topic_member`` — file-to-doc associations
          - ``semantic_neighbor`` — written by ``build_semantic_edges``
            during ``refresh_themes``; reorder ensures themes runs first.
        """
        if not self._library:
            return

        # Scope: current source + depends_on
        allowed_sources = set()
        if self.config.source_name:
            allowed_sources.add(self.config.source_name)
        for dep in self.config.dependencies:
            allowed_sources.add(dep)

        all_docs = self._library.list_documents()
        if not all_docs:
            return

        if allowed_sources:
            scoped_docs = [
                doc for doc in all_docs
                if any(
                    src in (sf or '')
                    for sf in (doc.source_files or ())
                    for src in allowed_sources
                )
                or not doc.source_files
            ]
        else:
            scoped_docs = all_docs

        _logger.info(
            'Cross-refs (graph-based): %d docs scoped (from %d total)',
            len(scoped_docs), len(all_docs),
        )
        if not scoped_docs:
            return

        scoped_ids = {doc.id for doc in scoped_docs}
        doc_titles = {doc.id: doc.title for doc in scoped_docs}
        total = len(scoped_docs)
        updated = 0
        from time import perf_counter
        _read = _write = 0.0

        # One bulk graph load + in-memory walks for ALL docs, instead of a
        # per-doc query storm (library.get_related_batch — equivalence-tested).
        _r = perf_counter()
        related_by_id = self._scoped_lib().get_related_batch(
            [doc.id for doc in scoped_docs], max_hops=2, limit=10,
        )
        _read += perf_counter() - _r

        for i, doc in enumerate(scoped_docs):
            if progress_callback:
                progress_callback(
                    f'Crossref {doc.title[:40]}', i, total,
                )

            related = related_by_id.get(doc.id) or []
            if not related:
                continue

            # Filter neighbors to in-scope docs only.
            refs = [
                CrossReference(
                    source_id=doc.id,
                    target_id=r['id'],
                    ref_type='graph',
                    context=f"Related (graph distance {r['distance']:.2f})",
                )
                for r in related
                if r['id'] in scoped_ids
            ]
            if not refs:
                continue

            try:
                new_content = inject_related_section(
                    doc.content, refs, doc_titles,
                )
            except Exception as e:
                # One bad doc (e.g., title with regex metachars, malformed
                # content) shouldn't kill crossref injection for every
                # remaining doc. Log and skip; user can re-run safely.
                _logger.warning(
                    'inject_related_section failed for %s (%s): %s',
                    doc.id, doc.title, e,
                )
                continue
            if new_content != doc.content:
                _w = perf_counter()
                self._library.update_document(doc.id, content=new_content)
                _write += perf_counter() - _w
                updated += 1

        _logger.info(
            'Cross-refs: %d/%d docs updated (graph load %.2fs, writes %.2fs)',
            updated, total, _read, _write,
        )
        if progress_callback:
            progress_callback('Crossrefs complete', total, total)

    async def check_staleness(self) -> dict:
        """Check which files need documentation updates.

        Returns:
            Dict with counts and lists of stale/undocumented files.
        """
        if not self._staleness:
            msg = 'Orchestrator must be used as async context manager'
            raise RuntimeError(msg)

        discover = (
            find_catalog_files
            if self.config.catalog_only_generator
            else find_python_files
        )
        all_files = discover(self.config.source_path)
        stale = self._staleness.get_stale_files(all_files, base_path=self.config.source_path, is_exempt=self._staleness_exempt())
        undocumented = self._staleness.get_undocumented_files(all_files, base_path=self.config.source_path)

        return {
            'total_files': len(all_files),
            'stale_files': len(stale),
            'undocumented_files': len(undocumented),
            'up_to_date': len(all_files) - len(stale),
            'stale_paths': [str(p) for p in stale],
            'undocumented_paths': [str(p) for p in undocumented],
        }

    async def generate_for_module(
        self,
        module_path: Path,
        doc_types: tuple[DocType, ...] | None = None,
    ) -> list[GeneratedDoc]:
        """Generate documentation for a single module.

        Args:
            module_path: Path to the Python module.
            doc_types: Types of documentation to generate.

        Returns:
            List of generated documents.
        """
        if not self._generator:
            msg = 'Orchestrator must be used as async context manager'
            raise RuntimeError(msg)

        return await self._generator.generate_for_file(module_path, doc_types)

    async def generate_for_package(
        self,
        package_path: Path,
        doc_types: tuple[DocType, ...] | None = None,
    ) -> list[GeneratedDoc]:
        """Generate documentation for a package (directory).

        Args:
            package_path: Path to the package directory.
            doc_types: Types of documentation to generate.

        Returns:
            List of generated documents.
        """
        if not self._generator:
            msg = 'Orchestrator must be used as async context manager'
            raise RuntimeError(msg)

        group = self._analyzer.analyze_directory(package_path)
        return await self._generator.generate_for_group(
            group, doc_types, concurrency=self.config.concurrency
        )

    async def run_reverse_augment(
        self,
        related_sources: dict[str, Path],
    ) -> int:
        """Run the cross-source reverse-augment phase for this
        orchestrator's source.

        Builds a CrossSourceGraph from the target source's manifest
        plus any related sources' manifests, regenerates docs for
        files with cross-source consumers (with consumer context
        injected into the prompt), and persists the regenerated docs
        via this orchestrator's existing ``_store_document``.

        Args:
            related_sources: Other sources whose ``.scip`` may contain
                references INTO this source's symbols. Maps source
                name → absolute root path. Sources without a manifest
                yet are silently skipped.

        Returns:
            Number of regenerated documents persisted.

        Should be called AFTER ``run()`` completes — the cross-source
        graph is meaningful only when this source's docs have already
        been generated through the normal pipeline.
        """
        if not self._generator:
            msg = 'Orchestrator must be used as async context manager'
            raise RuntimeError(msg)

        from docgen.reverse_augment import run_reverse_augment_for_source

        # Honor the same doc-gen excludes the main run applied: reverse-augment
        # must not regenerate a file the source excludes (e.g. test dirs) just
        # because another source references it. Build the source's included
        # doc-gen set the same way run() discovers files, and pass it as the
        # filter so the expensive cross-source pass respects exclude_dirs/exclude.
        from config import DEFAULT_EXCLUDE_FILE_PATTERNS
        from docgen.staleness import find_catalog_files, find_python_files

        _discover = (
            find_catalog_files
            if self.config.catalog_only_generator
            else find_python_files
        )
        _root = self.config.source_path
        allowed_files: set[str] = set()
        for _f in _discover(
            _root,
            exclude_patterns=(
                DEFAULT_EXCLUDE_FILE_PATTERNS + self.config.exclude_patterns
            ),
            exclude_dir_names=self.config.exclude_dir_names,
        ):
            try:
                allowed_files.add(_f.relative_to(_root).as_posix())
            except ValueError:
                allowed_files.add(_f.as_posix())

        # Persist each file's augmented docs as they're produced, so the phase
        # can mark a file fresh ONLY after it's durably stored (skip-check →
        # generate → persist → mark, per file). This makes the pass both
        # reuse-aware on re-run (the marker_store skip) and resumable after an
        # abort — completed files skip, the rest retry — instead of the old
        # "regenerate everything, then store everything" which re-billed the
        # whole consumed set on every run.
        persisted = 0

        async def _persist(_source_name: str, _file: str, docs: list) -> None:
            nonlocal persisted
            for doc in docs:
                stored = await self._store_document(doc)
                if stored is not None:
                    persisted += 1

        await run_reverse_augment_for_source(
            target_source=self.config.source_name,
            target_source_root=self.config.source_path,
            related_sources=related_sources,
            analyzer=self._analyzer.analyze_file,
            generator=self._generator,
            allowed_files=allowed_files,
            marker_store=self._staleness,
            persist=_persist,
        )
        return persisted
    def _staleness_exempt(self):
        """Predicate marking a source-relative path staleness-exempt, built
    from ``config.ignore_staleness``; ``None`` for the default (no
    exemption -- the legacy staleness path).
    """
        ignore = self.config.ignore_staleness
        if not ignore:
            return None
        from config import ignore_staleness_matches
        return lambda rel: ignore_staleness_matches(ignore, rel)


async def run_pipeline(
    source_path: Path,
    db_path: Path | None = None,
    model: str = 'gpt-5.2',
    api_key: str | None = None,
    doc_types: tuple[DocType, ...] = ('explanation', 'architecture', 'catalog', 'qa', 'gotcha', 'diagram'),
    force: bool = False,
    dry_run: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> PipelineResult:
    """Convenience function to run the documentation generation pipeline.

    Args:
        source_path: Path to source code directory.
        db_path: Path to Ariadne library database.
        model: LLM model to use.
        api_key: API key for LLM provider.
        doc_types: Types of documentation to generate.
        force: Force regeneration even if not stale.
        dry_run: Don't write to database.
        progress_callback: Optional progress callback.

    Returns:
        PipelineResult with summary statistics.
    """
    config = OrchestratorConfig(
        source_path=source_path,
        db_path=db_path or Path('ariadne.db'),
        model=model,
        api_key=api_key,
        doc_types=doc_types,
        force_regenerate=force,
        dry_run=dry_run,
    )

    async with DocGenOrchestrator(config) as orchestrator:
        return await orchestrator.run(progress_callback)
