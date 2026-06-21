"""Admin operations — listing, sync, generation, and feedback."""
from __future__ import annotations

import logging
from datetime import datetime

from ariadne_mcp.models import (
    AdminActionResponse,
    AffectedDocument,
    BranchStatusResponse,
    DirCostModel,
    DiscoverResponse,
    DocTypeCostModel,
    DocumentSummary,
    EstimateResponse,
    ExclusionSaving,
    FeedbackResponse,
    GitInfo,
    IndexerPlan,
    LanguageCount,
    ListResponse,
    ModelPrice,
    SourceAddResponse,
    SourceEntry,
    SourceListResponse,
    SyncStatusResponse,
)

_logger = logging.getLogger(__name__)


class AdminMixin:
    """Admin operations: listing, branch management, generation, feedback.

    Expects the composed class to provide:
    - self.library: Library
    - self.config: Config
    - self._resolve_source(), self._source_path()
    - self.get_branch()
    - self.clear_cache()
    """

    # ------------------------------------------------------------------
    # Source configuration (onboarding "Connect" / "Scope")
    # ------------------------------------------------------------------

    def source_add(
        self,
        name: str,
        *,
        path: str | None = None,
        depends_on: list[str] | None = None,
        parent: str | None = None,
        branches: list[str] | None = None,
        ref: str | None = None,
        exclude: list[str] | None = None,
        exclude_dirs: list[str] | None = None,
        exempt_dirs: list[str] | None = None,
        ignore_staleness: bool | None = None,
        set_default: bool | None = None,
    doc_types_by_language: dict | None = None) -> SourceAddResponse:
        """Create or update a source in ariadne.yaml.

        Mirrors ``ariadne source add``: bootstraps the config file when a
        project has none, applies the "first source becomes default" rule,
        and updates only the provided (non-None) fields on re-runs. Returns
        the persisted source config plus detected git metadata.
        """
        from pathlib import Path

        from cli.integration import _resolve_writable_config
        from git_ops import get_repo_info

        cfg = _resolve_writable_config()
        existed = cfg.get_source_config(name) is not None
        if not existed and not path:
            raise ValueError(f'A path is required to create source {name!r}.')
        # Validate the path up front so the error surfaces at "Connect",
        # not three steps later at discover.
        if path is not None and not Path(path).expanduser().is_dir():
            raise ValueError(
                f'Path does not exist or is not a directory: {path}')

        if not cfg.set_source_config(
            name,
            path=path,
            depends_on=depends_on,
            parent=parent,
            branches=branches,
            ref=ref,
            exclude=exclude,
            exclude_dirs=exclude_dirs,
            exempt_dirs=exempt_dirs,
            ignore_staleness=ignore_staleness,
        doc_types_by_language=doc_types_by_language):
            raise RuntimeError(f'Failed to write source {name!r} to config.')

        make_default = (
            set_default if set_default is not None
            else cfg.default_source is None
        )
        if make_default:
            cfg.set_default_source(name)

        # The service's cached Config (if any) predates this write — drop it
        # so later calls re-read the rebound global singleton.
        self._config = None

        sc = cfg.get_source_config(name)
        resolved = cfg.get_source_path(name)
        git = None
        if resolved is not None and Path(resolved).is_dir():
            git = GitInfo(**get_repo_info(Path(resolved)))

        ignore = sc.ignore_staleness
        return SourceAddResponse(
            source=name,
            path=sc.path,
            created=not existed,
            is_default=cfg.default_source == name,
            depends_on=list(sc.depends_on),
            parent=sc.parent,
            branches=list(sc.branches),
            ref=sc.ref,
            exclude=list(sc.exclude),
            exclude_dirs=list(sc.exclude_dirs),
            exempt_dirs=list(sc.exempt_dirs),
            ignore_staleness=list(ignore) if isinstance(ignore, tuple) else ignore,
            git=git,
        doc_types_by_language={k: list(v) for k, v in sc.doc_types_by_language.items()})

    def list_sources(self) -> SourceListResponse:
        """List the sources configured in ariadne.yaml — powers the
        onboarding dependency picker (depends_on)."""
        cfg = self.config
        entries: list[SourceEntry] = []
        for name in cfg.sources:
            sc = cfg.get_source_config(name)
            if sc is None:
                continue
            entries.append(SourceEntry(
                name=name,
                path=sc.path,
                is_default=(name == cfg.default_source),
                depends_on=list(sc.depends_on),
            ))
        return SourceListResponse(
            sources=entries, default_source=cfg.default_source)

    def estimate(
        self,
        source: str | None = None,
        *,
        model: str | None = None,
        doc_types: list[str] | None = None,
    ) -> EstimateResponse:
        """Cost preview for documenting a source (onboarding "Preview").

        Delegates to :func:`cli.generate_cost.build_estimate` — the same
        no-LLM cost model ``dry-run`` uses — and maps it to the wire schema:
        totals (live + batched), per-directory tree, per-doc-type split,
        language histogram, and the model price list.
        """
        from cli.generate_cost import build_estimate, exclusion_savings
        from docgen.pricing import LLM_PRICING, _supported_doc_types_for
        from docgen.prompts import LANGUAGE_DOC_TYPES

        src = self._resolve_source(source)
        if src is None:
            raise ValueError(
                'No source specified and no default_source configured.')
        cfg = self.config
        model = model or cfg.model
        dts = tuple(doc_types) if doc_types else None
        est = build_estimate(cfg, src, model=model, doc_types=dts, db_path=cfg.db_path)
        savings = exclusion_savings(
            cfg, src, model=model, doc_types=dts, db_path=cfg.db_path)

        total = est.total
        rates = total.rates or (0.0, 0.0)
        file_count = est.file_count or 1
        # Per-language applicable doc types for the matrix: start from the
        # static map, then overlay the *effective* set for each language
        # actually present — so unmapped-but-supported languages like
        # ``dockerfile`` (explanation-only, via the default the cost model
        # uses) render correctly instead of as all-"not applicable".
        language_doc_types = {
            lang: list(dts) for lang, dts in LANGUAGE_DOC_TYPES.items()
        }
        for lang, _count in est.languages:
            language_doc_types[lang] = list(_supported_doc_types_for(lang))
        return EstimateResponse(
            source=src,
            model=model,
            input_per_million=rates[0],
            output_per_million=rates[1],
            file_count=est.file_count,
            total_calls=total.total_calls,
            input_tokens=total.input_tokens,
            output_tokens=total.output_tokens,
            embedding_tokens=total.embedding_tokens,
            total_cost_usd=total.total_cost_usd,
            total_cost_batched_usd=est.total_batched.total_cost_usd,
            cost_lower_bound=total.cost_lower_bound,
            cost_upper_bound=total.cost_upper_bound,
            embedding_cost_usd=total.embedding_cost_usd,
            languages=[
                LanguageCount(
                    language=lang, files=n,
                    percent=round(100.0 * n / file_count, 1))
                for lang, n in est.languages
            ],
            by_doc_type=[
                DocTypeCostModel(
                    doc_type=dt, count=ce.total_calls,
                    cost_usd=ce.total_cost_usd,
                    cost_batched_usd=est.by_doc_type_batched[dt].total_cost_usd)
                for dt, ce in est.by_doc_type
            ],
            by_directory=[
                DirCostModel(
                    rel_path=nc.rel_path, docs=nc.docs,
                    total_usd=nc.total, ingestion_usd=nc.ingestion_cost)
                for nc in est.by_directory.values()
            ],
            available_models=[
                ModelPrice(
                    model=m, input_per_million=ipm, output_per_million=opm)
                for m, (ipm, opm) in LLM_PRICING.items()
            ],
            language_doc_types=language_doc_types,
            exclusion_savings=[
                ExclusionSaving(
                    pattern=s.pattern, kind=s.kind, files=s.files,
                    saved_usd=s.saved_usd, saved_batched_usd=s.saved_batched_usd)
                for s in savings
            ],
        )

    def discover(self, source: str | None = None) -> DiscoverResponse:
        """Detect languages + the SCIP index plan for a source and write
        its manifest (onboarding "Discover").

        Delegates the detection/persistence to
        :func:`cli.index.run_discover`, then layers on the language
        histogram + file/dir counts the UI shows.
        """
        from cli.dry_run import _discover_files_for_estimate
        from cli.generate_cost import language_histogram
        from cli.index import run_discover

        src = self._resolve_source(source)
        if src is None:
            raise ValueError(
                'No source specified and no default_source configured.')
        cfg = self.config
        result = run_discover(cfg, src)
        source_path = result['source_path']

        files = _discover_files_for_estimate(cfg, src, source_path)
        file_count = len(files)
        dirs = {path.parent for path, _size in files}
        hist = language_histogram(files)

        # run_discover may have written index_kinds — drop the cached config
        # so later calls re-read it.
        self._config = None
        return DiscoverResponse(
            source=src,
            file_count=file_count,
            dir_count=len(dirs),
            languages=[
                LanguageCount(
                    language=lang, files=n,
                    percent=round(100.0 * n / (file_count or 1), 1))
                for lang, n in hist
            ],
            indexers=[IndexerPlan(**ix) for ix in result['indexers']],
            index_kinds=result['index_kinds'],
            manifest_written=True,
        )

    # ------------------------------------------------------------------
    # List
    # ------------------------------------------------------------------

    def list_all(
        self,
        include_expired: bool = False,
        source: str | None = None,
    ) -> ListResponse:
        """List documents within the resolved source's closure.

        ``source`` selects which source's closure scopes the result;
        if omitted, it falls back to ``config.default_source``. The
        returned list never contains rows from outside that closure.
        """
        scoped = self._resolve_scope(source)
        docs = scoped.list_documents_lite()
        now = datetime.now()
        summaries: list[DocumentSummary] = []

        for doc in docs:
            expires_at_str = doc.metadata.get('expires_at')
            is_expired = False

            if expires_at_str:
                try:
                    is_expired = datetime.fromisoformat(expires_at_str) < now
                except ValueError:
                    pass

            if is_expired and not include_expired:
                continue

            summaries.append(DocumentSummary(
                id=doc.id,
                title=doc.title,
                content_type=doc.content_type,
                status=doc.metadata.get('status', 'stable'),
                branches=doc.metadata.get('branches', []),
                expires_at=expires_at_str,
                is_expired=is_expired,
                source_files=doc.source_files,
            ))

        event_id = self.library.log_usage('ariadne_list_all', None, len(summaries))

        return ListResponse(
            total=len(summaries),
            documents=summaries,
            event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Branch status
    # ------------------------------------------------------------------

    def branch_status(self, source: str | None = None) -> BranchStatusResponse:
        """Show documents affected by current branch changes."""
        from git_ops import get_changed_files_vs_main, get_current_branch

        source_name = self._resolve_source(source)
        if not source_name:
            event_id = self.library.log_usage('ariadne_branch_status', source, 0)
            return BranchStatusResponse(
                branch='unknown',
                message='No source specified and no default_source in config',
                event_id=event_id,
            )

        source_path = self._source_path(source_name)
        if source_path is None or not source_path.exists():
            event_id = self.library.log_usage('ariadne_branch_status', source, 0)
            return BranchStatusResponse(
                branch='unknown',
                message=f'Source path not found: {source_path}',
                event_id=event_id,
            )

        branch = get_current_branch(source_path)
        main_branch = self.config.main_branch

        if branch in (main_branch, 'master'):
            event_id = self.library.log_usage('ariadne_branch_status', source, 0)
            return BranchStatusResponse(
                branch=branch or 'unknown',
                message=f'On {branch} branch, no branch-specific docs needed',
                event_id=event_id,
            )

        changed_files = get_changed_files_vs_main(source_path, main_branch)
        scoped = self._resolve_scope(source)
        affected_docs = (
            scoped.find_documents_by_source_files(changed_files)
            if changed_files
            else []
        )

        event_id = self.library.log_usage(
            'ariadne_branch_status', source, len(affected_docs),
        )

        return BranchStatusResponse(
            branch=branch or 'unknown',
            comparing_against=main_branch,
            changed_files=changed_files,
            affected_documents=[
                AffectedDocument(id=d.id, title=d.title, content_type=d.content_type)
                for d in affected_docs
            ],
            event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Sync status
    # ------------------------------------------------------------------

    def sync_status(self, source: str | None = None) -> SyncStatusResponse:
        """Check when documentation was last synced."""
        from git_ops import get_commit_message

        source_name = self._resolve_source(source)
        if not source_name:
            event_id = self.library.log_usage('ariadne_sync_status', source, 0)
            return SyncStatusResponse(
                source='unknown',
                status='error',
                event_id=event_id,
            )

        state = self.library.get_sync_state(source_name)
        if state is None:
            event_id = self.library.log_usage('ariadne_sync_status', source, 0)
            return SyncStatusResponse(
                source=source_name,
                status='never_synced',
                event_id=event_id,
            )

        git_hash, synced_at = state
        commit_message = None
        source_path = self._source_path(source_name)
        if source_path and source_path.exists():
            commit_message = get_commit_message(git_hash, source_path)

        event_id = self.library.log_usage('ariadne_sync_status', source, 1)

        return SyncStatusResponse(
            source=source_name,
            status='synced',
            git_hash=git_hash,
            synced_at=synced_at,
            commit_message=commit_message,
            event_id=event_id,
        )

    # ------------------------------------------------------------------
    # Admin: branch sync, generate, merge
    # ------------------------------------------------------------------

    def branch_sync(
        self,
        source: str | None = None,
        dry_run: bool = True,
    ) -> AdminActionResponse:
        """Regenerate affected documents for current branch."""
        from git_ops import run_ariadne_cli

        status = self.branch_status(source)

        if status.message and ('no branch-specific' in status.message or 'not found' in status.message):
            return AdminActionResponse(output=status.message)

        if not status.affected_documents:
            return AdminActionResponse(output='No documents affected by branch changes.')

        if dry_run:
            lines = [
                f'Dry run — would regenerate {len(status.affected_documents)} document(s):',
                '',
            ]
            for d in status.affected_documents:
                lines.append(f'  - {d.title} ({d.content_type})')
            lines.append('')
            lines.append('Run with dry_run=false to regenerate these documents.')
            return AdminActionResponse(output='\n'.join(lines))

        source_name = self._resolve_source(source)
        output = run_ariadne_cli(
            ['sync', '--vs-main', '--branch', '--source', source_name or ''],
        )
        self.clear_cache()
        return AdminActionResponse(output=output)

    def generate(
        self,
        path: str,
        source: str | None = None,
        types: str = 'explanation,architecture',
        dry_run: bool = True,
    ) -> AdminActionResponse:
        """Generate documentation for a subdirectory."""
        from git_ops import run_ariadne_cli

        source_name = self._resolve_source(source)
        cli_args = [
            'generate',
            '--source', source_name or '',
            '--types', types,
            '--path', path,
        ]
        if dry_run:
            cli_args.append('--dry-run')

        output = run_ariadne_cli(cli_args)
        self.clear_cache()
        return AdminActionResponse(output=output)

    async def generate_file(
        self,
        file_path: str,
        source_name: str | None = None,
    ) -> AdminActionResponse:
        """Generate docs for a single file using DocGenOrchestrator directly.

        Unlike :meth:`generate` which shells out to the CLI, this method
        invokes the orchestrator in-process for a single file.
        """
        from pathlib import Path as _Path

        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        resolved_source = self._resolve_source(source_name)
        source_path = self._source_path(resolved_source) if resolved_source else None
        if source_path is None:
            return AdminActionResponse(output=f'Cannot resolve source for {file_path!r}')

        target = _Path(file_path)

        # Determine dependencies
        dependencies: tuple[str, ...] = ()
        if resolved_source:
            try:
                dependencies = tuple(self.config.get_source_dependencies(resolved_source))
            except Exception:
                pass

        config = OrchestratorConfig(
            source_path=source_path,
            db_path=_Path(self.config.db_path),
            staleness_db_path=_Path(self.config.staleness_db_path),
            model=self.config.model,
            source_name=resolved_source,
            dependencies=dependencies,
            target_path=target,
            force_regenerate=True,
        )

        async with DocGenOrchestrator(config) as orchestrator:
            result = await orchestrator.run()

        lines = [
            f'Files processed: {result.files_processed}',
            f'Files skipped: {result.files_skipped}',
            f'Docs created: {result.docs_created}',
            f'Docs failed: {result.docs_failed}',
        ]
        if result.errors:
            lines.append(f'Errors: {"; ".join(result.errors)}')
        self.clear_cache()
        return AdminActionResponse(output='\n'.join(lines))

    def merge(
        self,
        source: str | None = None,
        dry_run: bool = True,
        delete_consumed: bool = False,
    ) -> AdminActionResponse:
        """Detect merged branches and regenerate stable docs."""
        from git_ops import run_ariadne_cli

        if dry_run:
            try:
                from docgen.merge import preview_merge
                output = preview_merge(source, delete_consumed)
            except ImportError:
                output = 'Merge preview unavailable (docgen.merge not found).'
            return AdminActionResponse(output=output)

        source_name = self._resolve_source(source)
        cli_args = ['merge', '--source', source_name or '']
        if delete_consumed:
            cli_args.append('--delete-consumed')

        output = run_ariadne_cli(cli_args)
        self.clear_cache()
        return AdminActionResponse(output=output)

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def log_hit(self, event_id: int, feedback: str | None = None) -> FeedbackResponse:
        """Mark a usage event as a hit (useful result)."""
        success = self.library.mark_hit(event_id, feedback)
        return FeedbackResponse(
            success=success,
            message='Hit logged.' if success else 'Event not found.',
        )

    def log_miss(self, event_id: int, feedback: str) -> FeedbackResponse:
        """Mark a usage event as a miss (not useful)."""
        success = self.library.mark_miss(event_id, feedback)
        return FeedbackResponse(
            success=success,
            message='Miss logged.' if success else 'Event not found.',
        )
