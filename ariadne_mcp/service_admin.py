"""Admin operations — listing, sync, generation, and feedback."""
from __future__ import annotations

import logging
from datetime import datetime

from ariadne_mcp.models import (
    AdminActionResponse,
    AffectedDocument,
    BranchStatusResponse,
    DocumentSummary,
    FeedbackResponse,
    ListResponse,
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
