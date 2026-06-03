"""Core merge detection and execution logic.

This module provides shared logic for detecting merged branches and
regenerating stable docs after branch merges. Used by both CLI and MCP.
"""
from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import subprocess
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

if TYPE_CHECKING:
    from config import Config
    from library import Library
    from schema import Document

_logger = logging.getLogger(__name__)


def select_catalog_files(source_files: Iterable[str]) -> list[str]:
    """Filter ``source_files`` to those Ariadne can regenerate docs for.

    Shared by ``preview_merge`` (dry-run reporter) and ``execute_merge``
    (actual regeneration) so the count the user sees in the preview
    matches the files that actually get regenerated. Without a single
    helper, the two paths drifted apart — preview said "would
    regenerate N files" while execute regenerated more.
    """
    from docgen.catalog_writer import CATALOG_EXTS

    return [
        f for f in source_files
        if any(f.endswith(ext) for ext in CATALOG_EXTS)
    ]


@frozen
class MergeResult:
    """Result of a merge operation.

    Attributes:
        merged_branches: Branch names confirmed as merged to main.
        consumed_docs: IDs of experimental docs consumed by this merge.
        files_regenerated: Number of source files processed for regeneration.
        docs_created: Number of new/updated docs created.
        docs_failed: Number of doc regeneration failures.
        docs_deprecated: Number of experimental docs deprecated.
        docs_deleted: Number of experimental docs deleted.
    """

    merged_branches: tuple[str, ...]
    consumed_docs: tuple[str, ...]
    files_regenerated: int
    docs_created: int
    docs_failed: int
    docs_deprecated: int
    docs_deleted: int


def _get_remote_repo(source_path: Path) -> str:
    """Extract GitHub owner/repo from git remote URL.

    Args:
        source_path: Path to the git repository.

    Returns:
        Repository identifier in 'owner/repo' format.

    Raises:
        RuntimeError: If remote URL cannot be parsed.
    """
    result = subprocess.run(
        ['git', 'remote', 'get-url', 'origin'],
        cwd=source_path,
        capture_output=True,
        text=True,
        check=True,
    )
    url = result.stdout.strip()

    # Handle SSH format: git@github.com:owner/repo.git
    if url.startswith('git@'):
        path = url.split(':')[-1]
        return path.removesuffix('.git')

    # Handle HTTPS format: https://github.com/owner/repo.git
    parts = url.rstrip('/').split('/')
    repo = parts[-1].removesuffix('.git')
    owner = parts[-2]
    return f'{owner}/{repo}'


def find_merged_branches(
    source_path: Path,
    branch_names: list[str],
    main_branch: str = 'main',
) -> list[str]:
    """Check which branches have been merged to main via GitHub PRs.

    Uses `gh pr list` to detect merged PRs regardless of merge strategy
    (squash, rebase, or regular merge).

    Args:
        source_path: Path to the git repository.
        branch_names: Branch names to check.
        main_branch: Name of the main branch.

    Returns:
        List of branch names confirmed as merged.

    Raises:
        FileNotFoundError: If `gh` CLI is not installed.
    """
    # Verify gh is available
    try:
        subprocess.run(
            ['gh', '--version'],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        raise FileNotFoundError(
            'gh CLI required for merge detection. Install: https://cli.github.com/'
        )

    repo = _get_remote_repo(source_path)
    merged = []

    for branch in branch_names:
        try:
            result = subprocess.run(
                [
                    'gh', 'pr', 'list',
                    '--state', 'merged',
                    '--head', branch,
                    '--json', 'mergedAt',
                    '--limit', '1',
                    '--repo', repo,
                ],
                capture_output=True,
                text=True,
                check=True,
            )
            data = json.loads(result.stdout)
            if data:
                merged.append(branch)
        except subprocess.CalledProcessError:
            _logger.debug('Failed to check PR status for branch %s', branch)
            continue

    return merged


def resolve_branch_patterns(
    source_path: Path,
    patterns: list[str],
) -> dict[str, list[str]]:
    """Resolve branch glob patterns against remote branches.

    Args:
        source_path: Path to the git repository.
        patterns: Branch name patterns (may contain globs like 'feat/138-*').

    Returns:
        Mapping of pattern to list of matched remote branch names.
    """
    # Get all remote branches
    try:
        result = subprocess.run(
            ['git', 'branch', '-r'],
            cwd=source_path,
            capture_output=True,
            text=True,
            check=True,
        )
        remote_branches = []
        for line in result.stdout.strip().split('\n'):
            branch = line.strip()
            if ' -> ' in branch:
                continue  # Skip HEAD -> origin/main
            # Strip origin/ prefix
            branch = branch.removeprefix('origin/')
            remote_branches.append(branch)
    except subprocess.CalledProcessError:
        return {p: [] for p in patterns}

    resolved: dict[str, list[str]] = {}
    for pattern in patterns:
        if '*' in pattern or '?' in pattern or '[' in pattern:
            matches = [b for b in remote_branches if fnmatch.fnmatch(b, pattern)]
            resolved[pattern] = matches
        else:
            # Exact name — include if it exists in remotes, or just pass through
            resolved[pattern] = [pattern]

    return resolved


def collect_experimental_docs(library: Library) -> list[Document]:
    """Get all experimental documents from the library.

    Args:
        library: The Ariadne library instance.

    Returns:
        List of documents with status 'experimental'.
    """
    docs = library.list_documents()
    return [d for d in docs if d.metadata.get('status') == 'experimental']


def get_consumed_docs(
    library: Library,
    source_path: Path,
    main_branch: str = 'main',
) -> tuple[list[Document], list[str]]:
    """Find experimental docs whose branches have been merged.

    Orchestrates branch pattern resolution and merge detection to identify
    which experimental docs can be consumed (deprecated/deleted) after merge.

    Args:
        library: The Ariadne library instance.
        source_path: Path to the git repository.
        main_branch: Name of the main branch.

    Returns:
        Tuple of (consumed_docs, merged_branches).
    """
    experimental = collect_experimental_docs(library)
    if not experimental:
        return [], []

    # Extract all unique branch names/patterns from experimental docs
    all_patterns: set[str] = set()
    for doc in experimental:
        branches = doc.metadata.get('branches', [])
        if isinstance(branches, list):
            all_patterns.update(branches)

    if not all_patterns:
        return [], []

    # Resolve glob patterns to actual branch names
    resolved = resolve_branch_patterns(source_path, list(all_patterns))

    # Collect all concrete branch names to check
    concrete_branches: set[str] = set()
    for matches in resolved.values():
        concrete_branches.update(matches)

    if not concrete_branches:
        return [], []

    # Check which branches are merged
    merged = find_merged_branches(source_path, list(concrete_branches), main_branch)
    merged_set = set(merged)

    if not merged_set:
        return [], []

    # Filter docs to those whose branches overlap with merged branches
    consumed: list[Document] = []
    for doc in experimental:
        branches = doc.metadata.get('branches', [])
        if not isinstance(branches, list):
            continue

        doc_consumed = False
        for pattern in branches:
            matched_branches = resolved.get(pattern, [pattern])
            if any(b in merged_set for b in matched_branches):
                doc_consumed = True
                break

        if doc_consumed:
            consumed.append(doc)

    return consumed, merged


def preview_merge(
    source: str | None = None,
    delete_consumed: bool = False,
) -> str:
    """Preview what a merge operation would do.

    Shared by both MCP and CLI dry-run paths.

    Args:
        source: Source name (uses default if None).
        delete_consumed: Whether consumed docs would be deleted vs deprecated.

    Returns:
        Human-readable preview string.
    """
    from config import get_config
    from library import Library

    cfg = get_config()
    source_name = source or cfg.default_source

    if not source_name:
        return 'Error: No source specified and no default_source in config'

    source_path = cfg.get_source_path(source_name)
    if source_path is None or not source_path.exists():
        return f'Error: Source path not found: {source_path}'

    library = Library(Path(cfg.db_path))
    try:
        experimental = collect_experimental_docs(library)
        if not experimental:
            return 'Nothing to merge — no experimental docs found.'

        try:
            consumed_docs, merged_branches = get_consumed_docs(
                library, source_path, cfg.main_branch,
            )
        except FileNotFoundError as e:
            return f'Error: {e}'

        if not consumed_docs:
            return f'No merged branches detected ({len(experimental)} experimental doc(s) remain).'

        parts = [
            f'Merged branches: {", ".join(merged_branches)}',
            f'Experimental docs from merged branches: {len(consumed_docs)}',
            '',
        ]
        for doc in consumed_docs[:15]:
            branches = doc.metadata.get('branches', [])
            branch_str = f' [{", ".join(branches)}]' if branches else ''
            parts.append(f'  - {doc.title} ({doc.content_type}){branch_str}')
        if len(consumed_docs) > 15:
            parts.append(f'  ... and {len(consumed_docs) - 15} more')

        source_files: set[str] = set()
        for doc in consumed_docs:
            source_files.update(doc.source_files)
        catalog_files = select_catalog_files(source_files)
        parts.append('')
        parts.append(f'Source files to regenerate: {len(catalog_files)}')
        action = 'delete' if delete_consumed else 'deprecate'
        parts.append(f'Would {action} {len(consumed_docs)} experimental doc(s) and regenerate {len(catalog_files)} file(s).')
        parts.append('')
        parts.append('Run with dry_run=false to execute.')
        return '\n'.join(parts)
    finally:
        library.close()


async def execute_merge(
    library: Library,
    cfg: Config,
    source_name: str,
    source_path: Path,
    *,
    db_path: Path | None = None,
    since: str | None = None,
    skip_generate: bool = False,
    no_export: bool = False,
    delete_consumed: bool = False,
) -> MergeResult:
    """Execute the full merge workflow.

    1. Detect consumed experimental docs from merged branches
    2. Collect source files that need regeneration
    3. Regenerate stable docs via DocGenOrchestrator
    4. Deprecate or delete consumed experimental docs
    5. Export and update sync state

    Args:
        library: The Ariadne library instance.
        cfg: Ariadne configuration.
        source_name: Name of the source to process.
        source_path: Path to the source repository.
        db_path: Path to the library database (default from config).
        since: Git hash to compare from (default: last sync point).
        skip_generate: If True, skip regeneration and only deprecate.
        no_export: If True, skip markdown export.
        delete_consumed: If True, delete consumed docs instead of deprecating.

    Returns:
        MergeResult with counts of actions taken.
    """
    main_branch = cfg.main_branch

    # Step 1: Get consumed docs
    consumed_docs, merged_branches = get_consumed_docs(
        library, source_path, main_branch,
    )

    if not consumed_docs:
        return MergeResult(
            merged_branches=(),
            consumed_docs=(),
            files_regenerated=0,
            docs_created=0,
            docs_failed=0,
            docs_deprecated=0,
            docs_deleted=0,
        )

    # Step 2: Collect source files to regenerate
    source_files_to_regen: set[str] = set()
    for doc in consumed_docs:
        source_files_to_regen.update(doc.source_files)

    # If we have a since hash, also include git diff files
    if since:
        try:
            result = subprocess.run(
                ['git', 'diff', '--name-only', f'{since}..HEAD'],
                cwd=source_path,
                capture_output=True,
                text=True,
                check=True,
            )
            diff_files = [f for f in result.stdout.strip().split('\n') if f]
            source_files_to_regen.update(diff_files)
        except subprocess.CalledProcessError:
            pass

    # Filter to extensions covered by the catalog (multi-language).
    # Shared with ``preview_merge`` so dry-run counts match reality.
    catalog_files = select_catalog_files(source_files_to_regen)

    # Step 3: Regenerate docs
    docs_created = 0
    docs_failed = 0
    files_regenerated = 0

    if catalog_files and not skip_generate:
        from docgen.orchestrator import DocGenOrchestrator, OrchestratorConfig

        actual_db_path = db_path or Path(cfg.db_path)

        # SCIP source_config — Scala/Java sources go through SCIP; for
        # the legacy/Python-only path source_config stays None.
        scip_config = cfg.get_source_scip_config(source_name)

        config = OrchestratorConfig(
            source_path=source_path,
            db_path=actual_db_path,
            staleness_db_path=Path(cfg.staleness_db_path),
            model=cfg.model,
            doc_types=('explanation', 'architecture'),
            force_regenerate=True,
            source_config=scip_config,
        )

        async with DocGenOrchestrator(config) as orchestrator:
            semaphore = asyncio.Semaphore(orchestrator.config.concurrency)

            async def process_one(rel_path: str) -> tuple[int, int]:
                file_path = source_path / rel_path
                if not file_path.exists():
                    return 0, 0
                async with semaphore:
                    try:
                        result = await orchestrator._process_file(file_path)
                        return result.docs_generated, result.docs_failed
                    except Exception as e:
                        _logger.error('Error processing %s: %s', rel_path, e)
                        return 0, 1

            tasks = [process_one(rel_path) for rel_path in catalog_files]
            results = await asyncio.gather(*tasks)

            for created, failed in results:
                docs_created += created
                docs_failed += failed
            files_regenerated = len(catalog_files)

    # Step 4: Deprecate or delete consumed docs
    docs_deprecated = 0
    docs_deleted = 0

    for doc in consumed_docs:
        if delete_consumed:
            library.delete_document(doc.id)
            docs_deleted += 1
        else:
            updated_metadata = dict(doc.metadata)
            updated_metadata['status'] = 'deprecated'
            library.update_document(doc.id, metadata=updated_metadata)
            docs_deprecated += 1

    # Step 5: Export
    if not no_export:
        from export import LibraryExporter
        exporter = LibraryExporter(library)
        output_dir = cfg.resolve_docs_path(source_name)
        exporter.export_all(
            output_dir=output_dir,
            source_name=source_name,
            source_path=source_path,
        )

    # Step 6: Update sync state
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'],
            cwd=source_path,
            capture_output=True,
            text=True,
            check=True,
        )
        library.set_sync_state(source_name, result.stdout.strip())
    except subprocess.CalledProcessError:
        pass

    return MergeResult(
        merged_branches=tuple(merged_branches),
        consumed_docs=tuple(d.id for d in consumed_docs),
        files_regenerated=files_regenerated,
        docs_created=docs_created,
        docs_failed=docs_failed,
        docs_deprecated=docs_deprecated,
        docs_deleted=docs_deleted,
    )
