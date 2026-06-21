"""Staleness tracking for documentation.

This module provides tools for detecting when source files have changed
and documentation needs to be regenerated.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

from attrs import define, field, frozen


def _compute_file_hash(path: Path) -> str:
    """Compute SHA256 hash of a file's contents."""
    content = path.read_bytes()
    return hashlib.sha256(content).hexdigest()


@frozen
class SourceRecord:
    """Record of a documented source file.

    Attributes:
        path: Relative path to the source file.
        hash: SHA256 hash of the file contents when documented.
        documented_at: ISO timestamp of when documentation was generated.
        doc_ids: IDs of documents generated from this source.
    """

    path: str
    hash: str
    documented_at: str
    doc_ids: tuple[str, ...] = ()


@frozen
class PendingBatch:
    """Persisted record of an Anthropic batch that's been submitted
    but whose results haven't been fetched yet.

    Used by the orchestrator's resume path: a crash mid-poll means
    we've already paid Anthropic for the batch (24h SLA), so we must
    be able to recover it on the next ``ariadne generate`` instead of
    resubmitting and double-billing.

    Payloads are stored as JSON strings rather than parsed structures
    so this dataclass stays immutable + hashable + cheap to log.
    The orchestrator round-trips them through ``json.loads`` /
    ``json.dumps`` at the boundary.

    Attributes:
        batch_id: Anthropic-assigned batch identifier (also the PK).
        submitted_at: ISO 8601 UTC timestamp of submission.
        prompts_json: Serialized ``list[dict]`` — the ``PromptBundle``
            wire form (file/doc_type/system_prompt/user_prompt/title/
            metadata, with Path stringified).
        file_to_idxs_json: Serialized ``dict[str, list[int]]`` mapping
            source file path to indices into prompts.
        config_hash: Stable hash of the run config (doc_types,
            provider, model, source_path). The resume path refuses to
            adopt a pending batch whose config_hash mismatches the
            current run — submitting under different doc_types and
            then resuming under others would land docs the user
            didn't ask for.
    """

    batch_id: str
    submitted_at: str
    prompts_json: str
    file_to_idxs_json: str
    config_hash: str


@define
class StalenessTracker:
    """Tracks source file changes to detect stale documentation.

    This class maintains a database of source file hashes and their
    associated documentation. It can detect when source files have
    changed since their documentation was generated.

    Attributes:
        db_path: Path to the SQLite database file.
    """

    db_path: Path = field(converter=Path)
    # Per-language doc-type override (the doc-type screen's per-format
    # excludes), {language: (doc_type, ...)}: caps the effective set per
    # language so the gate/estimate ignore an excluded format. Default {}.
    doc_types_by_language: dict = field(factory=dict)
    _conn: sqlite3.Connection | None = field(default=None, init=False)
    _lock: asyncio.Lock = field(factory=asyncio.Lock, init=False)

    def __attrs_post_init__(self) -> None:
        """Initialize the database connection and schema."""
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _init_schema(self) -> None:
        """Create database tables if they don't exist."""
        if self._conn is None:
            return

        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS source_records (
                path TEXT PRIMARY KEY,
                hash TEXT NOT NULL,
                documented_at TEXT NOT NULL,
                doc_ids TEXT NOT NULL DEFAULT '[]'
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_source_hash ON source_records(hash)
        ''')
        # Pending batches (#45.3) — Anthropic batches that have been
        # submitted but not yet fetched. The orchestrator's resume path
        # consults this table on startup so a crash mid-poll doesn't
        # forfeit a batch the user's already paid for.
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS pending_batches (
                batch_id TEXT PRIMARY KEY,
                submitted_at TEXT NOT NULL,
                prompts_json TEXT NOT NULL,
                file_to_idxs_json TEXT NOT NULL,
                config_hash TEXT NOT NULL
            )
        ''')
        self._conn.execute('''
            CREATE INDEX IF NOT EXISTS idx_pending_config_hash
            ON pending_batches(config_hash)
        ''')
        # Reverse-augment freshness markers. Keyed by (source, path) — unlike
        # source_records (which has no source column), so the same relative
        # path in two sources doesn't collide. ``marker`` is sha256(source +
        # rendered consumer-context); an unchanged marker means the augment
        # prompt would be identical, so the file is skipped on re-run instead
        # of re-billed. See docgen.reverse_augment.augment_marker.
        self._conn.execute('''
            CREATE TABLE IF NOT EXISTS augment_markers (
                source TEXT NOT NULL,
                path TEXT NOT NULL,
                marker TEXT NOT NULL,
                PRIMARY KEY (source, path)
            )
        ''')
        self._conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> StalenessTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def get_augment_marker(self, source: str, path: str) -> str | None:
        """Return the stored reverse-augment freshness marker for
        ``(source, path)``, or None if never augmented. See
        docgen.reverse_augment for the marker semantics."""
        if self._conn is None:
            return None
        row = self._conn.execute(
            'SELECT marker FROM augment_markers WHERE source = ? AND path = ?',
            (source, path),
        ).fetchone()
        return row['marker'] if row else None

    def set_augment_marker(self, source: str, path: str, marker: str) -> None:
        """Record the reverse-augment freshness marker for ``(source, path)``.

        Called only after the file's augmented docs are durably stored, so a
        re-run skips it; an interrupted pass leaves unfinished files unmarked
        and they retry. Commits immediately for crash-durability."""
        if self._conn is None:
            return
        self._conn.execute(
            '''
            INSERT INTO augment_markers (source, path, marker)
            VALUES (?, ?, ?)
            ON CONFLICT(source, path) DO UPDATE SET marker = excluded.marker
            ''',
            (source, path, marker),
        )
        self._conn.commit()

    def record_documentation(
        self,
        source_path,
        doc_ids,
        base_path=None,
    ):
        """Record that a source file has been documented.

        Merges ``doc_ids`` with any existing record's doc_ids so a partial-type
        generation (e.g. adding qa/gotcha to a file that already has
        explanation/architecture) does NOT drop the doc_ids of the types it
        didn't regenerate. Doc ids are deterministic per (source, content_type),
        so the union dedupes cleanly — a regenerated type keeps its id, no stale
        accumulation. A re-run sees the now-complete file and skips it.
        """
        if self._conn is None:
            msg = 'Database connection not open'
            raise RuntimeError(msg)

        if base_path:
            rel_path = str(source_path.relative_to(base_path))
        else:
            rel_path = str(source_path)

        file_hash = _compute_file_hash(source_path)
        documented_at = datetime.now(UTC).isoformat()

        existing = self.get_record(rel_path)
        merged = list(dict.fromkeys(
            [*(existing.doc_ids if existing else ()), *doc_ids]
        ))

        self._conn.execute(
            '''
            INSERT OR REPLACE INTO source_records (path, hash, documented_at, doc_ids)
            VALUES (?, ?, ?, ?)
            ''',
            (rel_path, file_hash, documented_at, json.dumps(merged)),
        )
        self._conn.commit()

        return SourceRecord(
            path=rel_path,
            hash=file_hash,
            documented_at=documented_at,
            doc_ids=tuple(merged),
        )

    async def record_documentation_async(
        self,
        source_path: Path,
        doc_ids: list[str],
        base_path: Path | None = None,
    ) -> SourceRecord:
        """Async version with locking for concurrent access.

        Args:
            source_path: Path to the source file.
            doc_ids: IDs of the generated documents.
            base_path: Base path for computing relative paths.

        Returns:
            The created SourceRecord.
        """
        async with self._lock:
            return self.record_documentation(source_path, doc_ids, base_path)

    def get_record(self, source_path: Path | str) -> SourceRecord | None:
        """Get the record for a source file.

        Args:
            source_path: Path to the source file.

        Returns:
            The SourceRecord if found, None otherwise.
        """
        if self._conn is None:
            return None

        path_str = str(source_path)
        cursor = self._conn.execute(
            'SELECT * FROM source_records WHERE path = ?',
            (path_str,),
        )
        row = cursor.fetchone()

        if row is None:
            return None

        return SourceRecord(
            path=row['path'],
            hash=row['hash'],
            documented_at=row['documented_at'],
            doc_ids=tuple(json.loads(row['doc_ids'])),
        )

    def is_stale(
        self,
        source_path: Path,
        base_path: Path | None = None,
    is_exempt = None) -> bool:
        """Check if documentation for a source file is stale.

        A file is considered stale if:
        - It has no record (never documented)
        - Its current hash differs from the recorded hash

        Args:
            source_path: Path to the source file.
            base_path: Base path for computing relative paths.

        Returns:
            True if the documentation is stale, False otherwise.
        """
        if not source_path.exists():
            return False

        # Compute relative path
        if base_path:
            rel_path = str(source_path.relative_to(base_path))
        else:
            rel_path = str(source_path)

        record = self.get_record(rel_path)
        if record is None:
            return True  # Never documented

        current_hash = _compute_file_hash(source_path)
        if current_hash == record.hash:
            return False
        # Hash changed: stale, unless this path is staleness-exempt.
        if is_exempt is not None and is_exempt(rel_path):
            return False
        return True

    def get_stale_files(
        self,
        paths: list[Path],
        base_path: Path | None = None,
        requested_types: tuple[str, ...] | None = None,
        library: object | None = None,
    is_exempt = None) -> list[Path]:
        """Get all stale files from a list of paths.

        Language-agnostic: callers (e.g. ``find_catalog_files`` or
        ``find_python_files``) already filter the input list to the
        extensions they care about.

        Args:
            paths: List of source file paths to check.
            base_path: Base path for computing relative paths.
            requested_types: Doc types the caller intends to generate.
                When provided alongside ``library``, a file is treated
                as stale if ANY requested type isn't already on record
                — even if the source hash hasn't changed. Lets callers
                add doc types incrementally without ``--force``.
            library: Library used to resolve ``doc_ids → content_type``.
                Required when ``requested_types`` is set; ignored
                otherwise (backwards-compat for legacy callers).

        Returns:
            List of paths that have stale or missing documentation.
        """
        stale: list[Path] = []
        type_aware = bool(requested_types) and library is not None
        requested_set = set(requested_types or ())

        for path in paths:
            if not path.is_file():
                continue

            # Legacy hash-only path: never-documented or changed source.
            if self.is_stale(path, base_path, is_exempt=is_exempt):
                stale.append(path)
                continue

            if not type_aware:
                continue

            # Hash matches and we have type-aware filtering: stale only if
            # any requested doc_type is missing from the file's prior docs.
            if base_path:
                rel_path = str(path.relative_to(base_path))
            else:
                rel_path = str(path)
            record = self.get_record(rel_path)
            if record is None:
                # Should be unreachable since is_stale would have caught it,
                # but be defensive.
                stale.append(path)
                continue

            existing_types = self._existing_doc_types(record, library)

            # Filter requested_set down to what this file's LANGUAGE supports.
            # Without this, JSON/YAML/MD files (which only get `explanation`
            # per LANGUAGE_DOC_TYPES) are eternally stale: they can never have
            # architecture/qa/gotcha/diagram, so the issubset check forever
            # fails and the file gets retried every run for no progress.
            effective_requested = self._filter_requested_by_language(
                path, requested_set,
            )
            if not effective_requested.issubset(existing_types):
                stale.append(path)

        return stale

    def stale_doc_types(self, paths, base_path=None, requested_types=None,
                        library=None, is_exempt=None):
        """Per file, the requested doc types that still need generating.

        Returns ``{path: (doc_type, ...)}`` for files needing work. The requested
        set is first intersected with what the file's LANGUAGE supports
        (``LANGUAGE_DOC_TYPES``). A file whose source **changed** (hash mismatch)
        or that was never documented needs every effective type; an **unchanged**
        file needs only the effective types not already present in its recorded
        docs (resolved precisely via ``record.doc_ids``, never basename-matched).
        Files needing nothing are omitted, so generation/pricing do only the
        missing work.

        Staleness exemption suppresses only the **content-changed** signal: an
        exempt file whose source changed is not re-flagged, but a missing
        requested type still surfaces as a coverage gap (never-documented files
        always surface too). ``ignore_staleness`` quiets the regenerate nag; it
        does not hide absent docs.
        """
        out = {}
        requested_set = set(requested_types or ())
        for path in paths:
            if not path.is_file():
                continue
            rel_path = str(path.relative_to(base_path)) if base_path else str(path)
            effective = self._filter_requested_by_language(path, requested_set)
            if not effective:
                continue
            record = self.get_record(rel_path)
            if record is None:
                out[path] = tuple(sorted(effective))
                continue
            exempt = is_exempt is not None and is_exempt(rel_path)
            if not exempt and _compute_file_hash(path) != record.hash:
                out[path] = tuple(sorted(effective))
                continue
            if library is None:
                continue
            existing = self._existing_doc_types(record, library)
            missing = effective - existing
            if missing:
                out[path] = tuple(sorted(missing))
        return out

    def coverage_gaps(self, paths, base_path=None, requested_types=None, library=None):
        """Per file, the requested doc types ABSENT from its docs — ignoring content
        staleness (this does not hash) and exemption (a missing doc is a coverage
        gap that surfaces regardless of ``ignore_staleness``, which suppresses only
        the content-changed signal). A never-documented file needs every effective
        type; a recorded file needs only the effective types not already present
        (via ``record.doc_ids``). The cheap "what doc-type coverage is missing?"
        half of ``files_for_generation``; with no ``library`` only never-documented
        files are reported."""
        out = {}
        requested_set = set(requested_types or ())
        for path in paths:
            if not path.is_file():
                continue
            rel_path = str(path.relative_to(base_path)) if base_path else str(path)
            effective = self._filter_requested_by_language(path, requested_set)
            if not effective:
                continue
            record = self.get_record(rel_path)
            if record is None:
                out[path] = tuple(sorted(effective))
                continue
            if library is None:
                continue
            existing = self._existing_doc_types(record, library)
            missing = effective - existing
            if missing:
                out[path] = tuple(sorted(missing))
        return out

    def files_for_generation(self, all_files, *, base_path, requested_types,
                             library, is_exempt, restrict_to_files, force):
        """The files the next generate run will process and the doc types each
        needs — the single selection shared by ``DocGenOrchestrator.run`` and the
        dry-run estimate, so the preview can't diverge from the run.

        - ``force`` → every file at the full requested set (empty narrowing map).
        - commit-diff gate (``restrict_to_files`` not None) → files changed since
          the sync baseline UNION files missing a requested doc type
          (``coverage_gaps``), so a synced-but-unchanged source still fills
          newly-requested types. Changed files regenerate the full set (omitted
          from the map); coverage-gap files only their missing types.
        - otherwise → the full type-aware staleness pass.

        Returns ``(files_to_process, doc_types_by_file)``; the map omits files
        that regenerate the full requested set (callers fall back to it).
        """
        if force:
            return list(all_files), {}
        if restrict_to_files is not None:
            def _rel(f):
                try:
                    return f.relative_to(base_path).as_posix()
                except ValueError:
                    return f.as_posix()
            changed = [f for f in all_files if _rel(f) in restrict_to_files]
            changed_set = set(changed)
            gaps = self.coverage_gaps(
                all_files, base_path=base_path, requested_types=requested_types,
                library=library, )
            extra = [p for p in gaps if p not in changed_set]
            doc_types = {p: t for p, t in gaps.items() if p not in changed_set}
            return changed + extra, doc_types
        files = self.get_stale_files(
            all_files, base_path=base_path, requested_types=requested_types,
            library=library, is_exempt=is_exempt,
        )
        doc_types = self.stale_doc_types(
            files, base_path=base_path, requested_types=requested_types,
            library=library, is_exempt=is_exempt,
        )
        return files, doc_types

    def _filter_requested_by_language(
        self, path: 'Path', requested: set[str],
    ) -> set[str]:
        """Intersect ``requested`` doc types with what the file's language
        actually supports per ``LANGUAGE_DOC_TYPES``.

        Returns ``requested`` unchanged if the language is unknown — keeps
        the legacy behavior for unsupported extensions instead of silently
        widening or narrowing.
        """
        from docgen.catalog_extractor import _detect_language
        from docgen.prompts import filter_doc_types_for_language
        lang = _detect_language(path)
        if lang is None:
            return requested
        return set(filter_doc_types_for_language(tuple(requested), lang, override=self.doc_types_by_language))

    @staticmethod
    def _existing_doc_types(record, library):
        """Content types already documented for ``record``, resolved precisely via
        its ``doc_ids`` (never basename-matched). Callers pass a non-None
        ``library``. Single source of truth for the existing-types lookup shared
        by ``get_stale_files`` and ``stale_doc_types``."""
        existing = set()
        for doc_id in record.doc_ids:
            doc = library.get_document(doc_id)
            if doc is not None:
                existing.add(doc.content_type)
        return existing

    def get_undocumented_files(
        self,
        paths: list[Path],
        base_path: Path | None = None,
    ) -> list[Path]:
        """Get files that have never been documented.

        Language-agnostic — see ``get_stale_files``.
        """
        undocumented = []
        for path in paths:
            if not path.is_file():
                continue
            if base_path:
                rel_path = str(path.relative_to(base_path))
            else:
                rel_path = str(path)
            if self.get_record(rel_path) is None:
                undocumented.append(path)
        return undocumented

    def get_all_records(self) -> Iterator[SourceRecord]:
        """Iterate over all source records.

        Yields:
            SourceRecord for each documented file.
        """
        if self._conn is None:
            return

        cursor = self._conn.execute('SELECT * FROM source_records ORDER BY path')
        for row in cursor:
            yield SourceRecord(
                path=row['path'],
                hash=row['hash'],
                documented_at=row['documented_at'],
                doc_ids=tuple(json.loads(row['doc_ids'])),
            )

    def remove_record(self, source_path: Path | str) -> bool:
        """Remove the record for a source file.

        Args:
            source_path: Path to the source file.

        Returns:
            True if a record was removed, False otherwise.
        """
        if self._conn is None:
            return False

        path_str = str(source_path)
        cursor = self._conn.execute(
            'DELETE FROM source_records WHERE path = ?',
            (path_str,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def clear_all(self) -> int:
        """Remove all records.

        Returns:
            Number of records removed.
        """
        if self._conn is None:
            return 0

        cursor = self._conn.execute('DELETE FROM source_records')
        self._conn.commit()
        return cursor.rowcount

    # -------------------------------------------------------------------
    # Pending batches (#45.3) — Anthropic batches that have been
    # submitted but not yet fetched. The orchestrator's resume path
    # uses these to recover from a crash between submit and fetch.
    # -------------------------------------------------------------------

    def record_pending_batch(
        self,
        batch_id: str,
        prompts_json: str,
        file_to_idxs_json: str,
        config_hash: str,
    ) -> PendingBatch:
        """Persist a pending batch and return the record.

        ``INSERT OR REPLACE`` so re-recording the same ``batch_id``
        (e.g., a retried submit that landed twice) updates the
        timestamp instead of erroring on duplicate PK. Anthropic's
        batch IDs are server-assigned, so duplicate ``batch_id`` only
        happens on user error or harness re-runs.
        """
        if self._conn is None:
            msg = 'Database connection not open'
            raise RuntimeError(msg)

        submitted_at = datetime.now(UTC).isoformat()
        self._conn.execute(
            '''
            INSERT OR REPLACE INTO pending_batches
            (batch_id, submitted_at, prompts_json, file_to_idxs_json, config_hash)
            VALUES (?, ?, ?, ?, ?)
            ''',
            (batch_id, submitted_at, prompts_json, file_to_idxs_json, config_hash),
        )
        self._conn.commit()
        return PendingBatch(
            batch_id=batch_id,
            submitted_at=submitted_at,
            prompts_json=prompts_json,
            file_to_idxs_json=file_to_idxs_json,
            config_hash=config_hash,
        )

    def clear_pending_batch(self, batch_id: str) -> bool:
        """Delete a pending batch by id; return True iff a row was
        removed.

        Idempotent: a second call with the same id returns False
        (and doesn't raise) so ``ariadne batch clear`` can safely
        run on a typo or already-cleared id.
        """
        if self._conn is None:
            return False
        cursor = self._conn.execute(
            'DELETE FROM pending_batches WHERE batch_id = ?',
            (batch_id,),
        )
        self._conn.commit()
        return cursor.rowcount > 0

    def find_pending_batch(
        self, config_hash: str,
    ) -> PendingBatch | None:
        """Return the most recent pending batch matching ``config_hash``,
        or None.

        Ordered by ``rowid DESC`` so the freshest insertion wins
        regardless of timestamp resolution. The orchestrator's resume
        path calls this with the current run's config hash; non-match
        means "no recoverable batch for this run" and submit goes
        ahead normally.
        """
        if self._conn is None:
            return None
        cursor = self._conn.execute(
            '''
            SELECT batch_id, submitted_at, prompts_json,
                   file_to_idxs_json, config_hash
            FROM pending_batches
            WHERE config_hash = ?
            ORDER BY rowid DESC
            LIMIT 1
            ''',
            (config_hash,),
        )
        row = cursor.fetchone()
        if row is None:
            return None
        return PendingBatch(
            batch_id=row['batch_id'],
            submitted_at=row['submitted_at'],
            prompts_json=row['prompts_json'],
            file_to_idxs_json=row['file_to_idxs_json'],
            config_hash=row['config_hash'],
        )

    def list_pending_batches(self) -> list[PendingBatch]:
        """Return all pending batches, newest first.

        Surface for the ``ariadne batch list`` CLI (#45.10) — lets
        users see what's in flight or orphaned across multiple runs.
        """
        if self._conn is None:
            return []
        cursor = self._conn.execute(
            '''
            SELECT batch_id, submitted_at, prompts_json,
                   file_to_idxs_json, config_hash
            FROM pending_batches
            ORDER BY rowid DESC
            '''
        )
        return [
            PendingBatch(
                batch_id=row['batch_id'],
                submitted_at=row['submitted_at'],
                prompts_json=row['prompts_json'],
                file_to_idxs_json=row['file_to_idxs_json'],
                config_hash=row['config_hash'],
            )
            for row in cursor
        ]

    def normalize_paths(self, source_path: Path) -> list[tuple[str, str]]:
        """Normalize all stored paths to be relative to source_path.

        Detects paths with ``../`` prefixes or other inconsistencies and
        rewrites them as proper relative paths.

        Args:
            source_path: The resolved source directory that paths should
                be relative to.

        Returns:
            List of (old_path, new_path) tuples for every record that
            was updated.
        """
        if self._conn is None:
            return []

        source_path = source_path.resolve()
        changes: list[tuple[str, str]] = []

        cursor = self._conn.execute('SELECT path, hash, documented_at, doc_ids FROM source_records')
        rows = cursor.fetchall()

        for row in rows:
            old_path = row['path']
            # Resolve the stored path against source_path to get an absolute path,
            # then re-relativise it.
            try:
                absolute = (source_path / old_path).resolve()
                new_path = str(absolute.relative_to(source_path))
            except ValueError:
                # Path is outside source_path — leave it alone
                continue

            if new_path != old_path:
                changes.append((old_path, new_path))

        # Apply updates inside an explicit transaction
        if changes:
            # Partition into deletes (target exists) vs updates (target is new)
            existing_targets = {
                row[0]
                for row in self._conn.execute(
                    'SELECT path FROM source_records WHERE path IN ({})'.format(
                        ','.join('?' for _ in changes)
                    ),
                    [new for _, new in changes],
                ).fetchall()
            }

            deletes = [(old,) for old, new in changes if new in existing_targets]
            updates = [(new, old) for old, new in changes if new not in existing_targets]

            if deletes:
                self._conn.executemany(
                    'DELETE FROM source_records WHERE path = ?',
                    deletes,
                )
            if updates:
                self._conn.executemany(
                    'UPDATE source_records SET path = ? WHERE path = ?',
                    updates,
                )
            self._conn.commit()

        return changes

    def get_stats(self) -> dict:
        """Get statistics about tracked files.

        Returns:
            Dictionary with count statistics.
        """
        if self._conn is None:
            return {'total': 0, 'with_docs': 0}

        cursor = self._conn.execute('SELECT COUNT(*) FROM source_records')
        total = cursor.fetchone()[0]

        cursor = self._conn.execute(
            "SELECT COUNT(*) FROM source_records WHERE doc_ids != '[]'"
        )
        with_docs = cursor.fetchone()[0]

        return {
            'total': total,
            'with_docs': with_docs,
        }


def find_python_files(
    directory: Path,
    exclude_patterns: tuple[str, ...] | None = None,
    exclude_dir_names: tuple[str, ...] | None = None,
) -> list[Path]:
    """Find all Python files in a directory, pruning excluded subtrees.

    Args:
        directory: Directory to search.
        exclude_patterns: Glob patterns to exclude (per-file matching).
        exclude_dir_names: Directory NAMES pruned at the walk level —
            every file under any directory with one of these names is
            skipped at every depth. ``None`` (default) falls back to
            ``config.DEFAULT_EXCLUDE_POLICY``; pass an explicit tuple
            (e.g. the resolved ``Config.resolve_excluded_dirs`` result)
            to override. ``()`` is honored verbatim (full walk).

    Returns:
        Sorted list of Python file paths.
    """
    from config import DEFAULT_EXCLUDE_FILE_PATTERNS, DEFAULT_EXCLUDE_POLICY

    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_FILE_PATTERNS
    if exclude_dir_names is None:
        exclude_dir_names = DEFAULT_EXCLUDE_POLICY

    files: list[Path] = []
    excluded_set = frozenset(exclude_dir_names)

    for dirpath, dirnames, filenames in directory.walk():
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in excluded_set
            and not d.endswith('.egg-info')
        ]

        for filename in filenames:
            if not filename.endswith('.py'):
                continue

            path = dirpath / filename

            # Check exclude patterns (test files etc.)
            excluded = False
            for pattern in exclude_patterns:
                if path.match(pattern):
                    excluded = True
                    break

            if not excluded:
                files.append(path)

    return sorted(files)


def find_catalog_files(
    directory: Path,
    exclude_patterns: tuple[str, ...] | None = None,
    exclude_dir_names: tuple[str, ...] | None = None,
) -> list[Path]:
    """Find all catalog-supported files under ``directory``.

    Mirrors :func:`find_python_files` (same directory-pruning rules) but
    accepts every extension in ``docgen.catalog_writer.CATALOG_EXTS``.

    Args:
        directory: Directory to search.
        exclude_patterns: Glob patterns to exclude (per-file matching).
        exclude_dir_names: Directory NAMES pruned at the walk level.
            ``None`` (default) falls back to
            ``config.DEFAULT_EXCLUDE_POLICY``; pass an explicit tuple
            (typically ``Config.resolve_excluded_dirs(source_name)``)
            to override. ``()`` is honored verbatim (full walk).

    Returns:
        Sorted list of catalog-supported file paths.
    """
    from config import DEFAULT_EXCLUDE_FILE_PATTERNS, DEFAULT_EXCLUDE_POLICY
    from docgen.catalog_writer import CATALOG_EXTS, is_catalog_noise, is_vue_companion, is_catalog_file

    if exclude_patterns is None:
        exclude_patterns = DEFAULT_EXCLUDE_FILE_PATTERNS
    if exclude_dir_names is None:
        exclude_dir_names = DEFAULT_EXCLUDE_POLICY

    files: list[Path] = []
    excluded_set = frozenset(exclude_dir_names)

    for dirpath, dirnames, filenames in directory.walk():
        # Prune excluded directories in-place
        dirnames[:] = [
            d for d in dirnames
            if d not in excluded_set
            and not d.endswith('.egg-info')
        ]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if not is_catalog_file(Path(filename)):
                continue
            if is_vue_companion(filename):
                continue

            path = dirpath / filename
            if is_catalog_noise(path):
                continue

            # Check exclude patterns (test files etc.)
            excluded = False
            for pattern in exclude_patterns:
                if path.match(pattern):
                    excluded = True
                    break

            if not excluded:
                files.append(path)

    return sorted(files)
