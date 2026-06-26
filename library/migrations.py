"""One-off DB migrations.

Today this hosts ``migrate_doc_ids`` — backfill the deterministic-UUID5
scheme (``doc_id_for(source, type, primary_key)``) over docs that were
written under the original UUID4 scheme. This makes re-running
``ariadne generate`` idempotent and removes the title-collision dupes
the ID-as-UUID4 design produced (e.g. two ``llm`` modules both titled
``Llm Architecture`` getting two random IDs).

The migration is the prerequisite for running Ariadne against very
large codebases: without it the first re-run after deploy adds N new
duplicate rows next to the legacy ones.
"""
from __future__ import annotations

import ast
import json
import logging
import sqlite3
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import field, frozen

from schema import CATALOG_KIND_ELEMENT, doc_id_for

if TYPE_CHECKING:
    from sqlite3 import Connection

_logger = logging.getLogger(__name__)


# Content types that the orchestrator now writes with deterministic IDs
# via ``doc_id_for``. ``catalog`` / ``theme`` / ``finding`` use the older
# ``generate_deterministic_id`` scheme keyed on title or qualified-name —
# they're already stable and must not be touched.
_DOC_ID_FOR_CONTENT_TYPES: tuple[str, ...] = (
    'explanation', 'architecture', 'qa', 'diagram', 'gotcha',
)


@frozen
class MigrateDocIdsResult:
    """Outcome of one ``migrate_doc_ids`` call.

    Counts:
      - ``inspected``: rows whose content_type was in scope.
      - ``already_deterministic``: id == doc_id_for(source, type, key); skipped.
      - ``remapped``: id changed from legacy → deterministic.
      - ``duplicates_collapsed``: extra rows mapping to the same deterministic
        id; the newest survives, the rest are deleted (with their chunks /
        sections / theme_members / doc_graph rows).
      - ``skipped_no_source``: rows missing source_name or whose source isn't
        in the supplied source map; reported but not migrated.

    ``sample`` carries up to 10 (old, new, content_type) tuples for UI display.
    ``skipped_sample`` carries up to 20 (id, source_name, content_type, title)
    tuples — the rows that couldn't be remapped because their source_name was
    missing or not in the supplied source map.
    """
    inspected: int
    already_deterministic: int
    remapped: int
    duplicates_collapsed: int
    skipped_no_source: int
    sample: tuple[tuple[str, str, str], ...] = field(factory=tuple)
    skipped_sample: tuple[tuple[str, str | None, str, str], ...] = field(factory=tuple)
    skipped_source_names: tuple[tuple[str, int], ...] = field(factory=tuple)


def _compute_primary_key(
    *,
    title: str,
    source_files: list[str],
    metadata: dict[str, object],
    source_path: Path,
) -> str:
    """Mirror of ``DocGenOrchestrator._compute_deterministic_doc_id`` minus the
    final ``doc_id_for`` call. Kept exactly aligned so a migrated doc lands on
    the same id the orchestrator would use on its next run.
    """
    if metadata.get('group'):
        return f"group:{metadata.get('package_name', title)}"
    if metadata.get('topic'):
        return f"topic:{metadata.get('topic_title', title)}"
    if source_files:
        primary_file = source_files[0]
        try:
            return str(Path(primary_file).resolve().relative_to(source_path))
        except (ValueError, OSError):
            return primary_file
    return title


def _delete_doc_with_refs(conn: 'Connection', doc_id: str) -> None:
    """Manual cascade for losers under PRAGMA foreign_keys=OFF.

    With FK enforcement on, ``DELETE FROM documents`` would cascade through
    chunks/sections/themes/theme_members. We turn FK off during migration
    (so we can rewrite documents.id without ON UPDATE CASCADE), so we
    must do the cascade by hand here.
    """
    conn.execute('DELETE FROM chunks WHERE document_id = ?', (doc_id,))
    conn.execute('DELETE FROM sections WHERE document_id = ?', (doc_id,))
    conn.execute('DELETE FROM theme_members WHERE element_id = ?', (doc_id,))
    conn.execute(
        'DELETE FROM doc_graph WHERE source_id = ? OR target_id = ?',
        (doc_id, doc_id),
    )
    conn.execute('DELETE FROM themes WHERE doc_id = ?', (doc_id,))
    conn.execute('DELETE FROM documents WHERE id = ?', (doc_id,))


def _rename_doc_refs(conn: 'Connection', old_id: str, new_id: str) -> None:
    """Update every column that references ``old_id`` to ``new_id``.

    Includes ``documents.id`` itself; safe to call only with FK off.
    """
    conn.execute('UPDATE documents SET id = ? WHERE id = ?', (new_id, old_id))
    conn.execute(
        'UPDATE chunks SET document_id = ? WHERE document_id = ?',
        (new_id, old_id),
    )
    conn.execute(
        'UPDATE sections SET document_id = ? WHERE document_id = ?',
        (new_id, old_id),
    )
    conn.execute(
        'UPDATE themes SET doc_id = ? WHERE doc_id = ?',
        (new_id, old_id),
    )
    conn.execute(
        'UPDATE theme_members SET element_id = ? WHERE element_id = ?',
        (new_id, old_id),
    )
    conn.execute(
        'UPDATE doc_graph SET source_id = ? WHERE source_id = ?',
        (new_id, old_id),
    )
    conn.execute(
        'UPDATE doc_graph SET target_id = ? WHERE target_id = ?',
        (new_id, old_id),
    )


def _update_staleness_doc_ids(
    staleness_db: Path, id_map: dict[str, str],
) -> int:
    """Rewrite the ``doc_ids`` JSON column in the staleness DB.

    Each row's JSON array is parsed; any id present as a key in ``id_map``
    is replaced by its mapped value (others left alone). Returns number of
    rows updated.
    """
    if not staleness_db.exists() or not id_map:
        return 0
    updated = 0
    conn = sqlite3.connect(staleness_db)
    try:
        rows = conn.execute(
            'SELECT path, doc_ids FROM source_records WHERE doc_ids IS NOT NULL'
        ).fetchall()
        for path, blob in rows:
            try:
                ids = json.loads(blob) if blob else []
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(ids, list):
                continue
            new_ids = [id_map.get(i, i) for i in ids]
            if new_ids != ids:
                conn.execute(
                    'UPDATE source_records SET doc_ids = ? WHERE path = ?',
                    (json.dumps(new_ids), path),
                )
                updated += 1
        conn.commit()
    finally:
        conn.close()
    return updated


@frozen
class RepairLocationsResult:
    """Outcome of one ``repair_stringified_locations`` call.

    - ``inspected``: catalog ``element`` docs examined.
    - ``repaired``: ``location`` strings parsed back to dicts and rewritten.
    - ``already_dict``: locations already stored as dicts; left untouched.
    - ``unparseable``: ``location`` strings that did not parse to a dict; left
      untouched and surfaced (in ``unparseable_sample``) rather than dropped.
    """
    inspected: int
    repaired: int
    already_dict: int
    unparseable: int
    unparseable_sample: tuple[tuple[str, str], ...] = field(factory=tuple)


class MigrationsMixin:
    """One-off migration helpers attached to ``Library``.

    Expects the composed class to provide ``self._conn_provider``.
    """

    def migrate_doc_ids(
        self,
        *,
        source_name_to_path: dict[str, Path],
        content_types: Iterable[str] = _DOC_ID_FOR_CONTENT_TYPES,
        dry_run: bool = False,
        staleness_db_path: Path | None = None,
    ) -> MigrateDocIdsResult:
        """Backfill deterministic doc IDs for legacy UUID4 docs.

        Args:
            source_name_to_path: Map ``source_name → source_path`` so the
                migration can compute path-relative-to-source-root the
                way the orchestrator does. Sources missing from the map
                are skipped and counted in ``skipped_no_source``.
            content_types: Which content types are in scope. Defaults to
                LLM-written types — catalog/theme/finding already use the
                ``generate_deterministic_id`` scheme and stay untouched.
            dry_run: If True, return a result that reflects what would
                change but make no DB writes.
            staleness_db_path: Optional path to the staleness DB. When
                given, ``source_records.doc_ids`` JSON arrays are
                rewritten to point at the new IDs.

        Returns:
            ``MigrateDocIdsResult`` with the per-bucket counts.
        """
        ct_tuple = tuple(content_types)
        placeholders = ','.join('?' * len(ct_tuple))

        # Pull every in-scope row up front; the migration is offline.
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                f'''SELECT id, content_type, title, source_files, metadata,
                           source_name, updated_at
                    FROM documents
                    WHERE content_type IN ({placeholders})''',
                ct_tuple,
            ).fetchall()

        # Group every doc by its target (deterministic) id.
        # Each entry: (current_id, updated_at, content_type)
        groups: dict[str, list[tuple[str, str, str]]] = defaultdict(list)
        already_deterministic = 0
        skipped_no_source = 0
        skipped_sample: list[tuple[str, str | None, str, str]] = []
        skipped_counts: dict[str, int] = defaultdict(int)
        inspected = len(rows)

        for row in rows:
            current_id, ct, title, sf_json, meta_json, source_name, updated_at = row
            source_files = json.loads(sf_json) if sf_json else []
            metadata = json.loads(meta_json) if meta_json else {}

            if not source_name or source_name not in source_name_to_path:
                skipped_no_source += 1
                key = source_name or '<null>'
                skipped_counts[key] += 1
                if len(skipped_sample) < 20:
                    skipped_sample.append((current_id, source_name, ct, title))
                continue

            primary_key = _compute_primary_key(
                title=title,
                source_files=source_files,
                metadata=metadata,
                source_path=source_name_to_path[source_name],
            )
            target = doc_id_for(source_name, ct, primary_key)

            if current_id == target:
                already_deterministic += 1
                # Keep it in groups so we still detect duplicates that map
                # to this same target id (they'd lose to the deterministic one).
            groups[target].append((current_id, updated_at, ct))

        # Plan: per group, newest doc wins; others are losers.
        # Winner gets renamed (if its id != target). Losers are deleted.
        to_rename: list[tuple[str, str, str]] = []  # (old, new, content_type)
        to_delete: list[str] = []
        for target_id, members in groups.items():
            members.sort(key=lambda m: m[1], reverse=True)  # newest first
            winner_id, _, winner_ct = members[0]
            for old_id, _, _ in members[1:]:
                to_delete.append(old_id)
            if winner_id != target_id:
                to_rename.append((winner_id, target_id, winner_ct))

        sample = tuple(to_rename[:10])

        skipped_source_names = tuple(
            sorted(skipped_counts.items(), key=lambda kv: -kv[1])
        )

        if dry_run:
            return MigrateDocIdsResult(
                inspected=inspected,
                already_deterministic=already_deterministic,
                remapped=len(to_rename),
                duplicates_collapsed=len(to_delete),
                skipped_no_source=skipped_no_source,
                sample=sample,
                skipped_sample=tuple(skipped_sample),
                skipped_source_names=skipped_source_names,
            )

        # Apply. PRAGMA foreign_keys must be off so we can update documents.id
        # (no ON UPDATE CASCADE on the FK refs). Two-phase rename via temp ids
        # avoids any "new id collides with another doc that hasn't been renamed
        # yet" ordering hazard.
        with self._conn_provider.acquire() as conn:
            conn.commit()
            conn.execute('PRAGMA foreign_keys = OFF')
            try:
                for old_id in to_delete:
                    _delete_doc_with_refs(conn, old_id)

                tmp_prefix = '_migrate_tmp:'
                for old_id, _new_id, _ct in to_rename:
                    _rename_doc_refs(conn, old_id, f'{tmp_prefix}{old_id}')

                for old_id, new_id, _ct in to_rename:
                    _rename_doc_refs(conn, f'{tmp_prefix}{old_id}', new_id)

                conn.commit()
            finally:
                conn.execute('PRAGMA foreign_keys = ON')

        if staleness_db_path is not None:
            id_map = {old: new for old, new, _ in to_rename}
            for old in to_delete:
                # losers point nowhere now; staleness will rebuild on next run
                id_map.setdefault(old, old)
            _update_staleness_doc_ids(staleness_db_path, id_map)

        return MigrateDocIdsResult(
            inspected=inspected,
            already_deterministic=already_deterministic,
            remapped=len(to_rename),
            duplicates_collapsed=len(to_delete),
            skipped_no_source=skipped_no_source,
            sample=sample,
            skipped_sample=tuple(skipped_sample),
            skipped_source_names=skipped_source_names,
        )

    def repair_stringified_locations(self, *, dry_run: bool = False) -> RepairLocationsResult:
        """Repair catalog-element ``location`` metadata that a prior export→import
        round trip turned from a dict into a Python-repr string (see the
        ``import_from_markdown`` parser in export.py).

        Parses each string ``location`` back to a dict via ``ast.literal_eval`` —
        lossless, no regeneration. Only the ``location`` key on catalog ``element``
        docs is touched; a clean dict location is left as-is. A string that does
        not parse to a dict is left untouched and reported (never silently dropped).
        """
        inspected = 0
        repaired = 0
        already_dict = 0
        unparseable: list[tuple[str, str]] = []
        updates: list[tuple[str, str]] = []

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                "SELECT id, metadata FROM documents "
                "WHERE content_type = 'catalog' AND metadata IS NOT NULL"
            ).fetchall()

        for doc_id, meta_json in rows:
            metadata = json.loads(meta_json)
            if metadata.get('kind') != CATALOG_KIND_ELEMENT:
                continue
            inspected += 1
            loc = metadata.get('location')
            if isinstance(loc, dict):
                already_dict += 1
                continue
            if not isinstance(loc, str):
                continue
            try:
                parsed = ast.literal_eval(loc)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, dict):
                metadata['location'] = parsed
                updates.append((doc_id, json.dumps(metadata)))
                repaired += 1
            else:
                unparseable.append((doc_id, loc))

        if not dry_run:
            with self._conn_provider.acquire() as conn:
                conn.commit()
                for doc_id, new_meta in updates:
                    conn.execute(
                        'UPDATE documents SET metadata = ? WHERE id = ?',
                        (new_meta, doc_id),
                    )
                conn.commit()

        return RepairLocationsResult(
            inspected=inspected,
            repaired=repaired,
            already_dict=already_dict,
            unparseable=len(unparseable),
            unparseable_sample=tuple(unparseable[:20]),
        )


__all__ = [
    'MigrateDocIdsResult',
    'MigrationsMixin',
    'RepairLocationsResult',
]
