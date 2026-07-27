"""Themes mixin for Library — community-detection theme persistence (Themes plan, Phase 1).

Defines the three new tables (themes, theme_members, cluster_history), their
schema constants for use by Library.__attrs_post_init__, and the ThemesMixin
that exposes CRUD over them.

Theme docs themselves are stored in the existing `documents` table with
content_type='theme'; this mixin persists the cluster-level metadata that
points at those docs and tracks cluster membership / summarization state.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from attrs import frozen

from schema import _now_iso

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@frozen
class Theme:
    """A discovered cluster of catalog elements with a synthesized theme doc.

    Fields mirror the `themes` table; `coherent` is False when the LLM
    judged the cluster too noisy to summarize, and `dirty` is True when
    the cluster's membership has shifted since the last summary.
    """
    cluster_id: str
    doc_id: str
    member_count: int
    resolution: float
    last_built_at: str
    last_summarized_at: str
    summary_hash: str
    coherent: bool = True
    dirty: bool = False


@frozen
class ClusterMapping:
    """Maps a (run_id, cluster_id) to its predecessor for stability tracking.

    `prev_cluster_id` is None for clusters with no Jaccard match (genuinely new).
    `overlap_ratio` is the Jaccard score against the predecessor; None if no match.
    """
    cluster_id: str
    prev_cluster_id: str | None
    overlap_ratio: float | None


# ---------------------------------------------------------------------------
# Schema (referenced by Library.__attrs_post_init__)
# ---------------------------------------------------------------------------


_THEMES_SCHEMA = '''
CREATE TABLE IF NOT EXISTS themes (
    cluster_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL UNIQUE,
    member_count INTEGER NOT NULL,
    resolution REAL NOT NULL,
    last_built_at TEXT NOT NULL,
    last_summarized_at TEXT NOT NULL,
    summary_hash TEXT NOT NULL,
    coherent INTEGER NOT NULL DEFAULT 1,
    dirty INTEGER NOT NULL DEFAULT 0,
    association TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (doc_id) REFERENCES documents(id) ON DELETE CASCADE
)
'''

_THEME_MEMBERS_SCHEMA = '''
CREATE TABLE IF NOT EXISTS theme_members (
    cluster_id TEXT NOT NULL,
    element_id TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    joined_at TEXT NOT NULL,
    PRIMARY KEY (cluster_id, element_id),
    FOREIGN KEY (cluster_id) REFERENCES themes(cluster_id) ON DELETE CASCADE,
    FOREIGN KEY (element_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_theme_members_element ON theme_members(element_id);
CREATE INDEX IF NOT EXISTS idx_theme_members_cluster ON theme_members(cluster_id);
'''

_CLUSTER_HISTORY_SCHEMA = '''
CREATE TABLE IF NOT EXISTS cluster_history (
    run_id INTEGER NOT NULL,
    cluster_id TEXT NOT NULL,
    prev_cluster_id TEXT,
    overlap_ratio REAL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, cluster_id)
);
CREATE INDEX IF NOT EXISTS idx_cluster_history_prev ON cluster_history(prev_cluster_id);
'''
_THEME_SYNCED_HASHES_SCHEMA = '''
CREATE TABLE IF NOT EXISTS theme_synced_hashes (
    element_id TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL,
    FOREIGN KEY (element_id) REFERENCES documents(id) ON DELETE CASCADE
)
'''

# Per spool cross-check association (project × spool): the content signature of
# its last reconcile, so reconcile can SKIP re-clustering an unchanged
# association instead of rebuilding the (large) spool semantic index once per
# opted-in project (HIGH-A).
_SPOOL_ASSOC_SYNC_SCHEMA = '''
CREATE TABLE IF NOT EXISTS spool_assoc_sync (
    association TEXT PRIMARY KEY,
    content_hash TEXT NOT NULL
)
'''


# ---------------------------------------------------------------------------
# Mixin
# ---------------------------------------------------------------------------


class ThemesMixin:
    """CRUD over themes, theme_members, and cluster_history.

    Expects the composed class to provide:
    - self._conn_provider: _ConnectionProvider
    """

    # -- Theme CRUD --------------------------------------------------------

    def add_theme(
        self,
        *,
        cluster_id: str,
        doc_id: str,
        member_count: int,
        resolution: float,
        summary_hash: str,
        coherent: bool = True,
        dirty: bool = True,
        association: str = '',
    ) -> None:
        """Insert a new theme row.

        Defaults to dirty=True since newly created themes typically need a
        summarization pass; pass dirty=False for themes already synced.

        association names the clustering pass that owns this theme (''
        for the base global pass, a canonical scope key for a scoped
        spool pass); reconcile diffs stay within one association so
        independent passes don't delete each other's themes.
        """
        now = _now_iso()
        with self._conn_provider.acquire() as conn:
            conn.execute(
                '''INSERT INTO themes
                   (cluster_id, doc_id, member_count, resolution,
                    last_built_at, last_summarized_at, summary_hash,
                    coherent, dirty, association)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (
                    cluster_id, doc_id, member_count, resolution,
                    now, now, summary_hash,
                    int(coherent), int(dirty), association,
                ),
            )

    def get_theme(self, cluster_id: str) -> Theme | None:
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                '''SELECT cluster_id, doc_id, member_count, resolution,
                          last_built_at, last_summarized_at, summary_hash,
                          coherent, dirty
                   FROM themes WHERE cluster_id = ?''',
                (cluster_id,),
            ).fetchone()
        return self._row_to_theme(row) if row else None

    def list_themes(
        self,
        *,
        coherent_only: bool = True,
        source: str | None = None,
        source_names: tuple[str, ...] | None = None,
        association: str | None = None,
    ) -> list[Theme]:
        """List themes ordered by cluster_id.

        coherent_only filters out themes the LLM marked INCOHERENT.
        source filters by documents.source_name (single-value variant).
        source_names is the multi-value variant used by the closure
        wrapper — restrict themes whose summary doc has a source_name
        in the given tuple. source and source_names are mutually
        exclusive at the call site.
        association restricts to themes owned by one clustering pass
        (the reconcile partition; '' is the base global pass). It reads
        the themes.association column directly, so it needs no join and
        composes with the other filters.
        """
        needs_join = source is not None or source_names is not None
        sql = '''SELECT t.cluster_id, t.doc_id, t.member_count, t.resolution,
                        t.last_built_at, t.last_summarized_at, t.summary_hash,
                        t.coherent, t.dirty
                 FROM themes t'''
        if needs_join:
            sql += ' JOIN documents d ON d.id = t.doc_id'

        conditions: list[str] = []
        params: list[object] = []
        if coherent_only:
            conditions.append('t.coherent = 1')
        if source is not None:
            conditions.append('d.source_name = ?')
            params.append(source)
        if source_names is not None:
            if not source_names:
                return []
            placeholders = ','.join('?' * len(source_names))
            # Themes are cross-source by design (their summary doc is
            # written with source_name=NULL by cluster.py — see
            # ``docgen/themes.py`` module docstring). Admit both NULL
            # (unscoped, cross-source) and in-closure (explicitly tagged
            # per-source) summary docs.
            conditions.append(
                f'(d.source_name IS NULL OR d.source_name IN ({placeholders}))',
            )
            params.extend(source_names)
        if association is not None:
            conditions.append('t.association = ?')
            params.append(association)
        if conditions:
            sql += ' WHERE ' + ' AND '.join(conditions)
        sql += ' ORDER BY t.cluster_id'

        with self._conn_provider.acquire() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [self._row_to_theme(r) for r in rows]

    def _theme_owning_doc_id(self, cluster_id: str) -> str | None:
        """Return the summary doc_id for a theme, or None if missing."""
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                'SELECT doc_id FROM themes WHERE cluster_id = ?',
                (cluster_id,),
            ).fetchone()
        return None if row is None else str(row[0])

    def delete_theme(self, cluster_id: str) -> None:
        """Delete a theme. Cascades to theme_members via FK."""
        with self._conn_provider.acquire() as conn:
            conn.execute('DELETE FROM themes WHERE cluster_id = ?', (cluster_id,))

    def distinct_theme_associations(self) -> list[str]:
        """Distinct non-empty association keys — the scoped-pass partitions.

        The base global pass owns association='' and is excluded; each
        remaining value identifies one spool cross-check partition.
        """
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                "SELECT DISTINCT association FROM themes WHERE association != '' "
                'ORDER BY association',
            ).fetchall()
        return [str(r[0]) for r in rows]

    def delete_themes_for_association(self, association: str) -> None:
        """Delete every theme in a partition, summary docs included.

        Deletes through each summary document so the cascade (documents →
        themes → theme_members) clears the theme rows and their members too.
        Also clears the reconcile sync signature so a later re-add rebuilds.
        """
        for theme in self.list_themes(coherent_only=False, association=association):
            self.delete_document(theme.doc_id)
        with self._conn_provider.acquire() as conn:
            conn.execute(
                'DELETE FROM spool_assoc_sync WHERE association = ?',
                (association,),
            )

    def get_spool_assoc_hash(self, association: str) -> str | None:
        """The content signature of a spool association's last reconcile, or
        None. Lets reconcile SKIP re-clustering an unchanged association
        (HIGH-A) rather than rebuild the spool index per project."""
        with self._conn_provider.acquire() as conn:
            row = conn.execute(
                'SELECT content_hash FROM spool_assoc_sync WHERE association = ?',
                (association,),
            ).fetchone()
        return None if row is None else str(row[0])

    def set_spool_assoc_hash(self, association: str, content_hash: str) -> None:
        with self._conn_provider.acquire() as conn:
            conn.execute(
                'INSERT OR REPLACE INTO spool_assoc_sync '
                '(association, content_hash) VALUES (?, ?)',
                (association, content_hash),
            )

    # -- Membership --------------------------------------------------------

    def set_theme_members(
        self,
        cluster_id: str,
        members: list[tuple[str, float]],
    ) -> None:
        """Replace the member set for a cluster.

        members: list of (element_id, weight) pairs. Empty list clears membership.
        """
        now = _now_iso()
        with self._conn_provider.acquire() as conn:
            conn.execute(
                'DELETE FROM theme_members WHERE cluster_id = ?',
                (cluster_id,),
            )
            if members:
                conn.executemany(
                    '''INSERT INTO theme_members
                       (cluster_id, element_id, weight, joined_at)
                       VALUES (?, ?, ?, ?)''',
                    [(cluster_id, eid, weight, now) for eid, weight in members],
                )

    def get_theme_members(self, cluster_id: str) -> list[tuple[str, float]]:
        """Return (element_id, weight) pairs for a cluster, ordered by element_id."""
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                '''SELECT element_id, weight FROM theme_members
                   WHERE cluster_id = ? ORDER BY element_id''',
                (cluster_id,),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def get_themes_for_element(self, element_id: str) -> list[str]:
        """Return cluster_ids the element belongs to (an element can be in multiple)."""
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                '''SELECT cluster_id FROM theme_members
                   WHERE element_id = ? ORDER BY cluster_id''',
                (element_id,),
            ).fetchall()
        return [row[0] for row in rows]

    # -- Dirty / clean tracking -------------------------------------------

    def mark_theme_dirty(self, cluster_id: str) -> None:
        with self._conn_provider.acquire() as conn:
            conn.execute(
                'UPDATE themes SET dirty = 1 WHERE cluster_id = ?',
                (cluster_id,),
            )

    def mark_themes_clean(self, cluster_ids: list[str]) -> None:
        if not cluster_ids:
            return
        placeholders = ','.join('?' * len(cluster_ids))
        with self._conn_provider.acquire() as conn:
            conn.execute(
                f'UPDATE themes SET dirty = 0 WHERE cluster_id IN ({placeholders})',
                cluster_ids,
            )

    def get_dirty_themes(self) -> list[str]:
        with self._conn_provider.acquire() as conn:
            rows = conn.execute(
                'SELECT cluster_id FROM themes WHERE dirty = 1 ORDER BY cluster_id'
            ).fetchall()
        return [row[0] for row in rows]

    def update_summary_hash(self, cluster_id: str, summary_hash: str) -> None:
        """Update summary_hash and bump last_summarized_at."""
        now = _now_iso()
        with self._conn_provider.acquire() as conn:
            conn.execute(
                '''UPDATE themes
                   SET summary_hash = ?, last_summarized_at = ?
                   WHERE cluster_id = ?''',
                (summary_hash, now, cluster_id),
            )

    # -- Cluster history --------------------------------------------------

    def record_cluster_history(
        self,
        run_id: int,
        mappings: list[ClusterMapping],
    ) -> None:
        """Persist (run_id, cluster_id, prev_cluster_id, overlap_ratio) records."""
        if not mappings:
            return
        now = _now_iso()
        with self._conn_provider.acquire() as conn:
            conn.executemany(
                '''INSERT OR REPLACE INTO cluster_history
                   (run_id, cluster_id, prev_cluster_id, overlap_ratio, created_at)
                   VALUES (?, ?, ?, ?, ?)''',
                [
                    (run_id, m.cluster_id, m.prev_cluster_id, m.overlap_ratio, now)
                    for m in mappings
                ],
            )

    def latest_cluster_run(self) -> int | None:
        """Return the highest run_id in cluster_history, or None if empty."""
        with self._conn_provider.acquire() as conn:
            row = conn.execute('SELECT MAX(run_id) FROM cluster_history').fetchone()
        return row[0] if row and row[0] is not None else None

    # -- Internal ---------------------------------------------------------

    @staticmethod
    def _row_to_theme(row: tuple) -> Theme:
        return Theme(
            cluster_id=row[0],
            doc_id=row[1],
            member_count=row[2],
            resolution=row[3],
            last_built_at=row[4],
            last_summarized_at=row[5],
            summary_hash=row[6],
            coherent=bool(row[7]),
            dirty=bool(row[8]),
        )


__all__ = [
    '_CLUSTER_HISTORY_SCHEMA',
    '_THEMES_SCHEMA',
    '_THEME_MEMBERS_SCHEMA',
    'ClusterMapping',
    'Theme',
    'ThemesMixin',
]
