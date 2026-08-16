"""Persisting the graph — the step the wiring gate inspects.

``docgen/scip_index.py`` decides what SCIP means, ``docgen/scip_graph.py`` projects it into
rows, and this writes those rows down. Keeping it separate matters because this is where
two historical failures lived, both invisible at the time:

* **Scope of the delete.** Re-inserting one source while deleting globally leaves a
  multi-source store holding only the last build — the shape ``92b1d40`` fixed for
  ``doc_graph``'s ``imports``, and the same trap here.
* **Order of the delete.** ``scip_edges`` has no ``source_name``, so clearing one source's
  edges means joining through ``scip_symbols``. That join has to run *before* the symbols
  are removed, or it matches nothing and the edges are left orphaned.

Both are pinned by tests rather than by comments.
"""
from __future__ import annotations
from dataclasses import dataclass

from typing import TYPE_CHECKING

from docgen.scip_graph import CrossSourceEdge, CrossSourceSymbol, GraphRows

if TYPE_CHECKING:  # pragma: no cover - typing only
    from sqlite3 import Connection
_SYMBOL_COLUMNS = (
    'canonical_id', 'source_name', 'language', 'file', 'line_start', 'line_end',
    'kind', 'display_name', 'qualified_name', 'parent_qualified_name',
)
_EDGE_COLUMNS = (
    'caller_canonical_id', 'callee_canonical_id', 'edge_type', 'file', 'line',
    'confidence',
)


def clear_source(conn: 'Connection', source_name: str) -> None:
    """Remove one source's rows, edges first.

    Edges are deleted through a join on the symbols that are about to go, so the order is
    load-bearing: reverse it and the join finds nothing.
    """
    owned = ('SELECT canonical_id FROM scip_symbols WHERE source_name = ?')
    conn.execute(
        f'DELETE FROM scip_edges WHERE caller_canonical_id IN ({owned}) '
        f'OR callee_canonical_id IN ({owned})',
        (source_name, source_name),
    )
    conn.execute('DELETE FROM scip_symbols WHERE source_name = ?', (source_name,))
def save_rows(conn: 'Connection', rows: GraphRows, *, source_name: str) -> None:
    """Replace ``source_name``'s symbols and edges with ``rows``.

    Scoped to the one source: another source's rows are never touched.
    """
    clear_source(conn, source_name)

    symbol_sql = (
        f'INSERT OR REPLACE INTO scip_symbols ({", ".join(_SYMBOL_COLUMNS)}) '
        f'VALUES ({", ".join("?" * len(_SYMBOL_COLUMNS))})'
    )
    conn.executemany(symbol_sql, [
        (s.canonical_id, s.source_name, s.language, s.file, s.line_start,
         s.line_end, s.kind, s.display_name, s.qualified_name,
         s.parent_qualified_name)
        for s in rows.symbols.values()
    ])

    edge_sql = (
        f'INSERT OR REPLACE INTO scip_edges ({", ".join(_EDGE_COLUMNS)}) '
        f'VALUES ({", ".join("?" * len(_EDGE_COLUMNS))})'
    )
    conn.executemany(edge_sql, [
        (e.caller.canonical_id, e.callee.canonical_id, e.edge_type, e.file,
         e.line, e.confidence)
        for e in rows.edges
    ])
    conn.commit()
def load_rows(conn: 'Connection',
              source_name: str | None = None) -> GraphRows:
    """Read symbols and edges back, optionally for one source only.

    An edge whose endpoints are not both present is skipped and counted rather than
    returned with a hole in it.
    """
    where, args = '', ()
    if source_name is not None:
        where, args = ' WHERE source_name = ?', (source_name,)

    symbols: dict = {}
    for row in conn.execute(
            f'SELECT {", ".join(_SYMBOL_COLUMNS)} FROM scip_symbols{where}', args):
        symbols[row[0]] = CrossSourceSymbol(*row)

    edges: list = []
    dangling = 0
    for caller_id, callee_id, edge_type, file, line, confidence in conn.execute(
            f'SELECT {", ".join(_EDGE_COLUMNS)} FROM scip_edges'):
        caller = symbols.get(caller_id)
        callee = symbols.get(callee_id)
        if caller is None or callee is None:
            dangling += 1
            continue
        edges.append(CrossSourceEdge(
            caller=caller, callee=callee, edge_type=edge_type, file=file,
            line=line, confidence=confidence))

    return GraphRows(symbols=symbols, edges=edges, unresolved_callees=dangling)


@dataclass(frozen=True)
class OwnershipBackfillResult:
    """Counts from a canonical ownership reconciliation."""

    scanned_symbols: int
    candidate_edges: int
    removed_edges: int
    inserted_edges: int


def backfill_canonical_ownership(
    conn: "Connection", source_name: str, *, batch_size: int = 1000,
) -> OwnershipBackfillResult:
    """Reconstruct exact ownership edges from symbols already in a SCIP store.

    This is the spool repair path: a frozen pack may predate ownership-edge
    materialization even though its canonical symbol IDs already encode exact
    descriptor parents. The pass is bounded, source-scoped, and idempotent. It
    never needs a checkout, an indexer, or an LLM.
    """
    if batch_size < 1:
        raise ValueError("batch_size must be positive")

    from docgen.scip_descriptors import (
        _enclosing_symbol_from_symbol,
        _symbol_descriptor_kind,
    )

    candidate_table = "_scip_canonical_ownership"
    savepoint = "_scip_canonical_ownership_backfill"
    scanned_symbols = 0
    candidate_edges = 0
    removed_edges = 0
    inserted_edges = 0
    conn.execute(f"SAVEPOINT {savepoint}")
    try:
        conn.execute(f"DROP TABLE IF EXISTS temp.{candidate_table}")
        conn.execute(
            f"CREATE TEMP TABLE {candidate_table} ("
            "parent_id TEXT NOT NULL, child_id TEXT NOT NULL, "
            "file TEXT NOT NULL, line INTEGER NOT NULL, "
            "PRIMARY KEY (parent_id, child_id, file, line)) WITHOUT ROWID"
        )

        cursor = conn.execute(
            "SELECT canonical_id, file, line_start FROM scip_symbols "
            "WHERE source_name = ?",
            (source_name,),
        )
        while True:
            rows = cursor.fetchmany(batch_size)
            if not rows:
                break
            scanned_symbols += len(rows)
            candidates = []
            for child_id, file, line in rows:
                parent_id = _enclosing_symbol_from_symbol(child_id)
                if (
                    parent_id
                    and parent_id != child_id
                    and _symbol_descriptor_kind(parent_id)
                    in {"type", "term", "method"}
                ):
                    candidates.append((parent_id, child_id, file, line))
            if candidates:
                conn.executemany(
                    f"INSERT OR IGNORE INTO {candidate_table} "
                    "(parent_id, child_id, file, line) VALUES (?,?,?,?)",
                    candidates,
                )

        conn.execute(
            f"DELETE FROM {candidate_table} WHERE NOT EXISTS ("
            "SELECT 1 FROM scip_symbols AS parent "
            "JOIN scip_symbols AS child ON child.canonical_id = "
            f"{candidate_table}.child_id "
            f"WHERE parent.canonical_id = {candidate_table}.parent_id "
            "AND parent.source_name = ? AND child.source_name = ?)",
            (source_name, source_name),
        )
        candidate_edges = int(
            conn.execute(
                f"SELECT COUNT(*) FROM {candidate_table}"
            ).fetchone()[0]
        )

        owned = (
            "SELECT canonical_id FROM scip_symbols WHERE source_name = ?"
        )
        removed = conn.execute(
            "DELETE FROM scip_edges WHERE edge_type = 'contains' "
            f"AND (caller_canonical_id IN ({owned}) "
            f"OR callee_canonical_id IN ({owned})) "
            f"AND NOT EXISTS (SELECT 1 FROM {candidate_table} AS desired "
            "WHERE desired.parent_id = scip_edges.caller_canonical_id "
            "AND desired.child_id = scip_edges.callee_canonical_id "
            "AND desired.file = scip_edges.file "
            "AND desired.line = scip_edges.line)",
            (source_name, source_name),
        )
        removed_edges = max(0, int(removed.rowcount))

        conn.execute(
            "UPDATE scip_edges SET confidence = 'exact' "
            "WHERE edge_type = 'contains' "
            f"AND EXISTS (SELECT 1 FROM {candidate_table} AS desired "
            "WHERE desired.parent_id = scip_edges.caller_canonical_id "
            "AND desired.child_id = scip_edges.callee_canonical_id "
            "AND desired.file = scip_edges.file "
            "AND desired.line = scip_edges.line)"
        )
        inserted = conn.execute(
            "INSERT OR IGNORE INTO scip_edges ("
            "caller_canonical_id, callee_canonical_id, edge_type, "
            "file, line, confidence) "
            f"SELECT parent_id, child_id, 'contains', file, line, 'exact' "
            f"FROM {candidate_table}"
        )
        inserted_edges = max(0, int(inserted.rowcount))
        conn.execute(f"DROP TABLE temp.{candidate_table}")
    except Exception:
        conn.execute(f"ROLLBACK TO {savepoint}")
        conn.execute(f"RELEASE {savepoint}")
        raise
    conn.execute(f"RELEASE {savepoint}")
    return OwnershipBackfillResult(
        scanned_symbols=scanned_symbols,
        candidate_edges=candidate_edges,
        removed_edges=removed_edges,
        inserted_edges=inserted_edges,
    )
