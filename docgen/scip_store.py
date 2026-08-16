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
