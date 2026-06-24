"""Referential integrity for the data model (§9a). The binders keep
``data_access → schema_symbols`` consistent by construction (Layer 2 resolves
against Layer 1; the raw-SQL binder creates the nodes it references), and the
read path skips orphans — but the invariant was unspecified/unenforced. So
``persist_data_model`` runs a final-state check: a ``data_access`` row whose
``schema_symbol_id`` matches no ``schema_symbols`` row is surfaced as a
``data_model_gap``, never a silent read-time skip — the safety net that makes a
binder regression visible."""
from __future__ import annotations

from docgen.scip_persist import persist_data_model
from library import Library


def _gaps_for(db, source: str) -> list[str]:
    library = Library(db)
    try:
        with library._conn_provider.acquire() as conn:
            return [r[0] for r in conn.execute(
                'SELECT detail FROM data_model_gaps WHERE source_name = ?',
                (source,))]
    finally:
        library.close()


def test_persist_data_model_surfaces_orphan_data_access(tmp_path) -> None:
    """An orphan data_access row (no matching schema_symbol) → a gap naming it."""
    db = tmp_path / 'ariadne.db'
    library = Library(db)
    try:
        with library._conn_provider.acquire() as conn:
            conn.execute(
                "INSERT INTO data_access (source_name, consumer_symbol_id, "
                "schema_symbol_id, role, witness, confidence) VALUES "
                "('s', 'c', 'data sql s _._.ghost#col', 'filter', 'manual', "
                "'resolved')")
            conn.commit()
    finally:
        library.close()
    # No manifest at tmp_path → the ORM binders are skipped, but the integrity
    # check still runs over the final persisted state.
    persist_data_model(db, [('s', str(tmp_path))])
    gaps = _gaps_for(db, 's')
    assert any('ghost' in g and 'integrity' in g.lower() for g in gaps)
