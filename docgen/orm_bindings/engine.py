"""The shared ORM-binding engine (design §5).

Owns the invariant pipeline: iterate SCIP definition occurrences into a
``(file, line) -> canonical_id`` anchor map, run each detected strategy's
discovery, resolve FK targets to their tables, and emit uniform
``schema_symbols`` rows — recording unresolved bindings as surfaced gaps
(§5.0.1 #5), never silent. Per-ORM logic lives only in the strategies.
"""
from __future__ import annotations
import ast
from pathlib import Path

from typing import TYPE_CHECKING, Protocol, runtime_checkable

from attrs import frozen
from ast_utils import safe_ast_parse

if TYPE_CHECKING:
    from sqlite3 import Connection

    from docgen.scip_extractor import ScipIndex


@runtime_checkable
class OrmStrategy(Protocol):
    """The contract for a per-ORM strategy on the shared binding engine (design
    §5.0). To add an ORM: implement these four members and register the strategy
    in ``DEFAULT_STRATEGIES`` (``docgen/orm_bindings/__init__.py``) — the engine
    does everything else (SCIP anchoring, FK resolution, owning-symbol mapping,
    the derived->resolved promotion, uniform row emission, gap collection), so no
    engine change is needed. Strategies may be active simultaneously (a repo can
    mix ORMs); each tags its rows with its own ``name``.

    - ``name``: provenance tag, e.g. ``'orm:django'`` — stored as
      ``resolution_source`` / ``witness`` on this strategy's rows.
    - ``detect(scip_index, root) -> bool``: is this ORM present in the source?
      (cheap; gates the rest).
    - ``discover(scip_index, root, symbol_at) -> ([Table], gaps)``: Layer 1
      (structural) — the tables this ORM declares, each with its ``Col`` columns
      anchored on a SCIP producer symbol, plus surfaced gaps. ``symbol_at`` maps
      ``(relative_path, 0-indexed def line) -> canonical_id``.
    - ``recognize(scip_index, root, owning, schema, aliases) -> (rows, gaps)``:
      Layer 2 (access) — role-typed rows ``(consumer_symbol_id, table, column,
      role, file, line)`` for this ORM's query call sites, plus gaps.
      ``owning(path, 0-indexed line) -> enclosing symbol``; ``schema`` is
      ``{model: (table, {field: column})}`` from Layer 1; ``aliases`` is
      ``{relative_path: {local_name: imported_name}}`` so a query naming a model
      under an ``import … as`` alias resolves before the schema lookup.
    """

    name: str

    def detect(self, scip_index: 'ScipIndex', root) -> bool:
        ...

    def discover(self, scip_index: 'ScipIndex', root,
                 symbol_at: dict) -> tuple[list['Table'], tuple[str, ...]]:
        ...

    def recognize(self, scip_index: 'ScipIndex', root, owning,
                  schema: dict, aliases: dict) -> tuple[list, tuple[str, ...]]:
        ...


@frozen
class SchemaBindResult:
    """Outcome of ``persist_schema_symbols``: how many ``schema_symbols`` rows
    were written, and the bindings that could NOT be resolved — a recorded
    gap, never silently dropped (§5.0.1 #5)."""
    nodes_written: int
    gaps: tuple


def _symbol_at(scip_index: 'ScipIndex') -> dict:
    """``(relative_path, 0-indexed def line) -> canonical_id`` for every
    definition occurrence — the SCIP anchor a strategy resolves against
    (§5.0.1 #1), so there is no second parse to drift."""
    at = {}
    for doc in scip_index.documents:
        for occ in doc.occurrences:
            if occ.is_definition and occ.range:
                at[(doc.relative_path, occ.range[0])] = occ.symbol
    return at


def persist_schema_symbols(
    conn: 'Connection', source_name: str, scip_index: 'ScipIndex', *, strategies,
) -> SchemaBindResult:
    """Discover ORM models from the SCIP index and (re)write their
    ``schema_symbols`` (idempotent per source+strategy). The engine owns
    iteration, FK-target resolution, and emission; each strategy supplies only
    per-ORM discovery. Unresolved bindings (no SCIP anchor, unparsable source,
    FK target not found) are returned as surfaced gaps, never silent (§5.0.1
    #5)."""
    symbol_at = _symbol_at(scip_index)
    root = scip_index.source_root
    written = 0
    gaps: list[str] = []
    for strategy in strategies:
        if not strategy.detect(scip_index, root):
            continue
        conn.execute(
            'DELETE FROM schema_symbols WHERE source_name = ? AND resolution_source = ?',
            (source_name, strategy.name),
        )
        tables, strat_gaps = strategy.discover(scip_index, root, symbol_at)
        gaps.extend(strat_gaps)
        # model class name -> its table canonical id, for FK references_id.
        model_table = {
            t.model_name: f'data sql {source_name} _._.{t.table_name}'
            for t in tables
        }
        rows = []
        for t in tables:
            table_id = f'data sql {source_name} _._.{t.table_name}'
            rows.append((
                table_id, source_name, 'table', t.table_name, None, None,
                t.producer_symbol_id, strategy.name, t.confidence,
            ))
            for c in t.columns:
                references_id = None
                if c.fk_target_model is not None:
                    references_id = model_table.get(c.fk_target_model)
                    if references_id is None:
                        gaps.append(
                            f'{strategy.name}: {t.model_name}.{c.column_name} '
                            f'-> FK target {c.fk_target_model!r} not resolved'
                        )
                rows.append((
                    f'{table_id}#{c.column_name}', source_name, 'column',
                    t.table_name, c.column_name, references_id,
                    c.producer_symbol_id, strategy.name, c.confidence,
                ))
        conn.executemany(
            'INSERT OR IGNORE INTO schema_symbols '
            '(canonical_id, source_name, node_type, table_name, column_name, '
            ' references_id, producer_symbol_id, resolution_source, confidence) '
            'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
            rows,
        )
        written += len(rows)
    return SchemaBindResult(nodes_written=written, gaps=tuple(gaps))


@frozen
class Col:
    column_name: str
    producer_symbol_id: str
    confidence: str
    fk_target_model: str | None = None
    field_name: str = ''


@frozen
class Table:
    model_name: str
    table_name: str
    producer_symbol_id: str
    confidence: str
    columns: tuple


def _parse(root, relative_path):
    try:
        return safe_ast_parse((Path(root) / relative_path).read_text())
    except (OSError, SyntaxError, ValueError):
        return None  # unreadable / non-Python -> a gap, not a crash (§7)


def _str_const(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None
