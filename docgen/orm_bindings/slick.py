"""Slick ORM strategy (design §5.4) — Scala, scip-java.

Slick declares the schema with explicit string literals: a table is a class
``extends Table[Row](tag, "name")`` (optionally ``(tag, Some("schema"),
"name")``), each column a ``def c = column[T]("name")``. Both names are explicit
literals — there is no naming rule to derive — so every binding is ``exact``.
Discovery anchors ``producer_symbol_id`` on the SCIP def occurrence of the Table
subclass / column def; the literal names are read from the Scala source with
ast-grep (tree-sitter-scala). Unreadable sources / un-anchored / un-named
definitions are surfaced as gaps, never silently dropped (§5.0.1 #5).

Slick is the design's Layer-1-strong case (§5.4): its DSL references columns as
real ``def`` symbols, so SCIP references resolve and the Layer-2 recognizer adds
only the role.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ast_grep_py import SgRoot

from docgen.orm_bindings.engine import Col, Table

if TYPE_CHECKING:
    from docgen.scip_extractor import ScipIndex

_SCALA_EXTS = ('.scala', '.sbt')


class SlickStrategy:
    """Slick's ``schema_spec`` (§5.4), as discovery over the Scala source."""

    name = 'orm:slick'

    def detect(self, scip_index, root) -> bool:
        for doc in scip_index.documents:
            tree = _parse_scala(root, doc.relative_path)
            if tree is not None and _table_classes(tree):
                return True
        return False

    def discover(self, scip_index, root, symbol_at):
        tables, gaps = [], []
        for doc in scip_index.documents:
            tree = _parse_scala(root, doc.relative_path)
            if tree is None:
                continue
            for cls in _table_classes(tree):
                name_node = _child(cls, 'identifier')
                model = name_node.text()
                producer = symbol_at.get((doc.relative_path, _line(name_node)))
                if producer is None:
                    gaps.append(f'{self.name}: table {model} has no SCIP anchor')
                    continue
                table_name = _table_name(cls)
                if table_name is None:
                    gaps.append(f'{self.name}: table {model} has no name literal')
                    continue
                cols, col_gaps = _columns(
                    cls, doc.relative_path, symbol_at, model, self.name)
                gaps.extend(col_gaps)
                tables.append(Table(model, table_name, producer, 'exact', tuple(cols)))
        return tables, tuple(gaps)

    def recognize(self, scip_index, root, owning, schema, aliases):
        """Slick Layer-2 recognition (§5.4): add the role to the column refs.
        The engine's ``aliases`` map covers only Python ``import … as`` and never
        applies to Scala source; Slick extracts Scala import renames
        (``import x.{A => B}``) from its own ast-grep tree (``_scala_import_aliases``)
        and resolves the receiver's Table class through them before the lookup."""
        return _slick_recognize(scip_index, root, owning, schema, self.name)


def _parse_scala(root, rel_path):
    if not rel_path.endswith(_SCALA_EXTS):
        return None  # not a Scala source — another strategy / witness handles it
    try:
        text = (Path(root) / rel_path).read_text()
    except OSError:
        return None  # unreadable -> skipped, not a crash (§7)
    return SgRoot(text, 'scala').root()


def _child(node, kind):
    for c in node.children():
        if c.kind() == kind:
            return c
    return None


def _line(node):
    return node.range().start.line


def _table_classes(tree):
    return [c for c in tree.find_all(kind='class_definition') if _extends_table(c)]


def _extends_table(cls):
    """True iff the class ``extends Table[...]`` (a Slick table)."""
    ext = _child(cls, 'extends_clause')
    generic = _child(ext, 'generic_type') if ext is not None else None
    ti = _child(generic, 'type_identifier') if generic is not None else None
    return ti is not None and ti.text() == 'Table'


def _table_name(cls):
    """The table name = the (bare) string literal in ``Table[...](tag, "name")``;
    a ``Some("schema")`` schema arg is a call, not a bare string, so it is
    naturally excluded."""
    args = _child(_child(cls, 'extends_clause'), 'arguments')
    return _first_string(args) if args is not None else None


def _columns(cls, rel_path, symbol_at, model, witness):
    cols, gaps = [], []
    for name_node, call in _column_defs(cls):
        field = name_node.text()
        col_name = _first_string(_child(call, 'arguments'))
        if col_name is None:
            gaps.append(f'{witness}: column {model}.{field} has no name literal')
            continue
        producer = symbol_at.get((rel_path, _line(name_node)))
        if producer is None:
            gaps.append(f'{witness}: column {model}.{field} has no SCIP anchor')
            continue
        cols.append(Col(col_name, producer, 'exact', field_name=field))
    return cols, gaps


def _column_defs(cls):
    """``(def-name node, column call)`` for each ``def c = column[T](...)`` in
    the class body; the ``def *`` projection and non-``column`` defs are skipped."""
    body = _child(cls, 'template_body')
    if body is None:
        return []
    out = []
    for fdef in body.children():
        if fdef.kind() != 'function_definition':
            continue
        name_node = _child(fdef, 'identifier')
        call = _child(fdef, 'call_expression')
        if name_node is not None and call is not None and _is_column_call(call):
            out.append((name_node, call))
    return out


def _is_column_call(call):
    gf = _child(call, 'generic_function')
    ident = _child(gf, 'identifier') if gf is not None else None
    return ident is not None and ident.text() == 'column'


def _first_string(args):
    for c in args.children():
        if c.kind() == 'string':
            return c.text().strip('"')
    return None


# --- Layer 2 — access binding (§5.4 Access API) ----------------------------
# Slick query verbs that name columns via `_.col` lambdas -> role.
_SLICK_VERBS = {'filter': 'filter', 'withFilter': 'filter',
                'map': 'project', 'sortBy': 'order'}
_INSERT_OPS = {'+=', '++='}


def _slick_recognize(scip_index, root, owning, schema, witness):
    rows, gaps = [], []
    for doc in scip_index.documents:
        tree = _parse_scala(root, doc.relative_path)
        if tree is None:
            continue
        tq = _tablequery_map(tree)
        aliases = _scala_import_aliases(tree)
        for call in tree.find_all(kind='call_expression'):
            r, g = _decode_verb(call, doc.relative_path, owning, schema, tq, aliases, witness)
            rows.extend(r)
            gaps.extend(g)
        for infix in tree.find_all(kind='infix_expression'):
            r, g = _decode_insert(infix, doc.relative_path, owning, schema, tq, aliases, witness)
            rows.extend(r)
            gaps.extend(g)
    return rows, gaps


def _scala_import_aliases(tree):
    """``{local_name: imported_name}`` for each ``import pkg.{Orig => Local}``
    rename in the file — so a query naming a Table class under a Scala import
    alias resolves to the declared class before the schema lookup (§5.0). A hide
    selector (``{Orig => _}``) renames to a wildcard, not a class, so it binds no
    alias. This is Slick's own Scala-aware alias source; the engine's ``aliases``
    map covers only Python ``import … as`` and never applies to Scala source."""
    out = {}
    for ren in tree.find_all(kind='arrow_renamed_identifier'):
        idents = [c for c in ren.children() if c.kind() == 'identifier']
        if len(idents) == 2:
            out[idents[1].text()] = idents[0].text()
    return out


def _tablequery_map(tree):
    """``val x = TableQuery[Y]`` bindings -> ``{x: 'Y'}`` for receiver resolution."""
    out = {}
    for v in tree.find_all(kind='val_definition'):
        name = _child(v, 'identifier')
        cls = _tablequery_class(_child(v, 'generic_function'))
        if name is not None and cls is not None:
            out[name.text()] = cls
    return out


def _tablequery_class(node):
    """``'Y'`` if ``node`` is ``TableQuery[Y]``, else None."""
    if node is None or node.kind() != 'generic_function':
        return None
    ident = _child(node, 'identifier')
    if ident is None or ident.text() != 'TableQuery':
        return None
    ti = _child(_child(node, 'type_arguments'), 'type_identifier')
    return ti.text() if ti is not None else None


def _resolve_receiver(recv, tq):
    """Resolve a query receiver to its Table class name: a ``TableQuery`` val,
    inline ``TableQuery[Y]``, or the base of a chained ``q.verb(...).verb(...)``."""
    if recv.kind() == 'identifier':
        return tq.get(recv.text())
    if recv.kind() == 'generic_function':
        return _tablequery_class(recv)
    if recv.kind() == 'call_expression':
        fe = _child(recv, 'field_expression')
        return _resolve_receiver(_receiver(fe), tq) if fe is not None else None
    return None


def _receiver(field_expression):
    """The expression before ``.verb`` in ``<recv>.verb`` — always the first
    child of a field_expression."""
    return field_expression.children()[0]


def _verb(call):
    """``(verb, receiver)`` for ``<recv>.verb(...)``; ``(None, None)`` when the
    callee is not a ``<recv>.method`` form (e.g. ``column[T](...)``)."""
    fe = _child(call, 'field_expression')
    if fe is None:
        return None, None
    idents = [c for c in fe.children() if c.kind() == 'identifier']
    return (idents[-1].text() if idents else None), _receiver(fe)


def _decode_verb(call, doc_path, owning, schema, tq, aliases, witness):
    verb, recv = _verb(call)
    if verb not in _SLICK_VERBS:
        return [], []
    cls = _resolve_receiver(recv, tq)
    cls = aliases.get(cls, cls)  # resolve Scala import rename before the lookup
    if cls is None or cls not in schema:
        return [], []  # receiver not a known table query -> left alone, not guessed
    fields = _wildcard_fields(_child(call, 'arguments'))
    if not fields:
        return [], []
    role = 'write' if (verb == 'map' and _feeds_update(call)) else _SLICK_VERBS[verb]
    return _bind(call, cls, [(f, role) for f in fields],
                 doc_path, owning, schema, witness)


def _decode_insert(infix, doc_path, owning, schema, tq, aliases, witness):
    """``<query> += row`` / ``++= rows`` -> a whole-row write of every column."""
    op = _child(infix, 'operator_identifier')
    if op is None or op.text() not in _INSERT_OPS:
        return [], []
    kids = infix.children()
    cls = _resolve_receiver(kids[0] if kids else None, tq)
    cls = aliases.get(cls, cls)  # resolve Scala import rename before the lookup
    if cls is None or cls not in schema:
        return [], []
    table, field_map = schema[cls]
    return _bind(infix, cls, [(f, 'write') for f in field_map],
                 doc_path, owning, schema, witness)


def _bind(node, cls, field_roles, doc_path, owning, schema, witness):
    consumer = owning(doc_path, node.range().start.line)
    if consumer is None:
        return [], [f'{witness}: {cls} query has no owning symbol']
    table, field_map = schema[cls]
    rows, gaps = [], []
    for field, role in field_roles:
        if field in field_map:
            rows.append((consumer, table, field_map[field], role,
                         doc_path, node.range().start.line + 1))
        else:
            gaps.append(f'{witness}: {cls}.{field} -> undeclared field')
    return rows, gaps


def _wildcard_fields(args):
    """Field names from ``_.col`` lambda accesses anywhere in the call args; a
    non-wildcard field access (``obj.x``) is not a column reference."""
    out = []
    for fe in args.find_all(kind='field_expression'):
        idents = [c for c in fe.children() if c.kind() == 'identifier']
        if _child(fe, 'wildcard') is not None and idents:
            out.append(idents[0].text())
    return out


def _feeds_update(call):
    """True iff ``call``'s result is the receiver of an ``.update(...)`` — the
    Slick column-write idiom ``q.map(_.c).update(v)``."""
    par = call.parent()
    if par is None or par.kind() != 'field_expression':
        return False
    idents = [c for c in par.children() if c.kind() == 'identifier']
    return bool(idents) and idents[-1].text() == 'update'
