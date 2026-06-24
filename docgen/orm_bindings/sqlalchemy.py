"""SQLAlchemy ORM strategy (design §5.1) — schema declaration conventions.

Declarative style: table = ``__tablename__`` literal (exact); column = the first
positional string to ``mapped_column``/``Column`` (exact) else the attribute name
(derived); FKs from ``ForeignKey("table.col")`` carry their target table so the
engine resolves ``references_id``. Discovery is anchored on the SCIP definition
symbols (the engine supplies the anchor map); the metadata is read from the model
source with ``ast`` — one parse, no fuzzy match (§5.0.1). Unresolved bindings (no
SCIP anchor, unparsable source, non-literal ``__tablename__``, FK target not a
discovered table) are RECORDED as surfaced gaps, never silently dropped (§5.0,
§5.0.1 #5).
"""
from __future__ import annotations

import ast
from typing import TYPE_CHECKING

from docgen.orm_bindings.engine import Col, Table, _parse, _str_const

if TYPE_CHECKING:
    from docgen.scip_extractor import ScipIndex

_COLUMN_CALLS = {'mapped_column', 'Column'}


class SQLAlchemyStrategy:
    """SQLAlchemy's ``schema_spec`` (§5.1), as discovery over the model source."""

    name = 'orm:sqlalchemy'

    def recognize(self, scip_index, root, owning, schema, aliases):
        """SQLAlchemy Layer-2 recognition (§5.1), the engine's per-ORM hook."""
        return _sa_recognize(scip_index, root, owning, schema, aliases, self.name)

    def detect(self, scip_index, root) -> bool:
        for doc in scip_index.documents:
            tree = _parse(root, doc.relative_path)
            if tree is None:
                continue
            if any(_model_info(node)[0] for node in tree.body
                   if isinstance(node, ast.ClassDef)):
                return True
        return False

    def discover(self, scip_index, root, symbol_at):
        raw: list[tuple] = []
        gaps: list[str] = []
        for doc in scip_index.documents:
            tree = _parse(root, doc.relative_path)
            if tree is None:
                gaps.append(f'{self.name}: {doc.relative_path} did not parse')
                continue
            for cls in tree.body:
                if not isinstance(cls, ast.ClassDef):
                    continue
                is_model, table_name, table_conf = _model_info(cls)
                if not is_model:
                    continue
                producer = symbol_at.get((doc.relative_path, cls.lineno - 1))
                if producer is None:
                    gaps.append(f'{self.name}: model {cls.name} has no SCIP anchor')
                    continue
                if table_name is None:
                    gaps.append(
                        f'{self.name}: model {cls.name} __tablename__ not a literal')
                    continue
                cols = []
                for stmt in cls.body:
                    parsed = _column_attr(stmt)
                    if parsed is None:
                        continue
                    attr_name, call = parsed
                    col_symbol = symbol_at.get((doc.relative_path, stmt.lineno - 1))
                    if col_symbol is None:
                        gaps.append(
                            f'{self.name}: column {cls.name}.{attr_name} has no SCIP anchor')
                        continue
                    cols.append(_raw_column(attr_name, call, col_symbol))
                raw.append((cls.name, table_name, producer, table_conf, cols))
        return _resolve_fks(raw), tuple(gaps)


def _model_info(cls):
    """``(is_sqlalchemy_model, table_name_or_None, table_confidence)``. A model is
    a class that extends a base AND declares either a ``__tablename__`` or at least
    one ``mapped_column``/``Column`` attribute. The table name is the
    ``__tablename__`` string literal (``exact``), or — when ``__tablename__`` is
    absent — the lowercased class name (``derived``, the ``declarative_base``
    convention). A non-literal ``__tablename__`` yields no name (a gap, never guessed)."""
    if not cls.bases:
        return False, None, None
    value = _tablename_value(cls)
    if value is None:
        if not any(_column_attr(s) is not None for s in cls.body):
            return False, None, None
        return True, cls.name.lower(), 'derived'
    name = _str_const(value)
    if name is None:
        return True, None, None
    return True, name, 'exact'


def _tablename_value(cls):
    """The value node of a ``__tablename__ = …`` assignment, or None if absent."""
    for stmt in cls.body:
        if (isinstance(stmt, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == '__tablename__'
                        for t in stmt.targets)):
            return stmt.value
    return None


def _column_attr(stmt):
    """``(attr_name, call)`` if ``stmt`` is ``attr = mapped_column/Column(...)``
    or ``attr: T = mapped_column/Column(...)``, else None — so ``__tablename__``,
    docstrings, and plain assignments are skipped."""
    if isinstance(stmt, ast.AnnAssign):
        target, value = stmt.target, stmt.value
    elif isinstance(stmt, ast.Assign) and len(stmt.targets) == 1:
        target, value = stmt.targets[0], stmt.value
    else:
        return None
    if isinstance(target, ast.Name) and _is_column_call(value):
        return target.id, value
    return None


def _is_column_call(node):
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    name = (func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None)
    return name in _COLUMN_CALLS


def _raw_column(attr_name, call, producer):
    """A column whose ``fk_target_model`` field transiently carries the FK
    *target table* (resolved to the owning model by ``_resolve_fks``). Name =
    first positional string (exact) else the attribute name (derived)."""
    override = _positional_str(call)
    fk_table = _fk_target_table(call)
    if override is not None:
        return Col(override, producer, 'exact',
                    fk_target_model=fk_table, field_name=attr_name)
    return Col(attr_name, producer, 'derived',
                fk_target_model=fk_table, field_name=attr_name)


def _positional_str(call):
    """First positional string arg — the column-name override (§5.1)."""
    for arg in call.args:
        s = _str_const(arg)
        if s is not None:
            return s
    return None


def _fk_target_table(call):
    """Target table from ``ForeignKey("table.col")`` anywhere in the call args,
    else None (symbol-form FKs resolve via Layer 1, not here)."""
    for node in ast.walk(call):
        if (isinstance(node, ast.Call) and _callee_is(node.func, 'ForeignKey')
                and node.args):
            target = _str_const(node.args[0])
            if target is not None:
                return target.split('.', 1)[0]  # "orgs.id" -> "orgs"
    return None


def _callee_is(func, name):
    return ((isinstance(func, ast.Attribute) and func.attr == name)
            or (isinstance(func, ast.Name) and func.id == name))


def _resolve_fks(raw):
    """Resolve each FK's target *table* to the *model* that declares it, so the
    engine's model->table map binds ``references_id``; an unknown table is left
    as-is for the engine to surface as an unresolved-FK gap (never guessed)."""
    table_to_model = {table: model for model, table, _p, _conf, _cols in raw}
    tables = []
    for model, table, producer, table_conf, cols in raw:
        resolved = tuple(
            Col(c.column_name, c.producer_symbol_id, c.confidence,
                 fk_target_model=(
                     table_to_model.get(c.fk_target_model, c.fk_target_model)
                     if c.fk_target_model is not None else None),
                 field_name=c.field_name)
            for c in cols
        )
        tables.append(Table(model, table, producer, table_conf, resolved))
    return tables


# --- Layer 2 — access binding (§5.1 Access API) ----------------------------
# query roots that name the model in their first positional arg.
_ROOT_FUNCS = {'select', 'insert', 'update', 'delete'}
# chained verbs whose columns are Model.<attr> references in the call args.
_ATTR_VERBS = {'where': 'filter', 'filter': 'filter', 'with_entities': 'project',
               'only': 'project', 'order_by': 'order'}
# chained verbs whose columns are kwargs (col=value).
_KWARG_VERBS = {'filter_by': 'filter', 'values': 'write'}
_VERBS = {**_ATTR_VERBS, **_KWARG_VERBS}


def _sa_recognize(scip_index, root, owning, schema, aliases, witness):
    rows, gaps = [], []
    for doc in scip_index.documents:
        tree = _parse(root, doc.relative_path)
        if tree is None:
            continue  # Layer-1 discovery already recorded the parse gap
        file_aliases = aliases.get(doc.relative_path, {})
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                r, g = _decode_call(node, doc.relative_path, owning, schema,
                                    file_aliases, witness)
                rows.extend(r)
                gaps.extend(g)
    return rows, gaps


def _decode_call(node, doc_path, owning, schema, aliases, witness):
    """One query call site -> (rows, gaps). A query root contributes its
    projected ``select(Model.col)`` columns; a chained verb contributes the
    columns it names; ``session.add(Model(...))`` is a whole-row write. Anything
    else (a non-query call, a chain not rooted at a constructor) is left alone.
    The model name resolves through ``aliases`` (``import … as``) before the
    schema lookup, while AST attribute refs keep the local name as written."""
    func = node.func
    if isinstance(func, ast.Attribute) and func.attr == 'add':
        return _decode_add(node, doc_path, owning, schema, aliases, witness)
    if _is_query_root(node):
        root_call = node
    elif isinstance(func, ast.Attribute) and func.attr in _VERBS:
        root_call = _chain_root(node)
    else:
        return [], []
    if root_call is None:
        return [], []  # a verb-named chain not rooted at a query constructor
    model = _root_model(root_call)
    if model is None:
        return [], []  # e.g. select() with no model arg
    fields_role = _call_columns(node, root_call, model)
    if not fields_role:
        return [], []  # bare select(Model)/update(Model) names no column itself
    consumer = owning(doc_path, node.lineno - 1)
    if consumer is None:
        return [], [f'{witness}: {model} query has no owning symbol']
    resolved = aliases.get(model, model)  # resolve `import … as` before the lookup
    if resolved not in schema:
        return [], [f'{witness}: {model} query -> unknown model']
    table, field_map = schema[resolved]
    rows, gaps = [], []
    for field, role in fields_role:
        if field not in field_map:
            gaps.append(f'{witness}: {model}.{field} -> undeclared field')
        else:
            rows.append((consumer, table, field_map[field], role, doc_path, node.lineno))
    return rows, gaps


def _decode_add(node, doc_path, owning, schema, aliases, witness):
    """``session.add(Model(...))`` -> a whole-row write of every column (§5.1).
    ``add`` of a bare variable / non-model needs dataflow, so it is left alone
    rather than guessed. The constructed model name resolves through ``aliases``
    (``import … as``) before the schema lookup."""
    if not (len(node.args) == 1 and isinstance(node.args[0], ast.Call)
            and isinstance(node.args[0].func, ast.Name)):
        return [], []
    model = node.args[0].func.id
    resolved = aliases.get(model, model)
    if resolved not in schema:
        return [], []
    consumer = owning(doc_path, node.lineno - 1)
    if consumer is None:
        return [], [f'{witness}: session.add({model}) has no owning symbol']
    table, field_map = schema[resolved]
    return ([(consumer, table, col, 'write', doc_path, node.lineno)
             for col in field_map.values()], [])


def _call_columns(node, root_call, model):
    """``[(field, role)]`` a call names. The query root projects its
    ``Model.col`` constructor args; a kwarg verb reads its keywords; an attr verb
    reads the ``Model.col`` references in its args."""
    if node is root_call:
        return [(f, 'project') for f in _model_attrs(root_call.args, model)]
    verb = node.func.attr
    if verb in _KWARG_VERBS:
        return [(kw.arg, _KWARG_VERBS[verb]) for kw in node.keywords]
    return [(f, _ATTR_VERBS[verb]) for f in _model_attrs(node.args, model)]


def _is_query_root(call):
    func = call.func
    return ((isinstance(func, ast.Name) and func.id in _ROOT_FUNCS)
            or (isinstance(func, ast.Attribute) and func.attr == 'query'))


def _chain_root(node):
    """Walk a method chain down to its query-root constructor, or None."""
    cur = node
    while isinstance(cur, ast.Call):
        if _is_query_root(cur):
            return cur
        if isinstance(cur.func, ast.Attribute):
            cur = cur.func.value
        else:
            return None
    return None


def _root_model(root_call):
    """Model name from a query root's first positional arg: ``select(User)`` or
    the base of ``select(User.col)``."""
    if not root_call.args:
        return None
    arg = root_call.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    if isinstance(arg, ast.Attribute) and isinstance(arg.value, ast.Name):
        return arg.value.id
    return None


def _model_attrs(args, model):
    """Attribute names ``model.<attr>`` referenced anywhere in the call args
    (so ``User.email == x`` and ``User.email.in_(...)`` both yield ``email``)."""
    return [sub.attr for arg in args for sub in ast.walk(arg)
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
            and sub.value.id == model]
