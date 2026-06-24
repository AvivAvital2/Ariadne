"""Django ORM strategy (design §5.2) — schema declaration conventions.

Table = ``Meta.db_table`` (exact) else ``app_label_model`` (derived; recovered
when the app_label is ambiguous). Column = ``db_column`` (exact) else field
name (derived); FK/OneToOne append ``_id`` (derived) and carry their target
model so the engine can resolve ``references_id``. Discovery is anchored on the
SCIP definition symbols (the engine supplies the anchor map); the metadata is
read from the model source with ``ast`` — one parse, no fuzzy match (§5.0.1).

Also owns Django's Layer-2 query recognizer (§5.2 Access API) — the
``<Model>.objects.<verb>(...)`` decode the engine dispatches to via
``DjangoStrategy.recognize``: filter/exclude/get -> filter, create/update ->
write, values/only/defer -> project, order_by -> order, with ``Q``/``F``,
computed ``annotate``/``aggregate``, and whole-row ``bulk_create`` expansion
handled; undecodable forms (``__``-spanning lookups, queryset-variable
receivers, ``save()``) surfaced as gaps, never guessed (§5.0, §5.8).
"""
from __future__ import annotations

import ast
from pathlib import Path
from docgen.orm_bindings.engine import Col, Table, _parse, _str_const
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from docgen.scip_extractor import ScipIndex

_FK_FIELDS = {'ForeignKey', 'OneToOneField'}


class DjangoStrategy:
    """Django's ``schema_spec`` (§5.2), as discovery over the model source."""

    name = 'orm:django'

    def detect(self, scip_index, root) -> bool:
        for doc in scip_index.documents:
            tree = _parse(root, doc.relative_path)
            if tree is not None and _model_classdefs(tree):
                return True
        return False

    def discover(self, scip_index, root, symbol_at):
        tables: list[Table] = []
        gaps: list[str] = []
        for doc in scip_index.documents:
            tree = _parse(root, doc.relative_path)
            if tree is None:
                gaps.append(f'{self.name}: {doc.relative_path} did not parse')
                continue
            app_label = Path(doc.relative_path).parent.name
            for cls in _model_classdefs(tree):
                producer = symbol_at.get((doc.relative_path, cls.lineno - 1))
                if producer is None:
                    gaps.append(f'{self.name}: model {cls.name} has no SCIP anchor')
                    continue
                table_name, confidence = _table_name(cls, app_label)
                columns = []
                for stmt in cls.body:
                    parsed = _field(stmt)
                    if parsed is None:
                        continue
                    field_name, call = parsed
                    col_symbol = symbol_at.get((doc.relative_path, stmt.lineno - 1))
                    if col_symbol is None:
                        gaps.append(
                            f'{self.name}: field {cls.name}.{field_name} has no SCIP anchor'
                        )
                        continue
                    columns.append(_column(field_name, call, col_symbol))
                tables.append(
                    Table(cls.name, table_name, producer, confidence, tuple(columns))
                )
        return tables, tuple(gaps)
    def recognize(self, scip_index, root, owning, schema, aliases):
        """Django Layer-2 recognition (§5.2), the engine's per-ORM hook."""
        return _recognize(scip_index, root, owning, schema, aliases, self.name)


def _model_classdefs(tree):
    return [
        node for node in tree.body
        if isinstance(node, ast.ClassDef)
        and any(_is_model_base(base) for base in node.bases)
    ]


def _is_model_base(base):
    return (
        isinstance(base, ast.Attribute) and base.attr == 'Model'
        and isinstance(base.value, ast.Name) and base.value.id == 'models'
    )


def _table_name(cls, app_label):
    explicit = _meta_db_table(cls)
    if explicit is not None:
        return explicit, 'exact'
    if app_label:
        return f'{app_label}_{cls.name.lower()}', 'derived'
    return cls.name.lower(), 'recovered'  # app_label ambiguous (§5.2 note)


def _meta_db_table(cls):
    for node in cls.body:
        if isinstance(node, ast.ClassDef) and node.name == 'Meta':
            for stmt in node.body:
                value = _assign_value(stmt, 'db_table')
                if value is not None:
                    return _str_const(value)
    return None


def _field(stmt):
    if (
        isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
        and isinstance(stmt.value, ast.Call)
    ):
        return stmt.targets[0].id, stmt.value
    return None


def _column(field_name, call, producer):
    if _is_fk_call(call):
        return Col(f'{field_name}_id', producer, 'derived',
                    fk_target_model=_fk_target(call), field_name=field_name)
    db_column = _db_column_kwarg(call)
    if db_column is not None:
        return Col(db_column, producer, 'exact', field_name=field_name)
    return Col(field_name, producer, 'derived', field_name=field_name)


def _is_fk_call(call):
    func = call.func
    name = (func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None)
    return name in _FK_FIELDS


def _fk_target(call):
    if not call.args:
        return None
    arg = call.args[0]
    if isinstance(arg, ast.Name):
        return arg.id
    const = _str_const(arg)
    return const.rsplit('.', 1)[-1] if const is not None else None  # 'app.Org'->'Org'


def _db_column_kwarg(call):
    for kw in call.keywords:
        if kw.arg == 'db_column':
            return _str_const(kw.value)
    return None


def _assign_value(stmt, target_name):
    if (
        isinstance(stmt, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == target_name for t in stmt.targets)
    ):
        return stmt.value
    return None

# Django query verbs -> role + column-site (the access_spec, §5/§5.2).
_KWARG_VERBS = {'filter': 'filter', 'exclude': 'filter', 'get': 'filter',
                'create': 'write', 'update': 'write'}
_STRING_VERBS = {'values': 'project', 'values_list': 'project',
                 'only': 'project', 'defer': 'project', 'order_by': 'order'}
_WHOLE_ROW = {'bulk_create'}           # manager whole-row write -> expansion §10 P2
# (instance-level ``save()`` has no ``.objects`` receiver, so it is out of this
#  call-site recognizer's reach — a documented residue, like queryset-var chains.)
_COMPUTED = {'annotate', 'aggregate'}  # computed expressions -> not decoded


def _base_model(node):
    """The model name in a ``<Model>.objects[.verb(...)]*`` receiver chain,
    or None when the receiver is not a manager off a model."""
    while isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
        node = node.func.value
    if (isinstance(node, ast.Attribute) and node.attr == 'objects'
            and isinstance(node.value, ast.Name)):
        return node.value.id
    return None


_DISTINCTIVE = {'filter', 'exclude', 'order_by', 'values_list', 'only',
                'defer', 'annotate', 'aggregate', 'bulk_create', 'create'}


def _callee_name(func):
    return (func.attr if isinstance(func, ast.Attribute)
            else func.id if isinstance(func, ast.Name) else None)


def _q_field_names(call):
    """Field names inside Q(...) objects passed positionally to a filter verb
    (Q may be combined with | & ~, so walk each positional arg's subtree)."""
    names = []
    for arg in call.args:
        for sub in ast.walk(arg):
            if isinstance(sub, ast.Call) and _callee_name(sub.func) == 'Q':
                names.extend(kw.arg for kw in sub.keywords)
    return names


def _f_field_names(call):
    """Fields referenced by ``F("...")`` in keyword values — a read of the
    field's current value (e.g. ``update(views=F("views") + 1)``)."""
    return [s for kw in call.keywords
            for sub in ast.walk(kw.value)
            if isinstance(sub, ast.Call) and _callee_name(sub.func) == 'F'
            and sub.args and (s := _str_const(sub.args[0])) is not None]


def _computed_fields(call, field_map):
    """Declared field names referenced by string inside a computed call
    (e.g. ``Sum("price")``, ``F("price")``) — read as projections; non-field
    strings (aliases, relations) are left alone."""
    return [s for sub in ast.walk(call)
            if (s := _str_const(sub)) is not None and s in field_map]


def _columns_named(verb, call):
    """``[(field, role)]`` referenced by a column-naming verb. Filter verbs
    also pull ``Q(...)`` kwargs; ``F("...")`` in a kwarg value is a read
    (project) of the referenced field, alongside the key's own role (§5.2)."""
    if verb in _KWARG_VERBS:
        role = _KWARG_VERBS[verb]
        pairs = [(kw.arg, role) for kw in call.keywords]
        if role == 'filter':
            pairs += [(f, 'filter') for f in _q_field_names(call)]
        pairs += [(f, 'project') for f in _f_field_names(call)]
        return pairs
    role = _STRING_VERBS[verb]
    pairs = []
    for arg in call.args:
        f = _str_const(arg)
        if f is not None and verb == 'order_by':
            f = f.lstrip('-')
        pairs.append((f, role))
    return pairs


def _recognize(scip_index, root, owning, schema, aliases, witness):
    rows, gaps = [], []
    for doc in scip_index.documents:
        tree = _parse(root, doc.relative_path)
        if tree is None:
            continue  # slice-1 discovery already records the parse gap
        file_aliases = aliases.get(doc.relative_path, {})
        for node in ast.walk(tree):
            if not (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)):
                continue
            verb = node.func.attr
            model = _base_model(node.func.value)
            if model is None:
                # save() is a whole-row instance write whose receiver model is not
                # statically bound without dataflow (a §5 residue) -> surfaced, never
                # silently dropped (§5.0). Other distinctive verbs on a non-manager
                # receiver are unresolved queryset variables (also surfaced gaps).
                if verb == 'save':
                    gaps.append(f'{witness}: .save() whole-row instance write — '
                                'receiver model needs dataflow')
                elif verb in _DISTINCTIVE:
                    gaps.append(f'{witness}: .{verb}(...) on a non-manager '
                                'receiver — queryset variable not resolved')
                continue
            where = f'{model}.{verb}'
            consumer = owning(doc.relative_path, node.lineno - 1)
            if consumer is None:
                gaps.append(f'{witness}: {where} has no owning symbol')
                continue
            resolved = file_aliases.get(model, model)  # resolve `import … as`
            if resolved not in schema:
                gaps.append(f'{witness}: {where} -> unknown model')
                continue
            if verb in _WHOLE_ROW:
                # Whole-row write (bulk_create): expand to a write on every column of the
                # model's table — the §10 Phase-2 fix for the write-recall hole.
                table, field_map = schema[resolved]
                for column in field_map.values():
                    rows.append((consumer, table, column, 'write',
                                 doc.relative_path, node.lineno))
                continue
            table, field_map = schema[resolved]
            if verb in _COMPUTED:
                found = _computed_fields(node, field_map)
                if not found:
                    gaps.append(f'{witness}: {where} computed expression not decoded')
                for field in found:
                    rows.append((consumer, table, field_map[field], 'project',
                                 doc.relative_path, node.lineno))
                continue
            if verb not in _KWARG_VERBS and verb not in _STRING_VERBS:
                continue  # a manager/queryset method that names no column
            for field, role in _columns_named(verb, node):
                if field is None:
                    gaps.append(f'{witness}: {where} non-literal column argument')
                elif '__' in field:
                    gaps.append(f'{witness}: {where}({field}) spanning lookup deferred')
                elif field not in field_map:
                    gaps.append(f'{witness}: {where}({field}) -> undeclared field')
                else:
                    rows.append((consumer, table, field_map[field], role,
                                 doc.relative_path, node.lineno))
    return rows, gaps
