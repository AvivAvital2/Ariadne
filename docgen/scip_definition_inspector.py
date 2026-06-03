"""Phase 2s.b — definition-site RHS inspector.

Given source text, a line number, and a language tag, returns what
kind of expression sits on the RHS of the assignment at that line:

- ``'literal'`` — RHS is a single string literal.
- ``'getter_call'`` — RHS is a call to a known config getter
  (``config.getString("K")`` / ``config.get("K")`` /
  ``config["K"]``); ``config_key`` carries the literal first-arg
  string.
- ``'other'`` — anything else (function call to non-getter, complex
  expression, no assignment at the line, broken source).

Phase 2s consults this when finishing variable resolution: if the
var's def line classifies as a getter call, the resolver looks the
key up in Phase 2q ``config_values`` instead of returning the literal
key string as the value.

Per-language detection lives here behind a single dispatch entry
point so the resolver stays pure DB queries. Recognized getter
method names are intentionally common across languages (``get``,
``getString``, ``getInt``, ``getBoolean``, ``getDouble``, etc.) —
each language has slightly different conventions but the typesafe
config / dotenv-style ``get*`` family covers the vast majority of
real-world cases.
"""
from __future__ import annotations

import ast
from typing import Literal

from ast_grep_py import SgRoot
from attrs import frozen


_Kind = Literal['literal', 'getter_call', 'other']


@frozen
class InspectionResult:
    kind: _Kind
    config_key: str | None = None


# Method names treated as config getters across languages. Single-
# arg, returns the value associated with the key. Each language
# uses a subset of these in practice (Python tends to ``get``;
# JVM uses ``getString``/``getInt``/etc.).
_GETTER_METHODS: frozenset[str] = frozenset({
    'get',
    'getString', 'getStringList',
    'getInt', 'getIntList',
    'getBoolean', 'getBooleanList',
    'getLong', 'getLongList',
    'getDouble', 'getDoubleList',
    'getNumber', 'getNumberList',
    'getDuration', 'getDurationList',
    'getOrElse',
    'getValue', 'getRaw',
})


def inspect_definition_rhs(
    *, source_text: str, line: int, language: str,
) -> InspectionResult:
    """Inspect the RHS of an assignment at ``line`` in ``source_text``.

    Returns ``InspectionResult(kind='other')`` for unsupported
    languages or any case the per-language inspector can't classify
    confidently — never raises.
    """
    if language == 'python':
        return _inspect_python(source_text, line)
    if language == 'javascript':
        return _inspect_js(source_text, line)
    if language == 'scala':
        return _inspect_scala(source_text, line)
    return InspectionResult(kind='other')


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _inspect_python(source_text: str, line: int) -> InspectionResult:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return InspectionResult(kind='other')
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        if node.lineno != line:
            continue
        rhs = node.value
        if rhs is None:  # AnnAssign without initializer
            return InspectionResult(kind='other')
        return _classify_python_rhs(rhs)
    return InspectionResult(kind='other')


def _classify_python_rhs(rhs) -> InspectionResult:
    if isinstance(rhs, ast.Constant) and isinstance(rhs.value, str):
        return InspectionResult(kind='literal')
    if isinstance(rhs, ast.Call):
        if isinstance(rhs.func, ast.Attribute):
            if (
                rhs.func.attr in _GETTER_METHODS
                and rhs.args
                and isinstance(rhs.args[0], ast.Constant)
                and isinstance(rhs.args[0].value, str)
            ):
                return InspectionResult(
                    kind='getter_call',
                    config_key=rhs.args[0].value,
                )
        return InspectionResult(kind='other')
    if isinstance(rhs, ast.Subscript):
        slice_val = rhs.slice
        # Python 3.8 wrapped Subscript.slice in ast.Index; 3.9+ is
        # the expression directly. Support both for compatibility.
        if hasattr(ast, 'Index') and isinstance(
            slice_val, getattr(ast, 'Index'),
        ):
            slice_val = slice_val.value  # type: ignore[attr-defined]
        if (
            isinstance(slice_val, ast.Constant)
            and isinstance(slice_val.value, str)
        ):
            return InspectionResult(
                kind='getter_call',
                config_key=slice_val.value,
            )
        return InspectionResult(kind='other')
    return InspectionResult(kind='other')


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------


def _inspect_js(source_text: str, line: int) -> InspectionResult:
    try:
        root = SgRoot(source_text, 'javascript').root()
    except Exception:
        return InspectionResult(kind='other')
    for var_decl in root.find_all(kind='variable_declarator'):
        r = var_decl.range()
        if r.start.line + 1 != line:
            continue
        rhs = _js_var_decl_rhs(var_decl)
        if rhs is None:
            return InspectionResult(kind='other')
        return _classify_js_rhs(rhs)
    return InspectionResult(kind='other')


def _js_var_decl_rhs(var_decl):
    """Return the RHS expression child of a ``variable_declarator``,
    skipping the name, ``=``, type annotations, and punctuation."""
    skip = {'identifier', '=', ',', 'type_annotation', 'property_identifier'}
    rhs = None
    for c in var_decl.children():
        if c.kind() in skip:
            continue
        rhs = c
    return rhs


def _js_strip_quotes(text: str) -> str | None:
    if (
        len(text) >= 2
        and text[0] in ('"', "'", '`')
        and text[-1] == text[0]
    ):
        return text[1:-1]
    return None


def _classify_js_rhs(rhs) -> InspectionResult:
    kind = rhs.kind()
    if kind == 'string':
        return InspectionResult(kind='literal')
    if kind == 'template_string':
        if '${' not in rhs.text():
            return InspectionResult(kind='literal')
        return InspectionResult(kind='other')
    if kind == 'call_expression':
        children = list(rhs.children())
        if not children:
            return InspectionResult(kind='other')
        callee = children[0]
        if callee.kind() == 'member_expression':
            method_name = None
            for c in reversed(list(callee.children())):
                if c.kind() == 'property_identifier':
                    method_name = c.text()
                    break
            if method_name in _GETTER_METHODS:
                # First string arg from arguments node
                args = next(
                    (
                        x for x in rhs.children()
                        if x.kind() == 'arguments'
                    ),
                    None,
                )
                if args is not None:
                    for a in args.children():
                        if a.kind() in ('(', ')', ','):
                            continue
                        if a.kind() == 'string':
                            value = _js_strip_quotes(a.text())
                            if value is not None:
                                return InspectionResult(
                                    kind='getter_call',
                                    config_key=value,
                                )
                        break
        return InspectionResult(kind='other')
    if kind == 'subscript_expression':
        for c in rhs.children():
            if c.kind() == 'string':
                value = _js_strip_quotes(c.text())
                if value is not None:
                    return InspectionResult(
                        kind='getter_call',
                        config_key=value,
                    )
        return InspectionResult(kind='other')
    return InspectionResult(kind='other')


# ---------------------------------------------------------------------------
# Scala
# ---------------------------------------------------------------------------


def _inspect_scala(source_text: str, line: int) -> InspectionResult:
    try:
        root = SgRoot(source_text, 'scala').root()
    except Exception:
        return InspectionResult(kind='other')
    for kind_name in ('val_definition', 'var_definition'):
        for node in root.find_all(kind=kind_name):
            r = node.range()
            if r.start.line + 1 != line:
                continue
            rhs = _scala_val_def_rhs(node)
            if rhs is None:
                return InspectionResult(kind='other')
            return _classify_scala_rhs(rhs)
    return InspectionResult(kind='other')


def _scala_val_def_rhs(val_def):
    """Return the RHS of a val/var_definition. The Scala grammar
    yields the body as the last child after the ``=`` token; walk
    from the end and pick the first non-punctuation child."""
    skip = {'val', 'var', '=', ':', ','}
    rhs = None
    for c in val_def.children():
        kind = c.kind()
        if kind in skip:
            continue
        rhs = c
    return rhs


def _classify_scala_rhs(rhs) -> InspectionResult:
    kind = rhs.kind()
    if kind in ('string', 'string_literal'):
        return InspectionResult(kind='literal')
    if kind == 'call_expression':
        method = _scala_call_method_name(rhs)
        if method in _GETTER_METHODS:
            key = _scala_call_first_string_arg(rhs)
            if key is not None:
                return InspectionResult(
                    kind='getter_call',
                    config_key=key,
                )
        return InspectionResult(kind='other')
    return InspectionResult(kind='other')


def _scala_call_method_name(call):
    """For ``obj.method(...)`` parsed as ``call_expression`` with a
    ``field_expression`` callee, return ``'method'``. For bare-name
    calls, return that name."""
    children = list(call.children())
    if not children:
        return None
    callee = children[0]
    kind = callee.kind()
    if kind in ('field_expression', 'select_expression'):
        for c in reversed(list(callee.children())):
            if c.kind() in (
                'identifier', 'simple_identifier', 'name',
            ):
                return c.text()
        return None
    if kind in ('identifier', 'simple_identifier', 'name'):
        return callee.text()
    if kind == 'call_expression':
        # Apply-with-block — descend
        return _scala_call_method_name(callee)
    return None


def _scala_call_first_string_arg(call) -> str | None:
    args = next(
        (
            c for c in call.children()
            if c.kind() in (
                'arguments', 'arguments_list', 'argument_list',
            )
        ),
        None,
    )
    if args is None:
        return None
    for c in args.children():
        kind = c.kind()
        if kind in ('(', ')', ','):
            continue
        if kind in ('string', 'string_literal'):
            text = c.text()
            if (
                len(text) >= 2
                and text[0] == '"'
                and text[-1] == '"'
            ):
                return text[1:-1]
        return None
    return None


__all__ = ['InspectionResult', 'inspect_definition_rhs']
