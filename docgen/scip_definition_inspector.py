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
_PATH_GETTERS: frozenset[str] = frozenset({'getConfig'})


def inspect_definition_rhs(
    *, source_text: str, line: int, language: str,
) -> InspectionResult:
    """Inspect the RHS of an assignment at ``line`` in ``source_text``.

    Returns ``InspectionResult(kind='other')`` for unsupported
    languages or any case the per-language inspector can't classify
    confidently — never raises. Thin single-line form of
    :func:`inspect_definitions_at_lines`; both share one code path so
    they cannot diverge.
    """
    return inspect_definitions_at_lines(
        source_text=source_text, lines=(line,), language=language,
    ).get(line, InspectionResult(kind='other'))
def inspect_definitions_at_lines(
    *, source_text: str, lines, language: str,
) -> dict[int, InspectionResult]:
    """Classify the assignment RHS at each line in ``lines`` from a
    SINGLE parse of ``source_text`` — the batch form of
    :func:`inspect_definition_rhs`.

    Parsing is O(file); each requested line is then an O(1) map lookup,
    so classifying ``D`` lines costs **one** parse, not ``D``. Returns
    ``{line: InspectionResult}`` for the requested lines that carry an
    assignment; a line with no assignment is omitted (callers treat a
    missing line as ``'other'``). Unsupported languages → ``{}``.
    """
    wanted = set(lines)
    if language == 'python':
        return _python_lines(source_text, wanted)
    if language == 'javascript':
        return _astgrep_lines(
            source_text, 'javascript', wanted,
            kinds=('variable_declarator',),
            rhs_of=_js_var_decl_rhs, classify=_classify_js_rhs,
        )
    if language == 'scala':
        return _astgrep_lines(
            source_text, 'scala', wanted,
            kinds=('val_definition', 'var_definition'),
            rhs_of=_scala_val_def_rhs, classify=_classify_scala_rhs,
        )
    return {}


# ---------------------------------------------------------------------------
# Python
# ---------------------------------------------------------------------------


def _python_lines(
    source_text: str, wanted: set[int],
) -> dict[int, InspectionResult]:
    try:
        tree = ast.parse(source_text)
    except SyntaxError:
        return {}
    out: dict[int, InspectionResult] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Assign, ast.AnnAssign)):
            continue
        line = node.lineno
        if line not in wanted or line in out:
            continue
        rhs = node.value
        out[line] = (
            InspectionResult(kind='other') if rhs is None
            else _classify_python_rhs(rhs)
        )
    return out


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
                prefix = _python_config_prefix(rhs.func.value)
                return InspectionResult(
                    kind='getter_call',
                    config_key='.'.join([*prefix, rhs.args[0].value]),
                )
        return InspectionResult(kind='other')
    if isinstance(rhs, ast.Subscript):
        # 3.9+ exposes the slice expression directly (the 3.8 ast.Index
        # wrapper is gone on the supported runtime).
        slice_val = rhs.slice
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
def _python_config_prefix(node) -> list[str]:
    """Walk a receiver chain of ``getConfig("x")`` calls (each call's
    ``func.value``), returning the path segments outermost-first. Empty
    for a non-``getConfig`` receiver — the bare-key / opaque-receiver
    case where the dotted prefix is unknown."""
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in _PATH_GETTERS
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    ):
        return [*_python_config_prefix(node.func.value), node.args[0].value]
    return []


# ---------------------------------------------------------------------------
# JavaScript / TypeScript
# ---------------------------------------------------------------------------


def _astgrep_lines(
    source_text: str, language: str, wanted: set[int],
    *, kinds, rhs_of, classify,
) -> dict[int, InspectionResult]:
    # tree-sitter is error-tolerant (SgRoot never raises — it yields
    # ERROR nodes on bad input), so no parse guard is needed here.
    root = SgRoot(source_text, language).root()
    out: dict[int, InspectionResult] = {}
    for kind in kinds:
        for node in root.find_all(kind=kind):
            line = node.range().start.line + 1
            if line not in wanted or line in out:
                continue
            rhs = rhs_of(node)
            out[line] = (
                InspectionResult(kind='other') if rhs is None
                else classify(rhs)
            )
    return out


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
        method = _js_call_method_name(rhs)
        if method in _GETTER_METHODS:
            key = _js_call_first_string_arg(rhs)
            if key is not None:
                prefix = _js_config_prefix(_js_call_receiver(rhs))
                return InspectionResult(
                    kind='getter_call',
                    config_key='.'.join([*prefix, key]),
                )
        return InspectionResult(kind='other')
    if kind == 'subscript_expression':
        for c in rhs.children():
            if c.kind() == 'string':
                return InspectionResult(
                    kind='getter_call',
                    config_key=_js_strip_quotes(c.text()),
                )
        return InspectionResult(kind='other')
    return InspectionResult(kind='other')
def _js_call_method_name(call) -> str | None:
    """Method name of a ``recv.method(...)`` call (member-expression
    callee). ``None`` for a bare-name call, whose callee is a plain
    identifier."""
    callee = next(iter(call.children()), None)
    if callee is None or callee.kind() != 'member_expression':
        return None
    names = [
        c.text() for c in callee.children()
        if c.kind() == 'property_identifier'
    ]
    return names[-1] if names else None


def _js_call_receiver(call):
    """The receiver object of a ``recv.method(...)`` call — the first
    child of the callee ``member_expression``. Only ever called on
    member-callee getter calls (the method name was already matched),
    so the callee shape is guaranteed; no defensive branch is needed."""
    callee = next(iter(call.children()))
    return next(iter(callee.children()), None)


def _js_call_first_string_arg(call) -> str | None:
    """First positional argument if it's a string literal, else ``None``
    (empty args, or a dynamic/non-literal first arg)."""
    for child in call.children():
        if child.kind() != 'arguments':
            continue
        for a in child.children():
            if a.kind() in ('(', ')', ','):
                continue
            if a.kind() == 'string':
                return _js_strip_quotes(a.text())
            return None
    return None
def _astgrep_config_prefix(
    node, *, method_name, receiver, first_string_arg,
) -> list[str]:
    """Walk an ast-grep receiver chain of ``getConfig("x")`` calls,
    returning the path segments outermost-first. Shared by the Scala and
    JS inspectors, which differ only in the three node-accessor callables
    (``method_name`` / ``receiver`` / ``first_string_arg``), not in the
    walk itself. Empty for any non-``getConfig`` receiver (a plain
    identifier or a terminal value getter) — the bare-key case."""
    if node is None or node.kind() != 'call_expression':
        return []
    if method_name(node) not in _PATH_GETTERS:
        return []
    arg = first_string_arg(node)
    if arg is None:
        return []
    return [
        *_astgrep_config_prefix(
            receiver(node), method_name=method_name,
            receiver=receiver, first_string_arg=first_string_arg,
        ),
        arg,
    ]


def _js_config_prefix(node) -> list[str]:
    return _astgrep_config_prefix(
        node,
        method_name=_js_call_method_name,
        receiver=_js_call_receiver,
        first_string_arg=_js_call_first_string_arg,
    )


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
                prefix = _scala_config_prefix(_scala_call_receiver(rhs))
                return InspectionResult(
                    kind='getter_call',
                    config_key='.'.join([*prefix, key]),
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
def _scala_call_receiver(call):
    """The object a method call is invoked on — the first child of the
    callee ``field_expression``. ``None`` for a bare-name call, whose
    callee is a plain identifier (no receiver to walk)."""
    callee = next(iter(call.children()), None)
    if callee is not None and callee.kind() in (
        'field_expression', 'select_expression',
    ):
        return next(iter(callee.children()), None)
    return None


def _scala_config_prefix(node) -> list[str]:
    return _astgrep_config_prefix(
        node,
        method_name=_scala_call_method_name,
        receiver=_scala_call_receiver,
        first_string_arg=_scala_call_first_string_arg,
    )


__all__ = ['InspectionResult', 'inspect_definition_rhs']
