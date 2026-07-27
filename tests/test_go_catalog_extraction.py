"""Go catalog extraction — routes ``.go`` files through the SCIP extractor
(like Scala/Java) so a Go corpus produces real catalog elements (structs,
interfaces, funcs, methods), not just cross-source grounding.

The SCIP symbol→subtype mapping is the core; language detection, the Subtype
taxonomy, and the generator's class-like/callable groupings complete the chain
so Go elements render into architecture/explanation docs like every other
language.
"""
from __future__ import annotations

from typing import get_args

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
    _ScipSymbol,
    _subtype,
    extract,
)


def _sym(kind: str, **kw) -> _ScipSymbol:
    return _ScipSymbol(symbol='scip-go gomod mod 1 pkg/Foo#', kind=kind, **kw)


class TestGoSubtypeMapping:
    """SCIP SymbolKind × parent → Go Ariadne subtype. scip-go emits standard
    SCIP kinds; method-vs-function is disambiguated by the parent descriptor
    (a receiver type → method), mirroring the python/js branches."""

    def test_struct_and_interface(self) -> None:
        assert _subtype(_sym('Struct'), 'go') == 'go_struct'
        assert _subtype(_sym('Interface'), 'go') == 'go_interface'

    def test_method_vs_function_by_parent(self) -> None:
        # parent descriptor is a type (receiver) → method; else top-level func.
        assert _subtype(_sym('Method'), 'go',
                        parent_descriptor_kind='type') == 'go_method'
        assert _subtype(_sym('Function'), 'go',
                        parent_descriptor_kind='package') == 'go_function'
        # scip-go may label either way; parent kind is authoritative.
        assert _subtype(_sym('Function'), 'go',
                        parent_descriptor_kind='type') == 'go_method'
        assert _subtype(_sym('Method'), 'go',
                        parent_descriptor_kind='package') == 'go_function'
        # no parent info → default to top-level function
        assert _subtype(_sym('Function'), 'go') == 'go_function'

    def test_fields_consts_vars_types(self) -> None:
        assert _subtype(_sym('Field'), 'go') == 'go_field'
        assert _subtype(_sym('Constant'), 'go') == 'go_const'
        assert _subtype(_sym('Variable'), 'go') == 'go_var'
        assert _subtype(_sym('Type'), 'go') == 'go_type'
        assert _subtype(_sym('TypeAlias'), 'go') == 'go_type'

    def test_untracked_kind_is_none(self) -> None:
        assert _subtype(_sym('Namespace'), 'go') is None

    def test_go_branch_does_not_leak_into_other_languages(self) -> None:
        # a Struct kind is meaningless for python; the go mapping must not
        # apply to other languages.
        assert _subtype(_sym('Struct'), 'python') is None


class TestGoLanguageDetection:
    def test_go_extension_detected(self, tmp_path) -> None:
        from docgen.catalog_extractor import _detect_language
        assert _detect_language(tmp_path / 'main.go') == 'go'


class TestGoSubtypeTaxonomy:
    def test_go_subtypes_in_the_subtype_literal(self) -> None:
        import docgen.catalog_extractor as ce
        subtypes = get_args(ce.Subtype)
        for s in ('go_struct', 'go_interface', 'go_function', 'go_method',
                  'go_field', 'go_const', 'go_var', 'go_type'):
            assert s in subtypes, s

    def test_go_is_a_catalog_language(self) -> None:
        import docgen.catalog_extractor as ce
        assert 'go' in get_args(ce.Language)

    def test_go_grouped_class_like_and_callable(self) -> None:
        from docgen.generator import (
            _CALLABLE_SUBTYPES,
            _CLASS_LIKE_SUBTYPES,
        )
        assert {'go_struct', 'go_interface'} <= _CLASS_LIKE_SUBTYPES
        assert {'go_function', 'go_method'} <= _CALLABLE_SUBTYPES


class TestGoCatalogRoutesThroughScip:
    def test_go_is_scip_only_no_astgrep_fallback(self, tmp_path) -> None:
        # A .go file with no SCIP index declared returns [] (no ast-grep
        # grammar for Go) — exactly like Scala/Java, never a garbage parse.
        from docgen.catalog_extractor import extract_elements
        go_file = tmp_path / 'main.go'
        go_file.write_text('package main\nfunc Foo() {}\n')
        elements = extract_elements(
            go_file, source_root=tmp_path, source_config=None,
        )
        assert elements == []

    def test_extract_produces_go_elements_from_scip(self, tmp_path) -> None:
        # End-to-end: a synthetic scip-go index over one .go file yields the
        # right ElementInfo subtypes + qualified names (struct, receiver
        # method, top-level func). This is what makes docs real for Go.
        src = tmp_path / 'proj'
        (src / 'srv').mkdir(parents=True)
        go_file = src / 'srv' / 'server.go'
        go_file.write_text(
            'package srv\n'
            'type Server struct{}\n'
            'func (s *Server) Start() {}\n'
            'func New() *Server { return nil }\n'
        )
        base = 'scip-go gomod example.com/m v1'
        struct_sym = f'{base} srv/Server#'
        method_sym = f'{base} srv/Server#Start().'
        func_sym = f'{base} srv/New().'
        doc = _ScipDoc(
            relative_path='srv/server.go',
            occurrences=(
                _ScipOccurrence(symbol=struct_sym, range=(1, 5, 1, 11),
                                is_definition=True),
                _ScipOccurrence(symbol=method_sym, range=(2, 0, 2, 20),
                                is_definition=True),
                _ScipOccurrence(symbol=func_sym, range=(3, 0, 3, 30),
                                is_definition=True),
            ),
            symbols=(
                _ScipSymbol(symbol=struct_sym, kind='Struct'),
                _ScipSymbol(symbol=method_sym, kind='Method'),
                _ScipSymbol(symbol=func_sym, kind='Function'),
            ),
        )
        index = ScipIndex(documents=(doc,), source_root=src)
        elements = extract(go_file, source_root=src, index=index)
        by_subtype = {e.subtype: e for e in elements}
        assert by_subtype['go_struct'].qualified_name == 'srv.Server'
        # receiver method — parent descriptor is the Server type
        assert by_subtype['go_method'].qualified_name == 'srv.Server.Start'
        # top-level func — parent descriptor is the package
        assert by_subtype['go_function'].qualified_name == 'srv.New'
