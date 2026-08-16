"""Tests for the cross-source SCIP graph (SCIP-everywhere, Phase 2).

The cross-source graph joins multiple sources' SCIP indexes into a single
graph keyed on canonical SCIP symbol IDs. Within each language the symbol
strings are canonical and unambiguous, so cross-source joining is just
string equality. Cross-language joins do not happen — Python's
``scip-python python ...`` and Scala's ``scip-java maven ...`` symbol
strings cannot collide by construction.

Tests use synthetic ``_ScipDoc``/``_ScipSymbol``/``_ScipOccurrence``
intermediates (the same pattern as ``test_scip_extractor.py``); no
protobuf bindings required.

Conventions:
- 0-indexed line/col in the SCIP wire format → 1-indexed in our
  ``CrossSourceSymbol``/``CrossSourceEdge`` types (consistent with
  ``ElementInfo`` in the rest of Ariadne).
- ``range`` may be a 3-tuple (same line) or 4-tuple (multi-line).
- A reference's caller is the *tightest enclosing definition* in the
  same document (the def with the largest ``line_start`` whose range
  contains the reference line).
"""
from __future__ import annotations

from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Fixture builders — each returns a (source_name, ScipIndex, language) triple.
# ---------------------------------------------------------------------------


def _python_scalaproject_index():
    """scalaproject (Python) defines licensing.LicenseService.validate_token."""
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    cls_sym = 'scip-python python scalaproject 0.1 licensing/LicenseService#'
    method_sym = (
        'scip-python python scalaproject 0.1 '
        'licensing/LicenseService#validate_token().'
    )
    doc = _ScipDoc(
        relative_path='licensing.py',
        occurrences=(
            # class LicenseService:                  (line 1, 0-indexed -> 0)
            _ScipOccurrence(
                symbol=cls_sym, range=(0, 6, 0, 20), is_definition=True,
            ),
            # def validate_token(self):              (line 3..5, 0-indexed -> 2..4)
            _ScipOccurrence(
                symbol=method_sym, range=(2, 8, 4, 0), is_definition=True,
            ),
        ),
        symbols=(
            _ScipSymbol(
                symbol=cls_sym, kind='Class',
                display_name='LicenseService',
            ),
            _ScipSymbol(
                symbol=method_sym, kind='Method',
                display_name='validate_token',
            ),
        ),
    )
    return 'scalaproject', ScipIndex(documents=(doc,)), 'python'


def _python_pyproject_index_referencing_scalaproject():
    """biggerproject-backend (Python): auth.login() calls scalaproject's
    LicenseService.validate_token().

    The reference at line 12 (0-indexed: 11) sits inside the body of
    auth.login (defined at lines 11..13, 0-indexed: 10..12). The graph
    must identify auth.login as the caller of the cross-source edge.
    """
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    login_sym = 'scip-python python pyproject 0.1 auth/login().'
    scalaproject_method = (
        'scip-python python scalaproject 0.1 '
        'licensing/LicenseService#validate_token().'
    )

    doc = _ScipDoc(
        relative_path='auth.py',
        occurrences=(
            # def login(req):                        (lines 11..13)
            _ScipOccurrence(
                symbol=login_sym, range=(10, 4, 12, 0), is_definition=True,
            ),
            # call to LicenseService.validate_token  (line 12)
            _ScipOccurrence(
                symbol=scalaproject_method,
                range=(11, 8, 11, 25),
                is_definition=False,
            ),
        ),
        symbols=(
            _ScipSymbol(
                symbol=login_sym, kind='Method', display_name='login',
            ),
        ),
    )
    return 'biggerproject-backend', ScipIndex(documents=(doc,)), 'python'


def _scala_biggerproject_index_referencing_scalaproject_python():
    """Synthetic Scala source whose SCIP references a Python symbol.

    This case is *unrealistic* in real codebases (Scala doesn't import
    Python modules) and is included here to verify the join is keyed on
    SCIP scheme: the Scala index's ``scip-java`` symbols cannot collide
    with the Python index's ``scip-python`` symbols. We rely on this so
    the cross-source consumer never accidentally bridges languages.
    """
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    sm_sym = 'scip-java maven com.biggerproject biggerproject 1 com/biggerproject/SessionManager#refresh().'

    doc = _ScipDoc(
        relative_path='src/main/scala/com/biggerproject/SessionManager.scala',
        occurrences=(
            _ScipOccurrence(
                symbol=sm_sym, range=(20, 6, 22, 0), is_definition=True,
            ),
        ),
        symbols=(
            _ScipSymbol(
                symbol=sm_sym, kind='Method', display_name='refresh',
            ),
        ),
    )
    return 'biggerproject', ScipIndex(documents=(doc,)), 'scala'


def _scala_biggerproject_calling_scala_scalaproject():
    """Same-language Scala→Scala edge: biggerproject.SessionManager.refresh()
    calls scalaproject.LicenseService.validate_token() (Scala, not Python).

    This is the realistic same-language reverse-dep case. The Scala
    scalaproject index defines validate_token; biggerproject references it.
    """
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )

    scalaproject_method = (
        'scip-java maven com.scalaproject scalaproject 1 '
        'com/scalaproject/licensing/LicenseService#validate_token().'
    )
    scalaproject_class = (
        'scip-java maven com.scalaproject scalaproject 1 '
        'com/scalaproject/licensing/LicenseService#'
    )
    sm_sym = (
        'scip-java maven com.biggerproject biggerproject 1 '
        'com/biggerproject/SessionManager#refresh().'
    )

    scalaproject_doc = _ScipDoc(
        relative_path='src/main/scala/com/scalaproject/licensing/LicenseService.scala',
        occurrences=(
            _ScipOccurrence(
                symbol=scalaproject_class, range=(0, 6, 0, 19), is_definition=True,
            ),
            _ScipOccurrence(
                symbol=scalaproject_method, range=(2, 6, 4, 0), is_definition=True,
            ),
        ),
        symbols=(
            _ScipSymbol(
                symbol=scalaproject_class, kind='Class',
                display_name='LicenseService',
            ),
            _ScipSymbol(
                symbol=scalaproject_method, kind='Method',
                display_name='validate_token',
            ),
        ),
    )

    biggerproject_doc = _ScipDoc(
        relative_path='src/main/scala/com/biggerproject/SessionManager.scala',
        occurrences=(
            _ScipOccurrence(
                symbol=sm_sym, range=(20, 6, 22, 0), is_definition=True,
            ),
            _ScipOccurrence(
                symbol=scalaproject_method,
                range=(21, 8, 21, 30),
                is_definition=False,
            ),
        ),
        symbols=(
            _ScipSymbol(
                symbol=sm_sym, kind='Method', display_name='refresh',
            ),
        ),
    )
    return (
        ('scalaproject', ScipIndex(documents=(scalaproject_doc,)), 'scala'),
        ('biggerproject', ScipIndex(documents=(biggerproject_doc,)), 'scala'),
    )


# ---------------------------------------------------------------------------
# Phase 2b — failing tests for CrossSourceGraph.
# ---------------------------------------------------------------------------


class TestCrossSourceGraphConstruction:
    def test_empty_graph_has_no_symbols_or_edges(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        graph = CrossSourceGraph()
        graph.materialize()

        assert graph.has_scip('anything') is False
        assert graph.consumers_of_source('anything') == []

    def test_single_source_registers_definitions_as_symbols(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        name, index, language = _python_scalaproject_index()
        graph = CrossSourceGraph()
        graph.add_source(name, index=index, language=language)
        graph.materialize()

        assert graph.has_scip('scalaproject') is True

        # Two definitions in the fixture: the class and the method
        symbols = graph.symbols_in('scalaproject')
        assert len(symbols) == 2

        method = next(
            s for s in symbols if s.display_name == 'validate_token'
        )
        assert method.source_name == 'scalaproject'
        assert method.language == 'python'
        # qualified_name comes from _qualified_name_from_symbol
        assert method.qualified_name.endswith('LicenseService.validate_token')
        # 1-indexed line numbers (vs. 0-indexed in wire format)
        assert method.line_start == 3
        assert method.line_end == 5


class TestCrossSourceEdges:
    def test_cross_source_reference_produces_one_edge(self) -> None:
        """The simplest case: source B references a definition in source
        A. consumers_of_source('A') returns the cross-source edge."""
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        b_name, b_idx, b_lang = _python_pyproject_index_referencing_scalaproject()

        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.add_source(b_name, index=b_idx, language=b_lang)
        graph.materialize()

        consumers = graph.consumers_of_source('scalaproject')
        assert len(consumers) == 1, (
            'expected exactly one cross-source edge from '
            'biggerproject-backend.auth.login → scalaproject.validate_token'
        )

        edge = consumers[0]
        assert edge.callee.source_name == 'scalaproject'
        assert edge.callee.display_name == 'validate_token'
        assert edge.caller.source_name == 'biggerproject-backend'
        assert edge.caller.display_name == 'login'
        assert edge.confidence == 'exact'
        # Reference at 0-indexed line 11 → 1-indexed line 12
        assert edge.line == 12

    def test_within_source_reference_is_not_a_cross_source_edge(self) -> None:
        """A reference WITHIN scalaproject to its own symbol is a same-source
        edge, not a cross-source consumer. Reverse-augment cares only
        about cross-source consumers."""
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        method_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )
        helper_sym = 'scip-python python scalaproject 0.1 helpers/wrap().'
        cls_sym = 'scip-python python scalaproject 0.1 licensing/LicenseService#'

        # licensing.py: defines class + method
        license_doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=cls_sym, range=(0, 6, 0, 20), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym, range=(2, 8, 4, 0), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=cls_sym, kind='Class',
                    display_name='LicenseService',
                ),
                _ScipSymbol(
                    symbol=method_sym, kind='Method',
                    display_name='validate_token',
                ),
            ),
        )

        # helpers.py: defines wrap, references validate_token (same source)
        helpers_doc = _ScipDoc(
            relative_path='helpers.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=helper_sym, range=(10, 4, 12, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym,
                    range=(11, 8, 11, 25),
                    is_definition=False,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=helper_sym, kind='Method', display_name='wrap',
                ),
            ),
        )

        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(license_doc, helpers_doc)),
            language='python',
        )
        graph.materialize()

        # Same-source ref: helpers.wrap → licensing.validate_token
        # Should NOT appear in consumers_of_source('scalaproject') because
        # both endpoints are in scalaproject.
        assert graph.consumers_of_source('scalaproject') == []

        # But it SHOULD appear in callers_of(method_sym) — that query
        # is about call relationships regardless of source.
        callers = graph.callers_of(method_sym)
        assert len(callers) == 1
        assert callers[0].caller.display_name == 'wrap'

    def test_caller_is_tightest_enclosing_definition(self) -> None:
        """When a reference sits inside both a class (lines 0..50) and a
        method within that class (lines 15..25), the caller must be
        resolved to the method, not the class."""
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        cls_sym = 'scip-python python pyproject 0.1 service/AuthService#'
        method_sym = 'scip-python python pyproject 0.1 service/AuthService#login().'
        upstream_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )

        # Scalaproject defines validate_token
        scalaproject_doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=upstream_sym,
                    range=(2, 8, 4, 0), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=upstream_sym, kind='Method',
                    display_name='validate_token',
                ),
            ),
        )

        # pyproject defines AuthService (lines 1..51) and login (lines 16..26),
        # references upstream at line 21 — inside both
        pyproject_doc = _ScipDoc(
            relative_path='service.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=cls_sym, range=(0, 6, 50, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym,
                    range=(15, 4, 25, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=upstream_sym,
                    range=(20, 8, 20, 30),
                    is_definition=False,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=cls_sym, kind='Class',
                    display_name='AuthService',
                ),
                _ScipSymbol(
                    symbol=method_sym, kind='Method',
                    display_name='login',
                ),
            ),
        )

        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(scalaproject_doc,)),
            language='python',
        )
        graph.add_source(
            'biggerproject-backend',
            index=ScipIndex(documents=(pyproject_doc,)),
            language='python',
        )
        graph.materialize()

        consumers = graph.consumers_of_source('scalaproject')
        assert len(consumers) == 1
        # Caller must be the method, not the enclosing class
        assert consumers[0].caller.display_name == 'login'
        assert consumers[0].caller.kind == 'Method'

    def test_languages_do_not_bridge(self) -> None:
        """The Python scalaproject index and a Scala source must NOT produce
        cross-language edges, because their SCIP symbol schemes differ
        (``scip-python python ...`` vs ``scip-java maven ...``).

        This is a deliberate non-goal of the design (decision #4): no
        partial information across language boundaries.
        """
        from docgen.scip_cross_source import CrossSourceGraph

        py_name, py_idx, py_lang = _python_scalaproject_index()
        # Scala source defining a method but referring to Python
        # scalaproject would have a different SCIP scheme; we use a Scala
        # source with no references to test the language-isolation
        # property.
        sc_name, sc_idx, sc_lang = (
            _scala_biggerproject_index_referencing_scalaproject_python()
        )

        graph = CrossSourceGraph()
        graph.add_source(py_name, index=py_idx, language=py_lang)
        graph.add_source(sc_name, index=sc_idx, language=sc_lang)
        graph.materialize()

        # No edges at all — Scala source has no references; Python
        # source has no callers
        assert graph.consumers_of_source('scalaproject') == []
        assert graph.consumers_of_source('biggerproject') == []

    def test_same_language_jvm_edge_resolves(self) -> None:
        """The realistic JVM→JVM case: biggerproject (Scala) references
        scalaproject (Scala) — symbols share the ``scip-java maven`` scheme
        and resolve precisely."""
        from docgen.scip_cross_source import CrossSourceGraph

        scalaproject, biggerproject = _scala_biggerproject_calling_scala_scalaproject()

        graph = CrossSourceGraph()
        graph.add_source(
            scalaproject[0], index=scalaproject[1], language=scalaproject[2],
        )
        graph.add_source(
            biggerproject[0], index=biggerproject[1], language=biggerproject[2],
        )
        graph.materialize()

        consumers = graph.consumers_of_source('scalaproject')
        assert len(consumers) == 1
        edge = consumers[0]
        assert edge.callee.display_name == 'validate_token'
        assert edge.caller.source_name == 'biggerproject'
        assert edge.caller.display_name == 'refresh'


class TestEnrichDocGraphWithScip:
    """Phase 5 wiring — Library.enrich_doc_graph_with_scip adds
    SCIP-precise call edges to the existing doc_graph table for a
    source whose ``.scip`` is loaded.

    Edges are at file granularity (caller_file → callee_file) with
    edge_type ``'scip_calls'`` to distinguish from the existing
    import-derived ``'imports'`` edges. Both coexist; downstream
    consumers can filter by edge_type as needed.
    """

    def test_adds_scip_call_edges(self, tmp_path: Path) -> None:
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )
        from library import Library

        # Build a within-source SCIP graph: helpers/wrap calls
        # licensing/validate_token (different files in same source).
        method_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )
        helper_sym = 'scip-python python scalaproject 0.1 helpers/wrap().'
        cls_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#'
        )
        license_doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=cls_sym, range=(0, 6, 0, 20),
                    is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym, range=(2, 8, 4, 0),
                    is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=cls_sym, kind='Class',
                    display_name='LicenseService',
                ),
                _ScipSymbol(
                    symbol=method_sym, kind='Method',
                    display_name='validate_token',
                ),
            ),
        )
        helpers_doc = _ScipDoc(
            relative_path='helpers.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=helper_sym, range=(10, 4, 12, 0),
                    is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym, range=(11, 8, 11, 25),
                    is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=helper_sym, kind='Method', display_name='wrap',
            ),),
        )

        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(license_doc, helpers_doc)),
            language='python',
        )
        graph.materialize()

        db_path = tmp_path / 'enrich.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph.save_to(conn)

        # Project root used to resolve relative scip paths to absolute
        source_root = tmp_path / 'scalaproject_root'
        source_root.mkdir()

        added = lib.enrich_doc_graph_with_scip(
            'scalaproject', source_root,
        )
        assert added == 1

        with lib._conn_provider.acquire() as conn:
            rows = conn.execute(
                "SELECT source_id, target_id, edge_type "
                "FROM doc_graph WHERE edge_type = 'scip_calls'",
            ).fetchall()
        lib.close()

        assert len(rows) == 1
        source_id, target_id, edge_type = rows[0]
        assert 'helpers.py' in source_id
        assert 'licensing.py' in target_id

    def test_returns_zero_for_unindexed_source(
        self, tmp_path: Path,
    ) -> None:
        """Source name not in the SCIP graph → no edges added,
        return 0."""
        from library import Library

        db_path = tmp_path / 'empty.db'
        lib = Library(db_path)
        added = lib.enrich_doc_graph_with_scip(
            'ghost_source', tmp_path / 'ghost_root',
        )
        lib.close()
        assert added == 0

    def test_skips_self_edges(self, tmp_path: Path) -> None:
        """An edge where caller and callee live in the SAME file is
        a within-file call. Doc graph is at file granularity — a
        file-edge to itself is noise. Skip."""
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )
        from library import Library

        # Both caller and callee in same file
        a_sym = 'scip-java maven g a 1 com/x/A#a().'
        b_sym = 'scip-java maven g a 1 com/x/B#b().'
        doc = _ScipDoc(
            relative_path='Same.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=a_sym, range=(0, 0, 1, 0),
                    is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=b_sym, range=(2, 0, 3, 0),
                    is_definition=True,
                ),
                # A calls B (both in Same.scala)
                _ScipOccurrence(
                    symbol=b_sym, range=(0, 4, 0, 5),
                    is_definition=False,
                ),
            ),
            symbols=(
                _ScipSymbol(symbol=a_sym, kind='Method', display_name='a'),
                _ScipSymbol(symbol=b_sym, kind='Method', display_name='b'),
            ),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'mysrc', index=ScipIndex(documents=(doc,)), language='scala',
        )
        graph.materialize()

        db_path = tmp_path / 'self.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph.save_to(conn)

        added = lib.enrich_doc_graph_with_scip(
            'mysrc', tmp_path / 'src_root',
        )
        lib.close()
        # caller_file == callee_file → self-edge, skipped
        assert added == 0


class TestBuildGraphScipIntegration:
    """Phase 5 integration: ``Library.build_graph`` accepts an optional
    ``source_name`` and atomically includes SCIP-derived call edges
    when that source has a current SCIP graph in the DB.

    Backwards compat: callers that don't pass ``source_name`` get the
    pre-Phase-5 behavior (ast-grep import edges only).
    """

    def test_build_graph_signature_accepts_source_name(self) -> None:
        from library.graph import GraphMixin
        import inspect

        sig = inspect.signature(GraphMixin.build_graph)
        assert 'source_name' in sig.parameters
        # Optional with None default for backwards compat
        assert sig.parameters['source_name'].default is None

    def test_without_source_name_skips_scip_enrichment(
        self, tmp_path: Path,
    ) -> None:
        """If source_name is omitted, build_graph behaves as before —
        no SCIP edges added even if the SCIP graph IS loaded in DB."""
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )
        from library import Library

        # Set up SCIP data in DB
        m_sym = (
            'scip-python python scalaproject 0.1 lib/foo#bar().'
        )
        h_sym = 'scip-python python scalaproject 0.1 helpers/wrap().'
        doc = _ScipDoc(
            relative_path='lib/foo.py',
            occurrences=(_ScipOccurrence(
                symbol=m_sym, range=(0, 0, 1, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=m_sym, kind='Method', display_name='bar',
            ),),
        )
        helpers_doc = _ScipDoc(
            relative_path='helpers.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=h_sym, range=(0, 0, 1, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=m_sym, range=(0, 4, 0, 5), is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=h_sym, kind='Method', display_name='wrap',
            ),),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(doc, helpers_doc)),
            language='python',
        )
        graph.materialize()

        # Empty source root — build_graph won't find any Python files,
        # so the existing ast-grep walk produces no edges
        source_root = tmp_path / 'src_root'
        source_root.mkdir()

        db_path = tmp_path / 'no_name.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph.save_to(conn)

        # Call build_graph WITHOUT source_name
        lib.build_graph(source_root)

        with lib._conn_provider.acquire() as conn:
            scip_edge_count = conn.execute(
                "SELECT COUNT(*) FROM doc_graph "
                "WHERE edge_type = 'scip_calls'",
            ).fetchone()[0]
        lib.close()

        assert scip_edge_count == 0

    def test_with_source_name_atomically_includes_scip_edges(
        self, tmp_path: Path,
    ) -> None:
        """Passing source_name = 'scalaproject' triggers SCIP enrichment
        as part of build_graph. After the call, doc_graph contains
        BOTH ast-grep imports AND scip_calls edges."""
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )
        from library import Library

        # SCIP setup: helpers calls foo
        m_sym = 'scip-python python scalaproject 0.1 lib/foo#bar().'
        h_sym = 'scip-python python scalaproject 0.1 helpers/wrap().'
        foo_doc = _ScipDoc(
            relative_path='lib/foo.py',
            occurrences=(_ScipOccurrence(
                symbol=m_sym, range=(0, 0, 1, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=m_sym, kind='Method', display_name='bar',
            ),),
        )
        helpers_doc = _ScipDoc(
            relative_path='helpers.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=h_sym, range=(0, 0, 1, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=m_sym, range=(0, 4, 0, 5), is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=h_sym, kind='Method', display_name='wrap',
            ),),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(foo_doc, helpers_doc)),
            language='python',
        )
        graph.materialize()

        source_root = tmp_path / 'src_root'
        source_root.mkdir()

        db_path = tmp_path / 'with_name.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph.save_to(conn)

        # Call build_graph WITH source_name='scalaproject'
        lib.build_graph(source_root, source_name='scalaproject')

        with lib._conn_provider.acquire() as conn:
            scip_edge_count = conn.execute(
                "SELECT COUNT(*) FROM doc_graph "
                "WHERE edge_type = 'scip_calls'",
            ).fetchone()[0]
        lib.close()

        # The wrap → bar SCIP edge made it in
        assert scip_edge_count == 1


class TestEdgesInSource:
    """Phase 5 — within-source edges only. Used by cli_graph to replace
    its ast-grep-derived approximate edges with SCIP-precise ones."""

    def test_returns_within_source_edges(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        # scalaproject only — A.helpers.wrap calls A.LicenseService.validate_token
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        method_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )
        helper_sym = 'scip-python python scalaproject 0.1 helpers/wrap().'
        cls_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#'
        )
        license_doc = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=cls_sym, range=(0, 6, 0, 20),
                    is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym, range=(2, 8, 4, 0),
                    is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=cls_sym, kind='Class',
                    display_name='LicenseService',
                ),
                _ScipSymbol(
                    symbol=method_sym, kind='Method',
                    display_name='validate_token',
                ),
            ),
        )
        helpers_doc = _ScipDoc(
            relative_path='helpers.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=helper_sym, range=(10, 4, 12, 0),
                    is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=method_sym, range=(11, 8, 11, 25),
                    is_definition=False,
                ),
            ),
            symbols=(_ScipSymbol(
                symbol=helper_sym, kind='Method', display_name='wrap',
            ),),
        )

        graph = CrossSourceGraph()
        graph.add_source(
            'scalaproject',
            index=ScipIndex(documents=(license_doc, helpers_doc)),
            language='python',
        )
        graph.materialize()

        within = graph.edges_in_source('scalaproject')
        assert len(within) == 1
        edge = within[0]
        assert edge.caller.display_name == 'wrap'
        assert edge.callee.display_name == 'validate_token'

    def test_excludes_cross_source_edges(self) -> None:
        """A→B where A is in source X and B is in source Y: edges_in_source(X)
        returns empty (the edge crosses the source boundary)."""
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        b_name, b_idx, b_lang = _python_pyproject_index_referencing_scalaproject()
        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.add_source(b_name, index=b_idx, language=b_lang)
        graph.materialize()

        # Edge from pyproject→scalaproject is cross-source; not a within-source
        # edge for either side.
        assert graph.edges_in_source('scalaproject') == []
        assert graph.edges_in_source('biggerproject-backend') == []

    def test_unknown_source_returns_empty(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        graph = CrossSourceGraph()
        graph.materialize()
        assert graph.edges_in_source('ghost') == []


class TestCallersAndCallees:
    def test_callers_of_returns_all_call_sites(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        b_name, b_idx, b_lang = _python_pyproject_index_referencing_scalaproject()

        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.add_source(b_name, index=b_idx, language=b_lang)
        graph.materialize()

        method_sym = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )
        callers = graph.callers_of(method_sym)
        assert len(callers) == 1
        assert callers[0].caller.display_name == 'login'

    def test_callees_of_returns_outgoing_edges(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        b_name, b_idx, b_lang = _python_pyproject_index_referencing_scalaproject()

        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.add_source(b_name, index=b_idx, language=b_lang)
        graph.materialize()

        login_sym = 'scip-python python pyproject 0.1 auth/login().'
        callees = graph.callees_of(login_sym)
        assert len(callees) == 1
        assert callees[0].callee.display_name == 'validate_token'


class TestSymbolResolution:
    """Permissive symbol resolution per decision #3."""

    def test_resolve_exact_qualified_name(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.materialize()

        result = graph.resolve_symbol('licensing.LicenseService.validate_token')
        assert result.match_tier == 'exact'
        assert result.symbol.display_name == 'validate_token'

    def test_resolve_suffix_match_unique(self) -> None:
        """Typing the trailing component of a qualified name resolves
        to the unique tail-match."""
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.materialize()

        result = graph.resolve_symbol('validate_token')
        assert result.match_tier == 'suffix'
        assert result.symbol.display_name == 'validate_token'

    def test_resolve_no_match_returns_zero_candidates(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.materialize()

        result = graph.resolve_symbol('totally_nonexistent_symbol')
        assert result.symbol is None
        assert result.candidates == ()


class TestSchema:
    """Phase 2d — three new tables added to ariadne.db when Library is
    initialized: ``scip_symbols``, ``scip_edges``, ``scip_index_state``.
    """

    def test_scip_symbols_table_present(self, tmp_path: Path) -> None:
        from library import Library

        lib = Library(tmp_path / 'a.db')
        with lib._conn_provider.acquire() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='scip_symbols'",
            ).fetchone()
            cols = {
                r[1] for r in conn.execute(
                    'PRAGMA table_info(scip_symbols)',
                )
            }
        lib.close()

        assert row is not None, 'scip_symbols table not created'
        assert {
            'canonical_id', 'source_name', 'language', 'file',
            'line_start', 'line_end', 'kind', 'display_name',
            'qualified_name', 'parent_qualified_name',
        } <= cols

    def test_scip_edges_table_present(self, tmp_path: Path) -> None:
        from library import Library

        lib = Library(tmp_path / 'a.db')
        with lib._conn_provider.acquire() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='scip_edges'",
            ).fetchone()
            cols = {
                r[1] for r in conn.execute('PRAGMA table_info(scip_edges)')
            }
        lib.close()

        assert row is not None, 'scip_edges table not created'
        assert {
            'caller_canonical_id', 'callee_canonical_id', 'edge_type',
            'file', 'line', 'confidence',
        } <= cols

    def test_scip_index_state_table_present(self, tmp_path: Path) -> None:
        from library import Library

        lib = Library(tmp_path / 'a.db')
        with lib._conn_provider.acquire() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='scip_index_state'",
            ).fetchone()
            cols = {
                r[1] for r in conn.execute(
                    'PRAGMA table_info(scip_index_state)',
                )
            }
        lib.close()

        assert row is not None, 'scip_index_state table not created'
        assert {
            'source_name', 'scip_path', 'file_sha256', 'indexed_at',
            'indexer_version',
        } <= cols

    def test_init_is_idempotent(self, tmp_path: Path) -> None:
        """Re-opening the same DB does not error or duplicate tables."""
        from library import Library

        db_path = tmp_path / 'a.db'
        Library(db_path).close()
        # Re-open — must succeed
        Library(db_path).close()

    def test_api_endpoints_table_present(self, tmp_path: Path) -> None:
        """Phase 7a — Wave 4 (API surface tracking). The
        api_endpoints table holds Swagger/OpenAPI- or pattern-derived
        producer-side endpoint declarations, joining to scip_symbols
        via producer_symbol_id."""
        from library import Library

        lib = Library(tmp_path / 'a.db')
        with lib._conn_provider.acquire() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='api_endpoints'",
            ).fetchone()
            cols = {
                r[1] for r in conn.execute(
                    'PRAGMA table_info(api_endpoints)',
                )
            }
        lib.close()

        assert row is not None, 'api_endpoints table not created'
        assert {
            'endpoint_id', 'source_name', 'http_method',
            'path_template', 'producer_symbol_id', 'resolution_source',
        } <= cols

    def test_api_calls_table_present(self, tmp_path: Path) -> None:
        """Phase 7a — api_calls holds consumer-side HTTP call sites
        joining to api_endpoints + (via consumer_symbol_id) to
        scip_symbols. Confidence column captures whether URL
        resolution was literal, constant, templated, or ambiguous."""
        from library import Library

        lib = Library(tmp_path / 'a.db')
        with lib._conn_provider.acquire() as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master "
                "WHERE type='table' AND name='api_calls'",
            ).fetchone()
            cols = {
                r[1] for r in conn.execute('PRAGMA table_info(api_calls)')
            }
        lib.close()

        assert row is not None, 'api_calls table not created'
        assert {
            'consumer_symbol_id', 'endpoint_id', 'call_site_file',
            'call_site_line', 'resolution_source', 'confidence',
        } <= cols

    def test_api_tables_preserve_data_on_re_init(
        self, tmp_path: Path,
    ) -> None:
        """Schema init is idempotent — existing api_endpoints + api_calls
        rows survive a fresh Library() open."""
        from library import Library

        db_path = tmp_path / 'persist.db'

        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            conn.execute(
                'INSERT INTO api_endpoints VALUES (?, ?, ?, ?, ?, ?)',
                (
                    'mysrc:GET /api/login', 'mysrc', 'GET',
                    '/api/login', None, 'swagger',
                ),
            )
        lib.close()

        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            count = conn.execute(
                'SELECT COUNT(*) FROM api_endpoints',
            ).fetchone()[0]
        lib.close()

        assert count == 1

    def test_init_preserves_existing_scip_data_on_re_open(
        self, tmp_path: Path,
    ) -> None:
        """Schema init is the contract: re-opening a DB that already
        has scip_symbols rows must NOT drop those rows. ``CREATE TABLE
        IF NOT EXISTS`` plus ``CREATE INDEX IF NOT EXISTS`` is the
        idempotent pattern; this test guards against any future change
        that breaks data preservation (e.g., switching to ``CREATE
        TABLE`` without IF NOT EXISTS, or adding a destructive
        migration path).
        """
        from library import Library

        db_path = tmp_path / 'persist.db'

        # First open: populate scip_symbols and scip_edges
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            conn.execute(
                'INSERT INTO scip_symbols (canonical_id, source_name, language, file, line_start, line_end, kind, display_name, qualified_name, parent_qualified_name) '
                'VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                (
                    'sym1', 'mysrc', 'python', 'a.py',
                    1, 2, 'Method', 'foo', 'a.foo', 'a',
                ),
            )
            conn.execute(
                'INSERT INTO scip_edges VALUES (?, ?, ?, ?, ?, ?)',
                ('sym1', 'sym1', 'call', 'a.py', 1, 'exact'),
            )
            conn.execute(
                'INSERT INTO scip_index_state VALUES (?, ?, ?, ?, ?)',
                (
                    'mysrc', '/path/to/.scip',
                    'abc123' * 10, '2026-05-06T00:00:00Z',
                    'scip-python/1.5.0',
                ),
            )
        lib.close()

        # Second open — runs init_scip_schema again
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            sym_count = conn.execute(
                'SELECT COUNT(*) FROM scip_symbols',
            ).fetchone()[0]
            edge_count = conn.execute(
                'SELECT COUNT(*) FROM scip_edges',
            ).fetchone()[0]
            state_count = conn.execute(
                'SELECT COUNT(*) FROM scip_index_state',
            ).fetchone()[0]
        lib.close()

        assert sym_count == 1, 'scip_symbols data wiped on re-open'
        assert edge_count == 1, 'scip_edges data wiped on re-open'
        assert state_count == 1, 'scip_index_state data wiped on re-open'


class TestPersistence:
    """Phase 2f — CrossSourceGraph.save_to / load_from preserve state
    across an open/close cycle."""

    def test_save_then_load_round_trip(self, tmp_path: Path) -> None:
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        scalaproject, biggerproject = (
            _scala_biggerproject_calling_scala_scalaproject()
        )

        graph_a = CrossSourceGraph()
        graph_a.add_source(
            scalaproject[0], index=scalaproject[1], language=scalaproject[2],
        )
        graph_a.add_source(
            biggerproject[0], index=biggerproject[1], language=biggerproject[2],
        )
        graph_a.materialize()

        # Save to DB
        db_path = tmp_path / 'persist.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph_a.save_to(conn)
        lib.close()

        # Reload from DB into a fresh graph
        lib = Library(db_path)
        graph_b = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph_b.load_from(conn)
        lib.close()

        # Assert equivalence on the externally-visible state
        a_consumers = graph_a.consumers_of_source('scalaproject')
        b_consumers = graph_b.consumers_of_source('scalaproject')
        assert len(a_consumers) == len(b_consumers) == 1
        assert (
            a_consumers[0].callee.canonical_id
            == b_consumers[0].callee.canonical_id
        )
        assert (
            a_consumers[0].caller.display_name
            == b_consumers[0].caller.display_name
        )

    def test_save_replaces_edges_not_just_symbols(self, tmp_path: Path) -> None:
        """When ``save_to`` deletes a source's rows before inserting new
        ones, BOTH symbols and edges must be cleared. The edges delete
        depends on a join through scip_symbols — so it must run BEFORE
        the symbols delete, or the join returns empty and old edges
        leak through (orphaned, but still present)."""
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        # Initial state: a producer source + a consumer source with a
        # cross-source edge between them.
        scalaproject, biggerproject = (
            _scala_biggerproject_calling_scala_scalaproject()
        )

        graph_initial = CrossSourceGraph()
        graph_initial.add_source(
            scalaproject[0], index=scalaproject[1], language=scalaproject[2],
        )
        graph_initial.add_source(
            biggerproject[0], index=biggerproject[1], language=biggerproject[2],
        )
        graph_initial.materialize()
        assert len(graph_initial.consumers_of_source('scalaproject')) == 1

        db_path = tmp_path / 'edges.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph_initial.save_to(conn)
            edge_count_initial = conn.execute(
                'SELECT COUNT(*) FROM scip_edges',
            ).fetchone()[0]
        lib.close()
        assert edge_count_initial == 1

        # Now: re-save biggerproject with NO references to scalaproject
        # (e.g., the user removed the call site). The edge from
        # biggerproject → scalaproject should be gone afterwards.
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        sm_sym = (
            'scip-java maven com.biggerproject biggerproject 1 '
            'com/biggerproject/SessionManager#refresh().'
        )
        # New biggerproject: just defines refresh(), no reference to scalaproject
        biggerproject_doc_new = _ScipDoc(
            relative_path='src/main/scala/com/biggerproject/SessionManager.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=sm_sym, range=(20, 6, 22, 0),
                    is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=sm_sym, kind='Method', display_name='refresh',
                ),
            ),
        )
        graph_new = CrossSourceGraph()
        graph_new.add_source(
            'biggerproject',
            index=ScipIndex(documents=(biggerproject_doc_new,)),
            language='scala',
        )
        graph_new.materialize()

        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph_new.save_to(conn)
            remaining_edges = conn.execute(
                'SELECT COUNT(*) FROM scip_edges',
            ).fetchone()[0]
        lib.close()

        assert remaining_edges == 0, (
            'old biggerproject→scalaproject edge still in DB after re-save — '
            'edges DELETE must run before symbols DELETE, otherwise the '
            'join returns empty rows and edges orphan'
        )

    def test_save_to_with_zero_sources_is_no_op(self, tmp_path: Path) -> None:
        """A graph with no registered sources should not modify the DB
        when save_to is called. Used for accidental empty-pipeline
        scenarios."""
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        # Set up DB with existing data
        scalaproject, _ = _scala_biggerproject_calling_scala_scalaproject()
        primed = CrossSourceGraph()
        primed.add_source(
            scalaproject[0], index=scalaproject[1], language=scalaproject[2],
        )
        primed.materialize()

        db_path = tmp_path / 'empty.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            primed.save_to(conn)
            count_before = conn.execute(
                'SELECT COUNT(*) FROM scip_symbols',
            ).fetchone()[0]
        lib.close()
        assert count_before > 0

        # Now save an empty graph — must not delete anything
        empty = CrossSourceGraph()
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            empty.save_to(conn)
            count_after = conn.execute(
                'SELECT COUNT(*) FROM scip_symbols',
            ).fetchone()[0]
        lib.close()
        assert count_after == count_before

    def test_load_from_empty_db_produces_empty_graph(self, tmp_path: Path) -> None:
        """Loading a CrossSourceGraph from a fresh DB produces a graph
        with no symbols, no edges, and ``has_scip`` returning False for
        any source name."""
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = tmp_path / 'fresh.db'
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        assert graph.has_scip('scalaproject') is False
        assert graph.consumers_of_source('scalaproject') == []
        assert graph.symbols_in('scalaproject') == ()


    def test_save_replaces_existing_state(self, tmp_path: Path) -> None:
        """Re-saving a graph deletes prior rows for the registered
        sources before inserting — so a subsequent load reflects the
        latest materialization, not an additive merge."""
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )
        from library import Library

        # First state: scalaproject defines a single method
        sym1 = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#validate_token().'
        )
        doc1 = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym1, range=(2, 8, 4, 0), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=sym1, kind='Method',
                    display_name='validate_token',
                ),
            ),
        )
        graph_a = CrossSourceGraph()
        graph_a.add_source(
            'scalaproject',
            index=ScipIndex(documents=(doc1,)),
            language='python',
        )
        graph_a.materialize()

        db_path = tmp_path / 'replace.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph_a.save_to(conn)
        lib.close()

        # Second state: validate_token is gone, replaced by a new method
        sym2 = (
            'scip-python python scalaproject 0.1 '
            'licensing/LicenseService#refresh().'
        )
        doc2 = _ScipDoc(
            relative_path='licensing.py',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym2, range=(2, 8, 4, 0), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=sym2, kind='Method', display_name='refresh',
                ),
            ),
        )
        graph_b = CrossSourceGraph()
        graph_b.add_source(
            'scalaproject',
            index=ScipIndex(documents=(doc2,)),
            language='python',
        )
        graph_b.materialize()

        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph_b.save_to(conn)
        lib.close()

        # Reload — only the second state's symbols should appear
        lib = Library(db_path)
        graph_c = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph_c.load_from(conn)
        lib.close()

        names = {s.display_name for s in graph_c.symbols_in('scalaproject')}
        assert names == {'refresh'}, (
            'old validate_token should not survive a re-save'
        )


class TestManifestLoader:
    """Contract for ``load_source_from_manifest`` — Phase 2g.

    Reads ``<source_root>/.ariadne/manifest.json`` and registers each
    declared SCIP index with the graph under a single source_name.
    """

    def _write_manifest(self, source_root: Path, indexers: list) -> None:
        import json
        manifest_dir = source_root / '.ariadne'
        manifest_dir.mkdir(exist_ok=True)
        (manifest_dir / 'manifest.json').write_text(
            json.dumps({
                'ariadne_version': '1',
                'source_name': 'mysrc',
                'indexers': indexers,
            }),
            encoding='utf-8',
        )

    def test_loads_single_indexer_entry(self, tmp_path: Path) -> None:
        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )

        scalaproject, _ = _scala_biggerproject_calling_scala_scalaproject()
        # The fake factory returns the synthetic ScipIndex regardless
        # of the path argument — no disk I/O for this test.
        synthetic_index = scalaproject[1]

        def fake_factory(path, *, repo, max_staleness_days):
            return synthetic_index

        self._write_manifest(tmp_path, [{
            'kind': 'java',
            'cwd': '.',
            'scip_path': 'intermediate/index-java.scip',
        }])

        graph = CrossSourceGraph()
        load_source_from_manifest(
            graph, 'scalaproject', tmp_path, index_factory=fake_factory,
        )
        graph.materialize()

        assert graph.has_scip('scalaproject') is True
        # Symbols from the synthetic Scala index landed
        names = {s.display_name for s in graph.symbols_in('scalaproject')}
        assert 'LicenseService' in names
        assert 'validate_token' in names

    def test_loads_multiple_indexer_entries_under_one_source(
        self, tmp_path: Path,
    ) -> None:
        """Polyglot: one source_name, multiple indexer .scip files.
        Each entry registers under the SAME source_name, so the graph
        sees them as facets of the same source."""
        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )

        # Two synthetic indexes — one Scala-shape, one Python-shape
        _, scala_idx, _ = _scala_biggerproject_index_referencing_scalaproject_python()
        _, py_idx, _ = _python_scalaproject_index()

        # Map paths in manifest to the correct synthetic index
        index_map = {
            'intermediate/scala.scip': scala_idx,
            'intermediate/python.scip': py_idx,
        }

        def fake_factory(path, *, repo, max_staleness_days):
            rel = str(path.relative_to(tmp_path / '.ariadne'))
            return index_map[rel]

        self._write_manifest(tmp_path, [
            {'kind': 'java', 'cwd': '.',
             'scip_path': 'intermediate/scala.scip'},
            {'kind': 'python', 'cwd': 'scripts',
             'scip_path': 'intermediate/python.scip'},
        ])

        graph = CrossSourceGraph()
        load_source_from_manifest(
            graph, 'polyglot', tmp_path, index_factory=fake_factory,
        )
        graph.materialize()

        # Both indexes' symbols belong to the same source_name
        symbols = graph.symbols_in('polyglot')
        languages = {s.language for s in symbols}
        # Scala via 'java' kind → 'scala' language; python kind → 'python'
        assert 'scala' in languages
        assert 'python' in languages

    def test_skips_entries_without_scip_path(self, tmp_path: Path) -> None:
        """A manifest produced by ``ariadne discover`` BEFORE
        ``ariadne index`` ran has indexer entries but no scip_path.
        Loader skips them gracefully — partial state is allowed."""
        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )

        called = []

        def fake_factory(path, *, repo, max_staleness_days):
            called.append(path)
            raise AssertionError(
                'factory should not be called for entries without scip_path'
            )

        self._write_manifest(tmp_path, [
            {'kind': 'java', 'cwd': '.'},      # no scip_path yet
            {'kind': 'python', 'cwd': 'src'},   # same
        ])

        graph = CrossSourceGraph()
        # Should not raise
        load_source_from_manifest(
            graph, 'pending', tmp_path, index_factory=fake_factory,
        )
        assert called == []
        # No source registered (no actual indexes loaded)
        assert graph.has_scip('pending') is False

    def test_missing_manifest_raises_file_not_found(
        self, tmp_path: Path,
    ) -> None:
        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )

        graph = CrossSourceGraph()
        with pytest.raises(FileNotFoundError) as exc_info:
            load_source_from_manifest(graph, 'missing', tmp_path)
        # Error message tells the user how to recover
        assert 'discover' in str(exc_info.value)

    def test_unknown_kind_raises_value_error(self, tmp_path: Path) -> None:
        """A manifest with an indexer kind we don't recognize fails
        loud (not silently skipped)."""
        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )

        self._write_manifest(tmp_path, [
            {'kind': 'rust', 'cwd': '.', 'scip_path': 'foo.scip'},
        ])

        graph = CrossSourceGraph()
        with pytest.raises(ValueError) as exc_info:
            load_source_from_manifest(graph, 'mysrc', tmp_path)
        assert 'rust' in str(exc_info.value)

    def test_vue_mapping_translates_paths_and_lines(
        self, tmp_path: Path,
    ) -> None:
        """When a manifest entry declares ``vue_mapping``, the loader
        reads the JSON map and translates SCIP paths + line numbers
        back to the original ``.vue`` files. A symbol defined in
        ``Foo.vue.script.js:line=5`` (extracted) becomes
        ``Foo.vue:line=(5+offset)`` in the graph.
        """
        import json

        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        # Synthetic SCIP referring to the EXTRACTED companion path
        sym = (
            'scip-typescript npm frontend 0.1 '
            'src/components/Foo.vue.script#login().'
        )
        doc = _ScipDoc(
            relative_path='src/components/Foo.vue.script.js',
            occurrences=(
                _ScipOccurrence(
                    symbol=sym, range=(4, 0, 6, 0), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=sym, kind='Method', display_name='login',
                ),
            ),
        )
        synthetic_index = ScipIndex(documents=(doc,))

        def fake_factory(path, *, repo, max_staleness_days):
            return synthetic_index

        # Manifest declares vue_mapping path
        manifest_dir = tmp_path / '.ariadne'
        manifest_dir.mkdir()
        (manifest_dir / 'manifest.json').write_text(
            json.dumps({
                'ariadne_version': '1',
                'source_name': 'frontend',
                'indexers': [{
                    'kind': 'typescript',
                    'cwd': '.',
                    'scip_path': 'intermediate/index-ts.scip',
                    'vue_mapping': 'intermediate/vue-mapping.json',
                }],
            }),
            encoding='utf-8',
        )
        (manifest_dir / 'intermediate').mkdir()
        (manifest_dir / 'intermediate' / 'vue-mapping.json').write_text(
            json.dumps({
                'src/components/Foo.vue.script.js': {
                    'original': 'src/components/Foo.vue',
                    'line_offset': 10,
                    'block_type': 'script',
                },
            }),
            encoding='utf-8',
        )

        graph = CrossSourceGraph()
        load_source_from_manifest(
            graph, 'frontend', tmp_path, index_factory=fake_factory,
        )
        graph.materialize()

        symbols = graph.symbols_in('frontend')
        assert len(symbols) == 1
        s = symbols[0]
        # Path translated: .vue.script.js → .vue
        assert s.file == 'src/components/Foo.vue'
        # Line numbers offset: was 0-indexed lines (4..6) → 1-indexed (5..7)
        # plus the +10 vue offset → 1-indexed (15..17)
        assert s.line_start == 15
        assert s.line_end == 17

    def test_paths_not_in_vue_mapping_pass_through_unchanged(
        self, tmp_path: Path,
    ) -> None:
        """A doc whose ``relative_path`` is not in vue-mapping (e.g., a
        regular .ts file alongside .vue) must not be translated."""
        import json

        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        sym = 'scip-typescript npm frontend 0.1 src/utils#parse().'
        doc = _ScipDoc(
            relative_path='src/utils.ts',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 2, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='parse',
            ),),
        )

        def fake_factory(path, *, repo, max_staleness_days):
            return ScipIndex(documents=(doc,))

        manifest_dir = tmp_path / '.ariadne'
        manifest_dir.mkdir()
        (manifest_dir / 'manifest.json').write_text(
            json.dumps({
                'ariadne_version': '1', 'source_name': 'frontend',
                'indexers': [{
                    'kind': 'typescript', 'cwd': '.',
                    'scip_path': 'intermediate/index-ts.scip',
                    'vue_mapping': 'intermediate/vue-mapping.json',
                }],
            }),
            encoding='utf-8',
        )
        (manifest_dir / 'intermediate').mkdir()
        (manifest_dir / 'intermediate' / 'vue-mapping.json').write_text(
            json.dumps({
                'src/components/Other.vue.script.js': {
                    'original': 'src/components/Other.vue',
                    'line_offset': 5,
                },
            }),
            encoding='utf-8',
        )

        graph = CrossSourceGraph()
        load_source_from_manifest(
            graph, 'frontend', tmp_path, index_factory=fake_factory,
        )
        graph.materialize()

        symbols = graph.symbols_in('frontend')
        assert len(symbols) == 1
        # File and lines untouched (not in mapping)
        assert symbols[0].file == 'src/utils.ts'
        assert symbols[0].line_start == 1  # 0-indexed 0 → 1-indexed

    def test_propagates_scip_load_errors(self, tmp_path: Path) -> None:
        """If ``ScipIndex.load`` (the real factory) raises (missing
        file, stale, corrupt), the error propagates without being
        swallowed."""
        from docgen.scip_cross_source import (
            CrossSourceGraph, load_source_from_manifest,
        )
        from docgen.scip_config import ScipUnavailableError

        def failing_factory(path, *, repo, max_staleness_days):
            raise ScipUnavailableError(
                repo=repo, reason='index_missing',
            )

        self._write_manifest(tmp_path, [
            {'kind': 'python', 'cwd': '.',
             'scip_path': 'intermediate/index.scip'},
        ])

        graph = CrossSourceGraph()
        with pytest.raises(ScipUnavailableError):
            load_source_from_manifest(
                graph, 'mysrc', tmp_path, index_factory=failing_factory,
            )


class TestDeadCodeSignals:
    def test_zero_reference_symbols_in_isolated_source(self) -> None:
        """A source with definitions that nothing references should
        report all of them as zero-reference."""
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.materialize()

        unused = graph.symbols_with_zero_references('scalaproject')
        # Both class and method are unused (no references)
        names = {s.display_name for s in unused}
        assert names == {'LicenseService', 'validate_token'}

    def test_referenced_symbol_is_not_zero_reference(self) -> None:
        from docgen.scip_cross_source import CrossSourceGraph

        a_name, a_idx, a_lang = _python_scalaproject_index()
        b_name, b_idx, b_lang = _python_pyproject_index_referencing_scalaproject()

        graph = CrossSourceGraph()
        graph.add_source(a_name, index=a_idx, language=a_lang)
        graph.add_source(b_name, index=b_idx, language=b_lang)
        graph.materialize()

        unused = graph.symbols_with_zero_references('scalaproject')
        names = {s.display_name for s in unused}
        # validate_token is referenced by pyproject, so it's not unused
        assert 'validate_token' not in names
        # The class itself is still unused — no one references LicenseService
        # directly (only its method)
        assert 'LicenseService' in names
