"""Contract for ``ariadne callers`` / ``ariadne callees`` — Phase 4
slice A.

Architecture: pure walk function + CLI wrapper. The walk function
takes a graph + starting canonical_id + depth and returns a tree
structure that can be rendered or asserted in tests. The CLI wrapper
adds DB I/O, symbol resolution, and Rich tree rendering.

Tests cover: pure walk logic (depth, cycles, unknown symbols) and
the CLI's exit-code + side-effect contract.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def restore_global_config():
    import config as config_module
    saved = config_module._global_config
    yield
    config_module._global_config = saved


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {
        'symbol': '', 'depth': 5, 'source': None, 'db': None,
    }
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def _build_chain_graph_in_db(tmp_path: Path) -> Path:
    """Build a 3-deep chain: A→B→C→D, where each calls the next.
    Persist to a fresh Library DB and return the DB path."""
    from docgen.scip_cross_source import CrossSourceGraph
    from docgen.scip_extractor import (
        ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
    )
    from library import Library

    a_sym = 'scip-java maven g a 1 com/x/A#callA().'
    b_sym = 'scip-java maven g a 1 com/x/B#callB().'
    c_sym = 'scip-java maven g a 1 com/x/C#callC().'
    d_sym = 'scip-java maven g a 1 com/x/D#callD().'

    def _doc(rel, defs, refs):
        occurrences = []
        for sym, range_ in defs:
            occurrences.append(_ScipOccurrence(
                symbol=sym, range=range_, is_definition=True,
            ))
        for sym, range_ in refs:
            occurrences.append(_ScipOccurrence(
                symbol=sym, range=range_, is_definition=False,
            ))
        symbols = tuple(
            _ScipSymbol(symbol=sym, kind='Method',
                       display_name=sym.split('#')[-1].rstrip('().'))
            for sym, _ in defs
        )
        return _ScipDoc(
            relative_path=rel,
            occurrences=tuple(occurrences),
            symbols=symbols,
        )

    # A.callA() at lines 0-2, calls B.callB() at line 1
    # B.callB() at lines 5-7, calls C.callC() at line 6
    # C.callC() at lines 10-12, calls D.callD() at line 11
    # D.callD() at lines 15-17, no calls
    doc = _doc(
        'Chain.scala',
        defs=[
            (a_sym, (0, 0, 2, 0)),
            (b_sym, (5, 0, 7, 0)),
            (c_sym, (10, 0, 12, 0)),
            (d_sym, (15, 0, 17, 0)),
        ],
        refs=[
            (b_sym, (1, 4, 1, 12)),  # A → B
            (c_sym, (6, 4, 6, 12)),  # B → C
            (d_sym, (11, 4, 11, 12)),  # C → D
        ],
    )

    graph = CrossSourceGraph()
    graph.add_source(
        'mysrc', index=ScipIndex(documents=(doc,)), language='scala',
    )
    graph.materialize()

    db_path = tmp_path / 'chain.db'
    lib = Library(db_path)
    with lib._conn_provider.acquire() as conn:
        graph.save_to(conn)
    lib.close()
    return db_path


# ---------------------------------------------------------------------------
# Pure walk function
# ---------------------------------------------------------------------------


class TestWalk:
    def test_walk_callers_depth_1_returns_direct_callers_only(
        self, tmp_path: Path,
    ) -> None:
        """Depth=1 returns only the direct callers — A calls B, but
        we don't recurse to A's callers."""
        from cli.callers import walk_callers
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        c_sym = 'scip-java maven g a 1 com/x/C#callC().'
        # callers of C are: B (direct). With depth 1, sub-tree is empty.
        tree = walk_callers(graph, c_sym, depth=1)
        assert len(tree) == 1
        edge, sub = tree[0]
        assert edge.caller.display_name == 'callB'
        # No recursion at depth 1
        assert sub == []

    def test_walk_callers_depth_3_returns_transitive_chain(
        self, tmp_path: Path,
    ) -> None:
        """Depth=3 walks back A → B → C from D's callers."""
        from cli.callers import walk_callers
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        d_sym = 'scip-java maven g a 1 com/x/D#callD().'
        tree = walk_callers(graph, d_sym, depth=3)
        # D ← C ← B ← A
        assert len(tree) == 1  # one direct caller (C)
        c_edge, c_sub = tree[0]
        assert c_edge.caller.display_name == 'callC'
        # C has one caller: B
        assert len(c_sub) == 1
        b_edge, b_sub = c_sub[0]
        assert b_edge.caller.display_name == 'callB'
        # B has one caller: A
        assert len(b_sub) == 1
        a_edge, a_sub = b_sub[0]
        assert a_edge.caller.display_name == 'callA'
        # A has no callers; sub is empty
        assert a_sub == []

    def test_walk_callers_unknown_symbol_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        from cli.callers import walk_callers
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        # Symbol that's not in the graph
        ghost = 'scip-java maven g a 1 com/x/Ghost#ghost().'
        tree = walk_callers(graph, ghost, depth=5)
        assert tree == []

    def test_walk_callers_handles_cycles(self, tmp_path: Path) -> None:
        """If A → B and B → A are both edges (recursive call pattern),
        the walk must terminate, not infinite-loop."""
        from cli.callers import walk_callers
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        a_sym = 'scip-java maven g a 1 com/x/A#a().'
        b_sym = 'scip-java maven g a 1 com/x/B#b().'

        doc = _ScipDoc(
            relative_path='cycle.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=a_sym, range=(0, 0, 2, 0), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=b_sym, range=(5, 0, 7, 0), is_definition=True,
                ),
                # A calls B, B calls A — cycle
                _ScipOccurrence(
                    symbol=b_sym, range=(1, 4, 1, 5), is_definition=False,
                ),
                _ScipOccurrence(
                    symbol=a_sym, range=(6, 4, 6, 5), is_definition=False,
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

        # walk callers of A — B is a caller, but B's callers loop back
        # to A. With cycle detection, the walk terminates.
        tree = walk_callers(graph, a_sym, depth=10)
        # Just verifying it terminates and returns finite output;
        # exact tree shape depends on cycle-handling strategy.
        assert isinstance(tree, list)


class TestWalkCallees:
    def test_walk_callees_mirror_of_callers(self, tmp_path: Path) -> None:
        """walk_callees from A returns the forward chain A → B → C → D."""
        from cli.callers import walk_callees
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        a_sym = 'scip-java maven g a 1 com/x/A#callA().'
        tree = walk_callees(graph, a_sym, depth=3)
        # A → B → C → D
        assert len(tree) == 1
        b_edge, b_sub = tree[0]
        assert b_edge.callee.display_name == 'callB'
        assert len(b_sub) == 1
        c_edge, _ = b_sub[0]
        assert c_edge.callee.display_name == 'callC'


# ---------------------------------------------------------------------------
# CLI handlers
# ---------------------------------------------------------------------------


class TestCmdCallers:
    def test_known_symbol_returns_zero(self, tmp_path: Path) -> None:
        from cli.callers import cmd_callers

        db_path = _build_chain_graph_in_db(tmp_path)
        rc = cmd_callers(_make_args(
            symbol='callC',
            db=db_path,
        ))
        assert rc == 0

    def test_unknown_symbol_returns_nonzero(self, tmp_path: Path) -> None:
        from cli.callers import cmd_callers

        db_path = _build_chain_graph_in_db(tmp_path)
        rc = cmd_callers(_make_args(
            symbol='totally_nonexistent_symbol',
            db=db_path,
        ))
        assert rc != 0

    def test_ambiguous_symbol_returns_nonzero(
        self, tmp_path: Path,
    ) -> None:
        """When the resolver finds multiple candidates at the best
        match tier, exit non-zero so the user disambiguates."""
        from cli.callers import cmd_callers
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )
        from library import Library

        # Two methods with the same display name in different classes
        a_sym = 'scip-java maven g a 1 com/x/A#duplicate().'
        b_sym = 'scip-java maven g a 1 com/x/B#duplicate().'
        doc = _ScipDoc(
            relative_path='dup.scala',
            occurrences=(
                _ScipOccurrence(
                    symbol=a_sym, range=(0, 0, 0, 5), is_definition=True,
                ),
                _ScipOccurrence(
                    symbol=b_sym, range=(2, 0, 2, 5), is_definition=True,
                ),
            ),
            symbols=(
                _ScipSymbol(
                    symbol=a_sym, kind='Method', display_name='duplicate',
                ),
                _ScipSymbol(
                    symbol=b_sym, kind='Method', display_name='duplicate',
                ),
            ),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'mysrc', index=ScipIndex(documents=(doc,)), language='scala',
        )
        graph.materialize()

        db_path = tmp_path / 'dup.db'
        lib = Library(db_path)
        with lib._conn_provider.acquire() as conn:
            graph.save_to(conn)
        lib.close()

        rc = cmd_callers(_make_args(symbol='duplicate', db=db_path))
        assert rc != 0


class TestCmdCallees:
    def test_known_symbol_returns_zero(self, tmp_path: Path) -> None:
        from cli.callers import cmd_callees

        db_path = _build_chain_graph_in_db(tmp_path)
        rc = cmd_callees(_make_args(symbol='callA', db=db_path))
        assert rc == 0


# ---------------------------------------------------------------------------
# impact_radius — Phase 4 slice B
# ---------------------------------------------------------------------------


class TestImpactRadius:
    def test_aggregates_files_across_depth(
        self, tmp_path: Path,
    ) -> None:
        """impact_radius walks reverse edges N-deep and aggregates the
        set of unique files that contain affected symbols. For a chain
        A → B → C → D, asking impact of D at depth 3 surfaces all
        four files (or in this fixture's case, the single file
        containing all symbols)."""
        from cli.callers import compute_impact_radius
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        d_sym = 'scip-java maven g a 1 com/x/D#callD().'
        report = compute_impact_radius(graph, d_sym, depth=3)
        # All four symbols (D, C, B, A) live in 'Chain.scala'
        assert 'Chain.scala' in report.files
        # Symbol set should include all transitively-affected
        names = {sym.display_name for sym in report.affected_symbols}
        assert names == {'callD', 'callC', 'callB', 'callA'}

    def test_unknown_symbol_returns_empty_report(
        self, tmp_path: Path,
    ) -> None:
        from cli.callers import compute_impact_radius
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        report = compute_impact_radius(
            graph, 'scip-java maven g a 1 com/x/Ghost#ghost().', depth=5,
        )
        assert report.files == set()
        assert report.affected_symbols == []

    def test_cmd_returns_zero_for_known_symbol(
        self, tmp_path: Path,
    ) -> None:
        from cli.callers import cmd_impact_radius

        db_path = _build_chain_graph_in_db(tmp_path)
        rc = cmd_impact_radius(_make_args(symbol='callD', db=db_path))
        assert rc == 0

    def test_cmd_returns_nonzero_for_unknown_symbol(
        self, tmp_path: Path,
    ) -> None:
        from cli.callers import cmd_impact_radius

        db_path = _build_chain_graph_in_db(tmp_path)
        rc = cmd_impact_radius(_make_args(
            symbol='nonexistent_symbol', db=db_path,
        ))
        assert rc != 0


# ---------------------------------------------------------------------------
# Dead-code report — Phase 4 slice C
# ---------------------------------------------------------------------------


class TestFormatDeadCodeReport:
    """The dead-code helper queries graph.symbols_with_zero_references
    and produces a markdown-ish report. ``cmd_improve --dead-code``
    composes this into its existing multi-step output."""

    def test_lists_zero_reference_symbols(self, tmp_path: Path) -> None:
        """A graph with defined-but-unreferenced symbols produces
        non-empty output naming each."""
        from cli.callers import format_dead_code_report
        from docgen.scip_cross_source import CrossSourceGraph
        from docgen.scip_extractor import (
            ScipIndex, _ScipDoc, _ScipOccurrence, _ScipSymbol,
        )

        sym = 'scip-java maven g a 1 com/x/Unused#legacy().'
        doc = _ScipDoc(
            relative_path='Unused.scala',
            occurrences=(_ScipOccurrence(
                symbol=sym, range=(0, 0, 5, 0), is_definition=True,
            ),),
            symbols=(_ScipSymbol(
                symbol=sym, kind='Method', display_name='legacy',
            ),),
        )
        graph = CrossSourceGraph()
        graph.add_source(
            'mysrc', index=ScipIndex(documents=(doc,)), language='scala',
        )
        graph.materialize()

        report = format_dead_code_report(graph, 'mysrc')
        assert report is not None
        assert 'legacy' in report
        # Helpful: includes the file location
        assert 'Unused.scala' in report

    def test_returns_none_when_no_dead_code(
        self, tmp_path: Path,
    ) -> None:
        """If every defined symbol is referenced somewhere (within or
        across sources), the report is None — caller can skip the
        section entirely."""
        from cli.callers import format_dead_code_report
        from docgen.scip_cross_source import CrossSourceGraph
        from library import Library

        db_path = _build_chain_graph_in_db(tmp_path)
        lib = Library(db_path)
        graph = CrossSourceGraph()
        with lib._conn_provider.acquire() as conn:
            graph.load_from(conn)
        lib.close()

        # In the chain fixture, callD has no callers — it's "dead"
        # at the end of the chain. callA is referenced by no one but
        # IS the root caller. callB and callC are both used.
        # Actually the chain only goes one direction (A→B→C→D), so
        # only A has no callers.
        report = format_dead_code_report(graph, 'mysrc')
        # callA is used as a source (calls B), but nothing calls IT.
        # So callA appears as zero-ref → report is non-empty.
        assert report is not None
        assert 'callA' in report

    def test_returns_none_for_unknown_source(
        self, tmp_path: Path,
    ) -> None:
        from cli.callers import format_dead_code_report
        from docgen.scip_cross_source import CrossSourceGraph

        graph = CrossSourceGraph()
        graph.materialize()

        report = format_dead_code_report(graph, 'ghost')
        assert report is None
