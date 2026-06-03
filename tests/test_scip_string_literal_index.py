"""Contract for the string-literal index (Phase 2p / Layer C).

Layer C's resolution traversal walks SCIP refs from a sink call site
backward looking for literal values. The string-literal index makes
that lookup fast: every string literal in indexed source is
pre-extracted with its position and the symbol whose body encloses it.

Schema columns (asserted by ``TestSchema``):
``source_name``, ``file``, ``line_start``, ``col_start``, ``value``,
``owning_symbol_id``.

MVP scope (this slice): Python extractor only. JVM / TypeScript
extractors come in Phase 2p.b — same value class, same persistence
layer, just different per-language extraction.

Re-ingest semantics: ``persist_string_literals`` clears existing rows
for ``source_name`` before inserting (mirrors Phase 2q config_index
and Phase 7c swagger_ingest).

These tests are RED until ``docgen/scip_string_literal_index.py``
exists and ``library_scip.init_scip_schema`` creates the
``string_literals`` table.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the SCIP schema applied."""
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


class TestSchema:
    def test_init_scip_schema_creates_string_literals_table(
        self, conn: sqlite3.Connection,
    ) -> None:
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='table' AND name='string_literals'"
        )
        assert cur.fetchone() is not None

    def test_required_columns_present(
        self, conn: sqlite3.Connection,
    ) -> None:
        cur = conn.execute('PRAGMA table_info(string_literals)')
        cols = {row[1] for row in cur.fetchall()}
        for col in (
            'source_name', 'file', 'line_start', 'col_start',
            'value', 'owning_symbol_id',
        ):
            assert col in cols, (
                f'string_literals missing column: {col}'
            )

    def test_indexed_for_symbol_lookup(
        self, conn: sqlite3.Connection,
    ) -> None:
        """Phase 2s queries by ``(source_name, owning_symbol_id)``
        repeatedly during traversal — must be indexed."""
        cur = conn.execute(
            "SELECT name FROM sqlite_master "
            "WHERE type='index' AND tbl_name='string_literals'"
        )
        index_names = {row[0] for row in cur.fetchall()}
        # At least one index covering owning_symbol_id
        assert any(
            'symbol' in name.lower() or 'owning' in name.lower()
            for name in index_names
        )


# ---------------------------------------------------------------------------
# StringLiteral value class
# ---------------------------------------------------------------------------


class TestStringLiteralShape:
    def test_string_literal_is_frozen(self, tmp_path: Path) -> None:
        from docgen.scip_string_literal_index import StringLiteral

        lit = StringLiteral(
            file=tmp_path / 'app.py',
            line_start=10,
            col_start=4,
            value='hello',
            owning_symbol_id='mymod.func',
        )
        try:
            lit.value = 'mutated'  # type: ignore[misc]
        except Exception:
            return
        raise AssertionError(
            'StringLiteral should be @frozen',
        )


# ---------------------------------------------------------------------------
# Python extractor
# ---------------------------------------------------------------------------


def _write_py(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding='utf-8')
    return path


class TestExtractPythonLiterals:
    def test_simple_string_literal_extracted(
        self, tmp_path: Path,
    ) -> None:
        """A single ``"..."`` literal inside a function body is
        extracted with the function as the owning symbol."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'app.py',
            'def run():\n'
            '    cmd = "ls -l"\n'
            '    return cmd\n',
        )
        # Function spans lines 1-3 in 1-indexed terms
        symbols = [SymbolRange(
            canonical_id='app.run',
            line_start=1,
            line_end=3,
        )]

        literals = extract_python_literals(src, symbols=symbols)
        cmd_lit = next(
            (l for l in literals if l.value == 'ls -l'),
            None,
        )
        assert cmd_lit is not None
        assert cmd_lit.owning_symbol_id == 'app.run'
        assert cmd_lit.line_start == 2  # 1-indexed
        assert cmd_lit.file == src

    def test_multiple_literals_in_function(
        self, tmp_path: Path,
    ) -> None:
        """All string literals in a function get their own
        StringLiteral record."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'app.py',
            'def run():\n'
            '    a = "first"\n'
            '    b = "second"\n'
            '    return a, b\n',
        )
        symbols = [SymbolRange('app.run', 1, 4)]

        literals = extract_python_literals(src, symbols=symbols)
        values = {l.value for l in literals}
        assert 'first' in values
        assert 'second' in values

    def test_innermost_function_owns_literal(
        self, tmp_path: Path,
    ) -> None:
        """When a literal sits inside a nested function, the INNERMOST
        function owns it. Phase 2s needs this for correct caller
        attribution."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'app.py',
            'def outer():\n'
            '    def inner():\n'
            '        return "nested"\n'
            '    return inner\n',
        )
        # outer spans 1-4; inner spans 2-3
        symbols = [
            SymbolRange('app.outer', 1, 4),
            SymbolRange('app.outer.inner', 2, 3),
        ]

        literals = extract_python_literals(src, symbols=symbols)
        nested = next(
            (l for l in literals if l.value == 'nested'),
            None,
        )
        assert nested is not None
        # Innermost wins
        assert nested.owning_symbol_id == 'app.outer.inner'

    def test_module_level_literal_has_no_owning_symbol(
        self, tmp_path: Path,
    ) -> None:
        """A literal at module scope (not inside any function/class)
        has ``owning_symbol_id=None``. Phase 2s skips these for
        sink-resolution but they still get persisted for completeness."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'config.py',
            'BASE_URL = "https://api.example.com"\n'
            'def fetch():\n'
            '    return BASE_URL\n',
        )
        symbols = [SymbolRange('config.fetch', 2, 3)]

        literals = extract_python_literals(src, symbols=symbols)
        base = next(
            (l for l in literals if l.value == 'https://api.example.com'),
            None,
        )
        assert base is not None
        assert base.owning_symbol_id is None

    def test_dynamic_f_string_skipped(
        self, tmp_path: Path,
    ) -> None:
        """An f-string with non-constant parts cannot be resolved at
        index time. Skip — the resolution traversal in Phase 2s will
        mark it ambiguous separately."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'app.py',
            'def run(name):\n'
            '    cmd = f"echo {name}"\n'
            '    return cmd\n',
        )
        symbols = [SymbolRange('app.run', 1, 3)]

        literals = extract_python_literals(src, symbols=symbols)
        # The f-string with a variable should not appear as a literal
        for l in literals:
            assert 'echo' not in l.value or '{' not in l.value

    def test_constant_f_string_extracted(
        self, tmp_path: Path,
    ) -> None:
        """An f-string with NO interpolation is effectively a constant
        string and can be extracted (some Python codebases use f-strings
        unconditionally)."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'app.py',
            'def run():\n'
            '    return f"hello world"\n',
        )
        symbols = [SymbolRange('app.run', 1, 2)]

        literals = extract_python_literals(src, symbols=symbols)
        values = {l.value for l in literals}
        assert 'hello world' in values

    def test_bytes_literals_skipped(self, tmp_path: Path) -> None:
        """Bytes (``b"..."``) aren't string paths/URLs — skip."""
        from docgen.scip_string_literal_index import (
            SymbolRange,
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'app.py',
            'def run():\n'
            '    return b"\\x00\\x01"\n',
        )
        symbols = [SymbolRange('app.run', 1, 2)]

        literals = extract_python_literals(src, symbols=symbols)
        # No string literal here (it's bytes)
        assert literals == []

    def test_malformed_python_returns_empty(
        self, tmp_path: Path,
    ) -> None:
        """Parse errors → empty list, no crash. One bad file shouldn't
        abort indexing."""
        from docgen.scip_string_literal_index import (
            extract_python_literals,
        )

        src = _write_py(
            tmp_path / 'broken.py',
            'def oops(\n  # missing close paren\n',
        )
        assert extract_python_literals(src, symbols=[]) == []


# ---------------------------------------------------------------------------
# persist_string_literals — write path
# ---------------------------------------------------------------------------


class TestPersist:
    def test_inserts_rows(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
        )

        literals = [
            StringLiteral(
                file=tmp_path / 'app.py',
                line_start=10,
                col_start=4,
                value='hello',
                owning_symbol_id='app.run',
            ),
        ]
        count = persist_string_literals(
            source_name='myapp',
            literals=literals,
            conn=conn,
        )
        assert count == 1

        cur = conn.execute(
            'SELECT value, owning_symbol_id FROM string_literals '
            "WHERE source_name='myapp'"
        )
        row = cur.fetchone()
        assert row == ('hello', 'app.run')

    def test_re_ingest_replaces_old_rows(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
        )

        old = [StringLiteral(
            tmp_path / 'a.py', 1, 0, 'old', 'mod.f',
        )]
        new = [StringLiteral(
            tmp_path / 'a.py', 2, 0, 'new', 'mod.g',
        )]
        persist_string_literals(
            source_name='s', literals=old, conn=conn,
        )
        persist_string_literals(
            source_name='s', literals=new, conn=conn,
        )

        cur = conn.execute(
            "SELECT value FROM string_literals WHERE source_name='s'"
        )
        values = {row[0] for row in cur.fetchall()}
        assert values == {'new'}

    def test_isolated_per_source(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
        )

        persist_string_literals(
            source_name='A',
            literals=[StringLiteral(
                tmp_path / 'a.py', 1, 0, 'val_a', 'A.f',
            )],
            conn=conn,
        )
        persist_string_literals(
            source_name='B',
            literals=[StringLiteral(
                tmp_path / 'b.py', 1, 0, 'val_b', 'B.g',
            )],
            conn=conn,
        )

        cur = conn.execute(
            "SELECT value FROM string_literals WHERE source_name='A'"
        )
        assert cur.fetchone()[0] == 'val_a'

    def test_empty_list_clears_source(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
        )

        # Pre-populate
        persist_string_literals(
            source_name='s',
            literals=[StringLiteral(
                tmp_path / 'a.py', 1, 0, 'val', 'sym',
            )],
            conn=conn,
        )
        # Empty re-ingest clears
        persist_string_literals(
            source_name='s', literals=[], conn=conn,
        )
        cur = conn.execute(
            "SELECT COUNT(*) FROM string_literals WHERE source_name='s'"
        )
        assert cur.fetchone()[0] == 0

    def test_owning_symbol_can_be_null(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Module-level literals have ``owning_symbol_id=None`` —
        persist as SQL NULL."""
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
        )

        literals = [StringLiteral(
            file=tmp_path / 'config.py',
            line_start=1,
            col_start=11,
            value='https://api.example.com',
            owning_symbol_id=None,
        )]
        persist_string_literals(
            source_name='s', literals=literals, conn=conn,
        )

        cur = conn.execute(
            'SELECT value, owning_symbol_id FROM string_literals '
            "WHERE source_name='s'"
        )
        row = cur.fetchone()
        assert row[0] == 'https://api.example.com'
        assert row[1] is None


# ---------------------------------------------------------------------------
# query helpers
# ---------------------------------------------------------------------------


class TestQuery:
    def test_query_by_owning_symbol(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """Look up all literals enclosed by a given symbol — used by
        Phase 2s when walking back from a sink call site to find
        candidate literal arguments."""
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
            query_string_literals_by_symbol,
        )

        literals = [
            StringLiteral(
                tmp_path / 'a.py', 1, 0, 'first', 'mod.f',
            ),
            StringLiteral(
                tmp_path / 'a.py', 2, 0, 'second', 'mod.f',
            ),
            StringLiteral(
                tmp_path / 'a.py', 3, 0, 'other', 'mod.g',
            ),
        ]
        persist_string_literals(
            source_name='s', literals=literals, conn=conn,
        )

        f_literals = query_string_literals_by_symbol(
            source_name='s',
            owning_symbol_id='mod.f',
            conn=conn,
        )
        values = {l.value for l in f_literals}
        assert values == {'first', 'second'}

    def test_query_in_file(
        self, conn: sqlite3.Connection, tmp_path: Path,
    ) -> None:
        """All literals in a file (any owning symbol) — useful for
        per-file diagnostics."""
        from docgen.scip_string_literal_index import (
            StringLiteral,
            persist_string_literals,
            query_string_literals_in_file,
        )

        target = tmp_path / 'a.py'
        other = tmp_path / 'b.py'
        literals = [
            StringLiteral(target, 1, 0, 'in_target', 'sym1'),
            StringLiteral(target, 2, 0, 'also_target', None),
            StringLiteral(other, 1, 0, 'in_other', 'sym2'),
        ]
        persist_string_literals(
            source_name='s', literals=literals, conn=conn,
        )

        results = query_string_literals_in_file(
            source_name='s', file=target, conn=conn,
        )
        values = {l.value for l in results}
        assert values == {'in_target', 'also_target'}
