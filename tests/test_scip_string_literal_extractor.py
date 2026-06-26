"""Contract for the SCIP-driven string-literal extractor (Phase 2p).

This index is the substrate that route extractors (Phase 8a) and
resolution traversal (Phase 2s) query when they need a literal value at
a known position. Without it, every extractor re-implements its own
parser to read literal text — the workaround the architecture
explicitly aims to retire.

Per ``library_scip._STRING_LITERALS_SCHEMA``:

.. code-block:: sql

    CREATE TABLE string_literals (
        id               INTEGER PRIMARY KEY AUTOINCREMENT,
        source_name      TEXT NOT NULL,
        file             TEXT NOT NULL,
        line_start       INTEGER NOT NULL,    -- 1-indexed
        col_start        INTEGER NOT NULL,    -- 0-indexed
        value            TEXT NOT NULL,       -- unquoted contents
        owning_symbol_id TEXT                 -- canonical_id of enclosing
                                              -- function/method, or NULL
    )

Architecture:

- For each file in the SCIP index, parse with the language-appropriate
  AST tool (``ast`` for Python, ``ast-grep-py`` ``javascript`` grammar
  for ``.js``/``.ts``/etc., ``scala`` grammar for ``.scala``).
- Emit one row per string-literal node.
- Skip interpolated forms (Python f-strings, JS template literals with
  ``${...}``, Scala ``s"..."`` / ``f"..."``) — those are unresolved at
  index time and belong in Phase 2s's "ambiguous" tier, not in the
  literal index.
- Resolve ``owning_symbol_id`` by querying ``scip_symbols`` for the
  smallest-range function/method whose line range contains the
  literal. ``NULL`` if none.
- Re-ingest replaces all rows for ``source_name``.

These tests are RED until ``docgen/scip_string_literal_extractor.py``
implements ``ingest_string_literals``.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from docgen.scip_extractor import (
    ScipIndex,
    _ScipDoc,
    _ScipOccurrence,
)
from docgen.scip_string_literal_extractor import ingest_string_literals


@pytest.fixture
def conn():
    """Fresh in-memory SQLite with the SCIP schema applied."""
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _make_index(
    src_files: list[tuple[Path, Path]],
    source_root: Path,
) -> ScipIndex:
    """Build a synthetic ScipIndex for a set of files (empty occurrences).
    The extractor walks ``index.documents`` to know which files to scan; these
    extraction tests don't assert ownership, so no definition occurrences are
    needed. Ownership is resolved from the index's ``enclosing_range`` and is
    unit-tested in ``tests/test_scip_owning.py``; the one ingest-level wiring
    check (``test_string_inside_function_resolves_owner``) builds its index
    inline with a definition occurrence."""
    docs = tuple(
        _ScipDoc(
            relative_path=str(f.relative_to(source_root)),
            occurrences=(),
            symbols=(),
        )
        for f, _ in src_files
    )
    return ScipIndex(documents=docs, source_root=source_root)


def _query_literals(
    conn: sqlite3.Connection, source_name: str,
) -> list[tuple]:
    """Return ordered tuples (file, line, col, value, owning_symbol_id)
    so tests can assert without depending on the autoincrement id."""
    cur = conn.execute(
        '''SELECT file, line_start, col_start, value, owning_symbol_id
           FROM string_literals WHERE source_name = ?
           ORDER BY file, line_start, col_start''',
        (source_name,),
    )
    return cur.fetchall()


# ---------------------------------------------------------------------------
# Python — ast.parse for literal extraction
# ---------------------------------------------------------------------------


class TestPythonLiterals:
    def test_module_level_string(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "X = 'hello'\n"
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)

        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        # One row, value 'hello', owner NULL (module level)
        assert len(rows) == 1
        file, line, col, value, owner = rows[0]
        assert value == 'hello'
        assert line == 1
        assert owner is None

    def test_docstrings_are_excluded_as_non_data(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A docstring — the bare string that is the first statement of a
        module/class/function body — is documentation, never a data value, so it
        is not indexed (and never fed to the SQL parser). Assigned strings
        (module/class constants) are NOT docstrings and ARE indexed."""
        text = (
            '"""module docstring."""\n'
            'GREETING = "hello"\n'
            'def f():\n'
            '    "function docstring."\n'
            '    return "result"\n'
            'class C:\n'
            '    "class docstring."\n'
            '    attr = "value"\n'
        )
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj', source_root=tmp_path, conn=conn,
            index_factory=lambda: index)
        values = {r[3] for r in _query_literals(conn, 'myproj')}
        assert {'hello', 'result', 'value'} <= values         # data values indexed
        assert not ({'module docstring.', 'function docstring.',
                     'class docstring.'} & values)            # docstrings excluded

    def test_string_inside_function_resolves_owner(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """ingest wires the owning resolver: a literal inside a function body is
        stored with that function's canonical_id, resolved from the index's
        ``enclosing_range`` (no ``scip_symbols`` row). The resolver's own
        behaviours — method-vs-class, nested, term, parameter-exclusion — are
        unit-tested in ``tests/test_scip_owning.py``."""
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = (
            'def greet():\n'
            "    return 'hi'\n"
        )
        src = tmp_path / 'mod.py'
        src.write_text(text)
        fn_id = 'scip-python python myproj . mod/greet().'
        # greet's def occurrence: name-token range on line 0, body span 0..1.
        doc = _ScipDoc(
            relative_path='mod.py',
            occurrences=(
                _ScipOccurrence(symbol=fn_id, range=(0, 4, 9), is_definition=True,
                                enclosing_range=(0, 0, 1, 0)),
            ),
            symbols=())
        index = ScipIndex(documents=(doc,), source_root=tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        assert len(rows) == 1
        _, _, _, value, owner = rows[0]
        assert value == 'hi'
        assert owner == fn_id

    def test_fstring_with_interpolation_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = (
            "name = 'world'\n"
            "msg = f'hello {name}'\n"
        )
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        # Only the plain 'world' is captured. The f-string is skipped
        # entirely — neither 'hello ' nor the {name} part appears.
        values = [r[3] for r in rows]
        assert 'world' in values
        assert not any('hello' in v for v in values)

    def test_bytes_literal_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "data = b'binary'\n"
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        # b'binary' is bytes, not a str literal. Don't index.
        assert rows == []

    def test_raw_string_kept(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "pat = r'\\d+'\n"
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        assert len(rows) == 1
        # ast normalizes r-strings: r'\d+' gives the str value '\\d+'.
        assert rows[0][3] == '\\d+'


# ---------------------------------------------------------------------------
# JavaScript / TypeScript — ast-grep 'javascript' grammar
# ---------------------------------------------------------------------------


class TestJavaScriptLiterals:
    def test_single_quoted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "const x = 'hi';\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        values = [r[3] for r in rows]
        assert 'hi' in values

    def test_double_quoted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = 'const x = "hi";\n'
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'hi' in values

    def test_template_literal_without_interpolation(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "const x = `hi`;\n"
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'hi' in values

    def test_template_literal_with_interpolation_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = (
            "const name = 'world';\n"
            "const msg = `hello ${name}`;\n"
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        # 'world' captured (plain string), template literal skipped
        assert 'world' in values
        assert not any('hello' in v for v in values)

    def test_typescript_file_indexed(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "const x: string = 'ts';\n"
        src = tmp_path / 'app.ts'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'ts' in values


# ---------------------------------------------------------------------------
# Scala — ast-grep 'scala' grammar
# ---------------------------------------------------------------------------


class TestScalaLiterals:
    def test_plain_string(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = 'val x = "hi"\n'
        src = tmp_path / 'M.scala'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'hi' in values

    def test_interpolated_s_string_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = (
            'val name = "world"\n'
            'val msg = s"hello $name"\n'
        )
        src = tmp_path / 'M.scala'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        # 'world' kept; s"..." form skipped (interpolation).
        assert 'world' in values
        assert not any('hello' in v for v in values)


# ---------------------------------------------------------------------------
# Multiple files
# ---------------------------------------------------------------------------


class TestMultipleFiles:
    def test_literals_from_multiple_files(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        a = tmp_path / 'a.py'
        b_dir = tmp_path / 'sub'
        b_dir.mkdir()
        b = b_dir / 'b.py'
        a.write_text("x = 'from-a'\n")
        b.write_text("y = 'from-b'\n")
        index = _make_index([(a, tmp_path), (b, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        files = {r[0] for r in rows}
        values = {r[3] for r in rows}
        assert str(a.resolve()) in files
        assert str(b.resolve()) in files
        assert {'from-a', 'from-b'} <= values


# ---------------------------------------------------------------------------
# Re-ingest semantics
# ---------------------------------------------------------------------------


class TestReIngest:
    def test_replaces_same_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        src = tmp_path / 'mod.py'
        src.write_text("x = 'first'\n")
        index1 = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index1,
        )
        # Replace
        src.write_text("x = 'second'\n")
        index2 = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index2,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'first' not in values
        assert 'second' in values

    def test_preserves_other_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        # Pre-existing row from a different source
        conn.execute(
            '''INSERT INTO string_literals
               (source_name, file, line_start, col_start, value)
               VALUES (?, ?, ?, ?, ?)''',
            ('other', '/x.py', 1, 0, 'preserved'),
        )
        conn.commit()

        src = tmp_path / 'mod.py'
        src.write_text("x = 'mine'\n")
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        # myproj has 'mine'; other still has 'preserved'
        my = [r[3] for r in _query_literals(conn, 'myproj')]
        other = [r[3] for r in _query_literals(conn, 'other')]
        assert 'mine' in my
        assert other == ['preserved']


# ---------------------------------------------------------------------------
# Adversarial — error tolerance
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_missing_index_no_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        rc = ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
        )
        assert rc == 0

    def test_syntax_error_skips_only_that_file(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        broken = tmp_path / 'broken.py'
        good = tmp_path / 'good.py'
        broken.write_text('def oops(\n  # missing close\n')
        good.write_text("y = 'good'\n")
        index = _make_index(
            [(broken, tmp_path), (good, tmp_path)], tmp_path,
        )
        rc = ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        # broken.py contributes nothing, good.py emits 'good'
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'good' in values
        assert rc >= 1

    def test_unsupported_extension_skipped_cleanly(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        # .md is in the catalog ext set but we don't index its literals.
        # The extractor should walk past it without failing.
        md = tmp_path / 'README.md'
        py = tmp_path / 'app.py'
        md.write_text("# heading\n\n'this isnt a python literal'\n")
        py.write_text("x = 'real'\n")
        index = _make_index([(md, tmp_path), (py, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'real' in values
        # markdown content NOT scanned for literals
        assert 'this isnt a python literal' not in values

    def test_no_scip_symbols_owner_is_null(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """When scip_symbols is empty (e.g., literal extraction runs
        before cross-source-graph save), every owning_symbol_id is NULL.
        Don't blow up on the lookup."""
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        src = tmp_path / 'mod.py'
        src.write_text("def f():\n    return 'inside'\n")
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        assert len(rows) == 1
        assert rows[0][4] is None


# ---------------------------------------------------------------------------
# Bite — cases that fail under a "regex over source" implementation
# ---------------------------------------------------------------------------


class TestForcesStructuralParser:
    """The doc explicitly says Phase 2p is built by structural parsing,
    not regex over source. These tests fail loudly under a naive
    regex-based extractor — they articulate WHY the structural parser
    is required."""

    def test_quotes_inside_python_comment_not_indexed(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A regex like ``r"'[^']*'"`` would match ``'fake'`` inside a
        comment. The structural parser knows comments aren't string
        nodes."""
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = (
            "# this 'fake' is a comment with 'quoted' text\n"
            "real = 'actual'\n"
        )
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'actual' in values
        assert 'fake' not in values
        assert 'quoted' not in values

    def test_escaped_quote_in_python_string_preserved(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A naive impl that splits on the first matching quote chops the
        value at the escape. The structural parser produces the full
        unquoted value."""
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = 'msg = "say \\"hi\\""\n'
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        values = [r[3] for r in _query_literals(conn, 'myproj')]
        assert 'say "hi"' in values

    def test_python_triple_quoted_string_kept_with_full_value(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``'''multi\\nline'''`` → one row whose value is ``'multi\\nline'``.
        Embedded ``'`` characters inside don't terminate it."""
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = (
            "x = '''line1 with 'inner' quote\nline2'''\n"
        )
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        # One row whose value contains both 'inner' and the newline. A
        # regex that splits on `'` would emit 'line1 with ', 'inner',
        # ' quote ', etc. — wrong.
        assert len(rows) == 1
        value = rows[0][3]
        assert "'inner'" in value
        assert '\n' in value

    def test_python_adjacent_string_concat_is_one_literal(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Python concatenates adjacent string literals at parse time:
        ``'hello ' 'world'`` is one ``ast.Constant`` with value
        ``'hello world'``. A regex impl would emit two separate rows;
        the structural parser emits one."""
        from docgen.scip_string_literal_extractor import (
            ingest_string_literals,
        )

        text = "msg = 'hello ' 'world'\n"
        src = tmp_path / 'mod.py'
        src.write_text(text)
        index = _make_index([(src, tmp_path)], tmp_path)
        ingest_string_literals(
            source_name='myproj',
            source_root=tmp_path,
            conn=conn,
            index_factory=lambda: index,
        )
        rows = _query_literals(conn, 'myproj')
        values = [r[3] for r in rows]
        assert 'hello world' in values
        # ...and NOT two separate rows for 'hello ' / 'world'
        assert 'hello ' not in values
