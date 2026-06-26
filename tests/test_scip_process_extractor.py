"""Contract for SCIP-driven process invocation extractor (Phase 2t).

Same architectural shape as Phase 8b (HTTP client extractors) but
writes to ``process_invocations`` instead of ``http_client_calls``,
and matches sinks of ``kind='process_invocation'`` instead of
``'http_client'``.

Architecture:

- **SCIP** filters which call sites are real subprocess primitives
  via the Phase 2r registry filtered to ``kind='process_invocation'``.
  Adding a new subprocess library is a registry edit.
- **Per-language AST tools** (``ast`` for Python, tree-sitter for
  JS/Scala) walk call structure to find argument positions.
- **Phase 2p ``string_literals``** supplies the literal command/script
  value at the first arg's position. List args (``subprocess.run([
  "python", "x.py"])``) are deferred — they need either explicit list
  unpacking or Phase 2s resolution.
- **Phase 2d ``scip_symbols``** supplies ``caller_symbol_id``. The
  schema requires NOT NULL, so module-level subprocess calls are
  skipped.

Output: rows in ``process_invocations``. ``target_path`` is the
literal command/script string at the call site;
``target_symbol_id`` is reserved for Phase 2t.b fuzzy file-matching
(deferred).

Languages covered (v1):

- Python: ``subprocess.run``, ``subprocess.Popen``, ``subprocess.call``,
  ``subprocess.check_call``, ``subprocess.check_output``, ``os.system``
- JS/TS: ``child_process.spawn``, ``child_process.exec``,
  ``child_process.execFile``, ``child_process.spawnSync``,
  ``child_process.execSync``, ``child_process.fork``
- Scala: ``sys.process.Process(...)`` (apply with string arg)
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


# Synthetic SCIP symbols matching registry suffix patterns.
# Python — full scip-python descriptor with .py path component
_PY_SUBPROCESS_RUN_SYM = (
    'scip-python python pypi-stdlib 0 subprocess/__init__.py/run.'
)
_PY_SUBPROCESS_POPEN_SYM = (
    'scip-python python pypi-stdlib 0 subprocess/__init__.py/Popen#'
    '__init__.'
)
_PY_OS_SYSTEM_SYM = (
    'scip-python python pypi-stdlib 0 os.py/system.'
)

# JS — scip-typescript symbols for Node child_process
_JS_SPAWN_SYM = (
    'scip-typescript . . . child_process.d.ts/spawn.'
)
_JS_EXEC_SYM = (
    'scip-typescript . . . child_process.d.ts/exec.'
)

# JVM — scip-java for Scala sys.process
_SCALA_PROCESS_APPLY_SYM = (
    'scip-java semanticdb maven org.scala-lang scala-library 2.13.0 '
    'scala/sys/process/Process#apply().'
)


@pytest.fixture
def conn():
    from library.scip import init_scip_schema

    c = sqlite3.connect(':memory:')
    init_scip_schema(c)
    yield c
    c.close()


def _occ_at(
    text: str, marker: str, symbol: str, *, nth: int = 0,
) -> _ScipOccurrence:
    found = 0
    pos = -1
    n = len(text)
    i = 0
    while i <= n - len(marker):
        if text.startswith(marker, i):
            before_ok = (
                i == 0
                or not (text[i - 1].isalnum() or text[i - 1] == '_')
            )
            j = i + len(marker)
            after_ok = (
                j >= n
                or not (text[j].isalnum() or text[j] == '_')
            )
            if before_ok and after_ok:
                if found == nth:
                    pos = i
                    break
                found += 1
                i = j
                continue
        i += 1
    if pos < 0:
        raise ValueError(
            f'marker {marker!r} (nth={nth}) not found as a word',
        )
    line = text.count('\n', 0, pos)
    line_start = text.rfind('\n', 0, pos) + 1
    col = pos - line_start
    return _ScipOccurrence(
        symbol=symbol,
        range=(line, col, line, col + len(marker)),
        is_definition=False,
    )


def _make_index(
    src_files: list[tuple[Path, list[_ScipOccurrence]]],
    source_root: Path,
) -> ScipIndex:
    docs = tuple(
        _ScipDoc(
            relative_path=str(f.relative_to(source_root)),
            occurrences=tuple(occs),
            symbols=(),
        )
        for f, occs in src_files
    )
    return ScipIndex(documents=docs, source_root=source_root)


def _def_occ(symbol, line_start, line_end):
    """A definition occurrence for an enclosing function/method/class spanning
    1-indexed line_start..line_end: a name-token ``range`` on the def line plus
    the body ``enclosing_range`` — the shape scip-python emits and the owning
    resolver reads (replaces the retired scip_symbols owning fixture)."""
    return _ScipOccurrence(
        symbol=symbol, range=(line_start - 1, 4, 40), is_definition=True,
        enclosing_range=(line_start - 1, 0, line_end - 1, 0))


def _add_scip_symbol(
    conn: sqlite3.Connection,
    *,
    canonical_id: str,
    source_name: str,
    file: str,
    line_start: int,
    line_end: int,
    kind: str = 'Function',
    qualified_name: str = '',
    parent_qualified_name: str | None = None,
    language: str = 'python',
) -> None:
    conn.execute(
        '''INSERT INTO scip_symbols
           (canonical_id, source_name, language, file,
            line_start, line_end, kind, display_name,
            qualified_name, parent_qualified_name)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (canonical_id, source_name, language, file,
         line_start, line_end, kind,
         qualified_name.rsplit('.', 1)[-1] or canonical_id,
         qualified_name, parent_qualified_name),
    )
    conn.commit()


def _query_invocations(
    conn: sqlite3.Connection, source_name: str,
) -> list[dict]:
    cur = conn.execute(
        '''SELECT caller_symbol_id, target_path, target_symbol_id,
                  confidence, file, line_start, line_end
           FROM process_invocations WHERE source_name = ?
           ORDER BY file, line_start''',
        (source_name,),
    )
    cols = [
        'caller_symbol_id', 'target_path', 'target_symbol_id',
        'confidence', 'file', 'line_start', 'line_end',
    ]
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def _run_pipeline(
    *, source_name: str, source_root: Path,
    conn: sqlite3.Connection, index: ScipIndex,
) -> int:
    from docgen.scip_process_extractor import (
        ingest_process_invocations,
    )

    ingest_string_literals(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )
    return ingest_process_invocations(
        source_name=source_name, source_root=source_root,
        conn=conn, index_factory=lambda: index,
    )


# ---------------------------------------------------------------------------
# Python — subprocess + os
# ---------------------------------------------------------------------------


class TestPythonSubprocess:
    def test_subprocess_popen_paired_with_skip(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two ``Popen`` calls — one with a string literal (must
        emit), one with a variable (must skip). Bites a stub (no
        rows), an impl missing Popen from the registry (no rows
        either), and an over-broad impl (two rows)."""
        text = (
            'def run_proc():\n'
            '    cmd = "ls -la"\n'
            '    subprocess.Popen("ls -la")\n'
            '    subprocess.Popen(cmd)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        fn_id = 'scip-python . . . app.py/run_proc().'
        index = _make_index([
            (src, [
                _occ_at(
                    text, 'Popen', _PY_SUBPROCESS_POPEN_SYM, nth=0,
                ),
                _occ_at(
                    text, 'Popen', _PY_SUBPROCESS_POPEN_SYM, nth=1,
                ),_def_occ(fn_id, 1, 4)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal Popen call; got {targets}'
        )
        assert rows[0]['target_path'] == 'ls -la'

    def test_os_system_paired_with_skip(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Same shape for ``os.system`` — exercises the registry's
        ``os.system`` entry specifically while keeping the bite of a
        paired skip case."""
        text = (
            'def trigger():\n'
            '    cmd = "./run.sh"\n'
            '    os.system("./run.sh")\n'
            '    os.system(cmd)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        fn_id = 'scip-python . . . app.py/trigger().'
        index = _make_index([
            (src, [
                _occ_at(text, 'system', _PY_OS_SYSTEM_SYM, nth=0),
                _occ_at(text, 'system', _PY_OS_SYSTEM_SYM, nth=1),_def_occ(fn_id, 1, 4)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, f'expected only the literal; got {targets}'
        assert rows[0]['target_path'] == './run.sh'

    def test_list_arg_skipped_but_literal_arg_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two ``subprocess.run`` calls: one with a string literal
        (must emit), one with a list literal (must skip — v1 doesn't
        unpack sequences). Paired so a stub (zero rows) and a naive
        impl that captures lists both fail."""
        text = (
            'def deploy():\n'
            '    subprocess.run("python literal.py")\n'
            '    subprocess.run(["python", "list.py"])\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0),
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=1),_def_occ('scip-python . . . app/deploy().', 1, 3)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python literal.py'

    def test_variable_arg_skipped_but_literal_arg_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Same pairing pattern: literal arg emits, variable arg is
        skipped. Bites stubs and any AST impl that re-parses values
        from the source instead of going through Phase 2p."""
        text = (
            'def deploy():\n'
            '    cmd = "python via_var.py"\n'
            '    subprocess.run("python direct.py")\n'
            '    subprocess.run(cmd)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0),
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=1),_def_occ('scip-python . . . app/deploy().', 1, 4)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python direct.py'
        # Crucially, the literal value of the unrelated `cmd` var
        # is NOT what we capture — the impl can't peek at variable
        # definition lines via Phase 2p (those literals exist there
        # too, but at a different position from the call's arg).
        assert 'python via_var.py' not in targets

    def test_module_level_call_skipped_function_level_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Same fixture has TWO ``subprocess.run`` calls — one at
        module level (NULL caller_symbol_id, skipped per schema NOT
        NULL constraint) and one inside a function (emits). Pairing
        bites both a stub (no rows) and a naive impl that emits for
        both (two rows or NULL violation)."""
        text = (
            'subprocess.run("python boot.py")\n'
            'def deploy():\n'
            '    subprocess.run("python deploy.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0),
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=1),_def_occ('scip-python . . . app/deploy().', 2, 3)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the function-level call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python deploy.py'
        # Module-level call's literal must NOT be in the table at all
        assert 'python boot.py' not in targets


# ---------------------------------------------------------------------------
# Forces Phase 2p integration — bites a naive AST-only impl
# ---------------------------------------------------------------------------


class TestForcesPhase2pIntegration:
    """The contract says the extractor reads ``target_path`` from
    Phase 2p ``string_literals``, not by re-parsing the source. These
    tests fail under any impl that bypasses Phase 2p — they bite
    short-cuts at design time, not at PR-review time."""

    def test_phase_2p_is_load_bearing(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two-phase test that pins the Phase 2p contract:

        - WITH ``ingest_string_literals``: the call produces a row.
        - WITHOUT ``ingest_string_literals``: the SAME call produces
          no row.

        Pairing both halves catches a stub (no rows ever) AND an
        AST-only impl (rows even without Phase 2p). Either deviation
        breaks the test."""
        from docgen.scip_process_extractor import (
            ingest_process_invocations,
        )

        text = (
            'def deploy():\n'
            '    subprocess.run("python deploy.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [_occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM), _def_occ('scip-python . . . app/deploy().', 1, 2)]),
        ], tmp_path)

        # Phase 1: WITHOUT ingest_string_literals — must yield 0 rows
        ingest_process_invocations(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index_factory=lambda: index,
        )
        assert _query_invocations(conn, 'myapi') == [], (
            'extractor should not emit rows when string_literals '
            'is empty — Phase 2p is the value source, not a '
            'parallel AST walk'
        )

        # Phase 2: WITH ingest_string_literals — must yield 1 row
        ingest_string_literals(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index_factory=lambda: index,
        )
        ingest_process_invocations(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index_factory=lambda: index,
        )
        rows = _query_invocations(conn, 'myapi')
        assert len(rows) == 1, (
            'extractor should emit a row once string_literals is '
            'populated — pairs with the empty-string_literals branch '
            'above to confirm Phase 2p is load-bearing'
        )
        assert rows[0]['target_path'] == 'python deploy.py'

    def test_fstring_arg_skipped_but_literal_arg_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two ``subprocess.run`` calls in the same function: one with
        a literal arg (must emit), one with an f-string arg (must
        skip — Phase 2p doesn't index f-strings).

        Paired with the literal case so a stub (zero rows) AND a
        naive impl that captures f-string contents both fail."""
        text = (
            'def deploy(name):\n'
            '    subprocess.run("python literal.py")\n'
            '    subprocess.run(f"python {name}.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0),
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=1),_def_occ('scip-python . . . app/deploy().', 1, 3)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        # Exactly one row — the literal case
        assert len(rows) == 1, (
            f'expected one row from the literal call only; got '
            f'{[r["target_path"] for r in rows]}'
        )
        assert rows[0]['target_path'] == 'python literal.py'

    def test_adjacent_string_concat_yields_merged_literal(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Python parser merges ``"a" "b"`` into one
        ``ast.Constant("ab")``. Phase 2p indexes the merged literal
        at the position of the first quote. The extractor's lookup
        at that position should retrieve the merged string. A naive
        regex-over-source impl would emit two separate rows."""
        text = (
            'def deploy():\n'
            '    subprocess.run("python " "deploy.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [_occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM), _def_occ('scip-python . . . app/deploy().', 1, 2)]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        # One row with the merged value
        assert len(rows) == 1
        assert rows[0]['target_path'] == 'python deploy.py'


# ---------------------------------------------------------------------------
# JS/TS — child_process
# ---------------------------------------------------------------------------


class TestJsChildProcess:
    # ``child_process.spawn`` is exercised in
    # ``test_js_module_level_skipped_function_level_emitted`` which
    # already pairs positive + skip; no separate spawn-only test is
    # needed for regression coverage.

    def test_child_process_exec_paired_with_skip(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired regression guard for ``child_process.exec`` —
        literal arg emits, variable arg skips. Bites a stub, an
        impl missing ``exec`` from the registry, and an over-broad
        impl that captures variable args."""
        text = (
            'function trigger() {\n'
            "  const cmd = 'python deploy.py';\n"
            "  child_process.exec('python deploy.py');\n"
            '  child_process.exec(cmd);\n'
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        fn_id = 'scip-typescript . . . app.js/trigger().'
        index = _make_index([
            (src, [
                _occ_at(text, 'exec', _JS_EXEC_SYM, nth=0),
                _occ_at(text, 'exec', _JS_EXEC_SYM, nth=1),_def_occ(fn_id, 1, 5)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal exec call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python deploy.py'

    def test_js_module_level_skipped_function_level_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """JS counterpart of the Python paired test. Module-level
        spawn is skipped (no caller_symbol_id); function-level emits.
        Bites a stub (no rows) and a too-greedy impl (two rows or
        NULL constraint violation)."""
        text = (
            "child_process.spawn('python', ['boot.py']);\n"
            'function deploy() {\n'
            "  child_process.spawn('python', ['deploy.py']);\n"
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        fn_id = 'scip-typescript . . . app.js/deploy().'
        index = _make_index([
            (src, [
                _occ_at(text, 'spawn', _JS_SPAWN_SYM, nth=0),
                _occ_at(text, 'spawn', _JS_SPAWN_SYM, nth=1),_def_occ(fn_id, 2, 4)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        assert len(rows) == 1, (
            f'expected only function-level call; got '
            f'{[r["target_path"] for r in rows]}'
        )
        assert rows[0]['caller_symbol_id'] == fn_id


# ---------------------------------------------------------------------------
# Scala — sys.process.Process
# ---------------------------------------------------------------------------


class TestScalaProcess:
    def test_process_apply_paired_with_skip(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Paired regression guard for ``Process(...)`` — literal
        arg emits, variable arg skips. ``Process("x").run()`` is
        a chained call; the SCIP occurrence on ``Process`` matches
        the inner call_expression that holds the args. Bites a
        stub (no rows), an impl missing the JVM Process registry
        entry, and an impl that captures non-literal args."""
        text = (
            'class Runner {\n'
            '  def runIt(): Unit = {\n'
            '    val cmd = "python skipped.py"\n'
            '    Process("python literal.py").run()\n'
            '    Process(cmd).run()\n'
            '  }\n'
            '}\n'
        )
        src = tmp_path / 'Runner.scala'
        src.write_text(text)
        method_id = 'scip:Runner#runIt().'
        index = _make_index([
            (src, [
                _occ_at(
                    text, 'Process', _SCALA_PROCESS_APPLY_SYM,
                    nth=0,
                ),
                _occ_at(
                    text, 'Process', _SCALA_PROCESS_APPLY_SYM,
                    nth=1,
                ),_def_occ('scip:Runner#', 1, 7), _def_occ(method_id, 2, 6)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal Process call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python literal.py'
        assert rows[0]['caller_symbol_id'] == method_id
        # Phase 2p indexed "python skipped.py" at the val line; the
        # extractor must not pull it in just because it exists in
        # string_literals.
        assert 'python skipped.py' not in targets


# ---------------------------------------------------------------------------
# Phase 2s wiring — variable args resolve via scip_symbols + string_literals
# ---------------------------------------------------------------------------


class TestPhase2sVariableResolution:
    """Phase 2s shipped as a library; this test asserts Phase 2t
    consumes it. Currently red against the literal-only extractor —
    will go green when the extractor's arg resolution falls through
    to ``resolve_arg_value`` for identifier args.

    Paired with the existing literal-only and skip cases — this test
    only fires when both ``string_literals`` AND ``scip_symbols``
    cover the var's def site, so a stub still fails on the literal
    cases and the skip cases stay green."""

    def test_variable_arg_resolves_via_scip_symbols(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """A module-level constant referenced in a subprocess call
        gets resolved through Phase 2s. Confidence reflects the
        resolution path (``'resolved-constant'`` rather than
        ``'literal'``)."""
        text = (
            'SCRIPT = "python pipeline.py"\n'
            'def deploy():\n'
            '    subprocess.run(SCRIPT)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # The variable being referenced — SCRIPT defined on line 1
        _add_scip_symbol(
            conn, canonical_id='scip:app.SCRIPT',
            source_name='myapi', file=str(src.resolve()),
            line_start=1, line_end=1,
            kind='Variable', qualified_name='app.SCRIPT',
        )
        index = _make_index([
            (src, [_occ_at(
                text, 'run', _PY_SUBPROCESS_RUN_SYM,
            ), _def_occ('scip:app.deploy', 2, 3)]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['target_path'] == 'python pipeline.py'
        assert rows[0]['confidence'] == 'resolved-constant'
        assert rows[0]['caller_symbol_id'] == 'scip:app.deploy'

    def test_variable_with_no_def_still_skipped(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Negative half — when ``scip_symbols`` doesn't carry the
        variable's definition, Phase 2s returns unresolved and the
        extractor skips. Paired with a literal call so the test
        bites both stub (no rows ever) and over-eager-resolution
        (variable picked up without a def)."""
        text = (
            'def deploy():\n'
            '    subprocess.run("python literal.py")\n'
            '    subprocess.run(UNDEFINED_VAR)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        # NO scip_symbol for UNDEFINED_VAR — Phase 2s returns
        # unresolved → extractor skips
        index = _make_index([
            (src, [
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0),
                _occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=1),_def_occ('scip:app.deploy', 1, 3)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the literal call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python literal.py'


# ---------------------------------------------------------------------------
# Per-sink regression coverage — bites registry suffix drift on
# secondary sinks not exercised by an individual paired test
# ---------------------------------------------------------------------------


class TestPerSinkRegressionCoverage:
    """Combined paired tests for sinks without dedicated tests
    elsewhere. Bite both directions per sink: missing literal row
    fails (regression in sink registration); extra variable row
    fails (extractor over-broad)."""

    def test_python_secondary_subprocess_sinks(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``subprocess.call`` / ``check_call`` / ``check_output`` —
        each with a literal arg (must emit) and a variable arg
        (must skip). Six occurrences total → three rows."""
        text = (
            'def deploy():\n'
            '    cmd = "x"\n'
            '    subprocess.call("alpha.py")\n'
            '    subprocess.check_call("beta.py")\n'
            '    subprocess.check_output("gamma.py")\n'
            '    subprocess.call(cmd)\n'
            '    subprocess.check_call(cmd)\n'
            '    subprocess.check_output(cmd)\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        sym_call = (
            'scip-python python pypi-stdlib 0 '
            'subprocess/__init__.py/call.'
        )
        sym_check_call = (
            'scip-python python pypi-stdlib 0 '
            'subprocess/__init__.py/check_call.'
        )
        sym_check_output = (
            'scip-python python pypi-stdlib 0 '
            'subprocess/__init__.py/check_output.'
        )
        index = _make_index([
            (src, [
                _occ_at(text, 'call', sym_call, nth=0),
                _occ_at(text, 'check_call', sym_check_call, nth=0),
                _occ_at(
                    text, 'check_output',
                    sym_check_output, nth=0,
                ),
                _occ_at(text, 'call', sym_call, nth=1),
                _occ_at(text, 'check_call', sym_check_call, nth=1),
                _occ_at(
                    text, 'check_output',
                    sym_check_output, nth=1,
                ),_def_occ('scip-python . . . app/deploy().', 1, 8)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 3, (
            f'expected exactly three literal rows; got {targets}'
        )
        assert {'alpha.py', 'beta.py', 'gamma.py'} == set(targets)
        # Variable arg's value must NOT leak in
        assert 'x' not in targets

    def test_js_secondary_child_process_sinks(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """``child_process.execFile`` / ``spawnSync`` / ``execSync`` /
        ``fork`` — paired with variable-arg skip cases. Eight
        occurrences total → four rows."""
        text = (
            'function deploy() {\n'
            "  const cmd = 'noop';\n"
            "  child_process.execFile('alpha');\n"
            "  child_process.spawnSync('beta');\n"
            "  child_process.execSync('gamma');\n"
            "  child_process.fork('delta');\n"
            '  child_process.execFile(cmd);\n'
            '  child_process.spawnSync(cmd);\n'
            '  child_process.execSync(cmd);\n'
            '  child_process.fork(cmd);\n'
            '}\n'
        )
        src = tmp_path / 'app.js'
        src.write_text(text)
        sym_execfile = (
            'scip-typescript . . . child_process.d.ts/execFile.'
        )
        sym_spawnsync = (
            'scip-typescript . . . child_process.d.ts/spawnSync.'
        )
        sym_execsync = (
            'scip-typescript . . . child_process.d.ts/execSync.'
        )
        sym_fork = (
            'scip-typescript . . . child_process.d.ts/fork.'
        )
        index = _make_index([
            (src, [
                _occ_at(text, 'execFile', sym_execfile, nth=0),
                _occ_at(text, 'spawnSync', sym_spawnsync, nth=0),
                _occ_at(text, 'execSync', sym_execsync, nth=0),
                _occ_at(text, 'fork', sym_fork, nth=0),
                _occ_at(text, 'execFile', sym_execfile, nth=1),
                _occ_at(text, 'spawnSync', sym_spawnsync, nth=1),
                _occ_at(text, 'execSync', sym_execsync, nth=1),
                _occ_at(text, 'fork', sym_fork, nth=1),_def_occ('scip-python . . . app/deploy().', 1, 11)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 4, (
            f'expected four literal rows; got {targets}'
        )
        assert {'alpha', 'beta', 'gamma', 'delta'} == set(targets)
        assert 'noop' not in targets


# ---------------------------------------------------------------------------
# Adversarial — SCIP filtering + error tolerance
# ---------------------------------------------------------------------------


class TestAdversarial:
    def test_unrelated_run_filtered_but_subprocess_run_emitted(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Same fixture has TWO ``run`` calls — one is
        ``subprocess.run(...)`` (registry match), one is a custom
        ``job.run(...)`` (registry miss). SCIP disambiguates via the
        symbol. Bites a stub (no rows) and a naive impl that ignores
        the registry filter (two rows)."""
        text = (
            'def deploy():\n'
            '    subprocess.run("python real.py")\n'
            '    job.run("task")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [
                _occ_at(
                    text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0,
                ),
                _occ_at(
                    text, 'run',
                    'scip-python . . . myproj/Job#run.',
                    nth=1,
                ),_def_occ('scip-python . . . app/deploy().', 1, 3)
            ]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the subprocess.run call; got {targets}'
        )
        assert rows[0]['target_path'] == 'python real.py'

    def test_scip_filter_drives_extraction(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two ``subprocess.run`` calls with identical syntax — only
        ONE has a SCIP occurrence. The extractor must emit only that
        one. Bites a stub (no rows) AND any impl that walks AST
        without consulting SCIP (two rows)."""
        text = (
            'def deploy():\n'
            '    subprocess.run("with-scip.py")\n'
            '    subprocess.run("without-scip.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [_occ_at(
                text, 'run', _PY_SUBPROCESS_RUN_SYM, nth=0,
            ), _def_occ('scip-python . . . app/deploy().', 1, 3)]),  # only first
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert len(rows) == 1, (
            f'expected only the SCIP-occurrence call; got {targets}'
        )
        assert rows[0]['target_path'] == 'with-scip.py'
        assert 'without-scip.py' not in targets

    def test_malformed_python_doesnt_crash(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        broken = tmp_path / 'broken.py'
        good = tmp_path / 'good.py'
        broken.write_text('def oops(\n  # missing close paren\n')
        good_text = (
            'def deploy():\n'
            '    subprocess.run("python ok.py")\n'
        )
        good.write_text(good_text)
        index = ScipIndex(
            documents=(
                _ScipDoc(
                    relative_path='broken.py',
                    occurrences=(),
                    symbols=(),
                ),
                _ScipDoc(
                    relative_path='good.py',
                    occurrences=(
                        _occ_at(
                            good_text, 'run', _PY_SUBPROCESS_RUN_SYM,
                        ),
                        _def_occ('scip-python . . . app/deploy().', 1, 2),
                    ),
                    symbols=(),
                ),
            ),
            source_root=tmp_path,
        )
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert 'python ok.py' in targets

    def test_missing_index_returns_zero_present_index_emits(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        """Two-branch test: without an index → 0 rows, gracefully;
        with an index → ≥1 row from the same source. The second
        branch is what bites a stub that always returns 0."""
        from docgen.scip_process_extractor import (
            ingest_process_invocations,
        )

        # First branch — no index_factory, no .scip on disk
        rc_missing = ingest_process_invocations(
            source_name='myapi', source_root=tmp_path, conn=conn,
        )
        assert rc_missing == 0
        assert _query_invocations(conn, 'myapi') == []

        # Second branch — same source_name, real fixture index
        text = (
            'def deploy():\n'
            '    subprocess.run("python script.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [_occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM), _def_occ('scip-python . . . app/deploy().', 1, 2)]),
        ], tmp_path)
        ingest_string_literals(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index_factory=lambda: index,
        )
        rc_present = ingest_process_invocations(
            source_name='myapi', source_root=tmp_path, conn=conn,
            index_factory=lambda: index,
        )
        assert rc_present == 1
        rows = _query_invocations(conn, 'myapi')
        assert len(rows) == 1
        assert rows[0]['target_path'] == 'python script.py'


# ---------------------------------------------------------------------------
# Re-ingest semantics
# ---------------------------------------------------------------------------


class TestReIngest:
    def test_replaces_same_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        text1 = (
            'def deploy():\n'
            '    subprocess.run("python old.py")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text1)
        index1 = _make_index([
            (src, [_occ_at(text1, 'run', _PY_SUBPROCESS_RUN_SYM), _def_occ('scip-python . . . app/deploy().', 1, 2)]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index1,
        )
        # Replace
        text2 = (
            'def deploy():\n'
            '    subprocess.run("python new.py")\n'
        )
        src.write_text(text2)
        index2 = _make_index([
            (src, [_occ_at(text2, 'run', _PY_SUBPROCESS_RUN_SYM), _def_occ('scip-python . . . app/deploy().', 1, 2)]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index2,
        )
        rows = _query_invocations(conn, 'myapi')
        targets = [r['target_path'] for r in rows]
        assert 'python old.py' not in targets
        assert 'python new.py' in targets

    def test_preserves_other_source_rows(
        self, tmp_path: Path, conn: sqlite3.Connection,
    ) -> None:
        # Pre-existing row from another source
        conn.execute(
            '''INSERT INTO process_invocations
               (source_name, caller_symbol_id, target_path,
                target_symbol_id, confidence, file,
                line_start, line_end)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            ('other', 'caller:other', '/preserved.sh',
             None, 'literal', '/x.py', 1, 1),
        )
        conn.commit()

        text = (
            'def deploy():\n'
            '    subprocess.run("./mine.sh")\n'
        )
        src = tmp_path / 'app.py'
        src.write_text(text)
        index = _make_index([
            (src, [_occ_at(text, 'run', _PY_SUBPROCESS_RUN_SYM), _def_occ('scip-python . . . app/deploy().', 1, 2)]),
        ], tmp_path)
        _run_pipeline(
            source_name='myapi', source_root=tmp_path,
            conn=conn, index=index,
        )
        my = [
            r['target_path']
            for r in _query_invocations(conn, 'myapi')
        ]
        other = [
            r['target_path']
            for r in _query_invocations(conn, 'other')
        ]
        assert './mine.sh' in my
        assert other == ['/preserved.sh']
