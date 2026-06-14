"""Contract for ``_plan_indexing`` — the grouping/ordering that drives
the per-language Indexing display.

Design intent (from the user request):
  - Manifest indexer entries are GROUPED by language so all of one
    language's scopes run consecutively (no interleaving).
  - Languages are ordered by total file volume, SMALLEST FIRST, so the
    quick languages complete before the heavy one (e.g. the JVM compile).
  - A language with no entries never appears.

``_plan_indexing(entries, count_fn)`` returns an ordered list of
``(kind, entries_for_kind, total_files)`` tuples. ``count_fn`` maps an
entry to its file count (injected so this is pure / filesystem-free).
"""
from __future__ import annotations


def test_groups_by_kind_and_orders_by_volume_ascending():
    from cli.index import _plan_indexing

    # Interleaved manifest, mixed languages. Counts chosen so the
    # ascending-volume order (python=30, java=50, typescript=900)
    # differs from manifest order.
    entries = [
        {'kind': 'typescript', 'cwd': 'web', 'files': 500},
        {'kind': 'python', 'cwd': 'a', 'files': 10},
        {'kind': 'java', 'cwd': '.', 'files': 50},
        {'kind': 'python', 'cwd': 'b', 'files': 20},
        {'kind': 'typescript', 'cwd': 'widgets', 'files': 400},
    ]

    plan = _plan_indexing(entries, lambda e: e['files'])

    kinds = [kind for kind, _, _ in plan]
    assert kinds == ['python', 'java', 'typescript'], (
        'languages must be ordered smallest-volume-first'
    )

    totals = {kind: total for kind, _, total in plan}
    assert totals == {'python': 30, 'java': 50, 'typescript': 900}

    # Each language's scopes are grouped together, manifest order
    # preserved WITHIN a language.
    by_kind = {kind: ents for kind, ents, _ in plan}
    assert [e['cwd'] for e in by_kind['python']] == ['a', 'b']
    assert [e['cwd'] for e in by_kind['typescript']] == ['web', 'widgets']
    assert len(by_kind['java']) == 1


def test_absent_languages_do_not_appear():
    from cli.index import _plan_indexing

    entries = [
        {'kind': 'python', 'cwd': 'a', 'files': 5},
        {'kind': 'python', 'cwd': 'b', 'files': 7},
    ]
    plan = _plan_indexing(entries, lambda e: e['files'])
    assert [kind for kind, _, _ in plan] == ['python']


def test_only_python_streams_file_progress():
    """Only the Python adapter emits per-file events, so only its bar can
    be a determinate file counter; opaque adapters (java/typescript) get
    an animated indeterminate bar instead of a frozen 0/N."""
    from cli.index import _streams_file_progress

    assert _streams_file_progress('python') is True
    assert _streams_file_progress('typescript') is False
    assert _streams_file_progress('java') is False
    assert _streams_file_progress('anything-else') is False


def test_pulse_bar_only_for_java():
    """Only scip-java (single monolithic compile, no progress) gets the
    animated indeterminate bar. Python and TypeScript/JS keep the
    determinate file-counter bar."""
    from cli.index import _pulse_bar

    assert _pulse_bar('java') is True
    assert _pulse_bar('python') is False
    assert _pulse_bar('typescript') is False


def test_index_detail_text():
    from cli.index import _index_detail_text

    # Determinate (Python, TypeScript/JS): X/N files counter.
    assert _index_detail_text(1337, 512, pulse=False) == '512/1337 files'
    assert _index_detail_text(1337, 1337, pulse=False) == '1337/1337 files'
    # Pulse bar (Java): just the total N, no X/.
    assert _index_detail_text(2083, 0, pulse=True) == '2083 files'


def test_index_summary_renders_nested_indented_lines():
    """The per-language summary the callers print under ``✓ Index`` is
    one indented line per language: name, file count, m:ss elapsed."""
    from cli.dry_run import _print_index_summary, console

    summary = [
        {'language': 'Python', 'files': 1500, 'seconds': 8},
        {'language': 'Java', 'files': 900, 'seconds': 72},
    ]
    with console.capture() as cap:
        _print_index_summary(summary)
    out = cap.get()

    assert 'Python' in out and '1500 files' in out and '0:08' in out
    assert 'Java' in out and '900 files' in out and '1:12' in out
    # Every non-blank line is indented (nested under the ✓ Index line).
    for line in out.splitlines():
        if line.strip():
            assert line.startswith('    '), repr(line)


def test_volume_tie_breaks_deterministically_by_kind_name():
    from cli.index import _plan_indexing

    # Equal volumes → stable, deterministic order (kind name).
    entries = [
        {'kind': 'typescript', 'cwd': 'w', 'files': 100},
        {'kind': 'java', 'cwd': '.', 'files': 100},
    ]
    plan = _plan_indexing(entries, lambda e: e['files'])
    assert [kind for kind, _, _ in plan] == ['java', 'typescript']
