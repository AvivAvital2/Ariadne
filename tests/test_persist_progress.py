"""Per-file progress emission in the string-literals persist path — the step
whose long silent stall prompted this. ``ingest_string_literals`` must report
``(source, completed, total)`` per indexed file (so a bar advances smoothly and
lands at total/total), and ``persist_string_literals`` must forward the reporter
through. Synthetic fixtures only.
"""
from __future__ import annotations

import io

from rich.console import Console
from rich.progress import BarColumn, Progress, TextColumn

from cli.index import persist_progress
from docgen.scip_extractor import ScipIndex, _ScipDoc
from docgen.scip_persist import (
    persist_all_sources,
    persist_data_model,
    persist_string_literals,
)
from docgen.scip_string_literal_extractor import ingest_string_literals
from library import Library
from progress_util import _update_persist_task


def test_ingest_reports_per_file_progress(tmp_path) -> None:
    (tmp_path / 'a.py').write_text('x = "1"\n', encoding='utf-8')
    (tmp_path / 'b.py').write_text('y = "2"\n', encoding='utf-8')
    index = ScipIndex(
        documents=(_ScipDoc('a.py'), _ScipDoc('b.py')), source_root=tmp_path,
    )
    calls: list[tuple[str, int, int]] = []
    library = Library(tmp_path / 'db.sqlite')
    try:
        with library._conn_provider.acquire() as conn:
            ingest_string_literals(
                source_name='src1', source_root=tmp_path, conn=conn,
                index_factory=lambda: index,
                progress_callback=lambda lbl, done, total: calls.append(
                    (lbl, done, total)),
            )
    finally:
        library.close()
    # advances per file (2 docs) and lands at total/total
    assert calls == [('src1', 0, 2), ('src1', 1, 2), ('src1', 2, 2)]


def test_persist_string_literals_forwards_progress_callback(
    tmp_path, monkeypatch,
) -> None:
    received: dict = {}

    def fake_ingest(*, source_name, source_root, conn, progress_callback=None):
        received['cb'] = progress_callback
        return 0

    monkeypatch.setattr(
        'docgen.scip_string_literal_extractor.ingest_string_literals',
        fake_ingest,
    )

    def reporter(lbl, done, total):
        return None

    persist_string_literals(
        tmp_path / 'db.sqlite', [('src1', tmp_path)],
        progress_callback=reporter,
    )
    assert received['cb'] is reporter


def test_persist_data_model_reports_per_source_progress(tmp_path) -> None:
    """Per-source progress: one continuous bar advancing across sources (the
    dominant per-source cost is the atomic .scip load). Reports ('', i, n) →
    ('', n, n). Sources without a manifest are skipped internally, but the bar
    still advances per source."""
    calls: list[tuple[str, int, int]] = []
    persist_data_model(
        tmp_path / 'db.sqlite',
        [('a', tmp_path / 'a'), ('b', tmp_path / 'b')],
        progress_callback=lambda lbl, done, total: calls.append(
            (lbl, done, total)),
    )
    assert calls == [('', 0, 2), ('', 1, 2), ('', 2, 2)]


def test_persist_all_sources_reports_per_source_progress(tmp_path) -> None:
    """Same per-source contract over the graph-materialization load loop."""
    calls: list[tuple[str, int, int]] = []
    persist_all_sources(
        tmp_path / 'db.sqlite',
        [('a', tmp_path / 'a'), ('b', tmp_path / 'b')],
        progress_callback=lambda lbl, done, total: calls.append(
            (lbl, done, total)),
    )
    assert calls == [('', 0, 2), ('', 1, 2), ('', 2, 2)]


# --- the renderer (cli/index.py) -----------------------------------------


def _bare_progress() -> Progress:
    """A headless Progress (no real terminal) exposing the `detail` field the
    renderer sets, so we can assert task state directly."""
    return Progress(
        TextColumn('{task.description}'), BarColumn(),
        TextColumn('{task.fields[detail]}'),
        console=Console(file=io.StringIO()))


def test_update_persist_task_tracks_advances_and_resets_per_label() -> None:
    """First report creates the task; same-label advances it; a changed label
    resets it (fresh bar + ETA) with the new total; an empty label drops the
    sub-label and total 0 leaves the bar indeterminate."""
    progress = _bare_progress()
    state: dict = {}
    with progress:
        _update_persist_task(progress, state, 'data model', 'a', 0, 5)
        task = progress.tasks[0]
        assert (task.completed, task.total) == (0, 5)
        assert 'data model: a' in task.description

        _update_persist_task(progress, state, 'data model', 'a', 3, 5)
        assert progress.tasks[0].completed == 3  # same label → advance

        _update_persist_task(progress, state, 'data model', 'b', 0, 2)
        assert (progress.tasks[0].completed, progress.tasks[0].total) == (0, 2)
        assert 'data model: b' in progress.tasks[0].description  # change → reset

        _update_persist_task(progress, state, 'data model', '', 1, 3)
        assert progress.tasks[0].description.strip() == 'data model'  # no sub-label
        assert (progress.tasks[0].completed, progress.tasks[0].total) == (1, 3)


def test_persist_progress_yields_working_reporter() -> None:
    """The context manager yields a callable reporter that drives the bar
    without error across an advancing source."""
    with persist_progress(Console(file=io.StringIO()), 'string literals') as report:
        assert callable(report)
        report('src1', 0, 10)
        report('src1', 10, 10)
