"""The public retrieval battery (evals/run_battery.py): participation
verdicts come from the PRODUCTION breadth criterion, ground truth is
title-anchored, junk is measured on must-stay-silent rows."""
import importlib.util
import sqlite3
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    'run_battery',
    Path(__file__).resolve().parent.parent / 'evals' / 'run_battery.py',
)
run_battery = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(run_battery)


def _unit(*coords):
    v = np.asarray(coords, dtype=np.float32)
    return v / np.linalg.norm(v)


@pytest.fixture
def store(tmp_path):
    """Synthetic store: 4 env docs + 2 consumer docs on a 3-dim sphere.

    Axis 0 = the environment's streaming concern, axis 1 = the consumer's
    own domain, axis 2 = unrelated. Env docs cluster on axis 0; the
    consumer doc sits on axis 1.
    """
    conn = sqlite3.connect(tmp_path / 'eval.db')
    conn.execute(
        'CREATE TABLE documents (id TEXT PRIMARY KEY, title TEXT, '
        'source_name TEXT)')
    rows = [
        ('e1', 'Explanation of _client.py — streaming responses', 'spool:env'),
        ('e2', 'Explanation of _transports.py', 'spool:env'),
        ('e3', 'Gotchas of _decoders.py', 'spool:env'),
        ('e4', 'Explanation of README.md', 'spool:env'),
        ('c1', 'Explanation of reporter.py — nightly summaries', 'consumer'),
        ('c2', 'Explanation of models.py', 'consumer'),
    ]
    conn.executemany('INSERT INTO documents VALUES (?, ?, ?)', rows)
    conn.commit()
    vectors = {
        'e1': _unit(1.0, 0.05, 0.0),
        'e2': _unit(0.9, 0.1, 0.1),
        'e3': _unit(0.85, 0.0, 0.2),
        'e4': _unit(0.6, 0.3, 0.4),
        'c1': _unit(0.05, 1.0, 0.0),
        'c2': _unit(0.1, 0.9, 0.2),
    }
    ids = list(vectors)
    matrix = run_battery.Matrix(
        id_to_row={i: n for n, i in enumerate(ids)},
        M=np.stack([vectors[i] for i in ids]),
    )
    return run_battery.Store(
        conn=conn, matrix=matrix,
        spool_doc_ids=['e1', 'e2', 'e3', 'e4'],
    )


class TestScoreRow:
    def test_seam_question_speaks_and_finds_ground_truth(self, store):
        row = {
            'label': 'seam-streaming', 'archetype': 'peripheral',
            'want': 'speak', 'consumer': 'consumer',
            'question': 'How do I stream a large download?',
            'ground_truth': {'title_contains': ['streaming']},
        }
        # Query on the environment axis: every env doc outscores the
        # consumer's best -> breadth speaks, and the GT doc leads.
        result = run_battery.score_row(row, store, _unit(1.0, 0.02, 0.02))
        assert result.verdict == 'speak'
        assert result.correct
        assert result.gt_in_context
        assert result.gt_rank == 1
        assert result.junk == 0

    def test_repo_control_stays_silent(self, store):
        row = {
            'label': 'control-domain', 'archetype': 'peripheral',
            'want': 'silent', 'consumer': 'consumer',
            'question': 'How does our nightly summary get scheduled?',
        }
        # Query on the consumer's own axis: the consumer doc beats the
        # whole env window -> silent, zero junk, GT not applicable.
        result = run_battery.score_row(row, store, _unit(0.02, 1.0, 0.02))
        assert result.verdict == 'silent'
        assert result.correct
        assert result.gt_in_context is None
        assert result.junk == 0

    def test_leak_on_control_row_is_counted_as_junk(self, store):
        row = {
            'label': 'control-leaky', 'archetype': 'peripheral',
            'want': 'silent', 'consumer': 'consumer',
            'question': 'Ambiguous phrasing that drifts to the env.',
        }
        # Query drifts onto the env axis: breadth speaks against the
        # want -> incorrect, and the admitted docs are counted as junk.
        result = run_battery.score_row(row, store, _unit(1.0, 0.1, 0.0))
        assert result.verdict == 'speak'
        assert not result.correct
        assert result.junk > 0

    def test_speak_row_missing_ground_truth_is_flagged(self, store):
        row = {
            'label': 'seam-gt-miss', 'archetype': 'adopter',
            'want': 'speak', 'consumer': 'consumer',
            'question': 'Environment question whose GT title is absent.',
            'ground_truth': {'title_contains': ['retry budget']},
        }
        result = run_battery.score_row(row, store, _unit(1.0, 0.02, 0.02))
        assert result.verdict == 'speak'
        assert result.correct          # participation contract held...
        assert result.gt_in_context is False   # ...but the answer doc missed
        assert result.gt_rank is None


class TestReport:
    def test_report_totals_and_exit_code(self, store):
        rows = [
            {'label': 'a', 'archetype': 'peripheral', 'want': 'speak',
             'consumer': 'consumer', 'question': 'q1',
             'ground_truth': {'title_contains': ['streaming']}},
            {'label': 'b', 'archetype': 'peripheral', 'want': 'silent',
             'consumer': 'consumer', 'question': 'q2'},
        ]
        vectors = {'a': _unit(1.0, 0.02, 0.02), 'b': _unit(0.02, 1.0, 0.02)}
        report = run_battery.run(rows, store, vectors.__getitem__)
        assert report.correct == 2 and report.total == 2
        assert report.gt_hits == 1 and report.gt_total == 1
        assert report.junk_total == 0
        assert report.exit_code == 0

    def test_any_miss_fails_the_run(self, store):
        rows = [
            {'label': 'leak', 'archetype': 'peripheral', 'want': 'silent',
             'consumer': 'consumer', 'question': 'drifts'},
        ]
        report = run_battery.run(
            rows, store, {'leak': _unit(1.0, 0.1, 0.0)}.__getitem__)
        assert report.correct == 0
        assert report.exit_code == 1
